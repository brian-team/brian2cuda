'''
Module implementing the CUDA "standalone" device.
'''
import os
import inspect
from collections import defaultdict, Counter
import tempfile
from distutils import ccompiler
import re
from itertools import chain

import numpy as np

from brian2.codegen.cpp_prefs import get_compiler_and_args
from brian2.codegen.translation import make_statements
from brian2.core.clocks import Clock, defaultclock
from brian2.core.namespace import get_local_namespace
from brian2.core.preferences import prefs, PreferenceError
from brian2.core.variables import ArrayVariable, DynamicArrayVariable
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.devices.device import all_devices
from brian2.synapses.synapses import Synapses, SynapticPathway
from brian2.utils.filetools import copy_directory, ensure_directory
from brian2.utils.stringtools import get_identifiers, stripped_deindented_lines
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.utils.logger import get_logger
from brian2.units import second
from brian2.monitors import SpikeMonitor, StateMonitor, EventMonitor
from brian2.groups import Subgroup

from brian2.devices.cpp_standalone.device import CPPWriter, CPPStandaloneDevice
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup

from brian2cuda.utils.stringtools import replace_floating_point_literals
from brian2cuda.utils.gputools import select_gpu, get_nvcc_path
from brian2cuda.utils.logger import report_issue_message

from .codeobject import CUDAStandaloneCodeObject, CUDAStandaloneAtomicsCodeObject


__all__ = []

logger = get_logger('brian2.devices.cuda_standalone')


class CUDAWriter(CPPWriter):
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.source_files = set()
        self.header_files = set()

    def write(self, filename, contents):
        logger.diagnostic(f'Writing file {filename}:\n{contents}')
        if filename.lower().endswith('.cu'):
            self.source_files.add(filename)
        if filename.lower().endswith('.cpp'):
            self.source_files.add(filename)
        elif filename.lower().endswith('.h'):
            self.header_files.add(filename)
        elif filename.endswith('.*'):
            self.write(filename[:-1]+'cu', contents.cu_file)
            self.write(filename[:-1]+'h', contents.h_file)
            return
        fullfilename = os.path.join(self.project_dir, filename)
        if os.path.exists(fullfilename):
            if open(fullfilename, 'r').read()==contents:
                return
        open(fullfilename, 'w').write(contents)


class CUDAStandaloneDevice(CPPStandaloneDevice):
    '''
    The `Device` used for CUDA standalone simulations.
    '''

    def __init__(self):
        super(CUDAStandaloneDevice, self).__init__()
        ### Reset variables we don't need from CPPStandaloneDevice.__init__()
        # remove randomkit, which we don't use for CUDA Standalone
        self.include_dirs.remove('brianlib/randomkit')
        self.library_dirs.remove('brianlib/randomkit')

        # Add code line slots used in our benchmarks
        # TODO: Add to brian2 and remove here
        self.code_lines.update({'before_network_run': [],
                                'after_network_run': []})

        ### Attributes specific to CUDAStandaloneDevice:
        # only true during first run call (relevant for synaptic pre/post ID deletion)
        self.first_run = True
        # the minimal supported GPU compute capability
        self.minimal_compute_capability = 3.5
        # store the ID of the used GPU and it's compute capability
        self.gpu_id = None
        self.compute_capability = None
        # list of pre/post ID and delay arrays that are not needed in device memory
        self.delete_synaptic_pre = {}
        self.delete_synaptic_post = {}
        self.delete_synaptic_delay = {}
        # The following nested dictionary collects all codeobjects that use random
        # number generation (RNG).
        self.codeobjects_with_rng = {
            # All codeobjects that use the curand device api (binomial and
            # poisson with vectorized lambda)
            "device_api": {
                # This collects all codeobjects that run evey cycle of a clock
                "every_tick": [],
                # This collects all codeobjects that are running only once
                "single_tick": []
            },
            # All codeobjects that use the curand host api (rand, randn and poisson with
            # scalar lambda)
            "host_api": {
                # Dictionary of lists of codeobjects. Dictionary keys are the RNG types
                # (rand, randn, poisson_<idx>). For each `poisson(lamda)` function with
                # a different scalar lamda per codeobject, a new `poisson_<idx>` key is
                # added to the defaultdict. The dictionary values are lists of
                # codeobject.
                "all_runs": defaultdict(list),
                # Lists of defaultdict(list) with same structure as "all_runs", but now
                # codeobjects are seperate by `brian2.run()` calls (one list item per
                # run).
                "per_run": []
            },
        }
        # Dictionary to look up `lambda` values for all `poisson` calls with scalar
        # `lambda`, sorted by codeobj.name and poisson_name (`poisson-<idx>`):
        #   all_poisson_lamdas[codeobj.name][poisson_name] = lamda
        self.all_poisson_lamdas = defaultdict(dict)
        # List of multisynaptic index variables (for all Synapses with multisynaptic
        # index)
        self.multisyn_vars = []
        # List of names of all variables which are only required on host and will not
        # be copied to device memory
        self.variables_on_host_only = []

    def get_array_name(self, var, access_data=True, prefix=None):
        '''
        Return a globally unique name for `var`.

        Parameters
        ----------
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).
        prefix: {'_ptr', 'dev', 'd'}, optional
            Prefix for array name. Host pointers to device memory are prefixed
            with `dev`, device pointers to device memory are prefixed with `d`
            and pointers used in `scalar_code` and `vector_code` are prefixed
            with `_ptr` (independent of whether they are used in host or device
            code). The `_ptr` variables are declared as parameters in the
            kernel definition (KERNEL_PARAMETERS).
        '''
        # In single-precision mode we replace dt variables in codeobjects with
        # a single precision version, for details see #148
        if hasattr(var, 'real_var'):
            return self.get_array_name(var.real_var, access_data=access_data,
                                       prefix=prefix)

        prefix = prefix or ''
        choices = ['_ptr', 'dev', 'd', '']
        if prefix not in choices:
            raise ValueError(
                f"`prefix` has to be one of {choices} or `None`, got {prefix}"
            )

        if not access_data and prefix in ['_ptr', 'd']:
            raise ValueError(
                f"Don't use `'{prefix}'` prefix for a dynamic array object."
            )

        array_name = ''
        if isinstance(var, DynamicArrayVariable):
            if access_data:
                array_name = self.arrays[var]
            elif var.ndim == 1:
                array_name = self.dynamic_arrays[var]
            else:
                array_name = self.dynamic_arrays_2d[var]

        elif isinstance(var, ArrayVariable):
            array_name = self.arrays[var]
        else:
            raise TypeError(('Do not have a name for variable of type '
                             '%s') % type(var))

        return prefix + array_name

    def code_object_class(self, codeobj_class=None, fallback_pref=None):
        '''
        Return `CodeObject` class (either `CUDAStandaloneCodeObject` class or input)

        Parameters
        ----------
        codeobj_class : a `CodeObject` class, optional
            If this is keyword is set to None or no arguments are given, this method will return
            the default (`CUDAStandaloneCodeObject` class).
        fallback_pref : str, optional
            For the cuda_standalone device this option is ignored.

        Returns
        -------
        codeobj_class : class
            The `CodeObject` class that should be used
        '''
        # Ignore the requested pref (used for optimization in runtime)
        if codeobj_class is None:
            return CUDAStandaloneCodeObject
        else:
            return codeobj_class

    def get_array_read_write(self, abstract_code, variables):
        ##################################################################
        # This code is copied from CodeGenerator.translate() and
        # CodeGenerator.array_read_write() and should give us a set of
        # variables to which will be written or from which will be read in
        # `vector_code`
        ##################################################################
        vector_statements = {}
        for ac_name, ac_code in abstract_code.items():
            statements = make_statements(ac_code,
                                         variables,
                                         prefs['core.default_float_dtype'],
                                         optimise=True,
                                         blockname=ac_name)
            _, vector_statements[ac_name] = statements
        read = set()
        write = set()
        for statements in vector_statements.values():
            for stmt in statements:
                ids = get_identifiers(stmt.expr)
                # if the operation is inplace this counts as a read.
                if stmt.inplace:
                    ids.add(stmt.var)
                read = read.union(ids)
                write.add(stmt.var)
        read = set(varname for varname, var in variables.items()
                   if isinstance(var, ArrayVariable) and varname in read)
        write = set(varname for varname, var in variables.items()
                    if isinstance(var, ArrayVariable) and varname in write)
        return read, write

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None,
                    override_conditional_write=None, compiler_kwds=None):
        if prefs['core.default_float_dtype'] == np.float32 and 'dt' in variables:
            # In single-precision mode we replace dt variables in codeobjects with
            # a single precision version, for details see #148
            dt_var = variables['dt']
            new_dt_var = ArrayVariable(dt_var.name,
                                       dt_var.owner,
                                       dt_var.size,
                                       dt_var.device,
                                       dimensions=dt_var.dim,
                                       dtype=np.float32,
                                       constant=dt_var.constant,
                                       scalar=dt_var.scalar,
                                       read_only=dt_var.read_only,
                                       dynamic=dt_var.dynamic,
                                       unique=dt_var.unique)
            new_dt_var.real_var = dt_var
            new_dt_var.set_value(dt_var.get_value().item())
            variables['dt'] = new_dt_var
        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = dict(template_kwds)
        template_kwds['profiled'] = self.enable_profiling
        template_kwds['bundle_mode'] = prefs["devices.cuda_standalone.push_synapse_bundles"]
        no_or_const_delay_mode = False
        if isinstance(owner, (SynapticPathway, Synapses)) and "delay" in owner.variables and owner.variables["delay"].scalar:
            # catches Synapses(..., delay=...) syntax, does not catch the case when no delay is specified at all
            no_or_const_delay_mode = True
        template_kwds["no_or_const_delay_mode"] = no_or_const_delay_mode

        # Check if pre/post IDs are needed per synapse
        # This is the case  when presynapic/postsynapitc variables are used in synapses code or
        # if they are used to set synaptic variables after the first run call.
        # If in at least one `synapses` template for the same Synapses
        # object or in a `run_regularly` call (creates a `statupdate`) a
        # pre/post IDs are needed, we don't delete them. And if a synapses object that is only
        # run once after the first run call (e.g. syn.w['i<j'] = ...), we don't delete either.
        # If deleted, they will be deleted on the device in `run_lines` (see below)
        synapses_object_every_tick = False
        synapses_object_single_tick_after_run = False
        group_variable_set_templates = ['group_variable_set_conditional', 'group_variable_set']
        if isinstance(owner, Synapses):
            if template_name in ['synapses', 'stateupdate', 'summed_variable']:
                synapses_object_every_tick = True
            if not self.first_run and template_name in group_variable_set_templates:
                synapses_object_single_tick_after_run = True
        if synapses_object_every_tick or synapses_object_single_tick_after_run:
            read, write = self.get_array_read_write(abstract_code, variables)
            read_write = read.union(write)
            synaptic_pre_array_name = self.get_array_name(
                owner.variables['_synaptic_pre'], access_data=False
            )
            synaptic_post_array_name = self.get_array_name(
                owner.variables['_synaptic_post'], access_data=False
            )
            if synaptic_pre_array_name not in self.delete_synaptic_pre.keys():
                self.delete_synaptic_pre[synaptic_pre_array_name] = True
            if synaptic_post_array_name not in self.delete_synaptic_post.keys():
                self.delete_synaptic_post[synaptic_post_array_name] = True
            error_msg = ("'devices.cuda_standalone.no_{prepost}_references' "
                         "was set to True, but {prepost}synaptic index is "
                         "needed for variable {varname} in {owner.name}")
            # Check for all variable that are read or written to if they are
            # i/j or their indices are pre/post
            for varname in variables.keys():
                if varname in read_write:
                    idx = variable_indices[varname]
                    if idx == '_presynaptic_idx' or varname == 'i':
                        self.delete_synaptic_pre[synaptic_pre_array_name] = False
                        if prefs['devices.cuda_standalone.no_pre_references']:
                            raise PreferenceError(error_msg.format(prepost='pre',
                                                                   varname=varname,
                                                                   owner=owner))
                    if idx == '_postsynaptic_idx' or varname == 'j':
                        self.delete_synaptic_post[synaptic_post_array_name] = False
                        if prefs['devices.cuda_standalone.no_post_references']:
                            raise PreferenceError(error_msg.format(prepost='post',
                                                                   varname=varname,
                                                                   owner=owner))
            # Summed variables need the indices of their target variable, which
            # are not in the read_write set.
            if template_name == 'summed_variable':
                idx = template_kwds['_index_var'].name
                varname = template_kwds['_target_var'].name
                if idx == '_synaptic_pre':
                    self.delete_synaptic_pre[synaptic_pre_array_name] = False
                    if prefs['devices.cuda_standalone.no_pre_references']:
                        raise PreferenceError(error_msg.format(prepost='pre',
                                                               varname=varname,
                                                               owner=owner))
                if idx == '_synaptic_post':
                    self.delete_synaptic_post[synaptic_post_array_name] = False
                    if prefs['devices.cuda_standalone.no_post_references']:
                        raise PreferenceError(error_msg.format(prepost='post',
                                                               varname=varname,
                                                               owner=owner))
                if idx == '_syaptic_post':
                    self.delete_synaptic_post[synaptic_post_array_name] = False

        # Collect all variables written to in group_variable_set_templates so they can
        # be copied from device to host after being changed on device
        if template_name in group_variable_set_templates:
            read, write = self.get_array_read_write(abstract_code, variables)
            written_variables = {}
            for variable_name in write:
                var = variables[variable_name]
                varname = self.get_array_name(var, access_data=False)
                written_variables[var] = varname
            template_kwds['written_variables'] = written_variables
            # Do a similar check for the delay variables as for i/j. Delays can't be
            # modified in Synapses model code, so we only care about single tick objects
            # (setting delay values)
            if (
                not self.first_run
                and 'delay' in owner.variables
                and isinstance(owner, SynapticPathway)
            ):
                read_write = read.union(write)
                varname = owner.variables['delay'].name
                if varname in read_write:
                    synaptic_delay_array_name = self.get_array_name(
                        owner.variables['delay'], access_data=False
                    )
                    self.delete_synaptic_delay[synaptic_delay_array_name] = False

        if template_name == "synapses":
            prepost = template_kwds['pathway'].prepost
            synaptic_effects = "synapse"
            for varname in variables.keys():
                if varname in write:
                    idx = variable_indices[varname]
                    if (prepost == 'pre' and idx == '_postsynaptic_idx') or (prepost == 'post' and idx == '_presynaptic_idx'):
                        # The SynapticPathways 'target' group variables are modified
                        if synaptic_effects == "synapse":
                            synaptic_effects = "target"
                    if (prepost == 'pre' and idx == '_presynaptic_idx') or (prepost == 'post' and idx == '_postsynaptic_idx'):
                        # The SynapticPathways 'source' group variables are modified
                        synaptic_effects = "source"
            template_kwds["synaptic_effects"] = synaptic_effects
            logger.debug(f"Synaptic effects of Synapses object {name} modify {synaptic_effects} group variables.")
            # use atomics if possible (except for `synapses` mode, where we cann parallelise without)
            # TODO: this overwrites if somebody sets a codeobject in the Synapses(..., codeobj_class=...)
            if prefs['devices.cuda_standalone.use_atomics'] and synaptic_effects != 'synapses':
                codeobj_class = CUDAStandaloneAtomicsCodeObject
                logger.debug(
                    f"Using atomics in synaptic effect application of Synapses object "
                    f"{name}"
                )
            threads_expr = prefs.devices.cuda_standalone.threads_per_synapse_bundle
            pathway_name = template_kwds['pathway'].name
            replace_expr = {
                '{mean}': f'{pathway_name}_bundle_size_mean',
                '{std}': f'{pathway_name}_bundle_size_std',
                '{min}': f'{pathway_name}_bundle_size_min',
                '{max}': f'{pathway_name}_bundle_size_max',
            }
            for old, new in replace_expr.items():
                threads_expr = threads_expr.replace(old, new)
            template_kwds["threads_per_synapse_bundle"] = threads_expr
            template_kwds["bundle_threads_warp_multiple"] = prefs.devices.cuda_standalone.bundle_threads_warp_multiple
        if template_name in ["synapses_create_generator", "synapses_create_array"]:
            if owner.multisynaptic_index is not None:
                template_kwds["multisynaptic_idx_var"] = owner.variables[owner.multisynaptic_index]
            template_kwds["no_pre_references"] = False
            template_kwds["no_post_references"] = False
            if prefs['devices.cuda_standalone.no_pre_references']:
                template_kwds["no_pre_references"] = True
            if prefs['devices.cuda_standalone.no_post_references']:
                template_kwds["no_post_references"] = True
        template_kwds["launch_bounds"] = prefs["devices.cuda_standalone.launch_bounds"]
        template_kwds["sm_multiplier"] = prefs["devices.cuda_standalone.SM_multiplier"]
        template_kwds["syn_launch_bounds"] = prefs["devices.cuda_standalone.syn_launch_bounds"]
        template_kwds["calc_occupancy"] = prefs["devices.cuda_standalone.calc_occupancy"]
        if template_name in ["threshold", "spikegenerator"]:
            template_kwds["extra_threshold_kernel"] = prefs["devices.cuda_standalone.extra_threshold_kernel"]
        codeobj = super(CUDAStandaloneDevice, self).code_object(owner, name, abstract_code, variables,
                                                                template_name, variable_indices,
                                                                codeobj_class=codeobj_class,
                                                                template_kwds=template_kwds,
                                                                override_conditional_write=override_conditional_write,
                                                                compiler_kwds=compiler_kwds,
                                                                )
        return codeobj

    def check_openmp_compatible(self, nb_threads):
        if nb_threads > 0:
            raise NotImplementedError("Using OpenMP in a CUDA standalone project is not supported")

    def generate_objects_source(self, writer, arange_arrays, synapses, static_array_specs, networks):
        sm_multiplier = prefs.devices.cuda_standalone.SM_multiplier
        num_parallel_blocks = prefs.devices.cuda_standalone.parallel_blocks
        curand_generator_type = prefs.devices.cuda_standalone.random_number_generator_type
        curand_generator_ordering = prefs.devices.cuda_standalone.random_number_generator_ordering
        self.eventspace_arrays = {}
        self.spikegenerator_eventspaces = []
        for var, varname in self.arrays.items():
            if var.name.endswith('space'):  # get all eventspace variables
                self.eventspace_arrays[var] = varname
                #if hasattr(var, 'owner') and isinstance(v.owner, Clock):
                if isinstance(var.owner, SpikeGeneratorGroup):
                    self.spikegenerator_eventspaces.append(varname)
        for var in self.eventspace_arrays.keys():
            del self.arrays[var]
        subgroups_with_spikemonitor = set()
        for codeobj in self.code_objects.values():
            if isinstance(codeobj.owner, SpikeMonitor):
                if isinstance(codeobj.owner.source, Subgroup):
                    subgroups_with_spikemonitor.add(codeobj.owner.source.name)
        profile_statemonitor_copy_to_host = prefs.devices.cuda_standalone.profile_statemonitor_copy_to_host
        profile_statemonitor_vars = []
        for var in self.dynamic_arrays_2d.keys():
            is_statemon = isinstance(var.owner, StateMonitor)
            if (profile_statemonitor_copy_to_host
                    and isinstance(var.owner, StateMonitor)
                    and var.name == profile_statemonitor_copy_to_host):
                profile_statemonitor_vars.append(var)
        arr_tmp = self.code_object_class().templater.objects(
                        None, None,
                        array_specs=self.arrays,
                        dynamic_array_specs=self.dynamic_arrays,
                        dynamic_array_2d_specs=self.dynamic_arrays_2d,
                        zero_arrays=self.zero_arrays,
                        arange_arrays=arange_arrays,
                        synapses=synapses,
                        clocks=self.clocks,
                        static_array_specs=static_array_specs,
                        networks=networks,
                        code_objects=self.code_objects.values(),
                        get_array_filename=self.get_array_filename,
                        all_codeobj_with_host_rng=self.codeobjects_with_rng["host_api"]["all_runs"],
                        sm_multiplier=sm_multiplier,
                        num_parallel_blocks=num_parallel_blocks,
                        curand_generator_type=curand_generator_type,
                        curand_generator_ordering=curand_generator_ordering,
                        curand_float_type=c_data_type(prefs['core.default_float_dtype']),
                        eventspace_arrays=self.eventspace_arrays,
                        spikegenerator_eventspaces=self.spikegenerator_eventspaces,
                        multisynaptic_idx_vars=self.multisyn_vars,
                        profiled_codeobjects=self.profiled_codeobjects,
                        profile_statemonitor_copy_to_host=profile_statemonitor_copy_to_host,
                        profile_statemonitor_vars=profile_statemonitor_vars,
                        subgroups_with_spikemonitor=sorted(subgroups_with_spikemonitor),
                        variables_on_host_only=self.variables_on_host_only)
        # Reinsert deleted entries, in case we use self.arrays later? maybe unnecassary...
        self.arrays.update(self.eventspace_arrays)
        writer.write('objects.*', arr_tmp)

    def generate_main_source(self, writer):
        main_lines = []
        procedures = [('', main_lines)]
        runfuncs = {}
        run_counter = 0
        for func, args in self.main_queue:
            if func=='before_run_code_object':
                codeobj, = args
                main_lines.append('_before_run_%s();' % codeobj.name)
            elif func=='run_code_object':
                codeobj, = args
                codeobj.runs_every_tick = False
                # Need to check for RNG functions in code objects only run once (e.g.
                # when setting group variables before run). The results are stored in
                # `code_object.rng_calls`, `code_object.poisson_lamdas` and
                # `code_object.needs_curand_states`.
                prepare_codeobj_code_for_rng(codeobj)
                if codeobj.needs_curand_states:
                    self.codeobjects_with_rng["device_api"]["single_tick"].append(codeobj)
                    if isinstance(codeobj.owner, Synapses) \
                            and codeobj.template_name in ['group_variable_set_conditional', 'group_variable_set']:
                        # At curand state initalization, synapses are not generated yet.
                        # For codeobjects run every tick, this happens in the init() of
                        # the random number buffer called at first clock cycle of the network
                        main_lines.append('random_number_buffer.ensure_enough_curand_states();')
                main_lines.append(f'_run_{codeobj.name}();')
            elif func == 'after_run_code_object':
                codeobj, = args
                main_lines.append(f'_after_run_{codeobj.name}();')
            elif func=='run_network':
                net, netcode = args
                if run_counter == 0:
                    # These lines delete `i`/`j`/`delay` variables stored per synapse.
                    # They need to be called after before_run_synapses_push_spikes
                    # codeobjects, therefore just before the network creation, so I put
                    # them here
                    # TODO: Move this into before_run_synapses_push_spikes
                    for synaptic_pre, delete in self.delete_synaptic_pre.items():
                        if delete:
                            code = f'''
                                dev{synaptic_pre}.clear();
                                dev{synaptic_pre}.shrink_to_fit();
                            '''
                            main_lines.extend(stripped_deindented_lines(code))
                    for synaptic_post, delete in self.delete_synaptic_post.items():
                        if delete:
                            code = f'''
                                dev{synaptic_post}.clear();
                                dev{synaptic_post}.shrink_to_fit();
                            '''
                            main_lines.extend(stripped_deindented_lines(code))
                    for synaptic_delay, delete in self.delete_synaptic_delay.items():
                        if delete:
                            code = f'''
                                dev{synaptic_delay}.clear();
                                dev{synaptic_delay}.shrink_to_fit();
                            '''
                            main_lines.extend(stripped_deindented_lines(code))
                # The actual network code
                main_lines.extend(netcode)
                # Increment run counter
                run_counter += 1
            elif func=='set_by_constant':
                arrayname, value, is_dynamic = args
                size_str = f"{arrayname}.size()" if is_dynamic else f"_num_{arrayname}"
                rendered_value = CPPNodeRenderer().render_expr(repr(value))
                pointer_arrayname = f"dev{arrayname}"
                if arrayname.endswith('space'):  # eventspace
                    pointer_arrayname += f'[current_idx{arrayname}]'
                if is_dynamic:
                    pointer_arrayname = f"thrust::raw_pointer_cast(&dev{arrayname}[0])"
                # Set on host
                code = f'''
                    for(int i=0; i<{size_str}; i++)
                    {{
                        {arrayname}[i] = {rendered_value};
                    }}
                '''
                if arrayname not in self.variables_on_host_only:
                    # Copy to device
                    code += f'''
                        CUDA_SAFE_CALL(
                            cudaMemcpy(
                                {pointer_arrayname},
                                &{arrayname}[0],
                                sizeof({arrayname}[0])*{size_str},
                                cudaMemcpyHostToDevice
                            )
                        );
                    '''
                main_lines.extend(stripped_deindented_lines(code))
            elif func=='set_by_single_value':
                arrayname, item, value = args
                pointer_arrayname = f"dev{arrayname}"
                if arrayname.endswith('space'):  # eventspace
                    pointer_arrayname += f'[current_idx{arrayname}]'
                if arrayname in self.dynamic_arrays.values():
                    pointer_arrayname = f"thrust::raw_pointer_cast(&dev{arrayname}[0])"
                # Set on host
                code = f"{arrayname}[{item}] = {value};"
                if arrayname not in self.variables_on_host_only:
                    # Copy to device
                    code += f'''
                        CUDA_SAFE_CALL(
                            cudaMemcpy(
                                {pointer_arrayname} + {item},
                                &{arrayname}[{item}],
                                sizeof({arrayname}[{item}]),
                                cudaMemcpyHostToDevice
                            )
                        );
                    '''
                main_lines.extend(stripped_deindented_lines(code))
            elif func=='set_by_array':
                arrayname, staticarrayname, is_dynamic = args
                size_str = "_num_" + arrayname
                if is_dynamic:
                    size_str = arrayname + ".size()"
                pointer_arrayname = f"dev{arrayname}"
                if arrayname in self.dynamic_arrays.values():
                    pointer_arrayname = f"thrust::raw_pointer_cast(&dev{arrayname}[0])"
                # Set on host
                code = f'''
                    for(int i=0; i<_num_{staticarrayname}; i++)
                    {{
                        {arrayname}[i] = {staticarrayname}[i];
                    }}
                '''
                if arrayname not in self.variables_on_host_only:
                    # Copy to device
                    code += f'''
                        CUDA_SAFE_CALL(
                            cudaMemcpy(
                                {pointer_arrayname},
                                &{arrayname}[0],
                                sizeof({arrayname}[0])*{size_str},
                                cudaMemcpyHostToDevice
                            )
                        );
                    '''
                main_lines.extend(stripped_deindented_lines(code))
            elif func=='set_array_by_array':
                arrayname, staticarrayname_index, staticarrayname_value = args
                # Set on host
                code = f'''
                    for(int i=0; i<_num_{staticarrayname_index}; i++)
                    {{
                        {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                    }}
                '''
                if arrayname not in self.variables_on_host_only:
                    # Copy to device
                    code += f'''
                        CUDA_SAFE_CALL(
                            cudaMemcpy(
                                dev{arrayname},
                                &{arrayname}[0],
                                sizeof({arrayname}[0])*_num_{arrayname},
                                cudaMemcpyHostToDevice
                            )
                        );
                    '''
                main_lines.extend(stripped_deindented_lines(code))
            elif func=='resize_array':
                array_name, new_size = args
                code = f'''
                    {array_name}.resize({new_size});
                    THRUST_CHECK_ERROR(dev{array_name}.resize({new_size}));
                '''
                main_lines.extend(stripped_deindented_lines(code))
            elif func=='insert_code':
                main_lines.append(args)
            elif func=='start_run_func':
                name, include_in_parent = args
                if include_in_parent:
                    main_lines.append(f'{name}();')
                main_lines = []
                procedures.append((name, main_lines))
            elif func=='end_run_func':
                name, include_in_parent = args
                name, main_lines = procedures.pop(-1)
                runfuncs[name] = main_lines
                name, main_lines = procedures[-1]
            elif func=='seed':
                seed = args
                if seed is None:
                    # draw random seed in range of possible uint64 numbers
                    seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
                main_lines.append(f'random_number_buffer.set_seed({seed!r}ULL);')
            else:
                raise NotImplementedError("Unknown main queue function type "+func)

        # Store the GPU ID and it's compute capability. The latter can be overwritten in
        # self.generate_makefile() via preferences
        self.gpu_id, self.compute_capability = select_gpu()

        # generate the finalisations
        for codeobj in self.code_objects.values():
            if hasattr(codeobj.code, 'main_finalise'):
                main_lines.append(codeobj.code.main_finalise)

        user_headers = self.headers + prefs['codegen.cpp.headers']
        main_tmp = self.code_object_class().templater.main(None, None,
                                                           gpu_id=self.gpu_id,
                                                           main_lines=main_lines,
                                                           code_lines=self.code_lines,
                                                           code_objects=self.code_objects.values(),
                                                           report_func=self.report_func,
                                                           dt=float(defaultclock.dt),
                                                           user_headers=user_headers,
                                                           gpu_heap_size=prefs['devices.cuda_standalone.cuda_backend.gpu_heap_size']
                                                          )
        writer.write('main.cu', main_tmp)

    def generate_codeobj_source(self, writer):
        code_object_defs = defaultdict(list)
        host_parameters = defaultdict(list)
        kernel_parameters = defaultdict(list)
        kernel_constants = defaultdict(list)
        c_float_dtype = c_data_type(prefs['core.default_float_dtype'])
        c_int_dtype = 'unsigned int'
        # Generate data for non-constant values
        for codeobj in self.code_objects.values():
            code_object_defs_lines = []
            code_object_defs_lines_host_only = []
            host_parameters_lines = []
            kernel_parameters_lines = []
            kernel_constants_lines = []
            additional_code = []
            number_elements = ""
            if hasattr(codeobj, 'owner') and hasattr(codeobj.owner, '_N') and codeobj.owner._N != 0:
                number_elements = str(codeobj.owner._N)
            else:
                number_elements = "_N"
            # We need the functions to be sorted by keys for reproducable rng with a
            # given seed: For codeobjects that are only run once, we generate the random
            # numbers using the curand host API. For that, we insert code into the
            # `additional_code` block in the host function. If we use multiple random
            # function in one codeobject (e.g. rand() and randn()), the order in which
            # they are generated can differ between two codeobjects, which makes the
            # brian2.tests.test_neurongroup.test_random_values_fixed_seed fail.
            for k, v in sorted(codeobj.variables.items()):
                if k == 'dt' and prefs['core.default_float_dtype'] == np.float32:
                    # use the double-precision array versions for dt as kernel arguments
                    # they are cast to single-precision scalar dt in scalar_code
                    v = v.real_var

                # code objects which only run once
                if k in ["rand", "randn", "poisson"] and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create_generator":
                    curand_suffix = ''
                    if prefs['core.default_float_dtype'] == np.float64:
                        curand_suffix = 'Double'
                    if k == "randn":
                        num_calls = codeobj.rng_calls["randn"]
                        code = f'''
                            // Genenerate an array of random numbers on the device
                            // Make sure we generate an even number of random numbers
                            int32_t _randn_N = ({number_elements} % 2 == 0) ? {number_elements} : {number_elements} + 1;
                            {c_float_dtype}* dev_array_randn;
                            CUDA_SAFE_CALL(
                                cudaMalloc(
                                    (void**)&dev_array_randn,
                                    sizeof({c_float_dtype})*_randn_N*{num_calls}
                                )
                            );
                            CUDA_SAFE_CALL(
                                curandGenerateNormal{curand_suffix}(
                                    curand_generator,
                                    dev_array_randn,
                                    _randn_N*{num_calls},
                                    0,  // mean
                                    1   // stddev
                                )
                            );
                        '''
                        additional_code.append(code)
                        kernel_parameters_lines.append(
                            f"{c_float_dtype}* _ptr_array_{codeobj.name}_randn"
                        )
                        host_parameters_lines.append("dev_array_randn")
                    elif k == "rand":
                        num_calls = codeobj.rng_calls["rand"]
                        code = f'''
                            // Genenerate an array of random numbers on the device
                            // Make sure we generate an even number of random numbers
                            int32_t _rand_N = ({number_elements} % 2 == 0) ? {number_elements} : {number_elements} + 1;
                            {c_float_dtype}* dev_array_rand;
                            CUDA_SAFE_CALL(
                                cudaMalloc(
                                    (void**)&dev_array_rand,
                                    sizeof({c_float_dtype})*_rand_N*{num_calls}
                                )
                            );
                            CUDA_SAFE_CALL(
                                curandGenerateUniform{curand_suffix}(
                                    curand_generator,
                                    dev_array_rand,
                                    _rand_N*{num_calls}
                                )
                            );
                        '''
                        additional_code.append(code)
                        kernel_parameters_lines.append(
                            f"{c_float_dtype}* _ptr_array_{codeobj.name}_rand"
                        )
                        host_parameters_lines.append("dev_array_rand")
                    elif k == "poisson":
                        # We are assuming that there can be at most one poisson call per expression,
                        # else brian2 should raise a NotImplementedError due to multiple stateful function calls.
                        assert len(codeobj.poisson_lamdas) < 2, report_issue_message
                        if len(codeobj.poisson_lamdas) == 0:
                            ### On-the-fly poisson number generation (curand device API)
                            # If we have a poisson function call and no entry in
                            # `poisson_lamdas`, we must have a variable lamda and are
                            # using on-the-fly RNG We don't need to add any code, we
                            # will use the device implementation defined in
                            # cuda_generator.py
                            assert codeobj.needs_curand_states, report_issue_message
                        else:  # len(codeobj.poisson_lamdas) == 1
                            ### Pregenerated poisson number (curand host API)
                            # There only one poisson call, hence we have only `poisson_0`
                            poisson_name = 'poisson_0'
                            # curand generates `unsigned int`, we cast it to `int32_t` in our `_poisson` implementation
                            num_calls = codeobj.rng_calls[poisson_name]
                            lamda = codeobj.poisson_lamdas[poisson_name]
                            code = f'''
                                // Genenerate an array of random numbers on the device
                                // Make sure we generate an even number of random numbers
                                int32_t _{poisson_name}_N = ({number_elements} % 2 == 0) ? {number_elements} : {number_elements} + 1;
                                {c_int_dtype}* dev_array_{poisson_name};
                                CUDA_SAFE_CALL(
                                    cudaMalloc(
                                        (void**)&dev_array_{poisson_name},
                                        sizeof(unsigned int)*_{poisson_name}_N*{num_calls}
                                    )
                                );
                                CUDA_SAFE_CALL(
                                    curandGeneratePoisson(
                                        curand_generator,
                                        dev_array_{poisson_name},
                                        _{poisson_name}_N*{num_calls},
                                        {lamda}
                                    )
                                );
                            '''
                            additional_code.append(code)
                            kernel_parameters_lines.append(
                                f"{c_int_dtype}* _ptr_array_{codeobj.name}_{poisson_name}"
                            )
                            host_parameters_lines.append(f"dev_array_{poisson_name}")
                # Clock variables (t, dt, timestep)
                elif hasattr(v, 'owner') and isinstance(v.owner, Clock):
                    # Clocks only run on the host and the corresponding device variables are copied
                    # to the device only once in the beginning and in the end of a simulation.
                    # Therefore, we pass clock variables (t, dt, timestep) by value as kernel
                    # parameters whenever they are needed on the device. These values are translated
                    # into pointers in CUDACodeGenerator.determine_keywords(), such that they can be
                    # used in scalar/vector code.
                    arrayname = self.get_array_name(v)
                    dtype = c_data_type(v.dtype)
                    host_parameters_lines.append(
                        f"{arrayname}[0]")
                    kernel_parameters_lines.append(
                        f"const {dtype} _value{arrayname}")
                # ArrayVariables (dynamic and not)
                elif isinstance(v, ArrayVariable):
                    # These templates run on the host
                    host_codeobjects = ['synapses_create_generator',
                                        'synapses_create_array']
                    # These templates run on host and device (e.g. synapses_push_spikes
                    # has a before_run codeobject that runs on host only while the run
                    # codeobject runs on the device)
                    host_and_device_codeobjects = ['synapses_push_spikes',
                                                   'spatialstateupdate']
                    prefixes = ['dev']
                    if codeobj.template_name in host_codeobjects:
                        prefixes = ['']
                    elif codeobj.template_name in host_and_device_codeobjects:
                        # Start with no prefix, such that `_num{array}` variables are
                        # determined from the host vector if array is dynamic
                        prefixes = ['', 'dev']
                    for n_prefix, prefix in enumerate(prefixes):
                        try:
                            dyn_array_name = self.get_array_name(v,
                                                                 access_data=False,
                                                                 prefix=prefix)
                            array_name = self.get_array_name(v,
                                                             access_data=True,
                                                             prefix=prefix)
                            ptr_array_name = self.get_array_name(v,
                                                                 access_data=True,
                                                                 prefix='_ptr')
                            dtype = c_data_type(v.dtype)
                            if isinstance(v, DynamicArrayVariable):
                                if v.ndim == 1:

                                    code_object_defs_lines.append(
                                        f'{dtype}* const {array_name} = thrust::raw_pointer_cast(&{dyn_array_name}[0]);'
                                    )

                                    # Add host and kernel parameters only for device pointers
                                    if prefix == 'dev':
                                        # These lines are used to define the kernel call parameters, that
                                        # means only for codeobjects running on the device. The array names
                                        # always have a `_dev` prefix.
                                        host_parameters_lines.append(f"{array_name}")

                                        # These lines declare kernel parameters as the `_ptr` variables that
                                        # are used in `scalar_code` and `vector_code`.
                                        # TODO: here we should add const / __restrict and other optimizations
                                        #       for variables that are e.g. only read in the kernel
                                        kernel_parameters_lines.append(f"{dtype}* {ptr_array_name}")

                                    # Add size variables `_num{array}` only once and if
                                    # there are two prefixes, base it on host array
                                    # `{array}.size()`
                                    if len(prefixes) == 1 or prefix == '':
                                        code_object_defs_lines.append(
                                            f'const int _num{k} = {dyn_array_name}.size();'
                                        )
                                        host_parameters_lines.append(f"_num{k}")
                                        kernel_parameters_lines.append(f"const int _num{k}")

                            else:  # v is ArrayVariable but not DynamicArrayVariable
                                # Add host and kernel parameters only for device pointers
                                if prefix == 'dev':
                                    idx = ''
                                    if k.endswith('space'):
                                        bare_array_name = self.get_array_name(v)
                                        idx = f'[current_idx{bare_array_name}]'
                                    host_parameters_lines.append(f"{array_name}{idx}")
                                    kernel_parameters_lines.append(f'{dtype}* {ptr_array_name}')

                                # Add size variables `_num{array}` only once
                                if n_prefix == 0:
                                    code_object_defs_lines.append(f'const int _num{k} = {v.size};')
                                    kernel_constants_lines.append(f'const int _num{k} = {v.size};')

                        except TypeError:
                            pass

            # This rand stuff got a little messy... we pass a device pointer as kernel variable and have a hash define for rand() -> _ptr_..._rand[]
            # The device pointer is advanced every clock cycle in rand.cu and reset when the random number buffer is refilled (also in rand.cu)
            # TODO can we just include this in the k == 'rand' test above?
            # RAND
            if codeobj.rng_calls["rand"] >= 1 and codeobj.runs_every_tick:
                host_parameters_lines.append(f"dev_{codeobj.name}_rand")
                kernel_parameters_lines.append(
                    f"{c_float_dtype}* _ptr_array_{codeobj.name}_rand"
                )
            # RANDN
            if codeobj.rng_calls["randn"] >= 1 and codeobj.runs_every_tick:
                host_parameters_lines.append(f"dev_{codeobj.name}_randn")
                kernel_parameters_lines.append(
                    f"{c_float_dtype}* _ptr_array_{codeobj.name}_randn"
                )
            # POISSON (with scalar lamda)
            # Here, we don't use the hash define as for rand/n, instead we pass the
            # kernel paramter (_ptr...) directly to the _poisson function which returns
            # the correct element
            # TODO: We could do the same for rand/n and get rid of the hash define hack
            for rng_type in codeobj.rng_calls.keys():
                if rng_type not in ["rand", "randn"] and codeobj.runs_every_tick:
                    assert rng_type.startswith("poisson")
                    if codeobj.rng_calls[rng_type] >= 1:
                        host_parameters_lines.append(
                            f"dev_{codeobj.name}_{rng_type}"
                        )
                        kernel_parameters_lines.append(
                            f"{c_int_dtype}* _ptr_array_{codeobj.name}_{rng_type}"
                        )

            # Sometimes an array is referred to by to different keys in our
            # dictionary -- make sure to never add a line twice
            for line in code_object_defs_lines:
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)
            for line in host_parameters_lines:
                if not line in host_parameters[codeobj.name]:
                    host_parameters[codeobj.name].append(line)
            for line in kernel_parameters_lines:
                if not line in kernel_parameters[codeobj.name]:
                    kernel_parameters[codeobj.name].append(line)
            for line in chain(kernel_constants_lines):
                if not line in kernel_constants[codeobj.name]:
                    kernel_constants[codeobj.name].append(line)

            for line in additional_code:
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)

        # Generate the code objects
        for codeobj in self.code_objects.values():
            ns = codeobj.variables

            def _replace_constants_and_parameters(code):
                # HOST_CONSTANTS are equivalent to C++ Standalone's CONSTANTS
                code = code.replace('%HOST_CONSTANTS%', '\n\t\t'.join(code_object_defs[codeobj.name]))
                # KERNEL_CONSTANTS are the same for inside device kernels
                code = code.replace('%KERNEL_CONSTANTS%', '\n\t'.join(kernel_constants[codeobj.name]))
                # HOST_PARAMETERS are parameters that device kernels are called with from host code
                code = code.replace('%HOST_PARAMETERS%', ',\n\t\t\t'.join(host_parameters[codeobj.name]))
                # KERNEL_PARAMETERS are the same names of the same parameters inside the device kernels
                code = code.replace('%KERNEL_PARAMETERS%', ',\n\t'.join(kernel_parameters[codeobj.name]))
                code = code.replace('%CODEOBJ_NAME%', codeobj.name)
                return code

            # Before/after run code
            for block in codeobj.before_after_blocks:
                cu_code = getattr(codeobj.code, block + '_cu_file')
                cu_code = self.freeze(cu_code, ns)
                cu_code = _replace_constants_and_parameters(cu_code)
                h_code = getattr(codeobj.code, block + '_h_file')
                writer.write('code_objects/' + block + '_' + codeobj.name + '.cu',
                             cu_code)
                writer.write('code_objects/' + block + '_' + codeobj.name + '.h',
                             h_code)

            # Main code
            # TODO: fix these freeze/HOST_CONSTANTS hacks somehow - they work but not elegant.
            code = self.freeze(codeobj.code.cu_file, ns)

            if len(host_parameters[codeobj.name]) == 0:
                host_parameters[codeobj.name].append("0")
                kernel_parameters[codeobj.name].append("int dummy")

            code = _replace_constants_and_parameters(code)

            # substitue in generated code double types with float types in
            # single-precision mode
            if prefs['core.default_float_dtype'] == np.float32:
                # cast time differences (double) to float in event-drive updates
                sub = 't - lastupdate'
                if sub in code:
                    code = code.replace(sub, f'float({sub})')
                    logger.debug(f"Replaced {sub} with float({sub}) in {codeobj.name}")
                # replace double-precision floating-point literals with their
                # single-precision version (e.g. `1.0` -> `1.0f`)
                code = replace_floating_point_literals(code)
                logger.debug(
                    f"Replaced floating point literals by single precision version "
                    f"(appending `f`) in {codeobj.name}."
                )

            writer.write('code_objects/'+codeobj.name+'.cu', code)
            writer.write('code_objects/'+codeobj.name+'.h', codeobj.code.h_file)

    def generate_rand_source(self, writer):
        # Device side binomial functions and poisson functions with variables lambda use
        # the curand device api. The curand states (one per thread executed in parallel)
        # are initialized in rand.cu, where as many curand states are initialized as the
        # size of the largest codeobject with curand device api calls.
        needed_number_curand_states = {}
        for co in (self.codeobjects_with_rng["device_api"]["every_tick"]
                   + self.codeobjects_with_rng["device_api"]["single_tick"]):
            co_name = co.owner.name
            if co_name not in needed_number_curand_states:
                if isinstance(co.owner, Synapses):
                    # this is the pointer to the synapse object's N, which is a
                    # null pointer before synapses are generated and an int ptr
                    # after synapse generation (used to test if synapses
                    # already generated or not)
                    N_ptr = f'_array_{co_name}_N'
                    N_value = N_ptr + '[0]'
                else:
                    N_ptr = None
                    N_value = co.owner._N
                needed_number_curand_states[co_name] = (N_ptr, N_value)

        rand_tmp = self.code_object_class().templater.rand(None, None,
                                                           codeobjects_with_rng_per_run=self.codeobjects_with_rng["host_api"]["per_run"],
                                                           all_poisson_lamdas=self.all_poisson_lamdas,
                                                           needed_number_curand_states=needed_number_curand_states,
                                                           number_run_calls=len(self.codeobjects_with_rng["host_api"]["per_run"]),
                                                           profiled=self.enable_profiling,
                                                           curand_float_type=c_data_type(prefs['core.default_float_dtype']))
        writer.write('rand.*', rand_tmp)

    def copy_source_files(self, writer, directory):
        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(os.path.split(inspect.getsourcefile(self.code_object_class()))[0],
                                    'brianlib')
        brianlib_files = copy_directory(brianlib_dir, os.path.join(directory, 'brianlib'))
        for file in brianlib_files:
            if file.lower().endswith('.cpp'):
                writer.source_files.add('brianlib/'+file)
            if file.lower().endswith('.cu'):
                writer.source_files.add('brianlib/'+file)
            elif file.lower().endswith('.h'):
                writer.header_files.add('brianlib/'+file)

    def generate_network_source(self, writer):
        maximum_run_time = self._maximum_run_time
        if maximum_run_time is not None:
            maximum_run_time = float(maximum_run_time)
        network_tmp = self.code_object_class().templater.network(None, None,
                                                                 maximum_run_time=maximum_run_time,
                                                                 eventspace_arrays=self.eventspace_arrays,
                                                                 spikegenerator_eventspaces=self.spikegenerator_eventspaces)
        writer.write('network.*', network_tmp)

    def generate_synapses_classes_source(self, writer):
        synapses_classes_tmp = self.code_object_class().templater.synapses_classes(None, None)
        writer.write('synapses_classes.*', synapses_classes_tmp)

    def generate_run_source(self, writer):
        run_tmp = self.code_object_class().templater.run(None, None, run_funcs=self.runfuncs,
                                                        code_objects=self.code_objects.values(),
                                                        user_headers=self.headers,
                                                        array_specs=self.arrays,
                                                        clocks=self.clocks)
        writer.write('run.*', run_tmp)

    def generate_makefile(self, writer, cpp_compiler, cpp_compiler_flags, cpp_linker_flags, debug, disable_asserts):
        available_gpu_arch_flags = (
            '--gpu-architecture', '-arch', '--gpu-code', '-code', '--generate-code',
            '-gencode'
        )
        nvcc_compiler_flags = prefs.devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc
        gpu_arch_flags = []
        for flag in nvcc_compiler_flags:
            if flag.startswith(available_gpu_arch_flags):
                gpu_arch_flags.append(flag)
                nvcc_compiler_flags.remove(flag)
            elif flag.startswith(('-w', '--disable-warnings')):
                # add the flage to linker flags, else linking will give warnings
                cpp_linker_flags.append(flag)
        # Make the linker options (meant to be passed to `gcc`) compatible with `nvcc`
        for i, flag in enumerate(cpp_linker_flags):
            if flag.startswith('-Wl,'):
                # -Wl,<option> passes <option> directly to linker
                # for gcc `-Wl,<option>`, for nvcc `-Xlinker "<option>"`
                cpp_linker_flags[i] = flag.replace('-Wl,', '-Xlinker ')
        # Check if compute capability was set manually via preference
        compute_capability_pref = prefs.devices.cuda_standalone.cuda_backend.compute_capability
        # If GPU architecture was set via `extra_compile_args_nvcc` and
        # `compute_capability`, ignore `compute_capability`
        if gpu_arch_flags and compute_capability_pref is not None:
            logger.warn(
                "GPU architecture for compilation was specified via "
                "`prefs.devices.cuda_standalone.cuda_backend.compute_capability` and "
                "`prefs.devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc`. "
                "`prefs.devices.cuda_standalone.cuda_backend.compute_capability` will be ignored. "
                "To get rid of this warning, set "
                "`prefs.devices.cuda_standalone.brian_backend.compute_capability` to it's default "
                "value `None`"
            )
            # Ignore compute capability of chosen GPU and the one manually set via
            # `compute_capability` preferences.
            self.compute_capability = None
        # If GPU architecture was set only via `extra_compile_args_nvcc`, use that
        elif gpu_arch_flags:
            # Ignore compute capability of chosen GPU
            self.compute_capability = None
        # If GPU architecture was set only via `compute_capability` prefs, use that
        elif compute_capability_pref is not None:
            self.compute_capability = compute_capability_pref
        # If compute_capability wasn't set manually, the one from the chosen GPU is used
        # (stored in self.compute_capability, see self.generate_main_source())

        if self.compute_capability is not None:
            # check if compute capability is supported
            if self.compute_capability < self.minimal_compute_capability:
                raise NotImplementedError(
                    f"Compute capability `{self.compute_capability}` is not supported. "
                    f"Minimal supported compute capability is "
                    f"`{self.minimal_compute_capability}`."
                )

        # If GPU architecture is detected automatically or set via `compute_capability`
        # prefs, we still need to add it as a compile argument
        if not gpu_arch_flags:
            # Turn float (3.5) into string ("35")
            compute_capability_str = ''.join(str(self.compute_capability).split('.'))
            gpu_arch_flags.append(f"-arch=sm_{compute_capability_str}")

        # Log compiled GPU architecture
        if self.compute_capability is None:
            logger.info(
                f"Compiling device code with manually set architecture flags "
                f"({gpu_arch_flags}). Be aware that the minimal supported compute "
                f"capability is {self.minimal_compute_capability} "
                "(we are not checking your compile flags)"
            )
        else:
            logger.info(
                f"Compiling device code for compute capability "
                f"{self.compute_capability} (compiler flags: {gpu_arch_flags})"
            )

        nvcc_path = get_nvcc_path()
        if cpp_compiler=='msvc':
            # Check CPPStandaloneDevice.generate_makefile() for how to do things
            raise RuntimeError("Windows is currently not supported. See https://github.com/brian-team/brian2cuda/issues/225")
        else:
            # Generate the makefile
            if os.name=='nt':
                rm_cmd = 'del *.o /s\n\tdel main.exe $(DEPS)'
            else:
                rm_cmd = 'rm $(OBJS) $(PROGRAM) $(DEPS)'

            if debug:
                compiler_debug_flags = '-g -DDEBUG -G -DTHRUST_DEBUG'
                linker_debug_flags = '-g -G'
            else:
                compiler_debug_flags = ''
                linker_debug_flags = ''

            if disable_asserts:
                # NDEBUG precompiler macro disables asserts (both for C++ and CUDA)
                nvcc_compiler_flags += ['-NDEBUG']

            makefile_tmp = self.code_object_class().templater.makefile(
                None, None,
                source_files=' '.join(sorted(writer.source_files)),
                header_files=' '.join(sorted(writer.header_files)),
                cpp_compiler_flags=' '.join(cpp_compiler_flags),
                compiler_debug_flags=compiler_debug_flags,
                linker_debug_flags=linker_debug_flags,
                cpp_linker_flags=' '.join(cpp_linker_flags),
                nvcc_compiler_flags=' '.join(nvcc_compiler_flags),
                gpu_arch_flags=' '.join(gpu_arch_flags),
                nvcc_path=nvcc_path,
                rm_cmd=rm_cmd,
            )
            writer.write('makefile', makefile_tmp)

    def build(self, directory='output',
              compile=True, run=True, debug=False, clean=False,
              with_output=True, disable_asserts=False,
              additional_source_files=None,
              run_args=None, direct_call=True, **kwds):
        '''
        Build the project

        TODO: more details

        Parameters
        ----------
        directory : str, optional
            The output directory to write the project to, any existing files
            will be overwritten. If the given directory name is ``None``, then
            a temporary directory will be used (used in the test suite to avoid
            problems when running several tests in parallel). Defaults to
            ``'output'``.
        compile : bool, optional
            Whether or not to attempt to compile the project. Defaults to
            ``True``.
        run : bool, optional
            Whether or not to attempt to run the built project if it
            successfully builds. Defaults to ``True``.
        debug : bool, optional
            Whether to compile in debug mode. Defaults to ``False``.
        with_output : bool, optional
            Whether or not to show the ``stdout`` of the built program when run.
            Output will be shown in case of compilation or runtime error.
            Defaults to ``True``.
        clean : bool, optional
            Whether or not to clean the project before building. Defaults to
            ``False``.
        additional_source_files : list of str, optional
            A list of additional ``.cu`` files to include in the build.
        direct_call : bool, optional
            Whether this function was called directly. Is used internally to
            distinguish an automatic build due to the ``build_on_run`` option
            from a manual ``device.build`` call.
        '''
        if self.build_on_run and direct_call:
            raise RuntimeError('You used set_device with build_on_run=True '
                               '(the default option), which will automatically '
                               'build the simulation at the first encountered '
                               'run call - do not call device.build manually '
                               'in this case. If you want to call it manually, '
                               'e.g. because you have multiple run calls, use '
                               'set_device with build_on_run=False.')
        if self.has_been_run:
            raise RuntimeError('The network has already been built and run '
                               'before. To build several simulations in '
                               'the same script, call "device.reinit()" '
                               'and "device.activate()". Note that you '
                               'will have to set build options (e.g. the '
                               'directory) and defaultclock.dt again.')

        renames = {'project_dir': 'directory',
                   'compile_project': 'compile',
                   'run_project': 'run'}
        if len(kwds):
            msg = ''
            for kwd in kwds:
                if kwd in renames:
                    msg += ("Keyword argument '%s' has been renamed to "
                            "'%s'. ") % (kwd, renames[kwd])
                else:
                    msg += f"Unknown keyword argument '{kwd}'. "
            raise TypeError(msg)

        if debug and disable_asserts:
            logger.warn("You have disabled asserts in debug mode. Are you sure this is what you wanted to do?")

        if additional_source_files is None:
            additional_source_files = []
        if run_args is None:
            run_args = []
        if directory is None:
            directory = tempfile.mkdtemp(prefix='brian_standalone_')
        self.project_dir = directory
        ensure_directory(directory)

        # Determine compiler flags and directories
        cpp_compiler, cpp_default_extra_compile_args = get_compiler_and_args()
        extra_compile_args = self.extra_compile_args + cpp_default_extra_compile_args
        extra_link_args = self.extra_link_args + prefs['codegen.cpp.extra_link_args']

        codeobj_define_macros = [macro for codeobj in
                                 self.code_objects.values()
                                 for macro in
                                 codeobj.compiler_kwds.get('define_macros', [])]
        define_macros = (self.define_macros +
                         prefs['codegen.cpp.define_macros'] +
                         codeobj_define_macros)

        codeobj_include_dirs = [include_dir for codeobj in
                                self.code_objects.values()
                                for include_dir in
                                codeobj.compiler_kwds.get('include_dirs', [])]
        include_dirs = (self.include_dirs +
                        prefs['codegen.cpp.include_dirs'] +
                        codeobj_include_dirs)

        codeobj_library_dirs = [library_dir for codeobj in
                                self.code_objects.values()
                                for library_dir in
                                codeobj.compiler_kwds.get('library_dirs', [])]
        library_dirs = (self.library_dirs +
                        prefs['codegen.cpp.library_dirs'] +
                        codeobj_library_dirs)

        codeobj_runtime_dirs = [runtime_dir for codeobj in
                                self.code_objects.values()
                                for runtime_dir in
                                codeobj.compiler_kwds.get('runtime_library_dirs', [])]
        runtime_library_dirs = (self.runtime_library_dirs +
                                prefs['codegen.cpp.runtime_library_dirs'] +
                                codeobj_runtime_dirs)

        codeobj_libraries = [library for codeobj in
                             self.code_objects.values()
                             for library in
                             codeobj.compiler_kwds.get('libraries', [])]
        libraries = (self.libraries +
                     prefs['codegen.cpp.libraries'] +
                     codeobj_libraries)

        cpp_compiler_obj = ccompiler.new_compiler(compiler=cpp_compiler)
        cpp_compiler_flags = (ccompiler.gen_preprocess_options(define_macros,
                                                           include_dirs) +
                          extra_compile_args)
        cpp_linker_flags = (ccompiler.gen_lib_options(cpp_compiler_obj,
                                                  library_dirs=library_dirs,
                                                  runtime_library_dirs=runtime_library_dirs,
                                                  libraries=libraries) +
                        extra_link_args)

        codeobj_source_files = [source_file for codeobj in
                                self.code_objects.values()
                                for source_file in
                                codeobj.compiler_kwds.get('sources', [])]
        additional_source_files += codeobj_source_files

        for d in ['code_objects', 'results', 'static_arrays']:
            ensure_directory(os.path.join(directory, d))

        self.writer = CUDAWriter(directory)

        logger.diagnostic("Writing CUDA standalone project to directory "+os.path.normpath(directory))

        self.write_static_arrays(directory)

        # Check that all names are globally unique
        names = [obj.name for net in self.networks for obj in net.objects]
        non_unique_names = [name for name, count in Counter(names).items()
                            if count > 1]
        if len(non_unique_names):
            formatted_names = ', '.join(f"'{name}'"
                                        for name in non_unique_names)
            raise ValueError('All objects need to have unique names in '
                             'standalone mode, the following name(s) were used '
                             'more than once: %s' % formatted_names)

        net_synapses = [s for net in self.networks
                        for s in net.objects
                        if isinstance(s, Synapses)]

        # Collect all multisynaptic indices in all Synapses (that have them)
        self.multisyn_vars = []
        for syn in net_synapses:
            if syn.multisynaptic_index is not None:
                self.multisyn_vars.append(syn.variables[syn.multisynaptic_index])

        # Collect all variables that are stored on host only and should not be copied
        # from device to host at all (e.g. when set by constant or array or at the end
        # of the simulation)
        self.variables_on_host_only = []
        for var, varname in self.arrays.items():
            try:
                is_mon = isinstance(var.owner, (StateMonitor, SpikeMonitor, EventMonitor))
            except ReferenceError:
                # some variable ownders are weakreference that don't exist anymore
                # https://github.com/brian-team/brian2cuda/issues/296#issuecomment-1145085524
                continue
            if is_mon and var.name == 'N':
                # The size variable of monitors is managed on host via device vectors
                self.variables_on_host_only.append(varname)
            if var.name in ('t', 'dt', 'timestep'):
                # We manage time variables on host and pass them by value to kernels
                self.variables_on_host_only.append(varname)
        for var, varname in self.dynamic_arrays.items():
            varnames = ['_synaptic_pre', '_synaptic_post']
            try:
                # TODO: Manage Spike- & EventMonitor t also on host (modify template)
                is_monitor_t = isinstance(var.owner, StateMonitor) and var.name == 't'
            except ReferenceError:
                continue
            if var in self.multisyn_vars or var.name in varnames or is_monitor_t:
                self.variables_on_host_only.append(varname)

            if var.name == 'delay':
                # By default, delete all delay variables after
                # before_run_synapses_push_spikes, but only those that weren't already
                # set to be deleted during variable set calls (in
                # self.variableview_set_with_index_array and self.fill_with_array)
                if varname not in self.delete_synaptic_delay:
                    self.delete_synaptic_delay[varname] = True
                    # This avoids copying the delted delay array from device to host
                    # at the end of the simulation
                    self.variables_on_host_only.append(varname)

        self.generate_main_source(self.writer)

        # Create lists of codobjects using rand, randn, poisson or binomial across all
        # runs (needed for variable declarations).
        #   - Variables needed for device side rand/randn/poisson are declared in objects.cu:
        #     codeobjects_with_rng["host_api"]["all_runs"]['rand'/'rand'/'poisson'] are needed in `generate_objects_source`
        #   - Variables needed for device side binomial functions are initialized in rand.cu:
        #     codeobjects_with_rng["device_api"]["every_tick"] is needed in `generate_rand_source`
        for run_codeobj in self.codeobjects_with_rng["host_api"]["per_run"]:
            for key in run_codeobj.keys():
                # keys: 'rand', 'randn', 'poisson-<idx>'
                self.codeobjects_with_rng["host_api"]["all_runs"][key].extend(run_codeobj[key])

        self.generate_codeobj_source(self.writer)

        self.generate_objects_source(self.writer, self.arange_arrays,
                                     net_synapses, self.static_array_specs,
                                     self.networks)
        self.generate_network_source(self.writer)
        self.generate_synapses_classes_source(self.writer)
        self.generate_run_source(self.writer)
        self.generate_rand_source(self.writer)
        self.copy_source_files(self.writer, directory)

        self.writer.source_files.update(additional_source_files)

        self.generate_makefile(self.writer, cpp_compiler,
                               cpp_compiler_flags,
                               cpp_linker_flags,
                               debug,
                               disable_asserts)

        # Not sure what the best place is to call Network.after_run -- at the
        # moment the only important thing it does is to clear the objects stored
        # in magic_network. If this is not done, this might lead to problems
        # for repeated runs of standalone (e.g. in the test suite).
        for net in self.networks:
            net.after_run()

        logger.info("Using the following preferences for CUDA standalone:")
        for pref_name in prefs:
            if "devices.cuda_standalone" in pref_name:
                logger.info(f"\t{pref_name} = {prefs[pref_name]}")

        if compile:
            self.compile_source(directory, cpp_compiler, debug, clean)
            if run:
                self.run(directory, with_output, run_args)

    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=False, level=0, **kwds):
        ###################################################
        ### This part is copied from CPPStandaoneDevice ###
        ###################################################
        self.networks.add(net)
        if kwds:
            logger.warn(('Unsupported keyword argument(s) provided for run: '
                         '%s') % ', '.join(kwds.keys()))
        # We store this as an instance variable for later access by the
        # `code_object` method
        self.enable_profiling = profile

        # Allow setting `profile` in the `set_device` call (used e.g. in brian2cuda
        # SpeedTest configurations)
        if 'profile' in self.build_options:
            build_profile = self.build_options.pop('profile')
            if build_profile:
                self.enable_profiling = True

        all_objects = net.sorted_objects
        net._clocks = {obj.clock for obj in all_objects}
        t_end = net.t+duration
        for clock in net._clocks:
            clock.set_interval(net.t, t_end)

        # Get the local namespace
        if namespace is None:
            namespace = get_local_namespace(level=level+2)

        net.before_run(namespace)

        self.clocks.update(net._clocks)
        net.t_ = float(t_end)

        # TODO: remove this horrible hack
        for clock in self.clocks:
            if clock.name=='clock':
                clock._name = '_clock'

        # Extract all the CodeObjects
        # Note that since we ran the Network object, these CodeObjects will be sorted into the right
        # running order, assuming that there is only one clock
        code_objects = []
        for obj in all_objects:
            if obj.active:
                for codeobj in obj._code_objects:
                    code_objects.append((obj.clock, codeobj))

        # Code for a progress reporting function
        standard_code = '''
        std::string _format_time(float time_in_s)
        {
            float divisors[] = {24*60*60, 60*60, 60, 1};
            char letters[] = {'d', 'h', 'm', 's'};
            float remaining = time_in_s;
            std::string text = "";
            int time_to_represent;
            for (int i =0; i < sizeof(divisors)/sizeof(float); i++)
            {
                time_to_represent = int(remaining / divisors[i]);
                remaining -= time_to_represent * divisors[i];
                if (time_to_represent > 0 || text.length())
                {
                    if(text.length() > 0)
                    {
                        text += " ";
                    }
                    text += (std::to_string(time_to_represent)+letters[i]);
                }
            }
            //less than one second
            if(text.length() == 0)
            {
                text = "< 1s";
            }
            return text;
        }
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                %STREAMNAME% << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                %STREAMNAME% << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << _format_time(elapsed);
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    %STREAMNAME% << ", estimated " << _format_time(remaining) << " remaining.";
                }
            }

            %STREAMNAME% << std::endl << std::flush;
        }
        '''
        if report is None:
            report_func = ''
        elif report == 'text' or report == 'stdout':
            report_func = standard_code.replace('%STREAMNAME%', 'std::cout')
        elif report == 'stderr':
            report_func = standard_code.replace('%STREAMNAME%', 'std::cerr')
        elif isinstance(report, str):
            report_func = '''
            void report_progress(const double elapsed, const double completed, const double start, const double duration)
            {
            %REPORT%
            }
            '''.replace('%REPORT%', report)
        else:
            raise TypeError(('report argument has to be either "text", '
                             '"stdout", "stderr", or the code for a report '
                             'function'))

        if report_func != '':
            if self.report_func != '' and report_func != self.report_func:
                raise NotImplementedError('The CUDA standalone device does not '
                                          'support multiple report functions, '
                                          'each run has to use the same (or '
                                          'none).')
            self.report_func = report_func

        if report is not None:
            report_call = 'report_progress'
        else:
            report_call = 'NULL'

        ##############################################################
        ### From here on the code differs from CPPStandaloneDevice ###
        ##############################################################

        # For each codeobject of this run check if it uses rand, randn, poisson or
        # binomials. Store these as attributes of the codeobject and create
        # lists of codeobjects that use rand, randn, poisson or binomials. This only
        # checks codeobject in the network, meaning only the ones running every
        # clock cycle.
        # self.codeobjects_with_rng["host_api"]["per_run"] is a list (one per run) of defaultdicts
        # with keys 'rand', 'randn', 'poisson_<idx>' and values being lists of
        # codeobjects.
        self.codeobjects_with_rng["host_api"]["per_run"].append(defaultdict(list))
        run_idx = -1  # last list index

        # Count random number ocurrences in codeobjects run every tick
        for _, co in code_objects:  # (clock, code_object)
            prepare_codeobj_code_for_rng(co)
            if co.rng_calls["rand"] > 0:
                self.codeobjects_with_rng["host_api"]["per_run"][run_idx]['rand'].append(co)
            if co.rng_calls["randn"] > 0:
                self.codeobjects_with_rng["host_api"]["per_run"][run_idx]['randn'].append(co)
            for poisson_name, lamda in co.poisson_lamdas.items():
                self.all_poisson_lamdas[co.name][poisson_name] = lamda
                self.codeobjects_with_rng["host_api"]["per_run"][run_idx][poisson_name].append(co)
            if co.needs_curand_states:
                if co not in self.codeobjects_with_rng["device_api"]["every_tick"]:
                    self.codeobjects_with_rng["device_api"]["every_tick"].append(co)

        # To profile SpeedTests, we need to be able to set `profile` in
        # `set_device`. Here we catch that case.
        if 'profile' in self.build_options:
            build_profile = self.build_options.pop('profile')
            if build_profile:
                self.enable_profiling = True

        # Generate the updaters
        run_lines = []
        run_lines.append(f'{net.name}.clear();')

        # create all random numbers needed for the next clock cycle
        for clock in net._clocks:
            run_lines.append(f'{net.name}.add(&{clock.name}, _run_random_number_buffer);')

        all_clocks = set()
        for clock, codeobj in code_objects:
            run_lines.append(f'{net.name}.add(&{clock.name}, _run_{codeobj.name});')
            all_clocks.add(clock)

        # Under some rare circumstances (e.g. a NeuronGroup only defining a
        # subexpression that is used by other groups (via linking, or recorded
        # by a StateMonitor) *and* not calculating anything itself *and* using a
        # different clock than all other objects) a clock that is not used by
        # any code object should nevertheless advance during the run. We include
        # such clocks without a code function in the network.
        for clock in net._clocks:
            if clock not in all_clocks:
                run_lines.append(f'{net.name}.add(&{clock.name}, NULL);')

        run_lines.extend(self.code_lines['before_network_run'])
        # run everything that is run on a clock
        run_lines.append(
            f'{net.name}.run({float(duration)!r}, {report_call}, {float(report_period)!r});'
        )
        run_lines.extend(self.code_lines['after_network_run'])
        # for multiple runs, the random number buffer needs to be reset
        run_lines.append('random_number_buffer.run_finished();')

        self.main_queue.append(('run_network', (net, run_lines)))

        # Manually set the cache for the clocks, simulation scripts might
        # want to access the time (which has been set in code and is therefore
        # not accessible by the normal means until the code has been built and
        # run)
        for clock in net._clocks:
            self.array_cache[clock.variables['timestep']] = np.array([clock._i_end])
            self.array_cache[clock.variables['t']] = np.array([clock._i_end * clock.dt_])

        if self.build_on_run:
            if self.has_been_run:
                raise RuntimeError('The network has already been built and run '
                                   'before. Use set_device with '
                                   'build_on_run=False and an explicit '
                                   'device.build call to use multiple run '
                                   'statements with this device.')
            self.build(direct_call=False, **self.build_options)

        self.first_run = False

    def fill_with_array(self, var, *args, **kwargs):
        # If the delay variable is set after the first run call, do not delete it on the
        # device (which is happening by default)
        if not self.first_run and var.name == 'delay':
            synaptic_delay_array_name = self.get_array_name(var, access_data=False)
            self.delete_synaptic_delay[synaptic_delay_array_name] = False
        super().fill_with_array(var, *args, **kwargs)

    def variableview_set_with_index_array(self, variableview, *args, **kwargs):
        # If the delay variable is set after the first run call, do not delete it on the
        # device (which is happening by default)
        if not self.first_run and variableview.name == 'delay':
            synaptic_delay_array_name = self.get_array_name(variableview.variable,
                                                            access_data=False)
            self.delete_synaptic_delay[synaptic_delay_array_name] = False
        super().variableview_set_with_index_array(variableview, *args, **kwargs)


    def network_store(self, net, *args, **kwds):
        raise NotImplementedError(('The store/restore mechanism is not '
                                   'supported in CUDA standalone'))

    def network_restore(self, net, *args, **kwds):
        raise NotImplementedError(('The store/restore mechanism is not '
                                   'supported in CUDA standalone'))



def prepare_codeobj_code_for_rng(codeobj):
    '''
    Prepare a CodeObject for random number generation (RNG).

    There are two different ways that random numbers are generated in CUDA:
      1) Using a buffer system which is refilled from host code in regular intervals
         using the cuRAND host API. This is used for `rand()`, `randn()` and
         `poisson(lambda)` when `lambda` is a scalar. The buffer system is implemented
         in the `rand.cu` template.
      2) Using on-the-fly RNG from device code using the cuRAND device API. This is used
         for `binomial` and `poisson(lambda)` when `lambda` is a vectorized variable
         (different across neurons/synapses). This needs initilization of cuRAND random
         states, which is also happening in the `rand.cu` template.

    This function counts the number of `rand()`, `randn()` and `poisson(<lambda>)`
    appearances in `codeobj.code.cu_file` and stores this number in the
    `codeobj.rng_calls` dictionary (with keys `"rand"`, `"randn"` and `"poisson_<idx>"`
    ,one <idx> per `poisson()` call). If the codeobject uses the curand device API for
    RNG (for binomial of poisson with variable lambda), this function sets
    `codeobj.needs_curand_states = True`.

    For RNG functions that use the buffer system, this function replaces the function
    arguments in the generated code such that a pointer to the random number buffer and
    the correct index are passed as function arguments.

    For RNG functions that use on-the-fly RNG, the functions are not replaced
    since no pointer or index has to be passed.

    For the `poisson` RNG, the RNG type depends on the `lambda` value. For scalar
    `lambda`, we use the buffer system which is most efficient and most robust in the
    RNG. For vectorized `lambda` values, the host API is inefficient and instead the
    simple device API is used, which is the most efficient but least robust. For the two
    RNG systems to work, we overload the CUDA implementation of `_poisson` with
    `_poisson(double _lambda, ...)` and `_poisson(unsigned int* _poisson_buffer, ...)`.
    When the buffer system is used, we replace the `_poisson(<lambda>, ...)` calls with
    `_poisson(<int_pointer>, ...)` calls.

    For `poisson` with `lambda <= 0`, the returned random numbers are always `0`. This
    function makes sure that the `lambda` is replaced with a double literal for our
    overloaded `_poisson` function to work correctly.

    Parameters
    ----------
    codeobj: CodeObjects
        Codeobject with generated CUDA code in `codeobj.code.cu_file`.
    '''
    # synapses_create_generator uses host side random number generation
    if codeobj.template_name == 'synapses_create_generator':
        return

    ### RAND/N REGEX
    # regex explained
    # (?<!...) negative lookbehind: don't match if ... preceeds
    #     XXX: This only excludes #define lines which have exactly one space to '_rand'.
    #          As long is we don't have '#define  _rand' with 2 spaces, thats fine.
    # \b - non-alphanumeric character (word boundary, does not consume a character)
    pattern_template = r'(?<!#define )\b_{rng_func}\(_vectorisation_idx\)'
    rand_randn_pattern = {
        'rand': pattern_template.format(rng_func='rand'),
        'randn': pattern_template.format(rng_func='randn'),
    }

    # Store number of matches in codeobj.rng_calls dictionary
    for rng_type in ['rand', 'randn']:
        matches = re.findall(rand_randn_pattern[rng_type], codeobj.code.cu_file)
        num_calls = len(matches)
        codeobj.rng_calls[rng_type] = num_calls
        logger.diagnostic(
            f"Matched {num_calls} {rng_type} calls for {codeobj.name}"
        )

    ### POISSON REGEX
    # (?P<lambda>*?) Named group: Returns whatever is matched inside the brackets
    #                instead of the entire string and stores it in the variable
    #                `lamda`.
    #                `.*?` does a non-greedy match of all (*). This is the lambda value.
    poisson_pattern = r'(?<!#define )\b_poisson\((?P<lamda>.*?), _vectorisation_idx\)'
    matches_poisson = re.findall(poisson_pattern, codeobj.code.cu_file)

    # Collect the number of poisson calls separated by lambda values (we need to
    # generate poisson values separately for each lambda value when using cuRAND host
    # API). We call the poisson functions with different lambda `poisson-<idx>`, where
    # `<idx>` is enuemerates all poisson functions. It looks like this:
    #
    #   codeobj.rng_calls.keys() = ["poisson_0", "poisson_1", ...]
    #   codeobj.rng_calls["poisson_0"] = <number_of_calls_per_time_step>
    #
    # The lamda values for all poisson functions are stored in
    #   device.all_poisson_lamdas[codeobj.name]["poisson_0"] = <lambda_value>
    lamda_matches = {}
    poisson_device_api = False
    poisson_with_lamda_zero = []
    for i, lamda_match in enumerate(sorted(set(matches_poisson))):
        poisson_name = f"poisson_{i}"
        # Test if the lambda_match from poisson(<lambda_match>) is scalar or vectorized
        # (different across neurons/synapses of the codeobject owner)
        try:
            # Try to convert it to float, will raise ValueError if not possible
            # This will work only if `lambda_match` is a literal, e.g. in `poisson(5)`
            lamda = float(lamda_match)
            lamda_is_scalar = True
            logger.debug(
                f"Matched literal scalar lambda {lamda} for {poisson_name} in {codeobj.name}"
            )
        except ValueError:
            # lamda is not a float but a variable, e.g. `poisson(var)`
            # Check if lamda is scalar and constant (i.e. doesn't change during a run)
            # TODO: check if scalar variable can be set during in run. If so, check for
            # constant here as well!
            # TODO: make sure that a scalar and constand variable can't be changed
            # during a run!
            if lamda_match in codeobj.variables and codeobj.variables[lamda_match].scalar:
                lamda = codeobj.variables[lamda_match].value
                const = codeobj.variables[lamda_match].constant
                lamda_is_scalar = True
                logger.debug(
                    f"Matched non-literal scalar lambda {lamda} for {poisson_name} in {codeobj.name}"
                )
            else:
                # lamda is an array variable
                lamda = lamda_match
                lamda_is_scalar = False
                logger.debug(
                    f"Matched vectorized lambda {lamda} in {codeobj.name}"
                )

        if lamda_is_scalar:
            # We don't want to generate random numbers on the host for lambda <= 0,
            # which should return 0 anyways. We use the _poisson(double, ...) function,
            # which checks for lambda <= 0 before calling the curand devica API. Hence
            # we don't need host side RNG or curand states.
            lamda_matches[poisson_name] = lamda_match
            if lamda <= 0:
                # We need to replace '0' with '0.0' (double literal) for lambda in this
                # case, see comment below
                poisson_with_lamda_zero.append(poisson_name)
                continue
            assert lamda not in codeobj.poisson_lamdas.values()
            assert poisson_name not in codeobj.poisson_lamdas.keys()
            codeobj.poisson_lamdas[poisson_name] = lamda
            codeobj.rng_calls[poisson_name] = matches_poisson.count(lamda_match)
        else:
            codeobj.needs_curand_states = True
            poisson_device_api = True

    # We have two if/else code paths in synapses code (homog. / heterog. delay mode),
    # therefore we have twice as much matches for rand/randn/poisson-<idx> as actual
    # calls. Hence we half the number of detected calls here.
    if codeobj.template_name == 'synapses':
        for rng_type in codeobj.rng_calls.keys():
            assert codeobj.rng_calls[rng_type] % 2 == 0
            codeobj.rng_calls[rng_type] //= 2

    # RAND/N
    # Substitue the _vectorisation_idx of _rand/n calls such that different calls always
    # get different random numbers from the random number buffers
    # Substitute rand/n arguments twice for synapses templates
    repeat = 2 if codeobj.template_name == 'synapses' else 1
    for rng_type in ["rand", "randn"]:
        if codeobj.rng_calls[rng_type] > 0:
            for _ in range(repeat):
                for i in range(codeobj.rng_calls[rng_type]):
                    codeobj.code.cu_file = re.sub(
                        rand_randn_pattern[rng_type],
                        f"_{rng_type}(_vectorisation_idx + {i} * _N)",
                        codeobj.code.cu_file,
                        count=1
                    )

    # POISSON
    sub_repl_template = (
        "_poisson(_ptr_array_{codeobj.name}_{poisson_name}, _vectorisation_idx + {i} * _N)"
    )
    for poisson_name, lamda_match in lamda_matches.items():
        sub_pattern = poisson_pattern.replace(
            # use the correct lamda instead of matching the lamda
            "(?P<lamda>.*?)", lamda_match
        )
        if codeobj.rng_calls[poisson_name] > 0:
            for _ in range(repeat):
                for i in range(codeobj.rng_calls[poisson_name]):
                    sub_repl = sub_repl_template.format(
                        codeobj=codeobj, poisson_name=poisson_name, i=i
                    )
                    codeobj.code.cu_file = re.sub(
                        sub_pattern,
                        sub_repl,
                        codeobj.code.cu_file,
                        count=1
                    )
        elif poisson_name in poisson_with_lamda_zero:
            # Make sure the _poisson argument is a double literal. "0" fails since
            # it can be interpreted as both, null pointer or double and the _poisson
            # implementation is overloaded for `unsigned int *` and `double`.
            sub_repl= f"_poisson({float(lamda_match):.1f}, _vectorisation_idx)"
            codeobj.code.cu_file = re.sub(
                sub_pattern,
                sub_repl,
                codeobj.code.cu_file,
                count=0  # replace all ocurrences at once
            )

    # If the codeobjec does not need curand states for poisson, check if it needs
    # them for  binomial calls
    if not codeobj.needs_curand_states:
        match = re.search('_binomial\w*\(const int vectorisation_idx\)', codeobj.code.cu_file)
        if match is not None:
            codeobj.needs_curand_states = True


cuda_standalone_device = CUDAStandaloneDevice()

all_devices['cuda_standalone'] = cuda_standalone_device
