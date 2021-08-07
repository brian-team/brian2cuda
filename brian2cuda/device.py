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
import sys

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
from brian2.utils.stringtools import get_identifiers
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.utils.logger import get_logger
from brian2.units import second

from brian2.devices.cpp_standalone.device import CPPWriter, CPPStandaloneDevice
from brian2.input.spikegeneratorgroup import SpikeGeneratorGroup

from brian2cuda.utils.stringtools import replace_floating_point_literals
from brian2cuda.utils.gputools import select_gpu, get_nvcc_path

from .codeobject import CUDAStandaloneCodeObject, CUDAStandaloneAtomicsCodeObject


__all__ = []

logger = get_logger('brian2.devices.cuda_standalone')


class CUDAWriter(CPPWriter):
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.source_files = []
        self.header_files = []

    def write(self, filename, contents):
        logger.diagnostic('Writing file %s:\n%s' % (filename, contents))
        if filename.lower().endswith('.cu'):
            self.source_files.append(filename)
        if filename.lower().endswith('.cpp'):
            self.source_files.append(filename)
        elif filename.lower().endswith('.h'):
            self.header_files.append(filename)
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

        ### Attributes specific to CUDAStandaloneDevice:
        # specify minimal compute capability suppported by brian2cuda
        self.minimal_compute_capability = 3.5
        # only true during first run call (relevant for synaptic pre/post ID deletion)
        self.first_run = True
        # list of pre/post ID arrays that are not needed in device memory
        self.delete_synaptic_pre = {}
        self.delete_synaptic_post = {}
        # for each run, store a dictionary of codeobjects using rand, randn, binomial
        self.code_objects_per_run = []
        # and a dictionary with the same across all run calls
        self.all_code_objects = {'rand': [], 'randn': [], 'rand_or_randn': [], 'binomial': []}
        # and collect codeobjects run only once with binomial in separate list
        self.code_object_with_binomial_separate_call = []
        # store gpu_id and compute capability
        self.gpu_id = None
        self.compute_capability = None

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
            msg = "`prefix` has to be one of {choices} or `None`, got {prefix}"
            raise ValueError(msg.format(choices=choices, prefix=prefix))

        if not access_data and prefix in ['_ptr', 'd']:
            msg = "Don't use `'{prefix}'` prefix for a dynamic array object."
            raise ValueError(msg.format(prefix=prefix))

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
        for ac_name, ac_code in abstract_code.iteritems():
            statements = make_statements(ac_code,
                                         variables,
                                         prefs['core.default_float_dtype'],
                                         optimise=True,
                                         blockname=ac_name)
            _, vector_statements[ac_name] = statements
        read = set()
        write = set()
        for statements in vector_statements.itervalues():
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
        if isinstance(owner, Synapses):
            if template_name in ['synapses', 'stateupdate', 'summed_variable']:
                synapses_object_every_tick = True
            if not self.first_run and template_name in ['group_variable_set_conditional', 'group_variable_set']:
                synapses_object_single_tick_after_run = True
        if synapses_object_every_tick or synapses_object_single_tick_after_run:
            read, write = self.get_array_read_write(abstract_code, variables)
            read_write = read.union(write)
            synaptic_pre_array_name = self.get_array_name(owner.variables['_synaptic_pre'], False)
            synaptic_post_array_name = self.get_array_name(owner.variables['_synaptic_post'], False)
            if synaptic_pre_array_name not in self.delete_synaptic_pre.iterkeys():
                self.delete_synaptic_pre[synaptic_pre_array_name] = True
            if synaptic_post_array_name not in self.delete_synaptic_post.iterkeys():
                self.delete_synaptic_post[synaptic_post_array_name] = True
            error_msg = ("'devices.cuda_standalone.no_{prepost}_references' "
                         "was set to True, but {prepost}synaptic index is "
                         "needed for variable {varname} in {owner.name}")
            # Check for all variable that are read or written to if they are
            # i/j or their indices are pre/post
            for varname in variables.iterkeys():
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
        if template_name == "synapses":
            prepost = template_kwds['pathway'].prepost
            synaptic_effects = "synapse"
            for varname in variables.iterkeys():
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
            print('debug syn effect mode ', synaptic_effects)
            logger.debug("Synaptic effects of Synapses object {syn} modify {mod} group variables.".format(syn=name, mod=synaptic_effects))
            # use atomics if possible (except for `synapses` mode, where we cann parallelise without)
            # TODO: this overwrites if somebody sets a codeobject in the Synapses(..., codeobj_class=...)
            if prefs['devices.cuda_standalone.use_atomics'] and synaptic_effects != 'synapses':
                codeobj_class = CUDAStandaloneAtomicsCodeObject
                logger.debug("Using atomics in synaptic effect application of "
                             "Synapses object {syn}".format(syn=name))
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
        for var, varname in self.arrays.iteritems():
            if var.name.endswith('space'):  # get all eventspace variables
                self.eventspace_arrays[var] = varname
                #if hasattr(var, 'owner') and isinstance(v.owner, Clock):
                if isinstance(var.owner, SpikeGeneratorGroup):
                    self.spikegenerator_eventspaces.append(varname)
        for var in self.eventspace_arrays.iterkeys():
            del self.arrays[var]
        multisyn_vars = []
        for syn in synapses:
            if syn.multisynaptic_index is not None:
                multisyn_vars.append(syn.variables[syn.multisynaptic_index])
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
                        all_codeobj_with_rand=self.all_code_objects['rand'],
                        all_codeobj_with_randn=self.all_code_objects['randn'],
                        sm_multiplier=sm_multiplier,
                        num_parallel_blocks=num_parallel_blocks,
                        curand_generator_type=curand_generator_type,
                        curand_generator_ordering=curand_generator_ordering,
                        curand_float_type=c_data_type(prefs['core.default_float_dtype']),
                        eventspace_arrays=self.eventspace_arrays,
                        spikegenerator_eventspaces=self.spikegenerator_eventspaces,
                        multisynaptic_idx_vars=multisyn_vars,
                        profiled_codeobjects=self.profiled_codeobjects)
        # Reinsert deleted entries, in case we use self.arrays later? maybe unnecassary...
        self.arrays.update(self.eventspace_arrays)
        writer.write('objects.*', arr_tmp)

    def generate_main_source(self, writer):
        main_lines = []
        procedures = [('', main_lines)]
        runfuncs = {}
        run_counter = 0
        for func, args in self.main_queue:
            if func=='run_code_object':
                codeobj, = args
                codeobj.runs_every_tick = False
                # need to check for rand/randn/binomial for objects only run
                # once, stored in `code_object.rand(n)_calls`.
                uses_binomial = check_codeobj_for_rng(codeobj, check_binomial=True)
                if uses_binomial:
                    self.code_object_with_binomial_separate_call.append(codeobj)
                    if isinstance(codeobj.owner, Synapses) \
                            and codeobj.template_name in ['group_variable_set_conditional', 'group_variable_set']:
                        # At curand state initalization, synapses are not generated yet.
                        # For codeobjects run every tick, this happens in the init() of
                        # tha random number buffer called at first clock cycle of the network
                        main_lines.append('random_number_buffer.ensure_enough_curand_states();')
                main_lines.append('_run_%s();' % codeobj.name)
            elif func=='run_network':
                net, netcode = args
                if run_counter == 0:
                    # These lines delete `i`/`j` variables stored per synapse. They need to be called after
                    # synapses_initialise_queue codeobjects, therefore just before the network creation, so I
                    # put them here
                    for arrname, boolean in self.delete_synaptic_pre.iteritems():
                        if boolean:
                            lines = '''
                            dev{synaptic_pre}.clear();
                            dev{synaptic_pre}.shrink_to_fit();
                            '''.format(synaptic_pre=arrname)
                            main_lines.extend(lines.split('\n'))
                    for arrname, boolean in self.delete_synaptic_post.iteritems():
                        if boolean:
                            lines = '''
                            dev{synaptic_post}.clear();
                            dev{synaptic_post}.shrink_to_fit();
                            '''.format(synaptic_post=arrname)
                            main_lines.extend(lines.split('\n'))
                main_lines.extend(netcode)
                run_counter += 1
            elif func=='set_by_constant':
                arrayname, value, is_dynamic = args
                size_str = arrayname + ".size()" if is_dynamic else "_num_" + arrayname
                code = '''
                for(int i=0; i<{size_str}; i++)
                {{
                    {arrayname}[i] = {value};
                }}
                '''.format(arrayname=arrayname, size_str=size_str,
                           value=CPPNodeRenderer().render_expr(repr(value)))
                main_lines.extend(code.split('\n'))
                pointer_arrayname = "dev{arrayname}".format(arrayname=arrayname)
                if arrayname.endswith('space'):  # eventspace
                    pointer_arrayname += '[current_idx{arrayname}]'.format(arrayname=arrayname)
                if is_dynamic:
                    pointer_arrayname = "thrust::raw_pointer_cast(&dev{arrayname}[0])".format(arrayname=arrayname)
                line = '''
                CUDA_SAFE_CALL(
                        cudaMemcpy({pointer_arrayname},
                                   &{arrayname}[0],
                                   sizeof({arrayname}[0])*{size_str},
                                   cudaMemcpyHostToDevice)
                        );
                '''.format(arrayname=arrayname, size_str=size_str, pointer_arrayname=pointer_arrayname,
                           value=CPPNodeRenderer().render_expr(repr(value)))
                main_lines.extend([line])
            elif func=='set_by_single_value':
                arrayname, item, value = args
                pointer_arrayname = "dev{arrayname}".format(arrayname=arrayname)
                if arrayname.endswith('space'):  # eventspace
                    pointer_arrayname += '[current_idx{arrayname}]'.format(arrayname=arrayname)
                if arrayname in self.dynamic_arrays.values():
                    pointer_arrayname = "thrust::raw_pointer_cast(&dev{arrayname}[0])".format(arrayname=arrayname)
                code = '''
                {arrayname}[{item}] = {value};
                CUDA_SAFE_CALL(
                        cudaMemcpy({pointer_arrayname} + {item},
                                   &{arrayname}[{item}],
                                   sizeof({arrayname}[{item}]),
                                   cudaMemcpyHostToDevice)
                        );
                '''.format(pointer_arrayname=pointer_arrayname, arrayname=arrayname, item=item, value=value)
                main_lines.extend([code])
            elif func=='set_by_array':
                arrayname, staticarrayname, is_dynamic = args
                size = "_num_" + arrayname
                if is_dynamic:
                    size = arrayname + ".size()"
                code = '''
                for(int i=0; i<_num_{staticarrayname}; i++)
                {{
                    {arrayname}[i] = {staticarrayname}[i];
                }}
                '''.format(arrayname=arrayname, staticarrayname=staticarrayname, size_str=size)
                pointer_arrayname = "dev{arrayname}".format(arrayname=arrayname)
                if arrayname in self.dynamic_arrays.values():
                    pointer_arrayname = "thrust::raw_pointer_cast(&dev{arrayname}[0])".format(arrayname=arrayname)
                main_lines.extend(code.split('\n'))
                line = '''
                CUDA_SAFE_CALL(
                        cudaMemcpy({pointer_arrayname}, &{arrayname}[0],
                                sizeof({arrayname}[0])*{size_str}, cudaMemcpyHostToDevice)
                        );
                '''.format(pointer_arrayname=pointer_arrayname, staticarrayname=staticarrayname, size_str=size, arrayname=arrayname)
                main_lines.extend([line])
            elif func=='set_array_by_array':
                arrayname, staticarrayname_index, staticarrayname_value = args
                code = '''
                for(int i=0; i<_num_{staticarrayname_index}; i++)
                {{
                    {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                }}
                CUDA_SAFE_CALL(
                        cudaMemcpy(dev{arrayname}, &{arrayname}[0],
                                sizeof({arrayname}[0])*_num_{arrayname}, cudaMemcpyHostToDevice)
                        );
                '''.format(arrayname=arrayname, staticarrayname_index=staticarrayname_index,
                           staticarrayname_value=staticarrayname_value)
                main_lines.extend(code.split('\n'))
            elif func=='resize_array':
                array_name, new_size = args
                main_lines.append('''
                    {array_name}.resize({new_size});
                    THRUST_CHECK_ERROR(dev{array_name}.resize({new_size}));
                '''.format(array_name=array_name, new_size=new_size))
            elif func=='insert_code':
                main_lines.append(args)
            elif func=='start_run_func':
                name, include_in_parent = args
                if include_in_parent:
                    main_lines.append('%s();' % name)
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
                main_lines.append('random_number_buffer.set_seed({seed!r}ULL);'.format(seed=seed))
            else:
                raise NotImplementedError("Unknown main queue function type "+func)

        # Store the GPU ID and it's compute capability. The latter can be overwritten in
        # self.generate_makefile() via preferces
        self.gpu_id, self.compute_capability = select_gpu()

        # generate the finalisations
        for codeobj in self.code_objects.itervalues():
            if hasattr(codeobj.code, 'main_finalise'):
                main_lines.append(codeobj.code.main_finalise)

        user_headers = self.headers + prefs['codegen.cpp.headers']
        main_tmp = self.code_object_class().templater.main(None, None,
                                                           gpu_id=self.gpu_id,
                                                           main_lines=main_lines,
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
        # Generate data for non-constant values
        for codeobj in self.code_objects.itervalues():
            code_object_defs_lines = []
            host_parameters_lines = []
            kernel_parameters_lines = []
            kernel_constants_lines = []
            additional_code = []
            number_elements = ""
            if hasattr(codeobj, 'owner') and hasattr(codeobj.owner, '_N') and codeobj.owner._N != 0:
                number_elements = str(codeobj.owner._N)
            else:
                number_elements = "_N"
            for k, v in codeobj.variables.iteritems():
                if k == 'dt' and prefs['core.default_float_dtype'] == np.float32:
                    # use the double-precision array versions for dt as kernel arguments
                    # they are cast to single-precision scalar dt in scalar_code
                    v = v.real_var

                # code objects which only run once
                if k in ["rand", "randn"] and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create_generator":
                    if k == "randn":
                        code_snippet='''
                            //genenerate an array of random numbers on the device
                            {dtype}* dev_array_randn;
                            CUDA_SAFE_CALL(
                                    cudaMalloc((void**)&dev_array_randn, sizeof({dtype})*{number_elements}*{codeobj.randn_calls})
                                    );
                            curandGenerateNormal{curand_suffix}(curand_generator, dev_array_randn, {number_elements}*{codeobj.randn_calls}, 0, 1);
                            '''.format(number_elements=number_elements, codeobj=codeobj, dtype=c_data_type(prefs['core.default_float_dtype']),
                                       curand_suffix='Double' if prefs['core.default_float_dtype']==np.float64 else '')
                        additional_code.append(code_snippet)
                        line = "{dtype}* _ptr_array_{name}_randn".format(dtype=c_data_type(prefs['core.default_float_dtype']), name=codeobj.name)
                        kernel_parameters_lines.append(line)
                        host_parameters_lines.append("dev_array_randn")
                    elif k == "rand":
                        code_snippet = '''
                            //genenerate an array of random numbers on the device
                            {dtype}* dev_array_rand;
                            CUDA_SAFE_CALL(
                                    cudaMalloc((void**)&dev_array_rand, sizeof({dtype})*{number_elements}*{codeobj.rand_calls})
                                    );
                            curandGenerateUniform{curand_suffix}(curand_generator, dev_array_rand, {number_elements}*{codeobj.rand_calls});
                            '''.format(number_elements=number_elements, codeobj=codeobj, dtype=c_data_type(prefs['core.default_float_dtype']),
                                       curand_suffix='Double' if prefs['core.default_float_dtype']==np.float64 else '')
                        additional_code.append(code_snippet)
                        line = "{dtype}* _ptr_array_{name}_rand".format(dtype=c_data_type(prefs['core.default_float_dtype']), name=codeobj.name)
                        kernel_parameters_lines.append(line)
                        host_parameters_lines.append("dev_array_rand")
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
                        "{arrayname}[0]".format(arrayname=arrayname))
                    kernel_parameters_lines.append(
                        "const {dtype} _value{arrayname}".format(
                            dtype=dtype, arrayname=arrayname))
                # ArrayVariables (dynamic and not)
                elif isinstance(v, ArrayVariable):
                    prefix = 'dev'
                    # These codeobjects run on the host
                    host_codeobjects = ['synapses_create_generator',
                                        'synapses_create_array',
                                        'synapses_initialise_queue']
                    if codeobj.template_name in host_codeobjects:
                        prefix = ''
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

                                line = '{dtype}* const {array_name} = thrust::raw_pointer_cast(&{dyn_array_name}[0]);'
                                line = line.format(dtype=dtype,
                                                   array_name=array_name,
                                                   dyn_array_name=dyn_array_name)
                                code_object_defs_lines.append(line)

                                line = 'const int _num{k} = {dyn_array_name}.size();'
                                line = line.format(k=k, dyn_array_name=dyn_array_name)
                                code_object_defs_lines.append(line)

                                # These lines are used to define the kernel call parameters, that
                                # means only for codeobjects running on the device. The array names
                                # always have a `_dev` prefix.
                                line = '{array_name}'.format(array_name=array_name)
                                host_parameters_lines.append(line)
                                host_parameters_lines.append("_num{k}".format(k=k))

                                # These lines declare kernel parameters as the `_ptr` variables that
                                # are used in `scalar_code` and `vector_code`.
                                # TODO: here we should add const / __restrict and other optimizations
                                #       for variables that are e.g. only read in the kernel
                                line = "{dtype}* {ptr_array_name}"
                                kernel_parameters_lines.append(line.format(dtype=dtype,
                                                                           ptr_array_name=ptr_array_name))

                                line = "const int _num{k}"
                                kernel_parameters_lines.append(line.format(k=k))

                        else:  # v is ArrayVariable but not DynamicArrayVariable
                            host_parameters_lines.append("{array_name}".format(array_name=array_name))
                            line = '{dtype}* {ptr_array_name}'.format(dtype=dtype,
                                                                      ptr_array_name=ptr_array_name)
                            kernel_parameters_lines.append(line)

                            code_object_defs_lines.append('const int _num{k} = {v.size};'.format(k=k, v=v))
                            kernel_constants_lines.append('const int _num{k} = {v.size};'.format(k=k, v=v))
                            if k.endswith('space'):
                                bare_array_name = self.get_array_name(v)
                                idx = '[current_idx{bare_array_name}]'.format(bare_array_name=bare_array_name)
                                host_parameters_lines[-1] += idx
                    except TypeError:
                        pass

            # This rand stuff got a little messy... we pass a device pointer as kernel variable and have a hash define for rand() -> _ptr_..._rand[]
            # The device pointer is advanced every clock cycle in rand.cu and reset when the random number buffer is refilled (also in rand.cu)
            # TODO can we just include this in the k == 'rand' test above?
            if codeobj.rand_calls >= 1 and codeobj.runs_every_tick:
                host_parameters_lines.append("dev_{name}_rand".format(name=codeobj.name))
                kernel_parameters_lines.append("{dtype}* _ptr_array_{name}_rand".format(dtype=c_data_type(prefs['core.default_float_dtype']),
                                                                                name=codeobj.name))
            if codeobj.randn_calls >= 1 and codeobj.runs_every_tick:
                host_parameters_lines.append("dev_{name}_randn".format(name=codeobj.name))
                kernel_parameters_lines.append("{dtype}* _ptr_array_{name}_randn".format(dtype=c_data_type(prefs['core.default_float_dtype']),
                                                                                name=codeobj.name))

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
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.variables
            # TODO: fix these freeze/HOST_CONSTANTS hacks somehow - they work but not elegant.
            code = self.freeze(codeobj.code.cu_file, ns)

            if len(host_parameters[codeobj.name]) == 0:
                host_parameters[codeobj.name].append("0")
                kernel_parameters[codeobj.name].append("int dummy")

            # HOST_CONSTANTS are equivalent to C++ Standalone's CONSTANTS
            code = code.replace('%HOST_CONSTANTS%', '\n\t\t'.join(code_object_defs[codeobj.name]))
            # KERNEL_CONSTANTS are the same for inside device kernels
            code = code.replace('%KERNEL_CONSTANTS%', '\n\t'.join(kernel_constants[codeobj.name]))
            # HOST_PARAMETERS are parameters that device kernels are called with from host code
            code = code.replace('%HOST_PARAMETERS%', ',\n\t\t\t'.join(host_parameters[codeobj.name]))
            # KERNEL_PARAMETERS are the same names of the same parameters inside the device kernels
            code = code.replace('%KERNEL_PARAMETERS%', ',\n\t'.join(kernel_parameters[codeobj.name]))
            code = code.replace('%CODEOBJ_NAME%', codeobj.name)
            code = '#include "objects.h"\n'+code

            # substitue in generated code double types with float types in
            # single-precision mode
            if prefs['core.default_float_dtype'] == np.float32:
                # cast time differences (double) to float in event-drive updates
                sub = 't - lastupdate'
                if sub in code:
                    code = code.replace(sub, 'float({})'.format(sub))
                    logger.debug("Replaced {sub} with float({sub}) in {codeobj}"
                                 "".format(sub=sub, codeobj=codeobj))
                # replace double-precision floating-point literals with their
                # single-precision version (e.g. `1.0` -> `1.0f`)
                code = replace_floating_point_literals(code)
                logger.debug("Replaced floating point literals by single "
                             "precision version (appending `f`) in {}."
                             "".format(codeobj))

            writer.write('code_objects/'+codeobj.name+'.cu', code)
            writer.write('code_objects/'+codeobj.name+'.h', codeobj.code.h_file)

    def generate_rand_source(self, writer):
        binomial_codeobjects = {}
        for co in self.all_code_objects['binomial']:
            name = co.owner.name
            if name not in binomial_codeobjects:
                if isinstance(co.owner, Synapses):
                    # this is the pointer to the synapse object's N, which is a
                    # null pointer before synapses are generated and an int ptr
                    # after synapse generation (used to test if synapses
                    # already generated or not)
                    test_ptr = '_array_{name}_N'.format(name=name)
                    N = test_ptr + '[0]'
                else:
                    test_ptr = None
                    N = co.owner._N
                binomial_codeobjects[name] = {'test_ptr': test_ptr, 'N': N}
        rand_tmp = self.code_object_class().templater.rand(None, None,
                                                           code_objects_per_run=self.code_objects_per_run,
                                                           binomial_codeobjects=binomial_codeobjects,
                                                           number_run_calls=len(self.code_objects_per_run),
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
                writer.source_files.append('brianlib/'+file)
            if file.lower().endswith('.cu'):
                writer.source_files.append('brianlib/'+file)
            elif file.lower().endswith('.h'):
                writer.header_files.append('brianlib/'+file)

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
                "value `None`".format(self.minimal_compute_capability)
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
                    "Compute capability `{}` is not supported. Minimal supported "
                    "compute capability is `{}`.".format(
                        self.compute_capability, self.minimal_compute_capability
                    )
                )

        # If GPU architecture is detected automatically or set via `compute_capability`
        # prefs, we still need to add it as a compile argument
        if not gpu_arch_flags:
            # Turn float (3.5) into string ("35")
            compute_capability_str = ''.join(str(self.compute_capability).split('.'))
            gpu_arch_flags.append("-arch=sm_{}".format(compute_capability_str))

        # Log compiled GPU architecture
        if self.compute_capability is None:
            logger.info(
                "Compiling device code with manually set architecture flags "
                "({}). Be aware that the minimal supported compute capability is {} "
                "(we are not checking your compile flags)".format(
                    gpu_arch_flags, self.minimal_compute_capability
                )
            )
        else:
            logger.info(
                "Compiling device code for compute capability {} (compiler flags: {})"
                "".format(self.compute_capability, gpu_arch_flags)
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
                source_files=' '.join(writer.source_files),
                header_files=' '.join(writer.header_files),
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
        # TODO: remove this when #83 is fixed
        if not self.build_on_run:
            run_count = 0
            for func, args in self.main_queue:
                if func == 'run_network':
                    run_count += 1
            if run_count > 1:
                logger.warn("Multiple run statements are currently error prone. "
                            "See #83, #85, #86.")

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
                    msg += "Unknown keyword argument '%s'. " % kwd
            raise TypeError(msg)

        if debug and disable_asserts:
            logger.warn("You have disabled asserts in debug mode. Are you sure this is what you wanted to do?")

        if additional_source_files is None:
            additional_source_files = []
        if run_args is None:
            run_args = []
        if directory is None:
            directory = tempfile.mkdtemp(prefix='brian_standalone_')

        cpp_compiler, cpp_extra_compile_args = get_compiler_and_args()
        cpp_compiler_flags = ' '.join(cpp_extra_compile_args)
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
        self.find_synapses()

        # Not sure what the best place is to call Network.after_run -- at the
        # moment the only important thing it does is to clear the objects stored
        # in magic_network. If this is not done, this might lead to problems
        # for repeated runs of standalone (e.g. in the test suite).
        for net in self.networks:
            net.after_run()

        # Check that all names are globally unique
        names = [obj.name for net in self.networks for obj in net.objects]
        non_unique_names = [name for name, count in Counter(names).items()
                            if count > 1]
        if len(non_unique_names):
            formatted_names = ', '.join("'%s'" % name
                                        for name in non_unique_names)
            raise ValueError('All objects need to have unique names in '
                             'standalone mode, the following name(s) were used '
                             'more than once: %s' % formatted_names)

        self.generate_main_source(self.writer)

        # Create lists of codobjects using rand, randn or binomial across all
        # runs (needed for variable declarations).
        #   - Variables needed for device side rand/randn are declared in objects.cu:
        #     all_code_objects['rand'/'rand'] are neede in `generate_objects_source`
        #   - Variables needed for device side binomial functions are initialized in rand.cu:
        #     all_code_objects['binomial'] is needed in `generate_rand_source`
        for run_codeobj in self.code_objects_per_run:
            self.all_code_objects['rand'].extend(run_codeobj['rand'])
            self.all_code_objects['randn'].extend(run_codeobj['randn'])
            self.all_code_objects['rand_or_randn'].extend(run_codeobj['rand_or_randn'])
            self.all_code_objects['binomial'].extend(run_codeobj['binomial'])
        # Device side binomial functions use curand device api. The curand states (one per thread
        # executed in parallel) are initialized in rand.cu. The `run_codeobj` dictionary above only
        # collects codeobjects run every tick in the network. Here, we add those codeobjects that
        # use device side binomial functions and are run only once (e.g. when setting group
        # variables before run). For rand/randn, the codeobject themselves take care of the
        # initialization. For binomial, we need to initialize them in rand.cu.
        # This line needs to be after `self.generate_main_source`, which populates
        # `self.code_object_with_binomial_separate_call` and before `self.generate_rand_source`
        for codeobj in self.code_object_with_binomial_separate_call:
            if codeobj not in self.all_code_objects['binomial']:
                self.all_code_objects['binomial'].append(codeobj)

        self.generate_codeobj_source(self.writer)
        self.generate_objects_source(self.writer, self.arange_arrays,
                                     self.net_synapses,
                                     self.static_array_specs,
                                     self.networks)
        self.generate_network_source(self.writer)
        self.generate_synapses_classes_source(self.writer)
        self.generate_run_source(self.writer)
        self.generate_rand_source(self.writer)
        self.copy_source_files(self.writer, directory)

        self.writer.source_files.extend(additional_source_files)

        self.generate_makefile(self.writer, cpp_compiler,
                               cpp_compiler_flags,
                               cpp_linker_flags,
                               debug,
                               disable_asserts)

        logger.info("Using the following preferences for CUDA standalone:")
        for pref_name in prefs:
            if "devices.cuda_standalone" in pref_name:
                logger.info("\t{} = {}".format(pref_name,prefs[pref_name]))
 
        logger.debug("Using the following brian preferences:")
        for pref_name in prefs:
            if pref_name not in prefs:
                logger.debug("\t{} = {}".format(pref_name,prefs[pref_name]))

        if compile:
            self.compile_source(directory, cpp_compiler, debug, clean)
            if run:
                self.run(directory, with_output, run_args)

    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=False, level=0, **kwds):
        ###################################################
        ### This part is copied from CPPStandaoneDevice ###
        ###################################################
        if kwds:
            logger.warn(('Unsupported keyword argument(s) provided for run: '
                         '%s') % ', '.join(kwds.keys()))
        # We store this as an instance variable for later access by the
        # `code_object` method
        self.enable_profiling = profile

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
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                %STREAMNAME% << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                %STREAMNAME% << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << elapsed << " s";
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    %STREAMNAME% << ", estimated " << remaining << " s remaining.";
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
        elif isinstance(report, basestring):
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

        # For each codeobject of this run check if it uses rand, randn or
        # binomials. Store these as attributes of the codeobject and create
        # lists of codeobjects that use rand, randn or binomials. This only
        # checks codeobject in the network, meaning only the ones running every
        # clock tick.
        code_object_rng = {'rand': [], 'randn': [], 'rand_or_randn': [], 'binomial': []}
        for _, co in code_objects:  # (clock, code_object)
            binomial_match = check_codeobj_for_rng(co, check_binomial=True)
            if co.rand_calls > 0:
                code_object_rng['rand'].append(co)
                code_object_rng['rand_or_randn'].append(co)
            if co.randn_calls > 0:
                code_object_rng['randn'].append(co)
                if co.rand_calls == 0:
                    # only add if it wasn't already added above
                    code_object_rng['rand_or_randn'].append(co)
            if binomial_match:
                code_object_rng['binomial'].append(co)


        # store the codeobject dictionary for each run
        self.code_objects_per_run.append(code_object_rng)

        # To profile SpeedTests, we need to be able to set `profile` in
        # `set_device`. Here we catch that case.
        if 'profile' in self.build_options:
            build_profile = self.build_options.pop('profile')
            if build_profile:
                self.enable_profiling = True

        # Generate the updaters
        run_lines = []
        run_lines.append('{net.name}.clear();'.format(net=net))

        # create all random numbers needed for the next clock cycle
        for clock in net._clocks:
            run_lines.append('{net.name}.add(&{clock.name}, _run_random_number_buffer);'.format(clock=clock, net=net))

        all_clocks = set()
        for clock, codeobj in code_objects:
            run_lines.append('{net.name}.add(&{clock.name}, _run_{codeobj.name});'.format(clock=clock,
                                                                                          net=net, codeobj=codeobj))
            all_clocks.add(clock)

        # Under some rare circumstances (e.g. a NeuronGroup only defining a
        # subexpression that is used by other groups (via linking, or recorded
        # by a StateMonitor) *and* not calculating anything itself *and* using a
        # different clock than all other objects) a clock that is not used by
        # any code object should nevertheless advance during the run. We include
        # such clocks without a code function in the network.
        for clock in net._clocks:
            if clock not in all_clocks:
                run_lines.append('{net.name}.add(&{clock.name}, NULL);'.format(clock=clock, net=net))

        # In our benchmark scripts we run one example `nvprof` run,
        # which is informative especially when not running in profile
        # mode. In order to have the `nvprof` call only profile the
        # kernels which are run every timestep, we add
        # `cudaProfilerStart()`, `cudaDeviceSynchronize()` and
        # `cudaProfilerStop()`. But this might be confusing for anybody
        # who runs `nvprof` on their generated code, since it will not
        # report any profiling info about kernels, that initialise
        # things only once in the beginning? Maybe get rid of it in a
        # release version? (TODO)
        run_lines.append('CUDA_SAFE_CALL(cudaProfilerStart());')
        # run everything that is run on a clock
        run_lines.append('{net.name}.run({duration!r}, {report_call}, {report_period!r});'.format(net=net,
                                                                                              duration=float(duration),
                                                                                              report_call=report_call,
                                                                                              report_period=float(report_period)))
        # for multiple runs, the random number buffer needs to be reset
        run_lines.append('random_number_buffer.run_finished();')
        # nvprof stuff
        run_lines.append('CUDA_SAFE_CALL(cudaDeviceSynchronize());')
        run_lines.append('CUDA_SAFE_CALL(cudaProfilerStop());')

        self.main_queue.append(('run_network', (net, run_lines)))

        # Manually set the cache for the clocks, simulation scripts might
        # want to access the time (which has been set in code and is therefore
        # not accessible by the normal means until the code has been built and
        # run)
        for clock in net._clocks:
            self.array_cache[clock.variables['timestep']] = np.array([clock._i_end])
            self.array_cache[clock.variables['t']] = np.array([clock._i_end * clock.dt_])

        # Initialize eventspaces with -1 before the network runs
        for codeobj in self.code_objects.values():
            if codeobj.template_name == "threshold" or codeobj.template_name == "spikegenerator":
                for key in codeobj.variables.iterkeys():
                    if key.endswith('space'):  # get the correct eventspace name
                        eventspace_name = self.get_array_name(codeobj.variables[key], False)
                        # In case of custom scheduling, the thresholder might come after synapses or monitors
                        # and needs to be initialized in the beginning of the simulation

                        # See generate_main_source() for main_queue formats

                        # Initialize entire eventspace array with -1 at beginning of main
                        self.main_queue.insert(
                            0,  # list insert position
                            # func            , (arrayname, value, is_dynamic)
                            ('set_by_constant', (eventspace_name, -1, False))
                        )
                        # Set the last value (index N) in the eventspace array to 0 (-> event counter)
                        self.main_queue.insert(
                            1,  # list insert position
                            (
                                'set_by_single_value',  # func
                                # arrayname     , item,                                , value
                                (eventspace_name, "_num_{} - 1".format(eventspace_name), 0)
                            )
                        )

        if self.build_on_run:
            if self.has_been_run:
                raise RuntimeError('The network has already been built and run '
                                   'before. Use set_device with '
                                   'build_on_run=False and an explicit '
                                   'device.build call to use multiple run '
                                   'statements with this device.')
            self.build(direct_call=False, **self.build_options)

        self.first_run = False


def check_codeobj_for_rng(codeobj, check_binomial=False):
    '''
    Count the number of `"rand()"` and `"randn()"` appearances in
    `codeobj.code.cu_file` and store them as attributes in `codeobj.rand_calls`
    and `codeobj.randn_calls`.

    Parameters
    ----------
    codeobj: CodeObjects
        Codeobject with generated CUDA code in `codeobj.code.cu_file`.
    check_binomial: bool, optional
        Wether to also check if `"binomial()"` appears. Default is False.

    Returns
    -------
    binomial_match: bool or None
        If `check_binomial` is True, this tells if `binomial(const int
        vectorisation_idx)` is appearing in `code`, else `None`.
    '''
    # synapses_create_generator uses host side random number generation
    if codeobj.template_name == 'synapses_create_generator':
        if check_binomial:
            return False
        else:
            return None

    # regex explained
    # (?<!...) negative lookbehind: don't match if ... preceeds
    #     XXX: This only excludes #define lines which have exactly one space to '_rand'.
    #          As long is we don't have '#define  _rand' with 2 spaces, thats fine.
    # \b - non-alphanumeric character (word boundary, does not consume a character)
    rand_pattern = r'(?<!#define )\b_rand\(_vectorisation_idx\)'
    matches_rand = re.findall(rand_pattern, codeobj.code.cu_file)
    codeobj.rand_calls = len(matches_rand)

    randn_pattern = r'(?<!#define )\b_randn\(_vectorisation_idx\)'
    matches_randn = re.findall(randn_pattern, codeobj.code.cu_file)
    codeobj.randn_calls = len(matches_randn)

    if codeobj.template_name == 'synapses':
        # We have two if/else code paths in synapses code (homog. / heterog. delay mode),
        # therefore we have twice as much matches for rand/randn
        assert codeobj.rand_calls % 2 == 0
        assert codeobj.randn_calls % 2 == 0
        codeobj.randn_calls /= 2
        codeobj.rand_calls /= 2

    if codeobj.rand_calls > 0:
        # substitute rand/n arguments twice for synapses templates
        repeat = 2 if codeobj.template_name == 'synapses' else 1
        for _ in range(repeat):
            for i in range(0, codeobj.rand_calls):
                codeobj.code.cu_file = re.sub(
                    rand_pattern,
                    "_rand(_vectorisation_idx + {i} * _N)".format(i=i),
                    codeobj.code.cu_file,
                    count=1)

    if codeobj.randn_calls > 0:
        # substitute rand/n arguments twice for synapses templates
        repeat = 2 if codeobj.template_name == 'synapses' else 1
        for _ in range(repeat):
            for i in range(0, codeobj.randn_calls):
                codeobj.code.cu_file = re.sub(
                    randn_pattern,
                    "_randn(_vectorisation_idx + {i} * _N)".format(i=i),
                    codeobj.code.cu_file,
                    count=1)

    binomial = None
    if check_binomial:
        binomial = False
        match = re.search('_binomial\w*\(const int vectorisation_idx\)', codeobj.code.cu_file)
        if match is not None:
            binomial = True

    return binomial


cuda_standalone_device = CUDAStandaloneDevice()

all_devices['cuda_standalone'] = cuda_standalone_device
