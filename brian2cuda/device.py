'''
Module implementing the CUDA "standalone" device.
'''
import os
import shutil
import inspect
from collections import defaultdict
import tempfile

import numpy as np

import brian2

from brian2.codegen.cpp_prefs import get_compiler_and_args
from brian2.codegen.translation import make_statements
from brian2.core.clocks import defaultclock
from brian2.core.namespace import get_local_namespace
from brian2.core.network import Network
from brian2.core.preferences import prefs, BrianPreference
from brian2.core.variables import *
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.devices.device import all_devices
from brian2.synapses.synapses import Synapses, SynapticPathway
from brian2.utils.filetools import copy_directory, ensure_directory
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.utils.logger import get_logger
from brian2.units import second

from .codeobject import CUDAStandaloneCodeObject
from brian2.devices.cpp_standalone.device import CPPWriter, CPPStandaloneDevice
from brian2.monitors.statemonitor import StateMonitor
from brian2.groups.neurongroup import Thresholder


__all__ = []

logger = get_logger('brian2.devices.cuda_standalone')


# Preferences
prefs.register_preferences(
    'devices.cuda_standalone',
    'CUDA standalone preferences',

    SM_multiplier = BrianPreference(
        default=1,
        docs='''
        The number of blocks per SM. By default, this value is set to 1.
        ''',
        ),

    parallel_blocks = BrianPreference(
        docs='''
        The total number of parallel blocks to use. The default is the number
        of streaming multiprocessors.
        ''',
        validator=lambda v: v is None or (isinstance(v, int) and v > 0),
        default=None),

    gpu_heap_size = BrianPreference(
        docs='''
        Size of the heap (in MB) used by malloc() and free() device system calls, which
        are used in the `cudaVector` implementation. `cudaVectors` are used to
        dynamically allocate device memory for `Spikemonitors` and the synapse
        queues in the `CudaSpikeQueue` implementation for networks with
        heterogeneously distributed delays.
        ''',
        validator=lambda v: isinstance(v, int) and v >= 0,
        default=128),

    curand_float_type=BrianPreference(
        docs='''
        Floating point type of generated random numbers (float/double).
        ''',
        validator=lambda v: v in ['float', 'double'],
        default='float'),

    launch_bounds=BrianPreference(
        docs='''
        Weather or not to use `__launch_bounds__` to optimise register usage in kernels.
        ''',
        validator=lambda v: isinstance(v, bool),
        default=False),

    syn_launch_bounds=BrianPreference(
        docs='''
        Weather or not to use `__launch_bounds__` in synapses and synapses_push to optimise register usage in kernels.
        ''',
        validator=lambda v: isinstance(v, bool),
        default=False),

    calc_occupancy=BrianPreference(
        docs='''
        Weather or not to use cuda occupancy api to choose num_threads and num_blocks.
        ''',
        validator=lambda v: isinstance(v, bool),
        default=True),

    extra_threshold_kernel=BrianPreference(
        docs='''
        Weather or not to use a extra threshold kernel for resetting or not.
        ''',
        validator=lambda v: isinstance(v, bool),
        default=True),

    random_number_generator_type=BrianPreference(
        docs='''Generator type (str) that cuRAND uses for random number generation.
            Setting the generator type automatically resets the generator ordering
            (prefs.devices.cuda_standalone.random_number_generator_ordering) to its default value.
            See cuRAND documentation for more details on generator types and orderings.''',
        validator=lambda v: v in ['CURAND_RNG_PSEUDO_DEFAULT',
                                  'CURAND_RNG_PSEUDO_XORWOW',
                                  'CURAND_RNG_PSEUDO_MRG32K3A',
                                  'CURAND_RNG_PSEUDO_MTGP32',
                                  'CURAND_RNG_PSEUDO_PHILOX4_32_10',
                                  'CURAND_RNG_PSEUDO_MT19937',
                                  'CURAND_RNG_QUASI_DEFAULT',
                                  'CURAND_RNG_QUASI_SOBOL32',
                                  'CURAND_RNG_QUASI_SCRAMBLED_SOBOL32',
                                  'CURAND_RNG_QUASI_SOBOL64',
                                  'CURAND_RNG_QUASI_SCRAMBLED_SOBOL64'],
        default='CURAND_RNG_PSEUDO_DEFAULT'),

    random_number_generator_ordering=BrianPreference(
        docs='''The ordering parameter (str) used to choose how the results of cuRAND
            random number generation are ordered in global memory.
            See cuRAND documentation for more details on generator types and orderings.''',
        validator=lambda v: not v or v in ['CURAND_ORDERING_PSEUDO_DEFAULT',
                                           'CURAND_ORDERING_PSEUDO_BEST',
                                           'CURAND_ORDERING_PSEUDO_SEEDED',
                                           'CURAND_ORDERING_QUASI_DEFAULT'],
        default=False)  # False will prevent setting ordering in objects.cu (-> curRAND will uset the correct ..._DEFAULT)
)


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
        self.active_objects = set()
        
    def code_object_class(self, codeobj_class=None):
        # Ignore the requested codeobj_class
        return CUDAStandaloneCodeObject

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None,
                    override_conditional_write=None):
        if template_kwds == None:
            template_kwds = {}
        if hasattr(self, 'profile'):
            template_kwds['profile'] = self.profile
        no_or_const_delay_mode = False
        if isinstance(owner, (SynapticPathway, Synapses)) and "delay" in owner.variables and owner.variables["delay"].scalar:
            # catches Synapses(..., delay=...) syntax, does not catch the case when no delay is specified at all
                no_or_const_delay_mode = True
        template_kwds["no_or_const_delay_mode"] = no_or_const_delay_mode
        if template_name == "synapses":
            ##################################################################
            # This code is copied from CodeGenerator.translate() and CodeGenerator.array_read_write()
            # and should give us a set of variables to which will be written in `vector_code`
            vector_statements = {}
            for ac_name, ac_code in abstract_code.iteritems():
                statements = make_statements(ac_code,
                                             variables,
                                             prefs['core.default_float_dtype'],
                                             optimise=True,
                                             blockname=ac_name)
                _, vector_statements[ac_name] = statements
            write = set()
            for statements in vector_statements.itervalues():
                for stmt in statements:
                    write.add(stmt.var)
            write = set(varname for varname, var in variables.items()
                        if isinstance(var, ArrayVariable) and varname in write)
            ##################################################################
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
            print('debug syn effect mdoe ', synaptic_effects)
            logger.debug("Synaptic effects of Synapses object {syn} modify {mod} group variables.".format(syn=name, mod=synaptic_effects))
        if template_name in ["synapses_create_generator", "synapses_create_array"]:
            if owner.multisynaptic_index is not None:
                template_kwds["multisynaptic_idx_var"] = owner.variables[owner.multisynaptic_index]
        template_kwds["launch_bounds"] = prefs["devices.cuda_standalone.launch_bounds"]
        template_kwds["sm_multiplier"] = prefs["devices.cuda_standalone.SM_multiplier"]
        template_kwds["syn_launch_bounds"] = prefs["devices.cuda_standalone.syn_launch_bounds"]
        template_kwds["calc_occupancy"] = prefs["devices.cuda_standalone.calc_occupancy"]
        if template_name == "threshold":
            template_kwds["extra_threshold_kernel"] = prefs["devices.cuda_standalone.extra_threshold_kernel"]
        codeobj = super(CUDAStandaloneDevice, self).code_object(owner, name, abstract_code, variables,
                                                               template_name, variable_indices,
                                                               codeobj_class=codeobj_class,
                                                               template_kwds=template_kwds,
                                                               override_conditional_write=override_conditional_write
                                                               )
        return codeobj
    
    def check_openmp_compatible(self, nb_threads):
        if nb_threads > 0:
            raise NotImplementedError("Using OpenMP in an CUDA standalone project is not supported")
        
    def generate_objects_source(self, writer, arange_arrays, synapses, static_array_specs, networks):
        codeobj_with_rand = [co for co in self.code_objects.values() if co.runs_every_tick and co.rand_calls > 0]
        codeobj_with_randn = [co for co in self.code_objects.values() if co.runs_every_tick and co.randn_calls > 0]
        sm_multiplier = prefs.devices.cuda_standalone.SM_multiplier
        num_parallel_blocks = prefs.devices.cuda_standalone.parallel_blocks
        curand_generator_type = prefs.devices.cuda_standalone.random_number_generator_type
        curand_generator_ordering = prefs.devices.cuda_standalone.random_number_generator_ordering
        self.eventspace_arrays = {}
        for var, varname in self.arrays.iteritems():
            if var.name.endswith('space'):  # get all eventspace variables
                self.eventspace_arrays[var] = varname
        for var in self.eventspace_arrays.iterkeys():
            del self.arrays[var]
        multisyn_vars = []
        for syn in synapses:
            if syn.multisynaptic_index is not None:
                multisyn_vars.append(syn.variables[syn.multisynaptic_index])
        arr_tmp = CUDAStandaloneCodeObject.templater.objects(
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
                        codeobj_with_rand=codeobj_with_rand,
                        codeobj_with_randn=codeobj_with_randn,
                        sm_multiplier=sm_multiplier,
                        num_parallel_blocks=num_parallel_blocks,
                        curand_generator_type=curand_generator_type,
                        curand_generator_ordering=curand_generator_ordering,
                        curand_float_type=prefs['devices.cuda_standalone.curand_float_type'],
                        eventspace_arrays=self.eventspace_arrays,
                        multisynaptic_idx_vars=multisyn_vars,
                        active_objects=self.active_objects,
                        profile=self.profile)
        # Reinsert deleted entries, in case we use self.arrays later? maybe unnecassary...
        self.arrays.update(self.eventspace_arrays)
        writer.write('objects.*', arr_tmp)

    def generate_main_source(self, writer, main_includes):
        main_lines = []
        procedures = [('', main_lines)]
        runfuncs = {}
        for func, args in self.main_queue:
            if func=='run_code_object':
                codeobj, = args
                codeobj.runs_every_tick = False
                main_lines.append('_run_%s();' % codeobj.name)
            elif func=='run_network':
                net, netcode = args
                main_lines.extend(netcode)
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
                line = "cudaMemcpy({pointer_arrayname}, &{arrayname}[0], sizeof({arrayname}[0])*{size_str}, cudaMemcpyHostToDevice);".format(
                           arrayname=arrayname, size_str=size_str, pointer_arrayname=pointer_arrayname, 
                           value=CPPNodeRenderer().render_expr(repr(value)))
                main_lines.extend([line])
            elif func=='set_by_single_value':
                arrayname, item, value = args
		pointer_arrayname = "dev{arrayname}".format(arrayname=arrayname)
                if arrayname in self.dynamic_arrays.values():
                    pointer_arrayname = "thrust::raw_pointer_cast(&dev{arrayname}[0])".format(arrayname=arrayname)
                code = '''
                {arrayname}[{item}] = {value};
                cudaMemcpy(&{pointer_arrayname}[{item}], &{arrayname}[{item}], sizeof({arrayname}[0]), cudaMemcpyHostToDevice);
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
                cudaMemcpy({pointer_arrayname}, &{arrayname}[0], sizeof({arrayname}[0])*{size_str}, cudaMemcpyHostToDevice);
                '''.format(pointer_arrayname=pointer_arrayname, staticarrayname=staticarrayname, size_str=size, arrayname=arrayname)
                main_lines.extend([line])
            elif func=='set_array_by_array':
                arrayname, staticarrayname_index, staticarrayname_value = args
                code = '''
                for(int i=0; i<_num_{staticarrayname_index}; i++)
                {{
                    {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                }}
                cudaMemcpy(dev{arrayname}, &{arrayname}[0], sizeof({arrayname}[0])*_num_{arrayname}, cudaMemcpyHostToDevice);
                '''.format(arrayname=arrayname, staticarrayname_index=staticarrayname_index,
                           staticarrayname_value=staticarrayname_value)
                main_lines.extend(code.split('\n'))
            elif func=='resize_array':
                array_name, new_size = args
                main_lines.append('''
                    {array_name}.resize({new_size});
                    dev{array_name}.resize({new_size});
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
                if seed is not None:
                    main_lines.append('curandSetPseudoRandomGeneratorSeed(curand_generator, {seed!r}ULL);'.format(seed=seed))
                    # generator offset needs to be reset to its default (=0)
                    main_lines.append('curandSetGeneratorOffset(curand_generator, 0ULL);')
                # else a random seed is set in objects.cu::_init_arrays()
            else:
                raise NotImplementedError("Unknown main queue function type "+func)

        # generate the finalisations
        for codeobj in self.code_objects.itervalues():
            if hasattr(codeobj.code, 'main_finalise'):
                main_lines.append(codeobj.code.main_finalise)
                
        main_tmp = CUDAStandaloneCodeObject.templater.main(None, None,
                                                          main_lines=main_lines,
                                                          code_objects=self.code_objects.values(),
                                                          report_func=self.report_func,
                                                          dt=float(defaultclock.dt),
                                                          additional_headers=main_includes,
                                                          gpu_heap_size=prefs['devices.cuda_standalone.gpu_heap_size']
                                                          )
        writer.write('main.cu', main_tmp)
        
    def generate_codeobj_source(self, writer):
        #check how many random numbers are needed per step
        for code_object in self.code_objects.itervalues():
            # TODO: this needs better checking, what if someone defines a custom funtion `my_rand()`?
            num_occurences_rand = code_object.code.cu_file.count("_rand(")
            num_occurences_randn = code_object.code.cu_file.count("_randn(")
            if num_occurences_rand > 0:
                # synapses_create_generator uses host side random number generation
                if code_object.template_name != "synapses_create_generator":
                    #first one is alway the definition, so subtract 1
                    code_object.rand_calls = num_occurences_rand - 1
                    for i in range(0, code_object.rand_calls):
                        code_object.code.cu_file = code_object.code.cu_file.replace(
                            "_rand(_vectorisation_idx)",
                            "_rand(_vectorisation_idx + {i} * _N)".format(i=i),
                            1)
            if num_occurences_randn > 0 and code_object.template_name != "synapses_create_generator":
                #first one is alway the definition, so subtract 1
                code_object.randn_calls = num_occurences_randn - 1
                for i in range(0, code_object.randn_calls):
                    code_object.code.cu_file = code_object.code.cu_file.replace(
                        "_randn(_vectorisation_idx)",
                        "_randn(_vectorisation_idx + {i} * _N)".format(i=i),
                        1)

        code_object_defs = defaultdict(list)
        host_parameters = defaultdict(list)
        device_parameters = defaultdict(list)
        kernel_variables = defaultdict(list)
        # Generate data for non-constant values
        for codeobj in self.code_objects.itervalues():
            code_object_defs_lines = []
            host_parameters_lines = []
            device_parameters_lines = []
            kernel_variables_lines = []
            additional_code = []
            number_elements = ""
            if hasattr(codeobj, 'owner') and hasattr(codeobj.owner, '_N') and codeobj.owner._N != 0:
                number_elements = str(codeobj.owner._N)
            else:
                number_elements = "_N"
            for k, v in codeobj.variables.iteritems():
                #code objects which only run once
                if k == "_python_randn" and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create_generator":
                    code_snippet='''
                        //genenerate an array of random numbers on the device
                        {dtype}* dev_array_randn;
                        cudaMalloc((void**)&dev_array_randn, sizeof({dtype})*{number_elements}*{codeobj.randn_calls});
                        if(!dev_array_randn)
                        {{
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof({dtype})*{number_elements}*{codeobj.randn_calls});
                        }}
                        curandGenerateNormal{curand_suffix}(curand_generator, dev_array_randn, {number_elements}*{codeobj.randn_calls}, 0, 1);
                        '''.format(number_elements=number_elements, codeobj=codeobj, dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                   curand_suffix='Double' if prefs['devices.cuda_standalone.curand_float_type']=='double' else '')
                    additional_code.append(code_snippet)
                    line = "{dtype}* par_array_{name}_randn".format(dtype=prefs['devices.cuda_standalone.curand_float_type'], name=codeobj.name)
                    device_parameters_lines.append(line)
                    kernel_variables_lines.append("{dtype}* _ptr_array_{name}_randn = par_array_{name}_randn;".format(dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                                                                                          name=codeobj.name))
                    host_parameters_lines.append("dev_array_randn")
                elif k == "_python_rand" and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create_generator":
                    code_snippet = '''
                        //genenerate an array of random numbers on the device
                        {dtype}* dev_array_rand;
                        cudaMalloc((void**)&dev_array_rand, sizeof({dtype})*{number_elements}*{codeobj.rand_calls});
                        if(!dev_array_rand)
                        {{
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof({dtype})*{number_elements}*{codeobj.rand_calls});
                        }}
                        curandGenerateUniform{curand_suffix}(curand_generator, dev_array_rand, {number_elements}*{codeobj.rand_calls});
                        '''.format(number_elements=number_elements, codeobj=codeobj, dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                   curand_suffix='Double' if prefs['devices.cuda_standalone.curand_float_type']=='double' else '')
                    additional_code.append(code_snippet)
                    line = "{dtype}* par_array_{name}_rand".format(dtype=prefs['devices.cuda_standalone.curand_float_type'], name=codeobj.name)
                    device_parameters_lines.append(line)
                    kernel_variables_lines.append("{dtype}* _ptr_array_{name}_rand = par_array_{name}_rand;".format(dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                                                                                          name=codeobj.name))
                    host_parameters_lines.append("dev_array_rand")
                elif isinstance(v, ArrayVariable):
                    if k in ['t', 'timestep', '_clock_t', '_clock_timestep', '_source_t', '_source_timestep'] and v.scalar:  # monitors have not scalar t variables
                        arrayname = self.get_array_name(v)
                        host_parameters_lines.append(arrayname + '[0]')
                        device_parameters_lines.append("{dtype} par_{name}".format(dtype=c_data_type(v.dtype), name=arrayname))
                        kernel_variables_lines.append("const {dtype} _ptr{name} = par_{name};".format(dtype=c_data_type(v.dtype), name=arrayname))
                    else:
                        try:
                            if isinstance(v, DynamicArrayVariable):
                                if v.dimensions == 1:
                                    dyn_array_name = self.dynamic_arrays[v]
                                    array_name = self.arrays[v]
                                    line = '{c_type}* const {array_name} = thrust::raw_pointer_cast(&dev{dyn_array_name}[0]);'
                                    line = line.format(c_type=c_data_type(v.dtype), array_name=array_name,
                                                       dyn_array_name=dyn_array_name)
                                    code_object_defs_lines.append(line)
                                    line = 'const int _num{k} = dev{dyn_array_name}.size();'
                                    line = line.format(k=k, dyn_array_name=dyn_array_name)
                                    code_object_defs_lines.append(line)

                                    host_parameters_lines.append(array_name)
                                    host_parameters_lines.append("_num" + k)

                                    line = "{c_type}* par_{array_name}"
                                    device_parameters_lines.append(line.format(c_type=c_data_type(v.dtype), array_name=array_name))
                                    line = "int par_num_{array_name}"
                                    device_parameters_lines.append(line.format(array_name=k))

                                    line = "{c_type}* _ptr{array_name} = par_{array_name};"
                                    kernel_variables_lines.append(line.format(c_type=c_data_type(v.dtype), array_name=array_name))
                                    line = "const int _num{array_name} = par_num_{array_name};"
                                    kernel_variables_lines.append(line.format(array_name=k))
                            else:  # v is ArrayVariable but not DynamicArrayVariable
                                arrayname = self.get_array_name(v)
                                host_parameters_lines.append("dev"+arrayname)
                                device_parameters_lines.append("%s* par_%s" % (c_data_type(v.dtype), arrayname))
                                kernel_variables_lines.append("%s* _ptr%s = par_%s;" % (c_data_type(v.dtype),  arrayname, arrayname))

                                code_object_defs_lines.append('const int _num%s = %s;' % (k, v.size))
                                kernel_variables_lines.append('const int _num%s = %s;' % (k, v.size))
                                if k.endswith('space'):
                                    host_parameters_lines[-1] += '[current_idx{arrayname}]'.format(arrayname=arrayname)
                        except TypeError:
                            pass
                    
            # This rand stuff got a little messy... we pass a device pointer as kernel variable and have a hash define for rand() -> _ptr_..._rand[]
            # The device pointer is advanced every clock cycle in rand.cu and reset when the random number buffer is refilled (also in rand.cu)
            # TODO can we just include this in the k == '_python_rand' test above?
            if codeobj.rand_calls >= 1 and codeobj.runs_every_tick:
                host_parameters_lines.append("dev_{name}_rand".format(name=codeobj.name))
                device_parameters_lines.append("{dtype}* par_array_{name}_rand".format(dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                                                                name=codeobj.name))
                kernel_variables_lines.append("{dtype}* _ptr_array_{name}_rand = par_array_{name}_rand;".format(dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                                                                  name=codeobj.name))
            if codeobj.randn_calls >= 1 and codeobj.runs_every_tick:
                host_parameters_lines.append("dev_{name}_randn".format(name=codeobj.name))
                device_parameters_lines.append("{dtype}* par_array_{name}_randn".format(dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                                                                name=codeobj.name))
                kernel_variables_lines.append("{dtype}* _ptr_array_{name}_randn = par_array_{name}_randn;".format(dtype=prefs['devices.cuda_standalone.curand_float_type'],
                                                                                  name=codeobj.name))

            # Sometimes an array is referred to by to different keys in our
            # dictionary -- make sure to never add a line twice
            for line in code_object_defs_lines:
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)
            for line in host_parameters_lines:
                if not line in host_parameters[codeobj.name]:
                    host_parameters[codeobj.name].append(line)
            for line in device_parameters_lines:
                if not line in device_parameters[codeobj.name]:
                    device_parameters[codeobj.name].append(line)
            for line in kernel_variables_lines:
                if not line in kernel_variables[codeobj.name]:
                    kernel_variables[codeobj.name].append(line)
            
            for line in additional_code:
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)
        
        # Generate the code objects
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.variables
            # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant.
            code = self.freeze(codeobj.code.cu_file, ns)
                        
            if len(host_parameters[codeobj.name]) == 0:
                host_parameters[codeobj.name].append("0")
                device_parameters[codeobj.name].append("int dummy")
                
            code = code.replace('%CONSTANTS%', '\n\t\t'.join(code_object_defs[codeobj.name]))
            code = code.replace('%HOST_PARAMETERS%', ',\n\t\t\t'.join(host_parameters[codeobj.name]))
            code = code.replace('%DEVICE_PARAMETERS%', ',\n\t'.join(device_parameters[codeobj.name]))
            code = code.replace('%KERNEL_VARIABLES%', '\n\t'.join(kernel_variables[codeobj.name]))
            code = code.replace('%CODEOBJ_NAME%', codeobj.name)
            code = '#include "objects.h"\n'+code
            
            writer.write('code_objects/'+codeobj.name+'.cu', code)
            writer.write('code_objects/'+codeobj.name+'.h', codeobj.code.h_file)
            
    def generate_rand_source(self, writer):
        codeobj_with_rand = [co for co in self.code_objects.values() if co.runs_every_tick and co.rand_calls > 0]
        codeobj_with_randn = [co for co in self.code_objects.values() if co.runs_every_tick and co.randn_calls > 0]
        rand_tmp = CUDAStandaloneCodeObject.templater.rand(None, None,
                                                           code_objects=self.code_objects.values(),
                                                           codeobj_with_rand=codeobj_with_rand,
                                                           codeobj_with_randn=codeobj_with_randn,
                                                           profile=self.profile,
                                                           curand_float_type=prefs['devices.cuda_standalone.curand_float_type'])
        writer.write('rand.*', rand_tmp)
    
    def copy_source_files(self, writer, directory):
        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(os.path.split(inspect.getsourcefile(CUDAStandaloneCodeObject))[0],
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
        network_tmp = CUDAStandaloneCodeObject.templater.network(None, None,
                                                                 maximum_run_time=maximum_run_time,
                                                                 eventspace_arrays=self.eventspace_arrays,
                                                                 profile=self.profile)
        writer.write('network.*', network_tmp)
        
    def generate_synapses_classes_source(self, writer):
        synapses_classes_tmp = CUDAStandaloneCodeObject.templater.synapses_classes(None, None)
        writer.write('synapses_classes.*', synapses_classes_tmp)
        
    def generate_run_source(self, writer, run_includes):
        run_tmp = CUDAStandaloneCodeObject.templater.run(None, None, run_funcs=self.runfuncs,
                                                        code_objects=self.code_objects.values(),
                                                        additional_headers=run_includes,
                                                        array_specs=self.arrays,
                                                        clocks=self.clocks,
                                                        profile=self.profile
                                                        )
        writer.write('run.*', run_tmp)
        
    def generate_makefile(self, writer, cpp_compiler, cpp_compiler_flags, nb_threads, disable_asserts=False):
        nvcc_compiler_flags = prefs.codegen.cuda.extra_compile_args_nvcc
        gpu_arch_flags = ['']
        disable_warnings = False
        for flag in nvcc_compiler_flags:
            if flag.startswith(('--gpu-architecture', '-arch', '--gpu-code', '-code', '--generate-code', '-gencode')):
                gpu_arch_flags.append(flag)
                nvcc_compiler_flags.remove(flag)
            elif flag.startswith(('-w', '--disable-warnings')):
                disable_warnings = True
                nvcc_compiler_flags.remove(flag)
        nvcc_optimization_flags = ' '.join(nvcc_compiler_flags)
        gpu_arch_flags = ' '.join(gpu_arch_flags)
        if cpp_compiler=='msvc':
            if nb_threads>1:
                openmp_flag = '/openmp'
            else:
                openmp_flag = ''
            # Generate the visual studio makefile
            source_bases = [fname.replace('.cpp', '').replace('/', '\\') for fname in writer.source_files]
            win_makefile_tmp = CUDAStandaloneCodeObject.templater.win_makefile(
                None, None,
                source_bases=source_bases,
                cpp_compiler_flags=cpp_compiler_flags,
                openmp_flag=openmp_flag,
                )
            writer.write('win_makefile', win_makefile_tmp)
        else:
            # Generate the makefile
            if os.name=='nt':
                rm_cmd = 'del *.o /s\n\tdel main.exe $(DEPS)'
            else:
                rm_cmd = 'rm $(OBJS) $(PROGRAM) $(DEPS)'
            makefile_tmp = CUDAStandaloneCodeObject.templater.makefile(None, None,
                source_files=' '.join(writer.source_files),
                header_files=' '.join(writer.header_files),
                cpp_compiler_flags=cpp_compiler_flags,
                nvcc_optimization_flags=nvcc_optimization_flags,
                gpu_arch_flags=gpu_arch_flags,
                disable_warnings=disable_warnings,
                disable_asserts=disable_asserts,
                rm_cmd=rm_cmd)
            writer.write('makefile', makefile_tmp)

    def build(self, directory='output',
              compile=True, run=True, debug=False, clean=True,
              profile=None, with_output=True, disable_asserts=False,
              additional_source_files=None, additional_header_files=None,
              main_includes=None, run_includes=None,
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
            ``True``.
        additional_source_files : list of str, optional
            A list of additional ``.cu`` files to include in the build.
        additional_header_files : list of str
            A list of additional ``.h`` files to include in the build.
        main_includes : list of str
            A list of additional header files to include in ``main.cu``.
        run_includes : list of str
            A list of additional header files to include in ``run.cu``.
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
                    msg += "Unknown keyword argument '%s'. " % kwd
            raise TypeError(msg)

        if not profile is None:
            raise TypeError("The profile argument has to be set in `set_device()`, not in `device.build()`.")

        if debug and disable_asserts:
            logger.warn("You have disabled asserts in debug mode. Are you sure this is what you wanted to do?")

        if additional_source_files is None:
            additional_source_files = []
        if additional_header_files is None:
            additional_header_files = []
        if main_includes is None:
            main_includes = []
        if run_includes is None:
            run_includes = []
        if run_args is None:
            run_args = []
        if directory is None:
            directory = tempfile.mkdtemp()
    
        cpp_compiler, cpp_extra_compile_args = get_compiler_and_args()
        cpp_compiler_flags = ' '.join(cpp_extra_compile_args)
        self.project_dir = directory
        ensure_directory(directory)
        
        for d in ['code_objects', 'results', 'static_arrays']:
            ensure_directory(os.path.join(directory, d))
            
        writer = CUDAWriter(directory)
        
        logger.diagnostic("Writing CUDA standalone project to directory "+os.path.normpath(directory))
        arange_arrays = sorted([(var, start)
                                for var, start in self.arange_arrays.iteritems()],
                               key=lambda (var, start): var.name)
    
        self.write_static_arrays(directory)
        self.find_synapses()
    
        # Not sure what the best place is to call Network.after_run -- at the
        # moment the only important thing it does is to clear the objects stored
        # in magic_network. If this is not done, this might lead to problems
        # for repeated runs of standalone (e.g. in the test suite).
        for net in self.networks:
            net.after_run()
            
        self.generate_main_source(writer, main_includes)
        self.generate_codeobj_source(writer)
        self.generate_objects_source(writer, arange_arrays, self.net_synapses, self.static_array_specs, self.networks)
        self.generate_network_source(writer)
        self.generate_synapses_classes_source(writer)
        self.generate_run_source(writer, run_includes)
        self.generate_rand_source(writer)
        self.copy_source_files(writer, directory)
        
        writer.source_files.extend(additional_source_files)
        writer.header_files.extend(additional_header_files)
        
        self.generate_makefile(writer, cpp_compiler, cpp_compiler_flags, nb_threads=0, disable_asserts=disable_asserts)
        
        if compile:
            self.compile_source(directory, cpp_compiler, debug, clean)
            if run:
                self.run(directory, with_output, run_args)

    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0, **kwds):

        if not isinstance(profile, bool) or not profile:  # everything but True
            raise TypeError("The profile argument has to be set in `set_device()`, not in `run()`")

        if 'profile' in self.build_options:
            profile = self.build_options.pop('profile')
            assert 'profile' not in self.build_options
            if not isinstance(profile, bool) and not profile == 'blocking':
                raise TypeError("Unknown profile argument in `set_device()`. Has to be bool or 'blocking'. "
                                "Got {} ({})".format(profile, type(profile)))
            self.profile = profile
        else:
            self.profile = False  # default

        ###################################################
        ### This part is copied from CPPStandaoneDevice ###
        ###################################################
        if kwds:
            logger.warn(('Unsupported keyword argument(s) provided for run: '
                         '%s') % ', '.join(kwds.keys()))
        net._clocks = {obj.clock for obj in net.objects}
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
        for obj in net.objects:
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
                raise NotImplementedError('The C++ standalone device does not '
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

        # For profiling variables we need a unique set of all active objects in the simulation over possibly multiple runs
        self.active_objects.update([obj[1].name for obj in code_objects])

        # Generate the updaters
        run_lines = ['{net.name}.clear();'.format(net=net)]

        # create all random numbers needed for the next clock cycle
	for clock in net._clocks:
            run_lines.append('{net.name}.add(&{clock.name}, _run_random_number_generation, &random_number_generation_timer_start, '
                    '&random_number_generation_timer_stop, &random_number_generation_profiling_info);'.format(clock=clock, net=net))

        all_clocks = set()
        for clock, codeobj in code_objects:
            run_lines.append('{net.name}.add(&{clock.name}, _run_{codeobj.name}, &{codeobj.name}_timer_start, '
                             '&{codeobj.name}_timer_stop, &{codeobj.name}_profiling_info);'.format(clock=clock,
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
                run_lines.append('{net.name}.add(&{clock.name}, NULL, NULL, NULL, NULL);'.format(clock=clock, net=net))

        if True:#self.profile and self.profile != 'blocking':  # self.profile == True
            run_lines.append('cudaProfilerStart();')
        run_lines.append('{net.name}.run({duration!r}, {report_call}, {report_period!r});'.format(net=net,
                                                                                              duration=float(duration),
                                                                                              report_call=report_call,
                                                                                              report_period=float(report_period)))
        if True:#self.profile and self.profile != 'blocking':  # self.profile == True
            run_lines.append('cudaDeviceSynchronize();')
            run_lines.append('cudaProfilerStop();')
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
                        # In case of custom scheduling, the thresholder might come after synapses or monitors
                        # and needs to be initialized in the beginning of the simulation
                        self.main_queue.insert(0, ('set_by_constant', (self.get_array_name(codeobj.variables[key], False), -1, False)))

        if self.build_on_run:
            if self.has_been_run:
                raise RuntimeError('The network has already been built and run '
                                   'before. Use set_device with '
                                   'build_on_run=False and an explicit '
                                   'device.build call to use multiple run '
                                   'statements with this device.')
            self.build(direct_call=False, **self.build_options)



                        
cuda_standalone_device = CUDAStandaloneDevice()

all_devices['cuda_standalone'] = cuda_standalone_device
