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
from brian2.core.clocks import defaultclock
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

logger = get_logger(__name__)


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
        logger.debug('Writing file %s:\n%s' % (filename, contents))
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
        
    def code_object_class(self, codeobj_class=None):
        # Ignore the requested codeobj_class
        return CUDAStandaloneCodeObject

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None,
                    override_conditional_write=None):
        if template_kwds == None:
            template_kwds = {}
        no_or_const_delay_mode = False
        if isinstance(owner, SynapticPathway):
            if owner.variables["delay"].scalar:
                # TODO: this only catches the case, where Synapses(..., delay=1*ms) syntax is used.
                # if no delay is specified at all, we get scalar==False, which should still be caught here.
                # TODO hard coded to False here! Switch to True when implemented
                no_or_const_delay_mode = False
        template_kwds["no_or_const_delay_mode"] = no_or_const_delay_mode
        if template_name == "synapses":
            serializing_mode = "syn"    #no serializing
            for varname in variables.iterkeys():
                if variable_indices[varname] == "_postsynaptic_idx":
                    if serializing_mode == "syn":
                        serializing_mode = "post"
                if variable_indices[varname] == "_presynaptic_idx":
                    serializing_mode = "pre"
            template_kwds["serializing_mode"] = serializing_mode
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
        multiplier = prefs.devices.cuda_standalone.SM_multiplier
        curand_generator_type = prefs.devices.cuda_standalone.random_number_generator_type
        curand_generator_ordering = prefs.devices.cuda_standalone.random_number_generator_ordering
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
                        multiplier=multiplier,
                        curand_generator_type=curand_generator_type,
                        curand_generator_ordering=curand_generator_ordering)
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
		for clock in net._clocks:
                    line = "{net.name}.add(&{clock.name}, _sync_clocks);".format(clock=clock, net=net)
                    netcode.insert(1, line)
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
                    main_lines.append('curandSetPseudoRandomGeneratorSeed(random_float_generator, {seed!r}ULL);'.format(seed=seed))
                    # generator offset needs to be reset to its default (=0)
                    main_lines.append('curandSetGeneratorOffset(random_float_generator, 0ULL);')
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
                                                          additional_headers=main_includes
                                                          )
        writer.write('main.cu', main_tmp)
        
    def generate_codeobj_source(self, writer):
        #check how many random numbers are needed per step
        for code_object in self.code_objects.itervalues():
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
                    additional_code.append('''
                        //genenerate an array of random numbers on the device
                        float* dev_array_randn;
                        cudaMalloc((void**)&dev_array_randn, sizeof(float)*''' + number_elements + ''' * ''' + str(codeobj.randn_calls) + ''');
                        if(!dev_array_randn)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*''' + number_elements + '''*''' + str(codeobj.randn_calls) + ''');
                        }
                        curandGenerateNormal(random_float_generator, dev_array_randn, ''' + number_elements + '''*''' + str(codeobj.randn_calls) + ''', 0, 1);''')
                    line = "float* _array_{name}_randn".format(name=codeobj.name)
                    device_parameters_lines.append(line)
                    host_parameters_lines.append("dev_array_randn")
                elif k == "_python_rand" and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create_generator":
                    additional_code.append('''
                        //genenerate an array of random numbers on the device
                        float* dev_array_rand;
                        cudaMalloc((void**)&dev_array_rand, sizeof(float)*''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');
                        if(!dev_array_rand)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');
                        }
                        curandGenerateUniform(random_float_generator, dev_array_rand, ''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');''')
                    line = "float* _array_{name}_rand".format(name=codeobj.name)
                    device_parameters_lines.append(line)
                    host_parameters_lines.append("dev_array_rand")
                elif isinstance(v, ArrayVariable):
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
                        else:
                            host_parameters_lines.append("dev"+self.get_array_name(v))
                            device_parameters_lines.append("%s* par_%s" % (c_data_type(v.dtype), self.get_array_name(v)))
                            kernel_variables_lines.append("%s* _ptr%s = par_%s;" % (c_data_type(v.dtype),  self.get_array_name(v), self.get_array_name(v)))

                            code_object_defs_lines.append('const int _num%s = %s;' % (k, v.size))
                            kernel_variables_lines.append('const int _num%s = %s;' % (k, v.size))
                    except TypeError:
                        pass
                    
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
                                                           codeobj_with_randn=codeobj_with_randn)
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
        network_tmp = CUDAStandaloneCodeObject.templater.network(None, None, maximum_run_time=maximum_run_time)
        writer.write('network.*', network_tmp)
        
    def generate_synapses_classes_source(self, writer):
        synapses_classes_tmp = CUDAStandaloneCodeObject.templater.synapses_classes(None, None)
        writer.write('synapses_classes.*', synapses_classes_tmp)
        
    def generate_run_source(self, writer, run_includes):
        run_tmp = CUDAStandaloneCodeObject.templater.run(None, None, run_funcs=self.runfuncs,
                                                        code_objects=self.code_objects.values(),
                                                        additional_headers=run_includes,
                                                        array_specs=self.arrays,
                                                        clocks=self.clocks
                                                        )
        writer.write('run.*', run_tmp)
        
    def generate_makefile(self, writer, cpp_compiler, cpp_compiler_flags, nb_threads):
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
                rm_cmd=rm_cmd)
            writer.write('makefile', makefile_tmp)

    def build(self, directory='output',
              compile=True, run=True, debug=False, clean=True,
              with_output=True,
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
        
        logger.debug("Writing CUDA standalone project to directory "+os.path.normpath(directory))
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
        
        self.generate_makefile(writer, cpp_compiler, cpp_compiler_flags, nb_threads=0)
        
        if compile:
            self.compile_source(directory, cpp_compiler, debug, clean)
            if run:
                self.run(directory, with_output, run_args)

    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0, **kwds):
        build_on_run = self.build_on_run
        self.build_on_run = False
        try:  # for testing we need to reset build_on_run in case of errors
            super(CUDAStandaloneDevice, self).network_run(net, duration, report, report_period, namespace, profile, level+1, **kwds)
        finally:
            self.build_on_run = build_on_run
        self.build_on_run = build_on_run
        for codeobj in self.code_objects.values():
            if codeobj.template_name == "threshold" or codeobj.template_name == "spikegenerator":
                for key in codeobj.variables.iterkeys():
                    if key.endswith('space'):  # get the correct eventspace name
                        self.main_queue.insert(0, ('set_by_constant', (self.get_array_name(codeobj.variables[key], False), -1, False)))
        for func, args in self.main_queue:
            if func=='run_network':
                net, netcode = args
                for clock in net._clocks:
                    lines = '''{net.name}.add(&{clock.name}, _run_random_number_generation);'''.format(clock=clock, net=net)
                    if lines not in netcode:
                        netcode.insert(1, lines)

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
