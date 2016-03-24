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
from brian2.devices.device import all_devices, get_device, set_device
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

prefs.register_preferences(
    'devices.cuda_standalone',
    'CUDA standalone preferences ',
    SM_multiplier = BrianPreference(
        default=1,
        docs='''
        The number of blocks per SM. By default, this value is set to 1.
        ''',
        ),
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
        no_or_const_delay_mode = "False"
        if isinstance(owner, SynapticPathway):
            if owner.variables["delay"].constant or owner.variables["delay"].scalar or owner.variables["delay"].size == 1:
                no_or_const_delay_mode = "True"
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
                        multiplier=multiplier)
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
                cudaMemcpy({pointer_arrayname}, &{arrayname}[{item}], sizeof({arrayname}[0]), cudaMemcpyHostToDevice);
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
                if code_object.template_name != "synapses_create":
                    #first one is alway the definition, so subtract 1
                    code_object.rand_calls = num_occurences_rand - 1
                    for i in range(0, code_object.rand_calls):
                        if code_object.owner.N != 0:
                            code_object.code.cu_file = code_object.code.cu_file.replace("_rand(_vectorisation_idx)", "_rand(_vectorisation_idx + " + str(i) + " * " + str(code_object.owner.N) + ")", 1)
                        else:
                            code_object.code.cu_file = code_object.code.cu_file.replace("_rand(_vectorisation_idx)", "_rand(_vectorisation_idx + " + str(i) + " * _N)", 1)
                else:
                    code_object.code.cu_file = code_object.code.cu_file.replace("_rand(_vectorisation_idx)", "(rand()/(float)RAND_MAX)")
            if num_occurences_randn > 0 and code_object.template_name != "synapses_create":
                #first one is alway the definition, so subtract 1
                code_object.randn_calls = num_occurences_randn - 1
                for i in range(0, code_object.randn_calls):
                    if code_object.owner.N != 0:
                        code_object.code.cu_file = code_object.code.cu_file.replace("_randn(_vectorisation_idx)", "_randn(_vectorisation_idx + " + str(i) + " * " + str(code_object.owner.N) + ")", 1)
                    else:
                        code_object.code.cu_file = code_object.code.cu_file.replace("_randn(_vectorisation_idx)", "_randn(_vectorisation_idx + " + str(i) + " * _N)", 1)

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
                if k == "_python_randn" and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create":
                    additional_code.append('''
                        //genenerate an array of random numbers on the device
                        float* dev_array_randn;
                        cudaMalloc((void**)&dev_array_randn, sizeof(float)*''' + number_elements + ''' * ''' + str(codeobj.randn_calls) + ''');
                        if(!dev_array_randn)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*''' + number_elements + '''*''' + str(codeobj.randn_calls) + ''');
                        }
                        curandGenerator_t uniform_gen;
                        curandCreateGenerator(&uniform_gen, CURAND_RNG_PSEUDO_DEFAULT);
                        curandSetPseudoRandomGeneratorSeed(uniform_gen, time(0));
                        curandGenerateNormal(uniform_gen, dev_array_randn, ''' + number_elements + '''*''' + str(codeobj.randn_calls) + ''', 0, 1);''')
                    line = "float* _array_{name}_randn".format(name=codeobj.name)
                    device_parameters_lines.append(line)
                    host_parameters_lines.append("dev_array_randn")
                elif k == "_python_rand" and codeobj.runs_every_tick == False and codeobj.template_name != "synapses_create":
                    additional_code.append('''
                        //genenerate an array of random numbers on the device
                        float* dev_array_rand;
                        cudaMalloc((void**)&dev_array_rand, sizeof(float)*''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');
                        if(!dev_array_rand)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');
                        }
                        curandGenerator_t normal_gen;
                        curandCreateGenerator(&normal_gen, CURAND_RNG_PSEUDO_DEFAULT);
                        curandSetPseudoRandomGeneratorSeed(normal_gen, time(0));
                        curandGenerateUniform(normal_gen, dev_array_rand, ''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');''')
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
        shutil.copy2(os.path.join(os.path.split(inspect.getsourcefile(Synapses))[0], 'stdint_compat.h'),
                     os.path.join(directory, 'brianlib', 'stdint_compat.h'))

    def generate_network_source(self, writer, compiler):
        if compiler=='msvc':
            std_move = 'std::move'
        else:
            std_move = ''
        network_tmp = CUDAStandaloneCodeObject.templater.network(None, None,
                                                             std_move=std_move)
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
        
    def generate_makefile(self, writer, compiler, native, compiler_flags, nb_threads):
        if compiler=='msvc':
            if native:
                arch_flag = ''
                try:
                    from cpuinfo import cpuinfo
                    res = cpuinfo.get_cpu_info()
                    if 'sse' in res['flags']:
                        arch_flag = '/arch:SSE'
                    if 'sse2' in res['flags']:
                        arch_flag = '/arch:SSE2'
                except ImportError:
                    logger.warn('Native flag for MSVC compiler requires installation of the py-cpuinfo module')
                compiler_flags += ' '+arch_flag
            
            if nb_threads>1:
                openmp_flag = '/openmp'
            else:
                openmp_flag = ''
            # Generate the visual studio makefile
            source_bases = [fname.replace('.cpp', '').replace('/', '\\') for fname in writer.source_files]
            win_makefile_tmp = CUDAStandaloneCodeObject.templater.win_makefile(
                None, None,
                source_bases=source_bases,
                compiler_flags=compiler_flags,
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
                compiler_flags=compiler_flags,
                rm_cmd=rm_cmd)
            writer.write('makefile', makefile_tmp)


    def build(self, directory='output',
              compile=True, run=True, debug=False, clean=True,
              with_output=True, native=True,
              additional_source_files=None, additional_header_files=None,
              main_includes=None, run_includes=None,
              run_args=None, **kwds):
        '''
        Build the project
        
        TODO: more details
        
        Parameters
        ----------
        directory : str
            The output directory to write the project to, any existing files will be overwritten.
        compile : bool
            Whether or not to attempt to compile the project
        run : bool
            Whether or not to attempt to run the built project if it successfully builds.
        debug : bool
            Whether to compile in debug mode.
        with_output : bool
            Whether or not to show the ``stdout`` of the built program when run.
        native : bool
            Whether or not to compile for the current machine's architecture (best for speed, but not portable)
        clean : bool
            Whether or not to clean the project before building
        additional_source_files : list of str
            A list of additional ``.cpp`` files to include in the build.
        additional_header_files : list of str
            A list of additional ``.h`` files to include in the build.
        main_includes : list of str
            A list of additional header files to include in ``main.cpp``.
        run_includes : list of str
            A list of additional header files to include in ``run.cpp``.
        '''
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

        compiler, extra_compile_args = get_compiler_and_args()
        compiler_flags = ' '.join(extra_compile_args)
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
        self.generate_network_source(writer, compiler)
        self.generate_synapses_classes_source(writer)
        self.generate_run_source(writer, run_includes)
        self.generate_rand_source(writer)
        self.copy_source_files(writer, directory)
        
        writer.source_files.extend(additional_source_files)
        writer.header_files.extend(additional_header_files)
        
        self.generate_makefile(writer, compiler, native, compiler_flags, nb_threads=0)
        
        if compile:
            self.compile_source(directory, compiler, debug, clean, native)
            if run:
                self.run(directory, with_output, run_args)
                
    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0, **kwds):
        CPPStandaloneDevice.network_run(self, net, duration, report, report_period, namespace, profile, level+1)
        for codeobj in self.code_objects.values():
            if codeobj.template_name == "threshold" or codeobj.template_name == "spikegenerator":
                self.main_queue.insert(0, ('set_by_constant', (self.get_array_name(codeobj.variables['_spikespace'], False), -1, False)))
        for func, args in self.main_queue:
            if func=='run_network':
                net, netcode = args
                for clock in net._clocks:
                    lines = '''{net.name}.add(&{clock.name}, _run_random_number_generation);'''.format(clock=clock, net=net)
                    run_action = netcode.pop()
                    if lines not in netcode:
                        netcode.append(lines);
                    netcode.append(run_action)


                        
cuda_standalone_device = CUDAStandaloneDevice()

all_devices['cuda_standalone'] = cuda_standalone_device

class CUDAStandaloneSimpleDevice(CUDAStandaloneDevice):
    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0, **kwds):
        super(CUDAStandaloneSimpleDevice, self).network_run(net, duration,
                                                     report=report,
                                                     report_period=report_period,
                                                     namespace=namespace,
                                                     profile=profile,
                                                     level=level+1,
                                                     **kwds)
        tempdir = tempfile.mkdtemp()
        self.build(directory=tempdir, compile=True, run=True, debug=False,
                   with_output=False)

cuda_standalone_simple_device = CUDAStandaloneSimpleDevice()

all_devices['cuda_standalone_simple'] = cuda_standalone_simple_device
