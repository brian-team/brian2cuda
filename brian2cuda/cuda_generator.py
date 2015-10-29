import itertools

import numpy

from brian2.utils.stringtools import (deindent, stripped_deindented_lines,
                                      word_substitute)
from brian2.utils.logger import get_logger
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.core.preferences import prefs, BrianPreference
from brian2.core.variables import ArrayVariable
from brian2.codegen.generators import CPPCodeGenerator, c_data_type

from brian2.codegen.generators.base import CodeGenerator

logger = get_logger(__name__)

__all__ = ['CUDACodeGenerator',
           'c_data_type'
           ]


# Preferences
prefs.register_preferences(
    'codegen.generators.cuda',
    'CUDA codegen preferences',
    restrict_keyword = BrianPreference(
        default='__restrict',
        docs='''
        The keyword used for the given compiler to declare pointers as restricted.
        
        This keyword is different on different compilers, the default works for
        gcc and MSVS.
        ''',
        ),
    flush_denormals = BrianPreference(
        default=False,
        docs='''
        Adds code to flush denormals to zero.
        
        The code is gcc and architecture specific, so may not compile on all
        platforms. The code, for reference is::

            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            
        Found at `<http://stackoverflow.com/questions/2487653/avoiding-denormal-values-in-c>`_.
        ''',
        ),
    )


class CUDACodeGenerator(CPPCodeGenerator):
    '''
    C++ language with CUDA library
    
    CUDA code templates should provide Jinja2 macros with the following names:
    
    ``main``
        The main loop.
    ``support_code``
        The support code (function definitions, etc.), compiled in a separate
        file.
        
    For user-defined functions, there are two keys to provide:
    
    ``support_code``
        The function definition which will be added to the support code.
    ``hashdefine_code``
        The ``#define`` code added to the main loop.
        
    See `TimedArray` for an example of these keys.
    '''

    class_name = 'cuda'

    def __init__(self, *args, **kwds):
        super(CUDACodeGenerator, self).__init__(*args, **kwds)
        
    def determine_keywords(self):
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        lines = []
        # It is possible that several different variable names refer to the
        # same array. E.g. in gapjunction code, v_pre and v_post refer to the
        # same array if a group is connected to itself
        handled_pointers = set()
        template_kwds = {}
        # Again, do the import here to avoid a circular dependency.
        from brian2.devices.device import get_device
        device = get_device()
        for varname, var in self.variables.iteritems():
            if isinstance(var, ArrayVariable):
                # This is the "true" array name, not the restricted pointer.
                array_name = device.get_array_name(var)
                pointer_name = self.get_array_name(var)
                if pointer_name in handled_pointers:
                    continue
                if getattr(var, 'dimensions', 1) > 1:
                    continue  # multidimensional (dynamic) arrays have to be treated differently
                line = self.c_data_type(var.dtype) + ' * ' + self.restrict + pointer_name + ' = ' + array_name + ';'
                lines.append(line)
                handled_pointers.add(pointer_name)

        pointers = '\n'.join(lines)

        # set up the functions
        user_functions = []
        support_code = ''
        hash_defines = ''
        for varname, variable in self.variables.items():
            if isinstance(variable, Function):
                user_functions.append((varname, variable))
                funccode = variable.implementations[self.codeobj_class].get_code(self.owner)
                if isinstance(funccode, basestring):
                    funccode = {'support_code': funccode}
                if funccode is not None:
                    support_code += '\n' + deindent(funccode.get('support_code', ''))
                    hash_defines += '\n' + deindent(funccode.get('hashdefine_code', ''))
                # add the Python function with a leading '_python', if it
                # exists. This allows the function to make use of the Python
                # function via weave if necessary (e.g. in the case of randn)
                if not variable.pyfunc is None:
                    pyfunc_name = '_python_' + varname
                    if pyfunc_name in self.variables:
                        logger.warn(('Namespace already contains function %s, '
                                     'not replacing it') % pyfunc_name)
                    else:
                        self.variables[pyfunc_name] = variable.pyfunc

        # delete the user-defined functions from the namespace and add the
        # function namespaces (if any)
        for funcname, func in user_functions:
            del self.variables[funcname]
            func_namespace = func.implementations[self.codeobj_class].get_namespace(self.owner)
            if func_namespace is not None:
                self.variables.update(func_namespace)

        keywords = {'pointers_lines': stripped_deindented_lines(pointers),
                    'support_code_lines': stripped_deindented_lines(support_code),
                    'hashdefine_lines': stripped_deindented_lines(hash_defines),
                    'denormals_code_lines': stripped_deindented_lines(self.denormals_to_zero_code()),
                    }
        keywords.update(template_kwds)
        return keywords

################################################################################
# Implement functions
################################################################################

# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor']:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CUDACodeGenerator,
                                                               code=None)

# Functions that need a name translation
for func, func_cuda in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                       ('abs', 'fabs')]:
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CUDACodeGenerator,
                                                               code=None,
                                                               name=func_cuda)

# Functions that need to be implemented specifically
randn_code = '''
    #define _randn(vectorisation_idx) (_array_%CODEOBJ_NAME%_randn[vectorisation_idx])
        '''
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CUDACodeGenerator,
                                                              code=randn_code,
                                                              name='_randn')

rand_code = '''
    #define _rand(vectorisation_idx) (_array_%CODEOBJ_NAME%_rand[vectorisation_idx])
    '''
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CUDACodeGenerator,
                                                             code=rand_code,
                                                             name='_rand')

clip_code = '''
    __device__ double _clip(const float value, const float a_min, const float a_max)
    {
        if (value < a_min)
            return a_min;
        if (value > a_max)
            return a_max;
        return value;
    }
    '''
DEFAULT_FUNCTIONS['clip'].implementations.add_implementation(CUDACodeGenerator,
                                                             code=clip_code,
                                                             name='_clip')

int_code = '''
    __device__ int int_(const bool value)
    {
        return value ? 1 : 0;
    }
    '''
DEFAULT_FUNCTIONS['int'].implementations.add_implementation(CUDACodeGenerator,
                                                            code=int_code,
                                                            name='int_')
