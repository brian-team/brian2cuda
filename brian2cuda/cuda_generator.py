import itertools

import numpy

from brian2.utils.stringtools import (deindent, stripped_deindented_lines,
                                      word_substitute)
from brian2.utils.logger import get_logger
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.core.preferences import prefs
from brian2.core.variables import ArrayVariable
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.codegen.generators.base import CodeGenerator

logger = get_logger(__name__)

__all__ = ['CUDACodeGenerator',
           'c_data_type'
           ]

# CUDA does not support modulo arithmetics for long double. Since we can't give a warning, we let the
# compilation fail, which gives an error message of type
# error: more than one instance of overloaded function "_brian_mod" matches the argument list: ...
# TODO: can we produce a more informative error message?
mod_support_code = ''
typestrs = ['unsigned char', 'char', 'unsigned short', 'short', 'unsigned int', 'int', 'unsigned long', 'long',
            'unsigned long long', 'long long', 'float', 'double']#, 'long double']
floattypestrs = ['float', 'double']#, 'long double']
for ix, xtype in enumerate(typestrs):
    for iy, ytype in enumerate(typestrs):
        hightype = typestrs[max(ix, iy)]
        if xtype in floattypestrs or ytype in floattypestrs:
            expr = 'fmod(fmod(x, y)+y, y)'
        else:
            expr = '((x%y)+y)%y'
        mod_support_code += '''
        inline __host__ __device__ {hightype} _brian_mod({xtype} ux, {ytype} uy)
        {{
            const {hightype} x = ({hightype})ux;
            const {hightype} y = ({hightype})uy;
            return {expr};
        }}
        '''.format(hightype=hightype, xtype=xtype, ytype=ytype, expr=expr)

_universal_support_code = deindent(mod_support_code)+'''
#ifdef _MSC_VER
#define _brian_pow(x, y) (pow((double)(x), (y)))
#else
#define _brian_pow(x, y) (pow((x), (y)))
#endif
'''

class CUDACodeGenerator(CodeGenerator):
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

    universal_support_code = _universal_support_code

    def __init__(self, *args, **kwds):
        super(CUDACodeGenerator, self).__init__(*args, **kwds)
        self.c_data_type = c_data_type
        
    @property
    def restrict(self):
        return prefs['codegen.generators.cpp.restrict_keyword'] + ' '

    @property
    def flush_denormals(self):
        return prefs['codegen.generators.cpp.flush_denormals']

    @staticmethod
    def get_array_name(var, access_data=True):
        # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device
        device = get_device()
        if access_data:
            return '_ptr' + device.get_array_name(var)
        else:
            return device.get_array_name(var, access_data=False)

    def translate_expression(self, expr):
        expr = word_substitute(expr, self.func_name_replacements)
        return CPPNodeRenderer().render_expr(expr).strip()

    def translate_statement(self, statement):
        var, op, expr, comment = (statement.var, statement.op,
                                  statement.expr, statement.comment)
        # For C++ we replace complex expressions involving boolean variables into a sequence of
        # if/then expressions with simpler expressions. This is provided by the optimise_statements
        # function.
        if statement.used_boolean_variables is not None and len(statement.used_boolean_variables):
            used_boolvars = statement.used_boolean_variables
            bool_simp = statement.boolean_simplified_expressions
            if op == ':=':
                # we have to declare the variable outside the if/then statement (which
                # unfortunately means we can't make it const but the optimisation is worth
                # it anyway).
                codelines = [self.c_data_type(statement.dtype) + ' ' + var + ';']
                op = '='
            else:
                codelines = []
            firstline = True
            # bool assigns is a sequence of (var, value) pairs giving the conditions under
            # which the simplified expression simp_expr holds
            for bool_assigns, simp_expr in bool_simp.iteritems():
                # generate a boolean expression like ``var1 && var2 && !var3``
                atomics = []
                for boolvar, boolval in bool_assigns:
                    if boolval:
                        atomics.append(boolvar)
                    else:
                        atomics.append('!'+boolvar)
                if firstline:
                    line = ''
                else:
                    line = 'else '
                # only need another if statement when we have more than one boolean variables
                if firstline or len(used_boolvars)>1:
                    line += 'if('+(' && '.join(atomics))+')'
                line += '\n    '
                line += var + ' ' + op + ' ' + self.translate_expression(simp_expr) + ';'
                codelines.append(line)
                firstline = False
            code = '\n'.join(codelines)
        else:
            if op == ':=':
                decl = self.c_data_type(statement.dtype) + ' '
                op = '='
                if statement.constant:
                    decl = 'const ' + decl
            else:
                decl = ''
            code = decl + var + ' ' + op + ' ' + self.translate_expression(expr) + ';'
        if len(comment):
            code += ' // ' + comment
        return code
    
    def translate_to_read_arrays(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # index and read arrays (index arrays first)
        for varname in itertools.chain(indices, read):
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            if varname not in write:
                line = 'const '
            else:
                line = ''
            line = line + self.c_data_type(var.dtype) + ' ' + varname + ' = '
            line = line + self.get_array_name(var, self.variables) + '[' + index_var + '];'
            lines.append(line)
        return lines

    def translate_to_declarations(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # simply declare variables that will be written but not read
        for varname in write:
            if varname not in read and varname not in indices:
                var = self.variables[varname]
                line = self.c_data_type(var.dtype) + ' ' + varname + ';'
                lines.append(line)
        return lines

    def translate_to_statements(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # the actual code
        for stmt in statements:
            line = self.translate_statement(stmt)
            if stmt.var in conditional_write_vars:
                subs = {}
                condvar = conditional_write_vars[stmt.var]
                lines.append('if(%s)' % condvar)
                lines.append('    '+line)
            else:
                lines.append(line)
        return lines

    def translate_to_write_arrays(self, statements):
        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        lines = []
        # write arrays
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var, self.variables) + '[' + index_var + '] = ' + varname + ';'
            lines.append(line)
        return lines

    def translate_one_statement_sequence(self, statements, scalar=False):
        # This function is refactored into four functions which perform the
        # four necessary operations. It's done like this so that code
        # deriving from this class can overwrite specific parts.
        lines = []
        # index and read arrays (index arrays first)
        lines += self.translate_to_read_arrays(statements)
        # simply declare variables that will be written but not read
        lines += self.translate_to_declarations(statements)
        # the actual code
        lines += self.translate_to_statements(statements)
        # write arrays
        lines += self.translate_to_write_arrays(statements)
        code = '\n'.join(lines)                
        return stripped_deindented_lines(code)

    def denormals_to_zero_code(self):
        if self.flush_denormals:
            return '''
            #define CSR_FLUSH_TO_ZERO         (1 << 15)
            unsigned csr = __builtin_ia32_stmxcsr();
            csr |= CSR_FLUSH_TO_ZERO;
            __builtin_ia32_ldmxcsr(csr);
            '''
        else:
            return ''

    def _add_user_function(self, varname, variable):
        impl = variable.implementations[self.codeobj_class]
        support_code = []
        hash_defines = []
        pointers = []
        user_functions = [(varname, variable)]
        funccode = impl.get_code(self.owner)
        if isinstance(funccode, basestring):
            funccode = {'support_code': funccode}
        if funccode is not None:
            # To make namespace variables available to functions, we
            # create global variables and assign to them in the main
            # code
            func_namespace = impl.get_namespace(self.owner) or {}
            for ns_key, ns_value in func_namespace.iteritems():
                if hasattr(ns_value, 'dtype'):
                    if ns_value.shape == ():
                        raise NotImplementedError((
                        'Directly replace scalar values in the function '
                        'instead of providing them via the namespace'))
                    type_str = self.c_data_type(ns_value.dtype) + '*'
                else:  # e.g. a function
                    type_str = 'py::object'
                support_code.append('static {0} _namespace{1};'.format(type_str,
                                                                       ns_key))
                pointers.append('_namespace{0} = {1};'.format(ns_key, ns_key))
            support_code.append(deindent(funccode.get('support_code', '')))
            hash_defines.append(deindent(funccode.get('hashdefine_code', '')))

        dep_hash_defines = []
        dep_pointers = []
        dep_support_code = []
        if impl.dependencies is not None:
            for dep_name, dep in impl.dependencies.iteritems():
                self.variables[dep_name] = dep
                hd, ps, sc, uf = self._add_user_function(dep_name, dep)
                dep_hash_defines.extend(hd)
                dep_pointers.extend(ps)
                dep_support_code.extend(sc)
                user_functions.extend(uf)

        return (dep_hash_defines + hash_defines,
                dep_pointers + pointers,
                dep_support_code + support_code,
                user_functions)

    def determine_keywords(self):
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        lines = []
        # it is possible that several different variable names refer to the
        # same array. E.g. in gapjunction code, v_pre and v_post refer to the
        # same array if a group is connected to itself
        handled_pointers = set()
        template_kwds = {}
        # again, do the import here to avoid a circular dependency.
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

        support_code += '\n' + deindent(self.universal_support_code)

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

# The CUDA Math library suports all C99 standard float and double math functions.
# To have support for integral types in device code, we need to wrap these functions
# and promote integral types to float. In host code, we can just use the standard
# math functions directly.

### Functions available in the C99 standard
func_translations = []
# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor']:
    func_translations.append((func, func))

# Functions that exist under a different name in C++
for func, func_cuda in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan')]:
    func_translations.append((func, func_cuda))

for func, func_cuda in func_translations:
    cuda_code = '''
        template <typename T>
        __host__ __device__
        float _brian_{func}(T value)
        {{
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            return {func}f((float)value);
        #else
            return {func}(value);
        #endif
        }}
        inline __host__ __device__
        double _brian_{func}(double value)
        {{
            return {func}(value);
        }}
        '''
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CUDACodeGenerator,
                                                               code=cuda_code.format(func=func_cuda),
                                                               name='_brian_{}'.format(func_cuda)
                                                               )

# std::abs is available and already overloaded for integral types in device code
abs_code = '''
#define _brian_abs std::abs
'''
DEFAULT_FUNCTIONS['abs'].implementations.add_implementation(CUDACodeGenerator,
                                                            code=abs_code,
                                                            name='_brian_abs')

### Functions that need to be implemented specifically
int_code = '''
    template <typename T>
    __host__ __device__
    int _brian_int(T value)
    {
        return (int)value;
    }
    template <>
    inline __host__ __device__
    int _brian_int(bool value)
    {
        return value ? 1 : 0;
    }
    '''

DEFAULT_FUNCTIONS['int'].implementations.add_implementation(CUDACodeGenerator,
                                                            code=int_code,
                                                            name='_brian_int')

clip_code = '''
    template <typename T1, typename T2>
    inline __host__ __device__
    T1 _brian_clip(const T1 value, const T2 a_min, const T2 a_max)
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
                                                             name='_brian_clip')

sign_code = '''
        template <typename T>
        __host__ __device__
        int _brian_sign(T val)
        {
            return (T(0) < val) - (val < T(0));
        }
        '''
DEFAULT_FUNCTIONS['sign'].implementations.add_implementation(CUDACodeGenerator,
                                                             code=sign_code,
                                                             name='_brian_sign')

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
