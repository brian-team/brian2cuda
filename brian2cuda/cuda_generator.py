import itertools

import numpy as np
import re

from brian2.utils.stringtools import (deindent, stripped_deindented_lines,
                                      word_substitute)
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.core.clocks import Clock
from brian2.core.preferences import prefs, BrianPreference, PreferenceError
from brian2.core.variables import ArrayVariable
from brian2.core.core_preferences import default_float_dtype_validator, dtype_repr
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.codegen.generators.base import CodeGenerator
from brian2.devices import get_device
from brian2cuda.utils.gputools import select_gpu, get_cuda_runtime_version


__all__ = ['CUDACodeGenerator',
           'CUDAAtomicsCodeGenerator'
           'c_data_type'
           ]


logger = get_logger('brian2.codegen.generators.cuda_generator')


class ParallelisationError(Exception):
    pass


def compute_capability_validator(cc):
    """
    This function checks compute capability preference and raises an error if it is
    larger than the minimal supported compute capability.
    """
    if cc is not None:
        pref_name = "prefs.codegen.generators.cuda.compute_capability"
        if not isinstance(cc, float):
            raise PreferenceError(
                "Preference `{}` has to be of type float "
                "(e.g. `6.1`), got `{}`".format(pref_name, type(cc))
            )

    return True


# Preferences
prefs.register_preferences(
    'codegen.generators.cuda',
    'CUDA codegen preferences',

    default_functions_integral_convertion=BrianPreference(
        docs='''The floating point precision to which integral types will be converted when
        passed as arguments to default functions that have no integral type overload in device
        code (sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, ceil, floor, arcsin, arccos, arctan)."
        NOTE: Convertion from 32bit and 64bit integral types to single precision (32bit) floating-point
        types is not type safe. And convertion from 64bit integral types to double precision (64bit)
        floating-point types neither. In those cases the closest higher or lower (implementation
        defined) representable value will be selected.''',
        validator=default_float_dtype_validator,
        representor=dtype_repr,
        default=np.float64),

    use_atomics=BrianPreference(
        docs='''Weather to try to use atomic operations for synaptic effect
        application. Since this avoids race conditions, effect application can
        be parallelised.''',
        validator=lambda v: isinstance(v, bool),
        default=True),

    compute_capability=BrianPreference(
        docs='''Manually set the compute capability for which CUDA code will be
        compiled. Has to be a float (e.g. `6.1`) or None. If None, compute capability is
        chosen depending on GPU in use. ''',
        validator=compute_capability_validator,
        default=None),
)


# This is a function since this code needs to be generated after user preferences are
# set (for GPU selection)
def _generate_atomic_support_code():
    #            (arg_dtype, int_dtype, name in type cast function)
    overloads = [('int', 'int', 'int'),
                 ('float', 'int', 'int'),
                 ('double', 'unsigned long long int', 'longlong')]

    cuda_runtime_version = get_cuda_runtime_version()

    # Note: There are atomic functions that are supported only for compute capability >=
    # 3.5. We don't check for those as we require at least 3.5. If we ever support
    # smaller CC, we need to adapt the code here (see Atomic Functions in CUDA
    # Programming Guide)
    device = get_device()
    assert device.minimal_compute_capability >= 3.5, "Need to adapt atomic support code"

    software_implementation_float_dtype = '''
        // software implementation
        {int_dtype}* address_as_int = ({int_dtype}*)address;
        {int_dtype} old = *address_as_int, assumed;

        do {{
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                            __{arg_dtype}_as_{val_type_cast}(val {op}
                                   __{val_type_cast}_as_{arg_dtype}(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        }} while (assumed != old);

        return __{val_type_cast}_as_{arg_dtype}(old);
        '''

    hardware_implementation = '''
        // hardware implementation
        return atomic{op_name}(address, val);
        '''

    atomic_support_code = ''
    for op_name, op in [('Add', '+'), ('Mul', '*'), ('Div', '/')]:
        for arg_dtype, int_dtype, val_type_cast in overloads:
            # atomicAdd had hardware implementations, depending on dtype, compute
            # capability and CUDA runtime version. This section uses them when possible.
            if op_name == 'Add':
                format_sw = True
                code = '''
                inline __device__ {arg_dtype} _brian_atomic{op_name}({arg_dtype}* address, {arg_dtype} val)
                {{
                '''
                if arg_dtype in ['int', 'float']:
                    code += '''
                    {hardware_implementation}
                    '''.format(hardware_implementation=hardware_implementation)
                elif arg_dtype == 'double':
                    if cuda_runtime_version >= 8.0:
                        # Check for CC in at runtime to use software or hardware
                        # implementation.
                        # Don't need to check for defined __CUDA__ARCH__, it is always defined
                        # in __device__ and __global__ functions (no host code path).
                        code += '''
                        #if (__CUDA_ARCH__ >= 600)
                        {hardware_implementation}
                        #else
                        {software_implementation}
                        #endif
                        '''.format(
                            hardware_implementation=hardware_implementation,
                            software_implementation=software_implementation_float_dtype
                        )
                    else:
                        # For runtime < 8.0, there is no atomicAdd hardware implementation
                        # support independent of CC.
                        code += '''
                        {software_implementation}
                        '''.format(
                            software_implementation=software_implementation_float_dtype
                        )
                        format_sw = True
                code += '''
                }}
                '''
                if format_sw:
                    code = code.format(
                        arg_dtype=arg_dtype, int_dtype=int_dtype,
                        val_type_cast=val_type_cast, op_name=op_name, op=op
                    )
                else:
                    code = code.format(arg_dtype=arg_dtype, op_name=op_name)

                atomic_support_code += code
            # For other atomic operations on int types, we can use the same template.
            elif arg_dtype == 'int':
                atomic_support_code += '''
                inline __device__ int _brian_atomic{op_name}(int* address, int val)
                {{
                    // software implementation
                    int old = *address, assumed;

                    do {{
                        assumed = old;
                        old = atomicCAS(address, assumed, val {op} assumed);
                    }} while (assumed != old);

                    return old;
                }}
                '''.format(op_name=op_name, op=op)
            # Else (not atomicAdd and not int types) use this template.
            else:
                # in the software implementation, we treat all data as integer types
                # (since atomicCAS is only define
                # and use atomicCAS to swap the memory with our our desired value
                code = '''
                inline __device__ {{arg_dtype}} _brian_atomic{{op_name}}({{arg_dtype}}* address, {{arg_dtype}} val)
                {{{{
                    {software_implementation}
                }}}}
                '''.format(software_implementation=software_implementation_float_dtype)
                # above, just add the software_implementation code, below add the other
                # format variables (also present in software_implementation)
                code = code.format(
                    arg_dtype=arg_dtype, int_dtype=int_dtype,
                    val_type_cast=val_type_cast, op_name=op_name, op=op,
                )
                atomic_support_code += code

    return atomic_support_code


# CUDA does not support modulo arithmetics for long double. Since we can't give a warning, we let the
# compilation fail, which gives an error message of type
# error: more than one instance of overloaded function "_brian_mod" matches the argument list: ...
# TODO: can we produce a more informative error message?
_typestrs = ['int', 'long', 'long long', 'float', 'double']#, 'long double']
_hightype_support_code = 'template < typename T1, typename T2 > struct _higher_type;\n'
for ix, xtype in enumerate(_typestrs):
    for iy, ytype in enumerate(_typestrs):
        hightype = _typestrs[max(ix, iy)]
        _hightype_support_code += '''
template < > struct _higher_type<{xtype},{ytype}> {{ typedef {hightype} type; }};
        '''.format(hightype=hightype, xtype=xtype, ytype=ytype)

_mod_support_code = '''
template < typename T1, typename T2 >
__host__ __device__ static inline typename _higher_type<T1,T2>::type
_brian_mod(T1 x, T2 y)
{{
    return x-y*floor(1.0*x/y);
}}
'''

_floordiv_support_code = '''
template < typename T1, typename T2 >
__host__ __device__ static inline typename _higher_type<T1,T2>::type
_brian_floordiv(T1 x, T2 y)
{{
    return floor(1.0*x/y);
}}
'''

_pow_support_code = '''
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

    _use_atomics = False

    # This will be generated when the first class instance calls `determine_keywords()`
    universal_support_code = None

    def __init__(self, *args, **kwds):
        super(CUDACodeGenerator, self).__init__(*args, **kwds)
        self.c_data_type = c_data_type
        self.warned_integral_convertion = False
        self.previous_convertion_pref = None
        self.uses_atomics = False

        # Set convertion types for standard C99 functions in device code
        # These are used in _add_user_function to format the function code
        if prefs.codegen.generators.cuda.default_functions_integral_convertion == np.float64:
            self.default_func_type = 'double'
            self.other_func_type = 'float'
        else:  # np.float32
            self.default_func_type = 'float'
            self.other_func_type = 'double'
        # set clip function to either use all float or all double arguments
        # see #51 for details
        if prefs['core.default_float_dtype'] == np.float64:
            self.float_dtype = 'float'
        else:  # np.float32
            self.float_dtype = 'double'


    @property
    def restrict(self):
        return prefs['codegen.generators.cpp.restrict_keyword'] + ' '

    @property
    def flush_denormals(self):
        return prefs['codegen.generators.cpp.flush_denormals']

    @staticmethod
    def get_array_name(var, access_data=True, prefix=None):
        '''
        Return a globally unique name for `var`.
        See CUDAStandaloneDevice.get_array_name for parameters.

        Here, `prefix` defaults to `'_ptr'` when `access_data=True`.

        `prefix='_ptr'` is used since the CUDACodeGenerator generates the
        `scalar_code` and `vector_code` snippets.
        '''
        # We have to do the import here to avoid circular import dependencies.
        from brian2.devices.device import get_device
        device = get_device()
        if access_data:
            if prefix is None:
                prefix = '_ptr'
            return device.get_array_name(var, access_data=True, prefix=prefix)
        else:
            return device.get_array_name(var, access_data=False, prefix=prefix)

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

    def translate_to_read_arrays(self, read, write, indices):
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
            line = line + self.get_array_name(var) + '[' + index_var + '];'
            lines.append(line)
        return lines

    def translate_to_declarations(self, read, write, indices):
        lines = []
        # simply declare variables that will be written but not read
        for varname in write:
            if varname not in read and varname not in indices:
                var = self.variables[varname]
                line = self.c_data_type(var.dtype) + ' ' + varname + ';'
                lines.append(line)
        return lines

    def conditional_write(self, line, statement, conditional_write_vars):
        lines = []
        if statement.var in conditional_write_vars:
            subs = {}
            condvar = conditional_write_vars[statement.var]
            lines.append('if(%s)' % condvar)
            lines.append('    ' + line)
        else:
            lines.append(line)
        return lines

    def translate_to_statements(self, statements, conditional_write_vars):
        lines = []
        # the actual code
        for stmt in statements:
            line = self.translate_statement(stmt)
            lines.extend(self.conditional_write(line, stmt,
                                                conditional_write_vars))
        return lines

    def translate_to_write_arrays(self, write):
        lines = []
        # write arrays
        for varname in write:
            index_var = self.variable_indices[varname]
            var = self.variables[varname]
            line = self.get_array_name(var) + '[' + index_var + '] = ' + varname + ';'
            lines.append(line)
        return lines

    def translate_one_statement_sequence(self, statements, scalar=False):
        # This function is refactored into four functions which perform the
        # four necessary operations. It's done like this so that code
        # deriving from this class can overwrite specific parts.
        all_unique = not self.has_repeated_indices(statements)

        read, write, indices, conditional_write_vars = self.arrays_helper(statements)
        try:
            # try to use atomics
            if not self._use_atomics or scalar or all_unique:
                raise ParallelisationError()
            # more complex translations to deal with repeated indices, which
            # could lead to race conditions when applied in parallel
            lines = self.parallelise_code(statements)
            self.uses_atomics = True
        except ParallelisationError:
            # don't use atomics
            lines = []
            # index and read arrays (index arrays first)
            lines += self.translate_to_read_arrays(read, write, indices)
            # simply declare variables that will be written but not read
            lines += self.translate_to_declarations(read, write, indices)
            # the actual code
            lines += self.translate_to_statements(statements,
                                                  conditional_write_vars)
            # write arrays
            lines += self.translate_to_write_arrays(write)
        code = '\n'.join(lines)

        # Check if 64bit integer types occur in the same line as a default function.
        # We can't get the arguments of the function call directly with regex due to
        # possibly nested paranthesis inside function paranthesis.
        statement_lines = self.translate_to_statements(statements,
                                                       conditional_write_vars)
        convertion_pref = prefs.codegen.generators.cuda.default_functions_integral_convertion
        # only check if there was no warning yet or if convertion preference has changed
        if not self.warned_integral_convertion or self.previous_convertion_pref != convertion_pref:
            for line in statement_lines:
                brian_funcs = re.search('_brian_(' + '|'.join(functions_C99) + ')', line)
                if brian_funcs is not None:
                    for identifier in get_identifiers(line):
                        if convertion_pref == np.float64:
                            # 64bit integer to floating-point conversions are not type safe
                            int64_type = re.search(r'\bu?int64_t\s*{}\b'.format(identifier), code)
                            if int64_type is not None:
                                logger.warn("Detected code statement with default function and 64bit integer type in the same line. "
                                            "Using 64bit integer types as default function arguments is not type safe due to convertion of "
                                            "integer to 64bit floating-point types in device code. (relevant functions: sin, cos, tan, sinh, "
                                            "cosh, tanh, exp, log, log10, sqrt, ceil, floor, arcsin, arccos, arctan)\nDetected code "
                                            "statement:\n\t{}\nGenerated from abstract code statements:\n\t{}\n".format(line, statements),
                                            once=True)
                                self.warned_integral_convertion = True
                                self.previous_convertion_pref = np.float64
                        else:  # convertion_pref = np.float32
                            # 32bit and 64bit integer to floating-point conversions are not type safe
                            int32_64_type = re.search(r'\bu?int(32|64)_t\s*{}\b'.format(identifier), code)
                            if int32_64_type is not None:
                                logger.warn("Detected code statement with default function and 32bit or 64bit integer type in the same line and the "
                                            "preference for default_functions_integral_convertion is 'float32'. "
                                            "Using 32bit or 64bit integer types as default function arguments is not type safe due to convertion of "
                                            "integer to single-precision floating-point types in device code. (relevant functions: sin, cos, tan, sinh, "
                                            "cosh, tanh, exp, log, log10, sqrt, ceil, floor, arcsin, arccos, arctan)\nDetected code "
                                            "statement:\n\t{}\nGenerated from abstract code statements:\n\t{}\n".format(line, statements),
                                            once=True)
                                self.warned_integral_convertion = True
                                self.previous_convertion_pref = np.float32
        return stripped_deindented_lines(code)

    def atomics_parallelisation(self, statement, conditional_write_vars,
                                used_variables):
        if not self._use_atomics:
            raise ParallelisationError()
        # Avoids circular import
        from brian2.devices.device import device

        # See https://github.com/brian-team/brian2/pull/531 for explanation
        used = set(get_identifiers(statement.expr))
        used = used.intersection(k for k in self.variables.keys()
                                 if k in self.variable_indices
                                 and self.variable_indices[k] != '_idx')
        used_variables.update(used)
        if statement.var in used_variables:
            raise ParallelisationError()

        expr = self.translate_expression(statement.expr)

        if statement.op == ':=' or self.variable_indices[statement.var] == '_idx' or not statement.inplace:
            if statement.op == ':=':
                decl = self.c_data_type(statement.dtype) + ' '
                op = '='
                if statement.constant:
                    decl = 'const ' + decl
            else:
                decl = ''
                op = statement.op
            line = '{decl}{var} {op} {expr};'.format(decl=decl,
                                                     var=statement.var, op=op,
                                                     expr=expr)
            line = [line]
        elif statement.inplace:
            sign = ''
            if statement.op == '+=':
                atomic_op = '_brian_atomicAdd'
            elif statement.op == '-=':
                # CUDA has hardware implementations for float (and for CC>=6.0
                # for double) only for atomicAdd, which is faster then our
                # software implementation
                atomic_op = '_brian_atomicAdd'
                sign = '-'
            elif statement.op == '*=':
                atomic_op = '_brian_atomicMul'
            elif statement.op == '/=':
                atomic_op = '_brian_atomicDiv'
            else:
                # TODO: what other inplace operations are possible? Can we
                # implement them with atomicCAS ?
                logger.info("Atomic operation for operation {op} is not implemented."
                            "".format(op=statement.op))
                raise ParallelisationError()

            line = '{atomic_op}(&{array_name}[{idx}], ({array_dtype}){sign}({expr}));'.format(
                atomic_op=atomic_op,
                array_name=self.get_array_name(self.variables[statement.var]),
                idx=self.variable_indices[statement.var],
                array_dtype=c_data_type(self.variables[statement.var].dtype),
                sign=sign, expr=expr)
            # this is now a list of 1 or 2 lines (potentially with if(...))
            line = self.conditional_write(line, statement, conditional_write_vars)
        else:
            raise ParallelisationError()

        if len(statement.comment):
            line[-1] += ' // ' + statement.comment

        return line

    def parallelise_code(self, statements):
        try:
            used_variables = set()
            all_read, all_write, all_indices, _ = self.arrays_helper(statements)
            lines = []
            # we are collecting all reads, which are not only loaded because
            # of in-place atomic operations and all writes, which are not
            # written to by atomics
            collected_reads = set()
            collected_writes = set()
            atomic_lines = []
            for stmt in statements:
                lines.append('//  Abstract code:  {var} {op} {expr}'.format(var=stmt.var,
                                                                            op=stmt.op,
                                                                            expr=stmt.expr))
                # We treat every statement individually with its own read and write code
                # to be on the safe side
                read, write, indices, conditional_write_vars = self.arrays_helper([stmt])
                # No need to load a variable if it is only in read because of
                # the in-place operation, but still load the index var, we
                # need it for the atomics
                if (stmt.inplace and
                        self.variable_indices[stmt.var] != '_idx' and
                        stmt.var not in get_identifiers(stmt.expr)):
                    read = read - {stmt.var}
                collected_reads = collected_reads.union(read)
                atomic_lines.extend(self.atomics_parallelisation(stmt,
                                                                 conditional_write_vars,
                                                                 used_variables))
                # Do not write back such values, the atomic functions have
                # modified the underlying array already
                if stmt.inplace and self.variable_indices[stmt.var] != '_idx':
                    write = write - {stmt.var}
                collected_writes = collected_writes.union(write)

            # we translate to read arrays altogether, otherwise we end up with
            # multiple declarations for statement identifiers

            # pass all_write here, since this determines the `const` keyword
            # (otherwise the variables written to by atomics would be `const`)
            # and all_indices, since we need those for the atomic calls
            lines.extend(self.translate_to_read_arrays(collected_reads,
                                                       all_write,
                                                       all_indices))
            # we need to declare variables which will be written to but not
            # read (e.g. in assigmenets like `v_post = 1`) and which are not
            # used in atomics (hence we use `collected_writes`)
            lines.extend(self.translate_to_declarations(all_read,
                                                        collected_writes,
                                                        all_indices))
            # add the atomic operations and statements
            lines.extend(atomic_lines)
            # only write variables which are not written to by atomics
            lines.extend(self.translate_to_write_arrays(collected_writes))
        except ParallelisationError:
            # logg info here, since this means we tried, but failed to use atomics
            logger.info("Failed to parallelise code by using atomic operations. "
                        "Falling back to serialized effect application. This "
                        "might be slow. Switching to `cpp_standalone` might be "
                        "faster. Code object name is {} and first line in "
                        "abstract code is: {}".format(self.name, statements[0]),
                        once=True)
            raise

        return lines


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
        kernel_lines = []
        user_functions = [(varname, variable)]
        funccode = impl.get_code(self.owner)

        ### Different from CPPCodeGenerator: We format the funccode dtypes here
        from brian2.devices.device import get_device
        device = get_device()
        if varname in functions_C99:
            funccode = funccode.format(default_type=self.default_func_type,
                                       other_type=self.other_func_type)
        if varname == 'clip':
            funccode = funccode.format(float_dtype=self.float_dtype)
        ###

        if isinstance(funccode, basestring):
            funccode = {'support_code': funccode}
        if funccode is not None:
            # To make namespace variables available to functions, we
            # create global variables and assign to them in the main
            # code
            func_namespace = impl.get_namespace(self.owner) or {}
            for ns_key, ns_value in func_namespace.iteritems():
                # This section is adapted from CPPCodeGenerator such that file
                # global namespace pointers can be used in both host and device
                # code.
                assert hasattr(ns_value, 'dtype'), \
                    'This should not have happened. Please report at ' \
                    'https://github.com/brian-team/brian2cuda/issues/new'
                if ns_value.shape == ():
                    raise NotImplementedError((
                    'Directly replace scalar values in the function '
                    'instead of providing them via the namespace'))
                type_str = self.c_data_type(ns_value.dtype) + '*'
                namespace_ptr = '''
                    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                    __device__ {dtype} _namespace{name};
                    #else
                    {dtype} _namespace{name};
                    #endif
                    '''.format(dtype=type_str, name=ns_key)
                support_code.append(namespace_ptr)
                # pointer lines will be used in codeobjects running on the host
                pointers.append('_namespace{name} = {name};'.format(name=ns_key))
                # kernel lines will be used in codeobjects running on the device
                kernel_lines.append('''
                    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
                    _namespace{name} = d{name};
                    #else
                    _namespace{name} = {name};
                    #endif
                    '''.format(name=ns_key))
            support_code.append(deindent(funccode.get('support_code', '')))
            hash_defines.append(deindent(funccode.get('hashdefine_code', '')))

        dep_hash_defines = []
        dep_pointers = []
        dep_support_code = []
        dep_kernel_lines = []
        if impl.dependencies is not None:
            for dep_name, dep in impl.dependencies.iteritems():
                if dep_name not in self.variables:
                    self.variables[dep_name] = dep
                    hd, ps, sc, uf, kl = self._add_user_function(dep_name, dep)
                    dep_hash_defines.extend(hd)
                    dep_pointers.extend(ps)
                    dep_support_code.extend(sc)
                    user_functions.extend(uf)
                    dep_kernel_lines.extend(kl)

        return (dep_hash_defines + hash_defines,
                dep_pointers + pointers,
                dep_support_code + support_code,
                user_functions,
                dep_kernel_lines + kernel_lines)

    def determine_keywords(self):
        # set up the restricted pointers, these are used so that the compiler
        # knows there is no aliasing in the pointers, for optimisation
        pointers = []
        # Add additional lines inside the kernel functions
        kernel_lines = []
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
                if getattr(var, 'ndim', 1) > 1:
                    continue  # multidimensional (dynamic) arrays have to be treated differently
                restrict = self.restrict
                # turn off restricted pointers for scalars for safety
                if var.scalar:
                    restrict = ' '
                line = '{0}* {1} {2} = {3};'.format(self.c_data_type(var.dtype),
                                                    restrict,
                                                    pointer_name,
                                                    array_name)
                pointers.append(line)
                handled_pointers.add(pointer_name)

        # set up the functions
        user_functions = []
        support_code = []
        hash_defines = []
        for varname, variable in self.variables.items():
            if isinstance(variable, Function):
                hd, ps, sc, uf, kl = self._add_user_function(varname, variable)
                user_functions.extend(uf)
                support_code.extend(sc)
                pointers.extend(ps)
                hash_defines.extend(hd)
                kernel_lines.extend(kl)

        # Generate universal_support_code once when the first codeobject is created.
        # Can't do it at import time since need access to user preferences
        # This is a class attribute (not instance attribute).
        if CUDACodeGenerator.universal_support_code is None:
            _atomic_support_code = _generate_atomic_support_code()
            CUDACodeGenerator.universal_support_code = (
                _hightype_support_code
                + _mod_support_code
                + _floordiv_support_code
                + _pow_support_code
                + _atomic_support_code
            )
        support_code.append(CUDACodeGenerator.universal_support_code)

        # Clock variables (t, dt, timestep) are passed by value to kernels and
        # need to be translated back into pointers for scalar/vector code.
        for varname, variable in self.variables.iteritems():
            if hasattr(variable, 'owner') and isinstance(variable.owner, Clock):
                # get arrayname without _ptr suffix (e.g. _array_defaultclock_dt)
                arrayname = self.get_array_name(variable, prefix='')
                line = "const {dtype}* _ptr{arrayname} = &_value{arrayname};"
                line = line.format(dtype=c_data_type(variable.dtype), arrayname=arrayname)
                if line not in kernel_lines:
                    kernel_lines.append(line)

        keywords = {'pointers_lines': stripped_deindented_lines('\n'.join(pointers)),
                    'support_code_lines': stripped_deindented_lines('\n'.join(support_code)),
                    'hashdefine_lines': stripped_deindented_lines('\n'.join(hash_defines)),
                    'denormals_code_lines': stripped_deindented_lines('\n'.join(self.denormals_to_zero_code())),
                    'kernel_lines': stripped_deindented_lines('\n'.join(kernel_lines)),
                    'uses_atomics': self.uses_atomics
                    }
        keywords.update(template_kwds)
        return keywords


class CUDAAtomicsCodeGenerator(CUDACodeGenerator):

    _use_atomics = True

################################################################################
# Implement functions
################################################################################

# The CUDA Math library suports all C99 standard float and double math functions.
# To have support for integral types in device code, we need to wrap these functions
# and convert integral types to floating-point types. In host code, we can just use
# the standard math functions directly.

### Functions available in the C99 standard
functions_C99 = []
func_translations = []
# Functions that exist under the same name in C++
for func in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
             'log10', 'sqrt', 'ceil', 'floor']:
    func_translations.append((func, func))

# Functions that exist under a different name in C++
for func, func_cuda in [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan')]:
    func_translations.append((func, func_cuda))

for func, func_cuda in func_translations:
    functions_C99.append(func)
    cuda_code = '''
        template <typename T>
        __host__ __device__
        {{default_type}} _brian_{func}(T value)
        {{{{
        #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
            return {func}(({{default_type}})value);
        #else
            return {func}(value);
        #endif
        }}}}
        inline __host__ __device__
        {{other_type}} _brian_{func}({{other_type}} value)
        {{{{
            return {func}(value);
        }}}}
        '''.format(func=func_cuda)
        # {default_type} and {other_type} will be formatted in CUDACodeGenerator.determine_keywords()
        # depending on user prefs (which are not yet set when this code snippet is created)
    DEFAULT_FUNCTIONS[func].implementations.add_implementation(CUDACodeGenerator,
                                                               code=cuda_code,
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

# TODO: make float version type safe or print warning (see #51)
clip_code = '''
    inline __host__ __device__
    {float_dtype} _brian_clip(const {float_dtype} value,
                              const {float_dtype} a_min,
                              const {float_dtype} a_max)
    {{
        if (value < a_min)
            return a_min;
        if (value > a_max)
            return a_max;
        return value;
    }}
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
    #define _randn(vectorisation_idx) (_ptr_array_%CODEOBJ_NAME%_randn[vectorisation_idx])
        '''
DEFAULT_FUNCTIONS['randn'].implementations.add_implementation(CUDACodeGenerator,
                                                              code=randn_code,
                                                              name='_randn')

rand_code = '''
    #define _rand(vectorisation_idx) (_ptr_array_%CODEOBJ_NAME%_rand[vectorisation_idx])
    '''
DEFAULT_FUNCTIONS['rand'].implementations.add_implementation(CUDACodeGenerator,
                                                             code=rand_code,
                                                             name='_rand')


# Add support for the `timestep` function added in Brian 2.3.1
timestep_code = '''
    __host__ __device__
    static inline int64_t _timestep(double t, double dt)
    {
        return (int64_t)((t + 1e-3*dt)/dt);
    }
    '''
DEFAULT_FUNCTIONS['timestep'].implementations.add_implementation(CUDACodeGenerator,
                                                                 code=timestep_code,
                                                                 name='_timestep')
