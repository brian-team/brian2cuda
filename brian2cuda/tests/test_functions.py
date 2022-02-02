import os

import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.core.functions import timestep
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.utils.logger import catch_logs
from brian2.tests.utils import assert_allclose
from brian2.codegen.generators import CodeGenerator
from brian2.codegen.codeobject import CodeObject


# TODO: Remove this and import from brian2.tests.utils once we track brian2 version
# with merged PR brian2#1315
def exc_isinstance(exc_info, expected_exception, raise_not_implemented=False):
    # XXX: This will fails once the function is importable from brian2
    try:
        from brian2.tests.utils import exc_isinstance as _
    except ImportError:
        pass
    else:
        raise AssertionError("Remove this function and import from brian2")

    if exc_info is None:
        return False
    if hasattr(exc_info, 'value'):
        exc_info = exc_info.value

    if isinstance(exc_info, expected_exception):
        return True
    elif raise_not_implemented and isinstance(exc_info, NotImplementedError):
        raise exc_info

    return exc_isinstance(exc_info.__cause__, expected_exception,
                          raise_not_implemented=raise_not_implemented)

@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_user_defined_function():
    set_device('cuda_standalone', directory=None)

    @implementation('cuda',"""
    __host__ __device__ inline double usersin(double x)
    {
        return sin(x);
    }
    """)
    # version of this test with cython and cpp implementation is in brian2.tests
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    default_dt = defaultclock.dt
    test_array = np.array([0, 1, 2, 3])
    G = NeuronGroup(len(test_array),
                    '''func = usersin(variable) : 1
                              variable : 1''')
    G.variable = test_array
    mon = StateMonitor(G, 'func', record=True)
    run(default_dt)
    assert_allclose(np.sin(test_array), mon.func_.flatten())


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_user_function_with_namespace_variable():
    set_device('cuda_standalone', directory=None)

    my_var = 0.1 * np.array(np.arange(5))

    @implementation('cuda', '''
    __host__ __device__ float foo(int i)
    {
        return _namespacemy_var[i];
    }''', namespace={'my_var': my_var})
    @check_units(i=1, result=1)
    def foo(i):
        return my_var[i]


    G = NeuronGroup(5, 'v : 1')
    G.run_regularly('v = foo(i)')
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], my_var)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_manual_user_defined_function_cuda_standalone_compiler_args():
    set_device('cuda_standalone', directory=None)

    @implementation('cuda', '''
    __host__ __device__ static inline double foo(const double x, const double y)
    {
        return x + y + _THREE;
    }''',  # just check whether we can specify the supported compiler args,
           # only the define macro is actually used
        headers=[], sources=[], libraries=[], include_dirs=[],
        library_dirs=[], runtime_library_dirs=[],
        define_macros=[('_THREE', '3')])
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3*volt

    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    G.x = 1*volt
    G.y = 2*volt
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)
    assert mon[0].func == [6] * volt


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_manual_user_defined_function_cuda_standalone_wrong_compiler_args1():
    set_device('cuda_standalone', directory=None)

    @implementation('cuda', '''
    static inline double foo(const double x, const double y)
    {
        return x + y + _THREE;
    }''',  some_arg=[])  # non-existing argument
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3*volt

    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    with pytest.raises(BrianObjectException) as exc:
        net.run(defaultclock.dt, namespace={'foo': foo})
    assert exc_isinstance(exc, ValueError)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_manual_user_defined_function_cuda_standalone_wrong_compiler_args2():
    set_device('cuda_standalone', directory=None)

    @implementation('cuda', '''
    static inline double foo(const double x, const double y)
    {
        return x + y + _THREE;
    }''',  headers='<stdio.h>')  # existing argument, wrong value type
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3*volt

    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    with pytest.raises(BrianObjectException) as exc:
        net.run(defaultclock.dt, namespace={'foo': foo})
    assert exc_isinstance(exc, TypeError)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_external_function_cuda_standalone():
    set_device('cuda_standalone', directory=None)
    this_dir = os.path.abspath(os.path.dirname(__file__))
    @implementation('cuda', '//all code in func_def_cuda.cu',
                    headers=['"func_def_cuda.h"'],
                    include_dirs=[this_dir],
                    sources=[os.path.join(this_dir, 'func_def_cuda.cu')])
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3*volt

    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    G.x = 1*volt
    G.y = 2*volt
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)
    assert mon[0].func == [6] * volt
