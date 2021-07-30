import os

from nose import SkipTest, with_setup
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_raises

from brian2 import *
from brian2.core.functions import timestep
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_and_delete
from brian2.tests.utils import assert_allclose
from brian2.codegen.generators import CodeGenerator
from brian2.codegen.codeobject import CodeObject


@attr('cuda-standalone', 'standalone-only')
@with_setup(teardown=reinit_and_delete)
def test_user_defined_function():
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
    assert_equal(np.sin(test_array), mon.func_.flatten())


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_and_delete)
def test_user_function_with_namespace_variable():

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
