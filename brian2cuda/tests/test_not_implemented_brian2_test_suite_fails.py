'''
Tests that fail from the brian2 test suite and need fixing.
TODO: are these tests copied from the brian2 test suite? Check and if so, delete this.
'''
import pytest
from numpy.testing import assert_equal, assert_array_equal

from brian2 import *

import brian2cuda


# cuda version of same test in brian2.tests.test_functions.py
# ERROR: unexpected keyword argument 'header'
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


#TODO: Update BinomialFunction with cuda implementation
@pytest.mark.standalone_compatible
def test_binomial():
    binomial_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    binomial_f = BinomialFunction(100, 0.1, approximate=False)

    # Just check that it does not raise an error and that it produces some
    # values
    G = NeuronGroup(1, '''x : 1
                          y : 1''')
    G.run_regularly('''x = binomial_f_approximated()
                       y = binomial_f()''')
    mon = StateMonitor(G, ['x', 'y'], record=0)
    run(1*ms)
    assert np.var(mon[0].x) > 0
    assert np.var(mon[0].y) > 0


#TODO: Update TimedArray with cuda implementations
@pytest.mark.standalone_compatible
def test_time_dependent_rate():
    # The following two groups should show the same behaviour
    timed_array = TimedArray(np.array([[0, 0],
                                       [1./defaultclock.dt, 0]])*Hz, dt=1*ms)
    group_1 = PoissonGroup(2, rates='timed_array(t, i)')
    group_2 = PoissonGroup(2, rates='int(i==0)*int(t>=1*ms)*(1/dt)')
    spikes_1 = SpikeMonitor(group_1)
    spikes_2 = SpikeMonitor(group_2)
    run(2*ms)

    assert_equal(spikes_1.count,
                 np.array([int(round(1*ms/defaultclock.dt)), 0]))
    assert_equal(spikes_2.count,
                 np.array([int(round(1 * ms / defaultclock.dt)), 0]))
    assert sum(spikes_1.t < 1*ms) == 0
    assert sum(spikes_2.t < 1*ms) == 0


#TODO: needs BinomialFunction
@pytest.mark.standalone_compatible
def test_poissoninput():
    # Test extreme cases and do a very basic test of an intermediate case, we
    # don't want tests to be stochastic
    G = NeuronGroup(10, '''x : volt
                           y : volt
                           y2 : volt
                           z : volt
                           z2 : volt
                           w : 1''')
    G.w = 0.5

    never_update = PoissonInput(G, 'x', 100, 0*Hz, weight=1*volt)
    always_update = PoissonInput(G, 'y', 50, 1/defaultclock.dt, weight=2*volt)
    always_update2 = PoissonInput(G, 'y2', 50, 1/defaultclock.dt, weight='1*volt + 1*volt')
    sometimes_update = PoissonInput(G, 'z', 10000, 50*Hz, weight=0.5*volt)
    sometimes_update2 = PoissonInput(G, 'z2', 10000, 50*Hz, weight='w*volt')

    assert_equal(never_update.rate, 0*Hz)
    assert_equal(never_update.N, 100)
    assert_equal(always_update.rate, 1/defaultclock.dt)
    assert_equal(always_update.N, 50)
    assert_equal(sometimes_update.rate, 50*Hz)
    assert_equal(sometimes_update.N, 10000)

    mon = StateMonitor(G, ['x', 'y', 'y2', 'z', 'z2'], record=True, when='end')

    run(1*ms)
    assert_equal(0, mon.x[:])
    assert_equal(np.tile((1+np.arange(mon.y[:].shape[1]))*50*2*volt, (10, 1)),
                 mon.y[:])
    assert_equal(np.tile((1+np.arange(mon.y[:].shape[1]))*50*2*volt, (10, 1)),
                 mon.y2[:])
    assert all(np.var(mon.z[:], axis=1) > 0)  # variability over time
    assert all(np.var(mon.z[:], axis=0) > 0)  # variability over neurons
    assert all(np.var(mon.z2[:], axis=1) > 0)  # variability over time
    assert all(np.var(mon.z2[:], axis=0) > 0)  # variability over neurons


#TODO Why do we get a NotImplementedError in this test?
# NotImplementedError: Cannot retrieve the values of state variables in standalone code before the simulation has been run.
@pytest.mark.standalone_compatible
def test_synapses_to_synapses():
    source = SpikeGeneratorGroup(3, [0, 1, 2], [0, 0, 0]*ms, period=2*ms)
    modulator = SpikeGeneratorGroup(3, [0, 2], [1, 3]*ms)
    target = NeuronGroup(3, 'v : integer')
    conn = Synapses(source, target, 'w : integer', on_pre='v += w')
    conn.connect(j='i')
    conn.w = 1
    modulatory_conn = Synapses(modulator, conn, on_pre='w += 1')
    modulatory_conn.connect(j='i')
    run(5*ms)
    # First group has its weight increased to 2 after the first spike
    # Third group has its weight increased to 2 after the second spike
    assert_array_equal(target.v, [5, 3, 4])


#TODO Why do we get a NotImplementedError in this test?
# NotImplementedError: Cannot retrieve the values of state variables in standalone code before the simulation has been run.
@pytest.mark.standalone_compatible
def test_synapses_to_synapses_different_sizes():
    prefs.codegen.target = 'numpy'
    source = NeuronGroup(100, 'v : 1', threshold='False')
    source.v = 'i'
    modulator = NeuronGroup(1, 'v : 1', threshold='False')
    target = NeuronGroup(100, 'v : 1')
    target.v = 'i + 100'
    conn = Synapses(source, target, 'w:1', multisynaptic_index='k')
    conn.connect(j='i', n=2)
    conn.w = 'i + j'
    modulatory_conn = Synapses(modulator, conn)
    modulatory_conn.connect('k_post == 1')  # only second synapse is targeted
    run(0*ms)
    assert_equal(modulatory_conn.w_post, 2*np.arange(100))


#TODO Why do we get a NotImplementedError in this test?
# NotImplementedError: Cannot retrieve the values of state variables in standalone code before the simulation has been run.
@pytest.mark.standalone_compatible
def test_synapses_to_synapses_summed_variable():
    source = NeuronGroup(5, '', threshold='False')
    target = NeuronGroup(5, '')
    conn = Synapses(source, target, 'w : integer')
    conn.connect(j='i')
    conn.w = 1
    summed_conn = Synapses(source, conn, '''w_post = x : integer (summed)
                                            x : integer''')
    summed_conn.connect('i>=j')
    summed_conn.x = 'i'
    run(defaultclock.dt)
    assert_array_equal(conn.w[:], [10, 10, 9, 7, 4])


##########################
###  TimedArray tests  ###
##########################

#TODO: Update TimedArray with cuda implementations

@pytest.mark.standalone_compatible
def test_timedarray_semantics():
    # Make sure that timed arrays are interpreted as specifying the values
    # between t and t+dt (not between t-dt/2 and t+dt/2 as in Brian1)
    ta = TimedArray(array([0, 1]), dt=0.4*ms)
    G = NeuronGroup(1, 'value = ta(t) : 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=0)
    run(0.8*ms)
    assert_equal(mon[0].value, [0, 0, 0, 0, 1, 1, 1, 1])
    assert_equal(mon[0].value, ta(mon.t))


@pytest.mark.standalone_compatible
def test_timedarray_no_units():
    ta = TimedArray(np.arange(10), dt=0.1*ms)
    G = NeuronGroup(1, 'value = ta(t) + 1: 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    run(1.1*ms)
    assert_equal(mon[0].value_, np.clip(np.arange(len(mon[0].t)), 0, 9) + 1)


@pytest.mark.standalone_compatible
def test_timedarray_with_units():
    ta = TimedArray(np.arange(10)*amp, dt=0.1*ms)
    G = NeuronGroup(1, 'value = ta(t) + 2*nA: amp', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    run(1.1*ms)
    assert_equal(mon[0].value, np.clip(np.arange(len(mon[0].t)), 0, 9)*amp + 2*nA)


@pytest.mark.standalone_compatible
def test_timedarray_2d():
    # 4 time steps, 3 neurons
    ta2d = TimedArray(np.arange(12).reshape(4, 3), dt=0.1*ms)
    G = NeuronGroup(3, 'value = ta2d(t, i) + 1: 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=0.1*ms)
    run(0.5*ms)
    assert_equal(mon[0].value_, np.array([0, 3, 6, 9, 9]) + 1)
    assert_equal(mon[1].value_, np.array([1, 4, 7, 10, 10]) + 1)
    assert_equal(mon[2].value_, np.array([2, 5, 8, 11, 11]) + 1)


@pytest.mark.standalone_compatible
def test_timedarray_no_upsampling():
    # Test a TimedArray where no upsampling is necessary because the monitor's
    # dt is bigger than the TimedArray's
    ta = TimedArray(np.arange(10), dt=0.01*ms)
    G = NeuronGroup(1, 'value = ta(t): 1', dt=0.1*ms)
    mon = StateMonitor(G, 'value', record=True, dt=1*ms)
    run(2.1*ms)
    assert_equal(mon[0].value, [0, 9, 9])

if __name__ == '__main__':
    test_user_defined_function()
    test_binomial()
    test_time_dependent_rate()
    test_poissoninput()
    test_synapses_to_synapses()
    test_synapses_to_synapses_different_sizes()
    test_synapses_to_synapses_summed_variable()
    test_timedarray_semantics()
    test_timedarray_no_units()
    test_timedarray_with_units()
    test_timedarray_2d()
    test_timedarray_no_upsampling()
