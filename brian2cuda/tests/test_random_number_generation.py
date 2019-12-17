from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_raises, assert_allclose

from brian2 import *
from brian2.monitors.statemonitor import StateMonitor
from brian2.core.clocks import defaultclock
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices, set_device, device

import brian2cuda


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rand():

    set_device('cuda_standalone', directory=None)
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(3*defaultclock.dt)

    print(mon.v[:5, :])
    assert_raises(AssertionError, assert_equal, mon.v[:, -1], 0)
    assert_raises(AssertionError, assert_equal, mon.v[:, -2], mon.v[:, -1])


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_number_generation_with_multiple_runs():

    set_device('cuda_standalone', directory=None, build_on_run=False)
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    run(2*defaultclock.dt)
    device.build(direct_call=False, **device.build_options)

    assert_raises(AssertionError, assert_equal, mon.v[:, -1], 0)
    assert_raises(AssertionError, assert_equal, mon.v[:, -2], mon.v[:, -1])


# adapted for standalone mode from brian2/tests/test_neurongroup.py
@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_values_fixed_and_random():
    '''
    Test setting same seed for codeobject run only once (generating random
    numbers once using CuRAND host API) and setting random seed
    for codeobject run every tick (using buffer system in rand.cu). Adapted
    from brian2.tests.test_neurongroup.test_random_values_fixed_and_random().
    '''

    set_device('cuda_standalone', directory=None, build_on_run=False)

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    mon = StateMonitor(G, 'v', record=True)

    # first run
    seed(13579)
    G.v = 'rand()'
    seed()
    run(2*defaultclock.dt)

    # second run
    seed(13579)
    G.v = 'rand()'
    seed()
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])

    # First time step should be identical (same seed)
    assert_allclose(first_run_values[:, 0], second_run_values[:, 0])
    # Second should be different (random seed)
    assert_raises(AssertionError, assert_allclose,
                  first_run_values[:, 1], second_run_values[:, 1])


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_values_codeobject_every_tick():
    '''
    Test setting same seed for codeobject run every tick (using buffer system
    in rand.cu)
    '''

    set_device('cuda_standalone', directory=None, build_on_run=False)

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    mon = StateMonitor(G, 'v', record=True)

    # first run
    seed(10124)
    G.v = 'rand()'
    run(2*defaultclock.dt)

    # second run
    seed(10124)
    G.v = 'rand()'
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])

    # First time step should be identical (same seed)
    assert_allclose(first_run_values[:, 0], second_run_values[:, 0])
    # Second should also be identical (same seed)
    assert_allclose(first_run_values[:, 1], second_run_values[:, 1])

@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_binomial_values():
    '''
    Test setting same seed for binomial functions (using CuRAND device API).
    '''
    set_device('cuda_standalone', directory=None, build_on_run=False)

    binomial_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    binomial_f = BinomialFunction(100, 0.1, approximate=False)

    G = NeuronGroup(1, '''x : 1
                          y : 1''',
                    threshold='x>y')
    G.run_regularly('''x = binomial_f_approximated()
                       y = binomial_f()''')

    # TODO: add somewhere else
    ## use binomial in codeobject that is run only once
    #G.x = 'binomial_f_approximated()'
    #G.y = 'binomial_f()'

    mon = StateMonitor(G, ['x', 'y'], record=True)

    # first run
    run(2*defaultclock.dt)

    # second run
    seed(11400)
    run(2*defaultclock.dt)

    # third run
    run(2*defaultclock.dt)

    # forth run
    seed(11400)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    run_values_1 = np.vstack([mon.x[:, [0, 1]], mon.y[:, [0, 1]]])
    run_values_2 = np.vstack([mon.x[:, [2, 3]], mon.y[:, [2, 3]]])
    run_values_3 = np.vstack([mon.x[:, [4, 5]], mon.y[:, [4, 5]]])
    run_values_4 = np.vstack([mon.x[:, [6, 7]], mon.y[:, [6, 7]]])

    # Two calls to binomial functions should return different numbers in all runs
    for n, values in enumerate([run_values_1, run_values_2, run_values_3, run_values_4]):
        assert_raises(AssertionError, assert_allclose, values[:, 0], values[:, 1])


    # 2. and 4. run set the same seed
    assert_allclose(run_values_2, run_values_4)
    # all other combinations should be different
    assert_raises(AssertionError, assert_allclose,
                  run_values_1, run_values_2)
    assert_raises(AssertionError, assert_allclose,
                  run_values_1, run_values_3)
    assert_raises(AssertionError, assert_allclose,
                  run_values_2, run_values_3)


if __name__ == '__main__':
    test_rand()
    test_random_number_generation_with_multiple_runs()
    test_random_values_fixed_and_random()
    test_random_values_codeobject_every_tick()
