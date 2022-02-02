import pytest

from brian2 import *
from brian2.devices.device import reinit_and_delete
from brian2.utils.logger import catch_logs
from brian2.stateupdaters.base import UnsupportedEquationsException
from numpy.testing import assert_equal
from brian2.tests.utils import assert_allclose


# Tests below are standalone-compatible versions of all tests from
# `brian2/tests/test_stateupdaters.py`, which aren't standalone-compatible (due to
# multiple run calls).


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_multiple_noise_variables_extended():
    # Some actual simulations with multiple noise variables
    eqs = '''dx/dt = y : 1
             dy/dt = - 1*ms**-1*y - 40*ms**-2*x : Hz
            '''
    all_eqs_noise = ['''dx/dt = y : 1
                        dy/dt = noise_factor*ms**-1.5*xi_1 + noise_factor*ms**-1.5*xi_2
                           - 1*ms**-1*y - 40*ms**-2*x : Hz
                     ''',
                     '''dx/dt = y + noise_factor*ms**-0.5*xi_1: 1
                        dy/dt = noise_factor*ms**-1.5*xi_2
                            - 1*ms**-1*y - 40*ms**-2*x : Hz
                     ''']
    G = NeuronGroup(2, eqs, method='euler')
    G.x = [0.5, 1]
    G.y = [0, 0.5] * Hz
    mon1 = StateMonitor(G, ['x', 'y'], record=True)
    net = Network(G, mon1)
    net.run(10*ms)

    monitors = []
    for eqs_noise in all_eqs_noise:
        monitors.append(dict())
        for method_name, method in [('euler', euler), ('heun', heun)]:
            with catch_logs('WARNING'):
                G = NeuronGroup(2, eqs_noise, method=method)
                G.x = [0.5, 1]
                G.y = [0, 0.5] * Hz
                mon = StateMonitor(G, ['x', 'y'], record=True)
                net = Network(G, mon)
                # We run it deterministically, but still we'd detect major errors (e.g.
                # non-stochastic terms that are added twice, see #330
                net.run(10*ms, namespace={'noise_factor': 0})
                monitors[-1][method_name] = mon

    device.build(direct_call=False, **device.build_options)

    no_noise_x, no_noise_y = mon1.x[:], mon1.y[:]
    for i in range(len(all_eqs_noise)):
        for method_name in ['euler', 'heun']:
            mon = monitors[i][method_name]
            assert_allclose(mon.x[:], no_noise_x,
                            err_msg=f'Method {method_name} gave incorrect results')
            assert_allclose(mon.y[:], no_noise_y,
                            err_msg=f'Method {method_name} gave incorrect results')


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_multiple_noise_variables_deterministic_noise(fake_randn_randn_fixture):
    all_eqs = ['''dx/dt = y : 1
                          dy/dt = -y / (10*ms) + dt**-.5*0.5*ms**-1.5 + dt**-.5*0.5*ms**-1.5: Hz
                     ''',
                     '''dx/dt = y + dt**-.5*0.5*ms**-0.5: 1
                        dy/dt = -y / (10*ms) + dt**-.5*0.5 * ms**-1.5 : Hz
                ''']
    all_eqs_noise = ['''dx/dt = y : 1
                          dy/dt = -y / (10*ms) + xi_1 * ms**-1.5 + xi_2 * ms**-1.5: Hz
                     ''',
                     '''dx/dt = y + xi_1*ms**-0.5: 1
                        dy/dt = -y / (10*ms) + xi_2 * ms**-1.5 : Hz
                     ''']

    monitors_no_noise = []
    monitors = []
    for eqs, eqs_noise in zip(all_eqs, all_eqs_noise):
        G = NeuronGroup(2, eqs, method='euler')
        G.x = [5,  17]
        G.y = [25, 5 ] * Hz
        mon = StateMonitor(G, ['x', 'y'], record=True)
        net = Network(G, mon)
        net.run(10*ms)
        monitors_no_noise.append(mon)
        monitors.append(dict())

        for method_name, method in [('euler', euler), ('heun', heun)]:
            with catch_logs('WARNING'):
                G = NeuronGroup(2, eqs_noise, method=method)
                G.x = [5,  17]
                G.y = [25, 5 ] * Hz
                mon = StateMonitor(G, ['x', 'y'], record=True)
                net = Network(G, mon)
                net.run(10*ms)
                monitors[-1][method_name] = mon

    device.build(direct_call=False, **device.build_options)

    for i in range(len(all_eqs)):
        mon = monitors_no_noise[i]
        no_noise_x, no_noise_y = mon.x[:], mon.y[:]
        for method_name in ['euler', 'heun']:
            mon = monitors[i][method_name]
            assert_allclose(mon.x[:], no_noise_x,
                            err_msg=f'Method {method_name} gave incorrect results')
            assert_allclose(mon.y[:], no_noise_y,
                            err_msg=f'Method {method_name} gave incorrect results')


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_pure_noise_deterministic(fake_randn_randn_fixture):
    sigma = 3
    eqs = Equations('dx/dt = sigma*xi/sqrt(ms) : 1')
    dt = 0.1*ms
    G_dict = {}
    for method in ['euler', 'heun', 'milstein']:
        G = NeuronGroup(1, eqs, dt=dt, method=method)
        run(10*dt)
        G_dict[method] = G

    device.build(direct_call=False, **device.build_options)

    for method in ['euler', 'heun', 'milstein']:
        G = G_dict[method]
        assert_allclose(G.x, sqrt(dt)*sigma*0.5/sqrt(1*ms)*10,
                        err_msg=f'method {method} did not give the expected result')


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_subexpressions():
    '''
    Make sure that the integration of a (non-stochastic) differential equation
    does not depend on whether it's formulated using subexpressions.
    '''
    # no subexpression
    eqs1 = 'dv/dt = (-v + sin(2*pi*100*Hz*t)) / (10*ms) : 1'
    # same with subexpression
    eqs2 = '''dv/dt = I / (10*ms) : 1
              I = -v + sin(2*pi*100*Hz*t): 1'''

    objects = {}
    methods = ['exponential_euler', 'rk2', 'rk4']  # euler is tested in test_subexpressions_basic
    for method in methods:
        G1 = NeuronGroup(1, eqs1, method=method)
        G1.v = 1
        G2 = NeuronGroup(1, eqs2, method=method)
        G2.v = 1
        mon1 = StateMonitor(G1, 'v', record=True)
        mon2 = StateMonitor(G2, 'v', record=True)
        net = Network(G1, mon1, G2, mon2)
        net.run(10*ms)
        objects[method] = (G1, G2, mon1, mon2)

    device.build(direct_call=False, **device.build_options)

    for method in methods:
        G1, G2, mon1, mon2 = objects[method]
        assert_equal(mon1.v, mon2.v, f'Results for method {method} differed!')


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_refractory():
    # Compare integration with and without the addition of refractoriness --
    # note that the cell here is not spiking, so it should never be in the
    # refractory period and therefore the results should be exactly identical
    # with and without (unless refractory)
    objects = {}
    eqs_base = 'dv/dt = -v/(10*ms) : 1'
    for method in ['linear', 'exact', 'independent', 'euler', 'exponential_euler', 'rk2', 'rk4']:
        G_no_ref = NeuronGroup(10, eqs_base, method=method)
        G_no_ref.v = '(i+1)/11.'
        G_ref = NeuronGroup(10, eqs_base + '(unless refractory)',
                            refractory=1*ms, method=method)
        G_ref.v = '(i+1)/11.'
        net = Network(G_ref, G_no_ref)
        net.run(10*ms)
        objects[method] = (G_no_ref, G_ref)

    device.build(direct_call=False, **device.build_options)

    for method in ['linear', 'exact', 'independent', 'euler', 'exponential_euler', 'rk2', 'rk4']:
        G_no_ref, G_ref = objects[method]
        assert_allclose(G_no_ref.v[:], G_ref.v[:],
                        err_msg=('Results with and without refractoriness '
                                 'differ for method %s.') % method)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_refractory_stochastic(fake_randn_randn_fixture):

    eqs_base = 'dv/dt = -v/(10*ms) + second**-.5*xi : 1'

    objects = {}
    for method in ['euler', 'heun', 'milstein']:
        G_no_ref = NeuronGroup(10, eqs_base, method=method)
        G_no_ref.v = '(i+1)/11.'
        G_ref = NeuronGroup(10, eqs_base + ' (unless refractory)',
                            refractory=1*ms, method=method)
        G_ref.v = '(i+1)/11.'
        net = Network(G_ref, G_no_ref)
        net.run(10*ms)
        objects[method] = (G_no_ref, G_ref)

    device.build(direct_call=False, **device.build_options)

    for method in ['euler', 'heun', 'milstein']:
        G_no_ref, G_ref = objects[method]
        assert_allclose(G_no_ref.v[:], G_ref.v[:],
                        err_msg=('Results with and without refractoriness '
                                 'differ for method %s.') % method)


if __name__ == '__main__':
    import brian2cuda
    from brian2cuda.tests.conftest import fake_randn
    for test in [
            test_multiple_noise_variables_extended,
            test_multiple_noise_variables_deterministic_noise,
            test_pure_noise_deterministic,
            test_subexpressions,
            test_refractory,
            test_refractory_stochastic,
    ]:
        print(test.__name__)
        pytestmarks = [mark.name for mark in test.pytestmark]
        build_on_run = True
        if 'multiple_runs' in pytestmarks:
            build_on_run = False
        set_device('cuda_standalone', build_on_run=build_on_run, directory=None)
        try:
            test()
        except  TypeError:
            # Functions using fake_randn_randn_fixture, set fake randn manually here
            orig_randn = DEFAULT_FUNCTIONS['randn']
            DEFAULT_FUNCTIONS['randn'] = fake_randn
            test(None)
            DEFAULT_FUNCTIONS['randn'] = orig_randn

        reinit_and_delete()
