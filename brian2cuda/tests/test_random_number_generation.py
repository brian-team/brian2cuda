from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_raises, assert_allclose

from brian2 import *
from brian2.monitors.statemonitor import StateMonitor
from brian2.core.clocks import defaultclock
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices, device

import brian2cuda
from brian2cuda.device import check_codeobj_for_rng


@attr('cuda_standalone', 'standalone-only')
def test_rand_randn_regex():
    # Since we don't build anython but only test a Python function in
    # device.py, we can run this standalone-only test without set_device and
    # device.build

    # dummy class to fit with check_codeobj_for_rng function
    class DummyCodeobj():
        def __init__(self, string):
            self.template_name = 'TemplateName'
            self.name = 'Name'
            self.code = lambda: 0
            self.code.cu_file = string
            self.rand_calls = 0
            self.randn_calls = 0

    # should match
    m = []
    m.append(DummyCodeobj(' _rand(_vectorisation_idx) slfkjwefss'))
    m.append(DummyCodeobj('_rand(_vectorisation_idx) slfkjwefss'))
    m.append(DummyCodeobj(' _rand(_vectorisation_idx)'))
    m.append(DummyCodeobj('_rand(_vectorisation_idx)'))
    m.append(DummyCodeobj('*_rand(_vectorisation_idx)'))
    m.append(DummyCodeobj('_rand(_vectorisation_idx)-'))
    m.append(DummyCodeobj('+_rand(_vectorisation_idx)-'))

    # should not match
    n = []
    n.append(DummyCodeobj('h_rand(_vectorisation_idx)'))
    n.append(DummyCodeobj('_rand_h(_vectorisation_idx)'))
    n.append(DummyCodeobj('#_rand_h(_vectorisation_idx)'))
    n.append(DummyCodeobj('# _rand_h(_vectorisation_idx)'))
    n.append(DummyCodeobj(' # _rand_h(_vectorisation_idx)'))
    n.append(DummyCodeobj('#define _rand(_vectorisation_idx)'))
    n.append(DummyCodeobj(' #define _rand(_vectorisation_idx)'))
    n.append(DummyCodeobj('  #define _rand(_vectorisation_idx)'))

    # this one is matched currently (double space)
    #n.append(DummyCodeobj('  #define  _rand(_vectorisation_idx)'))

    for i, co in enumerate(m):
        check_codeobj_for_rng(co)
        assert co.rand_calls == 1, "{}: matches: {} in '{}'".format(i,
                                                                  co.rand_calls,
                                                                  co.code.cu_file)

    for i, co in enumerate(n):
        check_codeobj_for_rng(co)
        assert co.rand_calls == 0, "{}: matches: {} in '{}'".format(i,
                                                                  co.rand_calls,
                                                                  co.code.cu_file)


@attr('cuda_standalone', 'standalone-only')
def test_rand_randn_occurence_counting():

    set_device('cuda_standalone')

    G_rand = NeuronGroup(10, '''dx/dt = rand()/ms : 1''', threshold='True', name='G_rand')
    S_rand = Synapses(G_rand, G_rand, on_pre='''x += rand()''', name='S_rand')
    S_rand.connect()

    G_randn = NeuronGroup(10, '''dx/dt = randn()/ms : 1''', threshold='True', name='G_randn')
    S_randn = Synapses(G_randn, G_randn, on_pre='''x += randn()''', name='S_randn')
    S_randn.connect()

    run(0*ms)

    assert len(device.all_code_objects['binomial']) == 0

    for code_object in device.all_code_objects['rand']:
        assert code_object.rand_calls == 1
        assert code_object.randn_calls == 0

    for code_object in device.all_code_objects['randn']:
        assert code_object.rand_calls == 0
        assert code_object.randn_calls == 1

    for code_object in device.all_code_objects['rand_or_randn']:
        assert code_object not in device.code_object_with_binomial_separate_call


@attr('cuda_standalone', 'standalone-only')
def test_binomial_occurence():

    set_device('cuda_standalone')

    my_f_a = BinomialFunction(100, 0.1, approximate=True)
    my_f = BinomialFunction(100, 0.1, approximate=False)


    G_my_f = NeuronGroup(10, '''dx/dt = my_f()/ms : 1''', threshold='True', name='G_my_f')
    G_my_f.x = 'my_f()'
    S_my_f = Synapses(G_my_f, G_my_f, on_pre='''x += my_f()''', name='S_my_f')
    S_my_f.connect()

    G_my_f_a = NeuronGroup(10, '''dx/dt = my_f_a()/ms : 1''', threshold='True', name='G_my_f_a')
    S_my_f_a = Synapses(G_my_f_a, G_my_f_a, on_pre='''x += my_f_a()''', name='S_my_f_a')
    S_my_f_a.connect()

    run(0*ms)

    for key in ['rand', 'randn', 'rand_or_randn']:
        assert len(device.all_code_objects[key]) == 0

    # all_code_objects['binomial'] collects all codeobjects that use binomials (run
    # every or single clock cycle). We have 5 objects
    assert len(device.all_code_objects['binomial']) == 5

    # Here we collect all codeobjects with binomial run only ones
    # This is only one from the G_my_f.x = 'my_f()' line above
    obj_list = device.code_object_with_binomial_separate_call
    assert len(obj_list) == 1
    assert obj_list[0].name == 'G_my_f_group_variable_set_conditional_codeobject'


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_rand():
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(3*defaultclock.dt)

    print(mon.v[:5, :])
    assert_raises(AssertionError, assert_equal, mon.v[:, -1], 0)
    assert_raises(AssertionError, assert_equal, mon.v[:, -2], mon.v[:, -1])


@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_number_generation_with_multiple_runs():
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    run(2*defaultclock.dt)
    device.build(direct_call=False, **device.build_options)

    assert_raises(AssertionError, assert_equal, mon.v[:, -1], 0)
    assert_raises(AssertionError, assert_equal, mon.v[:, -2], mon.v[:, -1])


# adapted for standalone mode from brian2/tests/test_neurongroup.py
# brian2.tests.test_neurongroup.test_random_values_fixed_and_random().
@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_values_fixed_and_random_seed():
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


# Test all binomial is single test
@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_binomial_values():
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    my_f = BinomialFunction(100, 0.1, approximate=False)

    # Test neurongroup every tick objects
    G = NeuronGroup(10,'''dx/dt = my_f_approximated()/ms: 1
                          dy/dt = my_f()/ms: 1''',
                    threshold='True')
    G.run_regularly('''x = my_f_approximated()
                       y = my_f()''')


    # Test synapses every tick objects (where N is not known at compile time)
    syn = Synapses(G, G,
                   model='''
                         dw/dt = my_f()/ms: 1
                         dv/dt = my_f_approximated()/ms: 1
                         ''',
                   on_pre='''
                          x += w
                          y += v
                          '''
                   # TODO: fails when having binomial here, why?
                   #on_pre='''
                   #       x += w * my_f()
                   #       y += v * my_f_approximated()
                   #       '''
                   )
    # Test synapses generation, which needs host side binomial
    syn.connect(condition='my_f_approximated() < 100')

    mon = StateMonitor(G, ['x', 'y'], record=True)
    w_mon = StateMonitor(syn, ['w', 'v'], record=np.arange(100))

    def init_group_variables(my_f, my_f_approximated):
        # Test codeobjects run only once outside the network,
        # G.x uses group_variable_set_conditional, G.x[:5] uses group_variable_set
        # Synapse objects (N not known at compile time)
        syn.w = 'my_f()'
        syn.w['i < j'] = 'my_f_approximated()'
        syn.v = 'my_f_approximated()'
        syn.v['i < j'] = 'my_f()'
        # Neurongroup object
        G.x = 'my_f_approximated()'
        G.x[:5] = 'my_f()'
        G.y = 'my_f()'
        G.y[:5] = 'my_f_approximated()'

    # first run
    init_group_variables(my_f, my_f_approximated)
    run(2*defaultclock.dt)

    # second run
    seed(11400)
    init_group_variables(my_f, my_f_approximated)
    run(2*defaultclock.dt)

    # third run
    seed()
    init_group_variables(my_f, my_f_approximated)
    run(2*defaultclock.dt)

    # forth run
    seed(11400)
    init_group_variables(my_f, my_f_approximated)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    run_values_1 = np.vstack([mon.x[:, [0, 1]], mon.y[:, [0, 1]],
                              w_mon.w[:, [0, 1]], w_mon.v[:, [0, 1]]])
    run_values_2 = np.vstack([mon.x[:, [2, 3]], mon.y[:, [2, 3]],
                              w_mon.w[:, [2, 3]], w_mon.v[:, [2, 3]]])
    run_values_3 = np.vstack([mon.x[:, [4, 5]], mon.y[:, [4, 5]],
                              w_mon.w[:, [4, 5]], w_mon.v[:, [4, 5]]])
    run_values_4 = np.vstack([mon.x[:, [6, 7]], mon.y[:, [6, 7]],
                              w_mon.w[:, [6, 7]], w_mon.v[:, [6, 7]]])

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


####### RAND / RANDN ######
### 1. rand/randn in neurongroup set_conditional template covered in brian2 test
### 2. randn in neurongroup stateupdater covered in same brian2 test
###    in brian2/tests/test_neurongroup.py:test_random_values_random/fixed_seed.py

### 3. rand/randn in synapses set_conditional template
@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_set_synapses_random_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1''')
    S.connect()
    seed()
    S.v1 = 'rand() + randn()'
    seed()
    S.v2 = 'rand() + randn()'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert np.var(S.v1[:] - S.v2[:]) > 0


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1''')
    S.connect()
    seed(12345678)
    S.v1 = 'rand() + randn()'
    seed(12345678)
    S.v2 = 'rand() + randn()'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert_allclose(S.v1[:], S.v2[:])


### 4. randn in synapses stateupdater and mixing seed random/fixed
@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_random_values_synapse_dynamics_fixed_and_random_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, 'dv/dt = -v/(10*ms) + 0.1*xi/sqrt(ms) : 1')
    S.connect()
    mon = StateMonitor(S, 'v', record=range(100))

    # first run
    S.v = 0
    seed()
    run(2*defaultclock.dt)

    # third run
    S.v = 0
    seed()
    run(2*defaultclock.dt)

    # third run
    S.v = 0
    seed(13579)
    run(2*defaultclock.dt)

    # fourth run
    S.v = 0
    seed(13579)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])
    third_run_values = np.array(mon.v[:, [4, 5]])
    fourth_run_values = np.array(mon.v[:, [6, 7]])

    # First and second run should be different (random seed)
    assert_raises(AssertionError, assert_allclose,
                  first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 5. rand/randn in host side rng (synapses_init tempalte)
@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_init_synapses_fixed_and_random_seed():
    G = NeuronGroup(10, 'z : 1')

    seed()
    S1 = Synapses(G, G)
    S1.connect('randn() + 0.5 < rand()')

    seed(12345678)
    S2 = Synapses(G, G)
    S2.connect('randn() + 0.5 < rand()')

    seed()
    S3 = Synapses(G, G)
    S3.connect('randn() + 0.5 < rand()')

    seed(12345678)
    S4 = Synapses(G, G)
    S4.connect('randn() + 0.5 < rand()')

    run(0*ms)  # for standalone

    idcs1 = np.hstack([S1.i[:], S1.j[:]])
    idcs2 = np.hstack([S2.i[:], S2.j[:]])
    idcs3 = np.hstack([S3.i[:], S3.j[:]])
    idcs4 = np.hstack([S4.i[:], S4.j[:]])

    # Pre/post idcs for first, second and third run should be different
    assert_raises(AssertionError, assert_equal,
                  idcs1, idcs2)
    assert_raises(AssertionError, assert_equal,
                  idcs1, idcs3)
    assert_raises(AssertionError, assert_equal,
                  idcs2, idcs3)
    # Pre/post idcs for second and fourth run should be equal (same seed)
    assert_equal(idcs2, idcs4)


####### BINOMIAL ######

### 1. binomial in neurongroup set_conditional templates
@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_binomial_values_random_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1''')
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    seed()
    G.v1 = 'my_f() + my_f_approximated()'
    seed()
    G.v2 = 'my_f() + my_f_approximated()'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_binomial_values_fixed_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1''')
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    seed(12345678)
    G.v1 = 'my_f() + my_f_approximated()'
    seed(12345678)
    G.v2 = 'my_f() + my_f_approximated()'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


### 2. binomial in neurongroup stateupdater and mixed seed random/fixed
@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_binomial_values_fixed_and_random_seed():
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*(my_f() + my_f_approximated())*xi/sqrt(ms) : 1')
    mon = StateMonitor(G, 'v', record=True)

    # first run
    G.v = 0
    seed()
    run(2*defaultclock.dt)

    # second run
    G.v = 0
    seed()
    run(2*defaultclock.dt)

    # third run
    G.v = 0
    seed(13579)
    run(2*defaultclock.dt)

    # fourth run
    G.v = 0
    seed(13579)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])
    third_run_values = np.array(mon.v[:, [4, 5]])
    fourth_run_values = np.array(mon.v[:, [6, 7]])

    # First and second run should be different (random seed)
    assert_raises(AssertionError, assert_allclose,
                  first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 3. binomial in synapses set_conditional template
@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_set_synapses_random_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1''')
    S.connect()

    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed()
    S.v1 = 'my_f() + my_f_approximated()'
    seed()
    S.v2 = 'my_f() + my_f_approximated()'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert np.var(S.v1[:] - S.v2[:]) > 0


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1''')
    S.connect()

    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed(12345678)
    S.v1 = 'my_f() + my_f_approximated()'
    seed(12345678)
    S.v2 = 'my_f() + my_f_approximated()'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert_allclose(S.v1[:], S.v2[:])


### 4. binomial in synapses stateupdater and mixing seed random/fixed
@attr('standalone-compatible', 'multiple-runs')
@with_setup(teardown=reinit_devices)
def test_binomial_values_synapse_dynamics_fixed_and_random_seed():
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, 'dv/dt = -v/(10*ms) + 0.1*(my_f() + my_f_approximated())*xi/sqrt(ms) : 1')
    S.connect()
    mon = StateMonitor(S, 'v', record=range(100))

    # first run
    S.v = 0
    seed()
    run(2*defaultclock.dt)

    # third run
    S.v = 0
    seed()
    run(2*defaultclock.dt)

    # third run
    S.v = 0
    seed(13579)
    run(2*defaultclock.dt)

    # fourth run
    S.v = 0
    seed(13579)
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])
    third_run_values = np.array(mon.v[:, [4, 5]])
    fourth_run_values = np.array(mon.v[:, [6, 7]])

    # First and second run should be different (random seed)
    assert_raises(AssertionError, assert_allclose,
                  first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 5. binomial in host side rng (synapses_init tempalte)
@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_binomial_values_init_synapses_fixed_and_random_seed():
    G = NeuronGroup(10, 'z : 1')

    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed()
    S1 = Synapses(G, G)
    S1.connect('my_f() < my_f_approximated()')

    seed(12345678)
    S2 = Synapses(G, G)
    S2.connect('my_f() < my_f_approximated()')

    seed()
    S3 = Synapses(G, G)
    S3.connect('my_f() < my_f_approximated()')

    seed(12345678)
    S4 = Synapses(G, G)
    S4.connect('my_f() < my_f_approximated()')

    run(0*ms)  # for standalone

    idcs1 = np.hstack([S1.i[:], S1.j[:]])
    idcs2 = np.hstack([S2.i[:], S2.j[:]])
    idcs3 = np.hstack([S3.i[:], S3.j[:]])
    idcs4 = np.hstack([S4.i[:], S4.j[:]])

    # Pre/post idcs for first, second and third run should be different
    assert_raises(AssertionError, assert_equal,
                  idcs1, idcs2)
    assert_raises(AssertionError, assert_equal,
                  idcs1, idcs3)
    assert_raises(AssertionError, assert_equal,
                  idcs2, idcs3)
    # Pre/post idcs for second and fourth run should be equal (same seed)
    assert_equal(idcs2, idcs4)

### Extra: group_set template (not conditional), only for neurongroup, not synapse
@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_binomial_set_template_random_seed():
    G = NeuronGroup(10, '''v1 : 1
                           v2 : 1''')
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed()
    G.v1[:8] = 'rand() + randn() + my_f() + my_f_approximated()'
    seed()
    G.v2[:8] = 'rand() + randn() + my_f() + my_f_approximated()'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_random_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, '''v1 : 1
                           v2 : 1''')
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed(12345678)
    G.v1[:8] = 'rand() + randn() + my_f() + my_f_approximated()'
    seed(12345678)
    G.v2[:8] = 'rand() + randn() + my_f() + my_f_approximated()'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


if __name__ == '__main__':
    test_rand()
    test_random_number_generation_with_multiple_runs()
    test_random_values_fixed_and_random()
    test_random_values_codeobject_every_tick()
    test_rand_randn_regex()
