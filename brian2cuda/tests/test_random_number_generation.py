from collections import OrderedDict, defaultdict

import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.monitors.statemonitor import StateMonitor
from brian2.core.clocks import defaultclock
from brian2.devices.device import device, reinit_and_delete
from brian2.tests.utils import assert_allclose

import brian2cuda
from brian2cuda.device import prepare_codeobj_code_for_rng


# dummy class to fit with prepare_codeobj_code_for_rng function
class _DummyCodeobj():
    def __init__(self, string):
        self.template_name = 'DummyTemplate'
        self.name = 'Dummy'
        self.variables = {}
        self.code = lambda: 0
        self.code.cu_file = string
        self.rng_calls = defaultdict(int)
        self.poisson_lamdas = defaultdict(float)
        self.needs_curand_states = False


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_rand_randn_regex():
    # Since we don't build anything but only test a Python function in
    # device.py, we can run this standalone-only test without set_device and
    # device.build

    # should match
    m = []
    m.append(_DummyCodeobj(' _rand(_vectorisation_idx) slfkjwefss'))
    m.append(_DummyCodeobj('_rand(_vectorisation_idx) slfkjwefss'))
    m.append(_DummyCodeobj(' _rand(_vectorisation_idx)'))
    m.append(_DummyCodeobj('_rand(_vectorisation_idx)'))
    m.append(_DummyCodeobj('*_rand(_vectorisation_idx)'))
    m.append(_DummyCodeobj('_rand(_vectorisation_idx)-'))
    m.append(_DummyCodeobj('+_rand(_vectorisation_idx)-'))

    # should not match
    n = []
    n.append(_DummyCodeobj('h_rand(_vectorisation_idx)'))
    n.append(_DummyCodeobj('_rand_h(_vectorisation_idx)'))
    n.append(_DummyCodeobj('#_rand_h(_vectorisation_idx)'))
    n.append(_DummyCodeobj('# _rand_h(_vectorisation_idx)'))
    n.append(_DummyCodeobj(' # _rand_h(_vectorisation_idx)'))
    n.append(_DummyCodeobj('#define _rand(_vectorisation_idx)'))
    n.append(_DummyCodeobj(' #define _rand(_vectorisation_idx)'))
    n.append(_DummyCodeobj('  #define _rand(_vectorisation_idx)'))

    # this one is matched currently (double space)
    #n.append(_DummyCodeobj('  #define  _rand(_vectorisation_idx)'))

    for i, co in enumerate(m):
        prepare_codeobj_code_for_rng(co)
        assert co.rng_calls["rand"] == 1, \
            f"{i}: matches: {co.rng_calls['rand']} in '{co.code.cu_file}'"

    for i, co in enumerate(n):
        prepare_codeobj_code_for_rng(co)
        assert co.rng_calls["rand"] == 0, \
            f"{i}: matches: {co.rng_calls['rand']} in '{co.code.cu_file}'"


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_poisson_regex():
    # Since we don't build anything but only test a Python function in
    # device.py, we can run this standalone-only test without set_device and
    # device.build

    co = _DummyCodeobj('''
        _rand(_vectorisation_idx)
        _poisson(l, _vectorisation_idx)
        _poisson(5, _vectorisation_idx)
        _poisson(.01, _vectorisation_idx)
        _poisson(5, _vectorisation_idx)
        _poisson(.01, _vectorisation_idx)
        _poisson(.01, _vectorisation_idx)
        _poisson(l, _vectorisation_idx)
        ''')
    prepare_codeobj_code_for_rng(co)

    # If lambda is a variable (l), nothing will be replaces (on-the-fly RNG)
    # If lambda is a literal (.01 and 5), it will be replaced by a
    # _ptr_array_{name}_poisson_<idx> variables, where the idx is ascending by
    # lambda value (lamda=.01 gets idx=0; lamda=5 gets idx=1)
    replaced_code = f'''
        _rand(_vectorisation_idx + 0 * _N)
        _poisson(l, _vectorisation_idx)
        _poisson(_ptr_array_{co.name}_poisson_1, _vectorisation_idx + 0 * _N)
        _poisson(_ptr_array_{co.name}_poisson_0, _vectorisation_idx + 0 * _N)
        _poisson(_ptr_array_{co.name}_poisson_1, _vectorisation_idx + 1 * _N)
        _poisson(_ptr_array_{co.name}_poisson_0, _vectorisation_idx + 1 * _N)
        _poisson(_ptr_array_{co.name}_poisson_0, _vectorisation_idx + 2 * _N)
        _poisson(l, _vectorisation_idx)
        '''

    for i, (cu_line, replaced_line) in enumerate(zip(co.code.cu_file.split('\n'),
                                                     replaced_code.split('\n'))):
        assert_equal(cu_line, replaced_line, err_msg=f"Line {i} is wrong")
    assert_equal(co.code.cu_file, replaced_code)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_rng_occurrence_counting():

    set_device('cuda_standalone', directory=None)

    G_rand = NeuronGroup(10, '''dx/dt = rand()/ms : 1''', threshold='True', name='G_rand')
    S_rand = Synapses(G_rand, G_rand, on_pre='''x += rand()''', name='S_rand')
    S_rand.connect()

    G_randn = NeuronGroup(10, '''dx/dt = randn()/ms : 1''', threshold='True', name='G_randn')
    S_randn = Synapses(G_randn, G_randn, on_pre='''x += randn()''', name='S_randn')
    S_randn.connect()

    G_poisson = NeuronGroup(10, '''dx/dt = poisson(1)/ms : 1''', threshold='True', name='G_poisson')
    S_poisson = Synapses(G_poisson, G_poisson, on_pre='''x += poisson(1)''', name='S_poisson')
    S_poisson.connect()

    run(0*ms)

    # Check that there is no device side RNG registered (no binomial or poisson with
    # vectorized lamda)
    for codeobjects in device.codeobjects_with_rng['device_api'].values():
        assert len(codeobjects) == 0, codeobjects

    # Check that for each codeobject that has rand, randn or poisson detected, the number
    # of occurrences is correct.
    for check_rng in ['rand', 'randn', 'poisson_0']:
        for code_object in device.codeobjects_with_rng['host_api']['all_runs'][check_rng]:
            for rng_type, num_occurrence in code_object.rng_calls.items():
                if rng_type == check_rng:
                    assert num_occurrence == 1, (
                        '{rng_type} occurs {num_occurrence} times (not 1)'
                    )
                else:
                    assert num_occurrence == 0, (
                        '{rng_type} occurs {num_occurrence} times (not 0)'
                    )

                if rng_type == 'poisson_0':
                    co_lamda = code_object.poisson_lamdas['poisson_0']
                    assert co_lamda == 1.0, co_lamda
                    d_lamda = device.all_poisson_lamdas[code_object.name]['poisson_0']
                    assert d_lamda == 1.0, f"{d_lamda} {code_object.name}"

                assert not code_object.needs_curand_states


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_binomial_occurrence():

    set_device('cuda_standalone', directory=None)

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

    for codeobjects in device.codeobjects_with_rng['host_api']['all_runs'].values():
        assert len(codeobjects) == 0

    # Check the number of codeobjects using curand and that are run every clock cycle
    every_tick = device.codeobjects_with_rng['device_api']['every_tick']
    assert len(every_tick) == 4, every_tick

    # Here we collect all codeobjects with binomial run only ones
    # This is only one from the G_my_f.x = 'my_f()' line above
    single_tick = device.codeobjects_with_rng['device_api']['single_tick']
    assert len(single_tick) == 1
    assert single_tick[0].name == 'G_my_f_group_variable_set_conditional_codeobject'


@pytest.mark.standalone_compatible
def test_rand():
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(3*defaultclock.dt)

    print(mon.v[:5, :])
    with pytest.raises(AssertionError):
        assert_equal(mon.v[:, -1], 0)
    with pytest.raises(AssertionError):
        assert_equal(mon.v[:, -2], mon.v[:, -1])


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_random_number_generation_with_multiple_runs():
    G = NeuronGroup(1000, 'dv/dt = rand() : second')
    mon = StateMonitor(G, 'v', record=True)

    run(1*defaultclock.dt)
    run(2*defaultclock.dt)
    device.build(direct_call=False, **device.build_options)

    with pytest.raises(AssertionError):
        assert_equal(mon.v[:, -1], 0)
    with pytest.raises(AssertionError):
        assert_equal(mon.v[:, -2], mon.v[:, -1])


# adapted for standalone mode from brian2/tests/test_neurongroup.py
# brian2.tests.test_neurongroup.test_random_values_fixed_and_random().
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_random_values_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values[:, 1], second_run_values[:, 1])


# adapted for standalone mode from brian2/tests/test_neurongroup.py
# brian2.tests.test_neurongroup.test_random_values_fixed_and_random().
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_poisson_scalar_values_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')


    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*poisson(5)/ms : 1')
    mon = StateMonitor(G, 'v', record=True)

    # first run
    seed(13579)
    G.v = 'poisson(5)'
    seed()
    run(2*defaultclock.dt)

    # second run
    seed(13579)
    G.v = 'poisson(5)'
    seed()
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])

    # First time step should be identical (same seed)
    assert_allclose(first_run_values[:, 0], second_run_values[:, 0])
    # Second should be different (random seed)
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values[:, 1], second_run_values[:, 1])


# adapted for standalone mode from brian2/tests/test_neurongroup.py
# brian2.tests.test_neurongroup.test_random_values_fixed_and_random().
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_poisson_vectorized_values_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

    G = NeuronGroup(10,
                    '''l: 1
                       dv/dt = -v/(10*ms) + 0.1*poisson(l)/ms : 1''')
    G.l = arange(10)
    mon = StateMonitor(G, 'v', record=True)

    # first run
    seed(13579)
    G.v = 'poisson(l)'
    seed()
    run(2*defaultclock.dt)

    # second run
    seed(13579)
    G.v = 'poisson(l)'
    seed()
    run(2*defaultclock.dt)

    device.build(direct_call=False, **device.build_options)

    first_run_values = np.array(mon.v[:, [0, 1]])
    second_run_values = np.array(mon.v[:, [2, 3]])

    # First time step should be identical (same seed)
    assert_allclose(first_run_values[:, 0], second_run_values[:, 0])
    # Second should be different (random seed)
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values[:, 1], second_run_values[:, 1])


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
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
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_binomial_values():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')


    # On Denis' local computer this test blows up all available RAM + SWAP when
    # compiling with all threads in parallel. Use half the threads instead.
    import socket
    if socket.gethostname() == 'selene':
        prefs.devices.cpp_standalone.extra_make_args_unix = ['-j4']

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
        with pytest.raises(AssertionError):
            assert_allclose(values[:, 0], values[:, 1])

    # 2. and 4. run set the same seed
    assert_allclose(run_values_2, run_values_4)
    # all other combinations should be different
    with pytest.raises(AssertionError):
        assert_allclose(run_values_1, run_values_2)
    with pytest.raises(AssertionError):
        assert_allclose(run_values_1, run_values_3)
    with pytest.raises(AssertionError):
        assert_allclose(run_values_2, run_values_3)


####### RAND / RANDN ######
### 1. rand/randn in neurongroup set_conditional template covered in brian2 test
### 2. randn in neurongroup stateupdater covered in same brian2 test
###    in brian2/tests/test_neurongroup.py:test_random_values_random/fixed_seed.py

### 3. rand/randn in synapses set_conditional template
@pytest.mark.standalone_compatible
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


@pytest.mark.standalone_compatible
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
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_random_values_synapse_dynamics_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')


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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 5. rand/randn in host side rng (synapses_init tempalte)
@pytest.mark.standalone_compatible
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
    with pytest.raises(AssertionError):
        assert_equal(idcs1, idcs2)
    with pytest.raises(AssertionError):
        assert_equal(idcs1, idcs3)
    with pytest.raises(AssertionError):
        assert_equal(idcs2, idcs3)
    # Pre/post idcs for second and fourth run should be equal (same seed)
    assert_equal(idcs2, idcs4)


####### BINOMIAL ######

### 1. binomial in neurongroup set_conditional templates
@pytest.mark.standalone_compatible
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


@pytest.mark.standalone_compatible
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
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_binomial_values_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 3. binomial in synapses set_conditional template
@pytest.mark.standalone_compatible
def test_binomial_values_set_synapses_random_seed():
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


@pytest.mark.standalone_compatible
def test_binomial_values_set_synapses_fixed_seed():
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
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_binomial_values_synapse_dynamics_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 5. binomial in host side rng (synapses_init tempalte)
@pytest.mark.standalone_compatible
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
    # Pre/post idcs for first, second and third run should be different
    with pytest.raises(AssertionError):
        assert_equal(idcs1, idcs2)
    with pytest.raises(AssertionError):
        assert_equal(idcs1, idcs3)
    with pytest.raises(AssertionError):
        assert_equal(idcs2, idcs3)
    # Pre/post idcs for second and fourth run should be equal (same seed)
    assert_equal(idcs2, idcs4)

### Extra: group_set template (not conditional), only for neurongroup, not synapse
@pytest.mark.standalone_compatible
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


@pytest.mark.standalone_compatible
def test_random_binomial_poisson_scalar_lambda_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, '''v1 : 1
                           v2 : 1''')
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed(12345678)
    G.v1[:8] = 'rand() + randn() + my_f() + my_f_approximated() + poisson(5)'
    seed(12345678)
    G.v2[:8] = 'rand() + randn() + my_f() + my_f_approximated() + poisson(5)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


@pytest.mark.standalone_compatible
def test_random_binomial_poisson_variable_lambda_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, '''v1 : 1
                           v2 : 1
                           l  : 1''')
    G.l = arange(10)
    my_f = BinomialFunction(100, 0.1, approximate=False)
    my_f_approximated = BinomialFunction(100, 0.1, approximate=True)

    seed(12345678)
    G.v1[:8] = 'rand() + randn() + my_f() + my_f_approximated() + poisson(l)'
    seed(12345678)
    G.v2[:8] = 'rand() + randn() + my_f() + my_f_approximated() + poisson(l)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


###### POISSON ######

### 1. poisson in neurongroup set_conditional templates
@pytest.mark.standalone_compatible
def test_poisson_scalar_lambda_values_random_seed():

    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1''')
    seed()
    G.v1 = 'poisson(5)'
    seed()
    G.v2 = 'poisson(5)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@pytest.mark.standalone_compatible
def test_poisson_variable_lambda_values_random_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1
                            x1 : 1
                            x2 : 1
                            l  : 1''')
    G.l = arange(100) / 10
    seed()
    G.v1 = 'poisson(l)'
    seed()
    G.v2 = 'poisson(l)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@pytest.mark.standalone_compatible
def test_poisson_scalar_lambda_values_fixed_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1''')
    seed(12345678)
    G.v1 = 'poisson(5)'
    seed(12345678)
    G.v2 = 'poisson(5)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


@pytest.mark.standalone_compatible
def test_poisson_variable_lambda_values_fixed_seed():
    G = NeuronGroup(100, '''v1 : 1
                            v2 : 1
                            l  : 1''')
    G.l = arange(100) / 10
    seed(12345678)
    G.v1 = 'poisson(l)'
    seed(12345678)
    G.v2 = 'poisson(l)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert_allclose(G.v1[:], G.v2[:])


### 2. poisson in neurongroup stateupdater and mixed seed random/fixed
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_poisson_scalar_lambda_values_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

    G = NeuronGroup(10, 'dv/dt = -v/(10*ms) + 0.1*poisson(5)*xi/sqrt(ms) : 1')

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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_poisson_variable_lambda_values_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')


    G = NeuronGroup(10,
                    '''l : 1
                       dv/dt = -v/(10*ms) + 0.1*poisson(l)*xi/sqrt(ms) : 1''')
    G.l = arange(10)

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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 3. poisson in synapses set_conditional template
@pytest.mark.standalone_compatible
def test_poisson_scalar_lambda_values_set_synapses_random_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1''')
    S.connect()

    seed()
    S.v1 = 'poisson(5)'
    seed()
    S.v2 = 'poisson(5)'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert np.var(S.v1[:] - S.v2[:]) > 0


@pytest.mark.standalone_compatible
def test_poisson_variable_lambda_values_set_synapses_random_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1
                          l  : 1''')
    S.connect()
    S.l = arange(100) / 10

    seed()
    S.v1 = 'poisson(l)'
    seed()
    S.v2 = 'poisson(l)'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert np.var(S.v1[:] - S.v2[:]) > 0


@pytest.mark.standalone_compatible
def test_poisson_scalar_lambda_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1''')
    S.connect()

    seed(12345678)
    S.v1 = 'poisson(5)'
    seed(12345678)
    S.v2 = 'poisson(5)'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert_allclose(S.v1[:], S.v2[:])


@pytest.mark.standalone_compatible
def test_poisson_variable_lambda_values_set_synapses_fixed_seed():
    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G, '''v1 : 1
                          v2 : 1
                          l  : 1''')
    S.connect()
    S.l = arange(100) / 10

    seed(12345678)
    S.v1 = 'poisson(l)'
    seed(12345678)
    S.v2 = 'poisson(l)'
    run(0*ms)  # for standalone
    assert np.var(S.v1[:]) > 0
    assert np.var(S.v2[:]) > 0
    assert_allclose(S.v1[:], S.v2[:])


### 4. poisson in synapses stateupdater and mixing seed random/fixed
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_poisson_scalar_lambda_values_synapse_dynamics_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G,
                 '''dv/dt = -v/(10*ms) + 0.1*poisson(5)*xi/sqrt(ms) : 1''')
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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_poisson_variable_lambda_values_synapse_dynamics_fixed_and_random_seed():

    if prefs.core.default_float_dtype is np.float32:
        # TODO: Make test single-precision compatible, see #262
        pytest.skip('Need double precision for this test')

    G = NeuronGroup(10, 'z : 1')
    S = Synapses(G, G,
                 '''l : 1
                    dv/dt = -v/(10*ms) + 0.1*poisson(l)*xi/sqrt(ms) : 1''')
    S.connect()
    S.l = arange(100) / 10
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
    with pytest.raises(AssertionError):
        assert_allclose(first_run_values, second_run_values)
    # Third and fourth run should be identical (same seed)
    assert_allclose(third_run_values, fourth_run_values)


### 5. poisson in host side rng (synapses_init tempalte)
@pytest.mark.standalone_compatible
def test_poisson_values_init_synapses_fixed_and_random_seed():
    G = NeuronGroup(10, 'l : 1')
    G.l = arange(10)

    seed()
    S1 = Synapses(G, G)
    S1.connect('poisson(5) < poisson(l_post + l_pre)')

    seed(12345678)
    S2 = Synapses(G, G)
    S2.connect('poisson(5) < poisson(l_post + l_pre)')

    seed()
    S3 = Synapses(G, G)
    S3.connect('poisson(5) < poisson(l_post + l_pre)')

    seed(12345678)
    S4 = Synapses(G, G)
    S4.connect('poisson(5) < poisson(l_post + l_pre)')

    run(0*ms)  # for standalone

    idcs1 = np.hstack([S1.i[:], S1.j[:]])
    idcs2 = np.hstack([S2.i[:], S2.j[:]])
    idcs3 = np.hstack([S3.i[:], S3.j[:]])
    idcs4 = np.hstack([S4.i[:], S4.j[:]])

    # Pre/post idcs for first, second and third run should be different
    with pytest.raises(AssertionError):
        assert_equal(idcs1, idcs2)
    with pytest.raises(AssertionError):
        assert_equal(idcs1, idcs3)
    with pytest.raises(AssertionError):
        assert_equal(idcs2, idcs3)
    # Pre/post idcs for second and fourth run should be equal (same seed)
    assert_equal(idcs2, idcs4)


### Extra: group_set template (not conditional), only for neurongroup, not synapse
@pytest.mark.standalone_compatible
def test_poisson_scalar_lambda_set_template_random_seed():
    G = NeuronGroup(10, '''v1 : 1
                           v2 : 1''')

    seed()
    G.v1[:8] = 'rand() + randn() + poisson(5)'
    seed()
    G.v2[:8] = 'rand() + randn() + poisson(5)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@pytest.mark.standalone_compatible
def test_poisson_variable_lambda_set_template_random_seed():
    G = NeuronGroup(10, '''v1 : 1
                           v2 : 1
                           l  : 1''')

    G.l = arange(10)
    seed()
    G.v1[:8] = 'rand() + randn() + poisson(l)'
    seed()
    G.v2[:8] = 'rand() + randn() + poisson(l)'
    run(0*ms)  # for standalone
    assert np.var(G.v1[:]) > 0
    assert np.var(G.v2[:]) > 0
    assert np.var(G.v1[:] - G.v2[:]) > 0


@pytest.mark.standalone_compatible
def test_single_tick_rng_uneven_group_size():
    # cuRAND host API fails when generating an uneven number random numbers, make sure
    # we take care of that (just test for run not failing)
    G = NeuronGroup(9, '''dv1/dt = rand() * randn() * poisson(l) / ms : 1
                          dv2/dt = rand() * randn() * poisson(5) / ms : 1
                          l  : 1''')

    G.l = arange(9)
    G.v1[:7] = 'rand() + randn() + poisson(l)'
    G.v2[:7] = 'rand() + randn() + poisson(5)'
    run(defaultclock.dt)


if __name__ == '__main__':
    import brian2cuda
    from brian2cuda.tests.conftest import fake_randn
    for test in [
        test_rand_randn_regex,
        test_poisson_regex,
        test_rng_occurrence_counting,
        test_binomial_occurrence,
        test_rand,
        test_random_number_generation_with_multiple_runs,
        test_random_values_fixed_and_random_seed,
        test_poisson_scalar_values_fixed_and_random_seed,
        test_poisson_vectorized_values_fixed_and_random_seed,
        test_random_values_codeobject_every_tick,
        test_binomial_values,
        test_random_values_set_synapses_random_seed,
        test_random_values_set_synapses_fixed_seed,
        test_random_values_synapse_dynamics_fixed_and_random_seed,
        test_random_values_init_synapses_fixed_and_random_seed,
        test_binomial_values_random_seed,
        test_binomial_values_fixed_seed,
        test_binomial_values_fixed_and_random_seed,
        test_binomial_values_set_synapses_random_seed,
        test_binomial_values_set_synapses_fixed_seed,
        test_binomial_values_synapse_dynamics_fixed_and_random_seed,
        test_binomial_values_init_synapses_fixed_and_random_seed,
        test_random_binomial_set_template_random_seed,
        test_random_binomial_poisson_scalar_lambda_values_set_synapses_fixed_seed,
        test_random_binomial_poisson_variable_lambda_values_set_synapses_fixed_seed,
        test_poisson_scalar_lambda_values_random_seed,
        test_poisson_variable_lambda_values_random_seed,
        test_poisson_scalar_lambda_values_fixed_seed,
        test_poisson_variable_lambda_values_fixed_seed,
        test_poisson_scalar_lambda_values_fixed_and_random_seed,
        test_poisson_variable_lambda_values_fixed_and_random_seed,
        test_poisson_scalar_lambda_values_set_synapses_random_seed,
        test_poisson_variable_lambda_values_set_synapses_random_seed,
        test_poisson_scalar_lambda_values_set_synapses_fixed_seed,
        test_poisson_variable_lambda_values_set_synapses_fixed_seed,
        test_poisson_scalar_lambda_values_synapse_dynamics_fixed_and_random_seed,
        test_poisson_variable_lambda_values_synapse_dynamics_fixed_and_random_seed,
        test_poisson_values_init_synapses_fixed_and_random_seed,
        test_poisson_scalar_lambda_set_template_random_seed,
        test_poisson_variable_lambda_set_template_random_seed,
        test_single_tick_rng_uneven_group_size,
    ]:
        print()
        print(test.__name__)
        pytestmarks = [mark.name for mark in test.pytestmark]
        build_on_run = True
        if 'multiple_runs' in pytestmarks:
            build_on_run = False
        set_device('cuda_standalone', build_on_run=build_on_run, directory=None)
        test()

        reinit_and_delete()



### This is an attempt at dynamically generating these test functions. Should be
### possible like this but something wasn't working and I'm leaving this for another time.
## 1. poisson in neurongroup set_conditional templates
#rng_setups = [
#    # (rng_type, prepare_func, func, extra_vars)
#    ('rand', '', 'rand()', ''),
#    ('randn', '', 'randn()', ''),
#    (
#        'binomial',
#        'my_f = BinomialFunction(100, 0.1, approximate=False)',
#        'my_f()',
#        ''
#    ),
#    (
#        'binomial_approx',
#        'my_f = BinomialFunction(100, 0.1, approximate=True)',
#        'my_f()',
#        ''
#    ),
#    ('poisson_scalar_l', '', 'poisson(5)', ''),
#    ('poisson_variable_l', 'G.l = arange(10)', 'poisson(l)', 'l : 1')
#]
#
#seed_setup = {
#    'random': None,
#    'fixed': 13579
#}
#
#for rng_type, prepare_func, func, extra_vars in rng_setups:
#    for random_fixed, seed in seed_setup.items():
#        function_def = """
#@pytest.mark.standalone_compatible
##def test_{rng_type}_values_{random_fixed}_seed():
#    G = NeuronGroup(100, '''v1 : 1
#                            v2 : 1
#                            {extra_vars}''')
#    {prepare_func}
#    seed({seed})
#    G.v1 = \'{func}\'
#    seed({seed})
#    G.v2 = '{func}'
#    run(0*ms)
#    assert np.var(G.v1[:]) > 0
#    assert np.var(G.v2[:]) > 0
#    if '{random_fixed}' == 'random':
#        assert np.var(G.v1[:] - G.v2[:]) > 0
#    else:
#        assert_allclose(G.v1[:], G.v2[:])
#""".format(rng_type=rng_type, prepare_func=prepare_func, func=func,
#           extra_vars=extra_vars, random_fixed=random_fixed, seed=seed)
#
#        # define function from string via `exec`
#        print(function_def)
#        exec(function_def)
#
#
#### 2. binomial in neurongroup stateupdater and mixed seed random/fixed
#function_def = """
#@pytest.mark.standalone_compatible
#@pytest.mark.multiple_runs
##def test_{rng_type}_values_fixed_and_random_seed():
#
#    # e.g. `my_f = BinomialFunction(...)`
#    {prepare_func}
#
#    G = NeuronGroup(10,
#                    '''{extra_vars}  # e.g. `l : 1`
#                       dv/dt = -v/(10*ms) + 0.1*({func})/ms : 1''')
#
#    # e.g. `G.l = arange(10)`
#    {init_vars}
#
#    mon = StateMonitor(G, 'v', record=True)
#
#    # first run
#    G.v = 0
#    seed()
#    run(2*defaultclock.dt)
#
#    # second run
#    G.v = 0
#    seed()
#    run(2*defaultclock.dt)
#
#    # third run
#    G.v = 0
#    seed(13579)
#    run(2*defaultclock.dt)
#
#    # fourth run
#    G.v = 0
#    seed(13579)
#    run(2*defaultclock.dt)
#
#    device.build(direct_call=False, **device.build_options)
#
#    first_run_values = np.array(mon.v[:, [0, 1]])
#    second_run_values = np.array(mon.v[:, [2, 3]])
#    third_run_values = np.array(mon.v[:, [4, 5]])
#    fourth_run_values = np.array(mon.v[:, [6, 7]])
#
#    # First and second run should be different (random seed)
#    with pytest.raises(AssertionError):
# 	     assert_allclose(first_run_values, second_run_values)
#    # Third and fourth run should be identical (same seed)
#    assert_allclose(third_run_values, fourth_run_values)
#"""
