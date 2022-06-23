import pytest
from numpy.testing import assert_equal, assert_array_equal

from brian2 import *
from brian2.tests.utils import assert_allclose
from brian2.devices.device import reinit_devices, set_device

import brian2cuda


@pytest.mark.standalone_compatible
def test_CudaSpikeQueue_push_outer_loop():

    # This test only works, if the CudaSpikeQueue::push() kernel is called with
    # <<<num_blocks, num_threads>>> as specified below
    # In each block for the last syn, we will enter the second loop cycle with a different delay.
    # In the example in spikequeue.h this triggers situations (a) for syn=512 and (b) for syn=1024.
    # syn                (0-512/3)   (512/3-1023) 1024
    # delay                     0              1     2
    # tid                (0-512/3)   (512/3-1023)    0
    # i                         0              0  1024
    # loop cycle                0              0     1
    # delay_start_idx           0            512   512

    num_blocks = 15
    num_threads = 1024
    N = (num_threads + 1) * num_blocks # each block gets 1025 synapses to push

    default_dt = defaultclock.dt
    delays = arange(N)%2 * default_dt

    inp = SpikeGeneratorGroup(1, [0], [0]*ms)
    G = NeuronGroup(N, 'v:1', threshold='v>1', reset='v=0')
    S = Synapses(inp, G, on_pre='v+=1.1')
    S.connect()
    S.delay = delays  # delays 0, dt
    mon = SpikeMonitor(G)

    run(3*default_dt)

    even_indices = np.arange(0,N,2)  # delay=0
    odd_indices = np.arange(1,N,2)  # delay=dt

    assert_allclose(S.delay[even_indices], 0*second)
    assert_allclose(S.delay[odd_indices], default_dt)
    assert_equal(sorted(mon.i[mon.t==default_dt]), even_indices)
    assert_equal(sorted(mon.i[mon.t==2*default_dt]), odd_indices)
    assert len(mon) == N


@pytest.mark.standalone_compatible
def test_CudaSpikeQueue_push_outer_loop2():

    # This test was testing a special case scenario in an old version of
    # CudaSpikeQueue::push(). It is now left here as a sanity check for spike
    # propagation.

    # This test creates a situation where one pre neuron is connected to 2048
    # post neurons, where all synapses have different delays except of the the
    # two synapses with ids 1023 and 1024 (last thread of first loop cycle and
    # first thread of last loop cycle in CudaSpikeQueue::push()), which have
    # the same delay. Additionaly all delay queues have different size when the
    # pre neuron spikes one time. The different delay queue sizes is just a
    # leftover of another test and does not have any effect on this test, but
    # I'll leave it as an example.

    num_blocks = 1
    prefs['devices.cuda_standalone.parallel_blocks'] = num_blocks
    # make sure we don't use less then 1024 threads due to register usage
    prefs['devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc'] += ['-maxrregcount=64']

    threads_per_block = 1024
    neurons_per_block = 2 * threads_per_block
    N = neurons_per_block * num_blocks # each block gets 2048 synapses to push

    default_dt = defaultclock.dt
    # for the first connection each thread has a different delay per block of
    # postsynatpic neurons
    delays_one_block = arange(neurons_per_block) * default_dt
    delays0 = tile(delays_one_block, num_blocks)
    # for the second connection we want to trigger situation (a), so per block the
    # last thread of first loop and first thread of last loop have the same delay
    delays_one_block[threads_per_block] = delays_one_block[threads_per_block - 1]
    delays = tile(delays_one_block, num_blocks)

    # inp neuron 0 fires the first (2*threads_per_block) time steps, s.t. all queues
    # have different size after that (first 2048, last 0)
    # inp neuron 1 fires once after that
    indices = zeros(neurons_per_block + 1)
    indices[-1] = 1
    times = arange(neurons_per_block + 1) * default_dt
    inp = SpikeGeneratorGroup(2, indices, times)
    G = NeuronGroup(N, 'v:1', threshold='v>1', reset='v=0')

    S = Synapses(inp, G, 'w:1', on_pre='v+=w')
    S.connect()
    # inp neuron 0 has no effect on target neurons (weight 0)
    S.w['i == 0'] = 0
    # inp neuron 1 makes G neurons spike
    S.w['i == 1'] = 1.1
    # delays from inp neuron 0 are equal to post neuron number j
    S.delay['i == 0'] = '(j % neurons_per_block) * default_dt'  # arange(2048) per block
    # delays from inp neuron 0 are the same except of the delay for the synapse
    # of the thread of the second loop in the spikequeue.push(), which is the
    # same as the previous synapse
    S.delay['i == 1'] = '(j % neurons_per_block) * default_dt' # arange(2048), but [..., 1023, 1023, 1025, ...]
    S.delay['i == 1 and j % neurons_per_block == threads_per_block'] = '(threads_per_block - 1) * default_dt'

    mon = SpikeMonitor(G)

    # first neurons_per_block time steps we resize queues
    # at time step neurons_per_block + 1 we send a spike that triggeres
    # postsynaptic spikes with delays = arange(neurons_per_block), but [..., 1023, 1023, 1035, ...]
    # then it takes neurons_per_block - 1 time steps for all effects to be applied
    time_steps = 2 * neurons_per_block + 2
    run(time_steps * default_dt)

    assert_allclose(S.delay[0, :], delays0)
    assert_allclose(S.delay[1, :], delays)

    assert_equal(len(mon), N)

    unique_mon_t = unique(mon.t)
    for n in range(time_steps):
        t = n * default_dt
        if n <= neurons_per_block or n == time_steps - 1:
            # no spikes
            assert t not in unique_mon_t
        else:
            try:
                mon_idx = sort(mon.i[mon.t == t])
                # first neuron (idx 0) spikes at time step neurons_per_block + 1
                spiking_neuron_idx = n - (neurons_per_block + 1)
                # neuron indices increment over blocks, get idx starting at 0 per block
                mon_idx_per_block = mod(mon_idx, neurons_per_block)
                assert_array_equal(sort(mon_idx), mon_idx)
                if spiking_neuron_idx == threads_per_block - 1:  # 1023
                    # [1023, 1024, 1023, 1024, ...]
                    expected_indices = tile([spiking_neuron_idx, spiking_neuron_idx+1], num_blocks)
                elif spiking_neuron_idx == threads_per_block:  # 1024
                    # []
                    expected_indices = array([])
                else:
                    # [spiking_neuron_idx, spiking_neuron_idx, ...]
                    expected_indices = tile([spiking_neuron_idx], num_blocks)
                assert_equal(mon_idx_per_block, expected_indices)
            except AssertionError:
                print('n =', n)
                print(mon_idx)
                raise


@pytest.mark.standalone_compatible
def test_circular_eventspaces():
    # test postsynaptic effects in no_or_const_delay_mode are applied correctly

    default_dt = defaultclock.dt

    inp = SpikeGeneratorGroup(1, [0], [0]*ms)
    G = NeuronGroup(5, 'v:1', threshold='v>1', reset='v=0')
    # synapses with homogenous delays
    S0 = Synapses(inp, G, on_pre='v+=1.1', delay=0*ms)
    S0.connect(i=0, j=0)
    S1 = Synapses(inp, G, on_pre='v+=1.1', delay=2*default_dt)
    S1.connect(i=0, j=1)
    S2 = Synapses(inp, G, on_pre='v+=1.1', delay=5*default_dt)
    S2.connect(i=0, j=2)
    # synapse with heterogeneous delays
    S3 = Synapses(inp, G, on_pre='v+=1.1')
    S3.connect(i=0, j=[3, 4])
    S3.delay = 'j*default_dt'
    mon = SpikeMonitor(G)

    run(6*default_dt)

    # neurons should spike in the timestep after effect application
    assert_allclose(mon.t[mon.i[:] == 0], 1*default_dt)
    assert_allclose(mon.t[mon.i[:] == 1], 3*default_dt)
    assert_allclose(mon.t[mon.i[:] == 2], 6*default_dt)
    assert_allclose(mon.t[mon.i[:] == 3], 4*default_dt)
    assert_allclose(mon.t[mon.i[:] == 4], 5*default_dt)


@pytest.mark.standalone_compatible
def test_circular_eventspaces_spikegenerator():
    # same test as test_circular_eventspaces() but with a SpikeGeneratorGroup spiking
    # for multiple time steps

    # Change reset to be before synapses, else the NeuronGroup only spikes every second
    # time step (see https://github.com/brian-team/brian2/issues/1332)
    prefs.core.network.default_schedule = ['start', 'groups', 'thresholds', 'resets', 'synapses', 'end']

    default_dt = defaultclock.dt

    # inp neuron 0 spikes every time step for the first n_timesteps
    n_timesteps = 12
    indices = [0] * n_timesteps
    times = arange(n_timesteps) * default_dt
    inp = SpikeGeneratorGroup(1, indices, times)
    # G neurons spike for each incoming spike
    G = NeuronGroup(5, 'v:1', threshold='v>1', reset='v=0')
    # synapses with homogenous delays
    S0 = Synapses(inp, G, on_pre='v+=1.1', delay=0*ms)
    S0.connect(i=0, j=0)
    S1 = Synapses(inp, G, on_pre='v+=1.1', delay=2*default_dt)
    S1.connect(i=0, j=1)
    S2 = Synapses(inp, G, on_pre='v+=1.1', delay=5*default_dt)
    S2.connect(i=0, j=2)
    # synapse with heterogeneous delays
    S3 = Synapses(inp, G, on_pre='v+=1.1')
    S3.connect(i=0, j=[3, 4])
    S3.delay = 'j*default_dt'
    mon = SpikeMonitor(G)

    run((n_timesteps + 6) * default_dt)

    # neurons should spike in the timestep after effect application
    # TODO: remove sorted() when #46 is fixed
    assert_allclose(sorted(mon.t[mon.i[:] == 0]), arange(1, n_timesteps + 1) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 1]), arange(3, n_timesteps + 3) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 2]), arange(6, n_timesteps + 6) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 3]), arange(4, n_timesteps + 4) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 4]), arange(5, n_timesteps + 5) * default_dt)


@pytest.mark.xfail(
    reason='See Brian2CUDA issue #222',
    condition="config.device == 'cuda_standalone'",
    raises=AssertionError
)
@pytest.mark.standalone_compatible
def test_circular_eventspaces_different_clock_spikegenerator():
    # same test as test_circular_eventspaces_spikegenerator() but with a
    # SpikeGeneratorGroup on a different clock (dt=2*defaultclock.dt)

    default_dt = defaultclock.dt

    # Neuron 0 spikes every second time step in first n_timesteps
    clock_multiplier = 2  # factor by which SpikeGeneratorGroup clock is slower
    n_timesteps = 12
    indices = [0] * (n_timesteps // clock_multiplier)
    times = arange(0, n_timesteps, clock_multiplier) * default_dt
    inp = SpikeGeneratorGroup(1, indices, times, dt=2*default_dt)
    G = NeuronGroup(5, 'v:1', threshold='v>1', reset='v=0')
    # synapses with homogenous delays
    S0 = Synapses(inp, G, on_pre='v+=1.1', delay=0*ms)
    S0.connect(i=0, j=0)
    S1 = Synapses(inp, G, on_pre='v+=1.1', delay=2*default_dt)
    S1.connect(i=0, j=1)
    S2 = Synapses(inp, G, on_pre='v+=1.1', delay=4*default_dt)
    S2.connect(i=0, j=2)
    # synapse with heterogeneous delays
    S3 = Synapses(inp, G, on_pre='v+=1.1')
    S3.connect(i=0, j=[3, 4])  # delays: 6, 8
    S3.delay = '2*j*default_dt'
    mon = SpikeMonitor(G)

    run((n_timesteps + 9) * default_dt)

    # neurons should spike in the timestep after effect application
    # TODO: remove sorted() when #46 is fixed
    assert_allclose(sorted(mon.t[mon.i[:] == 0]), arange(1, n_timesteps + 1, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 1]), arange(3, n_timesteps + 3, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 2]), arange(5, n_timesteps + 5, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 3]), arange(7, n_timesteps + 7, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 4]), arange(9, n_timesteps + 9, clock_multiplier) * default_dt)


@pytest.mark.xfail(
    reason='See Brian2CUDA issue #222',
    condition="config.device == 'cuda_standalone'",
    raises=AssertionError
)
@pytest.mark.standalone_compatible
def test_circular_eventspaces_different_clock_neurongroup():
    # same test as test_circular_eventspaces_different_clock_spikegenerator() but with a
    # NeuronGroup instead of SpikeGeneratorGroup (also on a different clock:
    # dt=2*defaultclock.dt)

    default_dt = defaultclock.dt

    # Neuron 0 spikes every second time step in first n_timesteps
    clock_multiplier = 2  # factor by which SpikeGeneratorGroup clock is slower
    n_timesteps = 12
    inp = NeuronGroup(
        1,
        'default_t_in_timesteps = timestep(t, default_dt): 1',
        threshold='default_t_in_timesteps % clock_multiplier == 0 and default_t_in_timesteps < n_timesteps',
        dt=2*default_dt,
    )
    G = NeuronGroup(5, 'v:1', threshold='v>1', reset='v=0')
    # synapses with homogenous delays
    S0 = Synapses(inp, G, on_pre='v+=1.1', delay=0*ms)
    S0.connect(i=0, j=0)
    S1 = Synapses(inp, G, on_pre='v+=1.1', delay=2*default_dt)
    S1.connect(i=0, j=1)
    S2 = Synapses(inp, G, on_pre='v+=1.1', delay=4*default_dt)
    S2.connect(i=0, j=2)
    # synapse with heterogeneous delays
    S3 = Synapses(inp, G, on_pre='v+=1.1')
    S3.connect(i=0, j=[3, 4])  # delays: 6, 8
    S3.delay = '2*j*default_dt'
    mon = SpikeMonitor(G)

    run((n_timesteps + 9) * default_dt)

    # neurons should spike in the timestep after effect application
    # TODO: remove sorted() when #46 is fixed
    assert_allclose(sorted(mon.t[mon.i[:] == 0]), arange(1, n_timesteps + 1, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 1]), arange(3, n_timesteps + 3, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 2]), arange(5, n_timesteps + 5, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 3]), arange(7, n_timesteps + 7, clock_multiplier) * default_dt)
    assert_allclose(sorted(mon.t[mon.i[:] == 4]), arange(9, n_timesteps + 9, clock_multiplier) * default_dt)


@pytest.mark.standalone_compatible
def test_synaptic_effect_modes():
    # make sure on_pre pathway changing pre variables has the same mode as
    # on_post pathway changing post variables

    num_blocks = 15
    num_threads = 1024
    N = num_blocks * num_threads
    # all neurons spike every timestep
    neuron = NeuronGroup(1, 'v:1', threshold='True')
    group = NeuronGroup(N, 'v:1', threshold='True')
    # the on_pre pathway does not serialize in syanptic_effects == 'target' mode
    # with no_or_const_delay_mode
    S0 = Synapses(group, neuron, on_pre='v_post+=1', delay=0*ms)
    S0.connect()
    # the on_post pathway serializes entirely in synaptic_effects == 'source' mode
    S1 = Synapses(group, neuron, on_post='v_post-=1')
    S1.connect()
    S1.post.delay = 0*ms
    mon = SpikeMonitor(neuron, variables=['v'])

    run(2*defaultclock.dt)

    # each timestep there are the same number of + and - to neuron.v
    assert_allclose(mon.v[:], 0)


@pytest.mark.parametrize(
    'expression',
    [
        'ceil({mean} + 2*{std})',
        '{max}',
        '{min}',
        '5',
    ]
)
@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_threads_per_synapse_bundle_prefs(expression):

        set_device('cuda_standalone', directory=None, with_output=False)

        prefs.devices.cuda_standalone.threads_per_synapse_bundle = expression

        default_dt = defaultclock.dt

        inp = SpikeGeneratorGroup(1, [0], [0]*ms)
        G = NeuronGroup(5, 'v:1', threshold='v>1', reset='v=0')
        # synapse with heterogeneous delays
        S = Synapses(inp, G, on_pre='v+=1.1')
        S.connect(i=0, j=[3, 4])
        S.delay = 'j*default_dt'
        mon = SpikeMonitor(G)

        run(6*default_dt)

        # neurons should spike in the timestep after effect application
        assert_allclose(mon.t[mon.i[:] == 3], 4*default_dt)
        assert_allclose(mon.t[mon.i[:] == 4], 5*default_dt)
