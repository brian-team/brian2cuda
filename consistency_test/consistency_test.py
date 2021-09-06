import pytest
from brian2 import *
from brian2.tests.utils import assert_allclose
from brian2.devices.device import reinit_devices, set_device, reset_device
import brian2cuda
import numpy as np
import random as py_random

np.random.seed(123)
py_random.seed(123)

@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_mushroom_body():

    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        # configuration options independent of device
        duration = 10 * second

        # Number of neurons
        N_AL = 100
        N_MB = 2500
        N_LB = 100
        # Constants
        g_Na = 7.15 * uS
        E_Na = 50 * mV
        g_K = 1.43 * uS
        E_K = -95 * mV
        g_leak = 0.0267 * uS
        E_leak = -63.56 * mV
        C = 0.3 * nF
        VT = -63 * mV
        # Those two constants are dummy constants, only used when populations only have
        # either inhibitory or excitatory inputs
        E_e = 0 * mV
        E_i = -92 * mV
        # Actual constants used for synapses
        NKCKC = N_MB
        if NKCKC > 10000:
            NKCKC = 10000
        g_scaling = NKCKC / 2500
        if g_scaling < 1:
            g_scaling = 1
        tau_PN_LHI = 1 * ms
        tau_LHI_iKC = 3 * ms
        tau_PN_iKC = 2 * ms
        tau_iKC_eKC = 10 * ms
        tau_eKC_eKC = 5 * ms
        w_LHI_iKC = 8.75 * nS
        w_eKC_eKC = 75 * nS
        tau_pre = tau_post = 10 * ms
        dApre = 0.1 * nS / g_scaling
        dApost = -dApre
        g_max = 3.75 * nS / g_scaling

        scale = .675

        traub_miles = '''
            dV/dt = -(1/C)*(g_Na*m**3*h*(V - E_Na) +
                            g_K*n**4*(V - E_K) +
                            g_leak*(V - E_leak) +
                            I_syn) : volt
            dm/dt = alpha_m*(1 - m) - beta_m*m : 1
            dn/dt = alpha_n*(1 - n) - beta_n*n : 1
            dh/dt = alpha_h*(1 - h) - beta_h*h : 1
            alpha_m = 0.32*(mV**-1)*(13*mV-V+VT)/
                     (exp((13*mV-V+VT)/(4*mV))-1.)/ms : Hz
            beta_m = 0.28*(mV**-1)*(V-VT-40*mV)/
                    (exp((V-VT-40*mV)/(5*mV))-1)/ms : Hz
            alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
            beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
            alpha_n = 0.032*(mV**-1)*(15*mV-V+VT)/
                     (exp((15*mV-V+VT)/(5*mV))-1.)/ms : Hz
            beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
            '''

        # Principal neurons (Antennal Lobe)
        n_patterns = 10
        n_repeats = int(duration / second * 10)
        p_perturb = 0.1

        patterns = np.repeat(
            np.array([np.random.choice(N_AL, int(0.2 * N_AL), replace=False) for _ in range(n_patterns)]),
            n_repeats, axis=0)

        # Make variants of the patterns
        to_replace = np.random.binomial(int(0.2 * N_AL), p=p_perturb, size=n_patterns * n_repeats)

        variants = []
        for idx, variant in enumerate(patterns):
            np.random.shuffle(variant)
            if to_replace[idx] > 0:
                variant = variant[:-to_replace[idx]]
            new_indices = np.random.randint(N_AL, size=to_replace[idx])
            variant = np.unique(np.concatenate([variant, new_indices]))
            variants.append(variant)

        training_size = (n_repeats - 10)
        training_variants = []
        for p in range(n_patterns):
            training_variants.extend(variants[n_repeats * p:n_repeats * p + training_size])
        py_random.shuffle(training_variants)
        sorted_variants = list(training_variants)
        for p in range(n_patterns):
            sorted_variants.extend(variants[n_repeats * p + training_size:n_repeats * (p + 1)])

        spike_time_randomness = rand(n_patterns * n_repeats) * 2 * ms

        spike_times = np.arange(n_patterns * n_repeats) * 50 * ms + 1 * ms + spike_time_randomness
        spike_times = spike_times.repeat([len(p) for p in sorted_variants])
        spike_indices = np.concatenate(sorted_variants)

        PN = SpikeGeneratorGroup(N_AL, spike_indices, spike_times)

        # iKC of the mushroom body
        I_syn = '''I_syn = g_PN_iKC*(V - E_e): amp
                   dg_PN_iKC/dt = -g_PN_iKC/tau_PN_iKC : siemens'''
        eqs_iKC = Equations(traub_miles) + Equations(I_syn)
        iKC = NeuronGroup(N_MB, eqs_iKC, threshold='V>0*mV', refractory='V>0*mV',
                          method='exponential_euler')

        # eKCs of the mushroom body lobe
        I_syn = '''I_syn = g_iKC_eKC*(V - E_e) + g_eKC_eKC*(V - E_i): amp
                   dg_iKC_eKC/dt = -g_iKC_eKC/tau_iKC_eKC : siemens
                   dg_eKC_eKC/dt = -g_eKC_eKC/tau_eKC_eKC : siemens'''
        eqs_eKC = Equations(traub_miles) + Equations(I_syn)
        eKC = NeuronGroup(N_LB, eqs_eKC, threshold='V>0*mV', refractory='V>0*mV',
                          method='exponential_euler')

        # Synapses
        PN_iKC = Synapses(PN, iKC, 'weight : siemens', on_pre='g_PN_iKC += scale*weight')
        iKC_eKC = Synapses(iKC, eKC,
                           '''g_raw : siemens
                              dApre/dt = -Apre / tau_pre : siemens (event-driven)
                              dApost/dt = -Apost / tau_post : siemens (event-driven)
                              ''',
                           on_pre='''g_iKC_eKC += g_raw
                                     Apre += dApre
                                     g_raw = clip(g_raw + Apost, 0, g_max)
                                     ''',
                           on_post='''
                                      Apost += dApost
                                      g_raw = clip(g_raw + Apre, 0, g_max)''',
                           delay=0 * ms)
        eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC', delay=0 * ms)
        # bu.insert_benchmark_point()
        pn_ikc_max_synapses = N_AL * N_MB
        p_e_array = TimedArray(np.random.rand(1, pn_ikc_max_synapses), dt=duration)
        # PN_iKC.connect(p=0.15)
        PN_iKC.connect('p_e_array(0*ms, i)<0.15')

        if (N_MB > 10000):
            iKC_eKC.connect(p=float(10000) / N_MB)
        else:
            iKC_eKC.connect()
        eKC_eKC.connect()
        # bu.insert_benchmark_point()

        # First set all synapses as "inactive", then set 20% to active
        pn_ikc_array = TimedArray(np.random.randn(1, pn_ikc_max_synapses), dt=duration)
        PN_iKC.weight = '10*nS + 1.25*nS*pn_ikc_array(0*ms, i + j*N_pre)'
        # PN_iKC.weight = '10*nS + 1.25*nS*randn()'

        ikc_ekc_max_synapses = N_MB * N_LB
        ikc_ekc_array1 = TimedArray(np.random.rand(1, ikc_ekc_max_synapses), dt=duration)
        iKC_eKC.g_raw = 'ikc_ekc_array1(0*ms, i +j*N_pre)*g_max/10/g_scaling'
        # iKC_eKC.g_raw = 'rand()*g_max/10/g_scaling'
        ikc_ekc_array2 = TimedArray(np.random.rand(1, ikc_ekc_max_synapses), dt=duration)
        ikc_ekc_array3 = TimedArray(np.random.randn(1, ikc_ekc_max_synapses), dt=duration)
        iKC_eKC.g_raw[
            'ikc_ekc_array2(0*ms, i+j*N_pre) < 0.2'] = '(2.5*nS + 0.5*nS*ikc_ekc_array3(0*ms, i+j*N_pre))/g_scaling'
        # iKC_eKC.g_raw['rand() < 0.2'] = '(2.5*nS + 0.5*nS*randn())/g_scaling'
        iKC.V = E_leak
        iKC.h = 1
        iKC.m = 0
        iKC.n = .5
        eKC.V = E_leak
        eKC.h = 1
        eKC.m = 0
        eKC.n = .5

        # if use_spikemon:
        PN_spikes = SpikeMonitor(PN)
        iKC_spikes = SpikeMonitor(iKC)
        eKC_spikes = SpikeMonitor(eKC)

        run(duration)

        device.build(directory=None, with_output=False)

        results[devicename] = {}
        results[devicename]['PN_spikes'] = PN_spikes.num_spikes
        results[devicename]['iKC_spikes'] = iKC_spikes.num_spikes
        results[devicename]['eKC_spikes'] = eKC_spikes.num_spikes

    for key in ['PN_spikes', 'iKC_spikes', 'eKC_spikes']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_brunel_hakim_homog_delays():

    previous_device = get_device()

    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        # configuration options
        duration = 10 * second

        # need to be set in child class
        sigmaext = None  # vold

        # heterogeneous delays
        heterog_delays = None  # string syntax

        name = "BrunelHakimwithHomogeneous"
        tags = ["Neurons", "Synapses", "Delays"]

        # all delays 2 ms
        homog_delays = 2 * ms

        sigmaext = 1 * mV
        muext = 25 * mV

        assert not (heterog_delays is not None and
                    homog_delays is not None), \
            "Can't set homog_delays and heterog_delays"
        Vr = 10 * mV
        theta = 20 * mV
        tau = 20 * ms
        delta = 2 * ms
        taurefr = 2 * ms
        C = 1000
        n = 100
        sparseness = float(C) / n
        J = .1 * mV
        muext = muext
        sigmaext = sigmaext
        num_neurons = n
        num_time_steps = int(duration // defaultclock.dt)
        custom_xi = TimedArray(
            np.random.rand(num_time_steps, num_neurons) / np.sqrt(defaultclock.dt),
            dt=defaultclock.dt
        )

        eqs = """
        dV/dt = (-V+muext + sigmaext * sqrt(tau) * custom_xi(t, i))/tau : volt
        """

        group = NeuronGroup(n, eqs, threshold='V>theta',
                            reset='V=Vr', refractory=taurefr)
        group.V = Vr

        conn = Synapses(group, group, on_pre='V += -J',
                        delay=homog_delays)
        conn.connect(p=sparseness)

        if heterog_delays is not None:
            assert homog_delays is None
            conn.delay = heterog_delays

        M = SpikeMonitor(group)
        LFP = PopulationRateMonitor(group)

        run(duration)

        device.build(directory=None, with_output=False)

        results[devicename] = {}
        results[devicename]['M_spikes'] = M.num_spikes
        results[devicename]['LFP_rate'] = LFP.rate[:]

        print("Values for {}".format(devicename))
        print(results[devicename]['M_spikes'])
        print(results[devicename]['LFP_rate'])

    for key in ['M_spikes', 'LFP_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_brunel_hakim_hetero_delays():
    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        duration = 10 * second
        n = 1000

        name = "BrunelHakimwithheterogeneousdelaysuniform"
        tags = ["Neurons", "Synapses", "Delays"]

        # delays [0, 4] ms

        r_array = TimedArray(np.random.rand(1, n), dt=duration)
        heterog_delays = "4*ms * r_array(0*ms,i)"
        # heterog_delays = "4*ms * rand()"

        # homogeneous delays
        homog_delays = None  # second

        # to have a similar network activity regime as for homogenous delays
        # or narrow delay distribution
        sigmaext = 0.33 * mV
        muext = 27 * mV

        assert not (heterog_delays is not None and
                    homog_delays is not None), \
            "Can't set homog_delays and heterog_delays"
        Vr = 10 * mV
        theta = 20 * mV
        tau = 20 * ms
        delta = 2 * ms
        taurefr = 2 * ms
        C = 1000
        sparseness = float(C) / n
        J = .1 * mV
        muext = muext
        sigmaext = sigmaext

        num_neurons = n
        num_time_steps = int(duration // defaultclock.dt)
        custom_xi = TimedArray(
            np.random.rand(num_time_steps, num_neurons) / np.sqrt(defaultclock.dt),
            dt=defaultclock.dt
        )

        eqs = """
        dV/dt = (-V+muext + sigmaext * sqrt(tau) * custom_xi(t, i))/tau : volt
        """
        # eqs = """
        # dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
        # """

        group = NeuronGroup(n, eqs, threshold='V>theta',
                            reset='V=Vr', refractory=taurefr)
        group.V = Vr

        conn = Synapses(group, group, on_pre='V += -J',
                        delay=homog_delays)

        max_num_synapses = n * n
        p_e_array = TimedArray(np.random.rand(1, max_num_synapses), dt=duration)
        conn.connect('p_e_array(0*ms, i)<sparseness')
        # conn.connect(p=sparseness)

        if heterog_delays is not None:
            assert homog_delays is None
            conn.delay = heterog_delays

        M = SpikeMonitor(group)
        LFP = PopulationRateMonitor(group)

        run(duration)
        device.build(directory=None, with_output=False)

        results[devicename] = {}
        results[devicename]['M_spikes'] = M.num_spikes
        results[devicename]['LFP_rate'] = LFP.rate[:]

        print("Values for {}".format(devicename))
        print(results[devicename]['M_spikes'])
        print(results[devicename]['LFP_rate'])

    for key in ['M_spikes', 'LFP_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_brunel_hakim_hetero_delays_narrow_dist():
    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        n = 100
        duration = 10 * second

        # delays 2 ms += dt
        r_array = TimedArray(np.random.rand(1, n), dt=duration)
        heterog_delays = "2*ms + 2 * dt * r_array(0*ms,i) - dt"
        # heterog_delays = "2*ms + 2 * dt * rand() - dt"

        sigmaext = 1 * mV
        muext = 25 * mV

        homog_delays = None  # second

        assert not (heterog_delays is not None and
                    homog_delays is not None), \
            "Can't set homog_delays and heterog_delays"
        Vr = 10 * mV
        theta = 20 * mV
        tau = 20 * ms
        delta = 2 * ms
        taurefr = 2 * ms
        C = 1000
        sparseness = float(C) / n
        J = .1 * mV
        muext = muext
        sigmaext = sigmaext
        num_neurons = n
        num_time_steps = int(duration // defaultclock.dt)
        custom_xi = TimedArray(
            np.random.rand(num_time_steps, num_neurons) / np.sqrt(defaultclock.dt),
            dt=defaultclock.dt
        )

        eqs = """
        dV/dt = (-V+muext + sigmaext * sqrt(tau) * custom_xi(t, i))/tau : volt
        """

        group = NeuronGroup(n, eqs, threshold='V>theta',
                            reset='V=Vr', refractory=taurefr)
        group.V = Vr

        conn = Synapses(group, group, on_pre='V += -J',
                        delay=homog_delays)
        conn.connect(p=sparseness)

        if heterog_delays is not None:
            assert homog_delays is None
            conn.delay = heterog_delays

        run(duration)
        device.build(directory=None, with_output=False)

        M = SpikeMonitor(group)
        LFP = PopulationRateMonitor(group)

        results[devicename] = {}
        results[devicename]['M_spikes'] = M.num_spikes
        results[devicename]['LFP_rate'] = LFP.rate[:]

        print("Values for {}".format(devicename))
        print(results[devicename]['M_spikes'])
        print(results[devicename]['LFP_rate'])

    for key in ['M_spikes', 'LFP_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cpp_standalone
@pytest.mark.cuda_standalone
def test_cobahh():
    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        duration = 10 * second

        uncoupled = False

        # fixed connectivity: 1000 neurons per synapse
        n = 4000
        p = lambda n: 1000. / n

        # weights set to tiny values, s.t. they are effectively zero but don't
        # result in compiler optimisations
        we_max_synapses = n * n
        we_array = TimedArray(np.random.rand(1, we_max_synapses), dt=duration)
        we = 'we_array(0*ms,i) * 1e-9*nS'
        # we = 'rand() * 1e-9*nS'

        wi_max_synapses = n * n
        wi_array = TimedArray(np.random.rand(1, wi_max_synapses), dt=duration)
        wi = 'wi_array(0*ms,i) * 1e-9*nS'
        # wi = 'rand() * 1e-9*nS'

        # Parameters
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV

        # Time constants
        taue = 5 * ms
        taui = 10 * ms

        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        n_recorded = 10

        P = NeuronGroup(n, model=eqs, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')

        if not uncoupled:
            num_exc = int(0.8 * n)
            Pe = P[:num_exc]
            Pi = P[num_exc:]
            Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we')
            Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi')

            # connection probability p can depend on network size n
            ce_max_synapses = n * n
            ce_array = TimedArray(np.random.rand(1, ce_max_synapses), dt=duration)
            Ce.connect('ce_array(0*ms, i)<1000./n')
            # Ce.connect(str(p(n)))

            ci_max_synapses = n * n
            ci_array = TimedArray(np.random.rand(1, ci_max_synapses), dt=duration)
            Ci.connect('ci_array(0*ms, i)<1000./n')
            # Ci.connect(str(p(n)))
            Ce.we = we  # excitatory synaptic weight
            Ci.wi = wi  # inhibitory synaptic weight

        # Initialization
        p_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.v = 'El + (p_array(0*ms, i) * 5 - 5)*mV'
        # P.v = 'El + (randn() * 5 - 5)*mV'

        p_ge_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.ge = '(p_ge_array(0*ms, i) * 1.5 + 4) * 10.*nS'
        # P.ge = '(randn() * 1.5 + 4) * 10.*nS'

        p_gi_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.gi = '(p_gi_array(0*ms, i) * 12 + 20) * 10.*nS'
        # P.gi = '(randn() * 12 + 20) * 10.*nS'

        spikemon = SpikeMonitor(P)
        popratemon = PopulationRateMonitor(P)
        # state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1 * second)
        trace = StateMonitor(P, 'v', record=range(n_recorded))

        run(duration)

        results[devicename] = {}
        results[devicename]['spikemon_spikes'] = spikemon.num_spikes
        results[devicename]['popratemon_rate'] = popratemon.rate[:]
        results[devicename]['trace_w'] = trace.w

        print("Values for {}".format(devicename))
        print(results[devicename]['spikemon_spikes'])
        print(results[devicename]['popratemon_rate'])
        print(results[devicename]['trace_w'])

    for key in ['spikemon_spikes', 'trace_w', 'popratemon_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cpp_standalone
@pytest.mark.cuda_standalone
def test_cobahh_pseudo_coupled_80():
    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        duration = 10 * second

        uncoupled = False
        n = 4000

        # fixed connectivity: 80 neurons per synapse
        p = lambda n: str(80. / n)
        # weights set to tiny values, s.t. they are effectively zero but don't
        # result in compiler optimisations
        we_max_synapses = n * n
        we_array = TimedArray(np.random.rand(1, we_max_synapses), dt=duration)
        we = 'we_array(0*ms,i) * 1e-9*nS'
        # we = 'rand() * 1e-9*nS'

        wi_max_synapses = n * n
        wi_array = TimedArray(np.random.rand(1, wi_max_synapses), dt=duration)
        wi = 'wi_array(0*ms,i) * 1e-9*nS'
        # wi = 'rand() * 1e-9*nS'

        # Parameters
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV

        # Time constants
        taue = 5 * ms
        taui = 10 * ms

        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        n_recorded = 10

        P = NeuronGroup(n, model=eqs, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')

        if not uncoupled:
            num_exc = int(0.8 * n)
            Pe = P[:num_exc]
            Pi = P[num_exc:]
            Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we', delay=0 * ms)
            Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi', delay=0 * ms)

            # connection probability p can depend on network size n
            ce_max_synapses = n * n
            ce_array = TimedArray(np.random.rand(1, ce_max_synapses), dt=duration)
            Ce.connect('ce_array(0*ms, i)<80./n')
            # Ce.connect(p = 80./ len(P))

            ci_max_synapses = n * n
            ci_array = TimedArray(np.random.rand(1, ci_max_synapses), dt=duration)
            Ci.connect('ci_array(0*ms, i)<80./n')
            # Ci.connect(p = 80./ len(P))

            Ce.we = we  # excitatory synaptic weight
            Ci.wi = wi  # inhibitory synaptic weight

        # Initialization
        p_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.v = 'El + (p_array(0*ms, i) * 5 - 5)*mV'
        # P.v = 'El + (randn() * 5 - 5)*mV'

        p_ge_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.ge = '(p_ge_array(0*ms, i) * 1.5 + 4) * 10.*nS'
        # P.ge = '(randn() * 1.5 + 4) * 10.*nS'

        p_gi_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.gi = '(p_gi_array(0*ms, i) * 12 + 20) * 10.*nS'
        # P.gi = '(randn() * 12 + 20) * 10.*nS'

        spikemon = SpikeMonitor(P)
        popratemon = PopulationRateMonitor(P)
        # state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1 * second)
        trace = StateMonitor(P, 'v', record=range(n_recorded))

        run(duration)

        results[devicename] = {}
        results[devicename]['spikemon_spikes'] = spikemon.num_spikes
        results[devicename]['popratemon_rate'] = popratemon.rate[:]
        results[devicename]['trace_w'] = trace.w

        print("Values for {}".format(devicename))
        print(results[devicename]['spikemon_spikes'])
        print(results[devicename]['popratemon_rate'])
        print(results[devicename]['trace_w'])

    for key in ['spikemon_spikes', 'trace_w', 'popratemon_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cpp_standalone
@pytest.mark.cuda_standalone
def test_cobahh_pseudo_coupled_1000():
    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        duration = 10 * second

        uncoupled = False

        n = 4000

        # fixed connectivity: 1000 neurons per synapse
        p = lambda n: str(1000. / n)

        # weights set to tiny values, s.t. they are effectively zero but don't
        # result in compiler optimisations
        we_max_synapses = n * n
        we_array = TimedArray(np.random.rand(1, we_max_synapses), dt=duration)
        we = 'we_array(0*ms,i) * 1e-9*nS'
        # we = 'rand() * 1e-9*nS'

        wi_max_synapses = n * n
        wi_array = TimedArray(np.random.rand(1, wi_max_synapses), dt=duration)
        wi = 'wi_array(0*ms,i) * 1e-9*nS'
        # wi = 'rand() * 1e-9*nS'

        # Parameters
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV

        # Time constants
        taue = 5 * ms
        taui = 10 * ms

        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        n_recorded = 10

        P = NeuronGroup(n, model=eqs, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')

        if not uncoupled:
            num_exc = int(0.8 * n)
            Pe = P[:num_exc]
            Pi = P[num_exc:]
            Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we', delay=0 * ms)
            Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi', delay=0 * ms)

            # connection probability p can depend on network size n
            ce_max_synapses = n * n
            ce_array = TimedArray(np.random.rand(1, ce_max_synapses), dt=duration)
            Ce.connect('ce_array(0*ms, i)<1000./n')
            # Ce.connect(p = 1000./ len(P))

            ci_max_synapses = n * n
            ci_array = TimedArray(np.random.rand(1, ci_max_synapses), dt=duration)
            Ci.connect('ci_array(0*ms, i)<1000./n')
            # Ci.connect(p = 1000./ len(P))
            Ce.we = we  # excitatory synaptic weight
            Ci.wi = wi  # inhibitory synaptic weight

        # Initialization
        p_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.v = 'El + (p_array(0*ms, i) * 5 - 5)*mV'
        # P.v = 'El + (randn() * 5 - 5)*mV'

        p_ge_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.ge = '(p_ge_array(0*ms, i) * 1.5 + 4) * 10.*nS'
        # P.ge = '(randn() * 1.5 + 4) * 10.*nS'

        p_gi_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.gi = '(p_gi_array(0*ms, i) * 12 + 20) * 10.*nS'
        # P.gi = '(randn() * 12 + 20) * 10.*nS'

        spikemon = SpikeMonitor(P)
        popratemon = PopulationRateMonitor(P)
        # state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1 * second)
        trace = StateMonitor(P, 'v', record=range(n_recorded))

        run(duration)

        results[devicename] = {}
        results[devicename]['spikemon_spikes'] = spikemon.num_spikes
        results[devicename]['popratemon_rate'] = popratemon.rate[:]
        results[devicename]['trace_w'] = trace.w

        print("Values for {}".format(devicename))
        print(results[devicename]['spikemon_spikes'])
        print(results[devicename]['popratemon_rate'])
        print(results[devicename]['trace_w'])

    for key in ['spikemon_spikes', 'trace_w', 'popratemon_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)


@pytest.mark.cpp_standalone
@pytest.mark.cuda_standalone
def test_cobahh_pseudo_uncoupled():
    previous_device = get_device()
    results = {}

    for devicename in ['cpp_standalone', 'cuda_standalone']:
        set_device(devicename, build_on_run=False, with_output=False)
        Synapses.__instances__().clear()
        reinit_devices()

        duration = 10 * second

        uncoupled = True

        # Parameters
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV

        # Time constants
        taue = 5 * ms
        taui = 10 * ms

        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV

        n = 4000

        we_max_synapses = n * n
        we_array = TimedArray(np.random.rand(1, we_max_synapses), dt=duration)
        we = 'we_array(0*ms,i) * 1e-9*nS'
        # we = 'rand() * 1e-9*nS'

        wi_max_synapses = n * n
        wi_array = TimedArray(np.random.rand(1, wi_max_synapses), dt=duration)
        wi = 'wi_array(0*ms,i) * 1e-9*nS'
        # wi = 'rand() * 1e-9*nS'

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        n_recorded = 10

        P = NeuronGroup(n, model=eqs, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')

        if not uncoupled:
            num_exc = int(0.8 * n)
            Pe = P[:num_exc]
            Pi = P[num_exc:]
            Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we', delay=0 * ms)
            Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi', delay=0 * ms)

            # connection probability p can depend on network size n
            ce_max_synapses = n * n
            ce_array = TimedArray(np.random.rand(1, ce_max_synapses), dt=duration)
            Ce.connect('ce_array(0*ms, i)<1000./n')
            # Ce.connect(p=1000. / len(P))

            ci_max_synapses = n * n
            ci_array = TimedArray(np.random.rand(1, ci_max_synapses), dt=duration)
            Ci.connect('ci_array(0*ms, i)<1000./n')
            # Ci.connect(p=1000. / len(P))

            Ce.we = we  # excitatory synaptic weight
            Ci.wi = wi  # inhibitory synaptic weight

        # Initialization
        p_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.v = 'El + (p_array(0*ms, i) * 5 - 5)*mV'
        # P.v = 'El + (randn() * 5 - 5)*mV'

        p_ge_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.ge = '(p_ge_array(0*ms, i) * 1.5 + 4) * 10.*nS'
        # P.ge = '(randn() * 1.5 + 4) * 10.*nS'

        p_gi_array = TimedArray(np.random.randn(1, n), dt=duration)
        P.gi = '(p_gi_array(0*ms, i) * 12 + 20) * 10.*nS'
        # P.gi = '(randn() * 12 + 20) * 10.*nS'

        spikemon = SpikeMonitor(P)
        popratemon = PopulationRateMonitor(P)
        # state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1 * second)
        trace = StateMonitor(P, 'v', record=range(n_recorded))

        run(duration)

        results[devicename] = {}
        results[devicename]['spikemon_spikes'] = spikemon.num_spikes
        results[devicename]['popratemon_rate'] = popratemon.rate[:]
        results[devicename]['trace_w'] = trace.w

        print("Values for {}".format(devicename))
        print(results[devicename]['spikemon_spikes'])
        print(results[devicename]['popratemon_rate'])
        print(results[devicename]['trace_w'])

    for key in ['spikemon_spikes', 'trace_w', 'popratemon_rate']:
        assert_allclose(results['cpp_standalone'][key], results['cuda_standalone'][key])

    reset_device(previous_device)