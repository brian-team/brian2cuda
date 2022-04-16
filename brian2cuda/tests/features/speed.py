'''
Check the speed of different Brian 2 configurations (with additional models for brian2cuda)
'''
import brian2
from brian2 import *
from brian2.tests.features import SpeedTest
from brian2.tests.features.speed import *

from brian2cuda.tests.features.cuda_configuration import insert_benchmark_point

from brian2.tests.features.speed import __all__
__all__.extend(['DenseMediumRateSynapsesOnlyHeterogeneousDelays',
                'SparseLowRateSynapsesOnlyHeterogeneousDelays',
                'COBAHHUncoupled',
                'COBAHHCoupled',
                'COBAHHPseudocoupled1000',
                'COBAHHPseudocoupled80',
                'BrunelHakimHomogDelays',
                'BrunelHakimHeterogDelays',
                'BrunelHakimHeterogDelaysNarrowDistr',
                'CUBAFixedConnectivityNoMonitor',
                'STDPCUDA',
                'STDPCUDAHomogeneousDelays',
                'STDPCUDAHeterogeneousDelays',
                'STDPCUDAHeterogeneousDelaysNarrowDistr',
                'STDPCUDARandomConnectivityHomogeneousDelays',
                'STDPCUDARandomConnectivityHomogeneousDelaysDuration50',
                'STDPCUDARandomConnectivityHomogeneousDelaysDuration100',
                'STDPCUDARandomConnectivityHomogeneousDelaysDuration200',
                'STDPCUDARandomConnectivityHeterogeneousDelays',
                'STDPCUDARandomConnectivityHeterogeneousDelaysNarrowDistr',
                'STDPCUDANoPostEffects',
                'STDPEventDriven',
                'MushroomBody',
                'StateMonitorBenchmarkCoalescedReads',
                'StateMonitorBenchmarkUncoalescedReads',
               ])


# Add a time measurement of the `brian2.run()` call in the `timed_run()` calls
class TimedSpeedTest(SpeedTest):
    def __init__(self, n):
        self.runtime = None
        super().__init__(n)

    def timed_run(self, duration):
        start = time.time()
        # Can't use `super().timed_run()` since the `level` argument would be too low
        brian2.run(duration, level=1)
        self.runtime = time.time() - start


class COBAHHBase(TimedSpeedTest):
    """Base class for COBAHH benchmarks with different connectivity"""

    category = "Full examples"
    n_label = 'Num neurons'

    # configuration options
    duration = 10*second

    uncoupled = False

    # if not uncoupled, these need to be set in child class
    we = None
    wi = None
    p = lambda self, n: None

    def run(self):
        # preference for memory saving
        prefs['devices.cuda_standalone.no_pre_references'] = True

        # Parameters
        area = 20000*umetre**2
        Cm = (1*ufarad*cm**-2) * area
        gl = (5e-5*siemens*cm**-2) * area

        El = -60*mV
        EK = -90*mV
        ENa = 50*mV
        g_na = (100*msiemens*cm**-2) * area
        g_kd = (30*msiemens*cm**-2) * area
        VT = -63*mV
        # Time constants
        taue = 5*ms
        taui = 10*ms
        # Reversal potentials
        Ee = 0*mV
        Ei = -80*mV

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
        alpha_m = 0.32*(mV**-1)*(13.*mV-v+VT)/
                 (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1.)*(v-VT-40.*mV)/
                (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms : Hz
        alpha_h = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms : Hz
        beta_h = 4./(1.+exp((40.*mV-v+VT)/(5.*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1.)*(15.*mV-v+VT)/
                 (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms : Hz
        beta_n = .5*exp((10.*mV-v+VT)/(40.*mV))/ms : Hz
        ''')

        P = NeuronGroup(self.n, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                        method='exponential_euler')

        if not self.uncoupled:
            logger.info(
                f"Connecting synapses with\n"
                f"\tprobability p(N={self.n}) = {self.p(self.n)}\n"
                f"\twe={self.we}\n"
                f"\twi={self.wi}"
            )
            num_exc = int(0.8 * self.n)
            Pe = P[:num_exc]
            Pi = P[num_exc:]
            Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we', delay=0*ms)
            Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi', delay=0*ms)
            # connection probability p can depend on network size n
            insert_benchmark_point("before_synapses_connect")
            Ce.connect(p=self.p(self.n))
            Ci.connect(p=self.p(self.n))
            insert_benchmark_point("after_synapses_connect")
            Ce.we = self.we  # excitatory synaptic weight
            Ci.wi = self.wi  # inhibitory synaptic weight
        else:
            logger.info("Simulating in uncoupled mode. No synapses.")

        # Initialization
        P.v = 'El + (randn() * 5 - 5)*mV'
        P.ge = '(randn() * 1.5 + 4) * 10.*nS'
        P.gi = '(randn() * 12 + 20) * 10.*nS'

        self.timed_run(self.duration)


class COBAHHUncoupled(COBAHHBase):
    """COBAHH from brian2 examples but without synapses and without monitors"""

    name = "COBAHH uncoupled (no synapses, no monitors)"
    # A100 with 40GB fails at 10**9, RTX2080 with 12GB fails at 10**(8.5)
    n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]
    n_range = [int(10**p) for p in n_power]
    uncoupled = True


class COBAHHCoupled(COBAHHBase):
    """COBAHH from brian2 examples without monitors"""

    name = "COBAHH (brian2 example, 2% coupling probabiliy, no monitors)"
    n_range = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 3781250]  #pass:3781250, fail: 3812500
    p = lambda self, n: 0.02  # connection probability
    we = 6 * nS  # excitatory synaptic weight
    wi = 67 * nS  # inhibitory synaptic weight


class COBAHHPseudocoupled1000(COBAHHBase):
    """
    COBAHH with 1000 synapses per neuron and all weights set to very small
    values, s.t. they effectively have not effect while copmiler optimisations
    are avoided (for better comparibility with the coupled case). This
    benchmark is used in in the Brian2GeNN paper. No monitors.
    """

    name = "COBAHH (1000 syn/neuron, weights zero, no monitors)"
    # A100 with 40GB fails at 10**(6.33), RTX2080 with 12GB fails at 10**6
    n_power = [2, 2.33, 2.66, 3, 3.33, 3.66, 4, 4.33, 4.66, 5, 5.33, 5.66, 6]
    n_range = [int(10**p) for p in n_power]
    # fixed connectivity: 1000 neurons per synapse
    p = lambda self, n: 1000. / n
    # weights set to tiny values, s.t. they are effectively zero but don't
    # result in compiler optimisations
    we = wi = 'rand() * 1e-9*nS'


class COBAHHPseudocoupledZeroWeights1000(COBAHHPseudocoupled1000):
    """
    COBAHH with 1000 synapses per neuron and all weights set to zero and
    without monitors.
    """

    name = "COBAHH (1000 syn/neuron, weights zero, no monitors)"
    we = wi = 0


class COBAHHPseudocoupled80(COBAHHBase):
    """
    COBAHH with 80 synapses per neuron and all weights set to very small
    values, s.t. they effectively have not effect while copmiler optimisations
    are avoided (for better comparibility with the coupled case). This
    benchmark with 1000 synapses per neuron is is used in in the Brian2GeNN
    paper. No monitors.
    """

    name = "COBAHH (80 syn/neuron, weights zero, no monitors)"
    #n_range = [100, 500, 1000, 5000, 10000, 20000, 40000, 80000, 150000, 300000, 900000, 3500000]
    # TITAN X
    #n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]  #pass: 3625000, fail: 3632813
    # A100 40GB
    n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  #pass: 9486831 (~10**6.98), fail: ?
    n_range = [int(10**p) for p in n_power]
    # fixed connectivity: 80 neurons per synapse
    p = lambda self, n: 80. / n
    # weights set to tiny values, s.t. they are effectively zero but don't
    # result in compiler optimisations
    we = wi = 'rand() * 1e-9*nS'


class COBAHHPseudocoupledZeroWeights80(COBAHHPseudocoupled80):
    """
    COBAHH with 80 synapses per neuron and all weights set to zero and without
    monitors.
    """

    name = "COBAHH (80 syn/neuron, weights zero, no monitors)"
    we = wi = 0


class BrunelHakimBase(TimedSpeedTest):
    """
    Base class for BrunelHakim benchmarks with different delay
    distributions
    """

    category = "Full examples"
    n_label = 'Num neurons'

    # configuration options
    duration = 10*second

    # need to be set in child class
    sigmaext = None  # vold
    muext = None  # volt
    # homogeneous delays
    homog_delay = None  # second
    # heterogeneous delays
    heterog_delay = None  # string syntax

    def run(self):
        # preference for memory saving
        prefs['devices.cuda_standalone.no_pre_references'] = True
        assert not (self.heterog_delay is not None and
                    self.homog_delay is not None), \
                "Can't set homog_delay and heterog_delay"
        Vr = 10*mV
        theta = 20*mV
        tau = 20*ms
        delta = 2*ms
        taurefr = 2*ms
        C = 1000
        sparseness = float(C)/self.n
        J = .1*mV
        muext = self.muext
        sigmaext = self.sigmaext

        logger.info(f"Simulating Brunel Hakim with muext={muext}, sigmaext={sigmaext}")
        eqs = """
        dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
        """

        group = NeuronGroup(self.n, eqs, threshold='V>theta',
                            reset='V=Vr', refractory=taurefr)
        group.V = Vr

        conn = Synapses(group, group, on_pre='V += -J',
                        delay=self.homog_delay)

        insert_benchmark_point("before_synapses_connect")
        conn.connect(p=sparseness)
        insert_benchmark_point("after_synapses_connect")

        if self.heterog_delay is not None:
            assert self.homog_delay is None
            logger.info(f'Setting heterogeneous delays: "{self.heterog_delay}"')
            conn.delay = self.heterog_delay
        else:
            logger.info(f'Setting homogeneous delays: "{self.homog_delay}"')

        self.timed_run(self.duration)


class BrunelHakimHomogDelays(BrunelHakimBase):
    """
    BrunelHakim with homogeneous delays from brian2 examples
    """
    name = "Brunel Hakim with homogeneous delays (2 ms)"
    tags = ["Neurons", "Synapses", "Delays"]
    # A100 with 40GB fails at 10**(6.33), RTX2080 with 12GB fails at 10**6
    # A100 passes with 2100000, fails with 2200000
    # TITAN X (also 12GB) passes with 912500, failes with 925000 (old results)
    n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, np.log10(2100000)]
    n_range = [int(10**p) for p in n_power]

    # all delays 2 ms
    homog_delay = 2*ms

    sigmaext = 1*mV
    muext = 25*mV


class BrunelHakimHeterogDelays(BrunelHakimBase):
    """
    BrunelHakim with heterogeneous delays with same mean delay and
    similar activity regime as brian2 example (with homogeneous delays).
    """
    name = "Brunel Hakim with heterogeneous delays (uniform [0, 4] ms)"
    tags = ["Neurons", "Synapses", "Delays"]
    # A100 with 40GB fails at 10**(6.5), RTX2080 with 12GB fails at 10**6
    n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    n_range = [int(10**p) for p in n_power]

    # delays [0, 4] ms
    heterog_delay = "4*ms * rand()"

    # to have a similar network activity regime as for homogenous delays
    # or narrow delay distribution
    sigmaext = 0.33*mV
    muext = 27*mV


class BrunelHakimHeterogDelaysNarrowDistr(BrunelHakimBase):
    """
    BrunelHakim with heterogeneous delays with narrow delay distribution
    with same mean delay and similar activity regime as brian2 example
    (with homogeneous delays)
    """
    name = "Brunel Hakim with heterogeneous delays (uniform 2 ms += dt)"
    tags = ["Neurons", "Synapses", "Delays"]
    #n_range = [100, 1000, 10000, 20000, 50000, 100000, 380000]
    # TITAN X
    #n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]  #pass: 423826, fail: 430661
    # A100
    n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]  #pass: 1350000 (~10**6.13), fail: 1500000
    n_range = [int(10**p) for p in n_power]

    # delays 2 ms +- dt
    heterog_delay = "2*ms + 2 * dt * rand() - dt"

    sigmaext = 1*mV
    muext = 25*mV


class SynapsesOnlyHeterogeneousDelays(TimedSpeedTest):
    category = "Synapses only with heterogeneous delays"
    tags = ["Synapses"]
    n_range = [100, 1000, 10000, 100000, 1000000]
    n_label = 'Num neurons'
    duration = 1 * second
    # memory usage will be approximately p**2*rate*dt*N**2*bytes_per_synapse/1024**3 GB
    # for CPU, bytes_per_synapse appears to be around 40?

    def run(self):
        N = self.n
        rate = self.rate
        M = int(rate * N * defaultclock.dt)
        if M <= 0:
            M = 1
        G = NeuronGroup(M, 'v:1', threshold='True')
        H = NeuronGroup(N, 'w:1')
        S = Synapses(G, H, on_pre='w += 1.0')
        insert_benchmark_point("before_synapses_connect")
        S.connect(True, p=self.p)
        insert_benchmark_point("after_synapses_connect")
        S.delay = '4*ms * rand()'
        #M = SpikeMonitor(G)
        self.timed_run(self.duration,
            # report='text',
            )
        #plot(M.t/ms, M.i, ',k')


class DenseMediumRateSynapsesOnlyHeterogeneousDelays(SynapsesOnlyHeterogeneousDelays):
    name = "Dense, medium rate"
    rate = 10 * Hz
    p = 1.0
    n_range = [100, 1000, 10000, 100000, 200000, 462500]  #fail: 468750


class SparseLowRateSynapsesOnlyHeterogeneousDelays(SynapsesOnlyHeterogeneousDelays):
    name = "Sparse, low rate"
    rate = 1 * Hz
    p = 0.2
    n_range = [100, 1000, 10000, 100000, 500000, 1000000, 3281250]  #fail: 3312500


class CUBAFixedConnectivityNoMonitor(TimedSpeedTest):

    category = "Full examples"
    name = "CUBA fixed connectivity, no monitor"
    tags = ["Neurons", "Synapses"]
    n_range = [100, 1000, 10000, 100000, 500000, 1000000, 3562500]  #fail: 3578125
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    def run(self):
        N = self.n
        Ne = int(.8 * N)

        taum = 20 * ms
        taue = 5 * ms
        taui = 10 * ms
        Vt = -50 * mV
        Vr = -60 * mV
        El = -49 * mV

        eqs = '''
        dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
        dge/dt = -ge/taue : volt (unless refractory)
        dgi/dt = -gi/taui : volt (unless refractory)
        '''

        P = NeuronGroup(
            N, eqs, threshold='v>Vt', reset='v = Vr', refractory=5 * ms)
        P.v = 'Vr + rand() * (Vt - Vr)'
        P.ge = 0 * mV
        P.gi = 0 * mV

        we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
        wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
        Ce = Synapses(P, P, on_pre='ge += we')
        Ci = Synapses(P, P, on_pre='gi += wi')
        insert_benchmark_point("before_synapses_connect")
        Ce.connect('i<Ne', p=80. / N)
        Ci.connect('i>=Ne', p=80. / N)
        insert_benchmark_point("after_synapses_connect")

        self.timed_run(self.duration)


class STDPCUDA(TimedSpeedTest):
    """
    STDP benchmark with postsynaptic effects. On average 1000 out of N
    presynaptic Poisson neurons are randomly connected to N/1000 postsynaptic
    neurons, s.t. N is the nubmer of synapses. STDP  is implemented as synaptic
    variables and presynaptic spikes changes postsynaptic conductances.
    """

    category = "Full examples"
    tags = ["Neurons", "Synapses"]
    n_label = 'Num neurons'
    name = "STDP (event-driven, ~N neurons, N synapses)"
    # TITAN X
    #n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  #pass 11325000, fail:11520000
    # A100
    n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, np.log10(19375000)]  #pass 19375000 (~10**7.28), fail:20000000
    n_range = [(int(10**p)//1000)*1000 for p in n_power]  # needs to be multiple of 1000

    # configuration options
    duration = 10*second
    post_effects = True
    # homog delay is used in Synapses constructor (for GeNN compatibility)
    homog_delay = 0*ms
    # heterog delay is used to set Synapses delay attribute
    heterog_delay = None

    # connectivity style (if not random, each pre neuron is connected to a different set
    # of K_poisson (1000) contiguous post neurons
    connectivity_random = False

    def run(self):
        # preference for memory saving
        prefs['devices.cuda_standalone.no_pre_references'] = True

        # we draw by random K_poisson out of N_poisson (on avg.) and connect
        # them to each post neuron
        N = self.n
        N_poisson = N
        K_poisson = 1000
        taum = 10*ms
        taupre = 20*ms
        taupost = taupre
        Ee = 0*mV
        vt = -54*mV
        vr = -60*mV
        El = -74*mV
        taue = 5*ms
        F = 15 * Hz
        gmax = .01
        dApre = .01
        dApost = -dApre * taupre / taupost * 1.05
        dApost *= gmax
        dApre *= gmax

        assert K_poisson == 1000
        assert N % K_poisson == 0, f"{N} != {K_poisson}"

        eqs_neurons = '''
        dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
        dge/dt = -ge / taue {} : 1
        '''

        on_pre = ''
        if self.post_effects:
            logger.info("Simulating standard STDP with postsynaptic effects")
            # normal mode => poissongroup spikes make effect on postneurons
            eqs_neurons = eqs_neurons.format('')
            on_pre += 'ge += w\n'
        else:
            logger.info("Simulating STDP without postsynaptic effects")
            # second mode => poissongroup spikes are inffective for postneurons
            # here: white noise process is added with similar mean and variance as
            # poissongroup input that is disabled in this case
            gsyn = K_poisson * F * gmax / 2. # assuming avg weight gmax/2 which holds approx. true for the bimodal distrib.
            eqs_neurons = eqs_neurons.format('+ gsyn + sqrt(gsyn) * xi')
            # eqs_neurons = eqs_neurons.format('')
        on_pre += '''Apre += dApre
                     w = clip(w + Apost, 0, gmax)'''

        input = PoissonGroup(N_poisson, rates=F)
        neurons = NeuronGroup(N/K_poisson, eqs_neurons, threshold='v>vt', reset='v = vr')
        S = Synapses(input, neurons,
                     '''w : 1
                        dApre/dt = -Apre / taupre : 1 (event-driven)
                        dApost/dt = -Apost / taupost : 1 (event-driven)''',
                     on_pre=on_pre,
                     on_post='''Apost += dApost
                         w = clip(w + Apre, 0, gmax)''',
                     delay=self.homog_delay
                    )
        insert_benchmark_point("before_synapses_connect")
        if self.connectivity_random:
            # random poisson neurons connect to a post neuron (K_poisson many on avg)
            S.connect(p=float(K_poisson)/N_poisson)
        else:
            # contiguous K_poisson many poisson neurons connect to a post neuron
            S.connect('i < (j+1)*K_poisson and i >= j*K_poisson')
        insert_benchmark_point("after_synapses_connect")
        S.w = 'rand() * gmax'

        if self.heterog_delay is not None:
            assert self.homog_delay is None
            logger.info(f'Setting heterogeneous delays: "{self.heterog_delay}"')
            S.delay = self.heterog_delay
        else:
            logger.info(f'Setting homogeneous delays: "{self.homog_delay}"')

        self.timed_run(self.duration)

class STDPCUDAHomogeneousDelays(STDPCUDA):
    homog_delay = 2*ms
    name = "STDP (event-driven, ~N neurons, N synapses, homogeneous delays)"

class STDPCUDAHeterogeneousDelays(STDPCUDA):
    homog_delay = None
    heterog_delay = "2 * 2*ms * rand()"
    name = "STDP (event-driven, ~N neurons, N synapses, heterogeneous delays)"
    # TITAN X
    #n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  #pass 11397000, fail:11422000
    # A100
    n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, np.log10(19375000)]  #pass 19375000 (~10**7.28), fail:20000000
    n_range = [(int(10**p)//1000)*1000 for p in n_power]  # needs to be multiple of 1000

class STDPCUDAHeterogeneousDelaysNarrowDistr(STDPCUDA):
    homog_delay = None
    # delays 2 ms +- dt
    heterog_delay = "2*ms + 2 * dt * rand() - dt"
    name = "STDP (event-driven, ~N neurons, N synapses, heterogeneous delays narrow)"
    # TITAN X
    #n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  #pass 11397000, fail:11422000
    # A100
    n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, np.log10(19375000)]  #pass 19375000 (~10**7.28), fail:20000000
    n_range = [(int(10**p)//1000)*1000 for p in n_power]  # needs to be multiple of 1000

class STDPCUDARandomConnectivityHomogeneousDelays(STDPCUDAHomogeneousDelays):
    # A100 with 40GB fails at 10**(8.5), RTX2080 with 12GB fails at 10**8
    n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    n_range = [(int(10**p)//1000)*1000 for p in n_power]  # needs to be multiple of 1000

    connectivity_random = True

# TODO: Allow changing simulation duration in `run_benchmark_suite.py`
class STDPCUDARandomConnectivityHomogeneousDelaysDuration50(STDPCUDARandomConnectivityHomogeneousDelays):
    duration = 50*second

class STDPCUDARandomConnectivityHomogeneousDelaysDuration100(STDPCUDARandomConnectivityHomogeneousDelays):
    duration = 100*second

class STDPCUDARandomConnectivityHomogeneousDelaysDuration200(STDPCUDARandomConnectivityHomogeneousDelays):
    duration = 200*second

class STDPCUDARandomConnectivityHeterogeneousDelays(STDPCUDAHeterogeneousDelays):
    # A100 with 40GB fails at 10**9, RTX2080 with 12GB fails at 10**(8.5)
    n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5]
    n_range = [(int(10**p)//1000)*1000 for p in n_power]  # needs to be multiple of 1000

    connectivity_random = True

class STDPCUDARandomConnectivityHeterogeneousDelaysNarrowDistr(STDPCUDAHeterogeneousDelaysNarrowDistr):
    connectivity_random = True

class STDPCUDANoPostEffects(STDPCUDA):
    """
    STDP benchmark without postsynaptic effects. On average 1000 out of N
    presynaptic Poisson neurons are randomly connected to N/1000 postsynaptic
    neurons, s.t. N is the nubmer of synapses. STDP is implemented as synaptic
    variables and presynaptic spikes have NO effect on postsynaptic variables.
    Postsynaptic neurons are driven with white noise of similar mean and
    variance as the input from the presynaptic Poisson neurons would create if
    they had postsynaptic effects.
    """

    name = "STDP (event-driven, ~N neurons, N synapses, NO postsynaptic effects)"
    post_effects = False


class STDPEventDriven(TimedSpeedTest):

    category = "Full examples"
    name = "STDP (event-driven)"
    tags = ["Neurons", "Synapses"]
    n_range = [100, 1000, 10000, 20000, 50000, 100000, 1000000, 5000000, 6542968]  #fail:6562500
    n_label = 'Num neurons'

    # configuration options
    duration = 1*second

    def run(self):
        N = self.n
        taum = 10*ms
        taupre = 20*ms
        taupost = taupre
        Ee = 0*mV
        vt = -54*mV
        vr = -60*mV
        El = -74*mV
        taue = 5*ms
        F = 15*Hz
        gmax = .01
        dApre = .01
        dApost = -dApre * taupre / taupost * 1.05
        dApost *= gmax
        dApre *= gmax

        eqs_neurons = '''
        dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
        dge/dt = -ge / taue : 1
        '''

        input_poisson = PoissonGroup(N, rates=F)
        neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr')
        S = Synapses(input_poisson, neurons,
                     '''w : 1
                        dApre/dt = -Apre / taupre : 1 (event-driven)
                        dApost/dt = -Apost / taupost : 1 (event-driven)''',
                     on_pre='''ge += w
                        Apre += dApre
                        w = clip(w + Apost, 0*siemens, gmax)''',
                     on_post='''Apost += dApost
                        w = clip(w + Apre, 0*siemens, gmax)'''
                    )
        insert_benchmark_point("before_synapses_connect")
        S.connect()
        insert_benchmark_point("after_synapses_connect")
        S.w = 'rand() * gmax'
        self.timed_run(self.duration)


class MushroomBody(TimedSpeedTest):

    category = "Full examples"
    name = "Mushroom Body example from brian2GeNN benchmarks"
    tags = ["Neurons", "Synapses"]

    # A100 with 40GB fails at 10**8, RTX2080 with 12GB fails at 10**(7.5)
    n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]
    n_range = [int(10**p) for p in n_power]
    n_label = 'Num neurons'

    # configuration options
    duration = 10*second

    def run(self):
        # preference for memory saving
        prefs['devices.cuda_standalone.no_pre_references'] = True

        import random as py_random
        # Number of neurons
        N_AL = 100
        N_MB = self.n
        N_LB = 100
        # Constants
        g_Na = 7.15*uS
        E_Na = 50*mV
        g_K = 1.43*uS
        E_K = -95*mV
        g_leak = 0.0267*uS
        E_leak = -63.56*mV
        C = 0.3*nF
        VT = -63*mV
        # Those two constants are dummy constants, only used when populations only have
        # either inhibitory or excitatory inputs
        E_e = 0*mV
        E_i = -92*mV
        # Actual constants used for synapses
        NKCKC= N_MB
        if NKCKC > 10000:
            NKCKC = 10000
        g_scaling = NKCKC/2500
        if g_scaling < 1:
            g_scaling= 1
        tau_PN_LHI = 1*ms
        tau_LHI_iKC = 3*ms
        tau_PN_iKC = 2*ms
        tau_iKC_eKC = 10*ms
        tau_eKC_eKC = 5*ms
        w_LHI_iKC = 8.75*nS
        w_eKC_eKC = 75*nS
        tau_pre = tau_post = 10*ms
        dApre = 0.1*nS/g_scaling
        dApost = -dApre
        g_max = 3.75*nS/g_scaling

        scale = .675

        traub_miles = '''
        dV/dt = -(1./C)*(g_Na*m**3.*h*(V - E_Na) +
                        g_K*n**4.*(V - E_K) +
                        g_leak*(V - E_leak) +
                        I_syn) : volt
        dm/dt = alpha_m*(1 - m) - beta_m*m : 1
        dn/dt = alpha_n*(1 - n) - beta_n*n : 1
        dh/dt = alpha_h*(1 - h) - beta_h*h : 1
        alpha_m = 0.32*(mV**-1)*(13.*mV-V+VT)/
                 (exp((13.*mV-V+VT)/(4.*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(V-VT-40.*mV)/
                (exp((V-VT-40.*mV)/(5.*mV))-1.)/ms : Hz
        alpha_h = 0.128*exp((17.*mV-V+VT)/(18.*mV))/ms : Hz
        beta_h = 4./(1.+exp((40.*mV-V+VT)/(5.*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1.)*(15.*mV-V+VT)/
                 (exp((15.*mV-V+VT)/(5.*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-V+VT)/(40.*mV))/ms : Hz
        '''

        # Principal neurons (Antennal Lobe)
        n_patterns = 10
        n_repeats = int(self.duration/second*10)
        p_perturb = 0.1
        patterns = np.repeat(np.array([np.random.choice(N_AL, int(0.2*N_AL), replace=False) for _ in range(n_patterns)]), n_repeats, axis=0)
        # Make variants of the patterns
        to_replace = np.random.binomial(int(0.2*N_AL), p=p_perturb, size=n_patterns*n_repeats)
        variants = []
        for idx, variant in enumerate(patterns):
            np.random.shuffle(variant)
            if to_replace[idx] > 0:
                variant = variant[:-to_replace[idx]]
            new_indices = np.random.randint(N_AL, size=to_replace[idx])
            variant = np.unique(np.concatenate([variant, new_indices]))
            variants.append(variant)

        training_size = (n_repeats-10)
        training_variants = []
        for p in range(n_patterns):
            training_variants.extend(variants[n_repeats * p:n_repeats * p + training_size])
        py_random.shuffle(training_variants)
        sorted_variants = list(training_variants)
        for p in range(n_patterns):
            sorted_variants.extend(variants[n_repeats * p + training_size:n_repeats * (p + 1)])

        spike_times = np.arange(n_patterns*n_repeats)*50*ms + 1*ms + rand(n_patterns*n_repeats)*2*ms
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
                                     g_raw = clip(g_raw + Apost, 0*siemens, g_max)
                                     ''',
                           on_post='''
                                      Apost += dApost
                                      g_raw = clip(g_raw + Apre, 0*siemens, g_max)''',
                           delay=0*ms)
        eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC', delay=0*ms)
        insert_benchmark_point("before_synapses_connect")
        PN_iKC.connect(p=0.15)

        if (N_MB > 10000):
            iKC_eKC.connect(p=float(10000)/N_MB)
        else:
            iKC_eKC.connect()
        eKC_eKC.connect()
        insert_benchmark_point("after_synapses_connect")

        # First set all synapses as "inactive", then set 20% to active
        PN_iKC.weight = '10*nS + 1.25*nS*randn()'
        iKC_eKC.g_raw = 'rand()*g_max/10/g_scaling'
        iKC_eKC.g_raw['rand() < 0.2'] = '(2.5*nS + 0.5*nS*randn())/g_scaling'
        iKC.V = E_leak
        iKC.h = 1
        iKC.m = 0
        iKC.n = .5
        eKC.V = E_leak
        eKC.h = 1
        eKC.m = 0
        eKC.n = .5

        #if use_spikemon:
        #    PN_spikes = SpikeMonitor(PN)
        #    iKC_spikes = SpikeMonitor(iKC)
        #    eKC_spikes = SpikeMonitor(eKC)

        self.timed_run(self.duration)


class StateMonitorBenchmarkBase(TimedSpeedTest):

    category = "Monitor only"
    tags = ["Monitors", "Neurons"]
    n_label = "Num recorded neurons"
    name = "StateMonitor benchmark"
    n_power = [1, 2, 3, 4, 5, 6]
    n_range = [int(10**p) for p in n_power]

    # configuration options
    duration = 1*second
    coalesced_reads = None

    def run(self):
        prefs.devices.cuda_standalone.profile_statemonitor_copy_to_host = 'v'

        warp_size = 32
        num_recorded_neurons = self.n
        num_neurons = int(32 * 10**6)
        G = NeuronGroup(num_neurons, 'v:1')
        G.v = 'i'
        assert self.coalesced_reads is not None, "Don't use base benchmark class"
        if self.coalesced_reads:
            # record first n neurons in neurongroup (coalesced reads on state variables)
            record = arange(num_recorded_neurons)
        else:
            # record n neurons in steps of 32 (warp size -> non-coalesced reads)
            record = arange(0, num_recorded_neurons * warp_size, warp_size)

        mon = StateMonitor(G, 'v', record=record)

        self.timed_run(self.duration)

        prefs.devices.cuda_standalone.profile_statemonitor_copy_to_host = None


class StateMonitorBenchmarkCoalescedReads(StateMonitorBenchmarkBase):
    """
    Record a state variable from N out of 32 x 10^6 neurons with
    consecutive neuron indices.
    """
    name = "StateMonitor benchmark (coalesced reads)"
    coalesced_reads = True


class StateMonitorBenchmarkUncoalescedReads(StateMonitorBenchmarkBase):
    """
    Record a state variable from N out of 32 x 10^6 neurons with
    non-consecutive neuron indices (record every 32nd neuron with 32 being the
    number of CUDA threads in a warp).
    """
    name = "StateMonitor benchmark (uncoalesced reads)"
    coalesced_reads = False


if __name__=='__main__':
    #prefs.codegen.target = 'numpy'
    ThresholderOnlyPoissonLowRate(10).run()
    #show()
