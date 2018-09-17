'''
Check the speed of different Brian 2 configurations (with additional models for brian2cuda)
'''
from brian2 import *
from brian2.tests.features import SpeedTest
from brian2.tests.features.speed import *

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
                'STDPEventDriven',
                'MushroomBody'
               ])


class COBAHHBase(SpeedTest):
    """Base class for COBAHH benchmarks with different connectivity"""

    category = "Full examples"
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    uncoupled = False

    # if not uncoupled, these need to be set in child class
    we = None
    wi = None
    p = lambda self, n: None

    def run(self):
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

        P = NeuronGroup(self.n, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                        method='exponential_euler')

        if not self.uncoupled:
            we = self.we  # excitatory synaptic weight
            wi = self.wi  # inhibitory synaptic weight
            num_exc = int(0.8 * self.n)
            Pe = P[:num_exc]
            Pi = P[num_exc:]
            Ce = Synapses(Pe, P, on_pre='ge+=we')
            Ci = Synapses(Pi, P, on_pre='gi+=wi')
            # connection probability p can depend on network size n
            Ce.connect(p=self.p(self.n))
            Ci.connect(p=self.p(self.n))

        # Initialization
        P.v = 'El + (randn() * 5 - 5)*mV'
        P.ge = '(randn() * 1.5 + 4) * 10.*nS'
        P.gi = '(randn() * 12 + 20) * 10.*nS'

        self.timed_run(self.duration)


class COBAHHUncoupled(COBAHHBase):
    """COBAHH from brian2 examples but without synapses and without monitors"""

    name = "COBAHH uncoupled (no synapses, no monitors)"
    n_float = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 6e7]  #fail: 7e7
    n_range = [int(n) for n in n_float]
    uncoupled = True


class COBAHHCoupled(COBAHHBase):
    """COBAHH from brian2 examples without monitors"""

    name = "COBAHH (brian2 example, 2% coupling probabiliy, no monitors)"
    n_range = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 3781250]  #fail: 3812500
    p = lambda self, n: 0.02  # connection probability
    we = 6 * nS  # excitatory synaptic weight
    wi = 67 * nS  # inhibitory synaptic weight


class COBAHHPseudocoupled1000(COBAHHBase):
    """
    COBAHH with 1000 synapses per neuron and all weights set to zero (used in
    brian2genn benchmarks) and without monitors.
    """

    name = "COBAHH (1000 syn/neuron, weights zero, no monitors)"
    n_range = [100, 500, 1000, 5000, 10000, 20000, 40000, 80000, 150000, 300000]  #fail: 500000
    # fixed connectivity: 1000 neurons per synapse
    p = lambda self, n: 1000. / n
    # weights set to zero
    we = wi = 0 * nS


class COBAHHPseudocoupled80(COBAHHBase):
    """
    COBAHH with 80 synapses per neuron and all weights set to zero (used in
    brian2genn benchmarks) and without monitors.
    """

    name = "COBAHH (80 syn/neuron, weights zero, no monitors)"
    n_range = [100, 500, 1000, 5000, 10000, 20000, 40000, 80000, 150000, 300000, 900000, 2000000]  #TODO: max size?
    # fixed connectivity: 80 neurons per synapse
    p = lambda self, n: 80. / n
    # weights set to zero
    we = wi = 0 * nS


class BrunelHakimBase(SpeedTest):
    """
    Base class for BrunelHakim benchmarks with different delay
    distributions
    """

    category = "Full examples"
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    # need to be set in child class
    sigmaext = None  # vold
    muext = None  # volt
    # homogeneous delays
    homog_delays = None  # second
    # heterogeneous delays
    heterog_delays = None  # string syntax

    def run(self):
        assert not (self.heterog_delays is not None and
                    self.homog_delays is not None), \
                "Can't set homog_delays and heterog_delays"
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

        eqs = """
        dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
        """

        group = NeuronGroup(self.n, eqs, threshold='V>theta',
                            reset='V=Vr', refractory=taurefr)
        group.V = Vr

        conn = Synapses(group, group, on_pre='V += -J',
                        delay=self.homog_delays)
        conn.connect(p=sparseness)

        if self.heterog_delays is not None:
            conn.delay = self.heterog_delays

        self.timed_run(self.duration)


class BrunelHakimHomogDelays(BrunelHakimBase):
    """
    BrunelHakim with homogeneous delays from brian2 examples
    """
    name = "Brunel Hakim with homogeneous delays (2 ms)"
    tags = ["Neurons", "Synapses", "Delays"]
    n_range = [100, 1000, 10000, 20000, 40000, 70000, 100000, 130000, 200000]  #pass: 393750, fail: 403125

    # all delays 2 ms
    homog_delays = 2*ms

    sigmaext = 1*mV
    muext = 25*mV


class BrunelHakimHeterogDelays(BrunelHakimBase):
    """
    BrunelHakim with heterogeneous delays with same mean delay and
    similar activity regime as brian2 example (with homogeneous delays).
    """
    name = "Brunel Hakim with heterogeneous delays (uniform [0, 4] ms)"
    tags = ["Neurons", "Synapses", "Delays"]
    n_range = [100, 1000, 10000, 20000, 50000, 100000, 218750]  #fail: 225000

    # delays [0, 4] ms
    heterog_delays = "4*ms * rand()"

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
    n_range = [100, 1000, 10000, 20000, 50000, 100000, 218750]  #TODO: max size?

    # delays 2 ms += dt
    heterog_delays = "2*ms + 2 * dt * rand() - dt"

    sigmaext = 1*mV
    muext = 25*mV


class SynapsesOnlyHeterogeneousDelays(SpeedTest):
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
        S.connect(True, p=self.p)
        S.delay = '4*ms * rand()'
        #M = SpikeMonitor(G)
        self.timed_run(self.duration,
            # report='text',
            )
        #plot(M.t/ms, M.i, ',k')


class DenseMediumRateSynapsesOnlyHeterogeneousDelays(SynapsesOnlyHeterogeneousDelays, SpeedTest):
    name = "Dense, medium rate"
    rate = 10 * Hz
    p = 1.0
    n_range = [100, 1000, 10000, 100000, 200000, 462500]  #fail: 468750


class SparseLowRateSynapsesOnlyHeterogeneousDelays(SynapsesOnlyHeterogeneousDelays, SpeedTest):
    name = "Sparse, low rate"
    rate = 1 * Hz
    p = 0.2
    n_range = [100, 1000, 10000, 100000, 500000, 1000000, 3281250]  #fail: 3312500


class CUBAFixedConnectivityNoMonitor(SpeedTest):

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
        Ce.connect('i<Ne', p=80. / N)
        Ci.connect('i>=Ne', p=80. / N)

        self.timed_run(self.duration)


class STDPEventDriven(SpeedTest):

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
                        w = clip(w + Apost, 0, gmax)''',
                     on_post='''Apost += dApost
                         w = clip(w + Apre, 0, gmax)'''
                    )
        S.connect()
        S.w = 'rand() * gmax'
        self.timed_run(self.duration)


class MushroomBody(SpeedTest):

    category = "Full examples"
    name = "Mushroom Body example from brian2GeNN benchmarks"
    tags = ["Neurons", "Synapses"]
    # scaling values taken from brian2GeNN benchmark
    scaling =  [0.05, 0.25, 0.5, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    n_range =  (2500 * array(scaling)).astype('int')
    n_label = 'Num neurons'

    # configuration options
    duration = 1*second

    def run(self):
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
                                     g_raw = clip(g_raw + Apost, 0, g_max)
                                     ''',
                           on_post='''
                                      Apost += dApost
                                      g_raw = clip(g_raw + Apre, 0, g_max)''',
                           )
        eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC')
        # bu.insert_benchmark_point()
        PN_iKC.connect(p=0.15)

        if (N_MB > 10000):
            iKC_eKC.connect(p=float(10000)/N_MB)
        else:
            iKC_eKC.connect()
        eKC_eKC.connect()
        # bu.insert_benchmark_point()

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






if __name__=='__main__':
    #prefs.codegen.target = 'numpy'
    ThresholderOnlyPoissonLowRate(10).run()
    #show()
