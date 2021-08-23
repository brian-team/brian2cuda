import random as py_random
from brian2 import *
import brian2cuda
import os
import matplotlib.pyplot as plt
from utils import get_directory
plt.switch_backend('agg')


#seed
np.random.seed(123)
py_random.seed(123)

device_name = "cpp_standalone"

codefolder = get_directory(device_name,delete_dir=True)

# preference for memory saving
set_device(device = device_name, directory=codefolder, debug=True)


category = "Full examples"
name = "MushroomBody"

# configuration options
duration = 10*second


# Number of neurons
N_AL = 100
N_MB = 2500
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
n_repeats = int(duration/second*10)
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

spike_time_randomness = rand(n_patterns*n_repeats)*2*ms

spike_times = np.arange(n_patterns*n_repeats)*50*ms + 1*ms + spike_time_randomness
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
                   delay=0*ms)
eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC', delay=0*ms)
# bu.insert_benchmark_point()
PN_iKC.connect(p=0.15)

if (N_MB > 10000):
    iKC_eKC.connect(p=float(10000)/N_MB)
else:
    iKC_eKC.connect()
eKC_eKC.connect()
# bu.insert_benchmark_point()

# First set all synapses as "inactive", then set 20% to active
pn_ikc_max_synapses = N_AL*N_MB
pn_ikc_array = TimedArray(np.random.randn(1, pn_ikc_max_synapses), dt= duration)
#PN_iKC.weight = '10*nS + 1.25*nS*pn_ikc_array(0*ms, i + j*N_post)'
PN_iKC.weight = '10*nS + 1.25*nS*randn()'

ikc_ekc_max_synapses = N_MB*N_LB
ikc_ekc_array1 = TimedArray(np.random.rand(1, ikc_ekc_max_synapses), dt= duration)
#iKC_eKC.g_raw = 'ikc_ekc_array1(0*ms, i +j*N_post)*g_max/10/g_scaling'
iKC_eKC.g_raw = 'rand()*g_max/10/g_scaling'
ikc_ekc_array2 = TimedArray(np.random.rand(1, ikc_ekc_max_synapses), dt= duration)
ikc_ekc_array3 = TimedArray(np.random.randn(1, ikc_ekc_max_synapses), dt= duration)
iKC_eKC.g_raw['ikc_ekc_array2(0*ms, i+j*N_post) < 0.2'] = '(2.5*nS + 0.5*nS*ikc_ekc_array3(0*ms, i+j*N_post))/g_scaling'
#iKC_eKC.g_raw['rand() < 0.2'] = '(2.5*nS + 0.5*nS*randn())/g_scaling'
iKC.V = E_leak
iKC.h = 1
iKC.m = 0
iKC.n = .5
eKC.V = E_leak
eKC.h = 1
eKC.m = 0
eKC.n = .5

#if use_spikemon:
PN_spikes = SpikeMonitor(PN)
iKC_spikes = SpikeMonitor(iKC)
eKC_spikes = SpikeMonitor(eKC)
run(duration)

if not os.path.exists(codefolder):
    os.mkdir(codefolder) # for plots and profiling txt file

plot_array = [PN_spikes, iKC_spikes, eKC_spikes]
plot_array_name = ['PN_spikes', 'iKC_spikes', 'eKC_spikes']

for p, M in enumerate(plot_array):
    subplot(3, 1, p+1)
    plot(M.t/ms, M.i, ',k')
    ylabel(plot_array_name[p])
    print('SpikeMon %s, average rate %.1f sp/s' %
              (plot_array_name[p], M.num_spikes/(duration/second*len(M.source))))
    #show()

plotpath = os.path.join(codefolder, '{}_{}.png'.format(name,device_name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
