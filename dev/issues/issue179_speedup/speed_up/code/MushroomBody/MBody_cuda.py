import random as py_random

from brian2 import *
import brian2genn
import sys
from utils import get_directory

MB_scaling = float(sys.argv[1])

extra_args = {}
device = sys.argv[2]
threads = int(sys.argv[3])
use_spikemon = sys.argv[4] == 'true'
do_run = sys.argv[5] == 'true'
if threads == -1:
    extra_args = {'use_GPU': False}
else:
    prefs.devices.cpp_standalone.openmp_threads = threads

prefs.devices.genn.path = "/cognition/home/subora/Documents/github_repository/brian2cuda/frozen_repos/genn"

codefolder = get_directory(device, delete_dir=False)

set_device(device,directory=codefolder,**extra_args)

print('Running with arguments: ', sys.argv)

# defaultclock.dt = 0.025*ms
# Constants
g_Na = 7.15*uS
E_Na = 50*mV
g_K = 1.43*uS
E_K = -95*mV
g_leak = 0.0267*uS
E_leak = -63.56*mV
C = 0.3*nF
# Those two constants are dummy constants, only used when populations only have
# either inhibitory or excitatory inputs
E_e = 0*mV
E_i = -92*mV
# Actual constants used for synapses
tau_PN_LHI = 1*ms
tau_LHI_iKC = 3*ms
tau_PN_iKC = 2*ms
tau_iKC_eKC = 10*ms
tau_eKC_eKC = 5*ms
w_LHI_iKC = 8.75*nS
w_eKC_eKC = 75*nS
tau_pre = tau_post = 10*ms
dApre = 0.1*nS/MB_scaling
dApost = -dApre
# tau_STDP = 10*ms
# g_0 = 0.125*nS
g_max = 3.75*nS/MB_scaling
# g_mid = g_max/2
# g_slope = g_mid
# tau_decay = 10e5*ms
# A = g_max/4
# offset = 0.01*A

scale = .675

# Number of neurons
N_AL = 100
N_MB = int(2500*MB_scaling)
N_LB = 100

traub_miles = '''
dV/dt = -(1/C)*(g_Na*m**3*h*(V - E_Na) +
                g_K*n**4*(V - E_K) +
                g_leak*(V - E_leak) +
                I_syn) : volt
dm/dt = alpha_m*(1 - m) - beta_m*m : 1
dn/dt = alpha_n*(1 - n) - beta_n*n : 1
dh/dt = alpha_h*(1 - h) - beta_h*h : 1

alpha_m = 0.32*(-52 - V/mV)/(exp((-52 - V/mV)/4) - 1)/ms: Hz
beta_m = 0.28*(25 + V/mV)/(exp((25 + V/mV)/5) - 1)/ms: Hz
alpha_h = 0.128*exp((-48 - V/mV)/18)/ms: Hz
beta_h = 4/(exp((-25 - V/mV)/5) + 1)/ms : Hz
alpha_n = 0.032*(-50 - V/mV)/(exp((-50 - V/mV)/5) - 1)/ms: Hz
beta_n = 0.5*exp((-55 - V/mV)/40)/ms : Hz
'''

# Principal neurons (Antennal Lobe)
n_patterns = 10
n_repeats = 10
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

# all_patterns = np.zeros((n_patterns*n_repeats, N_AL))
# for idx, p in enumerate(sorted_variants):
#     all_patterns[idx, p] = 1
# plt.imshow(all_patterns[-10*n_patterns:, :], interpolation='none')
# plt.show()

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
iKC.V = E_leak
iKC.h = 1
iKC.m = 0
iKC.n = .5

# eKCs of the mushroom body lobe
I_syn = '''I_syn = g_iKC_eKC*(V - E_e) + g_eKC_eKC*(V - E_i): amp
           dg_iKC_eKC/dt = -g_iKC_eKC/tau_iKC_eKC : siemens
           dg_eKC_eKC/dt = -g_eKC_eKC/tau_eKC_eKC : siemens'''
eqs_eKC = Equations(traub_miles) + Equations(I_syn)
eKC = NeuronGroup(N_LB, eqs_eKC, threshold='V>0*mV', refractory='V>0*mV',
                  method='exponential_euler')
eKC.V = E_leak
eKC.h = 1
eKC.m = 0
eKC.n = .5

# Synapses
PN_iKC = Synapses(PN, iKC, 'weight : siemens', on_pre='g_PN_iKC += scale*weight')
PN_iKC.connect(p=0.15)
PN_iKC.weight = '4.545*nS + 1.25*nS*randn()'

# iKC_eKC = Synapses(iKC, eKC,
#              '''
#              dg_raw/dt = (g_0 - g_raw)/tau_decay : siemens (event-driven)
#              g_syn = g_max*(tanh((g_raw - g_mid)/g_slope) + 1)/2 : siemens
#              dapre/dt =  -apre/tau_stdp : siemens (event-driven)
#              dapost/dt = -apost/tau_stdp : siemens (event-driven)
#              ''',
#              on_pre='''
#                                    apre += A
#                                    g_iKC_eKC += g_max*(tanh((g_raw - g_mid)/g_slope) + 1)/2
#                     ''',
#              on_post='''
#              g_raw += apre - offset
#              ''')
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
                   )
iKC_eKC.connect()
# First set all synapses as "inactive", then set 20% to active
iKC_eKC.g_raw = 'rand()*g_max/10'
iKC_eKC.g_raw['rand() < 0.2'] = '1.25*nS + 0.25*nS*randn()'

eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC')
eKC_eKC.connect()

if use_spikemon:
    PN_spikes = SpikeMonitor(PN)
    iKC_spikes = SpikeMonitor(iKC)
    eKC_spikes = SpikeMonitor(eKC)

import time
if do_run:
    runtime = (n_patterns*n_repeats+1)*50*ms
else:
    runtime = 0*second
start = time.time()
run(runtime, report='text')
took = (time.time()-start)
print('took %.1fs' % took)
neurons = N_AL + N_MB + N_LB
synapses = len(PN_iKC) + len(iKC_eKC) + len(eKC_eKC)

with open('benchmarks.txt', 'a') as f:
    data = [neurons, synapses, device, threads, use_spikemon, do_run, took]
    f.write('\t'.join('%s' % d for d in data) + '\n')
