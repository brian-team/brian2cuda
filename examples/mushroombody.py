'''
this model example is taken from https://github.com/brian-team/brian2genn_benchmarks/blob/master/Mbody_example.py
'''

###############################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# random seed for reproducible simulations
seed = None

# number of mushroom body neurons (N_MB)
N = 2500

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder = 'code'

# monitors (neede for plot generation)
monitors = True

# single precision
single_precision = False

# number of connectivity matrix partitions
# (None uses as many as there are SMs on the GPU)
partitions = None

# atomic operations
atomics = True

# push synapse bundles
bundle_mode = True

# runtime in seconds
runtime = 1

# number of C++ threads (default 0: single thread without OpenMP)
# (only for devicename == 'cpp_standalone')
cpp_threads = 0

# pre or post parallelization mode in Brian2GeNN (only for devicename == 'genn')
genn_mode = 'post'
###############################################################################
## CONFIGURATION
from utils import set_prefs, update_from_command_line

# create paramter dictionary that can be modified from command line
params = {'devicename': devicename,
          'seed': seed,
          'resultsfolder': resultsfolder,
          'codefolder': codefolder,
          'N': N,
          'runtime': runtime,
          'profiling': profiling,
          'monitors': monitors,
          'single_precision': single_precision,
          'partitions': partitions,
          'atomics': atomics,
          'bundle_mode': bundle_mode,
          'cpp_threads': cpp_threads,
          'genn_mode': genn_mode}


# update params from command line
update_from_command_line(params)

# do the imports after parsing command line arguments (quicker --help)
import os
import random as py_random
import matplotlib
matplotlib.use('Agg')

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda
elif params['devicename'] == 'genn':
    import brian2genn
    if params['profiling']:
        prefs['devices.genn.kernel_timing'] = True
        params['profiling'] = False

# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('runing example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

set_device(params['devicename'], directory=codefolder, compile=True, run=True,
           debug=False, build_on_run=False)

if params['seed'] is not None:
    seed(params['seed'])

# Number of neurons
N_MB = params['N']
N_AL = 100
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
n_repeats = int(params['runtime']*10)
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
                   )
eKC_eKC = Synapses(eKC, eKC, on_pre='g_eKC_eKC += scale*w_eKC_eKC')
PN_iKC.connect(p=0.15)

if (N_MB > 10000):
    iKC_eKC.connect(p=float(10000)/N_MB)
else:
    iKC_eKC.connect()
eKC_eKC.connect()

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

if params['monitors']:
    PN_spikes = SpikeMonitor(PN)
    iKC_spikes = SpikeMonitor(iKC)
    iKC_voltage = StateMonitor(iKC, 'V', record=0)
    eKC_spikes = SpikeMonitor(eKC)
    eKC_voltage = StateMonitor(eKC, 'V', record=0)
    # This samples 10000 weights (i.e. 1% for N_MB=10000)
    if N_MB >= 10000:
        # Record the weights at the beginning and end
        iKC_eKC_weights = StateMonitor(iKC_eKC, 'g_raw',
                                       record=np.arange(10000),
                                       dt=params['runtime']*second)

run(params['runtime']*second, report='text', profile=params['profiling'])
if params['monitors'] and N_MB >= 10000:
    iKC_eKC_weights.record_single_timestep()
device.build()

if not os.path.exists(params['resultsfolder']):
    os.mkdir(params['resultsfolder']) # for plots and profiling txt file
if params['profiling']:
    print(profiling_summary())
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

if params['monitors']:
    style_file = os.path.join(os.path.dirname(__file__), 'figures.mplstyle')
    plt.style.use(['seaborn-paper', style_file])
    fig = plt.figure(constrained_layout=True, figsize=(7.08, 7.08))
    axs = fig.subplot_mosaic(
    """
    AA
    BB
    CC
    DD
    EE
    FG
    """)
    
    axs["A"].plot(PN_spikes.t/second, PN_spikes.i, '.k')
    axs["A"].set(title='Projection neurons (spike generators)', ylabel='Neuron ID', xticklabels=[])

    axs["B"].plot(iKC_voltage.t/second, iKC_voltage[0].V/mV, color='#c53929')
    axs["B"].set(title='intrinsic Kenyon cells (HH-type neurons)', ylabel='$V_i$ [mV]', xticklabels=[])
    axs["C"].plot(iKC_spikes.t/second, iKC_spikes.i, ',k')
    axs["C"].set(ylabel='Neuron ID', xticklabels=[])

    axs["D"].plot(eKC_voltage.t/second, eKC_voltage[0].V/mV, color='#c53929')
    axs["D"].set(title='extrinsic Kenyon cells (HH-type neurons)', ylabel='$V_i$ [mV]', xticklabels=[])
    axs["E"].plot(eKC_spikes.t/second, eKC_spikes.i, '.k')
    axs["E"].set(ylabel='Neuron ID', xlabel='Time [s]')

    fig.align_labels()

    if N_MB >= 10000:
        values, _, _ = axs["F"].hist(iKC_eKC_weights.g_raw[:, 0]/g_max, bins=np.linspace(0, 1, 50, endpoint=True),
                                     color='dimgray')
        axs["F"].set(yscale='log', xlabel=r'$g_\mathrm{raw}/g_\mathrm{max}$')
        max_value = np.max(values)
        axs["F"].set(ylim=(1, 1.25*max_value), title='iKC→eKC weights (0s)')
        axs["G"].hist(iKC_eKC_weights.g_raw[:, 1]/g_max, bins=np.linspace(0, 1, 50, endpoint=True),
                      color='dimgray')
        axs["G"].set(yscale='log', xlabel=r'$g_\mathrm{raw}/g_\mathrm{max}$')
        axs["G"].set(ylim=(1, 1.25*max_value), yticklabels=[], title=f'iKC→eKC weights ({params["runtime"]:.0f}s)')

    plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name))
    savefig(plotpath, dpi=300)
    print('plot saved in {}'.format(plotpath))
    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
