"""
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschlaeger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 3: random network of HH neurons with exponential synaptic conductances

Clock-driven implementation
(no spike time interpolation)

R. Brette - Dec 2007
"""

###############################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
#devicename = 'cpp_standalone'

# random seed for reproducible simulations
seed = None

#------------------------------------------------------------------------------
# choose mode

# coupling with 2% probability, nonzero coupling strength
scenario = 'brian2-example'

# no coupling at all
#scenario = 'uncoupled'

# coupling with 1000 synapses per neuron on avg, effectively zero coupling strength
#scenario = 'pseudocoupled-1000'

# coupling with 80 synapses per neuron on avg, effectively zero coupling strength
#scenario = 'pseudocoupled-80'
#------------------------------------------------------------------------------

# number of neurons
N = 4000

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
          'scenario': scenario,
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

# add parameter restrictions
choices = {'scenario': ['brian2-example','uncoupled', 'pseudocoupled-80',
                        'pseudocoupled-1000']}

# update params from command line
update_from_command_line(params, choices=choices)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda
if params['devicename'] == 'genn':
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
           debug=False)

if params['seed'] is not None:
    seed(params['seed'])

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

P = NeuronGroup(params['N'], model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='exponential_euler')

# only synapses for the other scenarios
if params['scenario'] != 'uncoupled':
    Ne = int(0.8 * params['N'])
    Pe = P[:Ne]
    Pi = P[Ne:]
    Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we')
    Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi')
    if params['scenario'] == 'brian2-example':
        # 0.02*N syn per neuron with nonzero efficacy
        Ce.connect(p=0.02)
        Ci.connect(p=0.02)
    elif params['scenario'] == 'pseudocoupled-1000':
        # fixed avg no of 1000 syn per neuron with effectively zero efficacy
        Ce.connect(p=1000. / len(P))
        Ci.connect(p=1000. / len(P))
    elif params['scenario'] == 'pseudocoupled-80':
        # fixed avg no of 80 syn per neuron with effectively zero efficacy
        Ce.connect(p=80. / len(P))
        Ci.connect(p=80. / len(P))

    if params['scenario'] == 'brian2-example':
        # nonzero efficacy
        Ce.we = 6 * nS
        Ci.wi = 67 * nS
    else:
        # effectively zero efficacy
        # (but avoiding compilier optimisations of constants in generated code)
        Ce.we = 'rand() * 1e-9*nS'
        Ci.wi = 'rand() * 1e-9*nS'

# Initialization
P.v = 'El + (randn() * 5 - 5)*mV'
P.ge = '(randn() * 1.5 + 4) * 10.*nS'
P.gi = '(randn() * 12 + 20) * 10.*nS'

if params['monitors']:
    trace = StateMonitor(P, 'v', record=np.arange(0, P.N, P.N // 100))
    spikemon = SpikeMonitor(P)
    popratemon = PopulationRateMonitor(P)

run(params['runtime']*second, report='text', profile=params['profiling'])

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
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.08, 5))
    axes[0].plot(trace.t/ms, trace.v[0]/mV, color='black')
    axes[0].set_ylabel('$V_i$ [mV]')
    axes[1].plot(spikemon.t/ms, spikemon.i, ',k')
    axes[1].set_ylabel('Neuron ID')
    axes[2].plot(popratemon.t/ms, popratemon.smooth_rate(width=1*ms), color='#c53929')
    axes[2].set(xlabel='Time [ms]', ylabel='Population\nrate [s$^{-1}$]')
    for ax in axes:
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    fig.align_labels()
    plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name))
    plt.savefig(plotpath, dpi=300)
    print('plot saved in {}'.format(plotpath))
    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
