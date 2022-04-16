'''
This is a Brian script implementing a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies
(2007). Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman,
Harris, Zirpe, Natschlager, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel,
Vibert, Alvarez, Muller, Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience 23(3):349-98

Benchmark 2: random network of integrate-and-fire neurons with exponential
synaptic currents.

Clock-driven implementation with exact subthreshold integration
(but spike times are aligned to the grid).
'''

###############################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# random seed for reproducible simulations
seed = None

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
import matplotlib
matplotlib.use('Agg')

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda
if params['devicename'] == 'genn':
    import brian2genn
    if params['profiling']:
        params['profiling'] = False
        prefs['devices.genn.kernel_timing'] = True

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

taum = 20 * ms
taue = 5 * ms
taui = 10 * ms
Vt = -50 * mV
Vr = -60 * mV
El = -49 * mV

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

P = NeuronGroup(params['N'], eqs, threshold='v>Vt', reset='v = Vr', refractory=5 * ms,
                method='exact')
P.v = 'Vr + rand() * (Vt - Vr)'
P.ge = 0 * mV
P.gi = 0 * mV

we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ne = int(0.8 * params['N'])
Ce.connect('i<Ne', p=0.02)
Ci.connect('i>=Ne', p=0.02)

if params['monitors']:
    s_mon = SpikeMonitor(P)

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
    plot(s_mon.t/ms, s_mon.i, ',k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    #show()

    plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name))
    savefig(plotpath)
    print('plot saved in {}'.format(plotpath))
    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
