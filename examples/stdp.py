'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)
'''

###############################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# random seed for reproducible simulations
seed = None

# number of Poisson generators
# (and also expectation value of the number of randomly connected synapses)
N = 10000

# select weather spikes effect postsynaptic neurons
post_effects = True

# select weather we have (delays: 'homogeneous', 'heterogeneous', 'none')
delays = 'none'
# delays = 'homogeneous'
# delays = 'heterogeneous'

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder = 'code'

# monitors (needed for plot generation)
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
runtime = 100

# number of C++ threads (default 0: single thread without OpenMP)
# (only for devicename == 'cpp_standalone')
cpp_threads = 0

# pre or post parallelization mode in Brian2GeNN (only for devicename == 'genn')
genn_mode = 'post'
###############################################################################
## CONFIGURATION
from utils import set_prefs, update_from_command_line

# create parameter dictionary that can be modified from command line
params = {'devicename': devicename,
          'seed': seed,
          'delays': delays,
          'post_effects': post_effects,
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
choices = {'delays': ['none', 'homogeneous', 'heterogeneous']}

# update params from command line
update_from_command_line(params, choices=choices)

# do the imports after parsing command line arguments (quicker --help)
import os
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
           debug=False)

if params['seed'] is not None:
    seed(params['seed'])

# On average `K_poisson` Poisson neurons are connected to each LIF neuron
N_poisson = params['N']
K_poisson = 1000
connection_probability = float(K_poisson) / N_poisson # 10% connection probability if K_poisson=1000, N_poisson=10000
N_lif = params['N'] / K_poisson
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz * (1000./K_poisson) # this scaling is not active here since K_poisson == 1000
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

assert params['N'] % K_poisson == 0, \
    f"N (={N}) has to be a multiple of {K_poisson}"

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue {} : 1
'''

on_pre = ''
if params['post_effects']:
    # normal mode => poissongroup spikes make effect on postneurons
    eqs_neurons = eqs_neurons.format('')
    on_pre += 'ge += w\n'
else:
    # second mode => poissongroup spikes are inffective for postneurons
    # here: white noise process is added with similar mean and variance as
    # poissongroup input that is disabled in this case
    gsyn = K_poisson * F * gmax / 2. # assuming avg weight gmax/2 which holds approx. true for the bimodal distrib.
    eqs_neurons = eqs_neurons.format('+ gsyn + sqrt(gsyn) * xi')
on_pre += '''Apre += dApre
             w = clip(w + Apost, 0, gmax)'''

input = PoissonGroup(N_poisson, rates=F)
neurons = NeuronGroup(N_lif, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='exact')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre=on_pre,
             on_post='''Apost += dApost
                 w = clip(w + Apre, 0, gmax)'''
            )
S.connect(p=connection_probability)
S.w = 'rand() * gmax'

if params['delays'] == 'homogeneous':
    S.delay = 2*ms
elif params['delays'] == 'heterogeneous':
    S.delay = "2 * 2*ms * rand()"

if params['monitors']:
    weight_mon = StateMonitor(S, 'w', record=np.arange(5))
    inp_mon = SpikeMonitor(input[:100])
    neuron_mon = SpikeMonitor(neurons)

run(params['runtime']*second, report='text', profile=params['profiling'])

if not os.path.exists(params['resultsfolder']):
    os.mkdir(params['resultsfolder']) # for plots and profiling txt file
if params['profiling']:
    print(profiling_summary())
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

style_file = os.path.join(os.path.dirname(__file__), 'figures.mplstyle')
plt.style.use(['seaborn-paper', style_file])

if params['monitors']:
    # We show only the first second of activity, but show the weight development
    # for the full runtime
    figA, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(7.08, 7.08/2))
    axs[0].plot(inp_mon.t/ms, inp_mon.i, '.k')
    axs[0].set(title="Input spikes (Poisson generator)", ylabel="Neuron ID",
               xlim=(0, min(params['runtime'], 1)*1000))
    axs[1].plot(neuron_mon.t/ms, neuron_mon.i, '.', color='#c53929')
    axs[1].set(title="Leaky integrate-and-fire neurons", xlabel="Time [ms]", ylabel="Neuron ID")
    figA.align_labels()

    figB, axs = plt.subplot_mosaic("""
    AA
    BC
    """, constrained_layout=True, figsize=(7.08, 7.08/2))
    axs["A"].plot(weight_mon.t/second, weight_mon.w.T/gmax, color='dimgray')
    axs["A"].set(title="Weight evolution examples", ylabel="$w/g_\mathrm{max}$",
                 xlabel="Time [s]")
    # We cannot easily record the initial weight distribution, since it is
    # generated in the standalone code and the number of synapses is not known
    # until the synapses have been generated in the standalone code as well.
    # For illustration, we draw a new sample of initial weights here (the
    # distribution is flat and not very interesting in the first place)
    np.random.seed(params['seed'])
    initial_weights = np.random.uniform(0, 1, size=len(S))  # relative to gmax

    # we plot the final values first, to be able to use the same y limits in the
    # first plot
    values, _, _ = axs["C"].hist(S.w/gmax, bins=np.linspace(0, 1, 50, endpoint=True),
                                 color='dimgray')
    axs["C"].set(xlabel=r'$w/g_\mathrm{max}$', yticklabels=[])
    max_value = np.max(values)
    axs["C"].set(ylim=(1, 1.1*max_value), title=f'synaptic weights ({params["runtime"]:.0f}s)')
    axs["B"].hist(initial_weights, bins=np.linspace(0, 1, 50, endpoint=True),
                  color='dimgray')
    axs["B"].set(ylim=(1, 1.1*max_value), xlabel=r'$w/g_\mathrm{max}$',
                 title=f'synaptic weights (0s)')
    figB.align_labels()
else:  # we can still plot the final weight distribution
    pass
    subplot(2, 1, 1)
    plot(S.w / gmax, '.k')
    ylabel('Weight / gmax')
    xlabel('Synapse index')
    subplot(2, 1, 2)
    hist(S.w / gmax, 20)
    xlabel('Weight / gmax')
    tight_layout()
#show()

plotpath = os.path.join(params['resultsfolder'], '{}_A.png'.format(name))
figA.savefig(plotpath, dpi=300)
print('plot saved in {}'.format(plotpath))

plotpath = os.path.join(params['resultsfolder'], '{}_B.png'.format(name))
figB.savefig(plotpath, dpi=300)
print('plot saved in {}'.format(plotpath))

print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
