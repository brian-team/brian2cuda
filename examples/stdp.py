'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)
'''

###############################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# number of _synapses_ -- min. is 1000; and: only multiples of 1000 supported
N = 1000

# select weather spikes effect postsynaptic neurons
post_effects = True

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

# number of post blocks (None is default)
num_blocks = None

# atomic operations
atomics = True

# push synapse bundles
bundle_mode = True

###############################################################################
## CONFIGURATION
from collections import OrderedDict

params = OrderedDict([('devicename', devicename),
                      ('post_effects', post_effects),
                      ('resultsfolder', resultsfolder),
                      ('codefolder', codefolder),
                      ('N', N),
                      ('profiling', profiling),
                      ('monitors', monitors),
                      ('single_precision', single_precision),
                      ('num_blocks', num_blocks),
                      ('atomics', atomics),
                      ('bundle_mode', bundle_mode)])

from utils import set_prefs, update_from_command_line

# update params from command line
update_from_command_line(params)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda

# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('runing example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

set_device(params['devicename'], directory=codefolder, compile=True, run=True,
           debug=False)

# we draw by random K_poisson out of N_poisson (on avg.) and connect them to each post neuron
N_poisson = params['N']
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
assert params['N'] % K_poisson == 0

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
    # eqs_neurons = eqs_neurons.format('')
on_pre += '''Apre += dApre
             w = clip(w + Apost, 0, gmax)'''

input = PoissonGroup(N_poisson, rates=F)
neurons = NeuronGroup(params['N']/K_poisson, eqs_neurons, threshold='v>vt', reset='v = vr')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre=on_pre,
             on_post='''Apost += dApost
                 w = clip(w + Apre, 0, gmax)'''
            )
#S.connect(p=float(K_poisson)/N_poisson) # random poisson neurons connect to a post neuron (K_poisson many on avg)
S.connect('i < (j+1)*K_poisson and i >= j*K_poisson') # contiguous K_poisson many poisson neurons connect to a post neuron
S.w = 'rand() * gmax'

n = 2
if params['monitors']:
    n = 3
    mon = StateMonitor(S, 'w', record=[0, 1])
    s_mon = SpikeMonitor(input)

run(100*second, report='text', profile=profiling)

if not os.path.exists(params['resultsfolder']):
    os.mkdir(params['resultsfolder']) # for plots and profiling txt file
if profiling:
    print(profiling_summary())
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

subplot(n,1,1)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(n,1,2)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
if params['monitors']:
    subplot(n,1,3)
    plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
#show()

plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
