'''
Spike-timing dependent plasticity
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001)
'''
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *

#####################################################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# select weather spikes effect postsynaptic neurons
post_effects = True

# number of neurons
N = 1000

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder_base = 'code'

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder_base = 'code'

# monitors (neede for plot generation)
with_monitors = False

# single precision
single_precision = False

## the preferences below only apply for cuda_standalone

# number of post blocks (None is default)
num_blocks = None

# atomic operations
use_atomics = True

# push synapse bundles
bundle_mode = True

#####################################################################################################
## CONFIGURATION

from utils import parse_arguments

name = os.path.basename(__file__).replace('.py', '_') + devicename.replace('_standalone', '')

name = os.path.basename(__file__).replace('.py', '')
#TODO naming
name += '_' + devicename.replace('_standalone', '')

(N, single_precision, with_monitors, num_blocks, use_atomics, bundle_mode,
 name) = parse_arguments(N, single_precision, with_monitors, num_blocks,
                         use_atomics, bundle_mode, name)

if devicename == 'cuda_standalone':
    import brian2cuda
    import socket
    hostname = socket.gethostname() 
    if hostname in ['elnath', 'adhara']:
        prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_35')
        prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

    if num_blocks is not None:
        assert isinstance(num_blocks, int), 'num_blocks need to be integer'
        prefs['devices.cuda_standalone.parallel_blocks'] = num_blocks

    if not bundle_mode:
        prefs['devices.cuda_standalone.push_synapse_bundles'] = False

    if not use_atomics:
        prefs['codegen.generators.cuda.use_atomics'] = False

if single_precision:
    prefs['core.default_float_dtype'] = float32

codefolder = os.path.join(codefolder_base, name)
print('runing example {}'.format(name))
print('compiling model in {}'.format(codefolder))
set_device(devicename, directory=codefolder, compile=True, run=True, debug=False)

#####################################################################################################
## SIMULATION

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

on_pre='''ge += w
          Apre += dApre
          w = clip(w + Apost, 0, gmax)
       '''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre=on_pre,
             on_post='''Apost += dApost
                 w = clip(w + Apre, 0, gmax)'''
            )
S.connect()
S.w = 'rand() * gmax'

n = 2
if with_monitors:
    n = 3
    mon = StateMonitor(S, 'w', record=[0, 1])
    s_mon = SpikeMonitor(input)

run(100*second, report='text', profile=profiling)

if not os.path.exists(resultsfolder):
    os.mkdir(resultsfolder) # for plots and profiling txt file
if profiling:
    print(profiling_summary())
    profilingpath = os.path.join(resultsfolder, '{}.txt'.format(name))
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
if with_monitors:
    subplot(n,1,3)
    plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
#show()

plotpath = os.path.join(resultsfolder, '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
