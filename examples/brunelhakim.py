"""
Dynamics of a network of sparsely connected inhibitory current-based
integrate-and-fire neurons. Individual neurons fire irregularly at
low rate but the network is in an oscillatory global activity regime
where neurons are weakly synchronized.

Reference:
    "Fast Global Oscillations in Networks of Integrate-and-Fire
    Neurons with Low Firing Rates"
    Nicolas Brunel & Vincent Hakim
    Neural Computation 11, 1621-1671 (1999)

Modification to original brian2 example: changed delay from constant to
    sampled from uniform distribution; remark: widening the delay
    distribution reduces the oscillatory power of the network
    (cf. with bifurcation diagrams in  figs 8 of reference above)
"""

#####################################################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
#devicename = 'cpp_standalone'

# select homogeneous (constant/identical) or heterogeneous (distributed) delays
heterog_delays = False
#heterog_delays = True

# select heterogeneous delays distribution:
# the following flag enables to have a very narrow delay distribution [2ms-dt,
# 2ms+dt] such that network behaviour is almost identical to the constant delay
# case but we force usage of the heterogeneous delay synapse propagation
# mechanism
# remark: disabling leads to a wider delay distribution [0, 4ms] producing less
# network oscillations
# further remark: to have the network operate in a similar activity regime
# (particular w.r.t. oscillations) we change the sigmaext based on inspecting
# fig. 8 from (Brunel & Hakim 1999)
narrow_delaydistr = False
#narrow_delaydistr = True

# number of neurons
N = 5000

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

## the preferences below only apply for cuda_standalone

# number of post blocks (None is default)
num_blocks = None

# atomic operations
atomics = True

# push synapse bundles
bundle_mode = True

###############################################################################
## CONFIGURATION

params = {'devicename': devicename,
          'heterog_delays': heterog_delays,
          'narrow_delaydistr': narrow_delaydistr,
          'resultsfolder': resultsfolder,
          'codefolder': codefolder,
          'N': N,
          'profiling': profiling,
          'monitors': monitors,
          'single_precision': single_precision,
          'num_blocks': num_blocks,
          'atomics': atomics,
          'bundle_mode': bundle_mode}

from utils import set_prefs, update_from_command_line

# update params from command line
choices={'devicename': ['cuda_standalone', 'cpp_standalone', 'genn']}
update_from_command_line(params, choices=choices)

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

Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = .1*second
C = 1000
sparseness = float(C)/params['N']
J = .1*mV
if params['heterog_delays'] and not params['narrow_delaydistr']:
    # to have a similar network activity regime as for const delays (or narrow
    # delay distribution)
    sigmaext = 0.33*mV
    muext = 27*mV
else:
    # default values from brian2 example (say 'reference' regime)
    sigmaext = 1*mV
    muext = 25*mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr

# delayed synapses
if params['heterog_delays']:
    conn = Synapses(group, group, on_pre='V += -J')
    conn.connect(p=sparseness)
    if params['narrow_delaydistr']:
        conn.delay = "delta + 2 * dt * rand() - dt"
    else:
        conn.delay = "2 * delta * rand()"
else:
    conn = Synapses(group, group, on_pre='V += -J', delay=delta)
    conn.connect(p=sparseness)

if params['monitors']:
    M = SpikeMonitor(group)
    LFP = PopulationRateMonitor(group)

run(duration, report='text', profile=params['profiling'])

###############################################################################
## RESULTS COLLECTION

if not os.path.exists(params['resultsfolder']):
    os.mkdir(params['resultsfolder']) # for plots and profiling txt file
if params['profiling']:
    print(profiling_summary())
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

if params['monitors']:
    subplot(211)
    plot(M.t/ms, M.i, '.')
    xlim(0, duration/ms)

    subplot(212)
    plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
    xlim(0, duration/ms)
    #show()

    plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name))
    savefig(plotpath)
    print('plot saved in {}'.format(plotpath))
    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
