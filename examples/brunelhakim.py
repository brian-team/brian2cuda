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
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *

#####################################################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# select homogeneous (constant/identical) or heterogeneous (distributed) delays
heterog_delays = True
# heterog_delays = False

# select heterogeneous delays distribution:
# the following flag enables to have a very narrow delay distribution [2ms, 2ms+2dt] such that network
#     behaviour is almost identical to the constant delay case but we force usage of the heterogeneous delay
#     synapse propagation mechanism
# remark: disabling leads to a wider delay distribution [0, 2ms] producing less network oscillations
# further remark: to have the network operate in a similar activity regime (particular w.r.t. oscillations)
# we change the sigmaext based on inspecting fig. 8 from (Brunel & Hakim 1999)
narrow_delaydistr = True
# narrow_delaydistr = False

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder_base = 'code'

#####################################################################################################

if devicename == 'cuda_standalone':
    import brian2cuda

name = os.path.basename(__file__).replace('.py', '')
if heterog_delays:
    name = name + '_heterogdelay'
    if narrow_delaydistr:
        name = name.replace('heterogdelay', 'heterogdelay[2,2+2dt]')
    else:
        name = name.replace('heterogdelay', 'heterogdelay[0,2]')
else:
    name = name + '_homogdelay[2]'
name = name + '_' + devicename.replace('_standalone', '')

codefolder = os.path.join(codefolder_base, name)
print('runing example {}'.format(name))
print('compiling model in {}'.format(codefolder))
set_device(devicename, directory=codefolder, compile=True, run=True, debug=False)

N = 5000
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = .1*second
C = 1000
sparseness = float(C)/N
J = .1*mV
if heterog_delays and not narrow_delaydistr:
    # to have a similar network activity regime as for const delays (or narrow delay distribution)
    sigmaext = 0.33*mV
    muext = 27*mV
else:
    # default values from brian2 example (say 'reference' regime)
    sigmaext = 1*mV
    muext = 25*mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

group = NeuronGroup(N, eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr

# delayed synapses
if heterog_delays:
    conn = Synapses(group, group, on_pre='V += -J')
    conn.connect('rand()<sparseness')
    if narrow_delaydistr:
        conn.delay = "delta   +   2 * dt * rand()"
    else:
        conn.delay = "2 * delta * rand()"
else:
    conn = Synapses(group, group, on_pre='V += -J', delay=delta)
    conn.connect('rand()<sparseness')

M = SpikeMonitor(group)
LFP = PopulationRateMonitor(group)

run(duration, report='text', profile=profiling)

if not os.path.exists(resultsfolder):
    os.mkdir(resultsfolder) # for plots and profiling txt file
if profiling:
    print(profiling_summary())
    profilingpath = os.path.join(resultsfolder, '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

subplot(211)
plot(M.t/ms, M.i, '.')
xlim(0, duration/ms)

subplot(212)
plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(0, duration/ms)
#show()

plotpath = os.path.join(resultsfolder, '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
