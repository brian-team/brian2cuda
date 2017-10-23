#!/usr/bin/env python
'''
Spike-timing dependent plasticity.
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001).

This example is modified from ``synapses_STDP.py`` and writes a standalone
C++ project in the directory ``STDP_standalone``.

This version includes a further modification:
multiple pre- _and_ postsynaptic neurons (s.t. no. synpases is N).
'''
import matplotlib
matplotlib.use('Agg')

import os
example_name = os.path.splitext(os.path.basename(__file__))[0]

from brian2 import *
import brian2cuda

set_device('cuda_standalone', directory=example_name, compile=True, run=True, debug=False)

N = 1000000 # no of synapses
N_neuron = int(sqrt(N))
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

input = PoissonGroup(N_neuron, rates=F)
neurons = NeuronGroup(N_neuron, eqs_neurons, threshold='v>vt', reset='v = vr')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S.connect()
S.w = 'rand() * gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)
r_mon = PopulationRateMonitor(input)

run(100*second, report='text')

print(profiling_summary())

subplot(311)
suptitle(example_name)
plot(S.w / gmax, '.k', ms=1)
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(312)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(313)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
subplots_adjust(top=0.9)
savefig(example_name+'/'+example_name+'_plots.png')
#show()
