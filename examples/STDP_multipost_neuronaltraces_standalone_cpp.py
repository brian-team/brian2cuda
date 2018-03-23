#!/usr/bin/env python
'''
Spike-timing dependent plasticity.
Adapted from Song, Miller and Abbott (2000) and Song and Abbott (2001).

This example is modified from ``synapses_STDP.py`` and writes a standalone
C++ project in the directory ``STDP_standalone``.

This version includes two further modifications:
traces in neurons and multiple pre- _and_ postsynaptic neurons (s.t. no. synpases is N).
'''
import matplotlib
matplotlib.use('Agg')

import os
example_name = os.path.splitext(os.path.basename(__file__))[0]

from brian2 import *
set_device('cpp_standalone', directory=example_name, compile=True, run=True, debug=True)

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

input_poisson = PoissonGroup(N_neuron, rates=F)
# auxiliary input neurongroup to allow for (presynaptic) neuronal traces (state variable A here)
input_neurons = NeuronGroup(N_neuron, '''dA/dt = -A / taupre : 1
                                  v : volt''',
                            threshold='v > vt', reset = 'v = vr; A += dApre')
input_neurons.v = vr
# auxiliary input synapse where each poisson cell connects to exactly one neuron
S_input = Synapses(input_poisson, input_neurons, on_pre='v = vt+1*volt')
S_input.connect('i==j')

output_neuron = NeuronGroup(N_neuron, '''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                                  dge/dt = -ge / taue : 1
                                  dA/dt = -A / taupost : 1''',
                            threshold='v > vt', reset='v = vr; A += dApost')

S = Synapses(input_neurons, output_neuron,
             '''w : 1''',
             on_pre='''ge += w
                    w = clip(w + A_post, 0, gmax)''',
             on_post='''w = clip(w + A_pre, 0, gmax)''',
             )
S.connect()
S.w = 'rand() * gmax'

mon = StateMonitor(S, 'w', record=[0, 1])

s_mon = SpikeMonitor(input_poisson)
r_mon = PopulationRateMonitor(input_poisson)


run(100*second, report='text')

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
