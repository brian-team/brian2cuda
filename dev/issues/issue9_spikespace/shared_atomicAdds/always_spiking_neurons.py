#!/usr/bin/env python
"""
Network of always spiking neurons (in each timestep).
No synaptic connections between neurons exist.
"""
import matplotlib
matplotlib.use('Agg')

from brian2 import *
import brian2cuda

plotit = False
monitor = True
duration = 1*ms
N = 10000 # number of neurons

mode = 'cuda'
#mode = 'cpp'

BrianLogger.log_level_debug()

set_device('{}_standalone'.format(mode))


all_neurons = NeuronGroup(N, 'v:1', threshold='True')#i/2 == (i+1)/2')
#TODO: let every second neuron spike and debug spikespace!

if monitor:
    neuron_mon = SpikeMonitor(all_neurons)

run(duration)

device.build(directory='./', compile=True, run=True, debug=False)

if plotit:
    plot(neuron_mon.t/ms, neuron_mon.i, '.k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    savefig('rasterplot.png')

