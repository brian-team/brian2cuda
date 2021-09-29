'''
A pseudo MSO neuron, with two dendrites and one axon (fake geometry).
'''
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *
import brian2cuda # cuda_standalone device

name = os.path.basename(__file__).replace('.py', '')
codefolder = os.path.join('code', name)
print('runing example {}'.format(name))
print('compiling model in {}'.format(codefolder))
set_device('cuda_standalone', build_on_run=False) # multiple runs require this change (see below)

# Morphology
morpho = Soma(30*um)
morpho.axon = Cylinder(diameter=1*um, length=300*um, n=100)
morpho.L = Cylinder(diameter=1*um, length=100*um, n=50)
morpho.R = Cylinder(diameter=1*um, length=150*um, n=50)

# Passive channels
gL = 1e-4*siemens/cm**2
EL = -70*mV
eqs='''
Im = gL * (EL - v) : amp/meter**2
I : amp (point current)
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs,
                       Cm=1*uF/cm**2, Ri=100*ohm*cm, method='exponential_euler')
neuron.v = EL
neuron.I = 0*amp

# Monitors
mon_soma = StateMonitor(neuron, 'v', record=[0])
mon_L = StateMonitor(neuron.L, 'v', record=True)
mon_R = StateMonitor(neuron, 'v', record=morpho.R[75*um])

run(1*ms)
neuron.I[morpho.L[50*um]] = 0.2*nA  # injecting in the left dendrite
run(5*ms)
neuron.I = 0*amp
run(50*ms, report='text', profile=True)

# cf. https://brian2.readthedocs.io/en/stable/user/computation.html#multiple-run-calls
device.build( directory=codefolder, compile = True, run = True, debug=False)

print(profiling_summary())

subplot(211)
plot(mon_L.t/ms, mon_soma[0].v/mV, 'k')
plot(mon_L.t/ms, mon_L[morpho.L[50*um]].v/mV, 'r')
plot(mon_L.t/ms, mon_R[morpho.R[75*um]].v/mV, 'b')
ylabel('v (mV)')
subplot(212)
for x in linspace(0*um, 100*um, 10, endpoint=False):
    plot(mon_L.t/ms, mon_L[morpho.L[x]].v/mV)
xlabel('Time (ms)')
ylabel('v (mV)')
#show()

plotpath = os.path.join('plots', '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
