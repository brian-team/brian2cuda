from brian2 import *
from brian2cuda import *
import os
import matplotlib.pyplot as plt
from utils import get_directory
plt.switch_backend('agg')

# preference for memory saving
device_name = "cpp_standalone"
codefolder = get_directory(device_name)
set_device(device_name, directory = codefolder,debug=True)

n = 100
category = "Full examples"
n_label = 'Num neurons'

# configuration options
duration = 10*second


name = "BrunelHakimwithheterogeneousdelays"
tags = ["Neurons", "Synapses", "Delays"]
#n_range = [100, 1000, 10000, 20000, 50000, 100000, 380000]
n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]  #pass: 423826, fail: 430661
n_range = [int(10**p) for p in n_power]

# delays 2 ms += dt
heterog_delays = "2*ms + 2 * dt * rand() - dt"

sigmaext = 1*mV
muext = 25*mV

homog_delays = None  # second



assert not (heterog_delays is not None and
            homog_delays is not None), \
        "Can't set homog_delays and heterog_delays"
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
C = 1000
sparseness = float(C)/n
J = .1*mV
muext = muext
sigmaext = sigmaext

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

group = NeuronGroup(n, eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr

conn = Synapses(group, group, on_pre='V += -J',
                delay=homog_delays)
conn.connect(p=sparseness)

if heterog_delays is not None:
    assert homog_delays is None
    conn.delay = heterog_delays
    
M = SpikeMonitor(group)
LFP = PopulationRateMonitor(group)

run(duration)

if not os.path.exists(codefolder):
    os.makedirs(codefolder) # for plots and profiling txt file

subplot(211)
plot(M.t/ms, M.i, '.')
xlim(0, duration/ms)

subplot(212)
plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(0, duration/ms)
#show()

plotpath = os.path.join(codefolder, '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
