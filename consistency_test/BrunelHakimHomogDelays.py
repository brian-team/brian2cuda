from brian2 import *
import brian2cuda
import os
import matplotlib.pyplot as plt
from utils import get_directory
import sys
plt.switch_backend('agg')

np.random.seed(123)

# device_name = sys.argv[1]
# print("Running in device:")
# print(device_name)
device_name = "cpp_standalone"
codefolder = get_directory(device_name)
# preference for memory saving
set_device(device_name, directory = codefolder,debug=True)


category = "Full examples"
n_label = 'Num neurons'

# configuration options
duration = 10*second

# need to be set in child class
sigmaext = None  # vold


# heterogeneous delays
heterog_delays = None  # string syntax

name = "BrunelHakimwithHomogeneous"
tags = ["Neurons", "Synapses", "Delays"]

# all delays 2 ms
homog_delays = 2*ms

sigmaext = 1*mV
muext = 25*mV


assert not (heterog_delays is not None and
            homog_delays is not None), \
        "Can't set homog_delays and heterog_delays"
Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
C = 1000
n = 100
sparseness = float(C)/n
J = .1*mV
muext = muext
sigmaext = sigmaext

num_neurons = n
num_time_steps = int(duration // defaultclock.dt)
custom_xi = TimedArray(
    np.random.rand(num_time_steps, num_neurons) / np.sqrt(defaultclock.dt),
    dt=defaultclock.dt
)

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * custom_xi(t, i))/tau : volt
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

plotfolder = get_directory(device_name, basedir='plots')
os.makedirs(plotfolder, exist_ok=True)
plotpath = os.path.join(plotfolder, '{}_{}.pdf'.format(name,device_name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
