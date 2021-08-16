from brian2 import *
from brian2cuda import *
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# preference for memory saving
device_name = "cpp_standalone"
codefolder = "./consistency_test/BrunelHakimHeteroDelays/results/" + device_name
set_device(device_name, directory = codefolder,debug=True)
#prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
#prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.5
#prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0

category = "Full examples"
n_label = 'Num neurons'

# configuration options
duration = 10*second


name = "BrunelHakimwithheterogeneousdelaysuniform"
tags = ["Neurons", "Synapses", "Delays"]
#n_range = [100, 1000, 10000, 20000, 50000, 100000, 380000]  #pass: 389649, fail: 396484
n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]  #pass: 389649, fail: 396484
n_range = [int(10**p) for p in n_power]

# delays [0, 4] ms
heterog_delays = "4*ms * rand()"
# homogeneous delays
homog_delays = None  # second

# to have a similar network activity regime as for homogenous delays
# or narrow delay distribution
sigmaext = 0.33*mV
muext = 27*mV


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

if not os.path.exists("./consistency_test/BrunelHakimHeteroDelays/results/"):
    os.makedirs("./consistency_test/BrunelHakimHeteroDelays/results/") # for plots and profiling txt file

subplot(211)
plot(M.t/ms, M.i, '.')
xlim(0, duration/ms)

subplot(212)
plot(LFP.t/ms, LFP.smooth_rate(window='flat', width=0.5*ms)/Hz)
xlim(0, duration/ms)
#show()

plotpath = os.path.join("./consistency_test/BrunelHakimHeteroDelays/results/"+ device_name + "/", '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
