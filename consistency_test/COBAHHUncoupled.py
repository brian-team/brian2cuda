from brian2 import *
from brian2cuda import *
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

device_name = "cuda_standalone"
codefolder = "./consistency_test/COBAHHUncoupled/results/" + device_name

# preference for memory saving
set_device(device_name, directory = codefolder,debug=True)

#prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
#prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.5
#prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0


name = "COBAHHuncoupled"
category = "Full examples"
n_label = 'Num neurons'

# configuration options
duration = 10 *second

uncoupled = True

n_power = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]  #fail: 131250000
n_range = [int(10**p) for p in n_power]

# Parameters
area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area

El = -60*mV
EK = -90*mV
ENa = 50*mV
g_na = (100*msiemens*cm**-2) * area
g_kd = (30*msiemens*cm**-2) * area
VT = -63*mV

# Time constants
taue = 5*ms
taui = 10*ms

# Reversal potentials
Ee = 0*mV
Ei = -80*mV

# The model
eqs = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
         (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
         (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
''')

n_recorded = 10
n = 4000

P = NeuronGroup(n, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='exponential_euler')

if not uncoupled:
    num_exc = int(0.8 * n)
    Pe = P[:num_exc]
    Pi = P[num_exc:]
    Ce = Synapses(Pe, P, 'we : siemens (constant)', on_pre='ge+=we', delay=0*ms)
    Ci = Synapses(Pi, P, 'wi : siemens (constant)', on_pre='gi+=wi', delay=0*ms)

    # connection probability p can depend on network size n
    Ce.connect(p(n))
    Ci.connect(p(n))
    Ce.we = we  # excitatory synaptic weight
    Ci.wi = wi  # inhibitory synaptic weight

# Initialization
P.v = 'El + (randn() * 5 - 5)*mV'
P.ge = '(randn() * 1.5 + 4) * 10.*nS'
P.gi = '(randn() * 12 + 20) * 10.*nS'

spikemon = SpikeMonitor(P)
popratemon = PopulationRateMonitor(P)
#state_mon = StateMonitor(S, 'w', record=range(n_recorded), dt=0.1 * second)
trace = StateMonitor(P, 'v', record=range(n_recorded))

run(duration)

if not os.path.exists(codefolder):
    os.mkdir(codefolder) # for plots and profiling txt file


subplot(311)
plot(spikemon.t/ms, spikemon.i, ',k')
subplot(312)
plot(trace.t/ms, trace.v[:].T[:, :5])
subplot(313)
plot(popratemon.t/ms, popratemon.smooth_rate(width=1*ms))
#show()

plotpath = os.path.join(codefolder, '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
