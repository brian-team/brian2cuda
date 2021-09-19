from brian2 import *
import brian2cuda
import os
import matplotlib.pyplot as plt
from utils import get_directory
import sys

plt.switch_backend('agg')

device_name = sys.argv[1]
print("Running in device:")
print(device_name)
# device_name="cpp_standalone"
codefolder = get_directory(device_name)

np.random.seed(123)

# preference for memory saving
set_device(device_name, directory=codefolder, debug=True)

category = "Full examples"
tags = ["Neurons", "Synapses"]
n_label = 'Num neurons'
name = "STDP"

# configuration options
duration = 10 * second
post_effects = True
# homog delay is used in Synapses constructor (for GeNN compatibility)
homog_delay = 0 * ms
# heterog delay is used to set Synapses delay attribute
heterog_delay = None

# we draw by random K_poisson out of N_poisson (on avg.) and connect
# them to each post neuron
N = 1000
N_poisson = N
K_poisson = 1000
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

assert K_poisson == 1000
assert N % K_poisson == 0, "{} != {}".format(N, K_poisson)

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue {} : 1
'''

on_pre = ''
if post_effects:
    # normal mode => poissongroup spikes make effect on postneurons
    eqs_neurons = eqs_neurons.format('')
    on_pre += 'ge += w\n'
else:
    # second mode => poissongroup spikes are inffective for postneurons
    # here: white noise process is added with similar mean and variance as
    # poissongroup input that is disabled in this case
    gsyn = K_poisson * F * gmax / 2.  # assuming avg weight gmax/2 which holds approx. true for the bimodal distrib.
    num_time_steps = duration // defaultclock.dt
    num_neurons = int(N/K_poisson)
    custom_xi = TimedArray(np.random.rand(num_time_steps, num_neurons)* 1/np.sqrt(defaultclock.dt), dt=defaultclock.dt)
    eqs_neurons = eqs_neurons.format('+ gsyn + sqrt(gsyn) * custom_xi(t, i)')
    #eqs_neurons = eqs_neurons.format('+ gsyn + sqrt(gsyn) * xi')
    # eqs_neurons = eqs_neurons.format('')
on_pre += '''Apre += dApre
             w = clip(w + Apost, 0, gmax)'''

input = PoissonGroup(N_poisson, rates=F)
neurons = NeuronGroup(N / K_poisson, eqs_neurons, threshold='v>vt', reset='v = vr')
max_num_synapses = int(N / K_poisson)*N_poisson
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)'''
             #on_pre=on_pre,
             # on_post='''Apost += dApost
             #     w = clip(w + Apre, 0, gmax)''',
             #delay=homog_delay
             )
# S.connect(p=float(K_poisson)/N_poisson) # random poisson neurons connect to a post neuron (K_poisson many on avg)
rand_array = TimedArray(np.random.rand(1, max_num_synapses), dt=duration)
S.connect(
    'i < (j+1)*K_poisson and i >= j*K_poisson')  # contiguous K_poisson many poisson neurons connect to a post neuron
S.w = 'rand_array(0*ms, i+j*N_pre) * gmax'
#S.w = 'rand() * gmax'

if heterog_delay is not None:
    assert homog_delay is None
    S.delay = heterog_delay

n = 3
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)

run(duration)

if not os.path.exists(codefolder):
    os.mkdir(codefolder)  # for plots and profiling txt file

subplot(n, 1, 1)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(n, 1, 2)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(n, 1, 3)
plot(mon.t / second, mon.w.T / gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
# show()

plotfolder = get_directory(device_name, basedir='plots')
os.makedirs(plotfolder, exist_ok=True)
plotpath = os.path.join(plotfolder, '{}_{}.pdf'.format(name,device_name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
