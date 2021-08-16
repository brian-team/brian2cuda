from brian2 import *
from brian2cuda import *
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

device_name = "cpp_standalone"
codefolder = "./consistency_test/STDPCUDA/results/" + device_name

# preference for memory saving
set_device(device_name, directory = codefolder, debug=True)
#prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
#prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.5
#prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0

category = "Full examples"
tags = ["Neurons", "Synapses"]
n_label = 'Num neurons'
name = "STDP (event-driven, ~N neurons, N synapses)"
n_power = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  #pass 11325000, fail:11520000
n_range = [(int(10**p)//1000)*1000 for p in n_power]  # needs to be multiple of 1000

# configuration options
duration = 10*second
post_effects = True
# homog delay is used in Synapses constructor (for GeNN compatibility)
homog_delay = 0*ms
# heterog delay is used to set Synapses delay attribute
heterog_delay = None



# we draw by random K_poisson out of N_poisson (on avg.) and connect
# them to each post neuron
N = 1000
N_poisson = N
K_poisson = 1000
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
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
    gsyn = K_poisson * F * gmax / 2. # assuming avg weight gmax/2 which holds approx. true for the bimodal distrib.
    eqs_neurons = eqs_neurons.format('+ gsyn + sqrt(gsyn) * xi')
    # eqs_neurons = eqs_neurons.format('')
on_pre += '''Apre += dApre
             w = clip(w + Apost, 0, gmax)'''

input = PoissonGroup(N_poisson, rates=F)
neurons = NeuronGroup(N/K_poisson, eqs_neurons, threshold='v>vt', reset='v = vr')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre=on_pre,
             on_post='''Apost += dApost
                 w = clip(w + Apre, 0, gmax)''',
             delay=homog_delay
            )
#S.connect(p=float(K_poisson)/N_poisson) # random poisson neurons connect to a post neuron (K_poisson many on avg)
S.connect('i < (j+1)*K_poisson and i >= j*K_poisson') # contiguous K_poisson many poisson neurons connect to a post neuron
S.w = 'rand() * gmax'

if heterog_delay is not None:
    assert homog_delay is None
    S.delay = heterog_delay
    
n = 3
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)  

run(duration)

if not os.path.exists(codefolder):
    os.mkdir(codefolder) # for plots and profiling txt file

subplot(n,1,1)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(n,1,2)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(n,1,3)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
#show()

plotpath = os.path.join(codefolder, '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))

