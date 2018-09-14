"""
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschlaeger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 3: random network of HH neurons with exponential synaptic conductances

Clock-driven implementation
(no spike time interpolation)

R. Brette - Dec 2007
"""
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *

#####################################################################################################
## PARAMETERS

# select code generation standalone device
devicename = 'cuda_standalone'
# devicename = 'cpp_standalone'

# choose mode
scenario = 'brian2-example' # coupling with 2% probability, nonzero coupling strength
# scenario = 'benchmark-uncoupled' # no coupling at all
# scenario = 'benchmark-pseudocoupled' # coupling with 1000 synapses per neuron on avg, zero coupling strength

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder_base = 'code'

#####################################################################################################

if devicename == 'cuda_standalone':
    import brian2cuda

name = os.path.basename(__file__).replace('.py', '') + '_' + scenario + '_' + devicename.replace('_standalone', '')

codefolder = os.path.join(codefolder_base, name)
print('runing example {}'.format(name))
print('compiling model in {}'.format(codefolder))
set_device(devicename, directory=codefolder, compile=True, run=True, debug=False)

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

if scenario == 'brian2-example':
    we = 6 * nS  # excitatory synaptic weight
    wi = 67 * nS  # inhibitory synaptic weight
else:
    we = 0 * nS  # excitatory synaptic weight
    wi = 0 * nS  # inhibitory synaptic weight


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

P = NeuronGroup(4000, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='exponential_euler')

if scenario != 'benchmark-uncoupled': # only synapses for the other scenarios
    Pe = P[:3200]
    Pi = P[3200:]
    Ce = Synapses(Pe, P, on_pre='ge+=we')
    Ci = Synapses(Pi, P, on_pre='gi+=wi')

if scenario == 'brian2-example': # 0.02*N syn per neuron with nonzero efficacy
    Ce.connect(p=0.02)
    Ci.connect(p=0.02)

elif scenario == 'benchmark-pseudocoupled': # fixed avg no of 1000 syn per neuron with zero efficacy
    Ce.connect(p=1000. / len(P))
    Ci.connect(p=1000. / len(P))

else: # scenario == 'benchmark-uncoupled'
    pass # no

# Initialization
P.v = 'El + (randn() * 5 - 5)*mV'
P.ge = '(randn() * 1.5 + 4) * 10.*nS'
P.gi = '(randn() * 12 + 20) * 10.*nS'

trace = StateMonitor(P, 'v', record=np.arange(0, P.N, P.N // 100))
spikemon = SpikeMonitor(P)
popratemon = PopulationRateMonitor(P)

run(1 * second, report='text', profile=profiling)

if not os.path.exists(resultsfolder):
    os.mkdir(resultsfolder) # for plots and profiling txt file
if profiling:
    print(profiling_summary())
    profilingpath = os.path.join(resultsfolder, '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

subplot(311)
plot(spikemon.t/ms, spikemon.i, ',k')
subplot(312)
plot(trace.t/ms, trace.v[:].T[:, :5])
subplot(313)
plot(popratemon.t/ms, popratemon.smooth_rate(width=1*ms))
#show()

plotpath = os.path.join(resultsfolder, '{}.png'.format(name))
savefig(plotpath)
print('plot saved in {}'.format(plotpath))
print('the generated model in {} needs to removed manually if wanted'.format(codefolder))
