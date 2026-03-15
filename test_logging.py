import brian2cuda
from brian2 import *
set_device('cuda_standalone', build_on_run=False) 
prefs.logging.console_log_level = 'DEBUG'
G = NeuronGroup(1, 'dv/dt = -v / (10*ms) : 1')
run(1*ms)
