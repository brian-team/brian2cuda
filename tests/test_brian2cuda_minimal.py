from brian2 import *

def test_brian2cuda_minimal():
# select CUDA backend
set_device("cuda_standalone", build_on_run=False)

```
start_scope()

eqs = '''
dv/dt = (1 - v)/ms : 1
'''

G = NeuronGroup(10, eqs, threshold='v>1', reset='v=0', method='euler')
G.v = 0

run(1*ms)

# test passes if simulation runs
assert True
```
