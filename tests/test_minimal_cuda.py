import pytest
from brian2 import *
from brian2.devices.device import set_device

def test_minimal_cuda_code_generation():
    set_device("cuda_standalone", build_on_run=False)

    eqs = "dv/dt = -v / (10*ms) : 1"
    G = NeuronGroup(1, eqs, method="exact")
    G.v = 1

    run(1*ms)

    try:
        device.build(directory="output", compile=True, run=False)
    except Exception as e:
        pytest.fail(f"CUDA build failed: {e}")
