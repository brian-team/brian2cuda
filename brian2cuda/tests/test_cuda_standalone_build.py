import pytest
import shutil
from brian2 import *
from brian2cuda import *

cuda_available = shutil.which("nvcc") is not None

pytestmark = pytest.mark.skipif(
    not cuda_available,
    reason="CUDA compiler (nvcc) not available"
)

def test_cuda_standalone_build(tmp_path):
    set_device("cuda_standalone", build_on_run=False)

    start_scope()

    eqs = """
    dv/dt = -v / (10*ms) : 1
    """

    G = NeuronGroup(1, eqs, method="euler")
    G.v = 1

    M = StateMonitor(G, "v", record=True)

    run(1*ms)

    device.build(directory=str(tmp_path), compile=True, run=False)

    assert tmp_path.exists()