"""
Package implementing the CUDA "standalone" `Device` and `CodeObject`.
"""

from . import cuda_prefs
from .codeobject import CUDAStandaloneCodeObject
from .device import cuda_standalone_device
from . import binomial
from . import timedarray


from . import _version

__version__ = _version.get_versions()["version"]


# make the test suite available via brian2cuda.test()
from .tests import run as test


def example_run(device_name="cuda_standalone", directory=None, **build_options):
    """
    Run a simple example simulation to test whether Brian2CUDA is correctly set up.

    Parameters
    ----------
    device_name : str
        What device to use (default: "cuda_standalone").
    directory : str ,optional
        The output directory to write the project to, any existing files will be
        overwritten. If the given directory name is ``None`` (default for this example
        run), then a temporary directory will be used.
    build_options : dict, optional
        Additional options that will be forwarded to the ``device.build`` call,
    """
    from brian2.devices.device import device, set_device
    from brian2 import ms, NeuronGroup, run
    import brian2cuda
    import numpy as np
    from numpy.testing import assert_allclose

    set_device(device_name, build_on_run=False)
    N = 100
    tau = 10 * ms
    G = NeuronGroup(
        N,
        "dv/dt = -v / tau: 1",
        threshold="v > 1",
        reset="v = 0",
        refractory=5 * ms,
        method="linear",
    )
    G.v = "i / 100."
    run(1 * ms)
    device.build(direct_call=False, directory=directory, **build_options)
    assert_allclose(G.v, np.arange(N) / N * np.exp(-1 * ms / tau))
    device.reinit()
    device.activate()
    print("\nExample run was successful.")