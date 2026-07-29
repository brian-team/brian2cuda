"""
Package implementing the CUDA "standalone" `Device` and `CodeObject`.
"""
import logging
import os

from . import binomial, cuda_prefs, timedarray
from .codeobject import CUDAStandaloneCodeObject
from .device import cuda_standalone_device

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(
            root="..",
            relative_to=__file__,
            version_scheme="post-release",
            local_scheme="no-local-version",
        )
        __version_tuple__ = tuple(int(x) for x in __version__.split(".")[:3])
    except ImportError:
        logging.getLogger("brian2cuda").warn(
            "Cannot determine Brian2CUDA version, running from source and "
            "setuptools_scm is not installed."
        )
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0)


# make the test suite available via brian2cuda.test()
from .tests import run as test


def _load_preference_files():
    """
    Load brian2cuda preference files from standard locations.

    This function loads brian2cuda-specific preferences from user preference
    files, avoiding the validation error that occurs when external package
    preferences are added to the main preference file.

    Files are loaded in the following order (later files override earlier ones):
    1. ~/.brian2cuda_preferences (user-specific preferences)
    2. ./brian2cuda_preferences (project-specific preferences)

    Missing files are silently ignored. Invalid files generate warnings but
    do not prevent the package from loading.
    """
    from brian2 import prefs

    # Define preference file locations
    user_prefs_file = os.path.join(
        os.path.expanduser('~'), '.brian2cuda_preferences'
    )
    local_prefs_file = 'brian2cuda_preferences'

    preference_files = [user_prefs_file, local_prefs_file]

    for prefs_file in preference_files:
        try:
            prefs.read_preference_file(prefs_file)
        except OSError:
            # File doesn't exist, that's fine
            pass
        except Exception as e:
            # Log a warning for other errors (invalid format, etc.)
            logger = logging.getLogger('brian2cuda')
            logger.warning(
                f"Error reading preference file '{prefs_file}': {e}"
            )


# Load preference files when the package is imported
_load_preference_files()


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
    import numpy as np
    from numpy.testing import assert_allclose

    import brian2cuda
    from brian2 import NeuronGroup, ms, run
    from brian2.devices.device import device, set_device

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
