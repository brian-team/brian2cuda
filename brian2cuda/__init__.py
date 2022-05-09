'''
Package implementing the CUDA "standalone" `Device` and `CodeObject`.
'''

from . import cuda_prefs
from .codeobject import CUDAStandaloneCodeObject
from .device import cuda_standalone_device
from . import binomial
from . import timedarray

# make the test suite available via brian2cuda.test()
from .tests import run as test

# TODO: Remove for release, until then debug logs by default
from brian2.utils.logger import BrianLogger
BrianLogger.log_level_debug()
