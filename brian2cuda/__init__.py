'''
Package implementing the CUDA "standalone" `Device` and `CodeObject`.
'''

import cuda_prefs
from .codeobject import CUDAStandaloneCodeObject
from .device import cuda_standalone_device
import binomial
import timedarray

# TODO: Remove for release, until then debug logs by default
from brian2.utils.logger import BrianLogger
BrianLogger.log_level_debug()
