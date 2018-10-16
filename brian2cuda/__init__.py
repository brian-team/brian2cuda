'''
Package implementing the CUDA "standalone" `Device` and `CodeObject`.
'''

import cuda_prefs
from .codeobject import CUDAStandaloneCodeObject
from .device import cuda_standalone_device
import binomial
