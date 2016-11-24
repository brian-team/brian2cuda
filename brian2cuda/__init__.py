'''
Package implementing the CUDA "standalone" `Device` and `CodeObject`.
'''

from .codeobject import CUDAStandaloneCodeObject 
from .device import cuda_standalone_device
import cuda_prefs
