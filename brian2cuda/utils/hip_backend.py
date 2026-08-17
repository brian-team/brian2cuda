"""
Shared HIP/ROCm backend helpers.

This module only depends on the standard library so it can be imported from both
``brian2cuda.device`` and ``brian2cuda.cuda_generator`` without creating an import
cycle (``device`` -> ``codeobject`` -> ``cuda_generator``).
"""

import os
import shutil


def get_rocm_path():
    """Return the ROCm installation path.

    Resolution order mirrors how the CUDA backend resolves ``cuda_path`` (see
    ``brian2cuda.utils.gputools.get_cuda_path``):
    ``prefs.devices.hip_standalone.hip_backend.rocm_path`` if set, then the
    ``ROCM_PATH`` environment variable, then the ``/opt/rocm`` default.
    """
    try:
        from brian2.core.preferences import prefs
        rocm_path_pref = prefs.devices.hip_standalone.hip_backend.rocm_path
    except (AttributeError, ImportError):
        rocm_path_pref = None
    if rocm_path_pref:
        return os.path.expanduser(rocm_path_pref)
    return os.environ.get('ROCM_PATH', '/opt/rocm')


def is_hip_backend():
    """Return whether the HIP/ROCm backend should be used instead of CUDA.

    The backend is selected by the ``USE_HIP`` environment variable, or inferred
    from the toolchain: ``hipcc`` present and ``nvcc`` absent, or a ROCm
    installation present and no CUDA installation. This single implementation is
    shared by code generation (``cuda_generator``) and the build (``device``) so
    they can never disagree about which backend is active.
    """
    if os.environ.get('USE_HIP', '').lower() in ('1', 'true', 'yes'):
        return True

    hipcc_path = shutil.which('hipcc')
    nvcc_path = shutil.which('nvcc')
    if hipcc_path and not nvcc_path:
        return True

    rocm_path = get_rocm_path()
    if os.path.exists(os.path.join(rocm_path, 'bin', 'hipcc')):
        cuda_path = os.environ.get('CUDA_PATH', '/usr/local/cuda')
        if not os.path.exists(os.path.join(cuda_path, 'bin', 'nvcc')):
            return True

    return False
