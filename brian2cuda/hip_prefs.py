'''
Preferences that relate to the brian2cuda HIP/ROCm interface.
'''
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.logger import get_logger


logger = get_logger(__name__)


# The HIP standalone device shares its code-generation preferences with the CUDA
# standalone device (registered under ``devices.cuda_standalone`` in
# ``cuda_prefs``); ``device.py`` reads those from ``devices.cuda_standalone`` for
# both backends. Only the HIP backend's installation/build preferences differ, so
# the ``devices.hip_standalone`` category is registered empty here purely to
# establish the parent namespace required before ``hip_backend`` can be read.
prefs.register_preferences(
    'devices.hip_standalone',
    'Brian2CUDA HIP/ROCm preferences',
)

prefs.register_preferences(
    'devices.hip_standalone.hip_backend',
    'Preferences for the HIP backend in Brian2CUDA',

    gpu_heap_size = BrianPreference(
        docs='''Size of the heap (in MB) used by malloc() and free() device system calls,
        the HIP analogue of ``devices.cuda_standalone.cuda_backend.gpu_heap_size``. It is
        applied via ``hipDeviceSetLimit`` in the generated ``main.cu``.''',
        validator=lambda v: isinstance(v, int) and v >= 0,
        default=128),

    gpu_id=BrianPreference(
        docs='''The ID of the GPU that should be used for code execution. Default value is
        ``None``, in which case the first available GPU is used.

        If environment variable ``HIP_VISIBLE_DEVICES`` is set, this preference will be
        interpreted as ID from the visible devices.
        ''',
        default=None,
        validator=lambda v: v is None or isinstance(v, int)
    ),

    extra_compile_args_hipcc=BrianPreference(
        docs='Extra compile arguments (a list of strings) to pass to the hipcc compiler.',
        default=['-w', '-ffast-math']
    ),

    gpu_arch=BrianPreference(
        docs='''Manually set the GPU architecture for which HIP code will be
        compiled. Has to be a string (e.g. ``gfx90a``) or None. If None, architecture is
        detected automatically.''',
        validator=lambda v: v is None or isinstance(v, str),
        default=None
    ),

    rocm_path=BrianPreference(
        docs='''The path to the ROCm installation. If set, this preference takes
        precedence over environment variable ``ROCM_PATH``.''',
        default=None,
        validator=lambda v: v is None or isinstance(v, str)
    ),
)
