'''
Preferences that relate to the brian2cuda interface.
'''

from brian2.core.preferences import prefs, BrianPreference


prefs.register_preferences(
    'codegen.cuda',
    'CUDA compilation preferences',
    extra_compile_args_nvcc=BrianPreference(
        docs='''Extra compile arguments (a list of strings) to pass to the nvcc compiler.''',
        default=['-w', '-use_fast_math']
    )
)

prefs.register_preferences(
    'brian2cuda',
    'General brian2CUDA preferences',

    detect_gpus=BrianPreference(
        docs='''Whether to detect names and compute capabilities of all available GPUs.
        This needs access to `nvidia-smi` and `deviceQuery` binaries.''',
        default=True,
        validator=lambda v: isinstance(v, bool)
    ),

    gpu_id=BrianPreference(
        docs='''The ID of the GPU that should be used''',
        default=None,
        validator=lambda v: v is None or isinstance(v, int)
    ),

    cuda_path=BrianPreference(
        docs='''The path to the CUDA installation. If set, this preferences takes
        precedence over environment variable `CUDA_PATH`.''',
        default=None,
        validator=lambda v: v is None or isinstance(v, str)
    ),
)
