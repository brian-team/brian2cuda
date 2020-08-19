'''
Preferences that relate to the brian2cuda interface.
'''

from brian2.core.preferences import *


prefs.register_preferences(
    'codegen.cuda',
    'CUDA compilation preferences',
    extra_compile_args_nvcc=BrianPreference(
        docs='''Extra compile arguments (a list of strings) to pass to the nvcc compiler.''',
        default=['-w', '-use_fast_math', '-arch=sm_61']  # TODO: arch is hard coded to our GPU, should be deduced form the device
    )
)
