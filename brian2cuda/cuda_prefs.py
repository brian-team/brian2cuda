'''
Preferences that relate to the brian2cuda interface.
'''
import numpy as np

from brian2.core.preferences import prefs, BrianPreference
from brian2.core.core_preferences import default_float_dtype_validator, dtype_repr
from brian2.utils.logger import get_logger


logger = get_logger('brian2.devices.cuda_standalone.cuda_prefs')

def validate_bundle_size_expression(string):
    known_vars = ['mean', 'std', 'max', 'min']
    try:
        # Try formatting all known_vars with 0
        formatted = string.format(**dict(zip(known_vars, [0] * len(known_vars))))
    except KeyError as error:
        logger.error(
            f"Unknown formatting variable {error}. Known variables are:"
            f" {known_vars}"
        )
        return False

    # Replase names from C++ std with numpy version for eval test below
    formatted = formatted.replace("ceil", "np.ceil")
    formatted = formatted.replace("floor", "np.floor")

    try:
        eval(formatted)
    except Exception:
        logger.error(f"Can't evaluate expression '{string}'")
        return False

    return True


# Preferences
prefs.register_preferences(
    'devices.cuda_standalone',
    'Brian2CUDA preferences',

    SM_multiplier = BrianPreference(
        default=1,
        docs='The number of blocks per SM. By default, this value is set to 1.',
        ),

    parallel_blocks = BrianPreference(
        docs='''The total number of parallel blocks to use. If ``None``, the number of parallel
                blocks equals the number streaming multiprocessors on the GPU.''',
        validator=lambda v: v is None or (isinstance(v, int) and v > 0),
        default=1),

    launch_bounds=BrianPreference(
        docs='Wether or not to use ``__launch_bounds__`` to optimise register usage in kernels.',
        default=False),

    syn_launch_bounds=BrianPreference(
        docs='Wether or not to use ``__launch_bounds__`` in synapses and synapses_push to optimise register usage in kernels.',
        default=False),

    calc_occupancy=BrianPreference(
        docs='Wether or not to use cuda occupancy api to choose num_threads and num_blocks.',
        default=True),

    extra_threshold_kernel=BrianPreference(
        docs='Wether or not to use a extra threshold kernel for resetting.',
        default=True),

    random_number_generator_type=BrianPreference(
        docs='''Generator type (str) that cuRAND uses for random number generation.
            Setting the generator type automatically resets the generator ordering
            (prefs.devices.cuda_standalone.random_number_generator_ordering) to its default value.
            See cuRAND documentation for more details on generator types and orderings.''',
        validator=lambda v: v in ['CURAND_RNG_PSEUDO_DEFAULT',
                                  'CURAND_RNG_PSEUDO_XORWOW',
                                  'CURAND_RNG_PSEUDO_MRG32K3A',
                                  'CURAND_RNG_PSEUDO_MTGP32',
                                  'CURAND_RNG_PSEUDO_PHILOX4_32_10',
                                  'CURAND_RNG_PSEUDO_MT19937',
                                  'CURAND_RNG_QUASI_DEFAULT',
                                  'CURAND_RNG_QUASI_SOBOL32',
                                  'CURAND_RNG_QUASI_SCRAMBLED_SOBOL32',
                                  'CURAND_RNG_QUASI_SOBOL64',
                                  'CURAND_RNG_QUASI_SCRAMBLED_SOBOL64'],
        default='CURAND_RNG_PSEUDO_DEFAULT'),

    random_number_generator_ordering=BrianPreference(
        docs='''The ordering parameter (str) used to choose how the results of cuRAND
            random number generation are ordered in global memory.
            See cuRAND documentation for more details on generator types and orderings.''',
        validator=lambda v: not v or v in ['CURAND_ORDERING_PSEUDO_DEFAULT',
                                           'CURAND_ORDERING_PSEUDO_BEST',
                                           'CURAND_ORDERING_PSEUDO_SEEDED',
                                           'CURAND_ORDERING_QUASI_DEFAULT'],
        default=False),  # False will prevent setting ordering in objects.cu (-> curRAND will uset the correct ..._DEFAULT)

    push_synapse_bundles=BrianPreference(
        docs='''If True, synaptic events are propagated by pushing bundles of
        synapse IDs with same delays into the corresponding delay queue. If
        False, each synapse of a spiking neuron is pushed in the corresponding
        queue individually. For very small bundle sizes (number of synapses
        with same delay, connected to a single neuron), pushing single Synapses
        can be faster. This option only has effect for ``Synapses`` objects with
        heterogenous delays.''',
        default=True),

    threads_per_synapse_bundle=BrianPreference(
        docs='''The number of threads used per synapses bundle during effect
        application. This has to be a string, which can be passed to Python's ``eval``
        function. The string can can use ``{mean}``, ``{std}``, ``{max}`` and ``{min}``
        expressions, which refer to the statistics across all bundles, and the function
        'ceil'. The result of this expression will be converted to the next
        lower ``int`` (e.g. ``1.9`` will be cast to ``1.0``). Examples: ``'{mean} + 2 *
        {std}'`` will use the mean bunde size + 2 times the standard deviation over
        bundle sizes and round it to the next lower integer. If you want to round up
        instead, use ``'ceil({mean} + 2 * {std})'``.''',
        default="{max}",
        validator=validate_bundle_size_expression),

    bundle_threads_warp_multiple=BrianPreference(
        docs='''Whether to round the number of threads used per synapse bundle during
        effect application (see
        `devices.cuda_standalone.threads_per_synapse_bundle`) to a multiple of
        the warp size. Round to next multiple if preference is ``'up'``, round to
        previous multiple if ``'low'`` and don't round at all if ``False`` (default). If
        rounding down results in ``0`` threads, ``1`` thread is used instead.''',
        default=False,
        validator=lambda v: v in ['up', 'down', False]),

    no_pre_references=BrianPreference(
        docs='''Set this preference if you don't need access to ``i`` in any
        synaptic code string and no Synapses object applies effects to
        presynaptic variables. This preference is for memory optimization until
        unnecassary device memory allocations in synapse creation are fixed, it
        is only relevant if your network uses close to all memory.''',
        default=False),

    no_post_references=BrianPreference(
        docs='''Set this preference if you don't need access to ``j`` in any
        synaptic code string and no Synapses object applies effects to
        postsynaptic variables. This preference is for memory optimization until
        unnecassary device memory allocations in synapse creation are fixed, it
        is only relevant if your network uses close to all memory.''',
        default=False),

    default_functions_integral_convertion=BrianPreference(
        docs='''The floating point precision to which integral types will be converted when
        passed as arguments to default functions that have no integral type overload in device
        code (sin, cos, tan, sinh, cosh, tanh, exp, log, log10, sqrt, ceil, floor, arcsin, arccos, arctan)."
        NOTE: Convertion from 32bit and 64bit integral types to single precision (32bit) floating-point
        types is not type safe. And convertion from 64bit integral types to double precision (64bit)
        floating-point types neither. In those cases the closest higher or lower (implementation
        defined) representable value will be selected.''',
        validator=default_float_dtype_validator,
        representor=dtype_repr,
        default=np.float64),

    use_atomics=BrianPreference(
        docs='''Weather to try to use atomic operations for synaptic effect
        application. Since this avoids race conditions, effect application can
        be parallelised.''',
        validator=lambda v: isinstance(v, bool),
        default=True),

    profile_statemonitor_copy_to_host=BrianPreference(
        docs='''Profile the final device to host copy of StateMonitor data. This
        preference is used for benchmarking and assumes that there is only one active
        StateMonitor in the network. The parameter of this preference is the recorded
        variable for which the device to host copy is recorded (e.g. 'v').''',
        validator=lambda v: v is None or isinstance(v, str),
        default=None),
)

prefs.register_preferences(
    'devices.cuda_standalone.cuda_backend',
    'Preferences for the CUDA backend in Brian2CUDA',

    gpu_heap_size = BrianPreference(
        docs='''Size of the heap (in MB) used by malloc() and free() device system calls, which
        are used in the ``cudaVector`` implementation. ``cudaVectors`` are used to
        dynamically allocate device memory for ``SpikeMonitors`` and the synapse
        queues in the ``CudaSpikeQueue`` implementation for networks with
        heterogeneously distributed delays.''',
        validator=lambda v: isinstance(v, int) and v >= 0,
        default=128),

    detect_gpus=BrianPreference(
        docs='''Whether to detect names and compute capabilities of all available GPUs.
        This needs access to ``nvidia-smi`` and ``deviceQuery`` binaries.''',
        default=True,
        validator=lambda v: isinstance(v, bool)
    ),

    gpu_id=BrianPreference(
        docs='''The ID of the GPU that should be used for code execution. Default value is
        ``None``, in which case the GPU with the highest compute capability and lowest ID
        is used.

        If environment variable ``CUDA_VISIBLE_DEVICES`` is set, this preference will be
        interpreted as ID from the visible devices (e.g. with ``CUDA_VISIBLE_DEVICES=2``
        and ``gpu_id=0`` preference, the GPU 2 will be used).
        ''',
        default=None,
        validator=lambda v: v is None or isinstance(v, int)
    ),

    extra_compile_args_nvcc=BrianPreference(
        docs='Extra compile arguments (a list of strings) to pass to the nvcc compiler.',
        default=['-w', '-use_fast_math']
    ),

    compute_capability=BrianPreference(
        docs='''Manually set the compute capability for which CUDA code will be
        compiled. Has to be a float (e.g. ``6.1``) or None. If None, compute capability is
        chosen depending on GPU in use. ''',
        validator=lambda v: v is None or isinstance(v, float),
        default=None
    ),

    detect_cuda=BrianPreference(
        docs='''Whether to try to detect CUDA installation paths and version. Disable
        this if you want to generae CUDA standalone code on a system without CUDA
        installed.''',
        default=True,
        validator=lambda v: isinstance(v, bool)
    ),

    cuda_path=BrianPreference(
        docs='''The path to the CUDA installation. If set, this preferences takes
        precedence over environment variable ``CUDA_PATH``.''',
        default=None,
        validator=lambda v: v is None or isinstance(v, str)
    ),

    cuda_runtime_version=BrianPreference(
        docs='The CUDA runtime version.',
        default=None,
        validator=lambda v: v is None or isinstance(v, float)
    ),

    device_query_path=BrianPreference(
        docs='''Path to CUDA's deviceQuery binary. Used to detect a GPUs compute
        capability''',
        default=None,
        validator=lambda v: v is None or isinstance(v, str)
    ),

)
