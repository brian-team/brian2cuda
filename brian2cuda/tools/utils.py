"""Utility funcitons"""
import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def parse_arguments(parser, float_default=['float64']):

    parser.add_argument('--targets', '-t', nargs='*',
                        default=['cuda_standalone'], type=str,
                        choices=['cuda_standalone', 'genn', 'cpp_standalone'],
                        help=("Which codegeneration targets to use, can be "
                              "multiple. Only standalone targets. "
                              "[default: 'cuda_standalone']"))

    parser.add_argument('--float-dtype', '-fd', nargs='*',
                        default=float_default, choices=['float32', 'float64'],
                        help=("The default float datatype "
                              "(sets prefs['core.default_float_dtype']) "
                              "[default: {}]".format(float_default)))

    parser.add_argument('--no-bundles', '-nob', action='store_true',
                        help="Don't use bundle pushing")

    parser.add_argument('--no-atomics', '-noa', action='store_true',
                        help="Don't use atomics for effect application")

    parser.add_argument('--num-blocks', '-bl', default=None, type=int, nargs=1,
                        help=("Number of parallel blocks in post neuron blocks "
                              "structure. [default: number of SMs on GPU]"))

    parser.add_argument('--all-prefs', '-a', action='store_true',
                        help=("Run with all preference combinations."))

    parser.add_argument('--jobs', '-j', nargs=1, type=int, default=None,
                        help=("Number of jobs to run simultaneously for "
                              "compilations (passed as -j option to the make "
                              "command. [default: no job limit (make -j)]"))

    args = parser.parse_args()

    single_prefs_set = any([args.no_atomics, args.no_bundles,
                            args.num_blocks is not None])

    if args.all_prefs and single_prefs_set:
        raise AttributeError("Can't set other preferences in combination with "
                             "--all-prefs")

    float_dtypes = []
    for dtype in args.float_dtype:
        float_dtypes.append(getattr(np, dtype))

    return args, float_dtypes


def print_prefs_combinations(configurations, condition=None):
    for n, d in enumerate(configurations):
        if condition is None or condition[n]:
            print "\t{}. run".format(n + 1)
            if not d:
                print "\t\tdefault preferences"
            else:
                for k, v in d.items():
                    print "\t\tprefs[{}] = {}".format(k, v)


def set_preferences(args, prefs, fast_compilation=True, suppress_warnings=True):

    import socket

    if 'cuda_standalone' in args.targets:
        import brian2cuda

    if 'genn' in args.targets:
        import brian2genn

    if fast_compilation:
        # Switch off cpp compiler optimization to get faster compilation times
        prefs['codegen.cpp.extra_compile_args_gcc'].extend(['-w', '-O0'])
        prefs['codegen.cpp.extra_compile_args_msvc'].extend(['/Od'])
        print("Turning off  compiler optimizations for fast compilation")

    if suppress_warnings:
        # Surpress some warnings from nvcc compiler
        compile_args = ['-Xcudafe "--diag_suppress=declared_but_not_referenced"']
        prefs['codegen.cuda.extra_compile_args_nvcc'].extend(compile_args)
        print("Suppressing compiler warnings")

    # TODO: remove once we set CC automatically from hardware
    if socket.gethostname() == 'elnath':
        prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_35')
        prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

    if args.jobs is not None:
        k = 'devices.cpp_standalone.extra_make_args_unix'
        if '-j' not in prefs[k]:
            print("WARNING: -j is not anymore in default pref[{}]".format(k))
        new_j = ['-j' + str(args.jobs[0])]
        prefs[k].remove('-j')
        prefs[k].extend(new_j)
        print "Compiling with make {}".format(new_j[0])

    if args.all_prefs:
        all_prefs_list = [{'devices.cuda_standalone.parallel_blocks': 1},
                          {'devices.cuda_standalone.push_synapse_bundles': False},
                          {'codegen.generators.cuda.use_atomics': False}]

        # create a powerset (all combinations) of the all_prefs_list
        all_prefs_combinations = []
        for dict_tuple in powerset(all_prefs_list):
            all_prefs_combinations.append(dict(item for d in dict_tuple
                                               for item in d.items()))
    else:
        # just one prefs set (depending on other pref arguments)
        all_prefs_combinations = [None]
        prefs_dict = {}
        if args.no_atomics:
            k = 'codegen.generators.cuda.use_atomics'
            v = False
            prefs[k] = v
            prefs_dict[k] = v

        if args.no_bundles:
            k = 'devices.cuda_standalone.push_synapse_bundles'
            v = False
            prefs[k] = v
            prefs_dict[k] = v

        if args.num_blocks is not None:
            k = 'devices.cuda_standalone.parallel_blocks'
            v = args.num_blocks[0]
            prefs[k] = v
            prefs_dict[k] = v

    print "Running with the following prefs combinations:\n"
    if all_prefs_combinations[0] is not None:
        print_prefs_combinations(all_prefs_combinations)
    elif prefs_dict:
        print "1 run with explicitly set preferences:"
        for k, v in prefs_dict.items():
            print "\t\tprefs[{}] = {}".format(k, v)
    else:
        print "1 run with default preferences"
    print "\n"

    return all_prefs_combinations


def check_success(successes, configurations, float_dtypes=None, targets=None):

    if float_dtypes is not None and len(float_dtypes) != 1:
        float_pref = 'core.default_float_dtype'
        new_configurations = []
        for dtype in float_dtypes:
            for c in configurations:
                if c is None:
                    config = {}
                else:
                    config = c.copy()
                config.update({float_pref: dtype.__name__})
                new_configurations.append(config)
        configurations = new_configurations

    print "\nFINISHED ALL RUNS\n"
    if all(successes):
        print "ALL PASSED"
        return True
    else:
        print "{}/{} CONFIGURATIONS FAILED:".format(len(successes) -
                                                    sum(successes),
                                                    len(successes))
        fails = [not b for b in successes]
        print_prefs_combinations(configurations, condition=fails)
        return False


def dict_to_name(prefs_dict, float_dtype=None):
    if (prefs_dict is None or not prefs_dict) and float_dtype is None:
        return 'default'

    if float_dtype is not None:
        if prefs_dict is None:
            prefs_dict = {}
        prefs_dict['core.default_float_dtype'] = float_dtype.__name__

    names = []
    for k, v in prefs_dict.items():
        names.append(k.split('.')[-1] + '_' + str(v))
    return '__'.join(names)
