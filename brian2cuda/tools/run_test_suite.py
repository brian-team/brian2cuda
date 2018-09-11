"""
Run brian2 test suite on standalone architectures
"""

import argparse
parser = argparse.ArgumentParser(description='Run the brian2 testsuite on GPU.')
parser.add_argument('--targets', nargs='*', default=['cuda_standalone'], type=str,
                    choices=['cuda_standalone', 'genn', 'cpp_standalone'],
                    help=("Which codegeneration targets to use, can be multiple. "
                          "Only standalone targets. (default='cuda_standalone')"))
parser.add_argument('--float-dtype', nargs='*', default=['float32', 'float64'],
                    choices=['float32', 'float64'], help=("The "
                    "prefs['core.default_float_dtype'] with which tests should be run."))
parser.add_argument('--all-prefs', action='store_true',
                    help=("Run test suits with all preference combinations."))
parser.add_argument('--no-bundles', action='store_true',
                    help="Don't use bundle pushing")
parser.add_argument('--no-atomics', action='store_true',
                    help="Don't use atomics for effect application")
parser.add_argument('--num-parallel-blocks', default=None, type=int, nargs=1,
                    help="Number of parallel blocks in post neuron blocks structure.")
parser.add_argument('--no-long-tests', action='store_false',
                    help="Set to not run long tests. By default they are run.")
parser.add_argument('--no-reset-prefs', action='store_false',
                    help="Weather to reset prefs between tests or not.")
parser.add_argument('--fail-not-implemented', action='store_true',
                    help="Weather to reset prefs between tests or not.")
parser.add_argument('--test-parallel', nargs='?', const=None, default=[],
                    help="Weather to use multiple cores for testing. Optionally the"
                         "targets for which mpi should be used, can be passes as"
                         "arguments. If none are passes, all are run in parallel.")
parser.add_argument('--jobs', '-j', nargs=1, type=int, default=[12],
                    help="Number of cores to use for compilations, passed as -j option "
                         "to the make commands.")
args = parser.parse_args()

single_prefs_set = any([args.no_atomics, args.no_bundles,
                        args.num_parallel_blocks is not None])
assert not (args.all_prefs and single_prefs_set), \
        "Can't set other preferences in combination with --all-prefs"

import os
from StringIO import StringIO
import socket
import numpy as np

from brian2 import test, prefs
import brian2cuda

if args.no_reset_prefs:
    # reset to default preferences
    prefs.read_preference_file(StringIO(prefs.defaults_as_file))
# Switch off cpp compiler code optimization to get faster compilation times
prefs['codegen.cpp.extra_compile_args_gcc'].extend(['-w', '-O0'])
prefs['codegen.cpp.extra_compile_args_msvc'].extend(['/Od'])
# Surpress some warnings from nvcc compiler
prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-Xcudafe "--diag_suppress=declared_but_not_referenced"'])

prefs['devices.cpp_standalone.extra_make_args_unix'] = ['-j' + str(args.jobs[0])]

if args.all_prefs:
    # create a powerset (all combinations) of the all_prefs_list

    from itertools import chain, combinations

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    all_prefs_list = [{'devices.cuda_standalone.parallel_blocks': 1},
                      {'devices.cuda_standalone.push_synapse_bundles': False},
                      {'codegen.generators.cuda.use_atomics': False}]

    all_prefs_combinations = []
    for dict_tuple in powerset(all_prefs_list):
        all_prefs_combinations.append(dict(item for d in dict_tuple for item in d.items()))
    print "Running test suite for following prefs combinations:\n"
    for n, d in enumerate(all_prefs_combinations):
        print n, ":", d, "\n" 

else:
    # just set the preferences chosen
    all_prefs_combinations = [None]
    if args.no_atomics:
        prefs['codegen.generators.cuda.use_atomics'] = False

    if args.no_bundles:
        prefs['devices.cuda_standalone.push_synapse_bundles'] = False

    if args.num_parallel_blocks is not None:
        prefs['devices.cuda_standalone.parallel_blocks'] = args.num_parallel_blocks[0]

if socket.gethostname() == 'elnath':
    prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_35')
    prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

extra_test_dirs = os.path.abspath(os.path.dirname(brian2cuda.__file__))

if 'genn' in args.targets:
    import brian2genn

if args.test_parallel is None:
    args.test_parallel = args.targets

dtypes = []
for dtype in args.float_dtype:
    dtypes.append(getattr(np, dtype))

stored_prefs = prefs.as_file

for target in args.targets:

    test_in_parallel = []
    if target in args.test_parallel:
        test_in_parallel = [target]
    if target == 'cuda_standalone':
        preference_dictionaries = all_prefs_combinations
    else:
        preference_dictionaries = [None]

    for dtype in dtypes:
        for prefs_dict in preference_dictionaries:

            # reset prefs to stored prefs
            prefs.read_preference_file(StringIO(stored_prefs))

            if prefs_dict is not None:
                print "\nRUNNING test suite on CUDA_STANDALONE with prefs:"
                for k, v in prefs_dict:
                    prefs[k] = v
                    print "\tprefs[{}] = {}".format(k,v)

            test(codegen_targets=[],
                 long_tests=args.no_long_tests,
                 test_codegen_independent=False,
                 test_standalone=target,
                 reset_preferences=False,
                 fail_for_not_implemented=args.fail_not_implemented,
                 test_in_parallel=test_in_parallel,
                 extra_test_dirs=extra_test_dirs,
                 float_dtype=dtype
                )
