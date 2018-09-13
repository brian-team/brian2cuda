"""
Run brian2 test suite on standalone architectures
"""

import argparse
import utils

parser = argparse.ArgumentParser(description='Run the brian2 testsuite on GPU.')

parser.add_argument('--no-long-tests', action='store_false',
                    help="Set to not run long tests. By default they are run.")

parser.add_argument('--fail-not-implemented', action='store_true',
                    help="Weather to reset prefs between tests or not.")

parser.add_argument('--test-parallel', nargs='?', const=None, default=[],
                    help="Weather to use multiple cores for testing. Optionally the"
                         "targets for which mpi should be used, can be passes as"
                         "arguments. If none are passes, all are run in parallel.")

args, float_dtypes = utils.parse_arguments(parser, float_default=['float64',
                                                                  'float32'])

import os, sys
from StringIO import StringIO

from brian2 import test, prefs
import brian2cuda

all_prefs_combinations = utils.set_preferences(args, prefs)

extra_test_dirs = os.path.abspath(os.path.dirname(brian2cuda.__file__))

if args.test_parallel is None:
    args.test_parallel = args.targets

stored_prefs = prefs.as_file

all_successes = []
for target in args.targets:

    test_in_parallel = []
    if target in args.test_parallel:
        test_in_parallel = [target]

    if target == 'cuda_standalone':
        preference_dictionaries = all_prefs_combinations
    else:
        preference_dictionaries = [None]
    successes = []
    for dtype in float_dtypes:
        for n, prefs_dict in enumerate(preference_dictionaries):

            # reset prefs to stored prefs
            prefs.read_preference_file(StringIO(stored_prefs))

            if prefs_dict is not None:
                print ("{}. RUN: test suite on CUDA_STANDALONE with prefs:"
                       "".format(n + 1))
                if not prefs_dict:
                    print "default preferences"
                else:
                    for k, v in prefs_dict.items():
                        prefs[k] = v
                        print "\tprefs[{}] = {}".format(k,v)

            success = test(codegen_targets=[],
                           long_tests=args.no_long_tests,
                           test_codegen_independent=False,
                           test_standalone=target,
                           reset_preferences=False,
                           fail_for_not_implemented=args.fail_not_implemented,
                           test_in_parallel=test_in_parallel,
                           extra_test_dirs=extra_test_dirs,
                           float_dtype=dtype)

            successes.append(success)

    print "\nTARGET: {}".format(target.upper())
    all_success = utils.check_success(successes, all_prefs_combinations,
                                      float_dtypes=float_dtypes)
    all_successes.append(all_success)

if len(args.targets) > 1:
    print "\nFINISHED ALL TARGETS"

    if all(all_successes):
        print "\nALL TARGETS PASSED"
    else:
        print "\n{}/{} TARGETS FAILED:".format(sum(all_successes) -
                                               len(all_successes),
                                               len(all_successes))
        for n, target in enumerate(args.targets):
            if not all_successes[n]:
                print "\t{} failed.".format(target)
        sys.exit(1)

else:
    if not all_successes[0]:
        sys.exit(1)
