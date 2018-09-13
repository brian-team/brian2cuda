"""
Run brian2 test suite on standalone architectures
"""

import argparse
import utils

parser = argparse.ArgumentParser(description='Run a subset from the brian2 testsuite on GPU.')

parser.add_argument('test', nargs='*', type=str,
                    help=("Specify the test(s) to run. Has to be in the form of "
                          "package.tests.test_file:test_function "
                          "(e.g. brian2.tests.test_base:test_names) or "
                          "package.tests.test_file to run all tests in a "
                          "file."))

args, float_dtypes = utils.parse_arguments(parser, float_default=['float64'])

import sys, nose
from StringIO import StringIO
import numpy as np

from brian2.devices.device import set_device
from brian2.tests import make_argv
from brian2 import prefs

all_prefs_combinations = utils.set_preferences(args, prefs)

stored_prefs = prefs.as_file

all_successes = []
for target in args.targets:
    successes = []
    for dtype in float_dtypes:

        # reset prefs to stored prefs
        prefs.read_preference_file(StringIO(stored_prefs))

        if target == 'cuda_standalone':
            preference_dictionaries = all_prefs_combinations
        else:
            preference_dictionaries = [None]

        # set float type
        prefs['core.default_float_dtype'] = dtype

        sys.stderr.write('Running test(s) {} for device {} with float type {} '
                         '...\n'.format(args.test, target, dtype))
        set_device(target, directory=None, with_output=False)

        argv = make_argv(args.test, 'standalone-compatible,!codegen-independent')

        successes.append(nose.run(argv=argv))

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

elif not all_successes[0]:
    sys.exit(1)
