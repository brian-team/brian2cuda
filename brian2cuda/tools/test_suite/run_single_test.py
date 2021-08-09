"""
Run brian2 test suite on standalone architectures
"""

import argparse
import utils

parser = argparse.ArgumentParser(description='Run a subset from the brian2 testsuite on GPU.')

parser.add_argument('tests', nargs='*', type=str,
                    help=("Specify the test(s) to run. Has to be in the form "
                          "of package/tests/test_file::test_function "
                          "(e.g. brian2/tests/test_base.py::test_names) or "
                          "package/tests/test_file to run all tests in a "
                          "file. If the ``--brian`` or ``--brian2cuda`` option is set, "
                          "``test`` can be a pattern to match brian2 or brian2cuda "
                          "tests (passed to pytest via ``-k`` option)"))

mutual_exclusive = parser.add_mutually_exclusive_group(required=False)
mutual_exclusive.add_argument(
    '--brian2',
    action='store_true',
    help=(
        "Pass ``test`` string as ``-k`` option to select tests from brian2 test suite."
    )
)
mutual_exclusive.add_argument(
    '--brian2cuda',
    action='store_true',
    help=(
        "Pass ``test`` string as ``-k`` option to select tests from brian2cuda test "
        "suite."
    )
)

parser.add_argument('--only',
                    choices=['single-run', 'multi-run', 'standalone-only'],
                    default=None)

args = utils.parse_arguments(parser)

import sys, os, pytest
from StringIO import StringIO
import numpy as np

import brian2
import brian2cuda
from brian2.devices.device import set_device, reset_device
from brian2.tests import clear_caches, make_argv, PreferencePlugin
from brian2 import prefs
import brian2cuda

all_prefs_combinations = utils.set_preferences(args, prefs)

# target independent preferences
stored_prefs = prefs.as_file

pref_plugin = PreferencePlugin(prefs, fail_for_not_implemented=True)

additional_args = []
# Increase verbosity such that the paths and names of executed tests are shown
additional_args += ['-vvv']
# Set rootdir to directory that has brian2's conftest.py, such that it is laoded for all
# tests (even when outside the brian2 folder)
additional_args += ['--rootdir={}'.format(os.path.dirname(brian2.__file__))]

brian2_dir = os.path.join(os.path.abspath(os.path.dirname(brian2.__file__)))
b2c_dir = os.path.join(os.path.abspath(os.path.dirname(brian2cuda.__file__)), 'tests')


if args.brian2:
    tests = [brian2_dir]
    test_patterns = ' or '.join(args.tests)
    additional_args += ['-k {}'.format(test_patterns)]
elif args.brian2cuda:
    tests = [b2c_dir]
    test_patterns = ' or '.join(args.tests)
    additional_args += ['-k {}'.format(test_patterns)]
else:
    tests = []
    for test in args.tests:
        if test.startswith('brian2cuda/'):
            test = os.path.join(b2c_dir, test)
        elif test[0].startswith('brian2/'):
            test = os.path.join(brian2_dir, test)
        tests.append(test)

all_successes = []
for target in args.targets:

    if target == 'cuda_standalone':
        preference_dictionaries = all_prefs_combinations
    else:
        preference_dictionaries = [None]

    successes = []
    for n, prefs_dict in enumerate(preference_dictionaries):

        # did all tests for this preference combination succeed?
        pref_success = True

        # reset prefs to stored prefs
        prefs.read_preference_file(StringIO(stored_prefs))

        if prefs_dict is not None:
            print "{}. RUN: running on {} with prefs:".format(n + 1, target)
            # print and set preferences
            utils.print_single_prefs(prefs_dict, set_prefs=prefs)
        else:  # None
            print "Running {} with default preferences\n".format(target)
        sys.stdout.flush()

        # backup prefs such that reinit_device in the pytest test teardown resets
        # the preferences to what was set above (in restore_initial_state())
        # TODO after change to pytest, might just use that pytest prefs plugin?
        prefs._backup()

        if args.only is None or args.only == 'single-run':
            print ("Running standalone-compatible standard tests "
                   "(single run statement)\n")
            sys.stdout.flush()
            set_device(target, directory=None, with_output=False)
            argv = make_argv(tests, markers='standalone_compatible and not multiple_runs')
            exit_code = pytest.main(argv + additional_args, plugins=[pref_plugin])
            pref_success = pref_success and (exit_code == 0)

            clear_caches()
            reset_device()

        if args.only is None or args.only == 'multi-run':
            print ("Running standalone-compatible standard tests "
                   "(multiple run statements)\n")
            sys.stdout.flush()
            set_device(target, directory=None, with_output=False,
                       build_on_run=False)
            argv = make_argv(tests, markers='standalone_compatible and multiple-runs')
            exit_code = pytest.main(argv + additional_args, plugins=[pref_plugin])
            pref_success = pref_success and (exit_code == 0)

            clear_caches()
            reset_device()

        if args.only is None or args.only == 'standalone-only':
            print "Running standalone-specific tests\n"
            sys.stdout.flush()
            set_device(target, directory=None, with_output=False)
            argv = make_argv(tests, markers=target)
            exit_code = pytest.main(argv + additional_args, plugins=[pref_plugin])
            pref_success = pref_success and (exit_code == 0)

            successes.append(pref_success)

            clear_caches()
            reset_device()

    print "\nTARGET: {}".format(target.upper())
    all_success = utils.check_success(successes, all_prefs_combinations)
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
