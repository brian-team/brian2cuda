"""
Run brian2 test suite on standalone architectures
"""

import argparse
parser = argparse.ArgumentParser(description='Run a subset from the brian2 testsuite on GPU.')
parser.add_argument('test', nargs=1, type=str,
                    help=("Specify the test(s) to run. Has to be in the form of "
                          "package.tests.test_file:test_function "
                          "(e.g. brian2.tests.test_base:test_names)"))
parser.add_argument('--targets', nargs='*', default=['cuda_standalone'], type=str,
                    choices=['cuda_standalone', 'genn', 'cpp_standalone'],
                    help=("Which codegeneration targets to use, can be multiple. "
                          "Only standalone targets. (default=['cuda_standalone'])"))
parser.add_argument('--reset-prefs', action='store_true',
                    help="Weather to reset prefs between tests or not.")
args = parser.parse_args()

import sys
from brian2.devices.device import set_device
import nose

if 'cuda_standalone' in args.targets:
    import brian2cuda
if 'genn' in args.targets:
    import brian2genn

for target in args.targets:
    sys.stderr.write('Running test(s) {} for device {} ...\n'.format(args.test[0], target))
    set_device(target, directory=None, with_output=False)
    argv = ['nosetests', args.test[0],
            '-c=',  # no config file loading
            '-I', '^hears\.py$',
            '-I', '^\.',
            '-I', '^_',
            # Only run standalone tests
            '--nologcapture',
            '--exe']

    success = nose.run(argv=argv)
