"""
Run brian2 test suite on standalone architectures
"""

import argparse
parser = argparse.ArgumentParser(description='Run the brian2 testsuite on GPU.')
parser.add_argument('--targets', nargs='*', default=['cuda_standalone'], type=str,
                    choices=['cuda_standalone', 'genn', 'cpp_standalone'],
                    help=("Which codegeneration targets to use, can be multiple. "
                          "Only standalone targets. (default='cuda_standalone')"))
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
args = parser.parse_args()

import os
from StringIO import StringIO

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

prefs['devices.cpp_standalone.extra_make_args_unix'] = ['-j12']

extra_test_dirs = os.path.abspath(os.path.dirname(brian2cuda.__file__))

if 'genn' in args.targets:
    import brian2genn

if args.test_parallel is None:
    args.test_parallel = args.targets

for target in args.targets:

    test_in_parallel = []
    if target in args.test_parallel:
        test_in_parallel = [target]

    test(codegen_targets=[],
         long_tests=args.no_long_tests,
         test_codegen_independent=False,
         test_standalone=target,
         reset_preferences=False,
         fail_for_not_implemented=args.fail_not_implemented,
         test_in_parallel=test_in_parallel,
         extra_test_dirs=extra_test_dirs
        )
