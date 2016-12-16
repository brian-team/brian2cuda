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
parser.add_argument('--reset-prefs', action='store_true',
                    help="Weather to reset prefs between tests or not.")
parser.add_argument('--test-parallel', nargs='?', const=None, default=[],
                    help="Weather to use multiple cores for testing. Optionally the"
                         "targets for which mpi should be used, can be passes as"
                         "arguments. If none are passes, all are run in parallel.")
args = parser.parse_args()

from brian2 import test

if 'cuda_standalone' in args.targets:
    import brian2cuda
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
         reset_preferences=args.reset_prefs,
         fail_for_not_implemented=False,
         test_in_parallel=test_in_parallel
        )
  

