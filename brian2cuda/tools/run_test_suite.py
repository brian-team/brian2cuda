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
args = parser.parse_args()

from brian2 import test

if 'cuda_standalone' in args.targets:
    import brian2cuda
if 'genn' in args.targets:
    import brian2genn

for target in args.targets:
    test(codegen_targets=[],
         long_tests=args.no_long_tests,
         test_codegen_independent=False,
         test_standalone=target,
         reset_preferences=args.reset_prefs
        )
  

