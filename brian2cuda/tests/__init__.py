import brian2
from brian2.tests import run

#def run(codegen_targets=None, long_tests=False, test_codegen_independent=False,
#        test_standalone=None, test_openmp=False,
#        test_in_parallel=['codegen_independent', 'numpy', 'cython', 'cpp_standalone'],
#        reset_preferences=True, fail_for_not_implemented=True, test_GSL=False,
#        build_options=None, extra_test_dirs=None, float_dtype=None,
#        additional_args=None):
#    pass
#
#def test(codegen_targets=[], test_codegen_independent=False, long_tests=False,
#         test_standalone='cuda_standalone', test_in_parallel=[],
#         extra_test_dirs=os.path.abspath(__file__),
#         **kwargs):
#
#    additional_args = f'--confcutdir={os.path.commonpath([brian2.__file__, brian2cuda.__file__])}',
#
#    run(
#
#        success = test(codegen_targets=[],
#                       long_tests=args.no_long_tests,
#                       test_codegen_independent=False,
#                       test_standalone=target,
#                       reset_preferences=False,
#                       fail_for_not_implemented=not args.skip_not_implemented,
#                       test_in_parallel=test_in_parallel,
#                       extra_test_dirs=extra_test_dirs,
#                       float_dtype=None,
#                       additional_args=additional_args,
#                       build_options=build_options)
