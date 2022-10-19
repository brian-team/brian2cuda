import sys
import os
import traceback
from io import StringIO

import brian2
from brian2 import prefs
import brian2cuda

def run(test_standalone=['cuda_standalone'], long_tests=False, reset_preferences=True,
        fail_for_not_implemented=False, test_in_parallel=False, build_options=None,
        extra_test_dirs=None, float_dtype=None, quiet=True, debug=False,
        additional_args=[], threads=None):
    """
    Run the brian2cuda test suite. This includes all standalone-compatible tests from
    the brian2 test suite. Needs an installation of the pytest testing tool.

    For testing, the preferences will be reset to the default preferences.
    After testing, the user preferences will be restored.


    Parameters
    ----------
    test_standalone : list of str, optional
        Specify a list of standlone devices to run the tests for. Should be the names of
        standalone modes (e.g. ``'cuda_standalone'``) and expects that a device of that
        name and an accordingly named "simple" device (e.g. ``'cuda_standalone_simple'``
        exists that can be used for testing (see `CUDAStandaloneSimpleDevice` for
        details. Defaults to ``['cuda_standalone']``.
    long_tests : bool, optional
        Whether to run tests that take a long time. Defaults to ``False``.
    reset_preferences : bool, optional
        Whether to reset all preferences to the default preferences before running the
        test suite. Defaults to ``True`` to get test results independent of the user's
        preference settings but can be switched off when the preferences are actually
        necessary to pass the tests (e.g. for device-specific settings).
    fail_for_not_implemented : bool, optional
        Whether to fail for tests raising a `NotImplementedError`. Defaults to
        ``False``, because some of the Brian2 tests can't pass in Brian2CUDA due to the
        way the tests are implemented.
    test_in_parallel : bool, optional
        Whether to run multiple tests in parallel (requires ``xdist`` installed). If
        ``True``, use all available CPU threads to run one test per thread in parallel.
        Note: This can lead to increased RAM usage! If this this is enabled, consider
        reducing the number of threads during parallel compilation of each test (via the
        ``threads`` parameter).
        excessive CPU usage during compilation.
    build_options : dict, optional
        Non-default build options that will be passed as arguments to the
        `set_device` call for the device specified in ``test_standalone``.
    extra_test_dirs : list of str or str, optional
        Additional directories as a list of strings (or a single directory as
        a string) that will be searched for additional tests.
    float_dtype : np.dtype, optional
        Set the dtype to use for floating point variables to a value different
        from the default `core.default_float_dtype` setting.
    quiet : bool, optional
        Disable verbose output during test runs.
    debug : bool, optional
        Disable pytest output capture and enable brian2 output.
    additional_args : list of str, optional
        Optional command line arguments to pass to ``pytest``
    disable_compiler_optim : bool, optional
        Disable all compiler optimizations for host and device compiler. This speeds up
        compilation but reduces binary runtimes.
    threads : int, optional
        Number of CPU threads to use for parallel compilation. Default (``None``) uses
        all available threads. Using more threads requires more CPU RAM.
    """

    b2c_dir = os.path.abspath(os.path.dirname(brian2cuda.__file__))
    print(f"Running Brian2CUDA version {brian2cuda.__version__} from {b2c_dir}")

    brian2cuda_tests_dir = os.path.abspath(os.path.dirname(__file__))
    if extra_test_dirs is None:
        extra_test_dirs = [brian2cuda_tests_dir]
    elif isinstance(extra_test_dirs, str):
        extra_test_dirs = [brian2cuda_tests_dir, extra_test_dirs]
    else:
        extra_test_dirs = [brian2cuda_tests_dir, *extra_test_dirs]

    pytest_rootdir = os.path.dirname(b2c_dir)
    pytest_args = [
        # Set the pytest rootdir such that we know the node paths for --deselect
        f'--rootdir={pytest_rootdir}',
        # Set confcutdir, such that all `conftest.py` files inside the brian2 and
        # brian2cuda directories are loaded (this overwrites confcutdir set in brian2's
        # `make_argv`, which stops searching for `conftest.py` files outside the
        # `brian2` directory)
        f'--confcutdir={os.path.commonpath([brian2.__file__, brian2cuda.__file__])}',
    ]

    if quiet:
        # set verbosity to max(?)
        additional_args += ['-vvv']

    if not quiet:
        # set verbosity to max(?)
        pytest_args += ['-vvv']

    extra_build_options = {}
    if debug:
        # disable output capture
        pytest_args += ['-s']
        # enable brian2 output
        extra_build_options = {'with_output': True}

    if build_options:
        extra_build_options.update(**build_options)

    # Deselect some tests from the brian2 test suite
    ignore_brian2_tests = [
        # These tests assert on the number of warning and fail because of warnings
        # from brian2cuda (fixed versions of these tests are added to the brian2cuda
        # test suite)
        'tests/test_neurongroup.py::test_semantics_floor_division'
    ]
    for test in ignore_brian2_tests:
        test_abspath = os.path.abspath(
            os.path.join(os.path.dirname(brian2.__file__), test)
        )
        test_nodeid = os.path.relpath(test_abspath, pytest_rootdir)
        additional_args += ['--deselect', test_nodeid]

    pytest_args += additional_args

    if reset_preferences:
        print('Resetting to default preferences')
        stored_user_prefs = prefs.as_file
        prefs.read_preference_file(StringIO(prefs.defaults_as_file))

    # Surpress warnings from nvcc compiler
    nvcc_compiler_args = ['-Xcudafe "--diag_suppress=declared_but_not_referenced"']

    # Set number of threads for parallel compilation
    if threads is not None:
        print(f'Compiling in parallel with {threads} threads')
        k = 'devices.cpp_standalone.extra_make_args_unix'
        assert '-j' in prefs[k], f"``-j`` is not in default prefs[{k}] anymore"
        new_j = f'-j{threads}'
        prefs[k].remove('-j')
        prefs[k].append(new_j)

    prefs['devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc'].extend(
        nvcc_compiler_args
    )

    try:
        successes = []
        stored_prefs = prefs.as_file
        for target in test_standalone:

            # Reset preference between test runs (since brian2.test modifies them)
            prefs.read_preference_file(StringIO(stored_prefs))

            success = brian2.test(
                codegen_targets=[],
                long_tests=long_tests,
                test_codegen_independent=False,
                test_standalone=target,
                reset_preferences=False,
                fail_for_not_implemented=fail_for_not_implemented,
                test_in_parallel=[target] if test_in_parallel else [],
                extra_test_dirs=extra_test_dirs,
                float_dtype=float_dtype,
                additional_args=pytest_args,
                build_options=extra_build_options
            )

            successes.append(success)

        all_success = all(successes)
        if not all_success:
            num_fails = len(successes) - sum(successes)
            print(
                f"ERROR: Test suite(s) for {num_fails}/{len(successes)} standalone"
                f" targets did not complete successfully (see above)."
            )
        else:
            print(
                f"OK: Test suite(s) for {len(successes)}/{len(successes)} standalone "
                f"targets did complete successfully"
            )
        return all_success

    finally:
        if reset_preferences:
            # Restore the user preferences
            prefs.read_preference_file(StringIO(stored_user_prefs))
            prefs._backup()


if __name__ == "__main__":
    run()
