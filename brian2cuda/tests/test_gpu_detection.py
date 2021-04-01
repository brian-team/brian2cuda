import os
import logging

from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing import assert_raises, assert_equal

from brian2 import prefs, ms, run
from brian2.devices.device import reinit_devices
from brian2.utils.logger import catch_logs
from brian2.core.preferences import PreferenceError
from brian2cuda.utils.gputools import (
    reset_cuda_installation, get_cuda_installation, restore_cuda_installation
)


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_wrong_cuda_path_error():
    # store global _cuda_installation and environment variable before changing them
    cuda_installation_backup = get_cuda_installation()
    cuda_path_env = os.environ['CUDA_PATH']
    # reset cuda installation, such that it will be detected again during `run()`
    reset_cuda_installation()

    # Set wrong CUDA_PATH
    os.environ['CUDA_PATH'] = '/tmp'
    with assert_raises(RuntimeError):
        run(0*ms)

    # restore the cuda installation
    restore_cuda_installation(cuda_installation_backup)
    # reset env variable
    if cuda_path_env is None:
        del os.environ['CUDA_PATH']
    else:
        os.environ['CUDA_PATH'] = cuda_path_env


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_manual_setting_compute_capability():
    cc_before = prefs.codegen.generators.cuda.compute_capability

    compute_capability_pref = '3.5'
    prefs.codegen.generators.cuda.compute_capability = float(compute_capability_pref)
    with catch_logs(log_level=logging.INFO) as logs:
        run(0*ms)
    log_start = "Compiling device code for compute capability "
    log_start_num_chars = len(log_start)
    for log in logs:
        if log[2].startswith(log_start):
            compute_capability = log[2][log_start_num_chars:log_start_num_chars + 3]
            assert_equal(compute_capability_pref, compute_capability)

    # restore pref
    prefs.codegen.generators.cuda.compute_capability = cc_before


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_unsupported_compute_capability_error():
    cc_before = prefs.codegen.generators.cuda.compute_capability

    prefs.codegen.generators.cuda.compute_capability = 2.0
    with assert_raises(NotImplementedError):
        run(0*ms)

    # restore pref
    prefs.codegen.generators.cuda.compute_capability = cc_before


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_warning_compute_capability_set_twice():
    cc_before = prefs.codegen.generators.cuda.compute_capability

    prefs.codegen.generators.cuda.compute_capability = 3.5
    arch_flag = '-arch=sm_37'
    prefs.codegen.cuda.extra_compile_args_nvcc.append(arch_flag)
    with catch_logs() as logs:
        run(0*ms)

    assert len(logs) == 1, logs
    log = logs[0]
    assert log[0] == "WARNING"
    assert log[1] == "brian2.devices.cuda_standalone"
    assert log[2].startswith("GPU architecture for compilation was specified via ")

    # restore prefs
    prefs.codegen.generators.cuda.compute_capability = cc_before
    prefs.codegen.cuda.extra_compile_args_nvcc.remove(arch_flag)
