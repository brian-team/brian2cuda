import os
import logging

from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing import assert_raises, assert_equal

from brian2 import *
from brian2.devices.device import reinit_devices
from brian2.utils.logger import catch_logs
from brian2.core.preferences import PreferenceError


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_wrong_cuda_path_error():
    try:
        cuda_path = os.environ['CUDA_PATH']
    except KeyError:
        # CUDA_PATH not set
        cuda_path = None

    # Set wrong CUDA_PATH
    os.environ['CUDA_PATH'] = '/tmp'
    assert_raises(RuntimeError, run(0*ms))

    # restore CUDA_PATH to its previous value
    if cuda_path is None:
        del os.environ['CUDA_PATH']
    else:
        os.environ['CUDA_PATH'] = cuda_path

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_manual_setting_compute_capability():
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

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_unsupported_compute_capability_error():
    prefs.codegen.generators.cuda.compute_capability = 2.0
    assert_raises(NotImplementedError, run(0*ms))


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_warning_compute_capability_set_twice():
    prefs.codegen.generators.cuda.compute_capability = 3.5
    prefs.codegen.cuda.extra_compile_args_nvcc.append('-arch=sm_37')
    with catch_logs() as logs:
        run(0*ms)

    assert len(logs) == 1, logs
    log = logs[0]
    assert log[0] == "WARNING"
    assert log[1] == "brian2.devices.cuda_standalone"
    assert log[2].startswith("GPU architecture for compilation was specified via ")
