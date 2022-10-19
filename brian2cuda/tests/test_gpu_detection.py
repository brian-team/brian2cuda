import os
import logging
from io import StringIO

import pytest
from numpy.testing import assert_equal

from brian2 import prefs, ms, run, set_device
from brian2.utils.logger import catch_logs
from brian2.core.preferences import PreferenceError
from brian2cuda.utils.gputools import (
    reset_cuda_installation,
    get_cuda_installation,
    restore_cuda_installation,
    reset_gpu_selection,
    get_gpu_selection,
    restore_gpu_selection,
)


### Pytest fixtures ###
# Detecting the CUDA installation and GPU to use is happening only once and sets global
# variables, such that requesting information will not repeat the detection process. For
# testing the detection process, we need to reset the global variables with fixtures.
@pytest.fixture()
def reset_cuda_detection():
    # these function store, reset and restore the global _cuda_installation dictionary
    backup = get_cuda_installation()
    reset_cuda_installation()
    yield
    restore_cuda_installation(backup)


@pytest.fixture()
def reset_gpu_detection():
    # these function store, reset and restore the global _cuda_installation dictionary
    backup = get_gpu_selection()
    reset_gpu_selection()
    yield
    restore_gpu_selection(backup)


@pytest.fixture()
def use_default_prefs():
    # reset all preferences to their default values
    stored_prefs = prefs.as_file
    prefs.read_preference_file(StringIO(prefs.defaults_as_file))
    prefs._backup()
    yield
    # restore the user preferences
    prefs.read_preference_file(StringIO(stored_prefs))
    prefs._backup()


### Tests ###
@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_wrong_cuda_path_error(reset_cuda_detection, use_default_prefs, monkeypatch):
    set_device("cuda_standalone", directory=None)
    # Set wrong CUDA_PATH
    monkeypatch.setenv("CUDA_PATH", "/tmp")
    with pytest.raises(RuntimeError) as exc:
        run(0*ms)

    exc.match("Couldn't find `nvcc` binary .*")


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_wrong_cuda_path_error2(reset_cuda_detection, use_default_prefs, monkeypatch):
    # When cuda detection is off and compute capability not given, we don't want to
    # raise the `nvcc` not found error
    set_device("cuda_standalone", directory=None)
    prefs.devices.cuda_standalone.cuda_backend.detect_cuda = False
    # Set wrong CUDA_PATH
    monkeypatch.setenv("CUDA_PATH", "/tmp")
    with pytest.raises(RuntimeError) as exc:
        run(0*ms)

    raised_nvcc_not_found = True
    try:
        exc.match("Couldn't find `nvcc` binary .*")
    except AssertionError:
        raised_nvcc_not_found = False

    if raised_nvcc_not_found:
        raise AssertionError(
            "Raised `nvcc` not found error even though CUDA detection was disabled"
        )


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_wrong_cuda_path_warning(reset_cuda_detection, use_default_prefs, monkeypatch):
    # When cuda detection is off, gpu detection is off and gpu_id and compute_capability
    # are set, we should get a warning that nvcc is not found (but no error)
    set_device("cuda_standalone", directory=None, compile=False)
    prefs.devices.cuda_standalone.cuda_backend.detect_cuda = False
    prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 6.1
    # Set wrong CUDA_PATH
    monkeypatch.setenv("CUDA_PATH", "/tmp")

    with catch_logs() as logs:
        run(0*ms)

    assert len(logs) == 1, logs
    log = logs[0]
    assert log[0] == "WARNING"
    assert log[1] == "brian2.devices.cuda_standalone"
    assert log[2].startswith("Couldn't find `nvcc` binary ")


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_manual_setting_compute_capability(reset_gpu_detection):
    set_device("cuda_standalone", directory=None)
    compute_capability_pref = '3.5'
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = float(compute_capability_pref)
    with catch_logs(log_level=logging.INFO) as logs:
        run(0*ms)
    log_start = "Compiling device code for compute capability "
    log_start_num_chars = len(log_start)
    for log in logs:
        if log[2].startswith(log_start):
            compute_capability = log[2][log_start_num_chars:log_start_num_chars + 3]
            assert_equal(compute_capability_pref, compute_capability)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_unsupported_compute_capability_error(reset_gpu_detection):
    set_device("cuda_standalone", directory=None)
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 2.0
    with pytest.raises(NotImplementedError):
        run(0*ms)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_warning_compute_capability_set_twice(reset_gpu_detection, use_default_prefs):
    set_device("cuda_standalone", directory=None)
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 3.5
    prefs.devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc.append('-arch=sm_37')
    with catch_logs() as logs:
        run(0*ms)

    assert len(logs) == 1, logs
    log = logs[0]
    assert log[0] == "WARNING"
    assert log[1] == "brian2.devices.cuda_standalone"
    assert log[2].startswith("GPU architecture for compilation was specified via ")


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_no_gpu_detection_preference_error(reset_gpu_detection, use_default_prefs):
    set_device("cuda_standalone", directory=None)
    # reset cuda installation, such that it will be detected again during `run()`
    prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    # needs setting gpu_id and compute_capability as well
    with pytest.raises(PreferenceError):
        run(0*ms)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_no_gpu_detection_preference(reset_gpu_detection, use_default_prefs):
    set_device("cuda_standalone", directory=None)
    # Test that disabling gpu detection works when setting gpu_id and compute_capability
    prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 6.1
    run(0*ms)
