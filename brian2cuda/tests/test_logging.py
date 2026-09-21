import logging
import re

import pytest

from brian2 import *
from brian2.devices.device import reinit_and_delete, set_device
from brian2.utils.logger import BrianLogger, LOG_LEVELS

import brian2cuda
from brian2cuda.utils.logger import get_codegen_log_level, nvcc_log_flags

_CUDA_TAG = re.compile(r'\[brian2cuda\]\[(DEBUG|INFO|WARNING|ERROR)\]')
_BELOW_ERROR = frozenset({'DEBUG', 'INFO', 'WARNING'})


@pytest.mark.codegen_independent
def test_codegen_log_level_flags():
    # Console level -> -DB2C_LOG_LEVEL (clamped to DEBUG..ERROR)
    previous = BrianLogger.console_handler.level
    try:
        BrianLogger.console_handler.setLevel(LOG_LEVELS['DIAGNOSTIC'])
        assert get_codegen_log_level() == logging.DEBUG
        assert nvcc_log_flags() == [f'-DB2C_LOG_LEVEL={logging.DEBUG}']

        BrianLogger.console_handler.setLevel(LOG_LEVELS['ERROR'])
        assert get_codegen_log_level() == logging.ERROR
        assert nvcc_log_flags() == [f'-DB2C_LOG_LEVEL={logging.ERROR}']

        BrianLogger.console_handler.setLevel(LOG_LEVELS['CRITICAL'])
        assert get_codegen_log_level() == logging.ERROR
    finally:
        BrianLogger.console_handler.setLevel(previous)


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_log_level_gating(capfd, tmp_path):
    # DEBUG should emit below-ERROR CUDA tags; ERROR should not
    previous = BrianLogger.console_handler.level
    try:
        for level, directory in (
            (LOG_LEVELS['DEBUG'], tmp_path / 'debug'),
            (LOG_LEVELS['ERROR'], tmp_path / 'error'),
        ):
            BrianLogger.console_handler.setLevel(level)
            set_device('cuda_standalone', build_on_run=False,
                      directory=str(directory))
            NeuronGroup(1, 'v : 1')
            run(0 * ms)
            capfd.readouterr()  # drop compile noise
            device.build(directory=str(directory), with_output=False,
                         clean=True)
            tags = set(_CUDA_TAG.findall(capfd.readouterr().err))
            reinit_and_delete()

            if level == LOG_LEVELS['DEBUG']:
                assert tags & _BELOW_ERROR
            else:
                assert not (tags & _BELOW_ERROR)
    finally:
        BrianLogger.console_handler.setLevel(previous)
