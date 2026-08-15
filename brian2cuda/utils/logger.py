'''
Brian2CUDA logging helpers.

Map the Brian2 console level to nvcc ``-DB2C_LOG_LEVEL`` for ``B2C_LOG_*``
macros, and re-emit host WARNING/ERROR from ``results/cuda_log.txt`` after the
standalone process exits.
'''
import logging
import os
import re

from brian2.core.preferences import prefs
from brian2.utils.logger import BrianLogger, get_logger

__all__ = [
    'suppress_brian2_logs',
    'get_codegen_log_level',
    'nvcc_log_flags',
    'update_log_flags_stamp',
    'reemit_cuda_log',
    'report_issue_message',
]

logger = get_logger(__name__)

_CUDA_LOG_LINE = re.compile(
    r'^\[brian2cuda\]\[(ERROR|WARNING)\]\s?(.*)$'
)


def get_codegen_log_level():
    '''Brian2 console level clamped to DEBUG..ERROR for CUDA codegen.'''
    if BrianLogger.console_handler is not None:
        level = BrianLogger.console_handler.level
    else:
        level = getattr(
            logging, prefs['logging.console_log_level'].upper(), logging.INFO
        )
    # DIAGNOSTIC to DEBUG; CRITICAL to ERROR
    return min(max(level, logging.DEBUG), logging.ERROR)


def nvcc_log_flags():
    '''nvcc ``-D`` flags for compile-time ``B2C_LOG_*`` gating.'''
    return [f'-DB2C_LOG_LEVEL={get_codegen_log_level()}']


def update_log_flags_stamp(project_dir):
    '''Return True if log -D flags changed since the last build.'''
    stamp_path = os.path.join(project_dir, 'b2c_log_flags.stamp')
    new_stamp = '\n'.join(nvcc_log_flags()) + '\n'
    old_stamp = None
    if os.path.isfile(stamp_path):
        with open(stamp_path, encoding='utf-8') as f:
            old_stamp = f.read()
    with open(stamp_path, 'w', encoding='utf-8') as f:
        f.write(new_stamp)
    return old_stamp is not None and old_stamp != new_stamp


def reemit_cuda_log(results_dir):
    '''Re-emit persisted host WARNING/ERROR via the Brian2 logger.'''
    if not results_dir:
        return
    path = os.path.join(results_dir, 'cuda_log.txt')
    if not os.path.isfile(path):
        return

    emit = {'ERROR': logger.error, 'WARNING': logger.warn}
    level = None
    parts = []

    def flush():
        nonlocal level, parts
        if level is not None and parts:
            emit[level]('\n'.join(parts), name_suffix='cuda_log')
        level, parts = None, []

    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.rstrip('\n')
                match = _CUDA_LOG_LINE.match(line)
                if match:
                    flush()
                    level = match.group(1)
                    parts = [match.group(2)]
                elif level is not None:
                    parts.append(line)
            flush()
    except (IOError, OSError) as ex:
        logger.warn(f"Could not read CUDA log '{path}': {ex}",
                    name_suffix='cuda_log')


def suppress_brian2_logs():
    '''
    Suppress all logs coming from brian2.
    '''
    BrianLogger.suppress_hierarchy('brian2.equations')
    # we want logs from brian2.devices.cuda_standalone
    BrianLogger.suppress_hierarchy('brian2.devices.cpp_standalone')
    BrianLogger.suppress_hierarchy('brian2.devices.device')
    BrianLogger.suppress_hierarchy('brian2.groups')
    BrianLogger.suppress_hierarchy('brian2.core')
    BrianLogger.suppress_hierarchy('brian2.synapses')
    BrianLogger.suppress_hierarchy('brian2.monitors')
    BrianLogger.suppress_hierarchy('brian2.input')
    BrianLogger.suppress_hierarchy('brian2.__init__')
    BrianLogger.suppress_hierarchy('brian2.spatialneuron')
    BrianLogger.suppress_hierarchy('brian2.stateupdater')
    BrianLogger.suppress_hierarchy('brian2.hears')
    # we want logs from brian2.codegen.cuda
    BrianLogger.suppress_hierarchy('brian2.codegen.codeobject')
    BrianLogger.suppress_hierarchy('brian2.codegen.runtime')
    BrianLogger.suppress_hierarchy('brian2.codegen.generators')
    # we want logs from brian2.codegen.generators.cuda_generator
    BrianLogger.suppress_hierarchy('brian2.codegen.generators.cpp_generator')
    BrianLogger.suppress_hierarchy('brian2.codegen.generators.numpy_generator')
    BrianLogger.suppress_hierarchy('brian2.codegen.generators.base')


report_issue_message = (
    "This should not have happened. Please report this error to "
    "https://github.com/brian-team/brian2cuda/issues/new"
)
