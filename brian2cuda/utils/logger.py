'''
Brian2CUDA's logging system extensions
'''
from brian2.utils.logger import BrianLogger

__all__ = ['suppress_brian2_logs']

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
