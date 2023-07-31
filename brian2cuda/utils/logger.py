'''
Brian2CUDA's logging system extensions
'''
from brian2.utils.logger import BrianLogger

__all__ = ['suppress_brian2_logs']

def suppress_brian2_logs():
    '''
    Suppress all logs coming from brian2.
    '''
    BrianLogger.suppress_hierarchy('brian2')

report_issue_message = (
    "This should not have happened. Please report this error to "
    "https://github.com/brian-team/brian2cuda/issues/new"
)
