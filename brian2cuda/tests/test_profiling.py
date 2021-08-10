import pytest

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import set_device

import brian2cuda


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_changing_profile_arg():
    set_device('cuda_standalone', build_on_run=False)
    G = NeuronGroup(10000, 'v : 1')
    op1 = G.run_regularly('v = exp(-v)', name='op1')
    op2 = G.run_regularly('v = exp(-v)', name='op2')
    op3 = G.run_regularly('v = exp(-v)', name='op3')
    op4 = G.run_regularly('v = exp(-v)', name='op4')
    # Op 1 is active only during the first profiled run
    # Op 2 is active during both profiled runs
    # Op 3 is active only during the second profiled run
    # Op 4 is never active (only during the unprofiled run)
    op1.active = True
    op2.active = True
    op3.active = False
    op4.active = False
    run(100*defaultclock.dt, profile=True)
    op1.active = True
    op2.active = True
    op3.active = True
    op4.active = True
    run(100*defaultclock.dt, profile=False)
    op1.active = False
    op2.active = True
    op3.active = True
    op4.active = False
    run(100*defaultclock.dt, profile=True)
    device.build(directory=None, with_output=False)
    profiling_dict = dict(magic_network.profiling_info)
    # Note that for now, C++ standalone creates a new CodeObject for every run,
    # which is most of the time unnecessary (this is partly due to the way we
    # handle constants: they are included as literals in the code but they can
    # change between runs). Therefore, the profiling info is potentially
    # difficult to interpret
    assert len(profiling_dict) == 4  # 2 during first run, 2 during last run
    # The two code objects that were executed during the first run
    assert ('op1_codeobject' in profiling_dict and
            profiling_dict['op1_codeobject'] > 0*second)
    assert ('op2_codeobject' in profiling_dict and
            profiling_dict['op2_codeobject'] > 0*second)
    # Four code objects were executed during the second run, but no profiling
    # information was saved
    for name in ['op1_codeobject_1', 'op2_codeobject_1', 'op3_codeobject',
                 'op4_codeobject']:
        assert name not in profiling_dict
    # Two code objects were exectued during the third run
    assert ('op2_codeobject_2' in profiling_dict and
            profiling_dict['op2_codeobject_2'] > 0*second)
    assert ('op3_codeobject_1' in profiling_dict and
            profiling_dict['op3_codeobject_1'] > 0*second)

if __name__ == '__main__':
    test_changing_profile_arg()
