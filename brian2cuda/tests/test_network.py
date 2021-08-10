import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.tests.utils import assert_allclose

import brian2cuda


# Copied from brian2.tests.test_network:test_both_equal (which is 'codegen-independent')
@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_both_equal():

    set_device('cuda_standalone', directory=None, build_on_run=False)
    #check all objects added by Network.add() also have their contained_objects added to 'Network'
    tau = 10*ms
    diff_eqn='''dv/dt = (1-v)/tau : 1'''
    chg_code='''v = 2*v'''

    Ng = NeuronGroup(1,diff_eqn,method='exact')
    M1 = StateMonitor(Ng, 'v', record=True)
    netObj = Network(Ng,M1)
    Ng.run_regularly(chg_code, dt=20*ms)
    netObj.run(100*ms)

    start_scope()
    Ng = NeuronGroup(1,diff_eqn,method='exact')
    M2 = StateMonitor(Ng, 'v', record=True)
    Ng.run_regularly(chg_code, dt=20*ms)
    run(100*ms)

    device.build(direct_call=False, **device.build_options)

    assert (M1.v == M2.v).all()


if __name__ == '__main__':
    test_both_equal()
