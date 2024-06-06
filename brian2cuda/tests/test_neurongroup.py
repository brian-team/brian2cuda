import pytest

from brian2.core.magic import run
from brian2.groups.neurongroup import NeuronGroup
from brian2.synapses import Synapses
from brian2.units import second

@pytest.mark.standalone_compatible
def test_group_variable_set_copy_to_host():

    G = NeuronGroup(1, 'v : 1')
    # uses group_variable_set template
    G.v[:1] = '50'
    # connect template runs on host, requiring G.v on host after the group_variable_set
    # template call above (this tests that data is copied from device to host)
    S = Synapses(G, G)
    S.connect(condition='v_pre == 50')
    run(0*second)
    assert len(S) == 1, len(S)


@pytest.mark.standalone_compatible
def test_group_variable_set_conditional_copy_to_host():

    G = NeuronGroup(1, 'v : 1')
    # uses group_variable_set_conditional template
    G.v['i < 1'] = '50'
    # connect template runs on host, requiring G.v on host after the group_variable_set
    # template call above (this tests that data is copied from device to host)
    S = Synapses(G, G)
    S.connect(condition='v_pre == 50')
    run(0*second)
    assert len(S) == 1, len(S)


if __name__ == '__main__':
    import brian2
    brian2.set_device('cuda_standalone')
    test_group_variable_set_copy_to_host()
    #test_group_variable_set_conditional_copy_to_host()
