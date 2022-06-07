import os
import numpy as np
import pytest
from numpy.testing import assert_equal

from brian2.core.clocks import defaultclock
from brian2.core.magic import run
from brian2.groups.neurongroup import NeuronGroup
from brian2.tests import make_argv
from brian2.tests.utils import assert_allclose
from brian2.utils.logger import catch_logs

# Adapted from brian2/tests/test_neurongroup.py::test_semantics_floor_division
# (brian2 test asserts for 0 warnings, brian2cuda warns for int to float64 conversion)
@pytest.mark.standalone_compatible
def test_semantics_floor_division():
    # See github issues #815 and #661
    G = NeuronGroup(11, '''a : integer
                           b : integer
                           x : 1
                           y : 1
                           fvalue : 1
                           ivalue : integer''',
                    dtype={'a': np.int32, 'b': np.int64,
                           'x': np.float, 'y': np.double})
    int_values = np.arange(-5, 6)
    float_values = np.arange(-5.0, 6.0, dtype=np.double)
    G.ivalue = int_values
    G.fvalue = float_values
    with catch_logs() as l:
        G.run_regularly('''
        a = ivalue//3
        b = ivalue//3
        x = fvalue//3
        y = fvalue//3
        ''')
        run(defaultclock.dt)
    # XXX: brian2 test adapted here for 1 warning
    assert len(l) == 1
    assert_equal(G.a[:], int_values // 3)
    assert_equal(G.b[:], int_values // 3)
    assert_allclose(G.x[:], float_values // 3)
    assert_allclose(G.y[:], float_values // 3)


if __name__ == '__main__':
    import brian2
    brian2.set_device('cuda_standalone')
    test_semantics_floor_division()
