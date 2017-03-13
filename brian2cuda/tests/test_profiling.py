from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices, set_device

import brian2cuda


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_true():
    # Test that profiling info is not 0
    set_device('cuda_standalone', directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    run(defaultclock.dt, profile=True)

    prof_info = magic_network.profiling_info
    assert any([prof_info[n][1] for n in range(len(prof_info))])

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_blocking():
    # Test that profiling info is not 0
    set_device('cuda_standalone', directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    run(defaultclock.dt, profile='blocking')

    prof_info = magic_network.profiling_info
    assert any([prof_info[n][1] for n in range(len(prof_info))])

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_false():
    # Test that profiling info is not 0
    set_device('cuda_standalone', directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    run(defaultclock.dt, profile=False)

    prof_info = magic_network.profiling_info
    assert all([prof_info[n][1] for n in range(len(prof_info))])

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_error():
    # Test that profiling info is not 0
    set_device('cuda_standalone', directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    assert_raises(ValueError, run(defaultclock.dt, profile='error'))

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_warning():
    # Test that profiling info is not 0
    set_device('cuda_standalone', directory=None)

    prefs.devices.cuda_standalone.profile = True

    G = NeuronGroup(1, 'v:1', threshold='True')

    BrianLogger._log_messages.clear()
    with catch_logs() as logs:
        run(defaultclock.dt, profile=True)

    assert len(logs) == 1, len(logs)
    assert logs[0][0] == 'WARNING'
    assert logs[0][1] == 'brian2.devices.cuda_standalone'


if __name__ == '__main__':
    test_profile_warning()
