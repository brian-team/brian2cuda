from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices, set_device

import brian2cuda


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_blocking():
    # Test that profiling info is not 0
    set_device('cuda_standalone', profile='blocking', directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    run(defaultclock.dt)

    prof_info = magic_network.profiling_info
    assert any([prof_info[n][1] for n in range(len(prof_info))])

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_True():
    # Test that profiling info is not 0
    set_device('cuda_standalone', profile=True, directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    run(defaultclock.dt)

    prof_info = magic_network.profiling_info
    assert any([prof_info[n][1] for n in range(len(prof_info))])

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_False():
    # Test that profiling info is 0
    set_device('cuda_standalone', profile=False, directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    run(defaultclock.dt)

    prof_info = magic_network.profiling_info
    assert all([prof_info[n][1] for n in range(len(prof_info))])

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_in_run_raises():
    set_device('cuda_standalone', directory=None, build_on_run=False)

    G = NeuronGroup(1, 'v:1', threshold='True')

    assert_raises(TypeError, lambda: run(defaultclock.dt, profile=False))
    assert_raises(TypeError, lambda: run(defaultclock.dt, profile='string'))
    run(defaultclock.dt, profile=True)

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_wrong_raises():
    # Test that profiling info is not 0
    set_device('cuda_standalone', profile='wrong', directory=None)

    G = NeuronGroup(1, 'v:1', threshold='True')

    # error raised in device.build()
    assert_raises(TypeError, lambda: run(defaultclock.dt))

@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_profile_build_raises():
    # Test that profiling info is not 0
    set_device('cuda_standalone', directory=None, build_on_run=False)
    assert_raises(TypeError, lambda: device.build(profile=True))
    assert_raises(TypeError, lambda: device.build(profile=False))
    assert_raises(TypeError, lambda: device.build(profile='string'))

if __name__ == '__main__':
    #test_profile_in_run_raises()
    #test_profile_wrong_raises()
    test_profile_build_raises()
