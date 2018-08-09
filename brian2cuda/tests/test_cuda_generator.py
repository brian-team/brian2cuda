from nose import with_setup, SkipTest
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_raises
import numpy as np
import logging

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices, set_device
from brian2.tests.test_synapses import permutation_analysis_good_examples
from brian2.utils.stringtools import get_identifiers, deindent

import brian2cuda
from brian2cuda.cuda_generator import CUDACodeGenerator

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_default_function_implementations():
    ''' Test that all default functions work as expected '''
    # NeuronGroup variables are set in device code
    # Synapses are generated in host code
    # int function arguments use the template code
    # double arguments use the overloaded function

    ### sin
    # Promotes int to float in template device code
    G = NeuronGroup(1, 'v: 1')
    G.v = 'sin(i)'

    # Uses int overloaded C++ function in template host code
    S = Synapses(G, G, 'w: 1')
    S.connect(condition='sin(i)==0')

    # Uses double overloaded func (device code)
    G1 = NeuronGroup(1, 'v: 1')
    G1.v = 'sin(i*1.0)'

    # Uses double overloaded func (host code)
    S1 = Synapses(G, G, 'w: 1')
    S1.connect(condition='sin(i*1.0)==0')


    ### cos
    G2 = NeuronGroup(1, 'v: 1')
    G2.v = 'cos(i)'

    S2 = Synapses(G, G, 'w: 1')
    S2.connect(condition='cos(i)==1')

    G3 = NeuronGroup(1, 'v: 1')
    G3.v = 'cos(i*1.0)'

    S3 = Synapses(G, G, 'w: 1')
    S3.connect(condition='cos(i*1.0)==1')


    ### tan
    G4 = NeuronGroup(1, 'v: 1')
    G4.v = 'tan(i)'

    S4 = Synapses(G, G, 'w: 1')
    S4.connect(condition='tan(i)==0')

    G5 = NeuronGroup(1, 'v: 1')
    G5.v = 'tan(i*1.0)'

    S5 = Synapses(G, G, 'w: 1')
    S5.connect(condition='tan(i*1.0)==0')


    ### sinh
    G6 = NeuronGroup(1, 'v: 1')
    G6.v = 'sinh(i)'

    S6 = Synapses(G, G, 'w: 1')
    S6.connect(condition='sinh(i)==0')

    G7 = NeuronGroup(1, 'v: 1')
    G7.v = 'sinh(i*1.0)'

    S7 = Synapses(G, G, 'w: 1')
    S7.connect(condition='sinh(i*1.0)==0')


    ### cosh
    G8 = NeuronGroup(1, 'v: 1')
    G8.v = 'cosh(i)'

    S8 = Synapses(G, G, 'w: 1')
    S8.connect(condition='cosh(i)==1')

    G9 = NeuronGroup(1, 'v: 1')
    G9.v = 'cosh(i*1.0)'

    S9 = Synapses(G, G, 'w: 1')
    S9.connect(condition='cosh(i*1.0)==1')


    ### tanh
    G10 = NeuronGroup(1, 'v: 1')
    G10.v = 'tanh(i)'

    S10 = Synapses(G, G, 'w: 1')
    S10.connect(condition='tanh(i)==0')

    G11 = NeuronGroup(1, 'v: 1')
    G11.v = 'tanh(i*1.0)'

    S11 = Synapses(G, G, 'w: 1')
    S11.connect(condition='tanh(i*1.0)==0')


    ### exp
    G12 = NeuronGroup(1, 'v: 1')
    G12.v = 'exp(i)'

    S12 = Synapses(G, G, 'w: 1')
    S12.connect(condition='exp(i)==1')

    G13 = NeuronGroup(1, 'v: 1')
    G13.v = 'exp(i*1.0)'

    S13 = Synapses(G, G, 'w: 1')
    S13.connect(condition='exp(i*1.0)==1')


    ### log
    G14 = NeuronGroup(1, 'v: 1')
    G14.v = 'log(i+1)'

    S14 = Synapses(G, G, 'w: 1')
    S14.connect(condition='log(i+1)==0')

    G15 = NeuronGroup(1, 'v: 1')
    G15.v = 'log(i+1.0)'

    S15 = Synapses(G, G, 'w: 1')
    S15.connect(condition='log(i+1.0)==0')


    ### log10
    G16 = NeuronGroup(1, 'v: 1')
    G16.v = 'log10(i+1)'

    S16 = Synapses(G, G, 'w: 1')
    S16.connect(condition='log10(i+1)==0')

    G17 = NeuronGroup(1, 'v: 1')
    G17.v = 'log10(i+1.0)'

    S17 = Synapses(G, G, 'w: 1')
    S17.connect(condition='log10(i+1.0)==0')


    ### sqrt
    G18 = NeuronGroup(1, 'v: 1')
    G18.v = 'sqrt(i)'

    S18 = Synapses(G, G, 'w: 1')
    S18.connect(condition='sqrt(i)==0')

    G19 = NeuronGroup(1, 'v: 1')
    G19.v = 'sqrt(i*1.0)'

    S19 = Synapses(G, G, 'w: 1')
    S19.connect(condition='sqrt(i*1.0)==0')


    ### ceil
    G20 = NeuronGroup(1, 'v: 1')
    G20.v = 'ceil(i)'

    S20 = Synapses(G, G, 'w: 1')
    S20.connect(condition='ceil(i)==0')

    G21 = NeuronGroup(1, 'v: 1')
    G21.v = 'ceil(i*1.0)'

    S21 = Synapses(G, G, 'w: 1')
    S21.connect(condition='ceil(i*1.0)==0')


    ### floor
    G22 = NeuronGroup(1, 'v: 1')
    G22.v = 'floor(i)'

    S22 = Synapses(G, G, 'w: 1')
    S22.connect(condition='floor(i)==0')

    G23 = NeuronGroup(1, 'v: 1')
    G23.v = 'floor(i*1.0)'

    S23 = Synapses(G, G, 'w: 1')
    S23.connect(condition='floor(i*1.0)==0')


    ### arcsin
    G24 = NeuronGroup(1, 'v: 1')
    G24.v = 'arcsin(i)'

    S24 = Synapses(G, G, 'w: 1')
    S24.connect(condition='arcsin(i)==0')

    G25 = NeuronGroup(1, 'v: 1')
    G25.v = 'arcsin(i*1.0)'

    S25 = Synapses(G, G, 'w: 1')
    S25.connect(condition='arcsin(i*1.0)==0')


    ### arccos
    G26 = NeuronGroup(1, 'v: 1')
    G26.v = 'arccos(i+1)'

    S26 = Synapses(G, G, 'w: 1')
    S26.connect(condition='arccos(i+1)==0')

    G27 = NeuronGroup(1, 'v: 1')
    G27.v = 'arccos(i+1.0)'

    S27 = Synapses(G, G, 'w: 1')
    S27.connect(condition='arccos(i+1.0)==0')


    ### arctan
    G28 = NeuronGroup(1, 'v: 1')
    G28.v = 'arctan(i)'

    S28 = Synapses(G, G, 'w: 1')
    S28.connect(condition='arctan(i)==0')

    G29 = NeuronGroup(1, 'v: 1')
    G29.v = 'arctan(i*1.0)'

    S29 = Synapses(G, G, 'w: 1')
    S29.connect(condition='arctan(i*1.0)==0')


    ### abs
    G30 = NeuronGroup(1, 'v: 1')
    G30.v = 'abs(i-1)'

    S30 = Synapses(G, G, 'w: 1')
    S30.connect(condition='abs(i-1)==1')

    G31 = NeuronGroup(1, 'v: 1')
    G31.v = 'abs(i-1.0)'

    S31 = Synapses(G, G, 'w: 1')
    S31.connect(condition='abs(i-1.0)==1')


    ### int
    G32 = NeuronGroup(1, 'v: 1')
    G32.v = 'int(i>0)'

    S32 = Synapses(G, G, 'w: 1')
    S32.connect(condition='int(i>0)==0')

    G33 = NeuronGroup(1, 'v: 1')
    G33.v = 'int(i+0.1)'

    S33 = Synapses(G, G, 'w: 1')
    S33.connect(condition='int(i+0.1)==0')


    ### clip
    G34 = NeuronGroup(1, 'v: 1')
    G34.v = 'clip(i+1,-1,0)'

    S34 = Synapses(G, G, 'w: 1')
    S34.connect(condition='clip(i+1,-1,0)==0')


    G35 = NeuronGroup(1, 'v: 1')
    G35.v = 'clip(i+1,-1.0,0.0)'

    S35 = Synapses(G, G, 'w: 1')
    S35.connect(condition='clip(i+1,-1.0,0.0)==0')


    ### sign
    G36 = NeuronGroup(1, 'v: 1')
    G36.v = 'sign(i+1)'

    S36 = Synapses(G, G, 'w: 1')
    S36.connect(condition='sign(i+1)==1')


    ### timestep
    G38 = NeuronGroup(1, 'v: 1')
    G38.v = 'timestep(0.1*ms, 0.001*ms)'

    S38 = Synapses(G, G, 'w: 1')
    S38.connect(condition='timestep(0.1*ms, 0.001*ms) == 100')


    run(0*ms)

    assert_allclose([G38.v[0]], [100])
    assert_allclose([S38.N[:]], [1])

    assert_allclose([G36.v[0]], [1])
    assert_allclose([S36.N[:]], [1])

    assert_allclose([G34.v[0], G35.v[0]], [0,0])
    assert_allclose([S34.N[:], S35.N[:]], [1,1])

    assert_allclose([G32.v[0], G33.v[0]], [0, 0])
    assert_allclose([S32.N[:], S33.N[:]], [1, 1])

    assert_allclose([G30.v[0], G31.v[0]], [1, 1])
    assert_allclose([S30.N[:], S31.N[:]], [1, 1])

    assert_allclose([G28.v[0], G29.v[0]], [0, 0])
    assert_allclose([S28.N[:], S29.N[:]], [1, 1])

    assert_allclose([G26.v[0], G27.v[0]], [0, 0])
    assert_allclose([S26.N[:], S27.N[:]], [1, 1])

    assert_allclose([G24.v[0], G25.v[0]], [0, 0])
    assert_allclose([S24.N[:], S25.N[:]], [1, 1])

    assert_allclose([G22.v[0], G23.v[0]], [0, 0])
    assert_allclose([S22.N[:], S23.N[:]], [1, 1])

    assert_allclose([G20.v[0], G21.v[0]], [0, 0])
    assert_allclose([S20.N[:], S21.N[:]], [1, 1])

    assert_allclose([G18.v[0], G19.v[0]], [0, 0])
    assert_allclose([S18.N[:], S19.N[:]], [1, 1])

    assert_allclose([G16.v[0], G17.v[0]], [0, 0])
    assert_allclose([S16.N[:], S17.N[:]], [1, 1])

    assert_allclose([G14.v[0], G15.v[0]], [0, 0])
    assert_allclose([S14.N[:], S15.N[:]], [1, 1])

    assert_allclose([G12.v[0], G13.v[0]], [1, 1])
    assert_allclose([S12.N[:], S13.N[:]], [1, 1])

    assert_allclose([G10.v[0], G11.v[0]], [0, 0])
    assert_allclose([S10.N[:], S11.N[:]], [1, 1])

    assert_allclose([G8.v[0], G9.v[0]], [1, 1])
    assert_allclose([S8.N[:], S9.N[:]], [1, 1])

    assert_allclose([G6.v[0], G7.v[0]], [0, 0])
    assert_allclose([S6.N[:], S7.N[:]], [1, 1])

    assert_allclose([G4.v[0], G5.v[0]], [0, 0])
    assert_allclose([S4.N[:], S5.N[:]], [1, 1])

    assert_allclose([G2.v[0], G3.v[0]], [1, 1])
    assert_allclose([S2.N[:], S3.N[:]], [1, 1])

    assert_allclose([G.v[0], G1.v[0]], [0, 0])
    assert_allclose([S.N[:], S1.N[:]], [1, 1])


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_default_function_convertion_preference():

    if prefs.core.default_float_dtype is np.float32:
        raise SkipTest('Need double precision for this test')

    set_device('cuda_standalone', directory=None)

    unrepresentable_int = 2**24 + 1  # can't be represented as 32bit float

    prefs.codegen.generators.cuda.default_functions_integral_convertion = np.float32
    G = NeuronGroup(1, 'v: 1')
    G.variables.add_array('myarr', dtype=np.int32, size=1)
    G.variables['myarr'].set_value(unrepresentable_int)
    G.v = 'floor(myarr)'.format(unrepresentable_int)

    prefs.codegen.generators.cuda.default_functions_integral_convertion = np.float64
    G2 = NeuronGroup(1, 'v: 1')
    G2.variables.add_array('myarr', dtype=np.int32, size=1)
    G2.variables['myarr'].set_value(unrepresentable_int)
    G2.v = 'floor(myarr)'.format(unrepresentable_int)

    run(0*ms)

    assert G.v[0] != unrepresentable_int, G.v[0]
    assert G2.v[0] == unrepresentable_int, '{} != {}'.format(G2.v[0], unrepresentable_int)


@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_default_function_convertion_warnings():

    set_device('cuda_standalone', directory=None)

    BrianLogger._log_messages.clear()
    with catch_logs() as logs1:
        # warning
        G1 = NeuronGroup(1, 'v: 1')
        G1.variables.add_array('myarr', dtype=np.int64, size=1)
        G1.v = 'sin(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs2:
        # warning
        G2 = NeuronGroup(1, 'v: 1')
        G2.variables.add_array('myarr', dtype=np.uint64, size=1)
        G2.v = 'cos(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs3:
        # no warning
        G3 = NeuronGroup(1, 'v: 1')
        G3.variables.add_array('myarr', dtype=np.int32, size=1)
        G3.v = 'tan(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs4:
        # no warning
        G4 = NeuronGroup(1, 'v: 1')
        G4.variables.add_array('myarr', dtype=np.uint32, size=1)
        G4.v = 'arcsin(i*myarr)'

    prefs.codegen.generators.cuda.default_functions_integral_convertion = np.float32

    BrianLogger._log_messages.clear()
    with catch_logs() as logs5:
        # warning
        G5 = NeuronGroup(1, 'v: 1')
        G5.variables.add_array('myarr', dtype=np.int32, size=1)
        G5.v = 'log(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs6:
        # warning
        G6 = NeuronGroup(1, 'v: 1')
        G6.variables.add_array('myarr', dtype=np.uint32, size=1)
        G6.v = 'log10(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs7:
        # warning
        G7 = NeuronGroup(1, 'v: 1')
        G7.variables.add_array('myarr', dtype=np.int64, size=1)
        G7.v = 'floor(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs8:
        # warning
        G8 = NeuronGroup(1, 'v: 1')
        G8.variables.add_array('myarr', dtype=np.uint64, size=1)
        G8.v = 'ceil(i*myarr)'


    run(0*ms)

    assert len(logs1) == 1, len(logs1)
    assert logs1[0][0] == 'WARNING'
    assert logs1[0][1] == 'brian2.codegen.generators.cuda_generator'
    assert len(logs2) == 1, len(logs2)
    assert logs2[0][0] == 'WARNING'
    assert logs2[0][1] == 'brian2.codegen.generators.cuda_generator'
    assert len(logs3) == 0, len(logs3)
    assert len(logs4) == 0, len(logs4)
    assert len(logs5) == 1, len(logs5)
    assert logs5[0][0] == 'WARNING'
    assert logs5[0][1] == 'brian2.codegen.generators.cuda_generator'
    assert len(logs6) == 1, len(logs6)
    assert logs6[0][0] == 'WARNING'
    assert logs6[0][1] == 'brian2.codegen.generators.cuda_generator'
    assert len(logs7) == 1, len(logs7)
    assert logs7[0][0] == 'WARNING'
    assert logs7[0][1] == 'brian2.codegen.generators.cuda_generator'
    assert len(logs8) == 1, len(logs8)
    assert logs8[0][0] == 'WARNING'
    assert logs8[0][1] == 'brian2.codegen.generators.cuda_generator'



@attr('cuda_standalone', 'standalone-only')
@with_setup(teardown=reinit_devices)
def test_atomics_parallelisation():
    # Adapted from brian2.test_synapses:test_ufunc_at_vectorisation()
    for n, code in enumerate(permutation_analysis_good_examples):
        should_be_able_to_use_ufunc_at = not 'NOT_UFUNC_AT_VECTORISABLE' in code
        if should_be_able_to_use_ufunc_at:
            use_ufunc_at_list = [False, True]
        else:
            use_ufunc_at_list = [True]
        code = deindent(code)
        vars = get_identifiers(code)
        vars_src = []
        vars_tgt = []
        vars_syn = []
        vars_shared = []
        vars_const = {}
        for var in vars:
            if var.endswith('_pre'):
                vars_src.append(var[:-4])
            elif var.endswith('_post'):
                vars_tgt.append(var[:-5])
            elif var.endswith('_syn'):
                vars_syn.append(var[:-4])
            elif var.endswith('_shared'):
                vars_shared.append(var[:-7])
            elif var.endswith('_const'):
                vars_const[var[:-6]] = 42
        eqs_src = '\n'.join(var+':1' for var in vars_src)
        eqs_tgt = '\n'.join(var+':1' for var in vars_tgt)
        eqs_syn = '\n'.join(var+':1' for var in vars_syn)
        eqs_syn += '\n' + '\n'.join(var+':1 (shared)' for var in vars_shared)
        origvals = {}
        endvals = {}
        group_size = 1000
        syn_size = group_size**2
        try:
            BrianLogger._log_messages.clear()
            with catch_logs(log_level=logging.INFO) as caught_logs:
                for use_ufunc_at in use_ufunc_at_list:
                    set_device('cuda_standalone', directory=None,
                               compile=True, run=True, debug=False)
                    CUDACodeGenerator._use_atomics = use_ufunc_at
                    src = NeuronGroup(group_size, eqs_src, threshold='True', name='src')
                    tgt = NeuronGroup(group_size, eqs_tgt, name='tgt')
                    syn = Synapses(src, tgt, eqs_syn,
                                   on_pre=code.replace('_syn', '').replace('_const', '').replace('_shared', ''),
                                   name='syn', namespace=vars_const)
                    syn.connect()
                    for G, vars in [(src, vars_src), (tgt, vars_tgt), (syn, vars_syn)]:
                        for var in vars:
                            fullvar = var+G.name
                            if fullvar in origvals:
                                G.state(var)[:] = origvals[fullvar]
                            else:
                                if isinstance(G, Synapses):
                                    val = rand(syn_size)
                                else:
                                    val = rand(len(G))
                                G.state(var)[:] = val
                                origvals[fullvar] = val.copy()
                    Network(src, tgt, syn).run(5*defaultclock.dt)
                    for G, vars in [(src, vars_src), (tgt, vars_tgt), (syn, vars_syn)]:
                        for var in vars:
                            fullvar = var+G.name
                            val = G.state(var)[:].copy()
                            if fullvar in endvals:
                                assert_allclose(val, endvals[fullvar],
                                                err_msg='%d: %s' % (n, code),
                                                rtol=1e-5)
                            else:
                                endvals[fullvar] = val
                    device.reinit()
                    device.activate()
                cuda_generator_messages = [l for l in caught_logs
                                           if l[1]=='brian2.codegen.generators.cuda_generator']
                if should_be_able_to_use_ufunc_at:
                    assert len(cuda_generator_messages) == 0, cuda_generator_messages
                else:
                    assert len(cuda_generator_messages) == 1, cuda_generator_messages
                    log_lev, log_mod, log_msg = cuda_generator_messages[0]
                    assert log_msg.startswith('Failed to parallelise code'), log_msg
        finally:
            CUDACodeGenerator._use_atomics = False  #restore it
            device.reinit()
            device.activate()


if __name__ == '__main__':
    test_default_function_implementations()
    test_default_function_convertion_preference()
    test_default_function_convertion_warnings()
    test_atomics_parallelisation()
