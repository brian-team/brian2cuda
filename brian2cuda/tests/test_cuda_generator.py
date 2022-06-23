import pytest
import numpy as np
import logging

from brian2 import *
from brian2.tests.utils import assert_allclose
from brian2.utils.logger import catch_logs
from brian2.devices.device import set_device
from brian2.tests.test_synapses import permutation_analysis_good_examples
from brian2.utils.stringtools import get_identifiers, deindent

import brian2cuda
from brian2cuda.cuda_generator import CUDACodeGenerator


@pytest.mark.parametrize(
    'func,zero_point,offset',
    [
        # functions without offset (= 0)
        ('sin', 0, 0),
        ('cos', 1, 0),
        ('tan', 0, 0),
        ('sinh', 0, 0),
        ('cosh', 1, 0),
        ('tanh', 0, 0),
        ('exp', 1, 0),
        ('sqrt', 0, 0),
        ('ceil', 0, 0),
        ('floor', 0, 0),
        ('arcsin', 0, 0),
        ('arctan', 0, 0),
        ('int', 0, 0),
        # functions with offset
        ('arccos', 0, 1),
        ('log', 0, 1),
        ('log10', 0, 1),
        ('abs', 1, -1),
        ('sign', 1, 1),
        #
    ]
)
@pytest.mark.standalone_compatible
def test_default_function_implementations(func, zero_point, offset):

    # Promotes int to float in template device code
    G = NeuronGroup(1, 'v: 1')
    G.v = f'{func}(i + {offset})'

    # Uses int overloaded C++ function in template host code
    S = Synapses(G, G, 'w: 1')
    S.connect(condition=f'{func}(i + {offset}) == {zero_point}')

    # Uses double overloaded func (device code)
    G1 = NeuronGroup(1, 'v: 1')
    G1.v = f'{func}(i * 1.0 + {offset})'

    # Uses double overloaded func (host code)
    S1 = Synapses(G, G, 'w: 1')
    S1.connect(condition=f'{func}(i * 1.0 + {offset}) == {zero_point}')

    run(0*ms)

    assert_allclose([G.v[0], G1.v[0]], [zero_point, zero_point])
    assert_allclose([S.N[:], S1.N[:]], [1, 1])


@pytest.mark.standalone_compatible
def test_default_function_implementations_clip():

    G = NeuronGroup(1, 'v: 1')
    G.v = 'clip(i+1,-1,0)'

    S = Synapses(G, G, 'w: 1')
    S.connect(condition='clip(i+1,-1,0)==0')

    G1 = NeuronGroup(1, 'v: 1')
    G1.v = 'clip(i+1,-1.0,0.0)'

    S1 = Synapses(G, G, 'w: 1')
    S1.connect(condition='clip(i+1,-1.0,0.0)==0')

    run(0*ms)

    assert_allclose([G.v[0], G1.v[0]], [0, 0])
    assert_allclose([S.N[:], S1.N[:]], [1, 1])


@pytest.mark.standalone_compatible
def test_default_function_implementations_timestep():

    G = NeuronGroup(1, 'v: 1')
    G.v = 'timestep(0.1*ms, 0.001*ms)'

    S = Synapses(G, G, 'w: 1')
    S.connect(condition='timestep(0.1*ms, 0.001*ms) == 100')

    run(0*ms)

    assert_allclose([G.v[0]], [100])
    assert_allclose([S.N[:]], [1])


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
def test_default_function_convertion_preference():

    if prefs.core.default_float_dtype is np.float32:
        pytest.skip('Need double precision for this test')

    set_device('cuda_standalone', directory=None)

    unrepresentable_int = 2**24 + 1  # can't be represented as 32bit float

    prefs.devices.cuda_standalone.default_functions_integral_convertion = np.float32
    G = NeuronGroup(1, 'v: 1')
    G.variables.add_array('myarr', dtype=np.int32, size=1)
    G.variables['myarr'].set_value(unrepresentable_int)
    G.v = 'floor(myarr)'.format(unrepresentable_int)

    prefs.devices.cuda_standalone.default_functions_integral_convertion = np.float64
    G2 = NeuronGroup(1, 'v: 1')
    G2.variables.add_array('myarr', dtype=np.int32, size=1)
    G2.variables['myarr'].set_value(unrepresentable_int)
    G2.v = 'floor(myarr)'.format(unrepresentable_int)

    run(0*ms)

    assert G.v[0] != unrepresentable_int, G.v[0]
    assert G2.v[0] == unrepresentable_int, f'{G2.v[0]} != {unrepresentable_int}'


@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
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

    prefs.devices.cuda_standalone.default_functions_integral_convertion = np.float32

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



# Adapted from brian2.test_synapses:test_ufunc_at_vectorisation()
@pytest.mark.parametrize('code', permutation_analysis_good_examples)
@pytest.mark.cuda_standalone
@pytest.mark.standalone_only
@pytest.mark.long
def test_atomics_parallelisation(code):
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
                        assert_allclose(val, endvals[fullvar])
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


if __name__ == '__main__':
    test_default_function_implementations()
    test_default_function_convertion_preference()
    test_default_function_convertion_warnings()
    test_atomics_parallelisation()
