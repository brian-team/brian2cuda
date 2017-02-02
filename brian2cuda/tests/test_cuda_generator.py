from nose import with_setup
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_allclose, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices

import brian2cuda


@attr('standalone-compatible')
@with_setup(teardown=restore_initial_state)
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
    
    
    run(0*ms)
    
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
@with_setup(teardown=restore_initial_state)
def test_default_function_convertion_preference():

    unrepresentable_int = 2**24 + 1  # can't be represented as 32bit float

    prefs.codegen.generators.cuda.default_functions_integral_convertion = 'single_precision'
    G = NeuronGroup(1, 'v: 1')
    G.variables.add_array('myarr', unit=Unit(1), dtype=np.int32, size=1)
    G.variables['myarr'].set_value(unrepresentable_int)
    G.v = 'floor(myarr)'.format(unrepresentable_int)

    prefs.codegen.generators.cuda.default_functions_integral_convertion = 'double_precision'
    G2 = NeuronGroup(1, 'v: 1')
    G2.variables.add_array('myarr', unit=Unit(1), dtype=np.int32, size=1)
    G2.variables['myarr'].set_value(unrepresentable_int)
    G2.v = 'floor(myarr)'.format(unrepresentable_int)

    run(0*ms)

    assert G.v[0] != unrepresentable_int, G.v[0]
    assert G2.v[0] == unrepresentable_int, G.v2[0]


@attr('cuda_standalone', 'standalone_only')
@with_setup(teardown=restore_initial_state)
def test_default_function_convertion_warnings():

    BrianLogger._log_messages.clear()
    with catch_logs() as logs1:
        # warning
        G1 = NeuronGroup(1, 'v: 1')
        G1.variables.add_array('myarr', unit=Unit(1), dtype=np.int64, size=1)
        G1.v = 'sin(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs2:
        # warning
        G2 = NeuronGroup(1, 'v: 1')
        G2.variables.add_array('myarr', unit=Unit(1), dtype=np.uint64, size=1)
        G2.v = 'cos(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs3:
        # no warning
        G3 = NeuronGroup(1, 'v: 1')
        G3.variables.add_array('myarr', unit=Unit(1), dtype=np.int32, size=1)
        G3.v = 'tan(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs4:
        # no warning
        G4 = NeuronGroup(1, 'v: 1')
        G4.variables.add_array('myarr', unit=Unit(1), dtype=np.uint32, size=1)
        G4.v = 'arcsin(i*myarr)'

    prefs.codegen.generators.cuda.default_functions_integral_convertion = 'single_precision'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs5:
        # warning
        G5 = NeuronGroup(1, 'v: 1')
        G5.variables.add_array('myarr', unit=Unit(1), dtype=np.int32, size=1)
        G5.v = 'log(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs6:
        # warning
        G6 = NeuronGroup(1, 'v: 1')
        G6.variables.add_array('myarr', unit=Unit(1), dtype=np.uint32, size=1)
        G6.v = 'log10(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs7:
        # warning
        G7 = NeuronGroup(1, 'v: 1')
        G7.variables.add_array('myarr', unit=Unit(1), dtype=np.int64, size=1)
        G7.v = 'floor(i*myarr)'

    BrianLogger._log_messages.clear()
    with catch_logs() as logs8:
        # warning
        G8 = NeuronGroup(1, 'v: 1')
        G8.variables.add_array('myarr', unit=Unit(1), dtype=np.uint64, size=1)
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


if __name__ == '__main__':
    test_default_function_implementations()
    test_default_function_convertion_preference()
    test_default_function_convertion_warning()
