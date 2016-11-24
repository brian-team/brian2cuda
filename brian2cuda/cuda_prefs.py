'''
Preferences that relate to the brian2cuda interface.
'''

from brian2.core.preferences import *

# TODO: write tests for this!
def validate_generator_and_set_order(generator):
    '''
    Validate generator and set correct default order parameter of cuRAND random
    number generator (depending of generator type being pseudo- or quasirandom).
    '''

    if generator in ['CURAND_RNG_PSEUDO_DEFAULT',
                     'CURAND_RNG_PSEUDO_XORWOW',
                     'CURAND_RNG_PSEUDO_MRG32K3A',
                     'CURAND_RNG_PSEUDO_MTGP32',
                     'CURAND_RNG_PSEUDO_PHILOX4_32_10',
                     'CURAND_RNG_PSEUDO_MT19937',
                     'CURAND_RNG_QUASI_DEFAULT',
                     'CURAND_RNG_QUASI_SOBOL32',
                     'CURAND_RNG_QUASI_SCRAMBLED_SOBOL32',
                     'CURAND_RNG_QUASI_SOBOL64',
                     'CURAND_RNG_QUASI_SCRAMBLED_SOBOL64']:
        validation = True
        if 'PSEUDO' in generator.split('_'):
            order = 'CURAND_ORDERING_PSEUDO_DEFAULT'
        elif 'QUASI' in generator.split('_'):
            order = 'CURAND_ORDERING_QUASI_DEFAULT'
        prefs.devices.cuda_standalone.random_number_generator_order = order
    else:
        validation = False
    return validation

prefs.register_preferences(
    'devices.cuda_standalone',
    'Preferences that relate to the brian2cuda interface',

    SM_multiplier = BrianPreference(
        default=1,
        docs='''
        The number of blocks per SM. By default, this value is set to 1.
        ''',
        ),

    extra_compile_args_nvcc=BrianPreference(
        docs='''Extra compile arguments (a list of strings) to pass to the nvcc compiler.''',
        default=['-w', '-use_fast_math', '-arch=sm_20']  # TODO: we shouldn't set arch to sm_20 by default?
    ),

    random_number_generator_type=BrianPreference(
        docs='''Generator type (str) that cuRAND uses for random number generation.
            Setting the generator type automatically resets the generator order
            (prefs.devices.cuda_standalone.random_number_generator_order) to its default value.
            See cuRAND documentation for more details on generator types and orders.''',
        validator=validate_generator_and_set_order,
        default='CURAND_RNG_PSEUDO_DEFAULT'),

    random_number_generator_order=BrianPreference(
        docs='''The order parameter (str) used to choose how the results of cuRAND
            random number generation are ordered in global memory.
            See cuRAND documentation for more details on generator types and orders.''',
        validator=lambda v: v in ['CURAND_ORDERING_PSEUDO_DEFAULT',
                                  'CURAND_ORDERING_PSEUDO_BEST',
                                  'CURAND_ORDERING_PSEUDO_SEEDED',
                                  'CURAND_ORDERING_QUASI_DEFAULT'],
        default='CURAND_ORDERING_PSEUDO_DEFAULT')
)
                                                                                                      
