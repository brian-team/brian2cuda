'''
Module implementing the CUDA "standalone" `CodeObject`. Brian2CUDA implements
two different code objects. `CUDAStandaloneCodeObject` is the standard
implementation, which does not use atomic operations but serialized synaptic
effect application if race conditions are possible.
`CUDAStandaloneAtomicsCodeObject` uses atomic operations which allows parallel
effect applications even when race conditions are possible.
'''
from collections import defaultdict

from brian2.codegen.codeobject import CodeObject, constant_or_scalar
from brian2.codegen.targets import codegen_targets
from brian2.codegen.templates import Templater
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject
from brian2.core.functions import DEFAULT_FUNCTIONS
#from brian2.devices.cpp_standalone.codeobject import constant_or_scalar
from brian2.devices.device import get_device
from brian2.core.preferences import prefs

from brian2cuda.cuda_generator import (CUDAAtomicsCodeGenerator,
                                       CUDACodeGenerator, c_data_type)

__all__ = ['CUDAStandaloneCodeObject',
           'CUDAStandaloneAtomicsCodeObject']


class CUDAStandaloneCodeObject(CPPStandaloneCodeObject):
    '''
    CUDA standalone code object

    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2cuda', '.cu',
                          env_globals={'c_data_type': c_data_type,
                                       'constant_or_scalar': constant_or_scalar})
    generator_class = CUDACodeGenerator

    def __init__(self, *args, **kwargs):
        super(CUDAStandaloneCodeObject, self).__init__(*args, **kwargs)
        self.runs_every_tick = True  #default True, set False in generate_main_source
        self.rng_calls = defaultdict(int)
        # {name: lambda} dictionary for all poisson functions with same lamda for all
        # units of an codeobject (will be generated using the curand host side API)
        self.poisson_lamdas = defaultdict(float)
        # [name] list for all poisson functions where lamda is a variable itself (these
        # will we generated on the fly using the curand device API)
        self.poisson_variable = False
        self.binomial_function = False

    def __call__(self, **kwds):
        return self.run()

    def run(self):
        get_device().main_queue.append(('run_code_object', (self,)))


class CUDAStandaloneAtomicsCodeObject(CUDAStandaloneCodeObject):
    '''
    CUDA standalone code object which uses atomic operations for parallel
    execution

    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    generator_class = CUDAAtomicsCodeGenerator


codegen_targets.add(CUDAStandaloneCodeObject)
codegen_targets.add(CUDAStandaloneAtomicsCodeObject)

rand_code = '''
    #define _rand(vectorisation_idx) (_ptr_array_%CODEOBJ_NAME%_rand[vectorisation_idx])
    '''
rand_impls = DEFAULT_FUNCTIONS['rand'].implementations
rand_impls.add_implementation(CUDAStandaloneCodeObject,
                              code=rand_code,
                              name='_rand')

randn_code = '''
    #define _randn(vectorisation_idx) (_ptr_array_%CODEOBJ_NAME%_randn[vectorisation_idx])
        '''
randn_impls = DEFAULT_FUNCTIONS['randn'].implementations
randn_impls.add_implementation(CUDAStandaloneCodeObject,
                               code=randn_code,
                               name='_randn')
