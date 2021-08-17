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
        # Whether this code object runs on a clock or only once. Default is True, set
        # False in `CUDAStandaloneDevice.generate_main_source()`
        self.runs_every_tick = True
        # Dictionary collectin the number of RNG function calls in this code object.
        # Keys are: "rand", "randn", "poisson-<idx>" with one <idx> for each `poisson`
        # with different (scalar) lambda
        self.rng_calls = defaultdict(int)
        # {name: lambda} dictionary for all poisson functions with scalar lambda
        # (these random numbers will be generated using the curand host side API)
        self.poisson_lamdas = defaultdict(float)
        # Whether this codeobject uses curand device API (for binomial functions or
        # poisson with vectorized lambda) and needs curand states
        self.needs_curand_states = False

    def __call__(self, **kwds):
        return self.run()

    def compile_block(self, block):
        pass  # Compilation will be handled in device

    def run_block(self, block):
        if block == 'run':
            get_device().main_queue.append((block + '_code_object', (self,)))
        else:
            # Check the C++ code whether there is anything to run
            cu_code = getattr(self.code, block + '_cu_file')
            if len(cu_code) and 'EMPTY_CODE_BLOCK' not in cu_code:
                get_device().main_queue.append((block + '_code_object', (self,)))
                self.before_after_blocks.append(block)


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
