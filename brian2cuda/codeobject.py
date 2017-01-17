'''
Module implementing the CUDA "standalone" `CodeObject`
'''
from brian2.codegen.codeobject import CodeObject, constant_or_scalar
from brian2.codegen.targets import codegen_targets
from brian2.codegen.templates import Templater
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject
from brian2.core.functions import DEFAULT_FUNCTIONS
#from brian2.devices.cpp_standalone.codeobject import constant_or_scalar
from brian2.devices.device import get_device
from brian2.core.preferences import prefs

from brian2cuda.cuda_generator import (CUDACodeGenerator,
                                                     c_data_type)

__all__ = ['CUDAStandaloneCodeObject']


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
    no_or_const_delay_mode = False
    serializing_form = "syn"
    runs_every_tick = True  #default True, set False in generate_main_source
    rand_calls = 0
    randn_calls = 0

    def __call__(self, **kwds):
        return self.run()

    def run(self):
        get_device().main_queue.append(('run_code_object', (self,)))

codegen_targets.add(CUDAStandaloneCodeObject)

rand_code = '''
    #define _rand(vectorisation_idx) (_array_%CODEOBJ_NAME%_rand[vectorisation_idx])
    '''
rand_impls = DEFAULT_FUNCTIONS['rand'].implementations
rand_impls.add_implementation(CUDAStandaloneCodeObject,
                              code=rand_code,
                              name='_rand')

randn_code = '''
    #define _randn(vectorisation_idx) (_array_%CODEOBJ_NAME%_randn[vectorisation_idx])
        '''
randn_impls = DEFAULT_FUNCTIONS['randn'].implementations
randn_impls.add_implementation(CUDAStandaloneCodeObject,
                              code=randn_code,
                              name='_randn')
