import os
import shutil
import sys
import socket
import subprocess
import shlex
import multiprocessing

import brian2
from brian2.tests.features import (Configuration, DefaultConfiguration,
                                   run_feature_tests, run_single_feature_test)
from brian2.utils.logger import get_logger

try:
    import brian2genn
except ImportError:
    pass

logger = get_logger('brian2.devices.cuda_standalone.cuda_configuration')

__all__ = ['CUDAStandaloneConfiguration']

class CUDAStandaloneConfigurationBase(Configuration):
    '''Base class for CUDAStandaloneConfigurations'''
    name = None
    commit = None
    device_kwargs = {}
    extra_prefs = {}

    def before_run(self):
        # set brian preferences
        for key, value in self.extra_prefs.items():
            brian2.prefs[key] = value
        if self.name is None:
            raise NotImplementedError("You need to set the name attribute.")
        brian2.set_device('cuda_standalone', build_on_run=False,
                          **self.device_kwargs)

    def after_run(self):
        if os.path.exists('cuda_standalone'):
            shutil.rmtree('cuda_standalone')
        brian2.device.build(directory='cuda_standalone', compile=True, run=True,
                            with_output=True)

class DynamicConfigCreator(object):
    def __init__(self, config_name, git_commit=None, prefs={}, set_device_kwargs={}):
        # self.name is needed in brian2.tests.featues.base.run_feature_tests()
        # where we pretend this class is a Configuration class
        self.name = config_name
        self.git_commit = git_commit
        self.set_device_kwargs = set_device_kwargs
        self.stashed = None
        self.checked_out_feature = None

        # make sure float_dtypes that were converted to strings (see below)
        # are converted back into brian2.float... types
        dtype_str = 'core.default_float_dtype'
        if dtype_str in prefs and isinstance(prefs[dtype_str], str):
            prefs[dtype_str] = getattr(brian2, prefs[dtype_str])
        self.prefs = prefs

        if self.git_commit is not None:
            # check if config_name is a branch name
            try:
                commit_sha = self._subprocess(f'git rev-parse --verify {self.git_commit}')
            except subprocess.CalledProcessError as err:
                raise ValueError(f"`git_commit` is not a valid branch or commit sha: {err} {err.output}")
            self.name += f' ({commit_sha[:6]})'

        # make sure printing np.float32 prints actually 'float32' and not '<type ...>'
        print_prefs = prefs.copy()
        if 'core.default_float_dtype' in prefs:
            print_prefs['core.default_float_dtype'] = prefs['core.default_float_dtype'].__name__
        d = ""
        if git_commit is not None:
            d = "'"
        clsname = (f"DynamicConfigCreator('{config_name}',"
                   f"{d}{git_commit}{d},{print_prefs},{set_device_kwargs})")
        # little hack: this way in brian2.tests.features.base.result() the
        # DynamicCUDAStandaloneConfiguration class will be correctly recreated
        self.__name__ = clsname

    def __call__(self):
        # we can't acces the self from DynamicConfigCreator inside the nested
        # DynamicCUDAStandaloneConfiguration class -> make a copy
        DynamicConfigCreator_self = self
        class DynamicCUDAStandaloneConfiguration(CUDAStandaloneConfigurationBase):
            name = DynamicConfigCreator_self.name
            device_kwargs = DynamicConfigCreator_self.set_device_kwargs
            commit = DynamicConfigCreator_self.git_commit
            extra_prefs = DynamicConfigCreator_self.prefs

        return DynamicCUDAStandaloneConfiguration()

    def _subprocess(self, cmd, fails_ok=False, **kwargs):
        try:
            output = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, **kwargs)
        except subprocess.CalledProcessError as error:
            logger.diagnostic(f"Command '{cmd}' failed: {error}, {error.output}")
            output = None
            if not fails_ok:
                raise

        return output

    def git_checkout(self, reverse=False):
        assert self.git_commit is not None
        if reverse:
            # RESTORE PREVIOUS GIT STATE
            assert hasattr(self, 'original_branch')
            try:
                self._subprocess(f'git checkout {self.original_branch}')
                logger.diagnostic(f'Checked out original branch {self.original_branch}')
                if self.stashed:
                    self._subprocess('git stash pop')
                    logger.diagnostic("Popped original stash.")
            except subprocess.CalledProcessError as err:
                raise RuntimeError(f"Coulnd't restore previous git state! Error: "
                                   f"{err}. Output: {err.output}")
        else:
            # CHECK OUT TARGET COMMIT
            self.original_branch = self._subprocess('git rev-parse --abbrev-ref HEAD').rstrip()
            logger.diagnostic(f"Original branch is {self.original_branch}.")
            # stash the current changes if we are checking out another commit
            old_stash = self._subprocess('git rev-parse -q --verify refs/stash', fails_ok=True)
            self._subprocess('git stash')
            new_stash = self._subprocess('git rev-parse -q --verify refs/stash', fails_ok=True)
            # store shash commit sha if there was something to stash
            self.stashed = None
            if old_stash != new_stash:
                self.stashed = new_stash
                logger.diagnostic("Stashed current state in original branch.")
            try:
                self._subprocess(f'git checkout {self.git_commit}')
                logger.diagnostic(f"Checked out target {self.git_commit}.")
            except subprocess.CalledProcessError as err:
                raise RuntimeError("Couldn't check out target commit, "
                                   "trying to restore original ({}). "
                                                    "Output: {}.".format(self.git_commit,
                                                                         err.output))
                if self.stashed:
                    # pop the previous stash
                    self._subprocess('git stash pop')
                    logger.diagnostic("Popped original stash before raising error")
                raise

    def git_checkout_feature(self, module):
        module_file = module.replace('.', '/') + '.py'
        if self.stashed:
            sha = self.stashed
        else:
            sha = self.original_branch

        try:
            self._subprocess(f'git checkout {sha} -- :/{module_file}')
            logger.diagnostic(f"Checked out feature {module_file} from {sha}")
        except subprocess.CalledProcessError as err:
            raise ValueError(f"Couldn't check out module file: {err} {err.output}")

    def git_reset(self):
        try:
            self._subprocess('git reset HEAD :/')
            self._subprocess('git checkout :/')
            logger.diagnostic("Reset changes in current working directory")
        except subprocess.CalledProcessError as err:
            raise ValueError(f"Couldn't reset changes: {err} {err.output}")


class CUDAStandaloneConfiguration(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone'

class CUDAStandaloneConfigurationExtraThresholdKernel(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone with extra threshold reset kernel'
    extra_prefs = {'devices.cuda_standalone.extra_threshold_kernel': True}

class CUDAStandaloneConfigurationNoAssert(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (asserts disabled)'
    device_kwargs = {'disable_asserts': True}

class CUDAStandaloneConfigurationNoCudaOccupancyAPI(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (not using cuda occupancy API)'
    extra_prefs = {'devices.cuda_standalone.calc_occupancy': False}

class CUDAStandaloneConfigurationNoCudaOccupancyAPIProfileCPU(CUDAStandaloneConfigurationBase):
    name = "CUDA standalone (not cuda occupancy API and profile='blocking')"
    extra_prefs = {'devices.cuda_standalone.calc_occupancy': False}
    device_kwargs = {'profile': 'blocking'}

class CUDAStandaloneConfiguration2BlocksPerSM(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone with 2 blocks per SM'
    extra_prefs = {'devices.cuda_standalone.SM_multiplier': 2}

class CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone with 2 blocks per SM and __launch_bounds__'
    extra_prefs = {'devices.cuda_standalone.SM_multiplier': 2,
                   'devices.cuda_standalone.launch_bounds': True}

class CUDAStandaloneConfigurationSynLaunchBounds(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (SYN __launch_bounds__)'
    extra_prefs = {'devices.cuda_standalone.syn_launch_bounds': True}

class CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (2 bl/SM, SYN __launch_bounds__)'
    extra_prefs = {'devices.cuda_standalone.SM_multiplier':  2,
                   'devices.cuda_standalone.syn_launch_bounds': True}

class CUDAStandaloneConfigurationProfileGPU(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (profile=True)'
    device_kwargs = {'profile': True}

class CUDAStandaloneConfigurationProfileCPU(CUDAStandaloneConfigurationBase):
    name = "CUDA standalone (profile='blocking')"
    device_kwargs = {'profile': 'blocking'}

class CUDAStandaloneConfigurationBundles(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone bundles'
    commit = 'nemo_bundles'

class CUDAStandaloneConfigurationBundlesProfileCPU(CUDAStandaloneConfigurationBase):
    name = "CUDA standalone bundles (profile='blocking')"
    commit = 'nemo_bundles'
    device_kwargs = {'profile': 'blocking'}

class CPPStandaloneConfiguration(Configuration):
    name = 'C++ standalone'
    profile = False
    single_precision = False
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        if self.single_precision:
            brian2.prefs['core.default_float_dtype'] = brian2.float32
            brian2.prefs._backup()
        brian2.set_device('cpp_standalone', build_on_run=False, profile=self.profile)
        
    def after_run(self):
        if os.path.exists('cpp_standalone'):
            shutil.rmtree('cpp_standalone')
        brian2.device.build(directory='cpp_standalone', compile=True, run=True,
                            with_output=True)

class CPPStandaloneConfigurationProfile(CPPStandaloneConfiguration):
    single_precision = False
    profile = True

class CPPStandaloneConfigurationSinglePrecision(CPPStandaloneConfiguration):
    name = 'C++ standalone (single precision)'
    single_precision = True
    profile = False

class CPPStandaloneConfigurationSinglePrecisionProfile(CPPStandaloneConfiguration):
    name = 'C++ standalone (single precision)'
    single_precision = True
    profile = True

class CPPStandaloneConfigurationOpenMPMaxThreads(CPPStandaloneConfiguration):
    single_precision = False
    profile = False
    openmp_threads = None
    hostname = socket.gethostname()
    known = True
    if hostname.startswith("cognition"):
        openmp_threads = 24
    else:
        known = False
        openmp_threads = multiprocessing.cpu_count()
    name = f'C++ standalone (OpenMP, {openmp_threads} threads)'

    def before_run(self):
        brian2.prefs.reset_to_defaults()
        if self.single_precision:
            brian2.prefs['core.default_float_dtype'] = brian2.float32
        brian2.set_device('cpp_standalone', build_on_run=False, profile=self.profile)
        brian2.prefs['devices.cpp_standalone.openmp_threads'] = self.openmp_threads
        brian2.prefs._backup()
        if self.known:
            logger.info("Running CPPStandaloneConfigurationOpenMP with {} "
                        "threads".format(self.openmp_threads))
        else:
            logger.warn("Unknown hostname. Using number of logical cores ({}) "
                        "as threads for CPPStandaloneConfigurationOpenMP"
                        "".format(self.openmp_threads))

class CPPStandaloneConfigurationOpenMPMaxThreadsProfile(CPPStandaloneConfigurationOpenMPMaxThreads):
    single_precision = False
    profile = True

class CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision(CPPStandaloneConfigurationOpenMPMaxThreads):
    # TODO: this is pretty hacky... instead just to the hostname stuff outside class definitions and use the global var...
    name = f'C++ standalone (OpenMP, single_precision, {CPPStandaloneConfigurationOpenMPMaxThreads.openmp_threads} threads)'
    single_precision = True
    profile = False

class CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecisionProfile(CPPStandaloneConfigurationOpenMPMaxThreads):
    name = f'C++ standalone (OpenMP, single_precision, {CPPStandaloneConfigurationOpenMPMaxThreads.openmp_threads} threads)'
    single_precision = True
    profile = True

# Copied from brian2genn.correctness_testing
class GeNNConfigurationOptimized(Configuration):
    name = 'GeNN_optimized'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedProfile(Configuration):
    name = 'GeNN_optimized'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['devices.genn.kernel_timing'] = True
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedSinglePrecision(Configuration):
    name = 'GeNN_optimized (single precision)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['core.default_float_dtype'] = brian2.float32
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedSinglePrecisionProfile(Configuration):
    name = 'GeNN_optimized (single precision)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['core.default_float_dtype'] = brian2.float32
        brian2.prefs['devices.genn.kernel_timing'] = True
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedSpanTypePre(Configuration):
    name = 'GeNN_optimized (span type PRE)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['devices.genn.synapse_span_type'] = 'PRESYNAPTIC'
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedSpanTypePreProfile(Configuration):
    name = 'GeNN_optimized (span type PRE)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['devices.genn.synapse_span_type'] = 'PRESYNAPTIC'
        brian2.prefs['devices.genn.kernel_timing'] = True
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedSinglePrecisionSpanTypePre(Configuration):
    name = 'GeNN_optimized (single precision, span type PRE)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['core.default_float_dtype'] = brian2.float32
        brian2.prefs['devices.genn.synapse_span_type'] = 'PRESYNAPTIC'
        brian2.prefs._backup()
        brian2.set_device('genn')

class GeNNConfigurationOptimizedSinglePrecisionSpanTypePreProfile(Configuration):
    name = 'GeNN_optimized (single precision, span type PRE)'
    def before_run(self):
        brian2.prefs.reset_to_defaults()
        brian2.prefs['core.default_float_dtype'] = brian2.float32
        brian2.prefs['devices.genn.synapse_span_type'] = 'PRESYNAPTIC'
        brian2.prefs['devices.genn.kernel_timing'] = True
        brian2.prefs._backup()
        brian2.set_device('genn')
