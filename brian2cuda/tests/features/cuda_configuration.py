import brian2
import os
import shutil
import sys
import socket
import subprocess
import shlex

from brian2.tests.features import (Configuration, DefaultConfiguration,
                                   run_feature_tests, run_single_feature_test)
from brian2.core.preferences import prefs

from brian2.utils.logger import get_logger

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
        for key, values in self.extra_prefs:
            prefs[key] = value
        if self.name is None:
            raise NotImplementedError("You need to set the name attribute.")
        brian2.set_device('cuda_standalone', build_on_run=False,
                          **self.device_kwargs)
        if socket.gethostname() == 'elnath':
            if prefs['devices.cpp_standalone.extra_make_args_unix'] == ['-j12']:
                prefs['devices.cpp_standalone.extra_make_args_unix'] = ['-j24']
            prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_35')
            prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

    def after_run(self):
        if os.path.exists('cuda_standalone'):
            shutil.rmtree('cuda_standalone')
        brian2.device.build(directory='cuda_standalone', compile=True, run=True,
                            with_output=False)

class DynamicConfigCreator(object):
    def __init__(self, config_name, git_commit=None, prefs={}, set_device_kwargs={}):
        # self.name is needed in brian2.tests.featues.base.run_feature_tests()
        # where we pretend this class is a Configuration class
        self.name = config_name
        self.git_commit = git_commit
        self.prefs = prefs
        self.set_device_kwargs = set_device_kwargs
        self.stashed = None
        self.checked_out_feature = None

        if self.git_commit is not None:
            # check if config_name is a branch name
            try:
                commit_sha = self._subprocess('git rev-parse --verify {}'.format(self.git_commit))
            except subprocess.CalledProcessError as err:
                raise ValueError("`git_commit` is not a valid branch or commit sha: {} {}".format(err, err.output))
            self.name += ' ({})'.format(commit_sha[:6])

        clsname = ("DynamicConfigCreator('{config_name}',"
                   "{d}{git_commit}{d},{prefs},{set_device_kwargs})".format(
                       config_name=config_name,
                       git_commit=git_commit,
                       prefs=prefs,
                       set_device_kwargs=set_device_kwargs,
                       d="" if git_commit is None else "'"
                   ))
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

    def _subprocess(self, cmd, **kwargs):
        return subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, **kwargs)

    def git_checkout(self, reverse=False):
        assert self.git_commit is not None
        if reverse:
            # RESTORE PREVIOUS GIT STATE
            assert hasattr(self, 'original_branch')
            try:
                self._subprocess('git checkout {}'.format(self.original_branch))
                logger.diagnostic('Checked out original branch {}.'.format(self.original_branch))
                if self.stashed:
                    self._subprocess('git stash pop')
                    logger.diagnostic("Popped original stash.")
            except subprocess.CalledProcessError as err:
                raise RuntimeError("Coulnd't restore previous git state! Error: "
                                   "{}. Output: {}".format(err, err.output))
        else:
            # CHECK OUT TARGET COMMIT
            self.original_branch = self._subprocess('git rev-parse --abbrev-ref HEAD').rstrip()
            logger.diagnostic("Original branch is {}.".format(self.original_branch))
            # stash the current changes if we are checking out another commit
            old_stash = self._subprocess('git rev-parse -q --verify refs/stash')
            self._subprocess('git stash')
            new_stash = self._subprocess('git rev-parse -q --verify refs/stash')
            # store shash commit sha if there was something to stash
            self.stashed = None
            if old_stash != new_stash:
                self.stashed = new_stash
                logger.diagnostic("Stashed current state in original branch.")
            try:
                self._subprocess('git checkout {}'.format(self.git_commit))
                logger.diagnostic("Checkout out target {}.".format(self.git_commit))
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
            self._subprocess('git checkout {} -- :/{}'.format(sha, module_file))
            logger.diagnostic("Checked out feature {} from {}".format(module_file, sha))
        except subprocess.CalledProcessError as err:
            raise ValueError("Couldn't check out module file: {} {}".format(err, err.output))

    def git_reset(self):
        try:
            self._subprocess('git reset HEAD :/')
            self._subprocess('git checkout :/')
            logger.diagnostic("Reset changes in current working directory")
        except subprocess.CalledProcessError as err:
            raise ValueError("Couldn't reset changes: {} {}".format(err, err.output))


class CUDAStandaloneConfiguration(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone'

class CUDAStandaloneConfigurationExtraThresholdKernel(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone with extra threshold reset kernel'
    extra_prefs = {'devices.cuda_standalone.extra_threshold_kernel': True}

class CUDAStandaloneConfigurationNoAssert(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (asserts disabled)'
    device_kwargs = {'disable_asserts': True}

class CUDAStandaloneConfigurationCurandDouble(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone (curand_float_type = double)'
    extra_prefs = {'devices.cuda_standalone.curand_float_type': 'double'}

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
