import os
import shutil
import subprocess
import shlex
import time

import brian2
from brian2.tests.features import (Configuration, DefaultConfiguration,
                                   run_feature_tests, run_single_feature_test)
from brian2.utils.logger import get_logger
from brian2.devices import device, set_device, get_device
from brian2 import prefs
from brian2.tests.features.base import SpeedTest

try:
    import brian2genn
except ImportError:
    pass

logger = get_logger('brian2.devices.cuda_standalone.cuda_configuration')

__all__ = ['CUDAStandaloneConfiguration']


# Get information about the environment this script is run in
_slurm_cluster = os.environ.get("SLURM_CLUSTER_NAME", default=None)
_num_available_threads = len(os.sched_getaffinity(0))


SETUP_TIMER = '''
std::chrono::high_resolution_clock::time_point _benchmark_start, _benchmark_now;
_benchmark_start = std::chrono::high_resolution_clock::now();
std::ofstream _benchmark_file;
_benchmark_file.open("{fname}");
'''

TIME_DIFF = '''
_benchmark_now = std::chrono::high_resolution_clock::now();
_benchmark_file << "{name}" << " "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       _benchmark_now - _benchmark_start
                   ).count()
                << std::endl;
'''

CLOSE_TIMER = '''
_benchmark_file.close();
'''

def insert_benchmark_point(name):
    device.insert_code("main", TIME_DIFF.format(name=name))


class BenchmarkConfiguration(Configuration):
    '''Base class for all configurations used for benchmarking'''
    name = None
    devicename = None
    single_precision = False
    profile = False
    extra_prefs = {}
    device_kwargs = {}
    commit = None

    def __init__(self, feature_test):
        self.feature_test = feature_test
        self.feature_test_name = self.feature_test.__class__.__name__
        self.n = self.feature_test.n
        self.project_dir = None
        self.benchmark_file = None
        if self.name is None:
            raise NotImplementedError("You need to set the name attribute.")
        if self.devicename is None:
            raise NotImplementedError("You need to set the devicename attribute.")
        super().__init__()

    @classmethod
    def get_project_dir(cls, feature_name, n):
        directory = os.path.join(f"{cls.devicename}_runs",
                                 f"{feature_name}_{n}_{cls.name}")
        for symbol in [" ", ",", "(", ")"]:
            directory = directory.replace(symbol, "-")
        return directory

    def set_profiling(self):
        if self.profile:
            self.device_kwargs["profile"] = self.profile

    def before_run(self):
        self.project_dir = self.get_project_dir(self.feature_test_name, self.n)
        # Remove project directory if it already exists
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)

        prefs.reset_to_defaults()

        # Enable/disable profiling, can be overwritten (e.g. in GeNNConfiguration)
        self.set_profiling()

        set_device(self.devicename,
                   directory=self.project_dir,
                   build_on_run=True,  # GeNN doesn't allow building after run
                   with_output=True,  # Generates the results/stdout.txt files
                   **self.device_kwargs)

        # Need chrono header for timing functions
        prefs.codegen.cpp.headers += ["<chrono>"]
        if self.single_precision:
            prefs["core.default_float_dtype"] = brian2.float32

        # Set preferences
        for key, value in self.extra_prefs.items():
            prefs[key] = value

        prefs._backup()

        # Insert benchmarking code
        self.benchmark_file = os.path.join(self.project_dir, "results", "benchmark.time")
        device.insert_code("before_start", SETUP_TIMER.format(fname=self.benchmark_file))
        device.insert_code("after_start", TIME_DIFF.format(name="after_start"))
        device.insert_code("before_run", TIME_DIFF.format(name="before_run"))
        device.insert_code("after_run", TIME_DIFF.format(name="after_run"))
        device.insert_code("before_end", TIME_DIFF.format(name="before_end"))
        device.insert_code("after_end", TIME_DIFF.format(name="after_end"))
        device.insert_code("after_end", CLOSE_TIMER)

    def after_run(self):
        # Add Python-side compile + run time measurment (taken in
        # `TimedSpeedTest.timed_run()`) to benchmarking file
        with open(self.benchmark_file, "a") as file:
            file.write(f"total {self.feature_test.runtime}")


class CUDAStandaloneConfigurationBase(BenchmarkConfiguration):
    '''Base class for CUDAStandaloneConfigurations'''
    devicename = "cuda_standalone"

    def before_run(self):
        # Set the path for self-compiled deviceQuery on the TU HPC cluster
        if _slurm_cluster == "hpc":
            pref_key = "devices.cuda_standalone.cuda_backend.device_query_path"
            if pref_key not in self.extra_prefs:
                self.extra_prefs[pref_key] = (
                    "~/cuda-samples/bin/x86_64/linux/release/deviceQuery"
                )
        super().before_run()


class DynamicConfigCreator(object):
    def __init__(self, config_name, git_commit=None, profile=None,
                 single_precision=None, prefs={}, set_device_kwargs={}):
        # self.name is needed in brian2.tests.featues.base.run_feature_tests()
        # where we pretend this class is a Configuration class
        self.name = config_name
        self.git_commit = git_commit
        self.set_device_kwargs = set_device_kwargs
        self.stashed = None
        self.checked_out_feature = None
        self.profile = profile
        self.single_precision = single_precision

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

    @classmethod
    def get_project_dir(cls, feature_name, n):
        """
        The run_benchmarck_suite.py determines the project directory from the config.
        For DynamicConfigCreator configurations, we pass the call to the actual
        configuration class.
        """
        return CUDAStandaloneConfigurationBase.get_project_dir(feature_name, n)

    def __call__(self, feature_test):
        # we can't acces the self from DynamicConfigCreator inside the nested
        # DynamicCUDAStandaloneConfiguration class -> make a copy
        DynamicConfigCreator_self = self
        class DynamicCUDAStandaloneConfiguration(CUDAStandaloneConfigurationBase):
            name = DynamicConfigCreator_self.name
            devicename = CUDAStandaloneConfigurationBase.devicename
            single_precision = DynamicConfigCreator_self.single_precision
            profile = DynamicConfigCreator_self.profile
            extra_prefs = DynamicConfigCreator_self.prefs
            device_kwargs = DynamicConfigCreator_self.set_device_kwargs
            commit = DynamicConfigCreator_self.git_commit

        return DynamicCUDAStandaloneConfiguration(feature_test)

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
    profile = 'blocking'

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
    profile = True

class CUDAStandaloneConfigurationProfileCPU(CUDAStandaloneConfigurationBase):
    name = "CUDA standalone (profile='blocking')"
    profile = 'blocking'

class CUDAStandaloneConfigurationBundles(CUDAStandaloneConfigurationBase):
    name = 'CUDA standalone bundles'
    commit = 'nemo_bundles'

class CUDAStandaloneConfigurationBundlesProfileCPU(CUDAStandaloneConfigurationBase):
    name = "CUDA standalone bundles (profile='blocking')"
    commit = 'nemo_bundles'
    profile = 'blocking'


######################################
###  CPP STANDALONE CONFIGURATIONS ###
######################################
class CPPStandaloneConfigurationBase(BenchmarkConfiguration):
    name = 'C++ standalone'
    devicename = 'cpp_standalone'

class CPPStandaloneConfiguration(CPPStandaloneConfigurationBase):
    single_precision = False
    profile = False

class CPPStandaloneConfigurationProfile(CPPStandaloneConfigurationBase):
    single_precision = False
    profile = True

class CPPStandaloneConfigurationSinglePrecision(CPPStandaloneConfigurationBase):
    name = 'C++ standalone (single precision)'
    single_precision = True
    profile = False

class CPPStandaloneConfigurationSinglePrecisionProfile(CPPStandaloneConfigurationBase):
    name = 'C++ standalone (single precision)'
    single_precision = True
    profile = True

##################################
###  CPP OPENMP CONFIGURATIONS ###
##################################
class CPPStandaloneConfigurationOpenMPBase(CPPStandaloneConfigurationBase):
    name = f'C++ standalone (OpenMP, {_num_available_threads} threads)'

    def before_run(self):
        # Use as many threads as are available to this process
        pref_key = "devices.cpp_standalone.openmp_threads"
        if pref_key not in self.extra_prefs:
            self.extra_prefs[pref_key] = _num_available_threads
        super().before_run()


class CPPStandaloneConfigurationOpenMPMaxThreads(CPPStandaloneConfigurationOpenMPBase):
    single_precision = False
    profile = False

class CPPStandaloneConfigurationOpenMPMaxThreadsProfile(CPPStandaloneConfigurationOpenMPBase):
    single_precision = False
    profile = True

class CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision(CPPStandaloneConfigurationOpenMPBase):
    name = (
        f'C++ standalone (OpenMP, single_precision, {_num_available_threads} threads)'
    )
    single_precision = True
    profile = False

class CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecisionProfile(CPPStandaloneConfigurationOpenMPBase):
    name = (
        f'C++ standalone (OpenMP, single_precision, {_num_available_threads} threads)'
    )
    single_precision = True
    profile = True


#############################
###  GENN CONFIGURATIONS  ###
#############################

# Adapted from brian2genn.correctness_testing
class GeNNConfigurationBase(BenchmarkConfiguration):
    devicename = 'genn'

    # Overwrite how GeNN sets profiling
    def set_profiling(self):
        if self.profile:
            self.extra_prefs['devices.genn.kernel_timing'] = True

class GeNNConfiguration(GeNNConfigurationBase):
    name = 'GeNN'
    single_precision = False
    profile = False

class GeNNConfigurationProfile(GeNNConfigurationBase):
    name = 'GeNN'
    single_precision = False
    profile = True

class GeNNConfigurationSinglePrecision(GeNNConfigurationBase):
    name = 'GeNN (single precision)'
    single_precision = True
    profile = False

class GeNNConfigurationSinglePrecisionProfile(GeNNConfigurationBase):
    name = 'GeNN (single precision)'
    single_precision = True
    profile = True

class GeNNConfigurationSpanTypePre(GeNNConfigurationBase):
    name = 'GeNN (span type PRE)'
    single_precision = False
    profile = False
    extra_prefs = {'devices.genn.synapse_span_type': 'PRESYNAPTIC'}

class GeNNConfigurationSpanTypePreProfile(GeNNConfigurationBase):
    name = 'GeNN (span type PRE)'
    single_precision = False
    profile = True
    extra_prefs = {'devices.genn.synapse_span_type': 'PRESYNAPTIC'}

class GeNNConfigurationSinglePrecisionSpanTypePre(GeNNConfigurationBase):
    name = 'GeNN (single precision, span type PRE)'
    single_precision = True
    profile = False
    extra_prefs = {'devices.genn.synapse_span_type': 'PRESYNAPTIC'}

class GeNNConfigurationSinglePrecisionSpanTypePreProfile(GeNNConfigurationBase):
    name = 'GeNN (single precision, span type PRE)'
    single_precision = True
    profile = True
    extra_prefs = {'devices.genn.synapse_span_type': 'PRESYNAPTIC'}
