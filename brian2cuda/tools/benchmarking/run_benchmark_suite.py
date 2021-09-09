import argparse

parser = argparse.ArgumentParser(description='Run brian2cuda benchmarks')

parser.add_argument('-d', '--results-dir', type=str, default=None,
                    help="Directory where results will be stored")

parser.add_argument('--dry-run', action='store_true',
                    help=("Exit script after argument parsing. Used to check "
                          "validity of arguments"))

parser.add_argument('--no-nvprof', action='store_true',
                    help=("Don't run nvprof profiling"))

args = parser.parse_args()

if args.dry_run:
    import sys
    print(f"Dry run completed, {__file__} arguments valid.")
    sys.exit()

if args.results_dir is None:
    raise RuntimeError(
        "Don't run `run_benchmark_suite.py` directly. Use the shell script "
        "`run_benchmark_suite.sh` instead."
    )
else:
    # Remove traling "/" if present
    directory = args.results_dir.rstrip("/")

import os
import shutil
import subprocess
import sys

# run tests without X-server
import matplotlib
matplotlib.use('Agg')

# pretty plots
#import seaborn as sns

import time
import datetime

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.base import results
from brian2.utils.logger import BrianLogger

from brian2cuda.tests.features.speed import *

import brian2cuda
from brian2cuda.utils.logger import suppress_brian2_logs
from brian2cuda.tests.features.cuda_configuration import DynamicConfigCreator
from brian2cuda.tests.features.cuda_configuration import (CUDAStandaloneConfiguration,
                                                          CUDAStandaloneConfigurationNoAssert,
                                                          CUDAStandaloneConfigurationExtraThresholdKernel,
                                                          CUDAStandaloneConfigurationNoCudaOccupancyAPI,
                                                          CUDAStandaloneConfigurationNoCudaOccupancyAPIProfileCPU,
                                                          CUDAStandaloneConfiguration2BlocksPerSM,
                                                          CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds,
                                                          CUDAStandaloneConfigurationSynLaunchBounds,
                                                          CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds,
                                                          CUDAStandaloneConfigurationProfileGPU,
                                                          CUDAStandaloneConfigurationProfileCPU,
                                                          CPPStandaloneConfiguration,
                                                          CPPStandaloneConfigurationSinglePrecision,
                                                          CPPStandaloneConfigurationOpenMPMaxThreads,
                                                          CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision,
                                                          GeNNConfiguration,
                                                          GeNNConfigurationSpanTypePre,
                                                          GeNNConfigurationSinglePrecision,
                                                          GeNNConfigurationSinglePrecisionSpanTypePre)


#from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationCPU, GeNNConfigurationOptimized

from create_readme import create_readme
from helpers import pickle_results, translate_pkl_to_csv

#suppress_brian2_logs()
# Uncomment this to get brian2cuda logs
# (Set log_level_diagnostic() for DynamicConfigCreator diagnostic messages)
BrianLogger.log_level_debug()

configurations = [
    # configuration
    #NumpyConfiguration,
    #WeaveConfiguration,
    #LocalConfiguration,
    CPPStandaloneConfiguration,
    #CPPStandaloneConfigurationSinglePrecision,
    CPPStandaloneConfigurationOpenMPMaxThreads,
    #CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision,

    # max blocks
    DynamicConfigCreator('CUDA standalone (max blocks, atomics)'),

    #DynamicConfigCreator('CUDA standalone (single precision, max blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32}),


    # 1 block
    DynamicConfigCreator('CUDA standalone (1 block, atomics)',
                         prefs={'devices.cuda_standalone.parallel_blocks': 1}),

    #DynamicConfigCreator('CUDA standalone (single precision, 1 block, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 1}),


    ## 20 blocks
    #DynamicConfigCreator('CUDA standalone (20 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.parallel_blocks': 20}),

    #DynamicConfigCreator('CUDA standalone (single precision, 20 blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 20}),


    ## 40 blocks
    #DynamicConfigCreator('CUDA standalone (40 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.parallel_blocks': 40}),

    #DynamicConfigCreator('CUDA standalone (single precision, 40 blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 40}),


    ## 60 blocks
    #DynamicConfigCreator('CUDA standalone (60 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.parallel_blocks': 60}),

    #DynamicConfigCreator('CUDA standalone (single precision, 60 blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 60}),


    #DynamicConfigCreator('CUDA standalone (max blocks, no atomics)',
    #                     prefs={'devices.cuda_standalone.use_atomics': False}),

    #DynamicConfigCreator('CUDA standalone (1 block, no atomics)',
    #                     prefs={'devices.cuda_standalone.use_atomics': False,
    #                            'devices.cuda_standalone.parallel_blocks': 1}),

    #DynamicConfigCreator('CUDA standalone (no bundles, 15 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.push_synapse_bundles': False}),


    #DynamicConfigCreator('CUDA standalone (master)',
    #                     git_commit='master'),

    #DynamicConfigCreator("CUDA standalone (only compilation)",
    #                     set_device_kwargs={'compile': True, 'run': False}),

    #CUDAStandaloneConfigurationExtraThresholdKernel,
    #CUDAStandaloneConfigurationNoAssert,
    #CUDAStandaloneConfigurationNoCudaOccupancyAPI,
    #CUDAStandaloneConfigurationNoCudaOccupancyAPIProfileCPU,
    #CUDAStandaloneConfiguration2BlocksPerSM,
    #CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds,
    #CUDAStandaloneConfigurationSynLaunchBounds,
    #CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds,
    #CUDAStandaloneConfigurationProfileGPU,
    #CUDAStandaloneConfigurationProfileCPU,
    #GeNNConfigurationCPU,
    GeNNConfiguration,
    #GeNNConfigurationSinglePrecision,
    GeNNConfigurationSpanTypePre,
    #GeNNConfigurationSinglePrecisionSpanTypePre,
]

speed_tests = [# feature_test                           n_slice

               # paper benchmarks
               (BrunelHakimHomogDelays,                 slice(None)),
               (BrunelHakimHeterogDelays,               slice(None)),
               (BrunelHakimHeterogDelaysNarrowDistr,    slice(None)),
               (STDPCUDAHeterogeneousDelays,            slice(None)),
               (STDPCUDAHomogeneousDelays,              slice(None)),
               (STDPCUDA,                               slice(None)),
               (COBAHHPseudocoupled1000,                slice(None)),
               (COBAHHPseudocoupled80,                  slice(None)),
               (MushroomBody,                           slice(None)),
               (COBAHHUncoupled,                        slice(None)),

               ## other benchmarks
               #(CUBAFixedConnectivityNoMonitor, slice(None)),
               #(COBAHHCoupled, slice(None)),
               #(STDPEventDriven, slice(None)),

               #(VerySparseMediumRateSynapsesOnly, slice(None)),
               #(SparseMediumRateSynapsesOnly, slice(None)),
               #(DenseMediumRateSynapsesOnly, slice(None)),
               #(SparseLowRateSynapsesOnly, slice(None)),
               #(SparseHighRateSynapsesOnly, slice(None)),

               #(DenseMediumRateSynapsesOnlyHeterogeneousDelays, slice(None)),
               #(SparseLowRateSynapsesOnlyHeterogeneousDelays, slice(None)),

               #(LinearNeuronsOnly, slice(None)),
               #(HHNeuronsOnly, slice(None)),

               ### below uses monitors
               #(CUBAFixedConnectivity, slice(None)),
               #(COBAHHFixedConnectivity, slice(None, -1)),
]

#sns.set_palette(sns.color_palette("hls", len(configurations)))
#sns.set_palette(sns.color_palette("cubehelix", len(configurations)))

time_stemp = time.time()
date_str = datetime.datetime.fromtimestamp(time_stemp).strftime('%Y-%m-%d_%T')

if not os.path.exists(directory):
    os.makedirs(directory)

data_dir = os.path.join(directory, 'data')
plot_dir = os.path.join(directory, 'plots')
log_dir = os.path.join(directory, 'logs')
prof_dir = os.path.join(directory, 'nvprof')
os.makedirs(data_dir)
os.makedirs(plot_dir)
os.makedirs(log_dir)
os.makedirs(prof_dir)
print(f"Saving results in {plot_dir}.")

shutil.copy(os.path.realpath(__file__), os.path.join(directory, 'run_benchmark_suite.py'))

time_format = '%d.%m.%Y at %H:%M:%S'
script_start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)

with open(os.path.join(directory, 'git.diff'), 'w') as diff_file:
    subprocess.call(['git', 'diff'], stdout=diff_file)

try:
    for i, (speed_test, n_slice) in enumerate(speed_tests):
        test_name = speed_test.__name__
        start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        print(f"Starting {test_name} on {start}.")
        maximum_run_time = 1000*second
        res = run_speed_tests(configurations=configurations,
                              speed_tests=[speed_test],
                              n_slice=n_slice,
                              #n_slice=slice(0,1,None),
                              run_twice=False,
                              verbose=True,
                              maximum_run_time=maximum_run_time,
                              mark_not_completed=True,
                              ## this needs modification of brian2 code
                              profile_only_active=True
                              #profile_only_active=False
                             )
        end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        diff = datetime.datetime.strptime(end, time_format) - datetime.datetime.strptime(start, time_format)
        print(f"Running {test_name} took {diff}.")
        res.plot_all_tests()
        ## this needs modification of brian2 code
        #res.plot_all_tests(print_relative=True)
        savefig(os.path.join(plot_dir, f'speed_test_{speed_tests[i][1]}_absolute.png'))
        res.plot_all_tests(relative=True)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_relative.png'))
        res.plot_all_tests(profiling_minimum=0.05)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_profiling.png'))
        if 3 != len(get_fignums()):
            print(f"WARNING: There were {len(get_fignums())} plots created, but only {3 * (i + 1)} saved.")
        for f in get_fignums():
            close(f)

        # pickel results object and csv file to disk
        pkl_file = os.path.join(data_dir, test_name + '.pkl' )
        pickle_results(res, pkl_file)
        try:
            translate_pkl_to_csv(pkl_file)
        except KeyError as e:
            print("ERROR tranlating {} to csv:\n\tKeyError: {}", pkl_file, e)

        try:
            for key in res.brian_stdouts.keys():
                config_key, st_key, n_key = key
                suffix = f'{st_key}_{config_key}_{n_key}.txt'
                stdout_file = os.path.join(log_dir, f'stdout_{suffix}')
                with open(stdout_file, 'w+') as sfile:
                    sfile.write(res.brian_stdouts[key])
                print(f"Written {stdout_file}")
                stderr_file = os.path.join(log_dir, f'stderr_{suffix}')
                with open(stderr_file, 'w+') as sfile:
                    sfile.write(res.brian_stderrs[key])
                print(f"Written {stderr_file}")
                tb_file = os.path.join(log_dir, f'tb_{suffix}')
                with open(tb_file, 'w+') as tfile:
                    tfile.write(res.tracebacks[key])
                print(f"Written {tb_file}")
        except Exception as err:
            print(f"ERROR writing stdout files: {err}")
        except:
            print("ERROR writing stdout files and couldn't catch Exception...")

        # collect stdout logs
        for conf in configurations:
            for n in speed_test.n_range[n_slice]:
                proj_dir = conf.get_project_dir(feature_name=test_name, n=n)
                for stream in ['stdout', 'stderr']:
                    file = os.path.join(proj_dir, 'results', f'{stream}.txt')
                    if os.path.exists(file):
                        shutil.copy(
                            file,
                            os.path.join(
                                log_dir, os.path.basename(
                                    f'{stream}_{test_name}_{proj_dir}_{n}.txt'
                                )
                            )
                        )
                    else:
                        print(f"WARNING Couldn't save {file}, file not found.")

        # run nvprof on n_range[2] for all configurations
        if not args.no_nvprof:
            for conf in configurations:
                for n in speed_test.n_range[n_slice]:
                    proj_dir = conf.get_project_dir(feature_name=test_name, n=n)
                    main_args = ''
                    if 'genn_runs' in proj_dir:
                        main_args = f'test {speed_test.duration / second}'
                    ns = speed_test.n_range[n_slice]
                    idx = 4
                    max_runtime = 20
                    if isinstance(conf, DynamicConfigCreator):
                        conf_name = conf.name.replace(' ', '-').replace('(', '-').replace(')', '-')
                    else:
                        conf_name = conf.__name__
                    print(f"Rerunning {conf_name} with n = {speed_test.n_range[idx]} for nvprof profiling")
                    #tb, res, runtime, prof_info = results(conf, speed_test, speed_test.n_range[idx], maximum_run_time=maximum_run_time)
                    runtime = res.full_results[conf.name, speed_test.fullname(), n, 'All']
                    this_res = res.feature_result[conf.name, speed_test.fullname(), n]
                    tb = res.tracebacks[conf.name, speed_test.fullname(), n]
                    if not isinstance(this_res, Exception) and runtime < max_runtime:
                        nvprof_args = '--profile-from-start-off' if proj_dir.startswith("cuda_standalone") else ''
                        nvprof_n = speed_test.n_range[idx]
                        log_file=os.path.join(
                            prof_dir, f'nvprof_{test_name}_{conf_name}_{nvprof_n}.log'
                        )
                        cmd = (
                            f'cd {proj_dir}'
                            f' && nvprof'
                                f' {nvprof_args}'
                                f' --log-file ../{log_file}'
                                f' ./main {main_args}'
                        )
                        prof_start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                        cmd = cmd.replace('(', '\(').replace(')', '\)')
                        try:
                            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
                        except subprocess.CalledProcessError as err:
                            print(f"ERROR: nvprof failed with:{err} output: {err.output}")
                            raise
                        prof_end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                        prof_diff = datetime.datetime.strptime(prof_end, time_format) - datetime.datetime.strptime(prof_start, time_format)
                        print(f"Profiling took {prof_diff} for runtime of {runtime}")
                    elif isinstance(res, Exception):
                        print("Didn't run nvprof, got an Exception", res)
                    else:  # runtime >= max_runtime
                        print(f"Didn't run nvprof, runtime ({runtime}) >= max_runtime ({max_runtime})")

        # Delete project directories when done
        for conf in configurations:
            for n in speed_test.n_range[n_slice]:
                conf.delete_project_dir(feature_name=test_name, n=n)

finally:
    create_readme(directory)
    print(f"\nSummarized speed test results in {os.path.join(directory, 'README.md')}")
    script_end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
    script_diff = datetime.datetime.strptime(script_end, time_format) - datetime.datetime.strptime(script_start, time_format)
    print("Finished speed test on {}. Total time = {}.".format(
        datetime.datetime.fromtimestamp(time.time()).strftime(time_format), script_diff))


##res.plot_all_tests(relative=True)
#for n in get_fignums():
#    plt.figure(n)
#    savefig(plot_dir + '/speed_test_{}.png'.format(speed_tests[n-1][1]))

## Debug (includes profiling infos)
#from brian2.tests.features.base import results
#for x in results(LocalConfiguration, LinearNeuronsOnly, 10, maximum_run_time=10*second):
#    print x
