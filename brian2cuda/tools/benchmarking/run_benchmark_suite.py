import argparse

parser = argparse.ArgumentParser(description='Run brian2cuda benchmarks')

parser.add_argument('-d', '--results-dir', default=None,
                    help="Directory where results will be stored")

parser.add_argument('--dry-run', action='store_true',
                    help=("Exit script after argument parsing. Used to check "
                          "validity of arguments"))

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
    directory = args.results_dir

import os
import shutil
import glob
import subprocess
import sys
import socket
import shlex

# run tests without X-server
import matplotlib
matplotlib.use('Agg')

# pretty plots
#import seaborn as sns

import time
import datetime
import pickle as pickle

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
                                                          GeNNConfigurationOptimized,
                                                          GeNNConfigurationOptimizedSpanTypePre,
                                                          GeNNConfigurationOptimizedSinglePrecision,
                                                          GeNNConfigurationOptimizedSinglePrecisionSpanTypePre)


#from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationCPU, GeNNConfigurationOptimized

from create_readme import create_readme
from helpers import pickle_results, translate_pkl_to_csv

suppress_brian2_logs()
# Uncomment this to get brian2cuda logs
# (e.g. for DynamicConfigCreator diagnostic messages)
#BrianLogger.log_level_diagnostic()

configs = [# configuration                          project_directory
          #(NumpyConfiguration,                     None),
          #(WeaveConfiguration,                     None),
          #(LocalConfiguration,                     None),
          (CPPStandaloneConfiguration,              'cpp_standalone'),
          #(CPPStandaloneConfigurationSinglePrecision,              'cpp_standalone'),
          (CPPStandaloneConfigurationOpenMPMaxThreads,        'cpp_standalone'),
          #(CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision,        'cpp_standalone'),

          # max blocks
          (DynamicConfigCreator('CUDA standalone (max blocks, atomics)'),
           'cuda_standalone'),

          # (DynamicConfigCreator('CUDA standalone (single precision, max blocks, atomics)',
          #                       prefs={'core.default_float_dtype': float32}),
          #  'cuda_standalone'),


          # 1 block
          (DynamicConfigCreator('CUDA standalone (1 block, atomics)',
                                prefs={'devices.cuda_standalone.parallel_blocks': 1}),
           'cuda_standalone'),

          # (DynamicConfigCreator('CUDA standalone (single precision, 1 block, atomics)',
          #                       prefs={'core.default_float_dtype': float32,
          #                              'devices.cuda_standalone.parallel_blocks': 1}),
          #  'cuda_standalone'),


          # 20 blocks
          # (DynamicConfigCreator('CUDA standalone (20 blocks, atomics)',
          #                       prefs={'devices.cuda_standalone.parallel_blocks': 20}),
          #  'cuda_standalone'),

          # (DynamicConfigCreator('CUDA standalone (single precision, 20 blocks, atomics)',
          #                       prefs={'core.default_float_dtype': float32,
          #                              'devices.cuda_standalone.parallel_blocks': 20}),
          #  'cuda_standalone'),


          # 40 blocks
          # (DynamicConfigCreator('CUDA standalone (40 blocks, atomics)',
          #                       prefs={'devices.cuda_standalone.parallel_blocks': 40}),
          #  'cuda_standalone'),

          # (DynamicConfigCreator('CUDA standalone (single precision, 40 blocks, atomics)',
          #                       prefs={'core.default_float_dtype': float32,
          #                              'devices.cuda_standalone.parallel_blocks': 40}),
          #  'cuda_standalone'),


          # 60 blocks
          # (DynamicConfigCreator('CUDA standalone (60 blocks, atomics)',
          #                       prefs={'devices.cuda_standalone.parallel_blocks': 60}),
          #  'cuda_standalone'),

          # (DynamicConfigCreator('CUDA standalone (single precision, 60 blocks, atomics)',
          #                       prefs={'core.default_float_dtype': float32,
          #                              'devices.cuda_standalone.parallel_blocks': 60}),
          #  'cuda_standalone'),


          #(DynamicConfigCreator('CUDA standalone (max blocks, no atomics)',
          #                      prefs={'devices.cuda_standalone.use_atomics': False}),
          # 'cuda_standalone'),

          #(DynamicConfigCreator('CUDA standalone (1 block, no atomics)',
          #                      prefs={'devices.cuda_standalone.use_atomics': False,
          #                             'devices.cuda_standalone.parallel_blocks': 1}),
          # 'cuda_standalone'),

          #(DynamicConfigCreator('CUDA standalone (no bundles, 15 blocks, atomics)',
          #                      prefs={'devices.cuda_standalone.push_synapse_bundles': False}),
          # 'cuda_standalone'),


          #(DynamicConfigCreator('CUDA standalone (master)',
          #                      git_commit='master'),
          # 'cuda_standalone'),

          #(DynamicConfigCreator("CUDA standalone (only compilation)",
          #                      set_device_kwargs={'compile': True, 'run': False}),
          # 'cuda_standalone'),

          #(CUDAStandaloneConfiguration,             'cuda_standalone'),
          #(CUDAStandaloneConfigurationExtraThresholdKernel,             'cuda_standalone'),
          #(CUDAStandaloneConfigurationNoAssert,             'cuda_standalone'),
          #(CUDAStandaloneConfigurationNoCudaOccupancyAPI,      'cuda_standalone'),
          #(CUDAStandaloneConfigurationNoCudaOccupancyAPIProfileCPU,    'cuda_standalone'),
          #(CUDAStandaloneConfiguration2BlocksPerSM, 'cuda_standalone'),
          #(CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds, 'cuda_standalone'),
          #(CUDAStandaloneConfigurationSynLaunchBounds,     'cuda_standalone'),
          #(CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds, 'cuda_standalone'),
          #(CUDAStandaloneConfigurationProfileGPU,   'cuda_standalone'),
          #(CUDAStandaloneConfigurationProfileCPU,   'cuda_standalone'),
          #(GeNNConfiguration,                       'GeNNworkspace'),
          #(GeNNConfigurationCPU,                    'GeNNworkspace'),
          (GeNNConfigurationOptimized,              'GeNNworkspace'),
          # (GeNNConfigurationOptimizedSinglePrecision,'GeNNworkspace'),
          (GeNNConfigurationOptimizedSpanTypePre,   'GeNNworkspace'),
          #(GeNNConfigurationOptimizedSinglePrecisionSpanTypePre, 'GeNNworkspace'),
          ]

speed_tests = [# feature_test                     name                                  n_slice

               # paper benchmarks
               (BrunelHakimHomogDelays,                         'BrunelHakimHomogDelays',                       slice(None)         ),
               (BrunelHakimHeterogDelays,                       'BrunelHakimHeterogDelays',                     slice(None)         ),
               (BrunelHakimHeterogDelaysNarrowDistr,            'BrunelHakimHeterogDelaysNarrowDistr',          slice(None)         ),
               (STDPCUDAHeterogeneousDelays,                    'STDPCUDAHeterogeneousDelays',                  slice(None)         ),
               (STDPCUDAHomogeneousDelays,                      'STDPCUDAHomogeneousDelays',                    slice(None)         ),
               (STDPCUDA,                                       'STDPCUDA',                                     slice(None)         ),
               (COBAHHPseudocoupled1000,                        'COBAHHPseudocoupled1000',                      slice(None)         ),
               (COBAHHPseudocoupled80,                          'COBAHHPseudocoupled80',                        slice(None)         ),
               (MushroomBody,                                   'MushroomBody',                                 slice(None)         ),
               (COBAHHUncoupled,                                'COBAHHUncoupled',                              slice(0, -2)         ),

               ## other benchmarks
               #(CUBAFixedConnectivityNoMonitor,                 'CUBAFixedConnectivityNoMonitor',               slice(None)         ),
               #(COBAHHCoupled,                                  'COBAHHCoupled',                                slice(None)         ),
               #(STDPEventDriven,                                'STDPEventDriven',                              slice(None)         ),

               #(VerySparseMediumRateSynapsesOnly,               'VerySparseMediumRateSynapsesOnly',             slice(None)         ),
               #(SparseMediumRateSynapsesOnly,                   'SparseMediumRateSynapsesOnly',                 slice(None)         ),
               #(DenseMediumRateSynapsesOnly,                    'DenseMediumRateSynapsesOnly',                  slice(None)         ),
               #(SparseLowRateSynapsesOnly,                      'SparseLowRateSynapsesOnly',                    slice(None)         ),
               #(SparseHighRateSynapsesOnly,                     'SparseHighRateSynapsesOnly',                   slice(None)         ),

               #(DenseMediumRateSynapsesOnlyHeterogeneousDelays, 'DenseMediumRateSynapsesOnlyHeterogeneousDelays', slice(None)       ),
               #(SparseLowRateSynapsesOnlyHeterogeneousDelays,   'SparseLowRateSynapsesOnlyHeterogeneousDelays', slice(None)         ),

               #(LinearNeuronsOnly,                              'LinearNeuronsOnly',                            slice(None)         ),
               #(HHNeuronsOnly,                                  'HHNeuronsOnly',                                slice(None)         ),

               ### below uses monitors
               #(CUBAFixedConnectivity,                          'CUBAFixedConnectivity',                        slice(None)         ),
               #(COBAHHFixedConnectivity,                        'COBAHHFixedConnectivity',                      slice(None, -1)     ),
]

configurations = [config[0] for config in configs]
project_dirs = [config[1] for config in configs]

#sns.set_palette(sns.color_palette("hls", len(configurations)))
#sns.set_palette(sns.color_palette("cubehelix", len(configurations)))

# check if multiple Configurations with same project_dirs are specified
last_idx = {}
for proj_dir in project_dirs:
    if proj_dir is not None:
        first_i = project_dirs.index(proj_dir)
        last_i = len(project_dirs) - 1 - project_dirs[::-1].index(proj_dir)
        if first_i != last_i:
            print("WARNING there are multiple configurations using {d} as project "
                  "directory. Profiling and logfiles will only be saved for the last one {c}.".format(
                  d=proj_dir, c=configurations[last_i].__name__))
        last_idx[proj_dir] = last_i

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
    for n, (st, name, sl) in enumerate(speed_tests):
        start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        print(f"Starting {name} on {start}.")
        maximum_run_time = 1000*second
        res = run_speed_tests(configurations=configurations,
                              speed_tests=[st],
                              n_slice=sl,
                              #n_slice=slice(0,1,None),
                              run_twice=False,
                              verbose=True,
                              maximum_run_time=maximum_run_time,
                              mark_not_completed=True#,
                              ## this needs modification of brian2 code
                              #profile_only_active=True
                              #profile_only_active=False
                             )
        end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        diff = datetime.datetime.strptime(end, time_format) - datetime.datetime.strptime(start, time_format)
        print(f"Running {name} took {diff}.")
        res.plot_all_tests()
        ## this needs modification of brian2 code
        #res.plot_all_tests(print_relative=True)
        savefig(os.path.join(plot_dir, f'speed_test_{speed_tests[n][1]}_absolute.png'))
        res.plot_all_tests(relative=True)
        savefig(os.path.join(plot_dir, f'speed_test_{name}_relative.png'))
        res.plot_all_tests(profiling_minimum=0.05)
        savefig(os.path.join(plot_dir, f'speed_test_{name}_profiling.png'))
        if 3 != len(get_fignums()):
            print(f"WARNING: There were {len(get_fignums())} plots created, but only {3 * (n + 1)} saved.")
        for n in get_fignums():
            close(n)

        # pickel results object and csv file to disk
        pkl_file = os.path.join(data_dir, name + '.pkl' )
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
                tb_file = os.path.join(log_dir, f'tb_{suffix}')
                with open(tb_file, 'w+') as tfile:
                    tfile.write(res.tracebacks[key])
                print(f"Written {tb_file}")
        except Exception as err:
            print(f"ERROR writing stdout files: {err}")
        except:
            print("ERROR writing stdout files and couldn't catch Exception...")

        # save stdout log of last run (the other are deleted in run_speed_tests())
        for proj_dir in set(project_dirs):
            if not proj_dir is None and proj_dir in ['cuda_standalone', 'cpp_standalone']:
                config = configurations[last_idx[proj_dir]]
                stdout_file = os.path.join(proj_dir, 'results/stdout.txt')
                if os.path.exists(stdout_file):
                    shutil.copy(stdout_file,
                                os.path.join(log_dir, 'stdout_{st}_{conf}_{n}.txt'.format(st=name, conf=proj_dir,
                                                                                           n=st.n_range[sl][-1])))
                else:
                    print(f"WARNING Couldn't save {stdout_file},file not found.")

        # run nvprof on n_range[2]
        for conf, proj_dir in zip(configurations, project_dirs):
            main_arg = ''
            if proj_dir in ['cuda_standalone', 'GeNNworkspace']:
                if proj_dir == 'GeNNworkspace':
                    main_arg = f'test {st.duration / second} 1'
                ns = st.n_range[sl]
                idx = 4
                max_runtime = 20
                if isinstance(conf, DynamicConfigCreator):
                    conf_name = conf.name.replace(' ', '-').replace('(', '-').replace(')', '-')
                else:
                    conf_name = conf.__name__
                print(f"Rerunning {conf_name} with n = {st.n_range[idx]} for nvprof profiling")
                tb, res, runtime, prof_info = results(conf, st, st.n_range[idx], maximum_run_time=maximum_run_time)
                if not isinstance(res, Exception) and runtime < max_runtime:
                    option = '--profile-from-start-off' if proj_dir == 'cuda_standalone' else ''
                    cmd = 'cd {proj_dir} && nvprof {opt} --log-file ../{log_file} ./main {arg}'.format(
                        proj_dir=proj_dir, arg=main_arg, opt=option,
                        log_file=os.path.join(prof_dir, 'nvprof_{st}_{conf}_{n}.log'.format(
                            st=name, conf=conf_name, n=st.n_range[idx])))
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
                    print(f"Didn't run nvprof, runtime ({runtime}) >= max_runtime ({max_runtime}")
finally:
    create_readme(directory)
    print(f"\nSummarized speed test results in {directory + '/README.md'}")
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
