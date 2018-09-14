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
import seaborn as sns

import time
import datetime
import cPickle as pickle

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.base import results
from brian2.utils.logger import BrianLogger

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
                                                          CUDAStandaloneConfigurationProfileCPU)
from brian2cuda.tests.features.speed import *

from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationCPU, GeNNConfigurationOptimized

from create_readme import create_readme
from helpers import pickle_results, translate_pkl_to_csv

suppress_brian2_logs()
BrianLogger.log_level_diagnostic()

assert len(sys.argv)<= 2, 'Only one command line argument supported! Got {}'.format(len(sys.argv)-1)
if len(sys.argv) == 2:
    additional_dir_name = '_' + sys.argv[1]
else:
    additional_dir_name = ''

prefs['devices.cpp_standalone.extra_make_args_unix'] = ['-j12']

# host specific settings
if socket.gethostname() == 'elnath':
    prefs['devices.cpp_standalone.extra_make_args_unix'] = ['-j24']
    prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_35')
    prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

configs = [# configuration                          project_directory
          (NumpyConfiguration,                     None),
          (WeaveConfiguration,                     None),
          (LocalConfiguration,                     None),

          (DynamicConfigCreator('CUDA standalone'),
           'cuda_standalone'),

          (DynamicConfigCreator('CUDA standalone bundles',
                                git_commit='nemo_bundles'),
           'cuda_standalone'),

          (DynamicConfigCreator("CUDA standalone (profile='blocking')",
                                set_device_kwargs={'profile': 'blocking'}),
           'cuda_standalone'),
          (DynamicConfigCreator("CUDA standalone with 2 blocks per SM",
                                prefs={'devices.cuda_standalone.SM_multiplier': 2}),
           'cuda_standalone'),

          (CUDAStandaloneConfiguration,             'cuda_standalone'),
          (CUDAStandaloneConfigurationExtraThresholdKernel,             'cuda_standalone'),
          (CUDAStandaloneConfigurationNoAssert,             'cuda_standalone'),
          (CUDAStandaloneConfigurationNoCudaOccupancyAPI,      'cuda_standalone'),
          (CUDAStandaloneConfigurationNoCudaOccupancyAPIProfileCPU,    'cuda_standalone'),
          (CUDAStandaloneConfiguration2BlocksPerSM, 'cuda_standalone'),
          (CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds, 'cuda_standalone'),
          (CUDAStandaloneConfigurationSynLaunchBounds,     'cuda_standalone'),
          (CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds, 'cuda_standalone'),
          (CUDAStandaloneConfigurationProfileGPU,   'cuda_standalone'),
          (CUDAStandaloneConfigurationProfileCPU,   'cuda_standalone'),
          (CPPStandaloneConfiguration,              'cpp_standalone'),
          (GeNNConfiguration,                       'GeNNworkspace'),
          (CPPStandaloneConfigurationOpenMP,        'cpp_standalone'),
          (GeNNConfigurationCPU,                    'GeNNworkspace'),
          (GeNNConfigurationOptimized,              'GeNNworkspace')
          ]

speed_tests = [# feature_test                     name                                  n_slice

               (CUBAFixedConnectivityNoMonitor,                 'CUBAFixedConnectivityNoMonitor',               slice(None)         ),
               (COBAHHUncoupled,                                'COBAHHUncoupled',                              slice(None)         ),
               (COBAHHCoupled,                                  'COBAHHCoupled',                                slice(None)         ),
               (COBAHHPseudocoupled,                            'COBAHHPseudocoupled',                          slice(None)         ),
               (STDP,                                           'STDP',                                         slice(None)         ),
               (STDPEventDriven,                                'STDPEventDriven',                              slice(None)         ),
               (BrunelHakimModelScalarDelay,                    'BrunelHakimModelScalarDelay',                  slice(None)         ),
               (MushroomBody,                                   'MushroomBody',                                 slice(None)         ),

               (VerySparseMediumRateSynapsesOnly,               'VerySparseMediumRateSynapsesOnly',             slice(None)         ),
               (SparseMediumRateSynapsesOnly,                   'SparseMediumRateSynapsesOnly',                 slice(None)         ),
               (DenseMediumRateSynapsesOnly,                    'DenseMediumRateSynapsesOnly',                  slice(None)         ),
               (SparseLowRateSynapsesOnly,                      'SparseLowRateSynapsesOnly',                    slice(None)         ),
               (SparseHighRateSynapsesOnly,                     'SparseHighRateSynapsesOnly',                   slice(None)         ),

               (DenseMediumRateSynapsesOnlyHeterogeneousDelays, 'DenseMediumRateSynapsesOnlyHeterogeneousDelays', slice(None)       ),
               (SparseLowRateSynapsesOnlyHeterogeneousDelays,   'SparseLowRateSynapsesOnlyHeterogeneousDelays', slice(None)         ),
               (BrunelHakimModelHeterogeneousDelay,             'BrunelHakimModelHeterogeneousDelay',           slice(None)         ),

               (LinearNeuronsOnly,                              'LinearNeuronsOnly',                            slice(None)         ),
               (HHNeuronsOnly,                                  'HHNeuronsOnly',                                slice(None)         ),

               ## below uses monitors
               (CUBAFixedConnectivity,                          'CUBAFixedConnectivity',                        slice(None)         ),
               (COBAHHFixedConnectivity,                        'COBAHHFixedConnectivity',                      slice(None, -1)     ),
]

configurations = [config[0] for config in configs]
project_dirs = [config[1] for config in configs]

sns.set_palette(sns.color_palette("hls", len(configurations)))

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
date_str = datetime.datetime.fromtimestamp(time_stemp).strftime('%Y_%m_%d')

directory = 'results_{}{}'.format(date_str, additional_dir_name)
if os.path.exists(directory):
    new_dir = directory + '_bak_' + str(int(time.time()))
    print("Directory with name `{}` already exists. Renaming it to `{}`.".format(directory, new_dir))
    os.rename(directory, new_dir)
os.makedirs(directory)
data_dir = os.path.join(directory, 'data')
plot_dir = os.path.join(directory, 'plots')
log_dir = os.path.join(directory, 'logs')
prof_dir = os.path.join(directory, 'nvprof')
os.makedirs(data_dir)
os.makedirs(plot_dir)
os.makedirs(log_dir)
os.makedirs(prof_dir)
print("Saving results in {}.".format(plot_dir))

shutil.copy(os.path.realpath(__file__), os.path.join(directory, 'run_speed_test_script.py'))

time_format = '%d.%m.%Y at %H:%M:%S'
script_start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)

with open(os.path.join(directory, 'git.diff'), 'w') as diff_file:
    subprocess.call(['git', 'diff'], stdout=diff_file)

try:
    for n, (st, name, sl) in enumerate(speed_tests):
        start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        print("Starting {} on {}.".format(name, start))
        maximum_run_time = 1*60*60*second
        res = run_speed_tests(configurations=configurations,
                              speed_tests=[st],
                              n_slice=sl,
                              #n_slice=slice(0,1,None),
                              run_twice=False,
                              verbose=True,
                              maximum_run_time=maximum_run_time#,
                              ## this needs modification of brian2 code
                              #profile_only_active=True
                              #profile_only_active=False
                             )
        end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        diff = datetime.datetime.strptime(end, time_format) - datetime.datetime.strptime(start, time_format)
        print("Running {} took {}.".format(name, diff))
        res.plot_all_tests()
        ## this needs modification of brian2 code
        #res.plot_all_tests(print_relative=True)
        savefig(os.path.join(plot_dir, 'speed_test_{}_absolute.png'.format(speed_tests[n][1])))
        res.plot_all_tests(relative=True)
        savefig(os.path.join(plot_dir, 'speed_test_{}_relative.png'.format(name)))
        res.plot_all_tests(profiling_minimum=0.05)
        savefig(os.path.join(plot_dir, 'speed_test_{}_profiling.png'.format(name)))
        if 3 != len(get_fignums()):
            print("WARNING: There were {} plots created, but only {} saved.".format(len(get_fignums()), 3*(n+1)))
        for n in get_fignums():
            close(n)

        # pickel results object and csv file to disk
        pkl_file = os.path.join(data_dir, name + '.pkl' )
        pickle_results(res, pkl_file)
        try:
            translate_pkl_to_csv(pkl_file)
        except KeyError as e:
            print("ERROR tranlating {} to csv:\n\tKeyError: {}", pkl_file, e)

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
                    print("WARNING Couldn't save {},file not found.".format(stdout_file))

        # run nvprof on n_range[2]
        for conf, proj_dir in zip(configurations, project_dirs):
            main_arg = ''
            if proj_dir in ['cuda_standalone', 'GeNNworkspace']:
                if proj_dir == 'GeNNworkspace':
                    main_arg = 'test {time} 1'.format(time=st.duration/second)
                ns = st.n_range[sl]
                idx = 2
                max_runtime = 20
                if isinstance(conf, DynamicConfigCreator):
                    conf_name = conf.name.replace(' ', '-').replace('(', '-').replace(')', '-')
                else:
                    conf_name = conf.__name__
                print("Rerunning {} with n = {} for nvprof profiling".format(conf_name, st.n_range[idx]))
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
                        print("ERROR: nvprof failed with:{} output: {}".format(err, err.output))
                        raise
                    prof_end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                    prof_diff = datetime.datetime.strptime(prof_end, time_format) - datetime.datetime.strptime(prof_start, time_format)
                    print("Profiling took {} for runtime of {}".format(prof_diff, runtime))
                elif isinstance(res, Exception):
                    print("Didn't run nvprof, got an Exception", res)
                else:  # runtime >= max_runtime
                    print("Didn't run nvprof, runtime ({}) >= max_runtime ({}".format(runtime, max_runtime))
finally:
    create_readme(directory)
    print("\nSummarized speed test results in {}".format(directory + '/README.md'))
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
