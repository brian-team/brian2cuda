import os
import shutil
import glob
import subprocess

# run tests without X-server
import matplotlib
matplotlib.use('Agg')

# pretty plots
import seaborn

import time
import datetime
import cPickle as pickle

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.base import results

import brian2cuda
from brian2cuda.tests.features.cuda_configuration import CUDAStandaloneConfiguration
from brian2cuda.tests.features.speed import *

from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationCPU, GeNNConfigurationOptimized

from create_readme import create_readme

prefs['devices.cpp_standalone.extra_make_args_unix'] = ['-j12']

configs = [# configuration                       project_directory
          #(NumpyConfiguration,                  None),
          #(WeaveConfiguration,                  None),
          #(LocalConfiguration,                  None),
          (CUDAStandaloneConfiguration,         'cuda_standalone'),
          (CPPStandaloneConfiguration,          'cpp_standalone'),
          (GeNNConfiguration,                   'GeNNworkspace'),
          #(CPPStandaloneConfigurationOpenMP,    'cpp_standalone'),
          #(GeNNConfigurationCPU,                'GeNNworkspace'),
          #(GeNNConfigurationOptimized,          'GeNNworkspace')
          ]

speed_tests = [# feature_test                     name                                  n_slice
               (LinearNeuronsOnly,                     'LinearNeuronsOnly',                   slice(None)         ),
               (HHNeuronsOnly,                         'HHNeuronsOnly',                       slice(None)         ),

               (BrunelHakimModelScalarDelay,           'BrunelHakimModelScalarDelay',         slice(None)         ),
               (BrunelHakimModelHeterogeneousDelay,    'BrunelHakimModelHeterogeneousDelay',  slice(None)         ),

               (STDP,                                   'STDP',                                slice(None)         ),
               (STDPEventDriven,                        'STDPEventDriven',                     slice(None)         ),
               (STDPNotEventDriven,                     'STDPNotEventDriven',                  slice(None)         ),
               (STDPMultiPost,                          'STDPMultiPost',                        slice(None)         ),
               (STDPNeuronalTraces,                     'STDPNeuronalTraces',                   slice(None)         ),
               (STDPMultiPostNeuronalTraces,            'STDPMultiPostNeuronalTraces',          slice(None)         ),

               (VerySparseMediumRateSynapsesOnly,       'VerySparseMediumRateSynapsesOnly',    slice(None)         ),
               (SparseMediumRateSynapsesOnly,           'SparseMediumRateSynapsesOnly',        slice(None)         ),
               (DenseMediumRateSynapsesOnly,            'DenseMediumRateSynapsesOnly',         slice(None)         ),
               (SparseLowRateSynapsesOnly,              'SparseLowRateSynapsesOnly',           slice(None)         ),
               (SparseHighRateSynapsesOnly,             'SparseHighRateSynapsesOnly',          slice(None)         ),

               (AdaptationOscillation,                  'AdaptationOscillation',               slice(None)         ),
               (COBAHH,                                 'COBAHH',                              slice(None)         ),
               (Vogels,                                 'Vogels',                              slice(None)         ),
               (VogelsWithSynapticDynamic,              'VogelsWithSynapticDynamic',           slice(None)         ),

               (COBAHHFixedConnectivity,                'COBAHHFixedConnectivity',             slice(None, -1)     ),
               (CUBAFixedConnectivity,                 'CUBAFixedConnectivity',               slice(None)         ),
]

configurations = [config[0] for config in configs]
project_dirs = [config[1] for config in configs]

# check if multiple Configurations with same project_dirs are specified
last_idx = {}
for proj_dir in project_dirs:
    if proj_dir is not None:
        first_i = project_dirs.index(proj_dir)
        last_i = len(project_dirs) - 1 - project_dirs[::-1].index(proj_dir)
        if first_i != last_i:
            print("WARNING there are multiple configurations using {d} as project "
                  "directory. Profiling and logfiles will only be saved for the last one {c}.".format(
                  d=proj_dir, c=configurations[last_idx]))
        last_idx[proj_dir] = last_i

time_stemp = time.time()
date_str = datetime.datetime.fromtimestamp(time_stemp).strftime('%Y_%m_%d')

directory = 'results_{}'.format(date_str)
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
        st.duration = 10*second
        res = run_speed_tests(configurations=configurations,
                              speed_tests=[st],
                              n_slice=sl,
                              #n_slice=slice(0,2,None),
                              #run_twice=False,
                              verbose=True,
                              maximum_run_time=maximum_run_time)
        end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        diff = datetime.datetime.strptime(end, time_format) - datetime.datetime.strptime(start, time_format)
        print("Running {} took {}.".format(name, diff))
        res.plot_all_tests()
        savefig(os.path.join(plot_dir, 'speed_test_{}_absolute.png'.format(speed_tests[n][1])))
        res.plot_all_tests(relative=True)
        savefig(os.path.join(plot_dir, 'speed_test_{}_relative.png'.format(name)))
        res.plot_all_tests(profiling_minimum=0.15)
        savefig(os.path.join(plot_dir, 'speed_test_{}_profiling.png'.format(name)))
        if 3 != len(get_fignums()):
            print("WARNING: There were {} plots created, but only {} saved.".format(len(get_fignums()), 3*(n+1)))
        for n in get_fignums():
            close(n)

        # pickel results object to disk
        pkl_file = os.path.join(data_dir, name + '.pkl' )
        with open(pkl_file, 'wb') as output:
                pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)

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
                conf_name = conf.__name__
                print("Rerunning {} with n = {} for nvprof profiling".format(conf_name, st.n_range[idx]))
                tb, res, runtime, prof_info = results(conf, st, st.n_range[idx], maximum_run_time=maximum_run_time)
                if not isinstance(res, Exception) and runtime < max_runtime:
                    cmd = 'cd {proj_dir} && nvprof --log-file ../{log_file} ./main {arg}'.format(
                        proj_dir=proj_dir, arg=main_arg,
                        log_file=os.path.join(prof_dir, 'nvprof_{st}_{conf}_{n}.log'.format(
                            st=name, conf=conf_name, n=st.n_range[idx])))
                    prof_start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                    print(cmd)
                    x = os.system(cmd)
                    if x:
                        print('nvprof failed with {}'.format(x))
                    prof_end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                    prof_diff = datetime.datetime.strptime(prof_end, time_format) - datetime.datetime.strptime(prof_start, time_format)
                    print("Profiling took {} for runtime of {}".format(prof_diff, runtime))
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
