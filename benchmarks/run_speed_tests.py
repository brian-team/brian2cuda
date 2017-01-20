import os

# run tests without X-server
import matplotlib
matplotlib.use('Agg')

# pretty plots
import seaborn

import time
import datetime

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *

import brian2cuda
from brian2cuda.tests.features.cuda_configuration import CUDAStandaloneConfiguration
from brian2cuda.tests.features.speed import *

from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationCPU, GeNNConfigurationOptimized

prefs.devices.cpp_standalone.extra_make_args_unix = ['-j6']

time_stemp = time.time()
date_str = datetime.datetime.fromtimestamp(time_stemp).strftime('%Y_%m_%d')

configurations = [
                  CUDAStandaloneConfiguration,
                  #NumpyConfiguration,
                  #WeaveConfiguration,
                  #LocalConfiguration,
                  CPPStandaloneConfiguration,
                  #CPPStandaloneConfigurationOpenMP,
                  GeNNConfiguration,
                  #GeNNConfigurationCPU,
                  #GeNNConfigurationOptimized
                  ]

speed_tests = [# feature_test                     name                                  n_slice
               (LinearNeuronsOnly,               'LinearNeuronsOnly',                   slice(None)         ),
               (HHNeuronsOnly,                   'HHNeuronsOnly',                       slice(None)         ),

               (BrunelHakimModel,                'BrunelHakimModel',                    slice(None)         ),
               (BrunelHakimModelWithDelay,       'BrunelHakimModelWithDelay',           slice(None)         ),

#               (CUBAFixedConnectivity,           'CUBAFixedConnectivity',               slice(None)         ),
#               (VerySparseMediumRateSynapsesOnly,'VerySparseMediumRateSynapsesOnly',    slice(None)         ),
#               (SparseMediumRateSynapsesOnly,    'SparseMediumRateSynapsesOnly',        slice(None)         ),
#               (DenseMediumRateSynapsesOnly,     'DenseMediumRateSynapsesOnly',         slice(None)         ),
#               (SparseLowRateSynapsesOnly,       'SparseLowRateSynapsesOnly',           slice(None)         ),
#               (SparseHighRateSynapsesOnly,      'SparseHighRateSynapsesOnly',          slice(None)         ),
#
#               (AdaptationOscillation,           'AdaptationOscillation',               slice(None)         ),
#               (COBAHH,                          'COBAHH',                              slice(None)         ),
#               (STDPEventDriven,                 'STDPEventDriven',                     slice(None)         ),
#               (STDPNotEventDriven,              'STDPNotEventDriven',                  slice(None)         ),
#               (Vogels,                          'Vogels',                              slice(None)         ),
#               (VogelsWithSynapticDynamic,       'VogelsWithSynapticDynamic',           slice(None)         ),
#
#               (COBAHHFixedConnectivity,         'COBAHHFixedConnectivity',             slice(None, -1)     ),
#               (STDP,                            'STDP',                                slice(None)         ),
]

directory = 'results_{}'.format(date_str)
if os.path.exists(directory):
    new_dir = directory + '_bak_' + str(int(time.time()))
    print("Directory with name `{}` already exists. Renaming it to `{}`.".format(directory, new_dir))
    os.rename(directory, new_dir)
os.makedirs(directory)
print("Saving result plots in {}.".format(directory))

for n, (st, name, sl) in enumerate(speed_tests):
    print("Starting speed test on", datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y at %H:%M:%S'))
    res = run_speed_tests(configurations=configurations,
                          speed_tests=[st],
                          n_slice=sl,
                          #n_slice=slice(None,None,100),
                          run_twice=False,
                          verbose=True)
    res.plot_all_tests()
    savefig(directory + '/speed_test_{}.png'.format(speed_tests[n][1]))
    res.plot_all_tests(relative=True)
    savefig(directory + '/speed_test_{}_relative.png'.format(name))
    res.plot_all_tests(profiling_minimum=0.05)
    savefig(directory + '/speed_test_{}_profiling.png'.format(name))
    if (3*(n+1) != len(get_fignums())):
        print("WARNING: There were {} plots created, but only {} saved.".format(len(get_fignums()), 3*(n+1)))

print("Finished all speed test on", datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y at %H:%M:%S'))


##res.plot_all_tests(relative=True)
#for n in get_fignums():
#    plt.figure(n)
#    savefig(directory + '/speed_test_{}.png'.format(speed_tests[n-1][1]))

## Debug (includes profiling infos)
#from brian2.tests.features.base import results
#for x in results(LocalConfiguration, LinearNeuronsOnly, 10, maximum_run_time=10*second):
#    print x
