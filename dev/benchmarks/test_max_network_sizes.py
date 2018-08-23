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
import seaborn

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
from helpers import pickle_results


speed_tests = [# feature_test                     name                                  n_slice

               (CUBAFixedConnectivityNoMonitor,                  'CUBAFixedConnectivityNoMonitor',              slice(None)         ),
               (COBAHHConstantConnectionProbability,             'COBAHHConstantConnectionProbability',         slice(None)         ),
               (COBAHHFixedConnectivityNoMonitor,                'COBAHHFixedConnectivityNoMonitor',            slice(None)         ),
               (AdaptationOscillation,                          'AdaptationOscillation',                        slice(None, -1)     ),
               (Vogels,                                         'Vogels',                                       slice(None)         ),
               (STDPEventDriven,                                'STDPEventDriven',                              slice(None)         ),
               (BrunelHakimModelScalarDelay,                    'BrunelHakimModelScalarDelay',                  slice(None)         ),

               (VerySparseMediumRateSynapsesOnly,               'VerySparseMediumRateSynapsesOnly',             slice(None)         ),
               (SparseMediumRateSynapsesOnly,                   'SparseMediumRateSynapsesOnly',                 slice(None)         ),
               (DenseMediumRateSynapsesOnly,                    'DenseMediumRateSynapsesOnly',                  slice(None)         ),
               (SparseLowRateSynapsesOnly,                      'SparseLowRateSynapsesOnly',                    slice(None)         ),
               (SparseHighRateSynapsesOnly,                     'SparseHighRateSynapsesOnly',                   slice(None)         ),

               (STDPNotEventDriven,                             'STDPNotEventDriven',                           slice(None)         ),

               (DenseMediumRateSynapsesOnlyHeterogeneousDelays, 'DenseMediumRateSynapsesOnlyHeterogeneousDelays', slice(None)       ),
               (SparseLowRateSynapsesOnlyHeterogeneousDelays,   'SparseLowRateSynapsesOnlyHeterogeneousDelays', slice(None)         ),
               (BrunelHakimModelHeterogeneousDelay,             'BrunelHakimModelHeterogeneousDelay',           slice(None)         ),

               (LinearNeuronsOnly,                              'LinearNeuronsOnly',                            slice(None)         ),
               (HHNeuronsOnly,                                  'HHNeuronsOnly',                                slice(None)         ),
               (VogelsWithSynapticDynamic,                      'VogelsWithSynapticDynamic',                    slice(None)         ),

               ### below uses monitors
               (CUBAFixedConnectivity,                          'CUBAFixedConnectivity',                        slice(None)         ),
               (COBAHHFixedConnectivity,                        'COBAHHFixedConnectivity',                      slice(None)         ),
]

time_format = '%d.%m.%Y at %H:%M:%S'
for (ft, name, sl) in speed_tests:
    codes = []
    length = len(ft.n_range)
    sl_idcs = sl.indices(length)
    max_idx = sl_idcs[1] - sl_idcs[2]
    n_init = ft.n_range[max_idx]
    n = n_prev = n_init
    #n = ft.n_range[0]
    has_failed = False
    for i in range(10):
        start = time.time()
        script_start = datetime.datetime.fromtimestamp(start).strftime(time_format)
        print("LOG RUNNING {} with n={} (start at {})".format(name, n, script_start))
        tb, res, runtime, prof_info = results(CUDAStandaloneConfiguration, ft, n)
        print("LOG FINISHED {} with n={} in {:.2f}s".format(name, n, time.time()-start))
        codes.append((n, res, tb))
        diff = np.abs(n - n_prev)
        n_prev = n
        if isinstance(res, Exception):
            print("LOG FAILED:", res, tb)
            has_failed = True
            if i == 0:
                print("LOG FAILED at first run ({}) n={}\n\terror={}\nt\ttb={}".format(name, n, res, tb))
                break
            # assuming the first run passes, when we fail we always go down half the abs distance
            n -= int(0.5 * diff)
            print("LOG HALF DOWN")
        elif has_failed:
            n += int(0.5 * diff)
            print("LOG HALF UP")
        else:  # not has_failed
            n *= 2
            print("LOG DOUBLE")
        if i != 0 and diff <= 10000:
            print("LOG BREAK {} after n={} (diff={})".format(name, n, diff))
            break
    print("LOG FINAL RESULTS FOR {}:".format(name))
    for n, res, tb in codes:
        if isinstance(res, Exception):
            sym = 'ERROR'
        else:
            sym = 'PASS'
        print("\t{}\t\t{}".format(n, sym))
