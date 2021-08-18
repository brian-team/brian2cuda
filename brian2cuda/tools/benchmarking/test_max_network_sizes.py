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
import pickle as pickle

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

               # paper benchmarks
               (COBAHHUncoupled,                        'COBAHHUncoupled',                      slice(None)),
               (COBAHHPseudocoupled1000,                'COBAHHPseudocoupled1000',              slice(None)),
               (COBAHHPseudocoupled80,                  'COBAHHPseudocoupled80',                slice(None)),
               (BrunelHakimHomogDelays,                 'BrunelHakimHomogDelays',               slice(None)),
               (BrunelHakimHeterogDelays,               'BrunelHakimHeterogDelays',             slice(None)),
               (BrunelHakimHeterogDelaysNarrowDistr,    'BrunelHakimHeterogDelaysNarrowDistr',  slice(None)),
               (STDPCUDA,                               'STDPCUDA',                             slice(None)),
               (STDPCUDAHomogeneousDelays,              'STDPCUDAHomogeneousDelays',            slice(None)),
               (STDPCUDAHeterogeneousDelays,            'STDPCUDAHeterogeneousDelays',          slice(None)),
               (MushroomBody,                           'MushroomBody',                         slice(None)),


               #(CUBAFixedConnectivityNoMonitor,                  'CUBAFixedConnectivityNoMonitor',              slice(None)         ),
               #(COBAHHConstantConnectionProbability,             'COBAHHConstantConnectionProbability',         slice(None)         ),
               #(COBAHHFixedConnectivityNoMonitor,                'COBAHHFixedConnectivityNoMonitor',            slice(None)         ),
               #(AdaptationOscillation,                          'AdaptationOscillation',                        slice(None, -1)     ),
               #(Vogels,                                         'Vogels',                                       slice(None)         ),
               #(STDPEventDriven,                                'STDPEventDriven',                              slice(None)         ),
               #(BrunelHakimModelScalarDelay,                    'BrunelHakimModelScalarDelay',                  slice(None)         ),

               #(VerySparseMediumRateSynapsesOnly,               'VerySparseMediumRateSynapsesOnly',             slice(None)         ),
               #(SparseMediumRateSynapsesOnly,                   'SparseMediumRateSynapsesOnly',                 slice(None)         ),
               #(DenseMediumRateSynapsesOnly,                    'DenseMediumRateSynapsesOnly',                  slice(None)         ),
               #(SparseLowRateSynapsesOnly,                      'SparseLowRateSynapsesOnly',                    slice(None)         ),
               #(SparseHighRateSynapsesOnly,                     'SparseHighRateSynapsesOnly',                   slice(None)         ),

               #(STDPNotEventDriven,                             'STDPNotEventDriven',                           slice(None)         ),

               #(DenseMediumRateSynapsesOnlyHeterogeneousDelays, 'DenseMediumRateSynapsesOnlyHeterogeneousDelays', slice(None)       ),
               #(SparseLowRateSynapsesOnlyHeterogeneousDelays,   'SparseLowRateSynapsesOnlyHeterogeneousDelays', slice(None)         ),
               #(BrunelHakimModelHeterogeneousDelay,             'BrunelHakimModelHeterogeneousDelay',           slice(None)         ),

               #(LinearNeuronsOnly,                              'LinearNeuronsOnly',                            slice(None)         ),
               #(HHNeuronsOnly,                                  'HHNeuronsOnly',                                slice(None)         ),
               #(VogelsWithSynapticDynamic,                      'VogelsWithSynapticDynamic',                    slice(None)         ),

               #### below uses monitors
               #(CUBAFixedConnectivity,                          'CUBAFixedConnectivity',                        slice(None)         ),
               #(COBAHHFixedConnectivity,                        'COBAHHFixedConnectivity',                      slice(None)         ),
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
        if name.startswith('STDPCUDA'):
            n = (n//1000) * 1000
        start = time.time()
        script_start = datetime.datetime.fromtimestamp(start).strftime(time_format)
        print(f"LOG RUNNING {name} with n={n} (start at {script_start})")
        tb, res, runtime, prof_info = results(CUDAStandaloneConfiguration, ft, n)
        print(f"LOG FINISHED {name} with n={n} in {time.time() - start:.2f}s")
        codes.append((n, res, tb))
        diff = np.abs(n - n_prev)
        n_prev = n
        if isinstance(res, Exception):
            print("LOG FAILED:\n", res, tb)
            has_failed = True
            if i == 0:
                print(f"LOG FAILED at first run ({name}) n={n}\n\terror={res}\nt\ttb={tb}")
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
            print(f"LOG BREAK {name} after n={n} (diff={diff})")
            break
    print(f"LOG FINAL RESULTS FOR {name}:")
    for n, res, tb in codes:
        if isinstance(res, Exception):
            sym = 'ERROR'
        else:
            sym = 'PASS'
        print(f"\t{n}\t\t{sym}")
