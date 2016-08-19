# run tests without X-server
import matplotlib
matplotlib.use('Agg')

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *

import brian2cuda
from brian2cuda.tests.cuda_configuration import CUDAStandaloneConfiguration
from brian2cuda.tests.speed import *
#from brian2genn.correctness_testing import GeNNConfiguration

# Full testing
# res = run_speed_tests()
# res.plot_all_tests()
# show()


# Quick testing
res = run_speed_tests(configurations=[CUDAStandaloneConfiguration,
                                      #NumpyConfiguration,
                                      #WeaveConfiguration,
                                      #LocalConfiguration,
                                      CPPStandaloneConfiguration,
                                      #CPPStandaloneConfigurationOpenMP,
                                      #GeNNConfiguration,
                                      ],
                      speed_tests=[#LinearNeuronsOnly,
                                   #HHNeuronsOnly,
                                   #CUBAFixedConnectivity,
                                   #VerySparseMediumRateSynapsesOnly,
                                   #SparseMediumRateSynapsesOnly,
                                   #DenseMediumRateSynapsesOnly,
                                   #SparseLowRateSynapsesOnly,
                                   #SparseHighRateSynapsesOnly,

                                   #AdaptationOscillation,
                                   BrunelHakimModel,
                                   BrunelHakimModelWithDelay,
                                   #COBAHH,
                                   #STDPEventDriven,
                                   #STDPNotEventDriven,
                                   #Vogels,
                                   #VogelsWithSynapticDynamic
                                   ],
                      #n_slice=slice(None, None, 3),
                      #n_slice=slice(None, -1),
                      n_slice=slice(None, None, 100),
                      run_twice=False,
                      maximum_run_time=1*second,
                      verbose=True
                      )
res.plot_all_tests()
res.plot_all_tests(relative=True)
for n in get_fignums():
    savefig('speed_tests_fig{}.png'.format(n))
#show()

# Debug
# from brian2.tests.features.base import results
# for x in results(LocalConfiguration, LinearNeuronsOnly, 10, maximum_run_time=10*second):
#     print x
