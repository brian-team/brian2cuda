from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.monitors import SpikeMonitorTest, StateMonitorTest
from brian2.tests.features.input import SpikeGeneratorGroupTest

import brian2cuda
from brian2cuda.tests.cuda_configuration import CUDAStandaloneConfiguration

#import brian2genn
#from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationOptimized

# Test cuda, genn, cpp
res = run_feature_tests([CPPStandaloneConfiguration,
                         #GeNNConfiguration,
                         #GeNNConfigurationOptimized,
                         CUDAStandaloneConfiguration])
print res.tables_and_exceptions

# Quick testing
# print run_feature_tests(configurations=[Default...
#                                         NumpyConfiguration,
#                                         WeaveConfiguration],
#                         feature_tests=[SpikeGeneratorGroupTest]).tables_and_exceptions

# Specific testing
#from brian2.tests.features.synapses import SynapsesSTDP, SynapsesPost
#print run_feature_tests(feature_tests=[SynapsesPost]).tables_and_exceptions
#print run_feature_tests(feature_tests=[SynapsesPost],
#                        configurations=[DefaultConfiguration,
#                                        WeaveConfiguration]).exceptions
