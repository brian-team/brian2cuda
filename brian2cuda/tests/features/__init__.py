__all__ = ['FeatureTest',
           'SpeedTest',
           'InaccuracyError',
           'Configuration',
           'run_feature_tests']

from brian2.tests.features.base import *
import brian2.tests.features.neurongroup
import brian2.tests.features.synapses
import brian2.tests.features.monitors
import brian2.tests.features.input
import brian2cuda.tests.features.speed
