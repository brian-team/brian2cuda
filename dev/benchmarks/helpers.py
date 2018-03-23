import cPickle as pickle
import brian2.tests.features.base


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError
        return self.__getitem__(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def configurations_to_dict(configurations):
    if isinstance(configurations[0], dotdict):
        return configurations
    new_configurations = []
    for config in configurations:
        new_config = dotdict()
        new_config['name'] = config.name
        new_config['classname'] = config.__name__
        new_config['module'] = config.__module__
        new_configurations.append(new_config)
    return new_configurations


def pickle_results(results, filename):
    assert isinstance(results, brian2.tests.features.base.SpeedTestResults)
    to_pickle = brian2.tests.features.base.SpeedTestResults(
        results.full_results,
        configurations_to_dict(results.configurations),
        results.speed_tests
    )
    with open(filename, 'wb') as output:
        pickle.dump(to_pickle, output, pickle.HIGHEST_PROTOCOL)



