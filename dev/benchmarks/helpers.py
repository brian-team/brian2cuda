import os
import cPickle as pickle
import pandas as pd
import brian2
import brian2.tests.features.base
from collections import defaultdict


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


def translate_pkl_to_csv(pkl_file, profile_suffixes=None):
    '''
    Extract from a pickled SpeedTestResults object the recorded runtimes in
    .csv files.

    Parameters
    ----------
    pkl_file : str
        Path to pickle file.
    profile_suffixes : list (optional)
        List of profiled code object names for which seperate .csv files should
        be created. If None, all recorded code objects are translated to .csv.
    -------
    '''
    with open(pkl_file) as f:
        speed_test_res = pickle.load(f)

    # we have only one speed test per .pkl file
    assert len(speed_test_res.speed_tests) == 1, speed_test_res.speed_tests
    speed_test = speed_test_res.speed_tests[0]
    speed_test_name = speed_test.fullname()
    full_results_dict = speed_test_res.full_results
    ns = speed_test_res.get_ns(speed_test_name)
    configs = speed_test_res.configurations

    # get all suffixes if speed test was run with profiling
    recorded_suffixes = [key[-1] for key in full_results_dict.keys()]
    if profile_suffixes is None:
        profile_suffixes = recorded_suffixes
    assert set(profile_suffixes).issubset(set(recorded_suffixes)), \
            "Not all measurement found for recorded_suffixes {}. Must be in {}".format(
                profile_suffixes, recorded_suffixes)

    for suffix in profile_suffixes:
        file_name = os.path.splitext(pkl_file)[0] + '_' + suffix + '.csv'
        create_csv(file_name, speed_test_name, full_results_dict, configs,
                   ns, suffix)


def create_csv(file_name, speed_test_name, full_results_dict, configs, ns,
               suffix):
    '''
    Create .csv file for one profile suffix if not all time measurements are
    zero (in which case speed test was run without profiling).
    '''
    config_names = []
    result_dict = defaultdict(list)
    for config in configs:
        config_name = config.name
        config_names.append(config.name)
        for n in ns:
            res = full_results_dict[config_name,
                                    speed_test_name,
                                    n,
                                    suffix]
            res = brian2.asarray(res)
            if not brian2.any(res):
                # don't save codeobjects that were not profiled
                return
            result_dict[n].append(res)

    result_df = pd.DataFrame.from_dict(result_dict, orient='index',
                                       columns=config_names)
    result_df.sort_index().to_csv(file_name)


if __name__ == '__main__':
    translate_pkl_to_csv('results_2018_08_13_genn_300/data/BrunelHakimModelScalarDelay.pkl')
