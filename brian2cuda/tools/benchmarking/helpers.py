import os
import pickle as pickle
import pandas as pd
import brian2
from brian2.tests.features.base import SpeedTestResults
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
    assert isinstance(results, SpeedTestResults)
    to_pickle = SpeedTestResults(
        results.full_results,
        configurations_to_dict(results.configurations),
        results.speed_tests,
        results.brian_stdouts,
        results.brian_stderrs,
        results.tracebacks
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
    with open(pkl_file, 'rb') as f:
        speed_test_res = pickle.load(f)

    # we have only one speed test per .pkl file
    assert len(speed_test_res.speed_tests) == 1, speed_test_res.speed_tests
    speed_test = speed_test_res.speed_tests[0]
    speed_test_name = speed_test.fullname()
    full_results_dict = speed_test_res.full_results
    ns = speed_test_res.get_ns(speed_test_name)
    configs = speed_test_res.configurations
    num_genn_configs = len([c['classname'] for c in configs if
                            c['classname'].lower().startswith('genn')])

    recorded_suffixes = set([key[-1] for key in full_results_dict.keys()])
    recorded_not_genn_suffixes = set([key[-1] for key in full_results_dict.keys()
                                      if not key[0].lower().startswith('genn')])
    recorded_genn_suffixes = set([key[-1] for key in full_results_dict.keys()
                                  if key[0].lower().startswith('genn')])

    if len(recorded_suffixes) <= 3:
        assert 'lrcf' in recorded_suffixes
        assert 'All' in recorded_suffixes
        assert 'Overheads' in recorded_suffixes, recorded_suffixes
        pairs = [(recorded_suffixes, None)]
    else:
        pairs = [(recorded_not_genn_suffixes, False), (recorded_genn_suffixes, True)]

    for rec_suffixes, genn in pairs:
        genn_suffix = '_genn' if genn else ''
        if profile_suffixes is None:
            this_profile_suffixes = rec_suffixes
        # this is probably gonna fail because suffixes in GeNN and other are different
        assert set(this_profile_suffixes).issubset(set(rec_suffixes)), \
                "Not all measurement found for recorded_suffixes {}. Must be in {}".format(
                    this_profile_suffixes, rec_suffixes)

        for suffix in this_profile_suffixes:
            file_name = os.path.splitext(pkl_file)[0] + genn_suffix + '_' + suffix + '.csv'
            create_csv(file_name, speed_test_name, full_results_dict, configs,
                       ns, suffix, genn=genn)


def create_csv(file_name, speed_test_name, full_results_dict, configs, ns,
               suffix, genn=None):
    '''
    Create .csv file for one profile suffix if not all time measurements are
    zero (in which case speed test was run without profiling).
    '''
    import brian2
    config_names = []
    result_dict = defaultdict(list)
    for config in configs:
        config_name = config.name
        if genn is None or config_name.lower().startswith('genn') == genn:
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


def plot_from_pkl(pkl_file, plot_dir, base_config='C++ standalone'):

    import matplotlib.pyplot as plt

    with open(pkl_file, 'rb') as f:
        # SpeedTestResults object
        res = pickle.load(f)

    configs = res.configurations
    confignames = [config.name for config in configs]
    if base_config in confignames:
        base_idx = confignames.index(base_config)
        # swap base_config with position 0
        configs[base_idx], configs[0] = configs[0], configs[base_idx]
    else:
        print("Couldn't find base_config {} in configurations. Not changing the "
              "baseline for relative speedup plots.".format(base_config,
                                                            pkl_file))

    # only one benchmark per pkl file
    name = res.speed_tests[0].__name__
    res.plot_all_tests()
    plt.savefig(os.path.join(plot_dir, f'speed_test_{name}_absolute.png'))
    plt.close()
    res.plot_all_tests(relative=True)
    plt.savefig(os.path.join(plot_dir, f'speed_test_{name}_relative.png'))
    plt.close()
    res.plot_all_tests(profiling_minimum=0.05)
    plt.savefig(os.path.join(plot_dir, f'speed_test_{name}_profiling.png'))
    plt.close()


if __name__ == '__main__':
    translate_pkl_to_csv('results_2018_08_13_genn_300/data/BrunelHakimModelScalarDelay.pkl')
