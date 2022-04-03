import __main__
import argparse


def update_from_command_line(params, choices={}):
    '''
    Use argparse to overwrite parameters in the `params` dictionary with command
    line options. Try to import as little modules before calling this function
    as possible. This gives a much faster response to the `--help` flag.

    Parameters
    ----------
    params : dict
        Dictionary with experiment parameters.
    '''
    if not isinstance(choices, dict):
        raise TypeError("`choices` needs to be dict, got "
                        "{}".format(type(choices)))

    # make a copy such that we can delete used keys without modifying choices
    choices_copy = choices.copy()

    parser = argparse.ArgumentParser(description='Run brian2cuda example')

    for key, value in params.items():
        if key == 'runtime':
            dtype = float
        else:
            dtype = type(value)

        if value is None:
            dtype = int

        if key == 'devicename':
            choice = ['cuda_standalone', 'cpp_standalone', 'genn']
        elif key == 'genn_mode':
            choices = ['pre', 'post']
        else:
            choice = choices_copy.pop(key, None)

        cmd_arg = key.replace('_', '-')
        if dtype == bool:
            assert choice is None, "Can't have choices for bool arguments"
            feature_parser = parser.add_mutually_exclusive_group(required=False)
            feature_parser.add_argument('--{}'.format(cmd_arg),
                                        dest=key, action='store_true')
            feature_parser.add_argument('--no-{}'.format(cmd_arg),
                                        dest=key, action='store_false')
            parser.set_defaults(**{key: None})
        else:
            parser.add_argument('--{}'.format(cmd_arg), type=dtype,
                                choices=choice, default=None)

    if choices_copy:
        raise AttributeError("Undefined keys in choices: "
                             "{}".format(choices_copy.keys()))

    parser.add_argument('--name-suffix', type=str, default=None)

    args = parser.parse_args()

    if args.name_suffix is not None:
        params['name_suffix'] = args.name_suffix

    for key in sorted(params.keys()):
        arg = getattr(args, key)
        if arg is not None:
            print("Setting {} from command line to {}.".format(key, arg))
            params[key] = arg


def _update_config_name(name, params, only_keys=None, ignore_keys=None):
    '''
    Append to ``name`` string all key-value pair information in ``params``. If
    ``only_keys`` is given, only update for keys in ``only_keys``. If ``ignore_keys`` is
    given, only update for all keys not in ``ignore_keys``.
    '''
    assert only_keys is None or ignore_keys is None, \
        "Can't have both, ``ignore_keys`` and ``only_keys``"

    filtered_keys = []
    for key in params.keys():
        if ignore_keys is not None and key not in ignore_keys:
            filtered_keys.append(key)
        elif only_keys is not None and key in only_keys:
            filtered_keys.append(key)

    for key in filtered_keys:
        value = params[key]
        string = key.replace('_', '-')
        if isinstance(value, bool):
            string = string if value else f'no-{string}'
        else:
            string = f'{string}-{value}'
        name += '_' + string

    return name


def set_prefs(params, prefs):
    '''
    Set brian2 preferences depending on parameter dictionary and create a
    experiment name from the parameters.

    Parameters
    ----------
    params : dict
        Dictionary with experiment parameters.
    prefs : BrianGlobalPreferences
        Brian's preferences object (accesible from brian2.prefs).

    Returns
    -------
    str
        Experiment name deduced from the `params` dictionary.
    '''
    import os
    import socket
    from numpy import float32, float64

    name = os.path.basename(__main__.__file__).replace('.py', '')
    name += '_' + params['devicename'].replace('_standalone', '')

    ignores = ['devicename', 'resultsfolder', 'codefolder']
    only_cuda = ['bundle_mode', 'atomics', 'partitions']
    only_cpp = ['cpp_threads']
    only_genn = ['genn_mode']
    all_ignores = ignores + only_cuda + only_cpp + only_genn

    name = _update_config_name(name, params, ignore_keys=all_ignores)

    if params['devicename'] == 'cuda_standalone':
        prefs['devices.cuda_standalone.parallel_blocks'] = params['partitions']
        prefs['devices.cuda_standalone.push_synapse_bundles'] = params['bundle_mode']
        prefs['devices.cuda_standalone.use_atomics'] = params['atomics']
        name = _update_config_name(name, params, only_keys=only_cuda)
    elif params['devicename'] == 'cpp_standalone':
        prefs['devices.cpp_standalone.openmp_threads'] = params['cpp_threads']
        name = _update_config_name(name, params, only_keys=only_cpp)
    elif params['devicename'] == 'genn':
        if params['genn_pre_mode']:
            prefs['devices.genn.synapse_span_type'] = 'PRESYNAPTIC'
        name = _update_config_name(name, params, only_keys=only_genn)

    if params['single_precision']:
        prefs['core.default_float_dtype'] = float32
    else:
        prefs['core.default_float_dtype'] = float64

    if 'name_suffix' in params:
        name += '_' + params['name_suffix']

    hostname = socket.gethostname()
    name += '_' + hostname


    return name
