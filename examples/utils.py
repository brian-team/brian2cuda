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
        dtype = type(value)
        if value is None:
            dtype = int
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

    name = os.path.basename(__main__.__file__).replace('.py', '')
    name += '_' + params['devicename'].replace('_standalone', '')

    ignores = ['devicename', 'resultsfolder', 'codefolder']
    for key, value in params.items():
        if key not in ignores and value is not None:
            # add the parameter value to the name
            key = key.replace('_', '-')
            if isinstance(value, bool):
                string = key if value else 'no-{}'.format(key)
            else:
                string = '{}-{}'.format(key, value)
            name += '_' + string

    if 'name_suffix' in params:
        name += '_' + params['name_suffix']

    hostname = socket.gethostname()
    name += '_' + hostname

    if params['devicename'] == 'cuda_standalone':
        dev_no_to_cc = None
        if hostname == 'merope':
            dev_no_to_cc = {0: '35'}
        if hostname in ['elnath', 'adhara']:
            dev_no_to_cc = {0: '20'}
        elif hostname == 'sabik':
            dev_no_to_cc = {0: '61', 1: '52'}
        elif hostname == 'eltanin':
            dev_no_to_cc = {0: '52', 1: '61', 2: '61', 3: '61'}
        elif hostname == 'risha':
            dev_no_to_cc = {0: '70'}
        else:
            print(
                f"WARNING: can't recognize hostname. Compiling with "
                f"{prefs['devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc']}"
            )

        # TODO make this a preference
        #prefs['devices.cuda_standalone.default_device'] = gpu_device
        if dev_no_to_cc is not None:
            try:
                gpu_device = int(os.environ['CUDA_VISIBLE_DEVICES'])
            except KeyError:
                # default
                gpu_device = 0

            try:
                cc = dev_no_to_cc[gpu_device]
            except KeyError as err:
                raise AttributeError("unknown device number: {}".format(err))

            print("Compiling device code for compute capability "\
                    "{}.{}".format(cc[0], cc[1]))
            prefs['devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc'].remove('-arch=sm_35')
            prefs['devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc'].extend(['-arch=sm_{}'.format(cc)])

        if params['num_blocks'] is not None:
            prefs['devices.cuda_standalone.parallel_blocks'] = params['num_blocks']

        if not params['bundle_mode']:
            prefs['devices.cuda_standalone.push_synapse_bundles'] = False

        if not params['atomics']:
            prefs['devices.cuda_standalone.use_atomics'] = False

    if params['single_precision']:
        from numpy import float32
        prefs['core.default_float_dtype'] = float32

    return name
