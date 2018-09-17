import __main__
import os
import socket
import argparse


def update_from_command_line(params):
    '''
    Use argparse to overwrite parameters in the `params` dictionary with command
    line options. Try to import as little modules before calling this function
    as possible. This gives a much faster response to the `--help` flag.

    Parameters
    ----------
    params : dict
        Dictionary with experiment parameters.
    '''

    parser = argparse.ArgumentParser(description='Run brian2cuda example')

    for key, value in params.items():
        dtype = type(value)
        if dtype == bool:
            feature_parser = parser.add_mutually_exclusive_group(required=False)
            feature_parser.add_argument('--{}'.format(key.replace('_', '-')),
                                        dest=key, action='store_true')
            feature_parser.add_argument('--no-{}'.format(key.replace('_', '-')),
                                        dest=key, action='store_false')
            parser.set_defaults(**{key: None})
        else:
            parser.add_argument('--{}'.format(key), type=dtype, default=None)

    parser.add_argument('--name-suffix', type=str, default=None)

    args = parser.parse_args()

    if args.name_suffix is not None:
        params['name_suffix'] = args.name_suffix

    for key in sorted(params.keys()):
        arg = getattr(args, key)
        if arg is not None:
            print "Setting {} from command line to {}.".format(key, arg)
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

    name = os.path.basename(__main__.__file__).replace('.py', '')
    name += '_' + params['devicename'].replace('_standalone', '')

    for key, value in params.items():
        if key != 'devicename' and value is not None:
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
        if hostname in ['elnath', 'adhara']:
            prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_35')
            prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

        if params['num_blocks'] is not None:
            prefs['devices.cuda_standalone.parallel_blocks'] = params['num_blocks']

        if not params['bundle_mode']:
            prefs['devices.cuda_standalone.push_synapse_bundles'] = False

        if not params['atomics']:
            prefs['codegen.generators.cuda.use_atomics'] = False

    if params['single_precision']:
        prefs['core.default_float_dtype'] = float32

    return name
