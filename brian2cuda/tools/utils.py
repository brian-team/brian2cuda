"""Utility funcitons"""
import traceback
import time
import numpy as np
from itertools import chain, combinations
import subprocess


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def parse_arguments(parser):

    parser.add_argument('--targets', '-t', nargs='*',
                        default=['cuda_standalone'], type=str,
                        choices=['cuda_standalone', 'genn', 'cpp_standalone'],
                        help=("Which codegeneration targets to use, can be "
                              "multiple. Only standalone targets. "
                              "[default: 'cuda_standalone']"))

    parser.add_argument('--single-precision', '-sp', action='store_true',
                        help=("Set default float datatype to np.float32 "
                              "(sets prefs['core.default_float_dtype']) "))

    parser.add_argument('--no-bundles', '-nob', action='store_true',
                        help="Don't use bundle pushing")

    parser.add_argument('--no-atomics', '-noa', action='store_true',
                        help="Don't use atomics for effect application")

    parser.add_argument('--num-blocks', '-bl', default=None, type=int, nargs=1,
                        help=("Number of parallel blocks in post neuron blocks "
                              "structure. [default: number of SMs on GPU]"))

    parser.add_argument('--all-prefs', '-a', action='store_true',
                        help=("Run with all preference combinations."))

    parser.add_argument('--jobs', '-j', nargs=1, type=int, default=None,
                        help=("Number of jobs to run simultaneously for "
                              "compilations (passed as -j option to the make "
                              "command. [default: no job limit (make -j)]"))

    args = parser.parse_args()

    return args


def print_single_prefs(prefs_dict, set_prefs=None, return_lines=False):
    '''
    Print preferences stored in prefs_dict. If set_prefs is not None, it has to
    be brian2.prefs and the preferences in prefs_dict will be set.
    '''
    prints = []
    if not prefs_dict:
        prints.append("\t\tdefault preferences")
    else:
        for k, v in prefs_dict.items():
            if set_prefs is not None:
                # set preference
                set_prefs[k] = v
            try:
                # np.float32/64 has __name__ attribute
                v = v.__name__
            except AttributeError:
                # int and bool don't have __name__ attribute
                pass
            prints.append("\t\tprefs[{}] = {}".format(k, v))

    if return_lines:
        return prints
    else:
        print("\n".join(prints))


def print_prefs_combinations(configurations, condition=None, return_lines=False):
    prints = []
    for n, d in enumerate(configurations):
        if condition is None or condition[n]:
            prints.append("\t{}. run".format(n + 1))
            prints.extend(print_single_prefs(d, return_lines=True))

    if return_lines:
        return prints
    else:
        print("\n".join(prints))


def set_preferences(args, prefs, fast_compilation=True, suppress_warnings=True,
                    suppress_brian_infos=True, return_lines=False):

    import socket
    import brian2cuda  # need it for preferences (or set prefs conditinally)

    prints = []

    if suppress_brian_infos:
        # Suppress INFO log messages during testing
        from brian2.utils.logger import BrianLogger, LOG_LEVELS
        log_level = BrianLogger.console_handler.level
        BrianLogger.console_handler.setLevel(LOG_LEVELS['WARNING'])

    if 'genn' in args.targets:
        import brian2genn

    if fast_compilation:
        # Switch off cpp compiler optimization to get faster compilation times
        prefs['codegen.cpp.extra_compile_args_gcc'].extend(['-w', '-O0'])
        prefs['codegen.cpp.extra_compile_args_msvc'].extend(['/Od'])
        prints.append("Turning off  compiler optimizations for fast compilation")

    if suppress_warnings:
        # Surpress some warnings from nvcc compiler
        compile_args = ['-Xcudafe "--diag_suppress=declared_but_not_referenced"']
        prefs['codegen.cuda.extra_compile_args_nvcc'].extend(compile_args)
        prints.append("Suppressing compiler warnings")

    # TODO: remove once we set CC automatically from hardware
    # Could also just cudaDeviceQuery and grep capability from there?
    s = subprocess.run("nvidia-smi -q | grep 'Product Name' | awk -F': ' '{print $2}'",
                       shell=True, stdout=subprocess.PIPE)
    gpu_name = s.stdout.decode()
    if gpu_name.startswith("Tesla K40"):
        prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_61')
        prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_35'])
        print("INFO brian2cuda/tools/utils.py: Setting -arch=sm_35")

    if gpu_name.startswith("GeForce RTX 2080"):
        prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_61')
        prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_75'])
        print("INFO brian2cuda/tools/utils.py: Setting -arch=sm_75")

    #if socket.gethostname() == 'elnath':
    #    prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_61')
    #    prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_20'])

    #if socket.gethostname() == 'cognition13':
    #    prefs['codegen.cuda.extra_compile_args_nvcc'].remove('-arch=sm_61')
    #    prefs['codegen.cuda.extra_compile_args_nvcc'].extend(['-arch=sm_75'])

    # cognition14 hat die Tesla K40c, die hat sm 35 (current default?)

    if args.jobs is not None:
        k = 'devices.cpp_standalone.extra_make_args_unix'
        if '-j' not in prefs[k]:
            prints.append("WARNING: -j is not anymore in default pref[{}]".format(k))
        new_j = ['-j' + str(args.jobs[0])]
        prefs[k].remove('-j')
        prefs[k].extend(new_j)
        prints.append("Compiling with make {}".format(new_j[0]))

    if args.all_prefs:
        all_prefs_list = []
        if not args.single_precision:
            all_prefs_list.append({'core.default_float_dtype': np.float32})

        if args.num_blocks is None:
            all_prefs_list.append(
                {'devices.cuda_standalone.parallel_blocks': 1})

        if not args.no_bundles:
            all_prefs_list.append(
                {'devices.cuda_standalone.push_synapse_bundles': False})

        if not args.no_atomics:
            all_prefs_list.append(
                {'codegen.generators.cuda.use_atomics': False})

        # create a powerset (all combinations) of the all_prefs_list
        all_prefs_combinations = []
        for dict_tuple in powerset(all_prefs_list):
            all_prefs_combinations.append(dict(item for d in dict_tuple
                                               for item in d.items()))
    else:
        # just one prefs set (depending on other pref arguments)
        all_prefs_combinations = [None]

    fixed_prefs_dict = {}
    if args.single_precision:
        k = 'core.default_float_dtype'
        v = np.float32
        prefs[k] = v
        fixed_prefs_dict[k] = v

    if args.no_atomics:
        k = 'codegen.generators.cuda.use_atomics'
        v = False
        prefs[k] = v
        fixed_prefs_dict[k] = v

    if args.no_bundles:
        k = 'devices.cuda_standalone.push_synapse_bundles'
        v = False
        prefs[k] = v
        fixed_prefs_dict[k] = v

    if args.num_blocks is not None:
        k = 'devices.cuda_standalone.parallel_blocks'
        v = args.num_blocks[0]
        prefs[k] = v
        fixed_prefs_dict[k] = v

    if args.all_prefs:
        # add fixed_prefs_dict to all combinations
        for prefs_dict in all_prefs_combinations:
            prefs_dict.update(fixed_prefs_dict)

    prints.append("Running with the following prefs combinations:\n")
    if all_prefs_combinations[0] is not None:
        # TODO XXX
        prints.extend(print_prefs_combinations(all_prefs_combinations, return_lines=True))
    elif fixed_prefs_dict:
        prints.append("1 run with explicitly set preferences:")
        for k, v in fixed_prefs_dict.items():
            prints.append("\t\tprefs[{}] = {}".format(k, v))
    else:
        prints.append("1 run with default preferences")
    prints.append("\n")

    if return_lines:
        return all_prefs_combinations, prints
    else:
        print("\n".join(prints))
        return all_prefs_combinations


def check_success(successes, configurations, return_lines=False):

    prints = []
    prints.append("\nFINISHED ALL RUNS\n")
    if all(successes):
        prints.append("ALL PASSED")
        success = True
    else:
        prints.append("{}/{} CONFIGURATIONS FAILED:".format(
            len(successes) - sum(successes), len(successes)))
        fails = [not b for b in successes]
        prints.extend(print_prefs_combinations(configurations, condition=fails, return_lines=True))
        success = False

    if return_lines:
        return success, prints
    else:
        print("\n".join(prints))
        return success


def dict_to_name(prefs_dict):
    if prefs_dict is None or not prefs_dict:
        return 'default_preferences'

    names = []
    for k, v in prefs_dict.items():
        try:
            # np.float32/64 has __name__ attribute
            v = v.__name__
        except AttributeError:
            # int and bool don't have __name__ attribute
            pass
        names.append(k.split('.')[-1] + '_' + str(v))

    return '__'.join(names)


class PrintBuffer(object):
    """Collect lines to print in a buffer to print them later all at once."""

    def __init__(self, clusterbot=None):
        """
        Parameters
        ----------
        clusterbot : ClusterBot
            ClusterBot instance that sends messages to Slack. If None
            (default), messages will only be printed to stdout (using print).
        """
        self._lines = []
        self._clusterbot = clusterbot

    def add(self, new_lines):
        """
        Add ``new_lines`` to print buffer and add a timestep. Does not print yet.

        Parameters
        ----------
        new_lines : str or list
            A string or list of strings. Each string is one line to print.
        """
        import time
        timestemp = time.gmtime()
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", timestemp)
        if isinstance(new_lines, basestring):
            new_lines = [new_lines]
        for i in range(len(new_lines)):
            if i == 0:
                # add formatted timestemp
                prefix = formatted
            else:
                # add empty string offset
                prefix = ' ' * len(formatted)
            new_lines[i] = "{}  {}".format(prefix, new_lines[i])
        self._lines.extend(new_lines)

    def print_all(self):
        """
        Print all lines in buffer to stdout (using print) and delete buffer. If
        ``clusterbot`` was given at init, also send the message to Slack.
        """
        message = "\n".join(self._lines)
        # print to stdout
        print(message)
        # post to Slack
        if self._clusterbot is not None:
            try:
                self._clusterbot.send(message)
            except Exception:
                # doesn't catch KeyboardInterrupt or SystemExit
                print("ERROR CLUSTERBOT PRINT:", traceback.format_exc())
                pass
        # delete buffer
        self._lines = []
