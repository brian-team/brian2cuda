import argparse

parser = argparse.ArgumentParser(description='Run brian2cuda benchmarks')

parser.add_argument('-d', '--results-dir', type=str, default=None,
                    help="Directory where results will be stored")

parser.add_argument('--dry-run', action='store_true',
                    help=("Exit script after argument parsing. Used to check "
                          "validity of arguments"))

parser.add_argument('--no-nvprof', action='store_true',
                    help=("Don't run nvprof profiling"))

parser.add_argument('--no-slack', action='store_true',
                    help=("Don't send notifications to Slack"))

parser.add_argument('-k', '--keep-project-dir', action='store_true',
                    help=("Don't delete project directory"))

parser.add_argument('--profile', action='store_true',
                    help=("Run with codeobject / kernel profiling"))

args = parser.parse_args()

if args.dry_run:
    import sys
    print(f"Dry run completed, {__file__} arguments valid.")
    sys.exit()

if args.results_dir is None:
    raise RuntimeError(
        "Don't run `run_benchmark_suite.py` directly. Use the shell script "
        "`run_benchmark_suite.sh` instead."
    )
else:
    # Remove traling "/" if present
    directory = args.results_dir.rstrip("/")

import os
import shutil
import subprocess
import sys
import traceback

# run tests without X-server
import matplotlib
matplotlib.use('Agg')

import time
import datetime
import socket

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.base import results
from brian2.utils.logger import BrianLogger

from brian2cuda.tests.features.speed import *

import brian2cuda
from brian2cuda.utils.logger import suppress_brian2_logs
from brian2cuda.tests.features.cuda_configuration import DynamicConfigCreator
from brian2cuda.tests.features.cuda_configuration import (CUDAStandaloneConfiguration,
                                                          CUDAStandaloneConfigurationNoAssert,
                                                          CUDAStandaloneConfigurationExtraThresholdKernel,
                                                          CUDAStandaloneConfigurationNoCudaOccupancyAPI,
                                                          CUDAStandaloneConfiguration2BlocksPerSM,
                                                          CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds,
                                                          CUDAStandaloneConfigurationSynLaunchBounds,
                                                          CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds,
                                                          CPPStandaloneConfiguration,
                                                          CPPStandaloneConfigurationSinglePrecision,
                                                          CPPStandaloneConfigurationOpenMPMaxThreads,
                                                          CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision,
                                                          GeNNConfiguration,
                                                          GeNNConfigurationSpanTypePre,
                                                          GeNNConfigurationSinglePrecision,
                                                          GeNNConfigurationSinglePrecisionSpanTypePre)


#from brian2genn.correctness_testing import GeNNConfiguration, GeNNConfigurationCPU, GeNNConfigurationOptimized

from create_readme import create_readme
from helpers import pickle_results, translate_pkl_to_csv


def print_flushed(string, slack=True, new_reply=False, format_code=True):
    print(string, flush=True)
    to_return = None
    if slack and bot is not None:
        assert slack_thread is not None
        if format_code:
            string = f"```{os.linesep}{string}{os.linesep}```"
        try:
            if new_reply:
                new_append_message = bot.reply(slack_thread, string)
                to_return = new_append_message
            else:
                assert append_message is not None
                bot.append(append_message, string)
        except Exception as exc:
            print(f"ERROR: Failed to send message to slack ({exc})", flush=True)
    return to_return

bot = None
if not args.no_slack:
    try:
        from clusterbot import ClusterBot
    except ImportError:
        print("WARNING: clusterbot not installed. Can't notify slack.")
    else:
        print_flushed("Starting ClusterBot...", slack=False)
        try:
            bot = ClusterBot()
        except Exception as exc:
            print(
                f"ERROR: ClusterBot failed to initialize correctly. Can't notify "
                f"slack. Here is the error:\n{exc}"
            )
            bot = None


#suppress_brian2_logs()
# Uncomment this to get brian2cuda logs
# (Set log_level_diagnostic() for DynamicConfigCreator diagnostic messages)
BrianLogger.log_level_debug()


# The configuration classes are defined in
# `brian2cuda/tests/features/cuda_configuration.py`
configurations = [
    # configuration
    #NumpyConfiguration,
    #WeaveConfiguration,
    #LocalConfiguration,
    CPPStandaloneConfiguration,
    #CPPStandaloneConfigurationSinglePrecision,
    CPPStandaloneConfigurationOpenMPMaxThreads,
    #CPPStandaloneConfigurationOpenMPMaxThreadsSinglePrecision,

    # max blocks
    DynamicConfigCreator('CUDA standalone (max blocks, atomics)'),

    #DynamicConfigCreator('CUDA standalone (single precision, max blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32}),


    # 1 block
    DynamicConfigCreator('CUDA standalone (1 block, atomics)',
                         prefs={'devices.cuda_standalone.parallel_blocks': 1}),

    #DynamicConfigCreator('CUDA standalone (single precision, 1 block, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 1}),


    ## 20 blocks
    #DynamicConfigCreator('CUDA standalone (20 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.parallel_blocks': 20}),

    #DynamicConfigCreator('CUDA standalone (single precision, 20 blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 20}),


    ## 40 blocks
    #DynamicConfigCreator('CUDA standalone (40 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.parallel_blocks': 40}),

    #DynamicConfigCreator('CUDA standalone (single precision, 40 blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 40}),


    ## 60 blocks
    #DynamicConfigCreator('CUDA standalone (60 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.parallel_blocks': 60}),

    #DynamicConfigCreator('CUDA standalone (single precision, 60 blocks, atomics)',
    #                     prefs={'core.default_float_dtype': float32,
    #                            'devices.cuda_standalone.parallel_blocks': 60}),


    #DynamicConfigCreator('CUDA standalone (max blocks, no atomics)',
    #                     prefs={'devices.cuda_standalone.use_atomics': False}),

    #DynamicConfigCreator('CUDA standalone (1 block, no atomics)',
    #                     prefs={'devices.cuda_standalone.use_atomics': False,
    #                            'devices.cuda_standalone.parallel_blocks': 1}),

    #DynamicConfigCreator('CUDA standalone (no bundles, 15 blocks, atomics)',
    #                     prefs={'devices.cuda_standalone.push_synapse_bundles': False}),


    #DynamicConfigCreator('CUDA standalone (master)',
    #                     git_commit='master'),

    #DynamicConfigCreator("CUDA standalone (only compilation)",
    #                     set_device_kwargs={'compile': True, 'run': False}),

    #CUDAStandaloneConfigurationExtraThresholdKernel,
    #CUDAStandaloneConfigurationNoAssert,
    #CUDAStandaloneConfigurationNoCudaOccupancyAPI,
    #CUDAStandaloneConfiguration2BlocksPerSM,
    #CUDAStandaloneConfiguration2BlocksPerSMLaunchBounds,
    #CUDAStandaloneConfigurationSynLaunchBounds,
    #CUDAStandaloneConfiguration2BlocksPerSMSynLaunchBounds,
    #GeNNConfigurationCPU,
    GeNNConfiguration,
    #GeNNConfigurationSinglePrecision,
    GeNNConfigurationSpanTypePre,
    #GeNNConfigurationSinglePrecisionSpanTypePre,
]

# The `benchmark` classes are defined in brian2cuda/tests/features/speed.py. The
# `n_slice` parameter indexes the `n_range` class attribute of the respective benchmark
# class to determine the network sizes for which this benchmark should be run.
speed_tests = [# benchmark                                                  n_slice

               # paper benchmarks
               (COBAHHUncoupled,                                            slice(None)),
               (COBAHHPseudocoupled1000,                                    slice(None)),
               (BrunelHakimHomogDelays,                                     slice(None)),
               (BrunelHakimHeterogDelays,                                   slice(None)),
               (STDPCUDARandomConnectivityHomogeneousDelays,                slice(None)),
               (STDPCUDARandomConnectivityHeterogeneousDelays,              slice(None)),
               (MushroomBody,                                               slice(None)),

               ## other benchmarks
               #(COBAHHPseudocoupled80,                                      slice(None)),
               #(BrunelHakimHeterogDelaysNarrowDistr,                        slice(None)),
               #(STDPCUDA,                                                   slice(None)),
               #(STDPCUDAHomogeneousDelays,                                  slice(None)),
               #(STDPCUDAHeterogeneousDelays,                                slice(None)),
               #(STDPCUDARandomConnectivityHeterogeneousDelaysNarrowDistr    slice(None),

               #(CUBAFixedConnectivityNoMonitor, slice(None)),
               #(COBAHHCoupled, slice(None)),
               #(STDPEventDriven, slice(None)),

               #(VerySparseMediumRateSynapsesOnly, slice(None)),
               #(SparseMediumRateSynapsesOnly, slice(None)),
               #(DenseMediumRateSynapsesOnly, slice(None)),
               #(SparseLowRateSynapsesOnly, slice(None)),
               #(SparseHighRateSynapsesOnly, slice(None)),

               #(DenseMediumRateSynapsesOnlyHeterogeneousDelays, slice(None)),
               #(SparseLowRateSynapsesOnlyHeterogeneousDelays, slice(None)),

               #(LinearNeuronsOnly, slice(None)),
               #(HHNeuronsOnly, slice(None)),

               ### below uses monitors
               #(CUBAFixedConnectivity, slice(None)),
               #(COBAHHFixedConnectivity, slice(None, -1)),
]

slack_thread = None
append_message = None

script = os.path.basename(__file__)
host = socket.gethostname()

start_msg = f"Running `{script}` on `{host}`"
if bot is not None:
    try:
        bot.init_pbar(max_value=len(speed_tests), title=start_msg)
        slack_thread = bot.pbar_id
    except Exception as exc:
        print_flushed(f"Failed to init slack notifications, no slack messages will be sent\n{exc}")
        bot = None

print_flushed(start_msg, slack=False)

if args.profile:
    print_flushed("Profiling ON")

msg = "ENVIRONMENT:"
for key, value in os.environ.items():
    if not key.startswith("LESS"):
        msg += f"\n\t{key} = {value}"
print_flushed(msg, slack=False)

try:
    cpuinfo = subprocess.check_output(["lscpu"], encoding='UTF-8')
except subprocess.CalledProcessError as err:
    cpuinfo = err
print_flushed(f"\nCPU information\n{cpuinfo}", slack=False)

try:
    gpuinfo = subprocess.check_output(["deviceQuery"], encoding='UTF-8')
except (subprocess.CalledProcessError, FileNotFoundError) as err:
    gpuinfo = err
print_flushed(f"\nGPU information\n{gpuinfo}", slack=False)


gxx_binary = os.getenv("CXX", None)
if gxx_binary is None:
    gxx_binary = 'gcc'
try:
    gxxinfo = subprocess.check_output([gxx_binary, "--version"], encoding='UTF-8')
except subprocess.CalledProcessError as err:
    gxxinfo = err
print_flushed(f"\nCXX version\n{gxxinfo}", slack=False)

try:
    nvccinfo = subprocess.check_output(["nvcc", "--version"], encoding='UTF-8')
except subprocess.CalledProcessError as err:
    nvccinfo = err
print_flushed(f"\nNVCC version\n{nvccinfo}", slack=False)

# pretty plots
try:
    import seaborn as sns
    sns.set_palette(sns.color_palette("hls", len(configurations)))
    #sns.set_palette(sns.color_palette("cubehelix", len(configurations)))
except Exception:
    pass



time_stemp = time.time()
date_str = datetime.datetime.fromtimestamp(time_stemp).strftime('%Y-%m-%d_%T')

if not os.path.exists(directory):
    os.makedirs(directory)

data_dir = os.path.join(directory, 'data')
plot_dir = os.path.join(directory, 'plots')
log_dir = os.path.join(directory, 'logs')
prof_dir = os.path.join(directory, 'nvprof')
os.makedirs(data_dir)
os.makedirs(plot_dir)
os.makedirs(log_dir)
os.makedirs(prof_dir)
print_flushed(f"Saving results in {plot_dir}.")

shutil.copy(os.path.realpath(__file__), os.path.join(directory, 'run_benchmark_suite.py'))

time_format = '%d.%m.%Y at %H:%M:%S'
script_start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)

with open(os.path.join(directory, 'git.diff'), 'w') as diff_file:
    subprocess.call(['git', 'diff'], stdout=diff_file)

try:
    for i, (speed_test, n_slice) in enumerate(speed_tests):
        test_name = speed_test.__name__
        start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        append_message = print_flushed(f"Starting {test_name} on {start}.", new_reply=True)
        maximum_run_time = 1000*second
        res = run_speed_tests(configurations=configurations,
                              speed_tests=[speed_test],
                              n_slice=n_slice,
                              #n_slice=slice(0,1,None),
                              run_twice=False,
                              verbose=True,
                              maximum_run_time=maximum_run_time,
                              mark_not_completed=True,
                              profile=args.profile,
                             )
        end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
        diff = datetime.datetime.strptime(end, time_format) - datetime.datetime.strptime(start, time_format)
        print_flushed(f"Running {test_name} took {diff}.")
        res.plot_all_tests()
        ## this needs modification of brian2 code
        #res.plot_all_tests(print_relative=True)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_absolute.png'))
        res.plot_all_tests(relative=True)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_relative.png'))
        res.plot_all_tests(profiling_minimum=0.05)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_profiling.png'))
        res.plot_all_tests(only="python_", exclude=(), profiling_minimum=0)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_python_timers.png'))
        res.plot_all_tests(only="standalone_", exclude=(), profiling_minimum=0)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_standalone_timers.png'))
        res.plot_all_tests(only="standalone_",
                           exclude=("standalone_before_", "standalone_after_"),
                           profiling_minimum=0)
        savefig(os.path.join(plot_dir, f'speed_test_{test_name}_standalone_timers_without_extra_diffs.png'))
        if 6 != len(get_fignums()):
            print_flushed(f"WARNING: There were {len(get_fignums())} plots created, but only {6 * (i + 1)} saved.", slack=False)
        for f in get_fignums():
            close(f)

        # pickel results object and csv file to disk
        pkl_file = os.path.join(data_dir, test_name + '.pkl' )
        pickle_results(res, pkl_file)
        try:
            translate_pkl_to_csv(pkl_file)
        except KeyError as e:
            print_flushed(f"ERROR tranlating {pkl_file} to csv:\n\tKeyError: {e}")

        try:
            for key in res.brian_stdouts.keys():
                config_key, st_key, n_key = key
                suffix = f'{st_key}_{config_key}_{n_key}.txt'
                stdout_file = os.path.join(log_dir, f'stdout_{suffix}')
                with open(stdout_file, 'w+') as sfile:
                    sfile.write(res.brian_stdouts[key])
                print_flushed(f"Written {stdout_file}", slack=False)
                stderr_file = os.path.join(log_dir, f'stderr_{suffix}')
                with open(stderr_file, 'w+') as sfile:
                    sfile.write(res.brian_stderrs[key])
                print_flushed(f"Written {stderr_file}", slack=False)
                tb_file = os.path.join(log_dir, f'tb_{suffix}')
                with open(tb_file, 'w+') as tfile:
                    tfile.write(res.tracebacks[key])
                print_flushed(f"Written {tb_file}", slack=False)
        except Exception as err:
            print_flushed(f"ERROR writing stdout files: {err}", slack=False)
        except:
            print_flushed("ERROR writing stdout files and couldn't catch Exception...", slack=False)

        # collect stdout logs
        for conf in configurations:
            for n in speed_test.n_range[n_slice]:
                proj_dir = conf.get_project_dir(feature_name=test_name, n=n)
                for stream in ['stdout', 'stderr']:
                    file = os.path.join(proj_dir, 'results', f'{stream}.txt')
                    if os.path.exists(file):
                        shutil.copy(
                            file,
                            os.path.join(
                                log_dir, os.path.basename(
                                    f'{stream}_{test_name}_{proj_dir}_{n}.txt'
                                )
                            )
                        )
                    else:
                        print_flushed(f"WARNING Couldn't save {file}, file not found.", slack=False)

        # run nvprof on n_range[2] for all configurations
        if not args.no_nvprof:
            for conf in configurations:
                #for n in speed_test.n_range[n_slice]:
                ns = speed_test.n_range[n_slice]
                idx = 4 if len(ns) > 4 else len(ns) - 1
                nvprof_n = speed_test.n_range[idx]
                max_runtime = 1000
                proj_dir = conf.get_project_dir(feature_name=test_name, n=nvprof_n)
                main_args = ''
                if 'cpp_standalone_runs' in proj_dir:
                    break
                elif 'genn_runs' in proj_dir:
                    main_args = f'test {speed_test.duration / second}'
                if isinstance(conf, DynamicConfigCreator):
                    conf_name = conf.name.replace(' ', '-').replace('(', '-').replace(')', '-').replace(',', '-')
                else:
                    conf_name = conf.__name__
                print_flushed(f"Rerunning {conf_name} with n = {nvprof_n} for nvprof profiling")
                #tb, res, runtime, prof_info = results(conf, speed_test, speed_test.n_range[idx], maximum_run_time=maximum_run_time)
                runtime = res.full_results[conf.name, speed_test.fullname(), n, 'All']
                this_res = res.feature_results[conf.name, speed_test.fullname(), n]
                tb = res.tracebacks[conf.name, speed_test.fullname(), n]
                fullname = conf.get_fullname(test_name, nvprof_n)
                if not isinstance(this_res, Exception) and runtime < max_runtime:
                    nvprof_args = '--profile-from-start-off' if proj_dir.startswith("cuda_standalone") else ''
                    log_file=os.path.join(
                        os.path.abspath(prof_dir),
                        f'nvprof_{fullname}.log'
                    )
                    cmd = (
                        f'cd {proj_dir}'
                        f' && nvprof'
                            f' {nvprof_args}'
                            f' --log-file {log_file}'
                            f' ./main {main_args}'
                    )
                    prof_start = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                    cmd = cmd.replace('(', '\(').replace(')', '\)')
                    print_flushed(f"Running nvprof: {cmd}", slack=False)
                    try:
                        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
                    except subprocess.CalledProcessError as err:
                        print_flushed(f"ERROR: nvprof failed with: {err} output: {err.output}")
                    prof_end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
                    prof_diff = datetime.datetime.strptime(prof_end, time_format) - datetime.datetime.strptime(prof_start, time_format)
                    print_flushed(f"Profiling took {prof_diff} for runtime of {runtime}")
                elif isinstance(res, Exception):
                    print_flushed("Didn't run nvprof, got an Exception", res)
                else:  # runtime >= max_runtime
                    print_flushed(f"Didn't run nvprof, runtime ({runtime}) >= max_runtime ({max_runtime})")

        # Delete project directories when done
        if not args.keep_project_dir:
            for conf in configurations:
                for n in speed_test.n_range[n_slice]:
                    conf.delete_project_dir(feature_name=test_name, n=n)

        if bot is not None:
            try:
                bot.update_pbar()
            except:
                print_flushed("ERROR: Failed to update slack progress bar.", flush=True, slack=False)

except KeyboardInterrupt:
    print_flushed(f"\nCaught KeyboardInterrupt: Exiting...")
    final_msg = "❌ INTERRUPTED"
except Exception as exc:
    traceback.print_exc()
    print_flushed(f"\nCaught an exception: {exc}\nExiting...")
    final_msg = "❌ FAILED"
else:
    final_msg = "✅ FINISHED"
finally:
    create_readme(directory)
    append_message = print_flushed(
        f"\nSummarized speed test results in {os.path.join(directory, 'README.md')}",
        new_reply=True
    )
    script_end = datetime.datetime.fromtimestamp(time.time()).strftime(time_format)
    script_diff = datetime.datetime.strptime(script_end, time_format) - datetime.datetime.strptime(script_start, time_format)
    print_flushed("Finished speed test on {}. Total time = {}.".format(
        datetime.datetime.fromtimestamp(time.time()).strftime(time_format), script_diff))
    final_msg += f" `{script}` on `{host}` after {script_diff}"
    if bot is not None:
        try:
            bot.update(slack_thread, final_msg)
        except:
            print("ERROR: Failed to update final slack message.", flush=True)
    print_flushed(final_msg, slack=False)


##res.plot_all_tests(relative=True)
#for n in get_fignums():
#    plt.figure(n)
#    savefig(plot_dir + '/speed_test_{}.png'.format(speed_tests[n-1][1]))

## Debug (includes profiling infos)
#from brian2.tests.features.base import results
#for x in results(LocalConfiguration, LinearNeuronsOnly, 10, maximum_run_time=10*second):
#    print x
