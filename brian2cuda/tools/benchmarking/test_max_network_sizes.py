# run tests without X-server
import matplotlib
matplotlib.use('Agg')

import time
import datetime
import os
import socket

from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.base import results

import brian2cuda
from brian2cuda.tests.features.cuda_configuration import CUDAStandaloneConfiguration
from brian2cuda.tests.features.speed import *


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
        except:
            print("ERROR: Failed to send message to slack.", flush=True)
    return to_return


bot = None
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


speed_tests = [# feature_test                     name                                  n_slice

               # paper benchmarks
               (COBAHHUncoupled,                        'COBAHHUncoupled',                      slice(None)),
               (COBAHHPseudocoupled1000,                'COBAHHPseudocoupled1000',              slice(None)),
               (COBAHHPseudocoupled80,                  'COBAHHPseudocoupled80',                slice(None)),
               (BrunelHakimHomogDelays,                 'BrunelHakimHomogDelays',               slice(None)),
               (BrunelHakimHeterogDelays,               'BrunelHakimHeterogDelays',             slice(None)),
               (BrunelHakimHeterogDelaysNarrowDistr,    'BrunelHakimHeterogDelaysNarrowDistr',  slice(None)),
               (STDPCUDA,                               'STDPCUDA',                             slice(None)),
               (STDPCUDAHomogeneousDelays,              'STDPCUDAHomogeneousDelays',            slice(None)),
               (STDPCUDAHeterogeneousDelays,            'STDPCUDAHeterogeneousDelays',          slice(None)),
               (MushroomBody,                           'MushroomBody',                         slice(None)),


               #(CUBAFixedConnectivityNoMonitor,                  'CUBAFixedConnectivityNoMonitor',              slice(None)         ),
               #(COBAHHConstantConnectionProbability,             'COBAHHConstantConnectionProbability',         slice(None)         ),
               #(COBAHHFixedConnectivityNoMonitor,                'COBAHHFixedConnectivityNoMonitor',            slice(None)         ),
               #(AdaptationOscillation,                          'AdaptationOscillation',                        slice(None, -1)     ),
               #(Vogels,                                         'Vogels',                                       slice(None)         ),
               #(STDPEventDriven,                                'STDPEventDriven',                              slice(None)         ),
               #(BrunelHakimModelScalarDelay,                    'BrunelHakimModelScalarDelay',                  slice(None)         ),

               #(VerySparseMediumRateSynapsesOnly,               'VerySparseMediumRateSynapsesOnly',             slice(None)         ),
               #(SparseMediumRateSynapsesOnly,                   'SparseMediumRateSynapsesOnly',                 slice(None)         ),
               #(DenseMediumRateSynapsesOnly,                    'DenseMediumRateSynapsesOnly',                  slice(None)         ),
               #(SparseLowRateSynapsesOnly,                      'SparseLowRateSynapsesOnly',                    slice(None)         ),
               #(SparseHighRateSynapsesOnly,                     'SparseHighRateSynapsesOnly',                   slice(None)         ),

               #(STDPNotEventDriven,                             'STDPNotEventDriven',                           slice(None)         ),

               #(DenseMediumRateSynapsesOnlyHeterogeneousDelays, 'DenseMediumRateSynapsesOnlyHeterogeneousDelays', slice(None)       ),
               #(SparseLowRateSynapsesOnlyHeterogeneousDelays,   'SparseLowRateSynapsesOnlyHeterogeneousDelays', slice(None)         ),
               #(BrunelHakimModelHeterogeneousDelay,             'BrunelHakimModelHeterogeneousDelay',           slice(None)         ),

               #(LinearNeuronsOnly,                              'LinearNeuronsOnly',                            slice(None)         ),
               #(HHNeuronsOnly,                                  'HHNeuronsOnly',                                slice(None)         ),
               #(VogelsWithSynapticDynamic,                      'VogelsWithSynapticDynamic',                    slice(None)         ),

               #### below uses monitors
               #(CUBAFixedConnectivity,                          'CUBAFixedConnectivity',                        slice(None)         ),
               #(COBAHHFixedConnectivity,                        'COBAHHFixedConnectivity',                      slice(None)         ),
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

config = CUDAStandaloneConfiguration

total_start = time.time()
time_format = '%d.%m.%Y at %H:%M:%S'
try:
    for (ft, name, sl) in speed_tests:
        codes = []
        know_last_fail = False
        break_after_first = False
        if isinstance(sl, slice):
            # Example: ft.n_range = [10, 100, 1000, 10000]
            length = len(ft.n_range)
            # Example: with sl = slice(None) -> sl_idcs = (0, 4, 1)
            sl_idcs = sl.indices(length)
            # Example: max_idx = 4 - 1 = 3 (last index in n_range)
            max_idx = sl_idcs[1] - sl_idcs[2]
            if host == "gpu055":  # A100 GPU
                n_init = ft.n_range[-1] * 2
            else:
                # Example: n_init = 10000
                n_init = ft.n_range[-1]
        else:
            if isinstance(sl, int):
                sl = [sl]

            if len(sl) == 1:
                # Only test one n
                n_init = sl[0]
                break_after_first = True
            else:
                # sl is (last_success, last_fail)
                assert len(sl) == 2
                n_last_success = sl[0]
                n_last_fail = sl[1]
                n_init = int(n_last_success + (n_last_fail - n_last_success) / 2)
                know_last_fail = True
        n = n_prev = n_init
        #n = ft.n_range[0]
        has_failed = False
        for i in range(3):
            if name.startswith('STDPCUDA'):
                n = (n//1000) * 1000
            start = time.time()
            script_start = datetime.datetime.fromtimestamp(start).strftime(time_format)
            append_message = print_flushed(
                f"LOG RUNNING {name} with n={n} (start at {script_start})",
                new_reply=True
            )
            tb, res, runtime, prof_info = results(config, ft, n)
            took = datetime.timedelta(seconds=int(time.time() - start))
            print_flushed(f"LOG FINISHED {name} with n={n} in {took}")
            codes.append((n, res, tb))
            diff = np.abs(n - n_prev)
            n_prev = n
            if not know_last_fail:
                if isinstance(res, Exception):
                    print_flushed(f"LOG FAILED:\n{res}, {tb}")
                    if i == 0:
                        print_flushed(f"LOG FAILED at first run ({name}) n={n}\n\terror={res}\nt\ttb={tb}")
                        #print_flushed(f"LOG FAILED at first run ({name}) n={n}\n\terror={res}\nTraceback printed above.")
                        config.delete_project_dir(feature_name=ft.__name__, n=n_prev)
                        break
                    # assuming the first run passes, when we fail we always go down half the abs distance
                    n -= int(0.5 * diff)
                    print_flushed("LOG HALF DOWN")
                elif has_failed:
                    n += int(0.5 * diff)
                    print_flushed("LOG HALF UP")
                else:  # not has_failed
                    n *= 2
                    print_flushed("LOG DOUBLE")
            else:  # know_last_fail = True
                # we have (last_success, last_fail) numbers, choose always half between last success and last fail
                if isinstance(res, Exception):
                    print_flushed(f"LOG FAILED:\n{res}, {tb}")
                    if n < n_last_fail:
                        n_last_fail = n
                    print_flushed("LOG HALF DOWN TO LAST SUCCESS")
                else:  # passed
                    if n > n_last_success:
                        n_last_success = n
                    print_flushed("LOG HALF UP TO NEXT FAIL")
                n = int(n_last_success + (n_last_fail - n_last_success) / 2)

            if break_after_first or (i != 0 and diff <= 10000):
                print_flushed(f"LOG BREAK {name} after n={n} (diff={diff})")
                config.delete_project_dir(feature_name=ft.__name__, n=n_prev)
                break

            config.delete_project_dir(feature_name=ft.__name__, n=n_prev)

        print_flushed(f"LOG FINAL RESULTS FOR {name}:")
        for n, res, tb in codes:
            if isinstance(res, Exception):
                sym = 'ERROR'
            else:
                sym = 'PASS'
            print_flushed(f"\t{n}\t\t{sym}")

        if bot is not None:
            try:
                bot.update_pbar()
            except:
                print("ERROR: Failed to update slack progress bar.", flush=True)

except KeyboardInterrupt:
    print_flushed("\nCaught KeyboardInterrupt! Exiting...", slack=False)
    config.delete_project_dir(feature_name=ft.__name__, n=n_prev)
    final_msg = "❌ INTERRUPTED"
else:
    final_msg = "✅ FINISHED"

total_time = datetime.timedelta(seconds=int(time.time() - total_start))
final_msg += f" `{script}` on `{host}` after {total_time}"
if bot is not None:
    try:
        bot.update(slack_thread, final_msg)
    except:
        print("ERROR: Failed to update final slack message.", flush=True)
print_flushed(final_msg, slack=False)
