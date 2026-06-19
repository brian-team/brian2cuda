"""Codegen + compile smoke test for CI (no GPU required)."""
import argparse
import os
import shutil
import tempfile

from brian2 import *
import brian2cuda


def _build_test_network():
    """Same model as brian2cuda.tests.test_cuda_standalone.test_cuda_standalone."""
    tau = 1 * ms
    eqs = '''
    dV/dt = (-40*mV-V)/tau : volt (unless refractory)
    '''
    threshold = 'V>-50*mV'
    reset = 'V=-60*mV'
    refractory = 5 * ms
    N = 1000

    G = NeuronGroup(
        N, eqs,
        reset=reset,
        threshold=threshold,
        refractory=refractory,
        name='gp',
    )
    G.V = '-i*mV'
    M = SpikeMonitor(G)
    S = Synapses(G, G, 'w : volt', on_pre='V += w')
    S.connect('abs(i-j)<5 and i!=j')
    S.w = 0.5 * mV
    S.delay = '0*ms'

    net = Network(G, M, S)
    net.run(100 * ms)


def test_compile(platform):
    prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.5

    prefs.devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc.extend(
        ['-Xcudafe "--diag_suppress=declared_but_not_referenced"']
    )

    out = tempfile.mkdtemp(prefix="brian2cuda_ci_")
    try:
        set_device("cuda_standalone", build_on_run=False, directory=out)
        _build_test_network()
        device.build(directory=out, compile=True, run=False)

        binary = os.path.join(out, "main.exe" if platform == "windows" else "main")
        if not os.path.isfile(binary):
            raise FileNotFoundError(binary)
        print(f"OK: {platform} codegen + compile")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        default="windows" if os.name == "nt" else "linux",
        choices=["windows", "linux"],
    )
    args = parser.parse_args()
    test_compile(args.platform)
