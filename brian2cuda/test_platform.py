"""Codegen + compile smoke test for CI (no GPU required)."""
import argparse
import os
import shutil
import tempfile

from brian2 import *
import brian2cuda


def test_compile(platform):
    prefs.devices.cuda_standalone.cuda_backend.detect_gpus = False
    prefs.devices.cuda_standalone.cuda_backend.gpu_id = 0
    prefs.devices.cuda_standalone.cuda_backend.compute_capability = 7.5

    out = tempfile.mkdtemp(prefix="brian2cuda_ci_")
    try:
        set_device("cuda_standalone", build_on_run=False, directory=out)
        group = NeuronGroup(
            10,
            "dv/dt = (-60*mV - v) / (20*ms) : volt",
            threshold="v > -50*mV",
            reset="v = -60*mV",
            method="euler",
        )
        group.v = "-60*mV"
        run(1 * ms)
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
