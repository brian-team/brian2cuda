"""
Tools to get information about available GPUs.
"""

import os
import subprocess
import shutil
import shlex
import re
import distutils

from brian2.core.preferences import prefs, PreferenceError
from brian2.codegen.cpp_prefs import get_compiler_and_args
from brian2.utils.logger import get_logger

logger = get_logger("brian2.devices.cuda_standalone")

# To list all GPUs: nvidia-smi -L

# Some code here is adapted from
# https://github.com/cupy/cupy/blob/e6f8d91ffae7ee241ed235ddbeb725c04f593c33/cupy/_environment.py


# To minimize running external commands (`nvidia-smi`, `nvcc`, `deviceQuery`), we define
# these global variables that are computed from the external commands. This way we only
# run them once and whenever they are needed again, we use the global variables defined
# here.
_cuda_installation = {
    "cuda_path": None,
    "nvcc_path": None,
    "runtime_version": None,
}

_gpu_selection = {
    "available_gpus": None,
    "selected_gpu_id": None,
    "selected_gpu_compute_capability": None,
}


def get_cuda_path():
    """
    Detect the path to the CUDA installation (e.g. '/usr/local/cuda'). This takes into
    account user defined environmental variable `CUDA_PATH` and preference
    `prefs.brian2cuda.cuda_path`.
    """
    # If cuda_path was already detected, reuse the global variable
    global _cuda_installation
    if _cuda_installation["cuda_path"] is None:
        _cuda_installation["cuda_path"] = _get_cuda_path()
    return _cuda_installation["cuda_path"]


def get_nvcc_path():
    """Return the path to the `nvcc` binary."""
    # If nvcc_path was already detected, reuse the global variable
    global _cuda_installation
    if _cuda_installation["nvcc_path"] is None:
        _cuda_installation["nvcc_path"] = _get_nvcc_path()
    return _cuda_installation["nvcc_path"]


def get_cuda_runtime_version():
    """Return CUDA runtime version (as float, e.g. `11.2`)"""
    # If runtime_version was already detected, reuse the global variable
    global _cuda_installation
    if _cuda_installation["runtime_version"] is None:
        _cuda_installation["runtime_version"] = _get_cuda_runtime_version()
    return _cuda_installation["runtime_version"]


def get_cuda_installation():
    """Return new dictionary of cuda installation variables"""
    cuda_installation = {
        'cuda_path': get_cuda_path(),
        'nvcc_path': get_nvcc_path(),
        'runtime_version': get_cuda_runtime_version(),
    }
    global _cuda_installation
    assert (
        sorted(cuda_installation.keys()) == sorted(_cuda_installation.keys())
    ), "{} != {}".format(cuda_installation.keys(), _cuda_installation.keys())
    return cuda_installation


def get_gpu_selection():
    """Return dictionary of selected gpu variable"""
    gpu_id, compute_capability = select_gpu()
    gpu_selection = {
        'available_gpus': get_available_gpus(),
        'selected_gpu_id': gpu_id,
        'selected_gpu_compute_capability': compute_capability,
    }
    global _gpu_selection
    assert gpu_selection.keys() == _gpu_selection.keys()
    return gpu_selection


def get_available_gpus():
    """
    Return list of names of available GPUs, sorted by GPU ID as reported in
    `nvidia-smi`
    """
    global _gpu_selection
    if _gpu_selection["available_gpus"] is None:
        _gpu_selection["available_gpus"] = _get_available_gpus()
    return _gpu_selection["available_gpus"]


def select_gpu():
    """
    Select GPU for simulation, based on user preference `prefs.brian2cuda.gpu_id` or (if
    not provided) pick the GPU with highest compute capability. Returns tuple of
    (gpu_id, compute_capability) of type (int, float).
    """
    global _gpu_selection
    if _gpu_selection["selected_gpu_id"] is None:
        assert _gpu_selection["selected_gpu_compute_capability"] is None
        gpu_id, compute_capability = _select_gpu()
        _gpu_selection["selected_gpu_id"] = gpu_id
        _gpu_selection["selected_gpu_compute_capability"] = compute_capability
    return (
        _gpu_selection["selected_gpu_id"],
        _gpu_selection["selected_gpu_compute_capability"]
    )


def reset_cuda_installation():
    """
    Reset detected CUDA installation. This will detect the CUDA installation again when
    it is needed.
    """
    global _cuda_installation
    for key in _cuda_installation.keys():
        _cuda_installation[key] = None


def reset_gpu_selection():
    """Reset selected GPU. This will select a new GPU the next time it is needed."""
    global _gpu_selection
    for key in _gpu_selection.keys():
        _gpu_selection[key] = None


def restore_cuda_installation(cuda_installation):
    """Set global cuda installation dictionary to `cuda_installation`"""
    global _cuda_installation
    if _cuda_installation.keys() != cuda_installation.keys():
        raise KeyError(
            "`cuda_installation` has to have the following keys: {}. Got instead: "
            "{}".format(cuda_installation.keys(), _cuda_installation.keys())
        )
    _cuda_installation.update(cuda_installation)


def restore_gpu_selection(gpu_selection):
    """Set global gpu selection dictionary to `gpu_selection`"""
    global _gpu_selection
    if _gpu_selection.keys() != gpu_selection.keys():
        raise KeyError(
            "`gpu_selection` has to have the following keys: {}. Got instead: "
            "{}".format(gpu_selection.keys(), _gpu_selection.keys())
        )
    _gpu_selection.update(gpu_selection)


def _get_cuda_path():
    # Use preference if set
    cuda_path_pref = prefs.brian2cuda.cuda_path
    if cuda_path_pref is not None:
        logger.info(
            "CUDA installation directory given via preference "
            "`prefs.brian2cuda.cuda_path={}`".format(cuda_path_pref)
        )
        return cuda_path_pref

    # Use environment variable if set
    cuda_path = os.environ.get("CUDA_PATH", "")  # Nvidia default on Windows
    if os.path.exists(cuda_path):
        logger.info(
            "CUDA installation directory given via environment variable `CUDA_PATH={}`"
            "".format(cuda_path)
        )
        return cuda_path

    # Use nvcc path if `nvcc` binary in PATH
    # TODO: Remove this and use shutil.which once we moved to Python 3
    def which(pgm):
        path = os.getenv("PATH")
        for p in path.split(os.path.pathsep):
            p = os.path.join(p, pgm)
            if os.path.exists(p) and os.access(p, os.X_OK):
                return p
    nvcc_path = which("nvcc")
    import sys
    assert not sys.version.startswith("3"), "Update code here for Python 3!"
    #nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        cuda_path_nvcc = os.path.dirname(os.path.dirname(nvcc_path))
        logger.info(
            "CUDA installation directory detected via location of `nvcc` binary: {}"
            "".format(cuda_path_nvcc)
        )
        return cuda_path_nvcc

    # Use typical path if nothing else worked
    if os.path.exists("/usr/local/cuda"):
        cuda_path_usr = "/usr/local/cuda"
        logger.info(
            "CUDA installation directory found in standard location: {}"
            "".format(cuda_path_usr)
        )
        return cuda_path_usr

    # Raise error if cuda path not found
    raise RuntimeError(
        "Couldn't find the CUDA installation. Please set the environment variable "
        "`CUDA_PATH` to point to your CUDA installation directory (this should be the "
        "directory, where `./bin/nvcc` is located, e.g. `/usr/local/cuda`)"
    )


def _get_nvcc_path():
    # TODO: Check if NVCC is specific to cupy and if we want to support it?
    # If so, make sure cuda_path and nvcc_path fit together, see:
    # https://github.com/cupy/cupy/blob/cb29c07ccbae346841adb3c8bfa33aba463e2588/install/build.py#L65-L70
    #nvcc = os.environ.get("NVCC", None)
    #if nvcc:
    #    return distutils.util.split_quoted(nvcc)

    cuda_path = get_cuda_path()

    compiler, _ = get_compiler_and_args()
    if compiler == "msvc":  # Windows
        nvcc_bin = "bin/nvcc.exe"
    else:  # Unix
        nvcc_bin = "bin/nvcc"

    nvcc_path = os.path.join(cuda_path, nvcc_bin)
    if not os.path.exists(nvcc_path):
        raise RuntimeError("Couldn't find `nvcc` binary in {}.".format(nvcc_path))

    return nvcc_path


def _get_cuda_runtime_version():
    """ Get CUDA runtime version form `nvcc --version` """
    nvcc_path = get_nvcc_path()
    nvcc_output = _run_command_with_output(nvcc_path, "--version")
    nvcc_lines = nvcc_output.split("\n")
    # version_line example: "Cuda compilation tools, release 11.2, V11.2.67"
    version_line = nvcc_lines[3]
    assert version_line.startswith("Cuda compilation tools, release")
    # release_str example: "release 11.2"
    release_str = version_line.split(", ")[1]
    # runtime_version example: 11.2
    runtime_version_str = release_str.split(" ")[1]
    # return version as float
    return float(runtime_version_str)


def _select_gpu():
    gpu_id = prefs.brian2cuda.gpu_id
    compute_capability = prefs.codegen.generators.cuda.compute_capability
    gpu_list = None
    if prefs.brian2cuda.detect_gpus:
        if gpu_id is None:
            gpu_id, compute_capability = get_best_gpu()
        else:
            compute_capability = get_compute_capability(gpu_id)
        gpu_list = get_available_gpus()
    else:
        logger.info(
            "Automatic detection of GPU names and compute capabilities disabled, using "
            "manual preferences"
        )
        if gpu_id is None or compute_capability is None:
            raise PreferenceError(
                "Got `prefs.brian2cuda.detect_gpus` == `False`. Without GPU detection, "
                "you need to set `prefs.brian2cuda.gpu_id` and "
                "`prefs.codegen.generators.cuda.compute_capability` (got "
                "`{prefs.brian2cuda.gpu_id}` and "
                "`{prefs.codegen.generators.cuda.compute_capability}`).".format(
                    prefs=prefs
                )
            )

    gpu_name = ""
    if gpu_list is not None:
        gpu_name = " ({})".format(gpu_list[gpu_id])

    logger.info(
        "Compiling device code for GPU {gpu_id}{gpu_name}".format(
            gpu_id=gpu_id, gpu_name=gpu_name
        )
    )

    return gpu_id, compute_capability


def _run_command_with_output(command, *args):
    """
    Return the stdout from `command` run in a subprocess and produce meaningful error
    message if it fails. If `args` is empty, `command` can be a string with multiple
    arguments (e.g. `ls -l -a`). If `args` are given, `command` has to be just the
    binary (e.g. `ls`) and each `args` item needs to be a single argument.

    Examples
    --------
    >>> _run_command_with_output("ls -a -l")
    >>> _run_command_with_output("ls", "-a", "-l")
    """
    if not args:
        command_split = shlex.split(command)
    else:
        command_split = [command] + list(args)

    try:
        output = subprocess.check_output(command_split)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            "Running `{binary}` failed with error code {err.returncode}: {err.output}"
            "".format(binary=command_split[0], err=err)
        )
    # TODO: In Python 3 this needs to be FileNotFoundError
    except OSError as err:
        raise OSError(
            "Binary not found: `{binary}` ({err})".format(
                binary=command_split[0], err=err
            )
        )

    return output


def _get_available_gpus():
    """
    Detect available GPUs and return a list of their names, where list index corresponds
    to GPU id.
    """
    command = "nvidia-smi -L"
    try:
        gpu_info_lines = _run_command_with_output(command).split("\n")
    # TODO: In Python 3 version, replace OSError with FileNotFoundError
    except (RuntimeError, OSError) as excepted_error:
        new_error = RuntimeError(
            "Running `{command}` failed. If `nvidia-smi` is not available in your "
            "system, you can disable automatic detection of GPU name and compute "
            "capability by setting "
            "`prefs.devices.brian2cuda.detect_gpus` = `False`".format(
                command=command
            )
        )
        # TODO: In Python 3, do this:
        # raise new_error from excepted_error
        raise new_error

    gpu_list = []
    if gpu_info_lines is not None:
        for i, gpu_info in enumerate(gpu_info_lines):
            if gpu_info == "":  # last list item is empty
                continue

            # `gpu_info` example:
            # "GPU 0: GeForce MX150 (UUID: GPU-8abe566f-c211-11c1-7b73-8103bfd30198)"

            # Remove the UUID part
            gpu_info = gpu_info.split(" (UUID")[0]
            # Split ID and NAME parts
            id_str, gpu_name = gpu_info.split(": ")
            assert id_str.startswith("GPU ")
            gpu_id = id_str[4]
            assert int(gpu_id) == i
            gpu_list.append(gpu_name)
    return gpu_list


def get_compute_capability(gpu_id):
    """
    Get compute capability of GPU with ID `gpu_id`. Returns a float (e.g. `6.1`).
    """
    gpu_list = get_available_gpus()
    cuda_path = get_cuda_path()
    device_query_path = os.path.join(cuda_path, "extras", "demo_suite", "deviceQuery")
    if not os.path.exists(device_query_path):
        # Note: If `deviceQuery` is not reliably available of user systems, we could
        # 1. use this github gist to scrape compute capabilities for GPU names from the
        #    nvidia website:
        #    https://gist.github.com/huitseeker/b2c79e5b763d58b06b9985de2b3c0d4d
        # 2. add a preference to point to the self-compiled binary?
        raise RuntimeError(
            "Couldn't find `{}` binary to detect the compute capability of "
            "the GPU. Please open an issue at "
            "https://github.com/brian-team/brian2cuda/issues/new. To continue, you can "
            "set the compute capability manually via "
            "`prefs.codegen.generators.cuda.compute_capability` (visit "
            "https://developer.nvidia.com/cuda-gpus to find the compute capability of "
            "your GPU).".format(device_query_path)
        )
    device_query_output = _run_command_with_output(device_query_path)
    lines = device_query_output.split("\n")
    compute_capability = None
    for i, line in enumerate(lines):
        if line.startswith("Device "):
            # example line:
            # `Device 0: "GeForce MX150"`
            this_gpu_id = int(line[7])  # "Device i ..."  <- i in position 7
            if this_gpu_id == gpu_id:
                # Get GPU name: word in quotation
                gpu_name = re.findall(r'\"(.+?)\"', line)[0]
                # Make sure we got the right GPU here
                assert gpu_list[gpu_id] == gpu_name
                # The compute capability is shown 2 lines after the "Device ..." line
                # Example line:
                # `  CUDA Capability Major/Minor version number:    6.1`
                compute_capability_line = lines[i + 2]
                assert compute_capability_line.strip().startswith(
                    "CUDA Capability Major/Minor version number"
                ), "Unexpected line parsed: {}".format(compute_capability_line)
                # Last 3 chars are the compute capability
                major = int(compute_capability_line[-3])
                minor = int(compute_capability_line[-1])
                # Turn into float
                compute_capability = major + 0.1 * minor
    return compute_capability


def get_best_gpu():
    """
    Get the "best" GPU available. This currently chooses the GPU with highest compute
    capability and lowest GPU ID (as reported by `nvidia-smi`)
    """
    gpu_list = get_available_gpus()
    best_gpu_id = 0
    best_compute_capability = 0
    for gpu_id, gpu, in enumerate(gpu_list):
        compute_capability = get_compute_capability(gpu_id)
        if compute_capability > best_compute_capability:
            best_compute_capability = compute_capability
            best_gpu_id = gpu_id

    return best_gpu_id, best_compute_capability


if __name__ == "__main__":
    print(get_best_gpu())
    #a = nvidia_smi()
    #print(a)

