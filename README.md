
Brian2CUDA
==========

Brian2CUDA is an extension of the spiking neural network simulator [Brian2](https://github.com/brian-team/brian2), implementing a [Brian2 standalone device](http://brian2.readthedocs.io/en/stable/developer/devices.html) to generate C++/CUDA code to run simulations on NVIDIA general purpose graphics processing units (GPGPUs).

### Usage: 
Use your Brian2 code (see [Brian2 documentation](http://brian2.readthedocs.io/en/stable/index.html)) and modify the imports to:
```python
from brian2 import *
import brian2cuda
set_device("cuda_standalone")
```

### Installation
#### Requirements

##### Operating system
Only Linux environments are supported. For progress on Windows support, see #225.

##### Brian2
The correct Brian2 version with which this implementation is working is stored in a submodule in `frozen_repos/brian2` 

After cloning Brian2CUDA, you can initialize the submodule from inside this repository:

```bash
cd brian2cuda
git submodule update --init frozen_repos/brian2
```

Next, you need to incorporate a few changes into the Brian2 repository. These
changes are already included in the newest Brian2 version, but need to be
added manually to the Brian2 version that is used in `frozen_repos`. To add
these changes, apply the `brian2.diff` file to the repository:
```bash
cd frozen_repos
bash update_brian2.sh
# return to the brian2cuda root directory for the following installation
cd ..
```

Now you can install the correct Brian2 and Brian2CUDA versions using pip (Be careful if you already have a Brian2 version installed in your current Python environment!):
```bash
pip install .
pip install ./frozen_repos/brian2
```

If you want to install in developer mode, add the `-e` flag to the `pip` commands.

Or just add them to your `PYTHONPATH` (in which case you need to install their dependencies manually).

#### Comparison with Brian2GeNN

The [Brian2GeNN](https://github.com/brian-team/brian2genn) and
[GeNN](https://github.com/genn-team/genn) versions that can be used with the
same Brian2 version as Brian2CUDA are stored in submodules in
`frozen_repos/brian2genn` and `frozen_repos/genn`. You can initialize them
with:
```bash
git submodule update --init frozen_repos/brian2genn
git submodule update --init frozen_repos/genn
```
Now you can install Brian2GeNN either with pip:
```bash
pip install ./frozen_repos/brian2genn
```
or just add it to your `PYTHONPATH`.

For GeNN to work correctly, you need to set the environment variable `GENN_PATH`. You can use the following command:
```bash
source frozen_repos/init_genn.sh
```

If you are using a conda environment, make sure that you don't have any
compiler packages installed in your environment or unset the corresponding
environment variables. Otherwise you can get mismatches of compiler and linker
versions in GeNN. During the first GeNN run, a set of libraries are compiled
once. If you change you compiler or environment later on, you might have to
reinstall GeNN (remove the compiled filed).

Otherwise follow the instructions from the [GeNN repository](https://github.com/genn-team/genn).
