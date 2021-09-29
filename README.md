
Brian2CUDA
==========

Please note that this package is still under developement and has not been released yet.

Brian2CUDA is an extention of the spiking neural network simulator [Brian2](https://github.com/brian-team/brian2), implementing a [Brian2 standalone device](http://brian2.readthedocs.io/en/stable/developer/devices.html) to generate C++/CUDA code to run simluations on NVIDIA general purpose graphics processing units (GPGPUs).

### Usage: 
Use your Brian2 code (see [Brian2 documentation](http://brian2.readthedocs.io/en/stable/index.html)) and modify the imports to:
```python
from brian2 import *
import brian2cuda
set_device("cuda_standalone")
```

### Installation
#### Requirements
- Brian2: The correct version with which this implementation is working is stored in a submodule in `frozen_repos/brian2` 

After cloning brian2cuda, you can initialise the submodule from inside this repository:

```
cd brian2cuda
git submodule update --init frozen_repos/brian2
```

Next, you need to incorporate a few changes into the brian2 repo. These changes are already included in the newest brian2 version, but need to be added manually to the brian2 version that is used in `frozen_repose`. To add these changes, apply the `brian2.diff` file to the repository:
```
cd frozen_repos/brian2
git apply ../brian2.diff
# return to the brian2cuda root directory for the following installation
cd ../..
```

Now you can install the correct brian2 and brian2cuda versions using pip (Be careful if you already have a Brian2 version installed in your current Python environment!):
```
pip install .
pip install ./frozen_repos/brian2
```

If you want to install in developer mode, add the `-e` flag to the `pip` commands.

Or just add them to your `PYTHONPATH` (in which case you need to install their dependencies manually).

#### Comparison with brian2genn

The [brian2genn](https://github.com/brian-team/brian2genn) and [GeNN](https://github.com/genn-team/genn) versions that can be used with the same brian2 version as brian2cuda are stored in submodules in `frozen_repos/brian2genn` and `frozen_repos/genn`. You can initialise them with:
```
git submodule update --init frozen_repos/brian2genn
git submodule update --init frozen_repos/genn
```
If you are using LINUX and your CUDA is installed in `/usr/local/cuda/`, you can now just source `init_genn` to set the enironmental variables needed for GeNN to work:
```
source frozen_repos/init_genn
```
Otherwise modify the `CUDA_PATH` in `init_genn` accordingly or follow the instructions from the [GeNN repository](https://github.com/genn-team/genn)

Now you can install `brian2genn` either with pip:
```
pip install ./frozen_repos/brian2genn
```
or just add it to your `PYTHONPATH`.
