
brian2cuda
==========

Please note that this package is still under developement and some features are not yet usable.

Brian2cuda is an extention of the spiking neural network simulator [Brian2](https://github.com/brian-team/brian2), implementing a [Brian2 standalone device](http://brian2.readthedocs.io/en/stable/developer/devices.html) to generate C++/CUDA code to run simluations on NVIDIA general purpose graphics processing units (GPGPUs).

### Usage: 
Use your Brian2 code (see [Brian2 documentation](http://brian2.readthedocs.io/en/stable/index.html)) and modify the imports to:
```
from brian2 import *
import brian2cuda
set_device("cuda_standalone")
```

### Installation
The Brian2 version with which this implementation is working is stored in a submodule in `frozen_repos/brian` (currently Brian2 commit [fadc6a0aeb90d1b4d343470628457d8561536f67
](https://github.com/brian-team/brian2/tree/fadc6a0aeb90d1b4d343470628457d8561536f67)).

After cloning you can initialise the submodule by running this command from inside this repository:

```
git submodule update --init frozen_repos/brian2
```

Now you can install the correct Brian2 and brian2cuda versions using pip (Be careful if you already have a Brian2 version installed in your current Python environment!):
```
pip install .
pip install ./frozen_repos/brian2
```
Or just add them to your `PYTHONPATH` (in which case you need to install their dependencies manually).

### Comparison with brian2genn

The [brian2genn](https://github.com/brian-team/brian2genn) and [GeNN](https://github.com/genn-team/genn) versions that can be used for comparison are stored in submodules in `frozen_repos/brian2genn` and `frozen_repos/genn`. You can initialise them with:
```
git submodule update --init frozen_repos/brian2genn
git submodule update --init frozen_repos/genn
```
If you are using LINUX and your CUDA is installed in `/usr/local/cuda/`, you can now just source `init_genn` to set the enironmental variables needed for GeNN to work:
```
source frozen_repos/init_genn
```
Otherwise follow the instructions from the [GeNN repository](https://github.com/genn-team/genn)

Now you can install `brian2genn` either with pip:
```
pip install ./frozen_repos/brian2genn
```
or just add it to your `PYTHONPATH`.
