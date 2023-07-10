Brian2CUDA
==========

Brian2CUDA is an extension of the spiking neural network simulator
[Brian2](https://github.com/brian-team/brian2), written in Python. It
generates C++/CUDA code to run simulations on NVIDIA GPUs.

For **support**, please use the [Brian forum](https://brian.discourse.group/). If
you think you found a bug in Brian2CUDA, please report it at the
[GitHub issue tracker](https://github.com/brian-team/brian2cuda/issues).

For **installation and usage instructions**, check out the
[Brian2CUDA documentation](https://brian2cuda.readthedocs.io).
For information on general Brian2 usage, check out the
[Brian2 documentation](http://brian2.readthedocs.io).

## Quick start
### Installation

You can install Brian2CUDA via `pip`:

```bash
python -m pip install brian2cuda
```

This will install a compatible version of Brian2 as dependency. For installation requirements and GPU configuration, check out the [Brian2CUDA documentation](https://brian2cuda.readthedocs.io/en/latest/index.html).

### Usage
Use your Brian2 code (see [Brian2 documentation](http://brian2.readthedocs.io/en/stable/index.html)) and modify the imports to:

```python
# Standard Brian2 import
from brian2 import *

# Enable GPU usage via Brian2CUDA
import brian2cuda
set_device("cuda_standalone")
```

See [Brian2's standalone code generation](https://brian2.readthedocs.io/en/stable/user/computation.html?highlight=set_device#standalone-code-generation) for more options for the `set_device` call.


## Citation
If you use this software in a published article, please cite
[our Brian2CUDA publication](https://www.frontiersin.org/articles/10.3389/fninf.2022.883700):

> Alevi, D, Stimberg, M, Sprekeler, H, Obermayer, K, Augustin, M. “Brian2CUDA: flexible and efficient simulation of spiking neural network models on GPUs” Frontiers in Neuroinformatics (2022). doi: 10.3389/fninf.2022.883700.

## License
Brian2CUDA is free software licensed under the [GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Testing
To run the test suite on Google Collab (no installation or GPU required), click on the badge below:

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brian-team/brian2cuda/blob/master/brian2cuda/tools/test_suite/run_tests.ipynb)