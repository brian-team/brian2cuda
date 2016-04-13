# THIS README AND THE CODE IS NOT YET USABLE AND UNDER HEAVY DEVELOPMENT. PLEASE COME BACK LATER :-)
# brian2cuda
CUDA implementation for the spiking network simulator BRIAN2 to generate GPU code. The device parameters (no of shared multiprocessors, shared and global memory sizes etc.) are automatically set during runtime according to the specific CUDA capable device properties.

Usage: 
```
import brian2
import brian2cuda
set_device("cuda_standalone")
```

Currently the implementation requires the following specific commits to allow comparisons between brian2cuda and brian2genn.

The `frozen` branch from moritzaugustin/brian2 (```1d435d16d6ce73a96acba4245827c17dddebcc0e```).

brian-team/brian2genn (```e1e0b790088fbd40cb713baba2dfa062c9ada903```)

genn-team/genn (```f9ea2fe4aa73799e4242b1e8713733d7c2570478```)


The correct commits are stored in submodules in the `frozen_repos` directory. You can initialize them all by executing:
```
git submodule update --init
```
or individually with
```
git submodule update --init frozen_repos/<name>
```
where `<name>` is one of the submodule names in the directory.


For `brian2` and `brian2genn` you cann now either install the correct package versions, e.g. with `pip install ./frozen_repos/<name>` (careful if you have another version installed already!), or add the package to your PYTHONPATH.

Install genn:
If you are using LINUX and your cuda is installed at `/usr/local/cuda/`, simply source `init_genn`:
```
source frozen_repos/init_genn
```
Otherwise follow the instructions at genn-team/genn
