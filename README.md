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

brian-team/brian2 (```d82e7a4419e61d262126a96dc1bed35e37bac761```)
brian-team/brian2genn (```e1e0b790088fbd40cb713baba2dfa062c9ada903```)
genn-team/genn (```f9ea2fe4aa73799e4242b1e8713733d7c2570478```)


The correct brian2 commit is stored as a submodule of this repository. You can initialize it by executing:
```
git submodule update --init
```

This way the correct brian2 version will be accesible from the brian2cuda directory. If you want to use your own brian2 installation, don't initialize the submodule.
