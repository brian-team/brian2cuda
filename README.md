# THIS README AND THE CODE IS NOT YET USABLE AND UNDER HEAVY DEVELOPMENT. PLEASE COME BACK LATER :-)
# brian2cuda
CUDA implementation for the spiking network simulator BRIAN2 to generate GPU code. The device parameters (no of shared multiprocessors, shared and global memory sizes etc.) are automatically set during runtime according to the specific CUDA capable device properties.

Usage: 
```
import brian2
import brian2cuda
set_device("cuda_standalone")
```

Currently the implementation requires the following specific commits:
brian-team/brian2 (```d82e7a4419e61d262126a96dc1bed35e37bac761```) or the frozen branch of moritzaugustin/brian2
brian-team/brian2genn (```e1e0b790088fbd40cb713baba2dfa062c9ada903```)
genn-team/genn (```f9ea2fe4aa73799e4242b1e8713733d7c2570478```)
to allow comparisons between brian2cuda and brian2genn
