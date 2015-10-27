# THIS README AND THE CODE IS NOT YET USABLE AND UNDER HEAVY DEVELOPMENT. PLEASE COME BACK LATER :-)
# brian2cuda
CUDA implementation for the spiking network simulator BRIAN2 to generate GPU code. The device parameters (no of shared multiprocessors, shared and global memory sizes etc.) are automatically set during runtime according to the specific CUDA capable device properties.

Usage: 
```import brian2
import brian2cuda
set_device("cuda_standalone")```

Currently the implementation requires a specific commit (```d82e7a4419e61d262126a96dc1bed35e37bac761``` (check!) of the master branch of brian-team/brian2 to allow comparisons with brian-team/brian2genn (```commit-hash```) employing genn-team/genn (```commit-hash```).