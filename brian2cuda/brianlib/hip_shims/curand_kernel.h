// cuRAND header shim for HIP builds: only the HIP makefile puts this directory
// on the include path, so <curand_kernel.h> resolves to hipRAND through the
// compatibility header. CUDA builds never see it.
#include "../cuda_to_hip.h"
