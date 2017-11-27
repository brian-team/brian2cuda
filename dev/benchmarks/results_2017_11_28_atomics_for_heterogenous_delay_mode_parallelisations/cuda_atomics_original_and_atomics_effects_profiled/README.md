
# Benchmark results from 28.11.2017
## Description:



## Last git log:
```
commit 8987de24ed9f4a3b1a276496407fca1087f04004
Author: Denis Alevi <mail@denisalevi.de>
Date:   Mon Nov 20 14:31:09 2017 +0100

    Fix critical section to include the actual pushing

```
There is also a `git diff` saved in the current directory.

## Results

### BrunelHakimModelHeterogeneousDelay
![](plots/speed_test_BrunelHakimModelHeterogeneousDelay_absolute.svg)
![](plots/speed_test_BrunelHakimModelHeterogeneousDelay_profiling.svg)
![](plots/speed_test_BrunelHakimModelHeterogeneousDelay_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==22819== NVPROF is profiling process 22819, command: ./main
==22819== Profiling application: ./main
==22819== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.40%  374.75ms     10000  37.475us  2.2080us  87.617us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, double*, int*)
                   22.46%  146.65ms     10000  14.664us  1.7280us  960.13us  _run_synapses_pre_push_spikes_push_kernel(unsigned int, unsigned int, unsigned int, int*)
                    6.51%  42.496ms     10000  4.2490us  4.0640us  5.4400us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double, double*, double*, double*, bool*, float*)
                    4.30%  28.046ms     10000  2.8040us  2.7200us  4.3840us  _run_synapses_pre_push_spikes_advance_kernel(void)
                    3.74%  24.420ms     10000  2.4420us  2.1120us  2.9120us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    3.10%  20.239ms     10000  2.0230us  1.9520us  2.2400us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    2.38%  15.522ms     10000  1.5520us  1.4400us  1.9520us  _GLOBAL__N__69_tmpxft_0000578e_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
                    0.11%  732.55us         1  732.55us  732.55us  732.55us  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, __operator_&__(float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_args_st))>(curandStateXORWOW*, float2*, unsigned long, unsigned long, normal_args_st)
      API calls:   48.78%  862.78ms     60001  14.379us  2.6750us  965.88us  cudaDeviceSynchronize
                   45.73%  808.74ms     70001  11.553us  9.9030us  8.8572ms  cudaLaunch
                    3.59%  63.553ms    380005     167ns     135ns  324.80us  cudaSetupArgument
                    1.05%  18.499ms     70001     264ns     187ns  12.032us  cudaConfigureCall
                    0.82%  14.507ms     60002     241ns     191ns  11.691us  cudaGetLastError
                    0.02%  272.19us         1  272.19us  272.19us  272.19us  cudaMalloc
                    0.01%  153.60us         1  153.60us  153.60us  153.60us  cudaMemGetInfo
                    0.00%  30.182us         8  3.7720us  3.0160us  5.4880us  cudaFuncGetAttributes
                    0.00%  28.610us        39     733ns     616ns  2.0690us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  6.6020us        12     550ns     360ns  1.2870us  cudaDeviceGetAttribute
                    0.00%  2.7290us         3     909ns     660ns  1.3620us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationTestBrunelHeteroAtomicsProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==23535== NVPROF is profiling process 23535, command: ./main
==23535== Profiling application: ./main
==23535== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.76%  145.65ms     10000  14.565us  1.9840us  1.1359ms  _run_synapses_pre_push_spikes_push_kernel(unsigned int, unsigned int, unsigned int, int*)
                   18.94%  64.522ms     10000  6.4520us  3.6480us  8.1280us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, double*, int*)
                   12.92%  43.998ms     10000  4.3990us  4.1280us  5.5360us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double, double*, double*, double*, bool*, float*)
                    8.09%  27.540ms     10000  2.7530us  2.6880us  4.5120us  _run_synapses_pre_push_spikes_advance_kernel(void)
                    6.73%  22.929ms     10000  2.2920us  2.0800us  2.8170us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    5.80%  19.741ms     10000  1.9740us  1.8560us  2.1760us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    4.55%  15.502ms     10000  1.5500us  1.4400us  2.1760us  _GLOBAL__N__69_tmpxft_00005a32_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
                    0.21%  731.97us         1  731.97us  731.97us  731.97us  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, __operator_&__(float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_args_st))>(curandStateXORWOW*, float2*, unsigned long, unsigned long, normal_args_st)
      API calls:   55.46%  798.58ms     70001  11.408us  9.3530us  8.8769ms  cudaLaunch
                   37.87%  545.25ms     60001  9.0870us  2.4500us  1.1408ms  cudaDeviceSynchronize
                    4.14%  59.601ms    380005     156ns     125ns  312.80us  cudaSetupArgument
                    1.35%  19.377ms     70001     276ns     185ns  13.408us  cudaConfigureCall
                    1.16%  16.665ms     60002     277ns     203ns  11.227us  cudaGetLastError
                    0.02%  277.76us         1  277.76us  277.76us  277.76us  cudaMalloc
                    0.01%  156.74us         1  156.74us  156.74us  156.74us  cudaMemGetInfo
                    0.00%  32.165us         8  4.0200us  2.8580us  7.6380us  cudaFuncGetAttributes
                    0.00%  27.873us        39     714ns     603ns  1.7260us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  6.6820us        12     556ns     373ns  1.4220us  cudaDeviceGetAttribute
                    0.00%  2.8330us         3     944ns     675ns  1.4210us  cudaGetDevice

```

</p></details>


***

### BrunelHakimModelHeterogeneousDelay - display less kernels in profiling
![](plots/speed_test_min_15_BrunelHakimModelHeterogeneousDelay_profiling.svg)


