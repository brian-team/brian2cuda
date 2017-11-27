
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

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationBrunelHeterogAndPushAtomicResize**</summary><p>
Profile summary for `N = 1000`:

```
==2700== NVPROF is profiling process 2700, command: ./main
==2700== Profiling application: ./main
==2700== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   36.32%  123.91ms      2523  49.113us  14.176us  1.3924ms  _run_synapses_pre_push_spikes_push_kernel(unsigned int, unsigned int, unsigned int, int*)
                   18.81%  64.168ms     10000  6.4160us  3.5840us  8.5440us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, double*, int*)
                   12.89%  43.962ms     10000  4.3960us  4.1600us  5.4080us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double, double*, double*, double*, bool*, float*)
                    8.62%  29.419ms     10000  2.9410us  2.8800us  4.2880us  _run_synapses_pre_push_spikes_advance_kernel(void)
                    6.74%  22.995ms     10000  2.2990us  2.0160us  2.8800us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    6.03%  20.585ms     10000  2.0580us  2.0160us  4.0960us  [CUDA memcpy DtoH]
                    5.48%  18.689ms     10000  1.8680us  1.7280us  2.2400us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    4.89%  16.676ms     10000  1.6670us  1.6000us  2.7520us  _GLOBAL__N__69_tmpxft_000008bc_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
                    0.21%  732.10us         1  732.10us  732.10us  732.10us  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, __operator_&__(float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_args_st))>(curandStateXORWOW*, float2*, unsigned long, unsigned long, normal_args_st)
      API calls:   63.43%  648.25ms     62524  10.368us  8.7500us  8.8943ms  cudaLaunch
                   28.22%  288.45ms     10000  28.844us  18.477us  1.3838ms  cudaMemcpy
                    5.61%  57.386ms    350097     163ns     124ns  335.99us  cudaSetupArgument
                    1.38%  14.127ms     62524     225ns     161ns  321.95us  cudaConfigureCall
                    1.30%  13.336ms     52525     253ns     200ns  300.39us  cudaGetLastError
                    0.03%  268.04us         1  268.04us  268.04us  268.04us  cudaMalloc
                    0.02%  166.72us         1  166.72us  166.72us  166.72us  cudaMemGetInfo
                    0.00%  30.363us        39     778ns     650ns  2.4670us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  29.284us         8  3.6600us  2.8650us  6.1410us  cudaFuncGetAttributes
                    0.00%  13.545us         1  13.545us  13.545us  13.545us  cudaDeviceSynchronize
                    0.00%  6.1940us        12     516ns     337ns  1.4590us  cudaDeviceGetAttribute
                    0.00%  3.8130us         3  1.2710us     863ns  1.9980us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==1993== NVPROF is profiling process 1993, command: ./main
==1993== Profiling application: ./main
==1993== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.03%  352.83ms     10000  35.283us  2.0480us  87.201us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, double*, int*)
                   23.69%  149.15ms     10000  14.915us  1.6320us  1.3164ms  _run_synapses_pre_push_spikes_push_kernel(unsigned int, unsigned int, unsigned int, int*)
                    6.70%  42.158ms     10000  4.2150us  3.8080us  5.6320us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double, double*, double*, double*, bool*, float*)
                    4.32%  27.228ms     10000  2.7220us  2.4960us  4.5120us  _run_synapses_pre_push_spikes_advance_kernel(void)
                    3.61%  22.747ms     10000  2.2740us  1.9200us  3.7760us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    3.00%  18.918ms     10000  1.8910us  1.7280us  3.7440us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    2.53%  15.914ms     10000  1.5910us  1.3440us  3.8080us  _GLOBAL__N__69_tmpxft_000005de_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
                    0.12%  731.65us         1  731.65us  731.65us  731.65us  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, __operator_&__(float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_args_st))>(curandStateXORWOW*, float2*, unsigned long, unsigned long, normal_args_st)
      API calls:   89.24%  744.81ms     70001  10.639us  8.5770us  8.8242ms  cudaLaunch
                    7.22%  60.281ms    380005     158ns     121ns  336.34us  cudaSetupArgument
                    1.85%  15.427ms     70001     220ns     159ns  319.22us  cudaConfigureCall
                    1.60%  13.340ms     60002     222ns     175ns  326.16us  cudaGetLastError
                    0.04%  332.76us         1  332.76us  332.76us  332.76us  cudaDeviceSynchronize
                    0.03%  253.93us         1  253.93us  253.93us  253.93us  cudaMalloc
                    0.02%  146.47us         1  146.47us  146.47us  146.47us  cudaMemGetInfo
                    0.00%  29.198us         8  3.6490us  2.7670us  6.3670us  cudaFuncGetAttributes
                    0.00%  27.382us        39     702ns     578ns  1.8100us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  6.0870us        12     507ns     326ns  1.3870us  cudaDeviceGetAttribute
                    0.00%  3.7450us         3  1.2480us     822ns  2.0410us  cudaGetDevice

```

</p></details>


