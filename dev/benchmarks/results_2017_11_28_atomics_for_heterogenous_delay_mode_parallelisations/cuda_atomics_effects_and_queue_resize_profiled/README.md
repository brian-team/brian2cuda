
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

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationBrunelHeterogAndPushAtomicResizeProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==24531== NVPROF is profiling process 24531, command: ./main
==24531== Profiling application: ./main
==24531== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.91%  132.38ms      2521  52.511us  14.048us  1.0672ms  _run_synapses_pre_push_spikes_push_kernel(unsigned int, unsigned int, unsigned int, int*)
                   18.34%  64.052ms     10000  6.4050us  3.5520us  8.3520us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, double*, int*)
                   13.11%  45.786ms     10000  4.5780us  4.3840us  5.6320us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double, double*, double*, double*, bool*, float*)
                    7.89%  27.566ms     10000  2.7560us  2.7200us  4.1280us  _run_synapses_pre_push_spikes_advance_kernel(void)
                    6.60%  23.060ms     10000  2.3050us  2.0800us  2.8480us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    5.89%  20.553ms     10000  2.0550us  2.0160us  4.1920us  [CUDA memcpy DtoH]
                    5.25%  18.329ms     10000  1.8320us  1.6640us  2.1760us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    4.80%  16.747ms     10000  1.6740us  1.6000us  2.2080us  _GLOBAL__N__69_tmpxft_00005e15_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
                    0.21%  731.84us         1  731.84us  731.84us  731.84us  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, __operator_&__(float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_args_st))>(curandStateXORWOW*, float2*, unsigned long, unsigned long, normal_args_st)
      API calls:   46.45%  719.26ms     62522  11.504us  9.5190us  8.6020ms  cudaLaunch
                   35.15%  544.39ms     60001  9.0720us  2.4110us  1.0720ms  cudaDeviceSynchronize
                   12.81%  198.43ms     10000  19.842us  18.034us  330.92us  cudaMemcpy
                    3.57%  55.321ms    350089     158ns     123ns  330.68us  cudaSetupArgument
                    1.15%  17.835ms     62522     285ns     182ns  10.032us  cudaConfigureCall
                    0.83%  12.881ms     52523     245ns     209ns  9.8600us  cudaGetLastError
                    0.02%  250.79us         1  250.79us  250.79us  250.79us  cudaMalloc
                    0.01%  147.52us         1  147.52us  147.52us  147.52us  cudaMemGetInfo
                    0.00%  28.259us         8  3.5320us  2.7680us  5.4040us  cudaFuncGetAttributes
                    0.00%  26.485us        39     679ns     562ns  1.7750us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  6.3090us        12     525ns     358ns  1.3730us  cudaDeviceGetAttribute
                    0.00%  2.9100us         3     970ns     717ns  1.4180us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationTestBrunelHeteroAtomicsProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==23837== NVPROF is profiling process 23837, command: ./main
==23837== Profiling application: ./main
==23837== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.49%  157.31ms     10000  15.731us  1.8560us  1.1459ms  _run_synapses_pre_push_spikes_push_kernel(unsigned int, unsigned int, unsigned int, int*)
                   17.67%  62.479ms     10000  6.2470us  3.4240us  7.9360us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, double*, int*)
                   12.96%  45.814ms     10000  4.5810us  4.3530us  5.4080us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double, double*, double*, double*, bool*, float*)
                    7.81%  27.614ms     10000  2.7610us  2.7200us  4.1920us  _run_synapses_pre_push_spikes_advance_kernel(void)
                    6.48%  22.902ms     10000  2.2900us  2.0160us  2.8170us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    5.57%  19.698ms     10000  1.9690us  1.6960us  2.2080us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    4.81%  17.002ms     10000  1.7000us  1.6320us  2.2400us  _GLOBAL__N__69_tmpxft_00005b5a_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
                    0.21%  731.94us         1  731.94us  731.94us  731.94us  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, __operator_&__(float2 curand_normal_scaled2<curandStateXORWOW>(curandStateXORWOW*, normal_args_st))>(curandStateXORWOW*, float2*, unsigned long, unsigned long, normal_args_st)
      API calls:   54.28%  776.23ms     70001  11.088us  9.1570us  9.5432ms  cudaLaunch
                   39.57%  565.83ms     60001  9.4300us  2.4970us  1.1523ms  cudaDeviceSynchronize
                    3.99%  57.063ms    380005     150ns     121ns  325.75us  cudaSetupArgument
                    1.16%  16.531ms     70001     236ns     172ns  25.540us  cudaConfigureCall
                    0.96%  13.788ms     60002     229ns     191ns  12.473us  cudaGetLastError
                    0.02%  304.60us         1  304.60us  304.60us  304.60us  cudaMalloc
                    0.01%  168.13us         1  168.13us  168.13us  168.13us  cudaMemGetInfo
                    0.00%  31.295us        39     802ns     568ns  4.4480us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  29.348us         8  3.6680us  2.8260us  5.7480us  cudaFuncGetAttributes
                    0.00%  6.1630us        12     513ns     356ns  1.2920us  cudaDeviceGetAttribute
                    0.00%  3.1870us         3  1.0620us     733ns  1.7050us  cudaGetDevice

```

</p></details>


***

### BrunelHakimModelHeterogeneousDelay - display less kernels in profiling 
![](plots/speed_test_min_15_BrunelHakimModelHeterogeneousDelay_profiling.svg)


