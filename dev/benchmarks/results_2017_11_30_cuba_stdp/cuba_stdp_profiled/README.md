
# Benchmark results from 29.11.2017
## Description:



## Last git log:
```
commit 65e51048f25caaee2a6e0396269f90821d994f85
Author: Denis Alevi <mail@denisalevi.de>
Date:   Mon Nov 27 17:55:05 2017 +0100

    Add recent benchmark results

```
There is also a `git diff` saved in the current directory.

## Results

### CUBA
![](plots/speed_test_CUBA_absolute.svg)
![](plots/speed_test_CUBA_profiling.svg)
![](plots/speed_test_CUBA_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==6637== NVPROF is profiling process 6637, command: ./main
==6637== Profiling application: ./main
==6637== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   27.59%  59.367ms     10000  5.9360us  5.7280us  6.9130us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double, double*, double*, double*, bool*)
                   23.11%  49.736ms     10000  4.9730us  3.2960us  20.256us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double*, double, int*, int, int*, bool*)
                   21.48%  46.232ms     10000  4.6230us  3.2960us  15.424us  kernel_synapses_1_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, int*, double*, bool*)
                   11.66%  25.090ms     10000  2.5080us  2.2720us  3.0080us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    8.37%  18.003ms     10000  1.8000us  1.6640us  2.1760us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*, bool*)
                    7.79%  16.764ms     10000  1.6760us  1.6000us  2.0480us  _GLOBAL__N__69_tmpxft_000017f7_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
      API calls:   55.27%  767.72ms     60000  12.795us  10.547us  9.0021ms  cudaLaunch
                   35.89%  498.54ms     80001  6.2310us  2.4830us  372.18us  cudaDeviceSynchronize
                    6.50%  90.343ms    520000     173ns     138ns  371.16us  cudaSetupArgument
                    1.33%  18.502ms     60000     308ns     238ns  364.34us  cudaConfigureCall
                    0.99%  13.745ms     50000     274ns     217ns  21.746us  cudaGetLastError
                    0.01%  138.51us         1  138.51us  138.51us  138.51us  cudaMemGetInfo
                    0.00%  33.472us        39     858ns     721ns  1.8600us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  30.648us         8  3.8310us  3.1320us  5.3030us  cudaFuncGetAttributes
                    0.00%  6.3800us        12     531ns     343ns  1.3920us  cudaDeviceGetAttribute
                    0.00%  2.9800us         3     993ns     737ns  1.3910us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==5900== NVPROF is profiling process 5900, command: ./main
==5900== Profiling application: ./main
==5900== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   27.83%  60.653ms     10000  6.0650us  5.7920us  7.0400us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double, double*, double*, double*, bool*)
                   23.00%  50.122ms     10000  5.0120us  3.2960us  24.320us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double*, double, int*, int, int*, bool*)
                   20.65%  45.008ms     10000  4.5000us  3.2960us  17.824us  kernel_synapses_1_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, int*, double*, bool*)
                   11.47%  25.008ms     10000  2.5000us  2.2720us  3.1680us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    9.22%  20.085ms     10000  2.0080us  1.8560us  2.1760us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*, bool*)
                    7.83%  17.069ms     10000  1.7060us  1.6320us  2.2400us  _GLOBAL__N__69_tmpxft_00001511_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
      API calls:   85.16%  640.31ms     60000  10.671us  9.6060us  9.0686ms  cudaLaunch
                   11.50%  86.475ms    520000     166ns     135ns  344.23us  cudaSetupArgument
                    1.87%  14.092ms     60000     234ns     176ns  334.30us  cudaConfigureCall
                    1.43%  10.785ms     50000     215ns     189ns  10.220us  cudaGetLastError
                    0.02%  139.19us         1  139.19us  139.19us  139.19us  cudaMemGetInfo
                    0.00%  31.512us         8  3.9390us  3.0080us  5.7970us  cudaFuncGetAttributes
                    0.00%  29.967us        39     768ns     653ns  1.9770us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  12.868us         1  12.868us  12.868us  12.868us  cudaDeviceSynchronize
                    0.00%  6.2440us        12     520ns     331ns  1.3150us  cudaDeviceGetAttribute
                    0.00%  3.7510us         3  1.2500us     823ns  1.7170us  cudaGetDevice

```

</p></details>


***

### CUBA - less kernels displayed
![](plots/speed_test_CUBA-less_kernels_displayed_min_15_profiling.svg)


***

### STDPNotEventDriven
![](plots/speed_test_STDP_absolute.svg)
![](plots/speed_test_STDP_profiling.svg)
![](plots/speed_test_STDP_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==28576== NVPROF is profiling process 28576, command: ./main
==28576== Profiling application: ./main
==28576== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.71%  73.256ms     10000  7.3250us  3.2960us  22.720us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, int, double*, double, double*, int, int*, int, int*)
                   15.80%  43.329ms     10000  4.3320us  3.8720us  6.2400us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                   12.77%  35.035ms     10000  3.5030us  3.3920us  6.3360us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, int, double, double*, int, int*)
                    9.94%  27.271ms     10000  2.7270us  2.6240us  3.1680us  kernel_synapses_stateupdater_codeobject(unsigned int, unsigned int, double*, int, double*, int, double*, int*)
                    9.28%  25.455ms     10000  2.5450us  2.2400us  2.9120us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    6.66%  18.254ms     10000  1.8250us  1.7600us  2.0800us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    6.65%  18.226ms     10000  1.8220us  1.7600us  2.5600us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    6.20%  16.991ms     10000  1.6990us  1.6000us  1.9200us  _GLOBAL__N__70_tmpxft_00006daf_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    5.88%  16.118ms     10000  1.6110us  1.4720us  1.8560us  _GLOBAL__N__69_tmpxft_00006dad_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    0.12%  330.53us         1  330.53us  330.53us  330.53us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
      API calls:   58.50%  1.10914s     90001  12.323us  9.6560us  9.1188ms  cudaLaunch
                   32.75%  621.00ms    100001  6.2090us  2.3660us  355.02us  cudaDeviceSynchronize
                    5.78%  109.54ms    660005     165ns     124ns  14.341us  cudaSetupArgument
                    1.49%  28.313ms     90001     314ns     245ns  12.028us  cudaConfigureCall
                    1.45%  27.511ms     70002     393ns     230ns  366.98us  cudaGetLastError
                    0.01%  208.18us         1  208.18us  208.18us  208.18us  cudaMalloc
                    0.01%  131.79us         1  131.79us  131.79us  131.79us  cudaMemGetInfo
                    0.00%  55.331us        74     747ns     647ns  1.4820us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  44.531us        12  3.7100us  3.1290us  4.8360us  cudaFuncGetAttributes
                    0.00%  9.1380us        20     456ns     333ns     893ns  cudaDeviceGetAttribute
                    0.00%  4.2750us         5     855ns     719ns  1.3080us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==27879== NVPROF is profiling process 27879, command: ./main
==27879== Profiling application: ./main
==27879== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   26.99%  74.731ms     10000  7.4730us  3.2960us  27.648us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, int, double*, double, double*, int, int*, int, int*)
                   15.88%  43.964ms     10000  4.3960us  3.9360us  6.4000us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                   12.62%  34.946ms     10000  3.4940us  3.3920us  6.4960us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, int, double, double*, int, int*)
                    9.80%  27.129ms     10000  2.7120us  2.3680us  2.9440us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    9.58%  26.535ms     10000  2.6530us  2.5600us  3.0400us  kernel_synapses_stateupdater_codeobject(unsigned int, unsigned int, double*, int, double*, int, double*, int*)
                    6.59%  18.247ms     10000  1.8240us  1.7280us  2.0480us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    6.58%  18.231ms     10000  1.8230us  1.7600us  2.5600us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    6.20%  17.155ms     10000  1.7150us  1.6320us  1.9520us  _GLOBAL__N__70_tmpxft_00006ae9_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    5.65%  15.632ms     10000  1.5630us  1.4720us  1.6960us  _GLOBAL__N__69_tmpxft_00006ae5_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    0.12%  329.57us         1  329.57us  329.57us  329.57us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
      API calls:   85.57%  910.45ms     90001  10.116us  8.9060us  9.1466ms  cudaLaunch
                   10.88%  115.80ms    660005     175ns     132ns  353.03us  cudaSetupArgument
                    2.00%  21.262ms     90001     236ns     181ns  330.07us  cudaConfigureCall
                    1.50%  15.984ms     70002     228ns     182ns  318.18us  cudaGetLastError
                    0.02%  207.89us         1  207.89us  207.89us  207.89us  cudaMalloc
                    0.01%  132.37us         1  132.37us  132.37us  132.37us  cudaMemGetInfo
                    0.01%  55.857us        74     754ns     674ns  1.5500us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  44.986us        12  3.7480us  3.0050us  5.8370us  cudaFuncGetAttributes
                    0.00%  13.864us         1  13.864us  13.864us  13.864us  cudaDeviceSynchronize
                    0.00%  9.5470us        20     477ns     338ns  1.1980us  cudaDeviceGetAttribute
                    0.00%  4.8700us         5     974ns     851ns  1.4220us  cudaGetDevice

```

</p></details>


***

### STDPNotEventDriven - less kernels displayed
![](plots/speed_test_STDP-less_kernels_displayed_min_15_profiling.svg)


***

### STDPEventDriven
![](plots/speed_test_STDPEventDriven_absolute.svg)
![](plots/speed_test_STDPEventDriven_profiling.svg)
![](plots/speed_test_STDPEventDriven_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfigurationProfileCPU**</summary><p>
Profile summary for `N = 1000`:

```
==18877== NVPROF is profiling process 18877, command: ./main
==18877== Profiling application: ./main
==18877== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.24%  85.455ms     10000  8.5450us  3.3280us  25.984us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, double*, double, double*, int, int*, int, int*, int)
                   16.85%  43.327ms     10000  4.3320us  3.8400us  6.2080us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                   13.77%  35.393ms     10000  3.5390us  3.4240us  7.2320us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, double, double*, int, int*, int*, int)
                    9.92%  25.503ms     10000  2.5500us  2.2400us  2.9760us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    7.11%  18.278ms     10000  1.8270us  1.7600us  2.0800us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    6.37%  16.365ms     10000  1.6360us  1.4080us  1.7920us  _GLOBAL__N__70_tmpxft_00004798_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    6.31%  16.219ms     10000  1.6210us  1.5040us  1.8560us  _GLOBAL__N__69_tmpxft_00004796_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    6.31%  16.209ms     10000  1.6200us  1.5360us  2.5920us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    0.13%  330.27us         1  330.27us  330.27us  330.27us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
      API calls:   57.23%  936.43ms     80001  11.705us  9.9320us  9.2809ms  cudaLaunch
                   34.35%  562.06ms     90001  6.2450us  2.4600us  359.92us  cudaDeviceSynchronize
                    5.96%  97.491ms    580005     168ns     132ns  357.12us  cudaSetupArgument
                    1.41%  23.032ms     80001     287ns     242ns  13.914us  cudaConfigureCall
                    1.02%  16.685ms     60002     278ns     235ns  14.273us  cudaGetLastError
                    0.01%  200.02us         1  200.02us  200.02us  200.02us  cudaMalloc
                    0.01%  134.78us         1  134.78us  134.78us  134.78us  cudaMemGetInfo
                    0.00%  36.321us        10  3.6320us  3.0320us  4.6100us  cudaFuncGetAttributes
                    0.00%  28.911us        41     705ns     592ns  1.5350us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  7.7890us        16     486ns     346ns  1.1310us  cudaDeviceGetAttribute
                    0.00%  3.2980us         4     824ns     736ns  1.0260us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==18067== NVPROF is profiling process 18067, command: ./main
==18067== Profiling application: ./main
==18067== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.20%  86.044ms     10000  8.6040us  3.3600us  26.176us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, double*, double, double*, int, int*, int, int*, int)
                   16.74%  43.393ms     10000  4.3390us  3.8080us  5.9840us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                   13.67%  35.442ms     10000  3.5440us  3.4560us  7.0400us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, double, double*, int, int*, int*, int)
                    9.83%  25.469ms     10000  2.5460us  2.2400us  2.7520us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    7.17%  18.573ms     10000  1.8570us  1.7280us  2.0800us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    7.03%  18.222ms     10000  1.8220us  1.7280us  2.6240us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    6.26%  16.215ms     10000  1.6210us  1.4080us  1.7920us  _GLOBAL__N__70_tmpxft_0000448e_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    5.98%  15.512ms     10000  1.5510us  1.4400us  1.6960us  _GLOBAL__N__69_tmpxft_0000448c_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    0.13%  330.56us         1  330.56us  330.56us  330.56us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
      API calls:   83.75%  838.49ms     80001  10.480us  9.1490us  9.2085ms  cudaLaunch
                   12.30%  123.18ms    580005     212ns     154ns  365.89us  cudaSetupArgument
                    2.22%  22.230ms     80001     277ns     208ns  341.41us  cudaConfigureCall
                    1.68%  16.830ms     60002     280ns     217ns  348.09us  cudaGetLastError
                    0.02%  200.11us         1  200.11us  200.11us  200.11us  cudaMalloc
                    0.01%  131.26us         1  131.26us  131.26us  131.26us  cudaMemGetInfo
                    0.00%  37.933us        10  3.7930us  3.0410us  5.6940us  cudaFuncGetAttributes
                    0.00%  33.513us        41     817ns     707ns  1.6920us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  13.505us         1  13.505us  13.505us  13.505us  cudaDeviceSynchronize
                    0.00%  7.9010us        16     493ns     368ns  1.1420us  cudaDeviceGetAttribute
                    0.00%  4.0280us         4  1.0070us     817ns  1.4860us  cudaGetDevice

```

</p></details>


***

### STDPEventDriven - less kernels displayed
![](plots/speed_test_STDPEventDriven-less_kernels_displayed_min_15_profiling.svg)


