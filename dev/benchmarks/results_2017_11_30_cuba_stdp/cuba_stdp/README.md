
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

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==9905== NVPROF is profiling process 9905, command: ./main
==9905== Profiling application: ./main
==9905== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   27.60%  60.179ms     10000  6.0170us  5.8560us  7.0720us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double, double*, double*, double*, bool*)
                   23.08%  50.318ms     10000  5.0310us  3.2960us  23.232us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double*, double, int*, int, int*, bool*)
                   21.95%  47.850ms     10000  4.7840us  3.2960us  19.968us  kernel_synapses_1_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, int*, int, double, int*, int, int*, double*, bool*)
                   11.42%  24.905ms     10000  2.4900us  2.2720us  2.8800us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double, double*, double*, bool*)
                    8.26%  18.018ms     10000  1.8010us  1.6640us  2.1120us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*, bool*)
                    7.68%  16.743ms     10000  1.6740us  1.5360us  2.0800us  _GLOBAL__N__69_tmpxft_000024a7_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_97ebdcc0::_reset_neurongroup_thresholder_codeobject(int*)
      API calls:   85.15%  643.72ms     60000  10.728us  9.3820us  9.0568ms  cudaLaunch
                   11.40%  86.186ms    520000     165ns     134ns  363.54us  cudaSetupArgument
                    1.93%  14.574ms     60000     242ns     182ns  349.11us  cudaConfigureCall
                    1.50%  11.304ms     50000     226ns     194ns  14.049us  cudaGetLastError
                    0.02%  134.84us         1  134.84us  134.84us  134.84us  cudaMemGetInfo
                    0.00%  31.105us         8  3.8880us  3.0120us  5.6890us  cudaFuncGetAttributes
                    0.00%  30.378us        39     778ns     653ns  1.8930us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  13.359us         1  13.359us  13.359us  13.359us  cudaDeviceSynchronize
                    0.00%  6.2530us        12     521ns     334ns  1.3690us  cudaDeviceGetAttribute
                    0.00%  3.6520us         3  1.2170us     789ns  1.6000us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **GeNNConfigurationOptimized**</summary><p>
Profile summary for `N = 1000`:

```
==10402== NVPROF is profiling process 10402, command: ./main test 1.0 1
==10402== Profiling application: ./main test 1.0 1
==10402== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.74%  78.512ms     10000  7.8510us  7.0080us  10.336us  calcNeurons
                   35.42%  43.636ms     10000  4.3630us  1.9840us  20.512us  calcSynapses
                    0.65%  799.04us        56  14.268us     960ns  163.46us  [CUDA memcpy HtoD]
                    0.19%  234.66us        13  18.050us  1.9840us  155.30us  [CUDA memcpy DtoH]
      API calls:   67.87%  468.06ms        16  29.253ms  15.634us  464.71ms  cudaHostAlloc
                   29.73%  204.98ms     20000  10.248us  9.4610us  337.99us  cudaLaunch
                    1.01%  6.9362ms     20000     346ns     275ns  331.07us  cudaConfigureCall
                    0.81%  5.6041ms     20000     280ns     221ns  329.96us  cudaSetupArgument
                    0.31%  2.1674ms        73  29.690us     512ns  179.18us  cudaMemcpy
                    0.18%  1.2374ms        16  77.339us  9.8610us  230.18us  cudaMalloc
                    0.06%  398.46us        94  4.2380us     154ns  155.40us  cuDeviceGetAttribute
                    0.02%  118.62us         1  118.62us  118.62us  118.62us  cuDeviceTotalMem
                    0.01%  48.855us         1  48.855us  48.855us  48.855us  cuDeviceGetName
                    0.00%  22.545us        16  1.4090us     582ns  3.4920us  cudaGetSymbolAddress
                    0.00%  9.5420us         1  9.5420us  9.5420us  9.5420us  cudaSetDevice
                    0.00%  3.6290us         3  1.2090us     200ns  2.4090us  cuDeviceGetCount
                    0.00%  1.5380us         1  1.5380us  1.5380us  1.5380us  cudaGetDeviceCount
                    0.00%  1.1220us         2     561ns     362ns     760ns  cuDeviceGet

```

</p></details>


***

### STDP (with SpikeMonitor)
![](plots/speed_test_STDP_absolute.svg)
![](plots/speed_test_STDP_profiling.svg)
![](plots/speed_test_STDP_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==8893== NVPROF is profiling process 8893, command: ./main
==8893== Profiling application: ./main
==8893== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   32.18%  119.34ms     10000  11.934us  1.6000us  26.926ms  kernel_spikemonitor_codeobject(unsigned int, int*, double, int*, int*, int*, int, int*, double*, int, int*, int*)
                   20.94%  77.684ms     10000  7.7680us  3.3600us  25.728us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, double*, double, double*, int, int*, int, int*, int)
                   11.71%  43.439ms     10000  4.3430us  3.8400us  6.0800us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                    9.85%  36.549ms     10000  3.6540us  3.5520us  7.0080us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, double, double*, int, int*, int*, int)
                    6.79%  25.173ms     10000  2.5170us  2.1760us  3.6800us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    4.91%  18.216ms     10000  1.8210us  1.7280us  4.3200us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    4.78%  17.745ms     10000  1.7740us  1.5680us  3.6800us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    4.51%  16.723ms     10000  1.6720us  1.6000us  3.2640us  _GLOBAL__N__70_tmpxft_00002089_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    4.22%  15.645ms     10000  1.5640us  1.3760us  3.5200us  _GLOBAL__N__69_tmpxft_00002086_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    0.09%  330.21us         1  330.21us  330.21us  330.21us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
                    0.02%  68.192us         1  68.192us  68.192us  68.192us  _run_spikemonitor_codeobject_init(void)
      API calls:   85.86%  977.23ms     90002  10.857us  9.1510us  10.887ms  cudaLaunch
                   10.27%  116.88ms    700005     166ns     137ns  331.60us  cudaSetupArgument
                    2.12%  24.161ms     90002     268ns     176ns  335.73us  cudaConfigureCall
                    1.71%  19.473ms     70003     278ns     205ns  329.44us  cudaGetLastError
                    0.02%  213.58us         1  213.58us  213.58us  213.58us  cudaMalloc
                    0.01%  133.47us         1  133.47us  133.47us  133.47us  cudaMemGetInfo
                    0.00%  43.181us        11  3.9250us  3.2280us  6.4450us  cudaFuncGetAttributes
                    0.00%  32.048us        42     763ns     627ns  1.7760us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  13.358us         1  13.358us  13.358us  13.358us  cudaDeviceSynchronize
                    0.00%  7.7520us        16     484ns     356ns  1.1240us  cudaDeviceGetAttribute
                    0.00%  4.0510us         4  1.0120us     822ns  1.5030us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **GeNNConfigurationOptimized**</summary><p>
Profile summary for `N = 1000`:

```
==9420== NVPROF is profiling process 9420, command: ./main test 1.0 1
==9420== Profiling application: ./main test 1.0 1
==9420== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.20%  103.62ms     10000  10.362us  1.5680us  45.248us  calcSynapses
                   19.97%  41.214ms     10000  4.1210us  3.1040us  6.7200us  calcNeurons
                   17.73%  36.597ms     17812  2.0540us  2.0160us  4.7360us  [CUDA memcpy DtoH]
                   12.06%  24.885ms     10000  2.4880us  2.3680us  10.848us  learnSynapsesPost
                    0.05%  94.016us        70  1.3430us     960ns  2.0480us  [CUDA memcpy HtoD]
      API calls:   34.18%  358.40ms        20  17.920ms  8.3270us  356.55ms  cudaHostAlloc
                   32.17%  337.26ms     30000  11.241us  9.5510us  356.07us  cudaLaunch
                   31.72%  332.57ms     20095  16.549us     231ns  988.56us  cudaMemcpy
                    1.03%  10.770ms     30000     358ns     283ns  331.73us  cudaConfigureCall
                    0.77%  8.0617ms     30000     268ns     208ns  334.35us  cudaSetupArgument
                    0.08%  809.75us        20  40.487us  8.0280us  232.94us  cudaMalloc
                    0.04%  401.78us        94  4.2740us     161ns  156.02us  cuDeviceGetAttribute
                    0.01%  113.16us         1  113.16us  113.16us  113.16us  cuDeviceTotalMem
                    0.00%  37.103us         1  37.103us  37.103us  37.103us  cuDeviceGetName
                    0.00%  22.451us        20  1.1220us     525ns  5.8000us  cudaGetSymbolAddress
                    0.00%  9.5720us         1  9.5720us  9.5720us  9.5720us  cudaSetDevice
                    0.00%  3.2610us         3  1.0870us     219ns  2.3710us  cuDeviceGetCount
                    0.00%  1.6100us         1  1.6100us  1.6100us  1.6100us  cudaGetDeviceCount
                    0.00%  1.0470us         2     523ns     250ns     797ns  cuDeviceGet

```

</p></details>


***

### STDPEventDriven
![](plots/speed_test_STDPEventDriven_absolute.svg)
![](plots/speed_test_STDPEventDriven_profiling.svg)
![](plots/speed_test_STDPEventDriven_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==19561== NVPROF is profiling process 19561, command: ./main
==19561== Profiling application: ./main
==19561== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.06%  85.737ms     10000  8.5730us  3.3600us  26.176us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, double*, double, double*, int, int*, int, int*, int)
                   16.85%  43.713ms     10000  4.3710us  3.8720us  6.4320us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                   13.67%  35.462ms     10000  3.5460us  3.4560us  7.1040us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, double, double*, int, int*, int*, int)
                    9.83%  25.505ms     10000  2.5500us  2.2400us  2.8480us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    7.03%  18.243ms     10000  1.8240us  1.7600us  2.5920us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    7.01%  18.182ms     10000  1.8180us  1.6960us  2.2080us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    6.41%  16.614ms     10000  1.6610us  1.5360us  1.9520us  _GLOBAL__N__70_tmpxft_00004a64_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    6.01%  15.583ms     10000  1.5580us  1.4720us  1.7280us  _GLOBAL__N__69_tmpxft_00004a60_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    0.13%  330.21us         1  330.21us  330.21us  330.21us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
      API calls:   84.19%  838.94ms     80001  10.486us  9.1520us  8.9317ms  cudaLaunch
                   11.76%  117.21ms    580005     202ns     157ns  419.79us  cudaSetupArgument
                    2.27%  22.642ms     80001     283ns     205ns  337.22us  cudaConfigureCall
                    1.74%  17.290ms     60002     288ns     220ns  371.39us  cudaGetLastError
                    0.02%  198.76us         1  198.76us  198.76us  198.76us  cudaMalloc
                    0.01%  139.83us         1  139.83us  139.83us  139.83us  cudaMemGetInfo
                    0.00%  37.555us        10  3.7550us  3.0440us  6.0110us  cudaFuncGetAttributes
                    0.00%  31.926us        41     778ns     680ns  1.6620us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  13.490us         1  13.490us  13.490us  13.490us  cudaDeviceSynchronize
                    0.00%  8.0030us        16     500ns     369ns  1.0430us  cudaDeviceGetAttribute
                    0.00%  4.0740us         4  1.0180us     792ns  1.5710us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **GeNNConfigurationOptimized**</summary><p>
Profile summary for `N = 1000`:

```
==20069== NVPROF is profiling process 20069, command: ./main test 1.0 1
==20069== Profiling application: ./main test 1.0 1
==20069== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.14%  103.55ms     10000  10.355us  1.5680us  49.760us  calcSynapses
                   24.06%  40.741ms     10000  4.0740us  3.0720us  6.6880us  calcNeurons
                   14.71%  24.918ms     10000  2.4910us  2.3360us  10.560us  learnSynapsesPost
                    0.06%  94.593us        70  1.3510us     960ns  2.0800us  [CUDA memcpy HtoD]
                    0.03%  55.553us        19  2.9230us  2.0160us  4.8010us  [CUDA memcpy DtoH]
      API calls:   56.62%  434.02ms        20  21.701ms  16.555us  432.05ms  cudaHostAlloc
                   40.47%  310.26ms     30000  10.341us  9.4140us  347.35us  cudaLaunch
                    1.37%  10.508ms     30000     350ns     275ns  330.81us  cudaConfigureCall
                    1.08%  8.2824ms     30000     276ns     221ns  333.57us  cudaSetupArgument
                    0.25%  1.9098ms        95  20.103us     434ns  41.200us  cudaMemcpy
                    0.13%  998.34us        20  49.917us  13.252us  259.26us  cudaMalloc
                    0.05%  419.41us        94  4.4610us     183ns  162.60us  cuDeviceGetAttribute
                    0.02%  126.30us         1  126.30us  126.30us  126.30us  cuDeviceTotalMem
                    0.00%  38.221us         1  38.221us  38.221us  38.221us  cuDeviceGetName
                    0.00%  29.710us        20  1.4850us     972ns  6.2560us  cudaGetSymbolAddress
                    0.00%  9.9350us         1  9.9350us  9.9350us  9.9350us  cudaSetDevice
                    0.00%  3.4560us         3  1.1520us     236ns  2.5630us  cuDeviceGetCount
                    0.00%  1.7080us         1  1.7080us  1.7080us  1.7080us  cudaGetDeviceCount
                    0.00%  1.2540us         2     627ns     267ns     987ns  cuDeviceGet

```

</p></details>


***

### STDPNotEventDriven
![](plots/speed_test_STDPNotEventDriven_absolute.svg)
![](plots/speed_test_STDPNotEventDriven_profiling.svg)
![](plots/speed_test_STDPNotEventDriven_relative.svg)

<details><summary>Examplary `nvprof` results for **CUDAStandaloneConfiguration**</summary><p>
Profile summary for `N = 1000`:

```
==18533== NVPROF is profiling process 18533, command: ./main
==18533== Profiling application: ./main
==18533== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   27.01%  73.513ms     10000  7.3510us  3.2960us  29.120us  kernel_synapses_pre_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, int, double*, double, double*, int, int*, int, int*)
                   16.08%  43.771ms     10000  4.3770us  3.9040us  6.3360us  kernel_neurongroup_stateupdater_codeobject(unsigned int, unsigned int, double*, double*, double*)
                   12.83%  34.925ms     10000  3.4920us  3.3920us  6.4000us  kernel_synapses_post_codeobject(unsigned int, unsigned int, unsigned int, int*, unsigned int, double*, int, double*, int, double*, int, int*, int, double, double*, int, int*)
                   10.01%  27.253ms     10000  2.7250us  2.6240us  3.2000us  kernel_synapses_stateupdater_codeobject(unsigned int, unsigned int, double*, int, double*, int, double*, int*)
                    9.18%  24.982ms     10000  2.4980us  2.2080us  2.6880us  kernel_poissongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*, double*, float*)
                    6.70%  18.244ms     10000  1.8240us  1.7280us  2.0800us  kernel_neurongroup_resetter_codeobject(unsigned int, unsigned int, double*, int*)
                    6.70%  18.236ms     10000  1.8230us  1.7280us  2.6240us  kernel_neurongroup_thresholder_codeobject(unsigned int, unsigned int, int*, double*)
                    5.76%  15.677ms     10000  1.5670us  1.4720us  1.6960us  _GLOBAL__N__69_tmpxft_00004642_00000000_6_neurongroup_thresholder_codeobject_cpp1_ii_c0b8948b::_reset_neurongroup_thresholder_codeobject(int*)
                    5.59%  15.219ms     10000  1.5210us  1.4400us  1.9520us  _GLOBAL__N__70_tmpxft_00004643_00000000_6_poissongroup_thresholder_codeobject_cpp1_ii_7314966e::_reset_poissongroup_thresholder_codeobject(int*)
                    0.12%  330.63us         1  330.63us  330.63us  330.63us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int))>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
      API calls:   85.78%  968.25ms     90001  10.758us  9.3110us  9.2828ms  cudaLaunch
                   10.41%  117.51ms    660005     178ns     137ns  367.56us  cudaSetupArgument
                    2.19%  24.694ms     90001     274ns     200ns  349.17us  cudaConfigureCall
                    1.59%  17.897ms     70002     255ns     203ns  333.24us  cudaGetLastError
                    0.02%  201.51us         1  201.51us  201.51us  201.51us  cudaMalloc
                    0.01%  131.77us         1  131.77us  131.77us  131.77us  cudaMemGetInfo
                    0.00%  51.691us        74     698ns     591ns  1.8080us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                    0.00%  47.398us        12  3.9490us  3.1850us  6.2570us  cudaFuncGetAttributes
                    0.00%  13.229us         1  13.229us  13.229us  13.229us  cudaDeviceSynchronize
                    0.00%  9.0230us        20     451ns     339ns     845ns  cudaDeviceGetAttribute
                    0.00%  4.9330us         5     986ns     852ns  1.4630us  cudaGetDevice

```

</p></details>


<details><summary>Examplary `nvprof` results for **GeNNConfigurationOptimized**</summary><p>
Profile summary for `N = 1000`:

```
==19043== NVPROF is profiling process 19043, command: ./main test 1.0 1
==19043== Profiling application: ./main test 1.0 1
==19043== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   39.69%  65.436ms     10000  6.5430us  1.5680us  25.984us  calcSynapses
                   24.07%  39.692ms     10000  3.9690us  3.1040us  6.4320us  calcNeurons
                   20.72%  34.165ms     10000  3.4160us  3.1040us  6.0800us  calcSynapseDynamics
                   15.42%  25.426ms     10000  2.5420us  2.3680us  6.6880us  learnSynapsesPost
                    0.06%  96.800us        72  1.3440us     960ns  2.0800us  [CUDA memcpy HtoD]
                    0.04%  59.552us        21  2.8350us  2.0480us  4.7680us  [CUDA memcpy DtoH]
      API calls:   51.75%  397.10ms     40000  9.9270us  9.2210us  345.12us  cudaLaunch
                   44.63%  342.53ms        21  16.311ms  16.914us  340.57ms  cudaHostAlloc
                    1.75%  13.449ms     40000     336ns     278ns  330.54us  cudaConfigureCall
                    1.40%  10.778ms     40000     269ns     210ns  335.21us  cudaSetupArgument
                    0.26%  1.9587ms        97  20.192us     407ns  40.743us  cudaMemcpy
                    0.13%  990.84us        21  47.182us  13.183us  232.13us  cudaMalloc
                    0.05%  400.17us        94  4.2570us     154ns  155.67us  cuDeviceGetAttribute
                    0.01%  113.90us         1  113.90us  113.90us  113.90us  cuDeviceTotalMem
                    0.00%  36.839us         1  36.839us  36.839us  36.839us  cuDeviceGetName
                    0.00%  30.866us        21  1.4690us     942ns  6.1960us  cudaGetSymbolAddress
                    0.00%  9.2900us         1  9.2900us  9.2900us  9.2900us  cudaSetDevice
                    0.00%  3.2500us         3  1.0830us     238ns  2.4920us  cuDeviceGetCount
                    0.00%  1.6970us         1  1.6970us  1.6970us  1.6970us  cudaGetDeviceCount
                    0.00%  1.0870us         2     543ns     238ns     849ns  cuDeviceGet

```

</p></details>

