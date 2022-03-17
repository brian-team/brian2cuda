These Python scripts specify the benchmarks run for each figure in Alevi
et~al.~2022. To execute any of these simulations, copy the Python file to
`brian2cuda/tools/benchmarking/run_benchmark_suite.py` and execute
`brian2cuda/tools/benchmarking/run_benchmark_suite.sh`. E.g. to run the figure 7 benchmarks, execute:
```
cp /path/to/brian2cuda/tools/benchmarking/paper_2022/run_benchmarks_fig7.py /path/to/brian2cuda/tools/benchmarking/run_benchmark_suite.py
cd /path/to/brian2cuda/tools/benchmarking
bash run_benchmark_suite.sh --name figure7-benchmarks
```

To run all simulations for all simulations for all paper figures, you can run
```
cd /path/to/brian2cuda/tools/benchmarking/paper_2022
bash run_all.sh
```


