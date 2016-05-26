### Reproduce time measurements of atomicAdds on global and shared memory in thresholder kernel.

clock_t() is used to determine clock cycles spent to execute code block in kernel. 
For each kernel call the clock cycles per threads are printed and the total time for all threads and the average time per thread for the given block.
Time is calculated using:
```
(number of clock cycles) / (clock frequency)
```
The clock frequency is hardcoded to 980 kHz! Check compatibality with your device!

To get the measurements of the unmodified version (spikespace filled sequentially using global atomicAdd per thread), run within the `global_atomicAdds/` folder:

```
make all
./main
```

To get the measurements of the modfied version (spikespace filled parallel between block using atomicAdd on shared memory and then one global atomicAdd per block), run within the `shared_atomicAdds/` folder:
```
make all
./main
```


