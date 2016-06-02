import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns # pretty plotting

# time measurements are roughly the averages of 10 kernel calls (from 10 time steps)
# from nvprof profiler

# using original implementation from master branch (commit dee9bf7)
N_orig = [10000, 5000, 3334, 2000, 1000, 200, 100, 10, 1]
orig = [15.6, 8.9, 6.6, 4.8, 3.5, 2.38, 2.25, 2.20, 2.20]
plt.plot(N_orig, orig, label='atomicAdds on global mem')

# using implementation from branch issue9_spikespace (commit eb215d4)
N_mod = [10000, 2000, 1000, 100, 10, 1]
modified = [43.5, 13.4, 8.3, 3.1, 2.4, 2.3]
plt.plot(N_mod, modified, label='atomicAdds on shared mem')

plt.xlabel('number of spiking neurons per time step (total $= 10000$)')
plt.ylabel('average time per thresholder kernel call ($\mu$s)')
plt.title('Performance of thresholder using global and shared memory atomics')
plt.legend(loc='best')
plt.savefig('performance_atomics.png')
