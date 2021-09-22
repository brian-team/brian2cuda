import time
import subprocess
import statistics

times_list = []
for _ in range(10):
    subprocess.run("make clean", shell=True)
    start = time.time()
    subprocess.run("make", shell=True)
    took = time.time() - start
    times_list.append(took)

print("Mean, standard deviation and variance of the population:")
print(statistics.mean(times_list))
print(statistics.pstdev(times_list))
print(statistics.pvariance(times_list))
