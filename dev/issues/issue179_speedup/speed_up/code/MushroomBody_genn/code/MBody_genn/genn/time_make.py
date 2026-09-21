import time
import subprocess
import statistics
import os

times_list_make = []
times_list_make_j = []
for i in range(5):
    print("In round {}".format(i))
    wd = os.getcwd()
    os.chdir(wd+"/magicnetwork_model_CODE/")
    subprocess.run("make clean", shell=True)
    os.chdir(wd)
    start = time.time()
    subprocess.run("make", shell=True)
    took = time.time() - start
    times_list_make.append(took)

    #for make -j command
    wd = os.getcwd()
    os.chdir(wd+"/magicnetwork_model_CODE/")
    subprocess.run("make clean", shell=True)
    os.chdir(wd)
    start = time.time()
    subprocess.run("make -j", shell=True)
    took = time.time() - start
    times_list_make_j.append(took)

print("Mean, standard deviation and variance of the make process:")
print(statistics.mean(times_list_make))
print(statistics.pstdev(times_list_make))
print(statistics.pvariance(times_list_make))

print("Mean, std dev and variance of the make -j process:")
print(statistics.mean(times_list_make_j))
print(statistics.pstdev(times_list_make_j))
print(statistics.pvariance(times_list_make_j))


