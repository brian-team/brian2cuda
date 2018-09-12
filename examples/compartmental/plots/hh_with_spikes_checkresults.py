import numpy as np

cpp_res = np.load('hh_with_spikes_cpp_results.npz')

cuda_res = np.load('hh_with_spikes_cuda_results.npz')

for name in cpp_res.files:
    print('cpp_res[{}].shape = {}'.format(name, cpp_res[name].shape))
    #print('cpp_res[{}] = {}'.format(name, cpp_res[name]))

    print('cuda_res[{}].shape = {}'.format(name, cuda_res[name].shape))
    #print('cuda_res[{}] = {}'.format(name, cuda_res[name]))

    #print('cpp_res[{}] - cuda_res[{}] = {}'.format(name, name, cpp_res[name]-cuda_res[name]))