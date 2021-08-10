from brian2 import *

# The "random" values are always 0.5
@implementation('cuda',
                '''
                __device__ __global__ double randn(int vectorisation_idx)
                {
                    return 0.5;
                }
                ''')
@check_units(N=Unit(1), result=Unit(1))
def fake_randn(N):
    return 0.5*ones(N)

# TODO check brian2 stateupdaters.py for tests not standalone-compatible?
