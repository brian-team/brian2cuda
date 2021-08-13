'''
Module containing fixtures and hooks used by the pytest test suite.
'''
# Use brian2's pytest configuration for the brian2cuda tests (see PR #232 for details)
from brian2.conftest import *

# Add a cuda implementation for the fake_randn_randn_fixture,
# used in test_stateupdaters.py
fake_randn.implementations.add_implementation(
    'cuda',
    '''
    __host__ __device__ double randn(int vectorisation_idx)
    {
        return 0.5;
    }
    '''
)


# Register `cuda_standalone` marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda_standalone: to be used with standalone_only marker"
    )
