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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda_standalone: to be used with standalone_only marker"
    )
    config.addinivalue_line(
        "markers",
        "compilation_only: codegen and compile only; does not execute GPU code",
    )
    if not hasattr(config, "device"):
        config.device = "cuda_standalone"
        config.device_options = {"build_on_run": False, "directory": None}
        config.brian_prefs = {}
        config.fail_for_not_implemented = False
