'''
Module containing fixtures and hooks used by the pytest test suite.
'''
# We set `--rootidr=/path/to/brian`, such that `brian2/conftest.py` is loaded for all
# tests run in the test suite. This `conftest.py` will only be loaded for all tests in
# this directory

from brian2.conftest import fake_randn, fake_randn_randn_fixture

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
