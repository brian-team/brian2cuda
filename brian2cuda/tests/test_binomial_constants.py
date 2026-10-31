import re

import pytest

from brian2.input.binomial import _pre_calc_constants

from brian2cuda.binomial import _generate_cuda_code


_FLOAT_LITERAL = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def _generated_inversion_constants(n, p):
    support_code = _generate_cuda_code(
        n=n, p=p, use_normal=False, name='_binomial_test'
    )[0]['support_code']

    patterns = {
        'P': rf'\* ({_FLOAT_LITERAL}) \* px',
        'q': rf'X\*({_FLOAT_LITERAL})\)',
        'qn': rf'\bpx = ({_FLOAT_LITERAL});',
        'bound': rf'X > ({_FLOAT_LITERAL})',
    }
    return {
        name: float(re.search(pattern, support_code).group(1))
        for name, pattern in patterns.items()
    }


@pytest.mark.parametrize(
    ('n', 'p'),
    [
        (50, 0.5),
        (51, 0.5),
        (100, 0.3),
        (100, 0.1),
        (10, 0.8),
    ],
)
def test_binomial_inversion_constants_preserve_values(n, p):
    _, q, P, qn, bound = _pre_calc_constants(n, p)

    generated = _generated_inversion_constants(n, p)

    assert generated['P'] == P
    assert generated['q'] == q
    assert generated['qn'] == qn
    assert generated['bound'] == bound


@pytest.mark.parametrize(
    ('n', 'p'),
    [
        (51, 0.5),
        (100, 0.3),
    ],
)
def test_binomial_inversion_qn_remains_nonzero(n, p):
    generated = _generated_inversion_constants(n, p)

    assert generated['qn'] > 0
