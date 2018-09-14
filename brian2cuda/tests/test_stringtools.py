from nose.tools import eq_
from nose.plugins.attrib import attr

from brian2cuda.utils.stringtools import replace_floating_point_literals


@attr('codegen-independent')
def test_replace_floating_point_literals():
    float_literals = ['1.',
                      '.2',
                      '3.14',
                      '5e6',
                      '5e-6',
                      '5E+6',
                      '7.e8',
                      '9.0E-10',
                      '.11e12']

    others = ['-', '+', '/', '%', '*', '(', ')', ' ', ';', ':', '>', '<', '|',
              '&']

    for l in float_literals:
        for start in others:
            for end in others:
                string = start + l + end
                f_string = start + l + 'f' + end

                # test that floating point literals are correctly replaced by
                # single precision versions (`f` appended)
                replaced = replace_floating_point_literals(string)
                eq_(replaced, f_string)

                # test that single precision floating-point literals are not
                # touched (e.g `1.0f`)
                f_replaced = replace_floating_point_literals(f_string)
                eq_(f_replaced, f_string)

    not_float_literals = ['1', '100', '002', 'a1.b', '-.-']

    for l in not_float_literals:
        for start in others:
            for end in others:
                string = start + l + end

                # test that these these are not matched
                replaced = replace_floating_point_literals(string)
                eq_(replaced, string)


if __name__ == '__main__':
    test_replace_floating_point_literals()
