import pytest

from brian2cuda.utils.stringtools import (
    replace_floating_point_literals, _remove_lines, _reinsert_lines
)

def eq_(a, b):
    assert a == b, f"{repr(a)} != {repr(b)}"

@pytest.mark.codegen_independent
def test_replace_floating_point_literals():
    float_literals = ['1.', '.2', '3.14', '5e6', '5e-6', '5E+6', '7.e8',
                      '9.0E-10', '.11e12']

    delimiters = ['-', '+', '/', '%', '*', '(', ')', ' ', ';', ':', '>', '<',
                  '|', '&', ',', '=']

    for l in float_literals:
        for start in delimiters:
            for end in delimiters:
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

    # Couldn't find a way to fix this. Instead ignoring #include lines in code when
    # appending `f` (which was creating a bug in `#include name_11.h` files
    not_delimiters = ['_', 'a']
    #for l in float_literals:
    #    for d in not_delimiters:

    #        for string in [d + l, d + l + d, l + d]:
    #            replaced = replace_floating_point_literals(string)
    #            eq_(replaced, string)

    not_float_literals = ['1', '100', '002', 'a1.b', '-.-']

    for l in not_float_literals:
        for start in delimiters:
            for end in delimiters:
                string = start + l + end

                # test that these these are not matched
                replaced = replace_floating_point_literals(string)
                eq_(replaced, string)

    # test that multiple concatenated literals are correctly replaced
    concat = ''.join(a + b for a, b in zip(float_literals,
                                           delimiters[:len(float_literals)]))
    f_concat = ''.join(a + 'f' + b for a, b in zip(float_literals,
                                                   delimiters[:len(float_literals)]))

    # replacing double to single precision version
    replaced = replace_floating_point_literals(concat)
    eq_(replaced, f_concat)

    # not replacing sincle precision version
    f_replaced = replace_floating_point_literals(f_concat)
    eq_(f_replaced, f_concat)

    code_with_include = f'''
#include ignore this double {concat}
{concat}
    #include how about an indented hashtag? ignore this: {concat}
# don't ignore this: {concat}
'''

    solution = f'''
#include ignore this double {concat}
{f_concat}
    #include how about an indented hashtag? ignore this: {concat}
# don't ignore this: {f_concat}
'''
    code_replaced = replace_floating_point_literals(code_with_include)
    eq_(code_replaced, solution)


@pytest.mark.xfail(
    reason='See Brian2CUDA issue #254',
    raises=AssertionError
)
@pytest.mark.standalone_only
@pytest.mark.cuda_standalone
def test_regex_bug():
    strings = ["var12.method()", "file_10.h"]
    for string in strings:
        replaced = replace_floating_point_literals(string)
        eq_(replaced, string)


if __name__ == '__main__':
    test_replace_floating_point_literals()
    test_regex_bug()
