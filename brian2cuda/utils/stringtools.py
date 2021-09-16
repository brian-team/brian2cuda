'''
Brian2CUDA regex functions.
'''
import re
import os


def append_f(match):
    '''
    Append ``'f'`` to the string in ``match`` if it doesn't end with ``'f'``.
    Used in ``replace_floating_point_literals``.

    Parameters
    ----------
    match : re.MatchObject
        The return type of e.g. ``re.match`` or ``re.search``.

    Returns
    -------
    str
        The string returned from ``match.group()`` if it end with ``'f'``, else
        the string with ``'f'`` appended.
    '''
    string = match.group()
    if string.endswith('f'):
        # literal has already f
        return string
    f_string = string[:-1] + 'f' + string[-1]
    return f_string


def replace_floating_point_literals(code):
    '''
    Replace double-precision floating-point literals in ``code`` by
    single-precision literals.

    Parameters
    ----------
    code : str
        A string to replace the literals in. C++ syntax is assumed, s.t. e.g.
        `a1.b` would not be replaced.

    Returns
    -------
    str
        A copy of ``code``, with double-precision floating point literals
        replaced by single-precision flaoting-point literals (with an ``f``
        appended).

    Examples
    --------
    >>> replace_floating_point_literals('1.|.2=3.14:5e6>.5E-2<7.e-8==a1.b')
    1.f|.2f=3.14f:5e6f>.5E-2f<7.e-8f==a1.b

    '''
    # NOTE: this regex fails when `code` ends with a literal that should be
    # replaced. E.g `sldkfjwe11.e12` -> `sldkfjwe11.fe12`, because the regex
    # relies on the first non-digit after the literal, which is not existing at
    # the end of the string. If ``code`` is a valid c++ file, this can't happen
    # though.

    # NOTE: this regex replaces variables that end with two digits, e.g.
    # `var12.method()` or `file_10.h`. I didn't have time to dig into regex woodoo and
    # just ended up ignoring line starting with #include, which is where the error
    # popped up. But this should be fixed eventually. In principle this should be
    # possible with possesive matching in the literal match (meaning not giving away the
    # first digit in the examples above). But that is not supported in python's re
    # module. One could use the `regex` module instead, where the following might fix
    # this problem (double + is possesive match)
    #   postDot = '(\.\d++)'
    # I also tried this workaround I found on the web, but didn't work for me
    #   postDot = '(\.(?=(\d+))\1)'

    # numbers are not part of variable (e.g. `a1.method()`)
    # use negative lookbehind (?<!) in order to not consume the letter before
    # the match, s.t. consecutive literals don't overlap (otherwise only one
    # would be matched, e.g. `1.-.2;` -> `1.f-.2;` instead of `1.f-.2f;`)
    notVar = '(?<!([A-Za-z_]))'
    # anything that has a digit before the dot (e.g. 2.1, 3. 1002.0293)
    preDot = '([1-9]\d*\.\d*)'
    # anything that starts with dot (e.g. .1 .09382)
    postDot = '(\.\d+)'
    # exponential syntax (e.g. e12, E-4, e+100)
    Exp = '([Ee][+-]?\d+)'
    # not digit (match the first not digit to check if it is `f` already)
    notDigit = '([\D$])'
    regex = f'{notVar}((({preDot}|{postDot}){Exp}?)|(\d+{Exp})){notDigit}'
    # Remove lines starting with #include before regexing
    code, excluded_lines = _remove_lines(code, "#include")
    code = re.sub(regex, append_f, code)
    code = _reinsert_lines(code, excluded_lines)
    return code


def _remove_lines(code, *line_beginnings):
    """ Remove lines from `code` that start with `*line_beginnings` """
    excluded = []
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip(' ').startswith(line_beginnings):
            excluded.append((i, line))

    # Delete lines in reversed order to not mess up deletion index
    for i, _ in reversed(excluded):
        del lines[i]

    new_code = "\n".join(lines)
    return new_code, excluded


def _reinsert_lines(code, insert_lines):
    """ Reinsert previously removed lines into `code` """
    lines = code.split("\n")
    # Insert in forward order for indices to fit
    for i, line in insert_lines:
        lines.insert(i, line)
    new_code = "\n".join(lines)
    return new_code
