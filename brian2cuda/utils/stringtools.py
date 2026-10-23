'''
Brian2CUDA regex functions.
'''
import re


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
        The string returned from ``match.group()`` if it ends with ``'f'``, else
        the string with ``'f'`` appended.
    '''
    string = match.group()
    if string.endswith('f'):
        return string
    return string + 'f'


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
        replaced by single-precision floating-point literals (with an ``f``
        appended).

    Examples
    --------
    >>> replace_floating_point_literals('1.|.2=3.14:5e6>.5E-2<7.e-8==a1.b')
    1.f|.2f=3.14f:5e6f>.5E-2f<7.e-8f==a1.b

    '''
    # Require a non-identifier character before/after the literal so we do not
    # match inside names like `var12.method()`, `file_11.h`, or UUID fragments
    # that look like scientific notation (e.g. `81e3` in `...d81e3_...`).
    # Use lookaround (not consuming the delimiter) so consecutive literals and
    # end-of-string literals are handled correctly.
    not_ident = r'(?<![A-Za-z0-9_])'
    # Digit(s) before the dot, including leading zeros (e.g. 0.5, 2.1, 1002.)
    pre_dot = r'(\d+\.\d*)'
    # Anything that starts with a dot (e.g. .1, .09382)
    post_dot = r'(\.\d+)'
    # Exponential syntax (e.g. e12, E-4, e+100)
    exp = r'([Ee][+-]?\d+)'
    not_ident_after = r'(?![A-Za-z0-9_])'
    regex = (
        f'{not_ident}((({pre_dot}|{post_dot}){exp}?)|(\\d+{exp})){not_ident_after}'
    )
    # Keep skipping #include lines as a safety net for header paths
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
