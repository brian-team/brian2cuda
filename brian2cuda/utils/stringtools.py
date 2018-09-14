'''
Brian2CUDA regex functions.
'''
import re

def append_f(match):
    string = match.group()
    if string.endswith('f'):
        # literal has already f
        return string
    f_string = string[:-1] + 'f' + string[-1]
    return f_string

def replace_floating_point_literals(code, single_precision=True):
    #regex = '[^A-Za-z_](((([1-9][0-9]*\.[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?)|([0-9]+([Ee][+-]?[0-9]+)))\D'#'(?!f)'
    # numbers are not part of variable (e.g. `a1.method()`)
    notV = '[^A-Za-z_]'
    # anything that has a digit before the dot (e.g. 2.1, 3. 1002.0293)
    preDot = '([1-9]\d*\.\d*)'
    # anything that starts with dot (e.g. .1 .09382)
    postDot = '(\.\d+)'
    # exponential syntax (e.g. e12, E-4, e+100)
    E = '([Ee][+-]?\d+)'
    # not digit (match the first not digit to check if it is `f` already)
    notD = '(\D)'
    regex = '{notV}((({preDot}|{postDot}){E}?)|(\d+{E})){notD}'.format(
        notV=notV, preDot=preDot, postDot=postDot, E=E, notD=notD)
    code = re.sub(regex, append_f, code)
    return code

