try:
    from rpython_compat import *
    from star import *
except ImportError:
    from .rpython_compat import *
    from .star import *


@look_inside
def str_to_float(string):
    assert isinstance(string, str), "TypeError"
    return float(bytes2(string))

@look_inside
def str_repr(string):
    new = [u""]*len(string)
    for i, char in enumerate(string):
        if char == u"\a":
            new[i] = u"\\t"
        elif char == u"\b":
            new[i] = u"\\b"
        elif char == u"\f":
            new[i] = u"\\f"
        elif char == u"\n":
            new[i] = u"\\n"
        elif char == u"\t":
            new[i] = u"\\t"
        elif char == u"\v":
            new[i] = u"\\v"
        elif char == u"\\":
            new[i] = u"\\\\"
        elif char == u'"':
            new[i] = u'\\"'
        else:
            new[i] = char
    return u'"' + u"".join(new) + u'"'

@look_inside
def str_split(string, substring):
    assert isinstance(substring, str), "TypeError"
    assert isinstance(string, str), "TypeError"
    # RPython can't split python2-unicode but can split python2-str
    if PYTHON == 3:
        array = string.split(substring)
    else:
        array = bytes(string).split(bytes(substring))
    new = [u""] * len(array)
    for i in range(len(array)):
        new[i] = str(array[i])
    return new

@look_inside
def str_split_n(string, substring, n):
    assert isinstance(substring, str), "TypeError"
    assert isinstance(string, str), "TypeError"
    # RPython can't split python2-unicode but can split python2-str
    if PYTHON == 3:
        array = string.split(substring, n)
    else:
        array = bytes(string).split(bytes(substring), n)
    new = [u""] * len(array)
    for i in range(len(array)):
        new[i] = str(array[i])
    return new