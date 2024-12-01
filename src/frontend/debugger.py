from __future__ import print_function
from python3.star import *
from sys import stdout


def debug(text, level):
    assert isinstance(text, str), "TypeError"
    assert isinstance(level, int), "TypeError"
    if level > DEBUG_LEVEL: return
    print(u"\x1b[96m\r[DEBUG " + str(bytes2(level)) + u"]: " + u"\t"*entered + \
          text + u"\x1b[0m")

def warn(text):
    print(u"\x1b[93m\r[WARNING]: " + u"\t"*entered + text + u"\x1b[0m")

def error(text):
    print(u"\x1b[91m\r[ERROR]: " + u"\t"*entered + text + u"\x1b[0m")

def done_success():
    print(u"\x1b[92m[DONE]: " + u"\t"*entered + u"Success\x1b[0m")

def done_fail():
    print(u"\x1b[91m[DONE]: " + u"\t"*entered + u"Failure\x1b[0m")


entered = 0
DEBUG_LEVEL = 2