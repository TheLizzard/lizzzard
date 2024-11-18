from .star import *


class JitDriverDummy(object):
    def __init__(self, greens, reds):
        pass

    def jit_merge_point(self, **kwargs):
        pass

    def can_enter_jit(self, **kwargs):
        pass


if PYTHON == 2:
    try:
        from rpython.rlib.jit import JitDriver
    except ImportError:
        JitDriver = JitDriverDummy
else:
    JitDriver = JitDriverDummy

USE_JIT = True