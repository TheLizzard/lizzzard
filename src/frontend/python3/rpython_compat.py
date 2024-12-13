try:
    from star import *
except ImportError:
    from .star import *


class JitDriverDummy(object):
    def __init__(self, greens, reds, get_printable_location):
        pass

    def jit_merge_point(self, **kwargs):
        pass

    def can_enter_jit(self, **kwargs):
        pass


if PYTHON == 2:
    try:
        from rpython.rlib.jit import JitDriver, elidable, promote, \
                                     promote_unicode, jit_debug, hint, \
                                     look_inside
        from rpython.rlib.objectmodel import specialize
        from rpython.rlib.nonconst import NonConstant
        never = lambda: NonConstant(False)
        noargtype = specialize.argtype
        NO_RPYTHON = False
    except ImportError:
        NO_RPYTHON = True
else:
    NO_RPYTHON = True


if NO_RPYTHON:
    noargtype = lambda n: (lambda f: f)
@noargtype(0)
def identity(f):
    return f
@noargtype(0)
def identity_kw(f, **kw):
    return f


if NO_RPYTHON:
    JitDriver = JitDriverDummy
    promote_unicode = identity
    elidable = identity
    promote = identity
    hint = identity_kw
    jit_debug = lambda s: None
    look_inside = identity
    never = lambda: None

const_str = promote_unicode
const = promote

USE_JIT = False
USE_JIT = True

# https://readthedocs.org/projects/ppc/downloads/pdf/latest/