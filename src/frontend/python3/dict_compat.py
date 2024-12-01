from python3.rpython_compat import *
from python3.star import *


class dict:
    __slots__ = "under"

    def __init__(self):
        self.under = {}

    def get(self, key, default):
        assert isinstance(key, str), "TypeError"
        return self.under.get(key, default)

    def copy(self):
        ret = dict()
        ret.under = self.under.copy()
        return ret

    def __getitem__(self, key):
        assert isinstance(key, str), "TypeError"
        return self.under[key]

    def __setitem__(self, key, value):
        assert isinstance(key, str), "TypeError"
        self.under[key] = value

    def __contains__(self, key):
        assert isinstance(key, str), "TypeError"
        return key in self.under

    def pop(self, key, default):
        assert isinstance(key, str), "TypeError"
        return self.under.pop(key, default)

    def __repr__(self):
        return "Python3Dict" + repr(self.under)