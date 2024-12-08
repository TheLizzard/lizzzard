from python3.rpython_compat import *
from python3.star import *


_DICT_LENGTH = 256


class Dict:
    _immutable_fields_ = ["buckets"]
    __slots__ = "buckets"

    def __init__(self):
        self.buckets = [[] for i in range(_DICT_LENGTH)]

    @look_inside
    @elidable
    def get(self, key, default):
        for k, v in self.buckets[self._hash(key)]:
            if k == key:
                return v
        return default

    @look_inside
    @elidable
    def copy(self):
        ret = Dict()
        for i, bucket in enumerate(self.buckets):
            for key, value in bucket:
                ret.buckets[i].append((key, value))
        return ret

    @look_inside
    def __setitem__(self, key, value):
        arr = self.buckets[self._hash(key)]
        for i in range(len(arr)):
            if arr[i][0] == key:
                arr[i] = (key, value)
                return
        arr.append((key, value))

    @look_inside
    @elidable
    def __contains__(self, key):
        for k, v in self.buckets[self._hash(key)]:
            if k == key:
                return True
        return False

    @look_inside
    @elidable
    def _hash(self, key):
        assert isinstance(key, str), "TypeError"
        h = 0
        for char in key:
            h = h ^ ord(char)
        return h % _DICT_LENGTH

    def __repr__(self):
        return "Dict"


""" # Tried a functional approach to get rpython to constant fold - didn't work
@look_inside
def new_dict():
    return [const([]) for i in range(_DICT_LENGTH)]

@look_inside
@elidable
def dict_get(self, key, default):
    for k, v in self[_dict_hash(key)]:
        if k == key:
            return v
    return default

@look_inside
@elidable
def dict_copy(self):
    ret = new_dict()
    for i, bucket in enumerate(self):
        for key, value in bucket:
            ret[i].append((key, value))
    return ret

@look_inside
def dict_set(self, key, value):
    arr = self[_dict_hash(key)]
    for i in range(len(arr)):
        if arr[i][0] == key:
            arr[i] = (key, value)
            return
    arr.append((key, value))

@look_inside
@elidable
def dict_contains(self, key):
    for k, v in self[_dict_hash(key)]:
        if k == key:
            return True
    return False

@look_inside
@elidable
def _dict_hash(key):
    assert isinstance(key, str), "TypeError"
    h = 0
    for char in key:
        h = h ^ ord(char)
    return h % _DICT_LENGTH
# """