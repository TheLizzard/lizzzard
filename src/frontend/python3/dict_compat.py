try:
    from rpython_compat import *
    from star import *
except ImportError:
    from .rpython_compat import *
    from .star import *


_DICT_LENGTH = 256


class Dict:
    _immutable_fields_ = ["buckets"]
    __slots__ = "buckets"

    def __init__(self):
        self.buckets = [[] for i in range(_DICT_LENGTH)]

    # This shouldn't really be elidable when (not ENV_IS_LIST)
    #   but it makes it 10x faster when (ENV_IS_LIST)
    @look_inside
    @elidable
    def get(self, key, default):
        for k, v in self.buckets[self._hash(key)]:
            if k == key:
                return v
        return default

    @look_inside
    def __getitem__(self, key): # Returns None if KeyError
        return self.get(key, None)

    @look_inside
    def copy(self):
        ret = Dict()
        for i, bucket in enumerate(self.buckets):
            ret.buckets[i].extend(bucket)
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
    def _hash(self, key):
        assert isinstance(key, str), "TypeError"
        h = 0
        for char in key:
            h = h ^ ord(char)
        return h % _DICT_LENGTH

    def __repr__(self):
        return "Dict"


if __name__ == "__main__":
    a = Dict()
    a["5"] = 20
    b = a.copy()
    print(b.get("5", None))
    print(b.get("4", None))


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