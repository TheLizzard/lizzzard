from .star import *


if PYTHON == 2:
    def int_to_bytes(integer, size):
        assert isinstance(integer, int), "TypeError"
        assert isinstance(size, int), "TypeError"
        assert 0 <= integer < (1<<(size*8)), "ValueError"
        assert size > 0, "ValueError"
        result = bytearray()
        for _ in range(size):
            result.append(integer & 0xFF)
            integer >>= 8
        result.reverse()
        return bytes2(result)
else:
    def int_to_bytes(integer, size):
        assert isinstance(integer, int), "TypeError"
        assert isinstance(size, int), "TypeError"
        assert 0 <= integer < (1<<(size*8)), "ValueError"
        assert size > 0, "ValueError"
        return integer.to_bytes(size, "big")


if PYTHON == 2:
    def int_from_bytes(data):
        result = 0
        for byte in data:
            result = (result << 8) + ord(byte)
        return result
else:
    def int_from_bytes(data):
        assert isinstance(data, bytes), "TypeError"
        assert len(data) > 0, "ValueError"
        return int.from_bytes(data, "big")


if PYTHON == 2:
    def int_to_str(data):
        assert isinstance(data, int), "TypeError"
        return bytes(data).decode("utf-8")
else:
    def int_to_str(data):
        assert isinstance(data, int), "TypeError"
        return str(data)