from __future__ import annotations

def int_to_bytes(integer:int, size:int) -> bytes:
    assert isinstance(integer, int), "TypeError"
    assert isinstance(size, int), "TypeError"
    assert 0 <= integer < (1<<(size*8)), "ValueError"
    assert size > 0, "ValueError"
    return integer.to_bytes(size, "big")

def int_from_bytes(data:bytes) -> int:
    assert isinstance(data, bytes), "TypeError"
    assert len(data) > 0, "ValueError"
    return int.from_bytes(data, "big")