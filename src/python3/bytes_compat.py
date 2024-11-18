from __future__ import annotations

def bytes_startswith(longer:bytes, shorter:bytes) -> bool:
    assert isinstance(shorter, bytes), "TypeError"
    assert isinstance(longer, bytes), "TypeError"
    if len(shorter) > len(longer):
        return False
    return longer[0:len(shorter)] == shorter

def bytes_decode(data:bytes) -> str:
    assert isinstance(data, bytes), "TypeError"
    return data.decode("utf-8")

def bytes_join_empty(iterable:Iterable[bytes]) -> bytes:
    return b"".join(iterable)