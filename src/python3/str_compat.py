from __future__ import annotations

def str_encode(string:str) -> bytes:
    assert isinstance(string, str), "TypeError"
    return string.encode("utf-8")