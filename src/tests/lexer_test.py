from __future__ import annotations
from io import StringIO
import sys
import os

ROOT:str = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from lexer import Tokeniser


TESTS = (
          # Single token tests
          (r"0", "0", ""),
          (r"0b00110", "6", ""),
          (r"0b1", "1", ""),
          (r"0x1", "1", ""),
          (r"0x10", "16", ""),
          (r"0.5e-001a", "0.5e-001", "a"),
          (r"abc_01=", "abc_01", "="),
          (r"== 10", "==", " 10"),
          (r"//= 5", "//=", " 5"),
          (r"'hello \x00world'+", '0hello \x00world', "+"),
          (r'"hello \u0123world"+', '0hello \u0123world', "+"),
        )
FULL_TESTS = (
                # Indentation/De-indentation tests
                ("a\n b\n c\n d", "a\n\tb\nc\nd\xff"),
                ("a\n b\n  c\n d", "a\n\tb\n\tc\xff\nd\xff"),
                ("a\n b\n  c\n d\n", "a\n\tb\n\tc\xff\nd\xff"),
                ("a\n\tb\n\t c\n", "a\n\tb\n\tc\xff\xff"),
                (r"5 ** 10", ("5", "**", "10")),
                ("x\n\t#c", "x"),
                ("x\n\t#c\ny", "x\ny"),
                ("...", ["..."])
             )


if __name__ == "__main__":
    for code, expected, expected_rest in TESTS:
        t:Tokeniser = Tokeniser(StringIO(code))
        got:str = t.read()
        rest:str = t.under.read(999)
        assert got == expected, f"Test error in {code!r}, read {got!r} token"
        assert rest == expected_rest, f"Test error in {code!r}"


    for code, expected in FULL_TESTS:
        t:Tokeniser = Tokeniser(StringIO(code))
        for exp in expected:
            assert t.read() == exp, f"Test error in {code!r}"
        for i in range(10):
            assert t.read() == "\n", "Should be empty"
    print("\x1b[92m[TEST]: All tests passed\x1b[0m")


"""

code:str = '''
a\n b\n c\n d
'''[1:-1]

t:Tokeniser = Tokeniser(StringIO(code))
# print(repr("".join(map(repr, t.read(len(expected))))), repr(expected),
#       sep="\n")
for i in range(10):
    tok = t.read()
    print([tok, t.indentation, bool(t), t.buffer, t.under.peek(999)])


# """