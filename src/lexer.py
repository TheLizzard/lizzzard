from __future__ import annotations
from io import TextIOBase
from enum import Enum


DEBUG_THROW:bool = False
DEBUG_READ:bool = False
USING_COLON:bool = True

class TokenType(Enum):
    IDENTIFIER:int = 1
    STRING:int = 2
    NUMBER:int = 3
    OTHER:int = 4
    EOF:int = 5
    INDENT:int = 6
    NEWLINE:int = 7


# "\t"   means start indentation of new block
# "\n"   means new line
# "\xff" means end the indentation of the block
#          if 2 lines have same indentation this isn't used
class Token:
    __slots__ = "token", "stamp", "type"

    def __init__(self, token:str, stamp:Stamp, type:TokenType) -> Token:
        assert isinstance(stamp, Stamp), "TypeError"
        assert isinstance(token, str|tuple), "TypeError"
        self.type:TokenType = type
        self.stamp:Stamp = stamp
        self.token:str = token

    def name_as(self, token:str) -> Token:
        return Token(token, self.stamp, self.type)

    def __eq__(self, other:str|Token) -> bool:
        if isinstance(other, str):
            return self.token == other
        elif isinstance(other, Token):
            return self.token == other.token
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.token)

    def isidentifier(self) -> bool:
        return self.type is TokenType.IDENTIFIER

    def isnumber(self) -> bool:
        return self.type is TokenType.NUMBER

    def isint(self) -> bool:
        return self.isnumber() and self.token.isnumeric()

    def isstring(self) -> bool:
        return self.type is TokenType.STRING

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        if self.isstring():
            return self.token[0]+repr(self.token[1:])
        else:
            return self.token


class Stamp:
    __slots__ = "line", "char", "line_string"

    def __init__(self, line:int, char:int, line_string:str) -> Stamp:
        self.line_string:str = line_string
        self.char:str = char
        self.line:int = line

        assert self.line_string


class FinishedWithError(SyntaxError): ...


class Buffer:
    __slots__ = "under", "buffer", "ran_out", "line", "char", "line_string"

    def __init__(self, buffer:TextIOBase) -> Buffer:
        self.under:TextIOBase = buffer
        self.ran_out:bool = False
        self.line_string:str = ""
        self.buffer:str = ""
        self.line:int = 1
        self.char:int = 0
        self.peek(1)

    def __bool__(self) -> bool:
        return (not self.ran_out) or bool(self.buffer)

    def throw(self, msg:str, *, stamp:Stamp=None) -> None:
        if stamp is None:
            stamp:Stamp = self.stamp()
        new_msg:str = msg + f" on line {stamp.line} character {stamp.char}\n"
        new_msg += stamp.line_string
        new_msg += " "*stamp.char + "^"
        if DEBUG_THROW:
            raise SyntaxError(new_msg)
        else:
            raise FinishedWithError(new_msg) from None

    def read(self, size:int, error_msg:str=None) -> str:
        self.peek(size)
        output, self.buffer = self.buffer[:size], self.buffer[size:]
        if (error_msg is not None) and (len(output) != size):
            self.throw(error_msg)
        self._update_line_char(output)
        return output

    def peek(self, size:int) -> str:
        if (len(self.buffer) < size) and (not self.ran_out):
            while True:
                char:str = self.under.read(1)
                self.buffer += char
                if char == "\n":
                    break
                elif not char:
                    self.ran_out:bool = True
                    break
        return self.buffer[:size]

    def stamp(self) -> Stamp:
        line_string:str = self.line_string + self.buffer.split("\n", 1)[0]
        return Stamp(self.line, self.char, line_string+"\n")

    def _update_line_char(self, text:str) -> None:
        self.line += text.count("\n")
        if "\n" in text:
            self.char:int = len(text) - text.rfind("\n") - 1
        else:
            self.char += len(text)
        self.line_string += text
        self.line_string:str = self.line_string.rsplit("\n")[-1]

    def read_empty_lines(self, comment_start:str, ignore:str) -> None:
        while self:
            while self.buffer.startswith("\n"):
                self.read(1)
            chunk:str = self.buffer.split("\n", 1)[0]
            if not chunk:
                self.peek(1)
                continue
            if (set(chunk.split(comment_start, 1)[0]) - set(ignore+"\n")):
                return None
            self.read(len(chunk))


Stack:type = list

class BoolContext:
    __slots__ = "stack"

    def __init__(self) -> BoolContext:
        self.stack:Stack[bool] = [False]

    def __bool__(self) -> bool:
        return self.stack[-1]

    def __enter__(self) -> BoolContext:
        self.stack.append(True)
        return self

    def __exit__(self, *args) -> bool:
        assert len(self.stack) > 1, "InternalError" # messed up stack
        assert self.stack.pop(), "InternalError" # messed up stack
        return False

    @property
    def inverse(self) -> _InverseBoolContext:
        return _InverseBoolContext(self)


class _InverseBoolContext(BoolContext):
    __slots__ = "original"

    def __init__(self, original:BoolContext) -> _InverseBoolContext:
        self.original:BoolContext = original
        self.stack:Stack[bool] = original.stack

    def __enter__(self) -> _InverseBoolContext:
        self.original.stack.append(False)
        return self

    def __exit__(self, *args) -> bool:
        assert len(self.original.stack) > 1, "InternalError" # messed up stack
        assert not self.original.stack.pop(), "InternalError" # messed up stack
        return False

    @property
    def inverse(self) -> BoolContext:
        return self.original


IndentStack:type = list[str]

class Tokeniser:
    __slots__ = "buffer", "indentation", "token_buffer", "freeze_indentation"

    def __init__(self, buffer:TextIOBase) -> Tokeniser:
        self.freeze_indentation:BoolContext = BoolContext()
        self.buffer:Buffer = Buffer(buffer)
        self.token_buffer:list[Token] = []
        self.indentation:IndentStack = []

        while self.buffer.peek(1) == "\n":
            self.buffer.read(1)
        if (self.buffer.peek(1) in list("\t ")) and USING_COLON:
            # We shouldn't start the file with regex["\n*\t"]
            #   so we leave it to the parser to kill us by
            #   starting with \t
            self.token_buffer.append(Token("\t", self.buffer.stamp(),
                                           TokenType.INDENT))

    def throw(self, msg:str, stamp_or_stamp:Stamp=None) -> None:
        if stamp_or_stamp is None:
            stamp:Stamp = self.buffer.stamp()
        elif isinstance(stamp_or_stamp, Token):
            stamp:Stamp = stamp_or_stamp.stamp
        elif isinstance(stamp_or_stamp, Stamp):
            stamp:Stamp = stamp_or_stamp
        else:
            raise TypeError("stamp_or_stamp must be of type Stamp|Token|None")
        self.buffer.throw(msg, stamp=stamp)

    def __bool__(self) -> bool:
        if self.buffer:
            return True
        if len(self.token_buffer) == 0:
            return False
        if set(self.token_buffer) != {"\n"}:
            return True
        return False

    def _eat_characters(self, chars:tuple[str]|str, invert:bool=False) -> None:
        if isinstance(chars, str):
            chars:tuple[str] = tuple(chars)
        while (self.buffer.peek(1) in chars) ^ invert:
            if not self.buffer.read(1):
                break

    def _eat_comment(self) -> None:
        assert self.buffer.read(1) == "#", "No comment to eat?"
        self._eat_characters("\r\n", invert=True)

    def peek(self, n:int=None) -> Token|list[Token]:
        while len(self.token_buffer) < (n or 1):
            self.token_buffer.append(self._read())
        if n is None:
            return self.token_buffer[0]
        else:
            return self.token_buffer[:n]

    def read(self, n:int|None=None) -> Token|list[Token]:
        self.peek(n or 1)
        if n is None:
            if DEBUG_READ: print(f"[DEBUG]: Read {self.token_buffer[0]!r}")
            return self.token_buffer.pop(0)
        output, self.token_buffer = self.token_buffer[:n], self.token_buffer[n:]
        if DEBUG_READ: print(f"[DEBUG]: Read {output!r}")
        return output
    read_token = read

    def remove_token_from_queue(self, token:Token) -> None:
        for i, t in enumerate(self.token_buffer):
            if t is token:
                self.token_buffer.pop(i)
                return
        self.throw("InternalError: remove_token_from_queue", token)

    def _read(self) -> Token:
        while True:
            stamp:Stamp = self.buffer.stamp()
            self._eat_characters(" \t")
            stamp:Stamp = self.buffer.stamp()
            token:str = self.buffer.peek(1)
            if not token:
                if self.indentation:
                    self.add_dentations_eof()
                return Token("\n", stamp, TokenType.EOF)
            elif token.isidentifier():
                ident:Token = self.read_identifier()
                if self.buffer.peek(1) in ("'", '"'):
                    return self.read_string(ident.token, type_stamp=stamp)
                else:
                    return ident
            elif token in "0123456789":
                return self.read_number()
            elif token in "\"'":
                return self.read_string("", type_stamp=stamp)
            elif token == "#":
                self._eat_comment()
                continue
            elif token in "$:[](){},.\t;":
                return Token(self.buffer.read(1), stamp, TokenType.OTHER)
            elif token in "+-*^%|<>=":
                self.buffer.read(1)
                if token == self.buffer.peek(1) == "*":
                    token += self.buffer.read(1)
                if token + self.buffer.peek(1) in ("--", "++", "->"):
                    token += self.buffer.read(1)
                elif self.buffer.peek(1) == "=":
                    token += self.buffer.read(1)
                return Token(token, stamp, TokenType.OTHER)
            elif token in "/":
                self.buffer.read(1)
                if self.buffer.peek(1) == "/":
                    token += self.buffer.read(1)
                if self.buffer.peek(1) == "=":
                    token += self.buffer.read(1)
                return Token(token, stamp, TokenType.OTHER)
            elif self.buffer.peek(2) == "!=":
                return Token(self.buffer.read(2), stamp, TokenType.OTHER)
            elif token == "\n":
                if self.freeze_indentation:
                    self.buffer.read(1)
                else:
                    self.read_newline_into_buffer()
                continue
            else:
                self.throw(f"[LEXER] unknown {token=!r}", stamp)
            raise NotImplementedError("Unreachable")

    def read_newline_into_buffer(self) -> None:
        assert self.buffer.read(1) == "\n", "InternalError"
        # Cleanup empty newlines
        self.buffer.read_empty_lines(comment_start="#", ignore=" \t")
        # Create newline token
        stamp:Stamp = self.buffer.stamp()
        newline_token:Token = Token("\n", stamp, TokenType.NEWLINE)
        # not USING_COLON
        if not USING_COLON:
            self.token_buffer.append(newline_token)
            return
        # Dentation
        for i, indent in enumerate(self.indentation):
            if self.buffer.peek(len(indent)) == indent:
                self.buffer.read(len(indent))
            elif self.buffer.peek(1) in ("\t", " "):
                self.throw("[LEXER] IndentationError")
            else:
                self.indentation, dentation = self.indentation[:i], \
                                              self.indentation[i:]
                token:Token = Token("\xff", stamp, TokenType.INDENT)
                for _ in dentation:
                    self.token_buffer.append(token)
                self.token_buffer.append(newline_token)
                break
        else:
            self.token_buffer.append(newline_token)
            # Indentation
            new_indent:str = ""
            while self.buffer.peek(1) in ("\t", " "):
                new_indent += self.buffer.read(1)
            if new_indent:
                indent_token:Token = Token("\t", stamp, TokenType.INDENT)
                self.indentation.append(new_indent)
                self.token_buffer.append(indent_token)

    def add_dentations_eof(self) -> None:
        # Add \xff for each indentation in self.indentations
        self.token_buffer.extend(["\xff"]*len(self.indentation))
        self.indentation.clear()

    def read_identifier(self) -> Token:
        token:str = ""
        stamp:Stamp = self.buffer.stamp()
        while True:
            token += self.buffer.read(1)
            ntoken:str = token + self.buffer.peek(1)
            if not ntoken.isidentifier():
                break
            if token == ntoken:
                break
        return Token(token, stamp, TokenType.IDENTIFIER)

    def read_number(self) -> Token:
        stamp:Stamp = self.buffer.stamp()
        output:str = self._parse_int()
        if self.buffer.peek(1) in ("x", "b"):
            _type:str = self.buffer.read(1)
            allowed:str = "01" if _type == "b" else "0123456789abcdef"
            base:int = 2 if _type == "b" else 16
            if output != "0":
                char:str = self.buffer.peek(1)
                if char and (char in allowed):
                    self.throw("[LEXER] Invalid binary integer literal", stamp)
                if char != "\n":
                    self.buffer.read(1)
                    stamp:Stamp = self.buffer.stamp()
                self.throw("[LEXER] Unexpected token after integer", stamp)
            output:str = ""
            while True:
                char:str = self.buffer.peek(1)
                if (not char) or (char not in allowed):
                    break
                output += self.buffer.read(1)
            return str(int(output, base=base))
        else:
            if self.buffer.peek(1) == ".":
                self.buffer.read(1)
                chunk:str = self._parse_int()
                output += "." + chunk
            if self.buffer.peek(1) == "e":
                self.buffer.read(1)
                chunk:str = self._parse_int()
                output += "e" + chunk
        return Token(output, stamp, TokenType.NUMBER)

    def _parse_int(self) -> str:
        stamp:Stamp = self.buffer.stamp()
        if self.buffer.peek(1) == "-":
            neg:str = self.buffer.read(1)
        else:
            neg:str = ""
        output:str = ""
        while True:
            char:str = self.buffer.peek(1)
            if not char:
                break
            if char not in "_0123456789":
                break
            else:
                output += char
                self.buffer.read(1)
        if (not output) or ("_" in output[0]+output[-1]):
            self.throw("[LEXER] Couldn't parse number literal", stamp)
        return neg+output.replace("_", "")

    def read_string(self, _type:str, type_stamp:Stamp) -> Token:
        stamp:Stamp = self.buffer.stamp()
        if len(set(_type)) != len(_type):
            self.throw(f"[LEXER] Invalid string prefix {_type!r}", type_stamp)
        if _type.strip("rfb") != "":
            self.throw(f"[LEXER] Invalid string prefix {_type!r}", type_stamp)
        _type:int = 1*("r" in _type) + 2*("f" in _type) + 4*("b" in _type)
        quote_type:str = self.buffer.read(1, "InteralError")
        multiline:bool = (self.buffer.peek(2) == quote_type*2)
        if multiline:
            self.buffer.read(2)
        output:str = ""
        while True:
            char:str = self.buffer.read(1, EOF_ERROR)
            if (char == "\n") and (not multiline):
                self.throw(EOL_ERROR, stamp)
            elif (char == "\\") and (_type & 1):
                if self.buffer.peek(1) in ("\n", quote_type):
                    char:str = self.buffer.read(1)
            elif char == "\\":
                char:str = self.buffer.read(1, EOF_ERROR)
                if char in STRING_ESCAPES:
                    char:str = STRING_ESCAPES[char]
                elif char in "xuU":
                    err_stamp:Stamp = self.buffer.stamp()
                    if char == "x":
                        data:str = self.buffer.read(2, EOF_ERROR)
                    elif char == "u":
                        data:str = self.buffer.read(4, EOF_ERROR)
                    elif char == "U":
                        data:str = self.buffer.read(8, EOF_ERROR)
                    try:
                        char:str = chr(self._hex_to_int(data))
                    except ValueError:
                        self.throw("[LEXER] Couldn't parse string literal " \
                                   "escape sequence", err_stamp)
            elif char == quote_type:
                if multiline:
                    if self.buffer.peek(2) == quote_type*2:
                        self.buffer.read(2)
                        break
                else:
                    break
            output += char
        return Token(str(_type)+output, stamp, TokenType.STRING)

    @staticmethod
    def _hex_to_int(string:str) -> int:
        output:int = 0
        for i, char in enumerate(string):
            output *= 16
            if char in "0123456789":
                output += ord(char)-48
            elif char.lower() in "abcdefg":
                output += ord(char)-87
            else:
                raise ValueError()
        return output


STRING_ESCAPES:dict[str:str] = {
                                 '"': '"',
                                 "'": "'",
                                 "\\": "\\",
                                 "a": "\a",
                                 "b": "\b",
                                 "f": "\f",
                                 "n": "\n",
                                 "r": "\r",
                                 "t": "\t",
                                 "v": "\v",
                               }
EOL_ERROR:str = "[LEXER] EOL while parsing string literal"
EOF_ERROR:str = "[LEXER] EOF while parsing string literal"