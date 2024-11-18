from __future__ import annotations
from enum import Enum, auto


DEBUG_THROW:bool = False
USING_COLON:bool = True
DEBUG_READ:bool = False


class TokenType(Enum):
    IDENTIFIER:int = auto()
    STRING:int = auto()
    OTHER:int = auto()
    FLOAT:int = auto()
    INT:int = auto()


class Token:
    __slots__ = "token", "stamp", "type"

    def __init__(self, token:str, stamp:Stamp, type:TokenType) -> Token:
        assert isinstance(type, TokenType), "TypeError"
        assert isinstance(stamp, Stamp), "TypeError"
        assert isinstance(token, str), "TypeError"
        assert len(token) > 0, "ValueError"
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

    def isint(self) -> bool:
        return self.type is TokenType.INT

    def isfloat(self) -> bool:
        return self.type is TokenType.FLOAT

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
        assert isinstance(line_string, str), "TypeError"
        assert isinstance(line, int), "TypeError"
        assert isinstance(char, int), "TypeError"
        self.line_string:str = line_string
        self.char:str = char
        self.line:int = line


class FinishedWithError(SyntaxError): ...


class PreLexer:
    __slots__ = "under", "buffer", "ran_out", "line", "char", "line_string"

    def __init__(self, buffer:TextIOBase) -> PreLexer:
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


class Tokeniser:
    __slots__ = "under", "indentation", "freeze_indentation", "ran_out", \
                "buffer"

    def __init__(self, buffer:TextIOBase) -> Tokeniser:
        self.freeze_indentation:BoolContext = BoolContext()
        self.under:PreLexer = PreLexer(buffer)
        self.indentation:Stack[str] = []
        self.buffer:list[Token] = []
        self.ran_out:bool = False
        self.read_newline_into_buffer(first=True)

    def _throw(self, msg:str, stamp:Stamp=None) -> None:
        stamp:Stamp = stamp or self.under.stamp()
        assert isinstance(stamp, Stamp), "TypeError"
        assert isinstance(msg, str), "TypeError"
        self.under.throw(msg, stamp=stamp)

    def throw(self, msg:str, token:Token=None) -> None:
        assert isinstance(token, Token|None), "TypeError"
        assert isinstance(msg, str), "TypeError"
        stamp:Stamp = self.under.stamp() if (token is None) else token.stamp
        self._throw(msg, stamp=stamp)

    def __bool__(self) -> bool:
        return (not self.ran_out) or bool(self.buffer)

    def remove_token_from_queue(self, token:Token) -> None:
        for i, t in enumerate(self.buffer):
            if t is token:
                self.buffer.pop(i)
                return
        raise RuntimeError("InternalError caused by caller")

    def _eat_characters(self, chars:str, invert:bool=False) -> None:
        assert isinstance(invert, bool), "TypeError"
        assert isinstance(chars, str), "TypeError"
        while (self.under.peek(1) in chars) ^ invert:
            if not self.under.read(1):
                break

    def _eat_comment(self) -> None:
        assert self.under.read(1) == "#", "No comment to eat?"
        self._eat_characters("\r\n", invert=True)

    def prepend_token(self, token:str) -> None:
        assert isinstance(token, str), "TypeError"
        self.buffer.insert(0, Token("\n", self.under.stamp(), TokenType.OTHER))

    def _get_first_or_newline(self, tokens:list[Token]) -> Token:
        assert isinstance(tokens, list), "TypeError"
        if len(tokens) == 0:
            return Token("\n", self.under.stamp(), TokenType.OTHER)
        return tokens[0]

    def peek(self) -> Token:
        return self._get_first_or_newline(self.peek_n(1))

    def read(self) -> Token:
        return self._get_first_or_newline(self.read_n(1))

    def peek_n(self, n:int) -> list[Token]:
        assert isinstance(n, int), "TypeError"
        while (len(self.buffer) < n) and (not self.ran_out):
            self.buffer.extend(self._read())
            # for i in range(100):
            #     tokens:list[Token] = self._read()
            #     self.buffer.extend(tokens)
            #     if tokens: break
            # else:
            #     raise RuntimeError("._read() didn't do anything 100 " \
            #                        "times in a row")
        return self.buffer[:n]

    def read_n(self, n:int) -> list[Token]:
        self.peek_n(n)
        output, self.buffer = self.buffer[:n], self.buffer[n:]
        if DEBUG_READ: print(f"[DEBUG]: Read {output!r}")
        return output

    def _read(self) -> list[Token]:
        stamp:Stamp = self.under.stamp()
        if not self:
            assert self.under.read(1) == "", "InternalError"
            return [Token("\n", stamp, TokenType.OTHER)]
        token:str = self.under.peek(1)
        ret:Token = None
        if token == "":
            self.ran_out, n = True, len(self.indentation)
            self.indentation.clear()
            return [Token("\xff", stamp, TokenType.OTHER)]*n + \
                   [Token("\n", stamp, TokenType.OTHER)]
        elif token in "\t ":
            assert self.under.read(1) == token, "Never fails"
        elif token.isidentifier():
            ident:Token = self.read_identifier()
            if self.under.peek(1) in ("'", '"'):
                ret:Token = self.read_string(ident.token, type_stamp=stamp)
            else:
                ret:Token = ident
        elif token == "?":
            q_mark:Stamp = self.under.stamp()
            self.under.read(1)
            if self.under.peek(1).isidentifier():
                ident:Token = self.read_identifier()
                ret:Token = Token("?"+ident.token, q_mark, TokenType.IDENTIFIER)
            else:
                ret:Token = Token("?", q_mark, TokenType.IDENTIFIER)
        elif token in "0123456789":
            ret:Token = self.read_number()
        elif token in "\"'":
            ret:Token = self.read_string("", type_stamp=stamp)
        elif token == "#":
            self._eat_comment()
        elif token == ".":
            if self.under.peek(2) == "..":
                self.under.read(3)
                ret:Token = Token("...", stamp, TokenType.OTHER)
            else:
                self.under.read(1)
                ret:Token = Token(".", stamp, TokenType.OTHER)
        elif token in "$:[](){},;":
            self.under.read(1)
            ret:Token = Token(token, stamp, TokenType.OTHER)
        elif token in "+-*^%|<>=":
            self.under.read(1)
            if (token == "-") and (self.under.peek(1) == ">"):
                token += self.under.read(1)
            else:
                if (token == "*") and (self.under.peek(1) == "*"): # **
                    token += self.under.read(1)
                if (token in "-+") and (self.under.peek(1) == token): # ++ --
                    token += self.under.read(1)
                elif self.under.peek(1) == "=":
                    token += self.under.read(1)
            ret:Token = Token(token, stamp, TokenType.OTHER)
        elif token in "/":
            self.under.read(1)
            if self.under.peek(1) == "/": # //
                token += self.under.read(1)
            if self.under.peek(1) == "=": # /= //=
                token += self.under.read(1)
            ret:Token = Token(token, stamp, TokenType.OTHER)
        elif self.under.peek(2) == "!=":
            ret:Token = Token(self.under.read(2), stamp, TokenType.OTHER)
        elif token == "\n":
            if self.freeze_indentation:
                self.under.read(1)
            else:
                return self.read_newline_into_buffer()
        else:
            self._throw(f"[LEXER] unknown {token=!r}", stamp)
        return [] if (ret is None) else [ret]

    def read_newline_into_buffer(self, *, first:bool=False) -> list[Token]:
        if not first:
            assert self.under.read(1) == "\n", "InternalError"
        # Cleanup empty newlines
        self.under.read_empty_lines("#", " \t")
        # Create stamp and newline token
        stamp:Stamp = self.under.stamp()
        newline:Token = Token("\n", stamp, TokenType.OTHER)
        # not USING_COLON
        if not USING_COLON:
            return [newline]
        # Dentation
        for i, indent in enumerate(self.indentation):
            if self.under.peek(len(indent)) == indent:
                self.under.read(len(indent))
            elif self.under.peek(1) in ("\t", " "):
                self._throw("[LEXER] IndentationError")
            else:
                token:Token = Token("\xff", stamp, TokenType.OTHER)
                ret:list[Token] = [token] * (len(self.indentation)-i)
                self.indentation:Stack[str] = self.indentation[:i]
                return ret + [newline]
        else:
            ret:list[Token] = [newline]
            # Indentation
            new_indent:str = ""
            while bool(self.under) and (self.under.peek(1) in "\t "):
                new_indent += self.under.read(1)
            if new_indent:
                indent_token:Token = Token("\t", stamp, TokenType.OTHER)
                self.indentation.append(new_indent)
                ret.append(indent_token)
            return ret

    def read_identifier(self) -> Token:
        token:str = ""
        stamp:Stamp = self.under.stamp()
        while True:
            token += self.under.read(1)
            ntoken:str = token + self.under.peek(1)
            if not ntoken.isidentifier():
                break
            if token == ntoken:
                break
        return Token(token, stamp, TokenType.IDENTIFIER)

    def read_number(self) -> Token:
        stamp:Stamp = self.under.stamp()
        output:str = self._parse_int()
        is_float:bool = False
        if self.under.peek(1) in ("x", "b"):
            _type:str = self.under.read(1)
            allowed:str = "01" if _type == "b" else "0123456789abcdef"
            base:int = 2 if _type == "b" else 16
            if output != "0":
                char:str = self.under.peek(1)
                if char and (char in allowed):
                    self._throw("[LEXER] Invalid binary integer literal", stamp)
                if char != "\n":
                    self.under.read(1)
                    stamp:Stamp = self.under.stamp()
                self._throw("[LEXER] Unexpected token after integer", stamp)
            output:str = ""
            while True:
                char:str = self.under.peek(1)
                if (not char) or (char not in allowed):
                    break
                output += self.under.read(1)
            return str(int(output, base=base))
        else:
            if self.under.peek(1) == ".":
                self.under.read(1)
                chunk:str = self._parse_int()
                output += "." + chunk
                is_float:bool = True
            if self.under.peek(1) == "e":
                self.under.read(1)
                chunk:str = self._parse_int()
                output += "e" + chunk
        token_type:TokenType = TokenType.FLOAT if is_float else TokenType.INT
        return Token(output, stamp, token_type)

    def _parse_int(self) -> str:
        stamp:Stamp = self.under.stamp()
        if self.under.peek(1) == "-":
            neg:str = self.under.read(1)
        else:
            neg:str = ""
        output:str = ""
        while True:
            char:str = self.under.peek(1)
            if not char:
                break
            if char not in "_0123456789":
                break
            else:
                output += char
                self.under.read(1)
        if (not output) or ("_" in output[0]+output[-1]):
            self._throw("[LEXER] Couldn't parse number literal", stamp)
        return neg+output.replace("_", "")

    def read_string(self, _type:str, type_stamp:Stamp) -> Token:
        stamp:Stamp = self.under.stamp()
        if len(set(_type)) != len(_type):
            self._throw(f"[LEXER] Invalid string prefix {_type!r}", type_stamp)
        if _type.strip("rfb") != "":
            self._throw(f"[LEXER] Invalid string prefix {_type!r}", type_stamp)
        _type:int = 1*("r" in _type) + 2*("f" in _type) + 4*("b" in _type)
        quote_type:str = self.under.read(1, "InteralError")
        multiline:bool = (self.under.peek(2) == quote_type*2)
        if multiline:
            self.under.read(2)
        output:str = ""
        while True:
            char:str = self.under.read(1, EOF_ERROR)
            if (char == "\n") and (not multiline):
                self._throw(EOL_ERROR, stamp)
            elif (char == "\\") and (_type & 1):
                if self.under.peek(1) in ("\n", quote_type):
                    char:str = self.under.read(1)
            elif char == "\\":
                char:str = self.under.read(1, EOF_ERROR)
                if char in STRING_ESCAPES:
                    char:str = STRING_ESCAPES[char]
                elif char in "xuU":
                    err_stamp:Stamp = self.under.stamp()
                    if char == "x":
                        data:str = self.under.read(2, EOF_ERROR)
                    elif char == "u":
                        data:str = self.under.read(4, EOF_ERROR)
                    elif char == "U":
                        data:str = self.under.read(8, EOF_ERROR)
                    try:
                        char:str = chr(self._hex_to_int(data))
                    except ValueError:
                        self._throw("[LEXER] Couldn't parse string literal " \
                                    "escape sequence", err_stamp)
            elif char == quote_type:
                if multiline:
                    if self.under.peek(2) == quote_type*2:
                        self.under.read(2)
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