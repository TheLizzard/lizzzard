from __future__ import annotations

from lexer import Token


class Expr:
    __slots__ = "t"


class Func(Expr):
    __slots__ = "args", "body", "t"

    def __init__(self, args:list[Var|Assign], body:Body, ret_type:Expr) -> Func:
        assert isinstance(args, list), "TypeError"
        assert isinstance(body, list), "TypeError"
        self.args:list[Var|Assign] = args
        self.body:Body = body
        self.t:list[T] = [None for _ in args] + [ret_type]

    def __repr__(self) -> str:
        return f"({self.args!r} => {self.body!r})"


class Class(Expr):
    __slots__ = "attributes", "slots"

    def __init__(self, attributes:list[Callable|Assignment]) -> Class:
        self.attributes:list[Callable|Assignment] = attributes
        self.slots:dict[Token:T] = dict()


class Assign:
    __slots__ = "identifier", "value"

    def __init__(self, identifier:Var, value:Expr) -> Assign:
        assert isinstance(identifier, Var|Expr), "TypeError"
        assert isinstance(value, Expr), "TypeError"
        self.identifier:Var = identifier
        self.value:Expr = value

    def __repr__(self) -> str:
        return f"{self.identifier} = {self.value!r}"


class Var(Expr):
    __slots__ = "identifier", "type", "read_only"

    def __init__(self, identifier:Token, read_only:bool=False) -> Var:
        assert isinstance(identifier, Token), "TypeError"
        self.identifier:Token = identifier
        self.read_only:bool = read_only
        self.type:Expr = None

    def __repr__(self) -> str:
        return f"Var[{self.identifier}]"


class Literal(Expr):
    __slots__ = "literal"

    def __init__(self, literal:Token) -> Literal:
        assert isinstance(literal, Token), "TypeError"
        self.literal:Token = literal

    def __repr__(self) -> str:
        return f"Literal[{self.literal}]"


class Op(Expr):
    __slots__ = "op", "args"

    def __init__(self, op:Token|None, *args:tuple[Expr]) -> Op:
        assert isinstance(op, Token|None), "TypeError"
        assert isinstance(args, tuple), "TypeError"
        self.args:list[Expr] = list(args)
        self.op:Token|None = op

    def __repr__(self) -> str:
        args:str = repr(self.args)[1:-1]
        return f"Op({self.op} {args})"


class If:
    __slots__ = "exp", "true", "false"

    def __init__(self, exp:Expr, true:Body, false:Body) -> If:
        assert isinstance(false, list), "TypeError"
        assert isinstance(true, list), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        self.false:Body = false
        self.true:Body = true
        self.exp:Expr = exp

    def __repr__(self) -> str:
        return f"If({self.exp!r}){self.true}{self.false}"


class While:
    __slots__ = "exp", "true"

    def __init__(self, exp:Expr, true:Body) -> While:
        assert isinstance(true, list), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        self.true:Body = true
        self.exp:Expr = exp

    def __repr__(self) -> str:
        return f"While({self.exp!r}){self.true}"


class For:
    __slots__ = "identifier", "exp", "body", "nobreak"

    def __init__(self, identifier:Token, exp:Expr, body:Body, nobreak:Body):
        self.identifier:Token = identifier
        self.nobreak:Body = nobreak
        self.body:Body = body
        self.exp:Expr = exp

    def __repr__(self) -> str:
        return f"For({self.identifier} in {self.exp!r}){self.body}"


class With:
    __slots__ = "identifier", "exp", "code", "catch", "fin"

    def __init__(self, identifier:Expr, exp:Expr, code:Body,
                 catch:list[tuple[list[Var],Expr,Body]], fin:Body) -> With:
        self.catch:list[tuple[list[Var],Expr,Body]] = catch
        self.identifier:Var = identifier
        self.code:Body = code
        self.fin:Body = fin
        self.exp:Expr = exp

    def __repr__(self) -> str:
        return f"With({self.identifier} = {self.exp!r}){self.code}" \
               f"{self.catch}{self.fin}"


class Return:
    __slots__ = "exp"

    def __init__(self, exp:Expr) -> Return:
        self.exp:Expr = exp

    def __repr__(self) -> str:
        if self.exp is None:
            return f"Return"
        return f"Return({self.exp!r})"


class Yield:
    __slots__ = "exp"

    def __init__(self, exp:Expr) -> Yield:
        self.exp:Expr = exp

    def __repr__(self) -> str:
        if self.exp is None:
            return f"Yield"
        return f"Yield({self.exp!r})"


class Raise:
    __slots__ = "exp"

    def __init__(self, exp:Expr) -> Raise:
        self.exp:Expr = exp

    def __repr__(self) -> str:
        if self.exp is None:
            return f"Raise"
        return f"Raise({self.exp!r})"


class Break:
    __slots__ = "n"

    def __init__(self, n:int) -> Break:
        self.n:int = n

    def __repr__(self) -> str:
        return f"Break({self.n})"


class Continue:
    __slots__ = "n"

    def __init__(self, n:int) -> Continue:
        self.n:int = n

    def __repr__(self) -> str:
        return f"Continue({self.n})"


class Match:
    __slots__ = "exp", "cases"

    def __init__(self, exp:Expr, cases:list[tuple[Expr,Body]]) -> Match:
        self.cases:list[tuple[Expr,Body]] = cases
        self.exp:Expr = exp

    def __repr__(self) -> str:
        return f"Match({self.exp!r}){self.cases}"


class GeneratorComp(Expr):
    __slots__ = "exp", "_for", "_if", "type"


class NonLocal:
    __slots__ = "identifier"

    def __init__(self, identifier:Token) -> NonLocal:
        self.identifier:Token = identifier

    def __repr__(self) -> str:
        return f"NonLocal({self.identifier})"


Macro:type = Func
Cmd:type = Expr|Assign|If|While|For|With|Return|Break|Continue|Raise| \
           Match|NonLocal|Yield
Body:type = list[Cmd]