from __future__ import annotations

from lexer import Token


class Expr:
    __slots__ = "type"

    def __init__(self) -> Expr:
        self.type:Expr = None


class DictPair(Expr):
    __slots__ = "exp1", "exp2"

    def __init__(self, exp1:Expr, exp2:Expr) -> DictPair:
        assert isinstance(exp1, Expr), "TypeError"
        assert isinstance(exp2, Expr), "TypeError"
        self.exp1:Expr = exp1
        self.exp2:Expr = exp2
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.exp1}:{self.exp2!r}"


class Func(Expr):
    __slots__ = "ft", "args", "body", "ret_type", "functional"

    def __init__(self, ft:Token, args:list[Var], body:Body, ret_type:Expr|None,
                 functional:bool) -> Func:
        assert isinstance(ret_type, Expr|None), "TypeError"
        assert isinstance(functional, bool), "TypeError"
        assert islist(body, Cmd|Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        assert islist(args, Var), "TypeError"
        self.ret_type:Expr|None = ret_type
        self.args:list[Var] = args
        self.body:Body = body
        self.ft:Token = ft
        super().__init__()

    def __repr__(self) -> str:
        return f"({self.args!r} => {self.body!r})"


class Class(Expr):
    __slots__ = "ft", "insides", "bases", "functional"

    def __init__(self, ft:Token, bases:list[Expr], insides:Body,
                 functional:bool) -> Class:
        assert isinstance(functional, bool), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        assert islist(insides, Cmd), "TypeError"
        assert islist(bases, Expr), "TypeError"
        self.bases:list[Expr] = bases
        self.insides:Body = insides
        self.ft:Token = ft
        super().__init__()

    def __repr__(self) -> str:
        return f"Class({repr(self.bases)[1:-1]}){self.insides}"


class Assign:
    __slots__ = "targets", "value"

    def __init__(self, targets:list[Assignable], value:Expr) -> Assign:
        assert islist(targets, Assignable), "TypeError"
        assert isinstance(value, Expr), "TypeError"
        assert len(targets) > 0, "ValueError"
        self.targets:list[Assignable] = targets
        self.value:Expr = value

    def __repr__(self) -> str:
        if len(self.targets) == 1:
            return f"{self.targets[0]} = {self.value}"
        return f"{self.targets} = {self.value}"


class Var(Expr):
    __slots__ = "identifier", "default"

    def __init__(self, identifier:Token) -> Var:
        assert isinstance(identifier, Token), "TypeError"
        self.identifier:Token = identifier
        self.default:Expr = None # for func def/calls
        super().__init__()

    def __repr__(self) -> str:
        if self.default is None:
            return f"Var[{self.identifier}]"
        else:
            return f"Var[{self.identifier} default={self.default}]"


class Literal(Expr):
    __slots__ = "literal"

    def __init__(self, literal:Token) -> Literal:
        assert isinstance(literal, Token), "TypeError"
        self.literal:Token = literal
        super().__init__()

    def __repr__(self) -> str:
        return f"Literal[{self.literal}]"


class Op(Expr):
    __slots__ = "ft", "op", "args"

    def __init__(self, ft:Token, op:Token, *args:tuple[Expr]) -> Op:
        assert isinstance(ft, Token), "TypeError"
        assert isinstance(op, Token), "TypeError"
        assert islist(args, Expr), "TypeError"
        self.args:list[Expr] = list(args)
        self.ft:Token = ft
        self.op:Token = op
        super().__init__()

    def __repr__(self) -> str:
        return f"Op({self.op} {repr(self.args)[1:-1]})"
Assignable:type = Op|Var


class If:
    __slots__ = "ft", "exp", "true", "false"

    def __init__(self, ft:Token, exp:Expr, true:Body, false:Body) -> If:
        assert islist(false, Cmd|Expr), "TypeError"
        assert islist(true, Cmd|Expr), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        self.false:Body = false
        self.true:Body = true
        self.exp:Expr = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        return f"If({self.exp!r}){self.true}{self.false}"


class While:
    __slots__ = "ft", "exp", "true"

    def __init__(self, ft:Token, exp:Expr, true:Body) -> While:
        assert islist(true, Cmd|Expr), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        self.true:Body = true
        self.exp:Expr = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        return f"While({self.exp!r}){self.true}"


class For:
    __slots__ = "ft", "identifier", "exp", "body", "nobreak"

    def __init__(self, ft:Token, identifier:Assignable, exp:Expr, body:Body,
                 nobreak:Body) -> For:
        assert isinstance(identifier, Assignable), "TypeError"
        assert islist(nobreak, Cmd|Expr), "TypeError"
        assert islist(body, Cmd|Expr), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        self.identifier:Assignable = identifier
        self.nobreak:Body = nobreak
        self.body:Body = body
        self.exp:Expr = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        return f"For({self.identifier} in {self.exp!r}){self.body}"


class With:
    __slots__ = "ft", "identifier", "exp", "code", "catch", "fin", "noerror"

    def __init__(self, ft:Token, identifier:Assignable|None, exp:Expr|None,
                 code:Body, catch:list[tuple[list[Var],Assignable|None,Body]],
                 fin:Body, noerror:Body) -> With:
        """
        with exp? as identifier?:
            <code>
        except <vars[0]|vars[1]|vars[2]|···>? as ?:
            <err body>
        finally:
            <fin>
        else:
            <noerror>
        """
        assert isinstance(identifier, Assignable|None), "TypeError"
        assert isinstance(exp, Expr|None), "TypeError"
        assert islist(noerror, Cmd|Expr), "TypeError"
        assert isinstance(catch, list), "TypeError"
        assert islist(code, Cmd|Expr), "TypeError"
        assert islist(fin, Cmd|Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        if exp is None:
            assert identifier is None, "ValueErrpr"
        for t in catch:
            assert isinstance(t, tuple), "TypeError"
            assert len(t) == 3, "TypeError"
            v, a, b = t
            assert isinstance(a, Assignable|None), "TypeError"
            assert islist(b, Cmd|Expr), "TypeError"
            assert islist(v, Var), "TypeError"

        self.catch:list[tuple[list[Var],Expr,Body]] = catch
        self.identifier:Assignable|None = identifier
        self.noerror:Body = noerror
        self.exp:Expr|None = exp
        self.code:Body = code
        self.fin:Body = fin
        self.ft:Token = ft

    def __repr__(self) -> str:
        return f"With({self.identifier} = {self.exp!r}){self.code}" \
               f"{self.catch}{self.fin}{self.noerror}"


class ReturnYield:
    __slots__ = "ft", "exp", "isreturn"

    def __init__(self, ft:Token, exp:Expr|None, isreturn:bool) -> ReturnYield:
        assert isinstance(isreturn, bool), "TypeError"
        assert isinstance(exp, Expr|None), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        self.isreturn:bool = isreturn
        self.exp:Expr|None = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        if self.isreturn:
            if self.exp is None:
                return "Return"
            return f"Return({self.exp})"
        else:
            if self.exp is None:
                return "Yield"
            return f"Yield({self.exp})"


class Raise:
    __slots__ = "ft", "exp"

    def __init__(self, ft:Token, exp:Expr|None) -> Raise:
        assert isinstance(exp, Expr|None), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        self.exp:Expr|None = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        if self.exp is None:
            return f"Raise"
        return f"Raise({self.exp!r})"


class BreakContinue:
    __slots__ = "ft", "n", "isbreak"

    def __init__(self, ft:Token, n:int, isbreak:bool) -> BreakContinue:
        assert isinstance(isbreak, bool), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        assert isinstance(n, int), "TypeError"
        assert n >= 1, "ValueError"
        self.isbreak:bool = isbreak
        self.ft:Token = ft
        self.n:int = n

    def __repr__(self) -> str:
        if self.isbreak:
            return f"Break({self.n})"
        else:
            return f"Continue({self.n})"


class MatchCase:
    __slots__ = "ft", "exp", "body"

    def __init__(self, ft:Token, exp:Expr, body:Body) -> MatchCase:
        assert islist(body, Cmd|Expr), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        self.body:Body = body
        self.exp:Expr = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        return f"Case[{self.exp}=>{self.body}]"


class Match:
    __slots__ = "ft", "exp", "cases"

    def __init__(self, ft:Token, exp:Expr, cases:list[MatchCase]) -> Match:
        assert isinstance(cases, list), "TypeError"
        assert isinstance(exp, Expr), "TypeError"
        assert isinstance(ft, Token), "TypeError"
        for c in cases:
            assert isinstance(c, MatchCase), "TypeError"
        self.cases:list[MatchCase] = cases
        self.exp:Expr = exp
        self.ft:Token = ft

    def __repr__(self) -> str:
        return f"Match({self.exp}){self.cases}"


# class GeneratorComp(Expr):
#     __slots__ = "exp", "_for", "_if", "type"


class NonLocal:
    __slots__ = "ft", "identifiers"

    def __init__(self, ft:Token, identifiers:list[Token]) -> NonLocal:
        assert islist(identifiers, Token), "TypeError"
        assert len(identifiers) > 0, "ValueError"
        assert isinstance(ft, Token), "TypeError"
        self.identifiers:list[Token] = identifiers
        self.ft:Token = ft

    def __repr__(self) -> str:
        identifiers:str = ",".join(map(str, self.identifiers))
        return f"NonLocal({identifiers})"


def islist(obj:object, T:type) -> bool:
    if not isinstance(obj, list|tuple):
        return False
    for o in obj:
        if not isinstance(o, T):
            return False
    return True


def get_first_token(node:Cmd|Expr|MatchCase) -> Token:
    assert isinstance(node, Cmd|Expr|MatchCase), "TypeError"
    if isinstance(node, DictPair):
        return get_first_token(node.exp1)
    elif isinstance(node, Assign):
        return get_first_token(node.targets[0])
    elif isinstance(node, Var):
        return node.identifier
    elif isinstance(node, Literal):
        return node.literal
    elif isinstance(node, Func|Class|Op|If|While|For|With|ReturnYield|Raise| \
                          BreakContinue|MatchCase|Match|NonLocal):
        return node.ft
    raise NotImplemented("TODO")


Macro:type = Func
Branch:type = If|While|For|Match|With
Cmd:type = Expr|Assign|Branch|ReturnYield|BreakContinue|Raise|NonLocal|Macro
Body:type = list[Cmd]