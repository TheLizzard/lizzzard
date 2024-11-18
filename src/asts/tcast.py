from __future__ import annotations

from asts.ccast import *

ODict:type = list # Ordered dictionary


class Type:
    __slots__ = ()

class TyInt(Type):
    __slots__ = ()
TINT:Type = TyInt()

class TyStr(Type):
    __slots__ = ()
TSTR:Type = TyStr()

class TyFunc(Type):
    __slots__ = "functional", "args", "ret_type", "ndefault"

    def __init__(self, functional:bool, args:ODict[str:Type|None],
                 ret_type:Type|None, ndefault:int) -> TyFunc:
        assert isinstance(ret_type, Type|None), "TypeError"
        assert isinstance(functional, bool), "TypeError"
        assert isinstance(ndefault, int), "TypeError"
        assert isinstance(args, ODict), "TypeError"
        self.args:ODict[str:Type|None] = args
        self.functional:bool = functional
        self.ret_type:Type = ret_type
        self.ndefault:int = ndefault

    def __repr__(self) -> str:
        return "->".join(list(map(lambda x: x[0], self.args))+[repr(ret_type)])


class TExpr:
    __slots__ = "type"


class TFunc(TExpr):
    __slots__ = "under", "body", "defaults"

    def __init__(self, under:Func) -> TFunc:
        assert isinstance(under, Func), "TypeError"
        args:ODict[str:Type|None] = []
        ndefaults:int = 0
        for arg in under.args:
            name:str = arg.identifier.token
            t:Expr|None = arg.type
            ndefaults += arg.default is not None
            args.append((name, None))
        ret:Expr|None = under.ret_type

        self.under:Func = under
        self.type:Type = TyFunc(under.functional, args, None, ndefaults)
        self.defaults:list[TExpr] = []
        self.body:list[TCmd] = []

    def __repr__(self) -> str:
        return f"({self.type} => {self.body!r})"


class TVar(Expr):
    __slots__ = "name"

    def __init__(self, under:Var) -> TVar:
        assert isinstance(under, Var), "TypeError"
        self.name:str = under.identifier.token

    def __repr__(self) -> str:
        return f"TVar[{self.name}]"


class TLiteralInt(Expr):
    __slots__ = "value"

    def __init__(self, under:Literal) -> TLiteral:
        assert isinstance(under, Literal), "TypeError"
        assert under.literal.isint(), "TypeError"
        self.value:int = int(under.literal.token)

    def __repr__(self) -> str:
        return f"TLiteralInt[{self.value}]"


class TLiteralStr(Expr):
    __slots__ = "value"

    def __init__(self, under:Literal) -> TLiteral:
        assert isinstance(under, Literal), "TypeError"
        assert under.literal.isstring(), "TypeError"
        self.value:str = str(under.literal.token)

    def __repr__(self) -> str:
        return f"TLiteralStr[{self.value}]"


class TOp(Expr):
    __slots__ = "under", "op", "args"

    def __init__(self, under:Op) -> TOp:
        assert isinstance(under, Op), "TypeError"
        self.op:str = under.op.token
        self.args:list[TExpr] = []
        self.under:Op = under

    def __repr__(self) -> str:
        return f"TOp({self.op} {repr(self.args)[1:-1]})"


class TAssign:
    __slots__ = "under", "target", "value"

    def __init__(self, under:Assign, target:str) -> TAssign:
        assert isinstance(under, Assign), "TypeError"
        self.value:TExpr|None = None
        self.under:Assign = under
        self.target:str = target

    def __repr__(self) -> str:
        return f"{self.target} = {self.value!r}"


# class TIf:
# class TWhile:
# class TFor:
# class TRaise:
# class TMatch:
# class TMatchCase:
# class TWith:
# class TReturnYield:


class TBreakContinue:
    __slots__ = "under", "n", "isbreak"

    def __init__(self, under:BreakContinue) -> TBreakContinue:
        assert isinstance(under, BreakContinue), "TypeError"
        self.isbreak:bool = under.isbreak
        self.under:BreakContinue = under
        self.n:int = under.n

    def __repr__(self) -> str:
        if self.isbreak:
            return f"TBreak({self.n})"
        else:
            return f"TContinue({self.n})"


class TNonLocal:
    __slots__ = "under", "names"

    def __init__(self, under:NonLocal) -> TNonLocal:
        assert isinstance(under, NonLocal), "TypeError"
        self.under:BreakContinue = under
        self.names:list[str] = []
        for name in under.identifiers:
            self.names.append(name.token)

    def __repr__(self) -> str:
        identifiers:str = ",".join(self.names)
        return f"TNonLocal({identifiers})"