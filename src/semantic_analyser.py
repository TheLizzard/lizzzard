from __future__ import annotations

from asts.ccast import *


INT_T:object = object()
STR_T:object = object()
NONE_T:object = object()
FUNC_T:object = object()
LINK_T:object = object()

class ReasonT:
    __slots__ = "reason", "ast_node"

    def __init__(self, reason:str, ast_node:Cmd) -> ReasonT:
        assert isinstance(ast_node, Cmd), "TypeError"
        assert isinstance(reason, str), "TypeError"
        self.ast_node:Cmd = ast_node
        self.reason:str = reason


Type:type = "INT_T | STR_T | NONE_T | FUNC_T | LinkT | Reason"
TEnv:type = dict[str:Type]
Success:type = bool


class SemanticFailed(Exception): ...


class SAnalyser:
    __slots__ = "ast"

    def __init__(self, ast:list[Cmd]) -> SAnalyser:
        self.ast:list[Cmd] = ast

    def analyse(self) -> Success:
        try:
            t, guaranteed = self._analyse(self.ast, {})
            if self.union(t, INT_T, {}) is not INT_T:
                self.raise_reason(ReasonT("Non int exit code", None))
        except SemanticFailed:
            return False
        else:
            return True

    def analyse(self, ast:list[Cmd], tenv:TEnv) -> tuple[Type|None,bool]:
        t:Type = None
        guaranteed:bool = True
        for cmd in ast:
            t2, g = self._analyse_cmd(cmd, tenv)
            t:Type = self.union(t or t2, t2)
            guaranteed &= g
        return t, guaranteed

    def _analyse_cmd(self, cmd:Cmd, tenv:TEnv) -> tuple[Type|None,bool]:
        if isinstance(cmd, Assign):
            for target in self.targets:
                assert isinstance(target, Var), "Not implemented yet"
                name:str = target.identifier.token
                env[name] = self._analyse_expr(target.value, tenv)
        elif isinstance(cmd, Expr):
            self._analyse_expr(cmd, tenv)
        elif isinstance(cmd, If):
            t:Type = self._analyse_expr(cmd.expr, tenv)
            if isinstance(t, ReasonT):
                self.raise_reason(t)
            true_tenv:TEnv = tenv.copy()
            false_tenv:TEnv = tenv.copy()
            t1:Type = self._analyse(cmd.true, true_tenv)
            t2:Type = self._analyse(cmd.false, false_tenv)
            for key in true_tenv.keys() & false_tenv.keys():
                t:Type = self.union(true_tenv[key], false_tenv[key], tenv)
                if t is None:
                    t = ReasonT(f"{key!r} is of type {true_tenv[key]} in the " \
                                f"true branch but it's of type " \
                                f"{false_tenv[key]} in the false branch.", cmd)
            return self.union(t1, t2), bool(t1 and t2)
        elif isinstance(cmd, While):
            t:Type = self._analyse_expr(cmd.expr, tenv)
            if isinstance(t, ReasonT):
                self.raise_reason(t)
            self._analyse(cmd.true, tenv.copy())
        elif isinstance(cmd, ReturnYield):
            assert cmd.isreturn, "Not implemented yield yet"
            return self._analyse_expr(cmd.exp), True
        elif isinstance(cmd, NonLocal):
            for identifier in cmd.identifiers:
                tenv[identifier.token] = LINK_T

    def _analyse_expr(self, expr:Expr, tenv:TEnv) -> Type:
        ...

    def union(self, t1:Type|None, t2:Type|None, tenv:TEnv) -> Type|None:
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        if t1 == t2:
            return t1
        if isinstance(t1, ReasonT):
            self.raise_reason(t1)
        if isinstance(t2, ReasonT):
            self.raise_reason(t2)

    def raise_reason(self, reason:ReasonT) -> None:
        assert isinstance(reason, ReasonT), "TypeError"
        print(f"[SEMANTIC ERROR]: {reason.reason}")
        print(f"[SEMANTIC ERROR]: in {reason.ast_node}")
        raise SemanticFailed()