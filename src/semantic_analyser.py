from __future__ import annotations

from lexer import Token
from asts.ccast import *
from asts.tcast import *


class SAnalyser:
    __slots__ = "variables", "ast"

    def __init__(self, ast:Body) -> SAnalyser:
        self.variables:dict[str:Variable] = {}
        self.ast:Body = ast

    def _read_body(self, body:Body) -> TBody:
        body:list[TCmd] = []
        for cmd in body:
            if isinstance(cmd, Assign):
                self._read_body__assign(body, cmd)

    def _read_body__assign(self, body:TBody, cmd:Assign) -> None:
        t:TExpr = self._read_expr(cmd.targets[0])
        tcmd:TAssign = TAssign(cmd, t)
        tcmd.value:TExpr = self._read_expr(cmd.value)
        body.append(tcmd)
        for target in cmd.targets[1:]:
            tcmd:TAssign = TAssign(cmd, self._read_expr(target))
            tcmd.value:TExpr = t
            body.append(tcmd)