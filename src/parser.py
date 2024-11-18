from __future__ import annotations
from sys import stderr

from lexer import Tokeniser, Token, FinishedWithError, TokenType
from lexer import USING_COLON
from asts.ccast import *


class Parser:
    __slots__ = "tokeniser"

    def __init__(self, tokeniser:Tokeniser) -> Parser:
        self.tokeniser:Tokeniser = tokeniser

    def _assert_read(self, expected:str, error_msg:str="") -> Token:
        token:Token = self.tokeniser.read()
        if token != expected:
            self.tokeniser.throw(error_msg, token)
        return token

    def _try_read(self, token:Token) -> bool:
        if not self.tokeniser: # if token=="\n" but no data, return False
            return False
        if self.tokeniser.peek() == token:
            self._assert_read(token)
            return True
        return False

    def _eat_newlines(self) -> None:
        while self._try_read("\n"):
            pass

    def read(self) -> Body|None:
        try:
            if (self.tokeniser.peek() == "\t") and USING_COLON:
                self.tokeniser.throw("File can't start with " \
                                     "indentation")
            return self._read_block(indented=False)
        except FinishedWithError as error:
            print(f"\x1b[91mSyntaxError: {error.msg}\x1b[0m", file=stderr)

    def _read_block(self, *, indented:bool) -> Body:
        """
        A block is list of lines of code where there is no de-indentation
        further from the start
        """
        self._eat_newlines()
        if indented:
            if USING_COLON:
                self._assert_read("\t", "Expected indented block")
        cmds:Body = []
        while self.tokeniser:
            cmds.extend(self._read_line())
            if USING_COLON:
                if self.to_end_line():
                    break
            else:
                if self.tokeniser.peek() == BLOCK_END:
                    break
            if USING_COLON:
                self._assert_read("\n", "Expected newline")
            else:
                self._eat_newlines()
        return cmds

    def to_end_line(self) -> bool:
        """
        Peek until a newline character
            If \xff before the newline character, read it and return True
            Else, return False
        """
        i:int = 1
        while True:
            tokens:list[Token] = self.tokeniser.peek(i)
            if tokens[-1] == "\n":
                if len(tokens) == 1:
                    return False
                if tokens[-2] != "\xff":
                    return False
                self.tokeniser.remove_token_from_queue(tokens[-2])
                return True
            i += 1

    def _read_line(self) -> Body:
        """
        Read commands until newline or finished
        """
        output:Body = []
        while True:
            output.append(self._read_line__partial())
            if not self._try_read(";"):
                return output

    def _read_line__partial(self) -> Cmd:
        """
        Read a statement
        Doesn't consume newlines
        """
        token:Token = self.tokeniser.peek()
        if token == "if":
            self._assert_read("if")
            exp:Expr = self._read_expr()
            true:Body = self._read_line__colon_block()
            if self.tokeniser.peek(2) == ["\n", "else"]:
                self._assert_read("\n")
                self._assert_read("else")
                if self.tokeniser.peek() == "if":
                    false:Body = self._read_line()
                elif self.tokeniser.peek() == BLOCK_START:
                    false:Body = self._read_line__colon_block()
                else:
                    self.tokeniser.throw(f'Expected "{BLOCK_START}" or "if"')
            else:
                false:Body = []
            return If(exp, true, false)
        elif token == "while":
            self._assert_read("while")
            exp:Expr = self._read_expr()
            true:Body = self._read_line__colon_block()
            return While(exp, true)
        elif token == "for":
            self._assert_read("for")
            identifier:Token = self.tokeniser.read()
            if not identifier.isidentifier():
                self.tokeniser.throw("Expected identifier after " \
                                     '"for" token', identifier)
            self._assert_read("in", 'Expected "in" after ' \
                                    '"for <identifier>"')
            exp:Expr = self._read_expr()
            body:Body = self._read_line__colon_block()
            if self.tokeniser.peek(2) == ["\n", "else"]:
                self._assert_read("\n")
                self._assert_read("else")
                nobreak:Body = self._read_line_colon_block()
            else:
                nobreak:Body = []
            return For(identifier, exp, body, nobreak)
        elif token in ("return", "yield"):
            self._assert_read(token)
            exp:Expr = None
            if self.tokeniser.peek() not in NOT_EXPR:
                exp:Expr = self._read_expr()
            return Return(exp) if token == "return" else Yield(exp)
        elif token in ("break", "continue"):
            _type:Token = token
            self._assert_read(token)
            token:Token = self.tokeniser.peek()
            if token.isint():
                self._assert_read(token)
                n:int = int(token.token)
                if n < 1:
                    self.tokeniser.throw("Break/Continue can only be used " \
                                         "with a +ve integer literal", token)
            else:
                n:int = 1
            if _type == "break":
                return Break(n)
            elif _type == "continue":
                return Continue(n)
            else:
                self.tokeniser.throw("InternalError unreachable")
        elif token == "with":
            exp = var = None
            self._assert_read("with")
            if self.tokeniser.peek() != BLOCK_START:
                exp:Expr = self._read_expr()
                self._assert_read("as", 'Expected "as" token after expression')
                err_token:Token = self.tokeniser.peek()
                var:Expr = self._read_expr()
                self._assert_check_assignable(var, err_token)
            if self.tokeniser.peek() != BLOCK_START:
                self.throw("Expected colon here")
            code:Body = self._read_line__colon_block()
            catch:list[tuple[list[Var],Expr,Body]] = []
            fin:Body = []
            self._eat_newlines()
            while self._try_read("except"):
                excs:list[Var] = []
                catch_var:Expr = None
                if self.tokeniser.peek() != BLOCK_START:
                    while self.tokeniser and (self.tokeniser.peek() != "as"):
                        excs.append(self._read_identifier())
                        if not self._try_read("|"):
                            break
                    if self._try_read("as"):
                        catch_var:Expr = self._read_expr()
                catch.append((excs, catch_var, self._read_line__colon_block()))
                self._eat_newlines()
            if self._try_read("finally"):
                fin:Body = self._read_line__colon_block()
            return With(var, exp, code, catch, fin)
        elif token == "raise":
            self._assert_read("raise")
            exp:Expr = None
            if self.tokeniser.peek() not in NOT_EXPR:
                exp:Expr = self._read_expr()
            return Raise(exp)
        elif token == "nonlocal":
            self._assert_read(token)
            identifier:Token = self.tokeniser.read()
            if not identifier.isidentifier():
                self.tokeniser.throw(f"Expected identifier after {token}")
            return NonLocal(identifier)
        else:
            type_set:bool = False
            exp:Expr = self._read_expr()
            if isinstance(exp, Var):
                if self._try_read(":"):
                    exp.type:Expr = self._read_expr()
                    type_set:bool = True
            if self._try_read("="):
                self._assert_check_assignable(exp, token)
                return Assign(exp, self._read_expr())
            elif type_set:
                self.tokeniser.throw("Missing assignment after the colon type")
            else:
                return exp

    def _assert_check_assignable(self, exp:Expr, fst_token:Token) -> None:
        if isinstance(exp, Var):
            return None
        elif isinstance(exp, Op):
            if exp.op == "·,·":
                exp.op:str = "()"
            if exp.op == "()":
                for sub_exp in exp.args:
                    self._assert_check_assignable(sub_exp, fst_token)
                return None
            elif exp.op == "idx":
                self._assert_check_assignable(exp.args[0], fst_token)
                return None
            elif exp.op == ".":
                self._assert_check_assignable(exp.args[0], fst_token)
                self._assert_check_assignable(exp.args[1], fst_token)
                return None
        self.tokeniser.throw("Invalid assignment target", fst_token)

    def _read_line__colon_block(self) -> Body:
        self._assert_read(BLOCK_START, f'Expected "{BLOCK_START}" here')
        with self.tokeniser.freeze_indentation.inverse:
            if self._try_read("\n"):
                block:Body = self._read_block(indented=True)
            else:
                block:Body = self._read_line()
        if BLOCK_END:
            self._assert_read(BLOCK_END, f'Expected "{BLOCK_END}" here')
        return block

    def _read_expr(self, precedence:int=0, exp:Expr=None) -> Expr:
        """
        For a proper description look at EXPR_PRECEDENCE
        """
        if precedence < len(EXPR_PRECEDENCE):
            operator_args:tuple|str|None = EXPR_PRECEDENCE[precedence]
        else:
            operator_args:tuple|str|None = None

        if isinstance(operator_args, tuple):
            if len(operator_args) == 3:
                _, ops, _ = operator_args
                exp:Exp = exp or self._read_expr(precedence+1)
                while True:
                    op:Token = self.tokeniser.peek()
                    if op not in ops:
                        break
                    self._assert_read(op)
                    exp:Expr = Op(op, exp, self._read_expr(precedence+1))
            elif len(operator_args) == 2:
                if operator_args[1] == "_":
                    ops, _ = operator_args
                    op:Token = self.tokeniser.peek()
                    if op in ops:
                        self._assert_read(op)
                        exp:Expr = Op(op, self._read_expr(precedence))
                    else:
                        exp:Exp = self._read_expr(precedence+1)
                else:
                    _, ops = operator_args
                    exp:Exp = exp or self._read_expr(precedence+1)
                    while True:
                        op:Token = self.tokeniser.peek()
                        if op not in ops:
                            break
                        self._assert_read(op)
                        exp:Expr = Op(op, exp)
            else:
                # Invalid length of tuple in EXPR_PRECEDENCE
                #  should have been either 2 or 3
                self.tokeniser.throw("InternalError: EXPR_PRECEDENCE 1")
        elif isinstance(operator_args, str):
            if operator_args == "if_else_expr":
                exp:Expr = self._read_expr(precedence+1)
                token:Token = self.tokeniser.peek()
                if token == "if":
                    self._assert_read("if")
                    cond:Expr = self._read_expr()
                    self._assert_read("else", "Missing else clause")
                    exp:Expr = Op(token, cond, exp, self._read_expr())
            elif operator_args == "call_or_idx":
                exp:Expr = self._read_expr_call_or_idx(precedence)
            elif operator_args == "literal":
                exp:Expr = self._read_expr_literal()
            elif operator_args == "·,·":
                exp:Expr = self._read_expr(precedence+1)
                istuple:bool = False
                op:Op = Op(None, exp)
                while True:
                    token:Token = self.tokeniser.peek()
                    if token != ",":
                        break
                    self._assert_read(",")
                    if not istuple:
                        op.op:Token = token.name_as("·,·")
                    istuple:bool = True
                    if self.tokeniser.peek() in NOT_EXPR:
                        break
                    op.args.append(self._read_expr(precedence+1))
                return op if istuple else exp
            else:
                # Invalid string in EXPR_PRECEDENCE
                self.tokeniser.throw("InternalError: EXPR_PRECEDENCE 2")
        else:
            # Invalid type in EXPR_PRECEDENCE
            self.tokeniser.throw("InternalError: EXPR_PRECEDENCE 3")
        return exp

    def _read_expr_call_or_idx(self, precedence:int) -> Expr:
        token:Token = self.tokeniser.peek()
        ismacro:bool = self._try_read("$")
        exp:Expr = self._read_expr(precedence+1)
        while True:
            op:Token = self.tokeniser.peek()
            if op == "[":
                self._assert_read("[")
                start:Expr = self._read_expr()
                if self._try_read(":"):
                    end:Expr = self._read_expr()
                else:
                    start, end = Literal(op.name_as("none")), start
                if self._try_read(":"):
                    step:Expr = self._read_expr()
                else:
                    step:Expr = Literal(op.name_as("none"))
                exp:Expr = Op(op.name_as("idx"), exp, start, end, step)
                self._assert_read("]", "Expected ] character")
            elif op == "(":
                if ismacro:
                    ismacro:bool = False
                    _type:str = "$call"
                    if not isinstance(exp, Var):
                        self.tokeniser.throw("Macros only exist at " \
                                             "compile time", token)
                else:
                    _type:str = "call"
                args = self._read_expr__expr_list(*"()",
                                                  self._read_func_call_arg)
                exp:Expr = Op(op.name_as(_type), exp, *args)
            else:
                break
        if ismacro:
            self.tokeniser.throw("Macros only exist at compile time",
                                 token)
        if self.tokeniser.peek() == ".":
            exp:Expr = self._read_expr(precedence+1, exp)
        return exp

    def _read_expr_literal(self) -> Expr:
        token:Token = self.tokeniser.peek()
        # int | float
        if token.isnumber() or token.isstring():
            self._assert_read(token)
            return Literal(token)
        # bool | None
        elif token in ("true", "false", "none"):
            self._assert_read(token)
            return Literal(token)
        # func
        elif token == "func":
            return self._read_func_def()
        # identifier
        elif token.isidentifier():
            return self._read_identifier()
        # array/brackets
        elif token == "(":
            with self.tokeniser.freeze_indentation:
                self._assert_read("(")
                if self._try_read(")"):
                    return Op(token.name_as("()"))
                else:
                    exp:Expr = self._read_expr()
                    self._assert_read(")", "Expected ) or comma here")
                    if self._is_tuple(exp):
                        return Op(token.name_as("()"), *exp.args)
                    else:
                        return exp
        # list
        elif token == "[":
            with self.tokeniser.freeze_indentation:
                self._assert_read("[")
                exp:Expr = self._read_expr()
                self._assert_read("]", "Expected ] or comma here")
                if self._is_tuple(exp):
                    return Op(token.name_as("[]"), *exp.args)
                else:
                    return Op(token.name_as("[]"), exp)
        # dict/set
        elif token == "{":
            return self._read_expr__set_dict()
        else:
            self.tokeniser.throw(f"Unexpected {token=!r}", token)

    @staticmethod
    def _is_tuple(exp:Expr) -> bool:
        if isinstance(exp, Op):
            if exp.op == "·,·":
                return True
        return False

    def _read_func_def(self) -> Expr:
        ret_type:Expr = None
        token:Token = self.tokeniser.read()
        if token != "func":
            self.tokeniser.throw("InternalError", token)
        args = self._read_expr__expr_list(*"()", self._read_func_def_arg)
        if self._try_read("->"):
            ret_type:Expr = self._read_expr()
        body:Body = self._read_line__colon_block()
        return Func(args, body, ret_type)

    def _read_identifier(self) -> Var:
        token:Token = self.tokeniser.peek()
        if not token.isidentifier():
            self.tokeniser.throw("expected identifier", token)
        self._assert_read(token)
        if token in KEYWORDS:
            self.tokeniser.throw(f"InternalError: {token=!r} in KEYWORDS " \
                                 f"but parsed as identifier")
        return Var(token)

    def _read_func_def_arg(self, prev:list[Var|Assign]) -> Var|Assign:
        err_token:Token = self.tokeniser.peek()
        var:Var = self._read_identifier()
        curr_arg:Token = var.identifier
        if self._try_read(":"):
            var.type:Expr = self._read_expr(precedence=1)
        if self._try_read("="):
            var:Assign = Assign(var, self._read_expr())
        else:
            for _ in filter(lambda x: not isinstance(x, Var), prev):
                self.tokeniser.throw("non-default argument follows " \
                                     "default argument", err_token)
        for arg in prev:
            arg:Token = (arg if isinstance(arg, Var) else arg.target).identifier
            if arg == curr_arg:
                self.tokeniser.throw("duplicate argument in func definition",
                                     err_token)
        return var

    def _read_func_call_arg(self, prev:list[Expr|Assign]) -> Expr|Assign:
        err_token:Token = self.tokeniser.peek()
        exp:Expr = self._read_expr(1) # don't read commas
        if self._try_read("="):
            value:Expr = self._read_expr(1) # don't read commas
            if not isinstance(exp, Var):
                self.tokeniser.throw("expression cannot contain assignment, " \
                                     'perhaps you meant "=="?', err_token)
            exp:Expr = Assign(exp, value)
        else:
            for p in prev:
                if not isinstance(p, Assign):
                    continue
                self.tokeniser.throw("non-default argument follows " \
                                     "default argument", err_token)
        return exp

    def __read_expr(self, prev:list[Expr]=None) -> Expr:
        return self._read_expr()

    def _read_expr__expr_list(self, start:str, end:str,
                              read_func) -> list[object]:
        args:list[object] = []
        with self.tokeniser.freeze_indentation:
            self._assert_read(start)
            while self.tokeniser.peek() != end:
                err_token:Token = self.tokeniser.peek()
                args.append(read_func(args))
                if self.tokeniser.peek() != ",":
                    break
                self._assert_read(",")
            self._assert_read(end, f"Expected {end} or comma here")
        return args

    def _read_expr__set_dict(self) -> Op:
        is_dict:bool = True
        def read_func(args:list[Expr|Assign]) -> Expr|Assign:
            nonlocal is_dict
            exp:Expr = self._read_expr(1) # don't read commas
            if (len(args) == 0) and (self.tokeniser.peek() != ":"):
                is_dict = False
                return exp
            if not is_dict:
                return exp
            self._assert_read(":", "Expected colon as part of a dict literal")
            exp2:Expr = self._read_expr(1) # don't read commas
            return Assign(exp, exp2)

        start_dict_set_token:Token = self.tokeniser.peek()
        args = self._read_expr__expr_list(*"{}", read_func)
        _type:str = "{:}" if is_dict else "{,}"
        return Op(start_dict_set_token.name_as(_type), *args)


BLOCK_START:str = ":" if USING_COLON else "{"
BLOCK_END:str = "" if USING_COLON else "}"
EXPR_PRECEDENCE = [
                    "·,·",
                    "if_else_expr", # right assosiative
                    ("_", ["and", "or"], "_"),
                    (["not"], "_"),
                    ("_", ["==", "!=", "<", ">", "<=", ">=", "is"], "_"),
                    ("_", ["&", "|", "^"], "_"),
                    ("_", ["<<", ">>"], "_"),
                    ("_", ["+", "-"], "_"),
                    ("_", ["*", "/", "//", "%"], "_"),
                    ("_", ["**"], "_"),
                    (["-", "~"], "_"),
                    (["++", "--"], "_"),
                    ("_", ["++", "--"]),
                    "call_or_idx",
                    ("_", ["."], "_"),
                    "literal",
                  ]
KEYWORDS = ("true", "false", "none", "return", "break", "continue", "for",
            "while", "if", "func", "else", "with", "except", "finally",
            "raise", "class", "match", "case", "nonlocal", "yield")
NOT_EXPR:list[str] = list("\xff\n;]})")