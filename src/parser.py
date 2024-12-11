from __future__ import annotations
from sys import stderr

from lexer import Tokeniser, Token, FinishedWithError, TokenType
from asts.ccast import *
import lexer # for lecter.USE_COLON


class Parser:
    __slots__ = "tokeniser"

    def __init__(self, tokeniser:Tokeniser, *, colon:bool=None) -> Parser:
        global BLOCK_START, BLOCK_END, USING_COLON
        if colon is None:
            colon:bool = lexer.USING_COLON
        lexer.USING_COLON = USING_COLON = colon
        BLOCK_START = "->" if USING_COLON else "{"
        BLOCK_END = "" if USING_COLON else "}"
        self.tokeniser:Tokeniser = tokeniser

    def throw(self, msg:str, node:Cmd|Expr) -> None:
        self.tokeniser.throw(msg, get_first_token(node))

    def _assert_read(self, expected:str, error_msg:str="") -> Token:
        assert isinstance(error_msg, str), "TypeError"
        assert isinstance(expected, str), "TypeError"
        token:Token = self.tokeniser.read()
        if token != expected:
            self.tokeniser.throw(error_msg, token)
        return token

    def _try_read(self, token:str) -> bool:
        assert isinstance(token, str), "TypeError"
        if not self.tokeniser:
            return False
        if self.tokeniser.peek() == token:
            self._assert_read(token)
            return True
        return False

    def _try_read_one_of(self, tokens:list[str]) -> bool:
        assert isinstance(tokens, list), "TypeError"
        for t in tokens: assert isinstance(t, str), "TypeError"
        token:Token = self.tokeniser.peek()
        if token in tokens:
            self._assert_read(token.token)
            return True
        else:
            return False

    def _try_read_after_newlines(self, token:str) -> bool:
        assert isinstance(token, str), "TypeError"
        i:int = 0
        while True:
            i += 1
            tokens:list[Token] = self.tokeniser.peek_n(i)
            if len(tokens) < i:
                return False
            if tokens[-1] != "\n":
                if tokens[-1] == token:
                    for _ in range(i-1): self._assert_read("\n")
                    self._assert_read(token)
                    return True
                else:
                    return False

    def _eat_newlines(self) -> None:
        raise DepricationError()

    # Read file/block/line
    def read(self) -> Body|None:
        try:
            return self._read_block(indented=False)
        except FinishedWithError as error:
            print(f"\x1b[91mSyntaxError: {error.msg}\x1b[0m", file=stderr)

    def _read_block(self, *, indented:bool) -> Body:
        """
        A block is list of lines of code where there is no de-indentation
        further from the start
        """
        if indented and USING_COLON:
            self._assert_read("\t", "Expected indented block")
        cmds:Body = []
        while True:
            # Eat newlines
            while self._try_read("\n"):
                pass
            if not self.tokeniser:
                break
            # If should stop:
            if not USING_COLON:
                if self.tokeniser.peek() == BLOCK_END:
                    break
            # Read a cmd into cmds
            cmds.extend(self._read_line())
            # If should stop
            if USING_COLON:
                if self.to_end_line():
                    break
            else:
                if self.tokeniser.peek() == BLOCK_END:
                    break
            # There must be a newline at the end of each line
            # Note that \xff always appears before \n
            self._assert_read("\n", "Expected newline")
        return cmds

    def _add_newline_start(self) -> None:
        """
        Adds a newline such that
            <Parser>._add_newline_start()
            assert <Parser>.tokeniser.peek() == "\n
        """
        assert not USING_COLON, "This function is only for not USING_COLON"
        self.tokeniser.prepend_token("\n")

    def to_end_line(self) -> bool:
        """
        Peek until a newline character
            If \xff before the newline character, read it and return True
            oherwise, return False
        This accounts for cases like this:
            x = ((func():
                    a), b)
        Note that after parsing `func():a`, there is no dentation and
          instead there is a ")"
        """
        assert USING_COLON, "This function is only for USING_COLON"
        i:int = 1
        while True:
            tokens:list[Token] = self.tokeniser.peek_n(i)
            if len(tokens) < i: break
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
            cmd:Cmd|None = self._read_line__partial()
            if cmd is not None:
                output.append(cmd)
            if not self._try_read(";"):
                return output

    def _read_line__partial(self) -> Cmd|None:
        """
        Read a statement
        Doesn't consume last newline
        """
        token:Token = self.tokeniser.peek()
        if token == "if":
            self._assert_read("if")
            exp:Expr = self._read_expr()
            true:Body = self._read_line__colon_block()
            if self._try_read_after_newlines("else"):
                if self.tokeniser.peek() == "if":
                    false:Body = self._read_line()
                elif self.tokeniser.peek() == BLOCK_START:
                    false:Body = self._read_line__colon_block()
                else:
                    self.tokeniser.throw(f'Expected "{BLOCK_START}" or "if"')
            else:
                false:Body = []
            return If(token, exp, true, false)
        elif token == "while":
            self._assert_read("while")
            exp:Expr = self._read_expr()
            true:Body = self._read_line__colon_block()
            return While(token, exp, true)
        elif token == "for":
            self._assert_read("for")
            identifier:Assignable = self._read_expr()
            self._assert_check_assignable(identifier)
            self._assert_read("in", 'Expected "in" after ' \
                                    '"for <identifier>"')
            exp:Expr = self._read_expr()
            body:Body = self._read_line__colon_block()
            if self._try_read_after_newlines("else"):
                nobreak:Body = self._read_line_colon_block()
            else:
                nobreak:Body = []
            return For(token, identifier, exp, body, nobreak)
        elif token in ("return", "yield"):
            self._assert_read(token.token)
            exp:Expr = None
            if self.tokeniser.peek() not in NOT_EXPR:
                exp:Expr = self._read_expr()
            return ReturnYield(token, exp, token=="return")
        elif token in ("break", "continue"):
            _type:Token = token
            self._assert_read(token.token)
            token:Token = self.tokeniser.peek()
            if token.isint():
                self._assert_read(token.token)
                n:int = int(token.token)
                if n < 1:
                    self.tokeniser.throw("Break/Continue can only be used " \
                                         "with a +ve integer literal", token)
            else:
                n:int = 1
            return BreakContinue(_type, n, _type=="break")
        elif token == "with":
            exp = var = None
            self._assert_read("with")
            if self.tokeniser.peek() != BLOCK_START:
                exp:Expr = self._read_expr()
                if self._try_read("as"):
                    var:Assignable = self._read_expr()
                    self._assert_check_assignable(var)
            code:Body = self._read_line__colon_block()
            catch:list[tuple[list[Var],Expr,Body]] = []
            fin, noerror = [], []
            while self._try_read_after_newlines("except"):
                excs:list[Var] = []
                catch_var:Expr = None
                if self.tokeniser.peek() != BLOCK_START:
                    while self.tokeniser and (self.tokeniser.peek() != "as"):
                        excs.append(self._read_identifier())
                        if not self._try_read("|"):
                            break
                    if self._try_read("as"):
                        catch_var:Expr = self._read_expr()
                        self._assert_check_assignable(catch_var)
                catch.append((excs, catch_var, self._read_line__colon_block()))
            if self._try_read_after_newlines("finally"):
                fin:Body = self._read_line__colon_block()
            if self._try_read_after_newlines("else"):
                noerror:Body = self._read_line__colon_block()
            return With(token, var, exp, code, catch, fin, noerror)
        elif token == "raise":
            self._assert_read("raise")
            exp:Expr = None
            if self.tokeniser.peek() not in NOT_EXPR:
                exp:Expr = self._read_expr()
            return Raise(token, exp)
        elif token == "nonlocal":
            self._assert_read(token.token)
            identifiers:list[Token] = []
            while True:
                identifier:Token = self.tokeniser.read()
                if not identifier.isidentifier():
                    self.tokeniser.throw("Expected identifier", identifier)
                identifiers.append(identifier)
                if not self._try_read(","):
                    break
            return NonLocal(token, identifiers)
        elif token == "...":
            self._assert_read("...")
            return None
        elif token == "match":
            self._assert_read("match")
            exp:Expr = self._read_expr()
            cases:list[MatchCase] = self._read_line__colon_block()
            for case in cases:
                if not isinstance(case, MatchCase):
                    self.throw('Expected "case" here', case)
            return Match(token, exp, cases)
        elif token == "case":
            self._assert_read("case")
            exp:Expr = self._read_expr()
            body:Body = self._read_line__colon_block()
            return MatchCase(token, exp, body)
        else:
            exps:list[Expr] = []
            while True:
                exps.append(self._read_expr())
                if not self._try_read("="):
                    break
            if len(exps) == 1:
                token:Token = self.tokeniser.peek()
                if self._try_read_one_of(MOD_ASSIGN_OPERATORS):
                    # x += 1
                    op_str:str = token.token.removesuffix("=")
                    target:Expr = exps[0]
                    self._assert_check_assignable(target)
                    value:Expr = self._read_expr()
                    renamed_token:Token = token.name_as(op_str)
                    op:Expr = Op(token, renamed_token, target, value)
                    return Assign([target], op)
                else:
                    # x
                    return exps[0]
            else:
                # x = y = 5
                value:Expr = exps.pop(-1)
                for exp in exps:
                    self._assert_check_assignable(exp)
                return Assign(exps, value)
        raise NotImplementedError("Unreachable")

    def _assert_check_assignable(self, exp:Expr) -> None:
        if isinstance(exp, Var):
            if not exp.identifier.token.startswith("?"):
                return None
        elif isinstance(exp, Op):
            if exp.op == "·,·":
                for sub_exp in exp.args:
                    self._assert_check_assignable(sub_exp)
                return None
            elif exp.op == "idx":
                self.throw("Can't assign to multiple elements in array " \
                           "at the same time", exp)
            elif exp.op == "simple_idx":
                self._assert_check_assignable(exp.args[0])
                return None
            elif exp.op == ".":
                self._assert_check_assignable(exp.args[0])
                self._assert_check_assignable(exp.args[1])
                return None
        self.throw("Invalid assignment target", exp)

    def _read_line__colon_block(self) -> Body:
        self._assert_read(BLOCK_START, f'Expected "{BLOCK_START}" here')
        if (not USING_COLON) and self._try_read(BLOCK_END):
            return []
        with self.tokeniser.freeze_indentation.inverse:
            if self._try_read("\n") or (not USING_COLON):
                block:Body = self._read_block(indented=True)
            else:
                block:Body = self._read_line()
        if BLOCK_END:
            self._assert_read(BLOCK_END, f'Expected "{BLOCK_END}" here')
        return block

    # Read expr
    def _read_expr(self, precedence:int=0, notype:bool=False,
                   istype:bool=False) -> Expr:
        """
        For a proper description look at TYPE_PRECEDENCE and EXPR_PRECEDENCE
        """
        precedence_data:list = TYPE_PRECEDENCE if istype else EXPR_PRECEDENCE
        assert precedence < len(precedence_data), "InternalError"
        special, assoc, operators = precedence_data[precedence]
        if special:
            if operators == ["·,·"]:
                return self._read_expr_basic_list(precedence, notype=notype,
                                                  istype=istype)
            elif operators == ["if_else_expr"]:
                assert not istype, f"{operators} is only for non-types"
                return self._read_expr_if_else(precedence, notype=notype)
            elif operators == ["func_call", "idx"]:
                assert not istype, f"{operators} is only for non-types"
                return self._read_expr_call_idx(precedence, notype=notype)
            elif operators == ["literal"]:
                assert not istype, f"{operators} is only for non-types"
                return self._read_expr_literal(notype=notype)
            elif operators == ["identifier"]:
                assert istype, f"{operators} is only for types"
                return self._read_identifier()
            elif operators == ["idx"]:
                assert istype, f"{operators} is only for types"
                return self._read_type_idx(precedence)
            raise NotImplementedError(f"special {operators=!r} not implemented")
        if assoc == "<_":
            op:Token = self.tokeniser.peek()
            if op in operators:
                self._assert_read(op.token)
                exp:Expr = self._read_expr(precedence, notype=notype,
                                           istype=istype)
                return Op(op, op, exp)
            else:
                return self._read_expr(precedence+1, notype=notype,
                                       istype=istype)
        elif assoc == "_<":
            exp:Exp = self._read_expr(precedence+1, notype=notype,
                                      istype=istype)
            while True:
                op:Token = self.tokeniser.peek()
                if op not in operators:
                    break
                self._assert_read(op.token)
                exp:Expr = Op(get_first_token(exp), op, exp)
            return exp
        elif assoc == "_<_":
            exp:Exp = self._read_expr(precedence+1, notype=notype, istype=istype)
            while True:
                op:Token = self.tokeniser.peek()
                if op not in operators:
                    break
                self._assert_read(op.token)
                exp2:Expr = self._read_expr(precedence+1, notype=notype,
                                            istype=istype)
                exp:Expr = Op(get_first_token(exp), op, exp, exp2)
            return exp
        raise NotImplementedError(f"{assoc=!r} not implemented/didn't return")

    def _read_type(self) -> Expr:
        return self._read_expr(1, notype=True, istype=True)

    def _read_type_idx(self, precedence:int) -> Expr:
        exp:Expr = self._read_expr(precedence+1, istype=True)
        while True:
            token:Token = self.tokeniser.peek()
            if not self._try_read("["):
                break
            idx:Expr = self._read_expr(istype=True)
            self._assert_read("]", "expected ] or comma")
            if self._is_tuple(idx):
                args:list[Expr] = idx.args
            else:
                args:list[Expr] = [idx]
            exp:Expr = Op(token, token.name_as("idx"), *args)
        return exp

    def _read_expr_basic_list(self, precedence:int=0, *, notype:bool,
                              istype:bool) -> Expr:
        """
        Reads "·,·"
        """
        exp:Expr = self._read_expr(precedence+1, notype=notype)
        comma_token:Token = self.tokeniser.peek()
        if not self._try_read(","):
            return exp # if not a basic list, just return
        args:list[Expr] = [exp]
        while self.tokeniser.peek() not in NOT_EXPR: # trailing comma
            args.append(self._read_expr(precedence+1, notype=notype,
                                        istype=istype))
            if not self._try_read(","):
                break
        return Op(comma_token, comma_token.name_as("·,·"), *args)

    def _read_expr_if_else(self, precedence:int, *, notype:bool) -> Expr:
        exp:Expr = self._read_expr(precedence+1, notype=notype)
        token:Token = self.tokeniser.peek()
        if self._try_read("if"):
            cond:Expr = self._read_expr(notype=notype)
            self._assert_read("else", "Missing else clause for if expr")
            return Op(get_first_token(exp), token, cond, exp,
                      self._read_expr(notype=notype))
        else:
            return exp

    def _read_expr_call_idx(self, precedence:int, *, notype:bool) -> Expr:
        token:Token = self.tokeniser.peek()
        ismacro:bool = self._try_read("$")
        exp:Expr = self._read_expr(precedence+1, notype=notype)
        while True:
            op:Token = self.tokeniser.peek()
            none:Expr = Literal(op.name_as("none"))
            if self._try_read("["):
                iscomplex:bool = False
                if self._try_read(":"):
                    start:Expr = none
                    iscomplex:bool = True
                else:
                    start:Expr = self._read_expr(notype=True)
                    iscomplex:bool = self._try_read(":")
                # I am not 100% how I managed to get this to work first try
                if iscomplex:
                    stop = step = none
                    if self._try_read(":"):
                        if self.tokeniser.peek() != "]":
                            step:Expr = self._read_expr(notype=True)
                    elif self.tokeniser.peek() != "]":
                        stop:Expr = self._read_expr(notype=True)
                        if self._try_read(":"):
                            if self.tokeniser.peek() != "]":
                                step:Expr = self._read_expr(notype=True)
                    exp:Expr = Op(get_first_token(exp), op.name_as("idx"), exp,
                                  start, stop, step)
                else:
                    exp:Expr = Op(get_first_token(exp), \
                                  op.name_as("simple_idx"), exp,
                                  start)
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
                exp:Expr = Op(get_first_token(exp), op.name_as(_type), exp,
                              *args)
                if self._try_read(":"):
                    exp.type:Expr = self._read_type()
            else:
                break
        if ismacro:
            self.tokeniser.throw("Macros only exist at compile time",
                                 token)
        return exp

    def _read_expr_literal(self, *, notype:bool) -> Expr:
        token:Token = self.tokeniser.peek()
        # int | float
        if token.isint() or token.isfloat() or token.isstring():
            self._assert_read(token.token)
            exp:Expr = Literal(token)
        # bool | None
        elif token in ("true", "false", "none"):
            self._assert_read(token.token)
            exp:Expr = Literal(token)
        # func
        elif token in ("func", "proc"):
            exp:Expr = self._read_func_proc_def()
        # class
        elif token in ("class", "record"):
            exp:Expr = self._read_class_record_def()
        # identifier
        elif token.isidentifier():
            exp:Expr = self._read_identifier()
        # (5,10) or (5+5)
        elif token == "(":
            with self.tokeniser.freeze_indentation:
                self._assert_read("(")
                if self.tokeniser.peek() == ")":
                    exp:Expr = Op(token, token.name_as("·,·"))
                else:
                    exp:Expr = self._read_expr()
                self._assert_read(")", "Expected ) or comma")
        # list
        elif token == "[":
            with self.tokeniser.freeze_indentation:
                self._assert_read("[")
                if self.tokeniser.peek() == "]":
                    exp:Expr = Op(token, token.name_as("[]"))
                else:
                    exp:Expr = self._read_expr()
                    if self._is_tuple(exp):
                        array:list[Expr] = exp.args
                    else:
                        array:list[Expr] = [exp]
                    exp:Expr = Op(token, token.name_as("[]"), *array)
                self._assert_read("]", "Expected ] or comma")
        # dict/set
        elif token == "{":
            exp:Expr = self._read_expr__set_dict()
        else:
            self.tokeniser.throw(f"Unexpected {token=!r}", token)

        if not notype:
            if self._try_read(":"):
                exp.type:Expr = self._read_type()
        return exp

    @staticmethod
    def _is_tuple(exp:Expr) -> bool:
        if isinstance(exp, Op):
            if exp.op == "·,·":
                return True
        return False

    def _read_identifier(self) -> Var:
        token:Token = self.tokeniser.peek()
        if not token.isidentifier():
            self.tokeniser.throw("expected identifier", token)
        self._assert_read(token.token)
        if token in KEYWORDS:
            self.tokeniser.throw(f"{token=!r} in KEYWORDS but parsed as " \
                                 f"identifier")
        return Var(token)

    def _read_class_record_def(self) -> Expr:
        class_token:Token = self.tokeniser.read()
        functional:bool = False
        if class_token == "record":
            functional:bool = True
        elif class_token != "class":
            self.tokeniser.throw("InternalError", class_token)

        token:Token = self.tokeniser.peek()
        bases:list[Expr] = []
        if token == "(":
            raw_bases:Expr = self._read_expr_literal(notype=False)
            if self._is_tuple(raw_bases):
                bases:list[Expr] = raw_bases.args
            else:
                bases:list[Expr] = [raw_bases]

        body:list[Cmd] = self._read_line__colon_block()
        for cmd in body:
            if not isinstance(cmd, Assign):
                self.throw("Expected assignment", cmd)

        return Class(class_token, bases, body, functional)

    def _read_func_proc_def(self) -> Expr:
        token:Token = self.tokeniser.read()
        functional:bool = False
        ret_type:Expr = None
        if token == "func":
            functional:bool = True
        elif token != "proc":
            self.tokeniser.throw("InternalError", token)
        args = self._read_expr__expr_list(*"()", self._read_func_proc_def_arg)
        if self._try_read(":"):
            ret_type:Expr = self._read_type()
        body:Body = self._read_line__colon_block()
        return Func(token, args, body, ret_type, functional)

    def _read_func_proc_def_arg(self, prev:list[Var]) -> Var:
        err_token:Token = self.tokeniser.peek()
        var:Var = self._read_identifier()
        curr_arg:Token = var.identifier
        if self._try_read(":"):
            var.type:Expr = self._read_type()
        if self._try_read("="):
            var.default:Expr = self._read_expr()
        else:
            for _ in filter(lambda x: not isinstance(x, Var), prev):
                self.tokeniser.throw("non-default argument follows " \
                                     "default argument", err_token)
        for arg in prev:
            arg:Token = (arg if isinstance(arg, Var) else arg.target).identifier
            if arg == curr_arg:
                self.tokeniser.throw("duplicate identifier in func definition",
                                     err_token)
        return var

    def _read_func_call_arg(self, prev:list[Expr|Var]) -> Expr|Var:
        err_token:Token = self.tokeniser.peek()
        exp:Expr = self._read_expr(1) # don't read commas
        if self._try_read("="):
            value:Expr = self._read_expr(1) # don't read commas
            if not isinstance(exp, Var):
                self.tokeniser.throw("expression cannot contain assignment, " \
                                     'perhaps you meant "=="', err_token)
            assert isinstance(exp, Var), "Impossible"
            exp.default:Expr = value
        else:
            for p in prev:
                if not isinstance(p, Var):
                    continue
                assert isinstance(p, Var), "Impossible"
                if p.default is None:
                    continue
                self.tokeniser.throw("non-default argument follows " \
                                     "default argument", err_token)
        return exp

    def _read_expr__set_dict(self) -> Op:
        is_dict:bool = True
        def read_func(args:list[Expr|DictPair]) -> Expr|DictPair:
            nonlocal is_dict
            exp:Expr = self._read_expr(1, notype=True) # don't read commas
            if (len(args) == 0) and (self.tokeniser.peek() != ":"):
                is_dict = False
                return exp
            if not is_dict:
                return exp
            self._assert_read(":", "Expected colon as part of a dict literal")
            exp2:Expr = self._read_expr(1) # don't read commas
            return DictPair(exp, exp2)

        start_token:Token = self.tokeniser.peek()
        args = self._read_expr__expr_list(*"{}", read_func)
        _type:str = "{:}" if is_dict else "{,}"
        return Op(start_token, start_token.name_as(_type), *args)

    def _read_expr__expr_list(self, start:str, end:str,
                              read_func) -> list[object]:
        assert isinstance(start, str), "TypeError"
        assert isinstance(end, str), "TypeError"
        args:list[object] = []
        with self.tokeniser.freeze_indentation:
            self._assert_read(start)
            while self.tokeniser.peek() != end:
                err_token:Token = self.tokeniser.peek()
                args.append(read_func(args))
                if self.tokeniser.peek() != ",":
                    break
                self._assert_read(",")
            self._assert_read(end, f"Expected {end} or comma")
        return args


EXPR_PRECEDENCE = [
                    # < means left assoc, > means with assoc
                    # [is_special, associativity, [*tokens]]
                    [True,  "",     ["·,·"]],
                    [True,  "",     ["if_else_expr"]], # right assoc
                    [False, "_<_",  ["and", "or"]],
                    [False, "<_",   ["not"]],
                    [False, "_<_",  ["==", "!=", "<", ">", "<=", ">=", "is"]],
                    [False, "_<_",  ["&", "|", "^"]],
                    [False, "_<_",  ["<<", ">>"]],
                    [False, "_<_",  ["+", "-"]],
                    [False, "_<_",  ["*", "/", "//", "%"]],
                    [False, "_<_",  ["**"]],
                    [False, "<_",   ["-", "~"]],
                    # [False, "<_",   ["++", "--"]], # name clashes with x++
                    # [False, "_<",   ["++", "--"]],
                    [True,  "",     ["func_call", "idx"]], # left assoc
                    [False, "_<_",  ["."]],
                    [True,  "",     ["literal"]],
                  ]
TYPE_PRECEDENCE = [
                    [True,  "",     ["·,·"]],
                    [False, "_<_",  ["|"]],
                    [True,  "",     ["idx"]], # left assoc
                    [False, "_<_",  ["."]],
                    [True,  "",     ["identifier"]],
                  ]
MOD_ASSIGN_OPERATORS = []
for op in ("+", "-", "*", "/", "//", "&", "^", "|", "<<", ">>"):
    MOD_ASSIGN_OPERATORS.append(op+"=")
KEYWORDS = ["true", "false", "none", "return", "break", "continue", "for",
            "while", "if", "func", "else", "with", "except", "finally",
            "raise", "class", "match", "case", "nonlocal", "yield", "proc",
            "record"]
NOT_EXPR:list[str] = list("\xff\n;]})")