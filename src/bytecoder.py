from __future__ import annotations
from contextlib import contextmanager
from functools import reduce
from operator import iconcat
from io import StringIO
from sys import stderr
import traceback

from frontend.bcast import *
from asts.ccast import *
from parser import *
from lexer import *

DEBUG_RAISE:bool = False


class Labels:
    __slots__ = "_free_label"

    def __init__(self) -> Labels:
        self.reset()

    def get_fl(self) -> int:
        current, self._free_label = self._free_label, self._free_label+1
        return current

    def reset(self) -> None:
        self._free_label:int = 0


class Regs:
    __slots__ = "_taken", "max_reg", "_parent"

    def __init__(self, parent:Regs=None) -> Regs:
        self._taken:list[bool] = [True,True] # 0,1
        self._parent:Regs = parent
        self.max_reg:int = 3

    def get_free_reg(self) -> int:
        for idx, taken in enumerate(self._taken):
            if not taken:
                self._taken[idx] = True
                return idx
        reg:int = len(self._taken)
        self._taken.append(True)
        if len(self._taken) >= (1<<(FRAME_SIZE*8)):
            raise RuntimeError("What are you doing that needs " \
                               "so many registers")
        self._notify_max_reg(reg)
        return reg

    def free_reg(self, reg:int, instructions:Block) -> None:
        assert 0 <= reg < len(self._taken), "ValueError"
        if reg > 1:
            assert self._taken[reg], "reg not taken"
            self._taken[reg] = False

    def clear_reg(self, reg:int, instructions:Block) -> None:
        if reg > 1:
            instructions.append(BLiteral(reg, BNONE, BLiteral.UNDEFINED_T))

    def _notify_max_reg(self, max_reg:int) -> None:
        self.max_reg:int = max(self.max_reg, max_reg)
        if self._parent is not None:
            self._parent._notify_max_reg(self.max_reg)


Env:type = list[str] # must be ordered
Nonlocals:type = dict[str:int]
LoopLabel:type = tuple[str,str] # Continue,Break jump labels
EnvReason:type = dict[str:Branch]


class State:
    __slots__ = "block", "blocks", "regs", "nonlocals", "loop_labels", \
                "master", "env", "not_env", "must_ret", "first_inst", \
                "flags", "must_end_loop", "uses", "class_state", "_attrs"

    def __init__(self, *, blocks:list[Block], nonlocals:Nonlocals,
                 block:Block, loop_labels:list[LoopLabel], regs:Regs,
                 env:Env, not_env:EnvReason, flags:FeatureFlags,
                 master:State=None, uses:dict[str:tuple[Token,bool]]) -> State:
        self.loop_labels:list[LoopLabel] = loop_labels
        self.uses:dict[str:tuple[Token,bool]] = uses
        self.nonlocals:Nonlocals = nonlocals
        self.blocks:list[Block] = blocks
        self.not_env:EnvReason = not_env
        self.must_end_loop:bool = False
        self.flags:FeatureFlags = flags
        self.class_state:bool = False
        self.first_inst:bool = True
        self._attrs:list[str] = []
        self.must_ret:bool = False
        self.master:State = master
        self.block:Block = block
        self.regs:Regs = regs
        self.env:Env = env

    @property
    def attrs(self) -> list[str]:
        state:State = self
        while state.master:
            state:State = state.master
        return state._attrs

    def add_attr(self, attr:str) -> None:
        if attr not in self.attrs:
            self.attrs.append(attr)

    @property
    def full_env(self) -> Env:
        return BUILTIN_HELPERS + self.env + sorted(list(self.not_env))

    # Copy helpers
    def _for_block(self) -> tuple[Nonlocals,State]:
        a:int = 0 if self.class_state else 1
        new_nl:NonLocals = {name:link+a for name,link in self.nonlocals.items()}
        master:State = self.master if self.class_state else self
        return new_nl, master

    def copy_for_func(self) -> State:
        new_nl, master = self._for_block()
        state:State = State(blocks=self.blocks, block=Block(), loop_labels=[],
                            nonlocals=new_nl, regs=Regs(self.regs), env=Env(),
                            not_env=self.not_env.copy(), flags=self.flags,
                            master=master, uses={})
        state.blocks.append(state.block)
        return state

    def copy_for_branch(self) -> State:
        return State(blocks=self.blocks, loop_labels=self.loop_labels,
                     block=self.block, nonlocals=self.nonlocals,
                     regs=self.regs, env=self.env.copy(), flags=self.flags,
                     not_env=self.not_env.copy(), master=self.master,
                     uses=self.uses.copy())

    def copy_for_class(self) -> State:
        new_nl, master = self._for_block()
        state:State = State(blocks=self.blocks, block=Block(), loop_labels=[],
                            nonlocals=new_nl, regs=Regs(self.regs), env=Env(),
                            not_env=self.not_env.copy(), flags=self.flags,
                            master=master, uses={})
        state.blocks.append(state.block)
        state.class_state:bool = True
        return state

    @staticmethod
    def new(flags:FeatureFlags) -> State:
        block:Block = Block()
        env:Env = Env(BUILTINS[len(BUILTIN_HELPERS):])
        uses:dict[str:tuple[Token,bool]] = {n:(None,True) for n in BUILTINS}
        return State(block=block, blocks=[block], nonlocals=Nonlocals(),
                     loop_labels=[], regs=Regs(), env=env, uses=uses,
                     not_env=EnvReason(), flags=flags)

    # Env helpers
    def _merge_branch(self, branch:Branch, other:State) -> None:
        if self.must_ret or self.must_end_loop:
            self.not_env:EnvReason = other.not_env
            self.env:Env = other.env
            return
        if other.must_ret or other.must_end_loop:
            return
        self.not_env |= other.not_env
        for name in other.env:
            if name in self.env:
                # TODO: merge self.env[name] with other.env[name]
                pass
            else:
                self.not_env[name] = branch
        for name in self.env:
            if name not in other.env:
                self.env.remove(name)
                if name in other.not_env:
                    self.not_env[name] = other.not_env[name]
                else:
                    self.not_env[name] = branch

    def merge_branch(self, branch:Branch, other:State) -> None:
        self._merge_branch(branch, other)
        self.must_end_loop:bool = (self.must_end_loop or self.must_ret) and \
                                  (other.must_end_loop or other.must_ret)
        self.must_ret &= other.must_ret

    def assert_read(self, name_token:Token, reg:int) -> None:
        assert isinstance(name_token, Token), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert reg > 1, "ValueError"
        if self.must_end_loop: return None
        if self.must_ret: return None
        state:State = self
        while state is not None:
            name:str = state._guard_env(name_token)
            if (not state.class_state) or (state is self): # skip class def vars
                if name in state.env:
                    if name not in self.uses:
                        self.uses[name] = (name_token, False)
                    self.append_bast(BStoreLoadDict(name, reg, False))
                    return
            state:State = state.master
        name_token.throw(f"variable {name!r} was not defined")

    def _guard_env(self, name_token:Token) -> None:
        name:str = name_token.token
        if name in self.not_env:
            branch_token:Token = self.not_env[name].ft
            branch_token.double_throw(f"Because of this branch",
                                      f"Variable {name!r} might not be " \
                                      f"defined here", name_token)
        return name

    def assert_can_nonlocal(self, name_token:Token) -> None:
        assert isinstance(name_token, Token), "TypeError"
        name:str = name_token.token
        if name in self.uses:
            used_token, iswrite = self.uses[name]
            used_token.double_throw("You used this variable before",
                                    "you declaired it as nonlocal", name_token)
        if name not in self.master.env:
            name_token.throw(f"variable {name!r} not defined in parent scope")

    def write_env(self, token:Token) -> None:
        assert isinstance(token, Token), "TypeError"
        name:str = token.token
        if name in self.nonlocals:
            # TODO: merge name with nonlocals
            pass
        else:
            if name in self.not_env:
                self.not_env.pop(name)
            self.env.append(name)
            used_token, iswrite = self.uses.get(name, (None,True))
            if not iswrite:
                used_token.double_throw("This variable read here refers to " \
                                        "a nonlocal variable",
                                        "But this variable write here is to " \
                                        "a nonlocal variable. Perhaps you " \
                                        f"forgot `nonlocal {name}`?", token)
            self.uses[name] = (token, True)

    # Reg helpers
    def free_reg(self, reg:int) -> None:
        assert isinstance(reg, int), "TypeError"
        self.regs.free_reg(reg, self.block)
        self.clear_reg(reg)

    def get_free_reg(self) -> int:
        return self.regs.get_free_reg()

    def clear_reg(self, reg:int) -> None:
        assert isinstance(reg, int), "TypeError"
        if self.flags.is_set("CLEAR_AFTER_USE"):
            self.regs.clear_reg(reg, self.block)

    def get_none_reg(self) -> int:
        reg:int = self.get_free_reg()
        self.append_bast(BLiteral(reg, BNONE, BLiteral.NONE_T))
        return reg

    @contextmanager
    def get_free_reg_wrapper(self) -> int:
        reg:int = self.get_free_reg()
        yield reg
        self.free_reg(reg)

    # Block helpers
    def append_bast(self, instruction:Bast) -> None:
        assert isinstance(instruction, Bast), "TypeError"
        self.block.append(instruction)

    # Fransforms
    def fransform_func(self) -> None:
        if not self.flags.is_set("ENV_IS_LIST"):
            return None
        nonlocals:Nonlocals = Nonlocals()
        fransformed_block:Block = Block()
        for bt in self.block:
            if isinstance(bt, BLoadLink):
                nonlocals[bt.name] = bt.link
                continue
            elif isinstance(bt, BDotDict) and self.flags.is_set("ENV_IS_LIST"):
                self.add_attr(bt.attr)
                attr_idx:int = self.attrs.index(bt.attr)
                bt:Bast = BDotList(bt.obj_reg, attr_idx, bt.reg, bt.storing)
            elif isinstance(bt, BStoreLoadDict):
                scope, link = self, 0
                if bt.name in nonlocals:
                    for _ in range(nonlocals[bt.name]):
                        scope, link = scope.master, link+1
                else:
                    while bt.name not in scope.full_env:
                        scope, link = scope.master, link+1
                        if scope is None:
                            raise NotImplementedError(f"{bt.name!r} cannot " \
                                                      f"be found")
                idx:int = scope.full_env.index(bt.name)
                bt:Bast = BStoreLoadList(link, idx, bt.reg, bt.storing)
            fransformed_block.append(bt)
        self.block[:] = fransformed_block

    def fransform_class(self) -> None:
        fransformed_block:Block = Block()
        for bt in self.block:
            if isinstance(bt, BStoreLoadDict):
                if bt.name not in self.nonlocals:
                    self.add_attr(bt.name)
                    if self.flags.is_set("ENV_IS_LIST"):
                        attr_idx:int = self.attrs.index(bt.name)
                        bt:Bast = BDotList(CLS_REG, attr_idx, bt.reg,
                                           bt.storing)
                    else:
                        bt:Bast = BDotDict(CLS_REG, bt.name, bt.reg,
                                           bt.storing)
            fransformed_block.append(bt)
        self.block[:] = fransformed_block
        self.fransform_func()
        # The stull bellow should be unneeded as long as the if above the
        #   add_attr(bt.name) reads:
        #   > if bt.name not in self.nonlocals
        #   and not:
        #   > if bt.name in self.env
        # This is because class scopes don't get an env in the interpreter
        #   so they don't get tmp variables. As an optimisation, I can
        #   put those tmp variables in self.regs # TODO
        for name, branch in self.not_env.items():
            if name in self.attrs: continue
            get_first_token(branch).throw(f"Variable {name!r} might not be " \
                                          f"defined because of this branch. " \
                                          f"No teporary variables are " \
                                          f"allowed in a class scope")


class Block(list[Bast]):
    __slots__ = ()

class SemanticError(Exception): ...

RegsSize = EnvSize = int


class ByteCoder:
    __slots__ = "ast", "_labels", "_flags", "_states", "_todo"

    def __init__(self, ast:Body, flags:FeatureFlags) -> ByteCoder:
        self._todo:list[Callable[None]] = []
        self._flags:FeatureFlags = flags
        self._labels:Labels = Labels()
        self._states:list[State] = []
        self.ast:Body = ast

    def _to_bytecode(self) -> tuple[int,int,list[str],list[Block]]:
        state:State = State.new(self._flags)
        self._states:list[State] = [state]
        self._labels.reset()
        for cmd in self.ast:
            reg:int = self._convert(cmd, state)
            state.free_reg(reg)
        with state.get_free_reg_wrapper() as tmp_reg:
            state.append_bast(BLiteral(tmp_reg, BLiteralInt(0), BLiteral.INT_T))
            state.append_bast(BRet(tmp_reg, False))
        state.fransform_func()
        while self._todo:
            self._todo.pop(0)()
        return state.regs.max_reg+1, len(state.full_env), state.attrs, \
               state.blocks

    def to_bytecode(self) -> tuple[RegsSize,EnvSize,list[str],list[Block]]:
        try:
            return self._to_bytecode()
        except (SemanticError, FinishedWithError) as error:
            if DEBUG_RAISE and isinstance(error, FinishedWithError):
                print(traceback.format_exc(), end="")
            print(f"\x1b[91mSemanticError: {error.msg}\x1b[0m", file=stderr)
            return 3, 0, [], []

    def _serialise(self, frame_size:int, env_size:int, attrs:list[str],
                   bytecode:Block) -> bytes:
        bast:bytearray = bytearray()
        for instruction in bytecode:
            bast += instruction.serialise()
        return serialise(self._flags, frame_size, env_size, attrs, bytes(bast))

    def serialise(self) -> bytes:
        frame_size, env_size, attrs, bytecode_blocks = self.to_bytecode()
        bytecode:Block = reduce(iconcat, bytecode_blocks, [])
        return self._serialise(frame_size, env_size, attrs, bytecode)

    def _convert_body(self, body:Body, state:State) -> None:
        for cmd in body:
            reg:int = self._convert(cmd, state)
            if reg > 1:
                state.free_reg(reg)

    def _convert(self, cmd:Cmd, state:State) -> int:
        assert isinstance(cmd, Cmd), "TypeError"
        if not isinstance(cmd, NonLocal):
            state.first_inst:bool = False
        res_reg:int = 0
        if isinstance(cmd, Assign):
            for target in cmd.targets:
                if isinstance(target, Var):
                    state.write_env(target.identifier)
            reg:int = self._convert(cmd.value, state)
            for target in cmd.targets:
                if isinstance(target, Var):
                    state.append_bast(BStoreLoadDict(target.identifier.token,
                                                     reg, True))
                elif isinstance(target, Op):
                    # res_reg:int = state.get_free_reg()
                    if target.op == "simple_idx":
                        args:list[int] = []
                        for exp in target.args:
                            args.append(self._convert(exp, state))
                        with state.get_free_reg_wrapper() as tmp_reg:
                            state.append_bast(BStoreLoadDict("simple_idx=",
                                                             tmp_reg, False))
                            state.append_bast(BCall([0,tmp_reg] + args + [reg]))
                        for reg in args:
                            state.free_reg(reg)
                    elif target.op == ".":
                        if not isinstance(target.args[1], Var):
                            raise NotImplementedError("Impossible")
                        attr:str = target.args[1].identifier.token
                        obj_reg:int = self._convert(target.args[0], state)
                        state.append_bast(BDotDict(obj_reg, attr, reg, True))
                        state.free_reg(obj_reg)
                    elif target.op == "·,·":
                        raise NotImplementedError("TODO")
                    else:
                        raise NotImplementedError("Impossible")
                else:
                    raise NotImplementedError("Impossible")
            state.free_reg(reg)

        elif isinstance(cmd, Literal):
            value:Token = cmd.literal
            if value.isint():
                literal:int = int(value.token)
                if literal in (0, 1):
                    res_reg:int = literal
                else:
                    res_reg:int = state.get_free_reg()
                    state.append_bast(BLiteral(res_reg, BLiteralInt(literal),
                                               BLiteral.INT_T))
            elif value.isstring():
                res_reg:int = state.get_free_reg()
                state.append_bast(BLiteral(res_reg, BLiteralStr(value.token),
                                           BLiteral.STR_T))
            elif value in ("true", "false"):
                res_reg:int = int(value == "true")
            elif value == "none":
                res_reg:int = state.get_none_reg()
            elif value.isfloat():
                raise NotImplementedError("TODO")
            else:
                raise NotImplementedError("Impossible")

        elif isinstance(cmd, Var):
            res_reg:int = state.get_free_reg()
            state.assert_read(cmd.identifier, res_reg)

        elif isinstance(cmd, Op):
            if cmd.op == "if":
                # Set up
                if_id:int = self._labels.get_fl()
                label_true:Bable = Bable("if_true_"+str(if_id))
                label_end:Bable = Bable("if_end_"+str(if_id))
                # Condition
                reg:int = self._convert(cmd.args[0], state)
                state.append_bast(BJump(label_true.id, reg, False))
                state.free_reg(reg)
                res_reg:int = state.get_free_reg()
                # If-false
                tmp_reg:int = self._convert(cmd.args[2], state)
                state.append_bast(BRegMove(res_reg, tmp_reg))
                state.free_reg(tmp_reg)
                state.append_bast(BJump(label_end.id, 1, False))
                # If-true
                state.append_bast(label_true)
                state.clear_reg(reg)
                tmp_reg:int = self._convert(cmd.args[1], state)
                state.append_bast(BRegMove(res_reg, tmp_reg))
                state.free_reg(tmp_reg)
                state.append_bast(label_end)
            elif cmd.op == ".":
                if not isinstance(cmd.args[1], Var):
                    raise NotImplementedError("Impossible")
                res_reg:int = state.get_free_reg()
                attr:str = cmd.args[1].identifier.token
                obj_reg:int = self._convert(cmd.args[0], state)
                state.append_bast(BDotDict(obj_reg, attr, res_reg, False))
                state.free_reg(obj_reg)
            else:
                res_reg:int = state.get_free_reg()
                used_regs:list[int] = []
                if cmd.op != "call":
                    func:int = state.get_free_reg()
                    state.assert_read(cmd.op, func)
                    used_regs.append(func)
                for arg in cmd.args:
                    reg:int = self._convert(arg, state)
                    used_regs.append(reg)
                state.append_bast(BCall([res_reg] + used_regs))
                for reg in used_regs:
                    state.free_reg(reg)

        elif isinstance(cmd, If):
            #   reg := condition
            #   jmpif reg if_true
            #   <if-false>
            #   jmp if_end
            # if_true:
            #   <if-true>
            # if_end:
            if_id:int = self._labels.get_fl()
            label_true:Bable = Bable("if_true_"+str(if_id))
            label_end:Bable = Bable("if_end_"+str(if_id))
            reg:int = self._convert(cmd.exp, state)
            state.append_bast(BJump(label_true.id, reg, False))
            state.free_reg(reg)
            other_state:State = state.copy_for_branch()
            for subcmd in cmd.false:
                tmp_reg:int = self._convert(subcmd, state)
                state.free_reg(tmp_reg)
            state.append_bast(BJump(label_end.id, 1, False))
            state.append_bast(label_true)
            state.clear_reg(reg)
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(subcmd, other_state)
                other_state.free_reg(tmp_reg)
            state.append_bast(label_end)
            state.merge_branch(cmd, other_state)

        elif isinstance(cmd, While):
            # while_start:
            #   reg := <condition>
            #   reg_neg := not(reg)
            #   jmpif (!reg) to while_end
            #   <while-body>
            #   jmp while_start
            # while_end:
            while_id:int = self._labels.get_fl()
            label_start:Bable = Bable(f"while_{while_id}_start")
            label_end:Bable = Bable(f"while_{while_id}_end")
            state.append_bast(label_start)
            reg:int = self._convert(cmd.exp, state)
            state.append_bast(BJump(label_end.id, reg, True))
            state.free_reg(reg)
            state.loop_labels.append((label_start.id, label_end.id))
            other_state:State = state.copy_for_branch()
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(subcmd, other_state)
                other_state.free_reg(tmp_reg)
            state.loop_labels.pop()
            state.append_bast(BJump(label_start.id, 1, False))
            state.append_bast(label_end)
            state.clear_reg(reg)
            state.merge_branch(cmd, other_state)

        elif isinstance(cmd, Func):
            func_id:int = self._labels.get_fl()
            # <INSTRUCTIONS>:
            #   res_reg := func_label
            res_reg:int = state.get_free_reg()
            func_literal:BLiteralFunc = BLiteralFunc(0, func_id, len(cmd.args))
            state.append_bast(BLiteral(res_reg, func_literal, BLiteral.FUNC_T))
            def todo() -> None:
                # <NEW BODY>:
                # label_start:
                #   <copy args>
                #   <function-body>
                label_start:Bable = Bable(f"{func_id}")
                nstate:State = state.copy_for_func()
                self._states.append(nstate)
                nstate.append_bast(label_start)
                for i, arg in enumerate(cmd.args, start=3):
                    token:Token = arg.identifier
                    nstate.write_env(token)
                    nstate.append_bast(BStoreLoadDict(token.token, i, True))
                for subcmd in cmd.body:
                    tmp_reg:int = self._convert(subcmd, nstate)
                    nstate.free_reg(tmp_reg)
                reg:int = nstate.get_free_reg()
                nstate.append_bast(BLiteral(reg, BNONE, BLiteral.NONE_T))
                nstate.append_bast(BRet(reg, False))
                nstate.fransform_func()
                func_literal.env_size = len(nstate.full_env)
            # Convert the Func to bytecode after the rest of the code in the
            #   currect scope. This allows for mutual recursion
            self._todo.append(todo)

        elif isinstance(cmd, NonLocal):
            # if not state.first_inst:
            #     cmd.ft.throw("nonlocal directives must be at the top of " \
            #                  "the scope")
            if state.master is None:
                cmd.ft.throw("variables in the global scope can't be nonlocal")
            for identifier_token in cmd.identifiers:
                identifier:str = identifier_token.token
                if identifier not in state.nonlocals:
                    if state.master.class_state:
                        state.master.assert_can_nonlocal(identifier_token)
                    else:
                        state.assert_can_nonlocal(identifier_token)
                    state.nonlocals[identifier] = 1
                link:int = state.nonlocals[identifier]
                state.append_bast(BLoadLink(identifier, link))

        elif isinstance(cmd, ReturnYield):
            if cmd.isreturn:
                if state.class_state:
                    get_first_token(cmd).throw("Cannot return from a class " \
                                               "scope")
                elif state.master is None:
                    # Returning from global scope
                    pass
                if cmd.exp is None:
                    reg:int = state.get_none_reg()
                else:
                    reg:int = self._convert(cmd.exp, state)
                state.append_bast(BRet(reg, False))
                state.free_reg(reg)
                state.must_ret:bool = True # Must be at the end
            else:
                if state.class_state:
                    get_first_token(cmd).throw("Class scope can't be a " \
                                               "generator")
                raise NotImplementedError("TODO")

        elif isinstance(cmd, BreakContinue):
            if cmd.n > len(state.loop_labels):
                action:str = "break" if cmd.isbreak else "continue"
                if len(state.loop_labels) == 0:
                    msg:str = "no loops to " + action
                else:
                    msg:str = action + " number too high"
                cmd.ft.throw(msg)
            start_label, end_label = state.loop_labels[-cmd.n]
            label:str = end_label if cmd.isbreak else start_label
            state.append_bast(BJump(label, 1, False))
            state.must_end_loop:bool = True # Must be at the end

        elif isinstance(cmd, Class):
            res_reg:int = state.get_free_reg()
            label:str = f"class_{self._labels.get_fl()}"
            bases:list[Reg] = []
            for base in cmd.bases:
                bases.append(self._convert(base, state))
            cls_literal:BLiteralClass = BLiteralClass(bases, label)
            state.append_bast(BLiteral(res_reg, cls_literal, BLiteral.CLASS_T))
            for base in bases:
                state.free_reg(base)
            def todo() -> None:
                nstate:State = state.copy_for_class()
                nstate.append_bast(Bable(label))
                cls_obj_reg:int = nstate.get_free_reg()
                assert cls_obj_reg == CLS_REG, "InternalError"
                for assignment in cmd.insides:
                    self._convert(assignment, nstate)
                nstate.free_reg(cls_obj_reg)
                nstate.append_bast(BRet(0, True))
                nstate.fransform_class()
            self._todo.append(todo)

        else:
            raise NotImplementedError(f"Not implemented {cmd!r}")

        return res_reg

    def to_file(self, filepath:str) -> None:
        assert isinstance(filepath, str), "TypeError"
        with open(filepath, "wb") as file:
            file.write(self.serialise())


if __name__ == "__main__":
    TEST1 = """
print("The following should be all 1s/`true`s")


f = func(x){
    g = func(){return x}
    return g
}
c = func(f){
    x = y = g = 0
    return f()
}
x = 5
y = 2
print(1, "\t", 7==c(f(x+y)))


f = func(x, y){
    return 0 if y == 0 else x + f(x, y-1)
}
x = 5
y = 10
z = f(x, y)
while (z > 5){
    z -= 1
}
while (z > 0){
    if (z == 5){
        break
    }
}
print(1, "\t", 5==z)


x = [5, 10, 15]
print(1, "\t", 10==x[1])
x[1] = 15
append(x, 20)
print(2, "\t", 15==x[1], 20==x[3])


x = 21
tmp = func(){
    nonlocal x
    g = func(){
        nonlocal x
        x += 1
    }
    func(h){
            func(a){a()}(h)
           }(g)
    func(h){h()}(g)
    return [g, (func(){nonlocal x; return x})]
}()
add = tmp[0]
get = tmp[1]
add()
add()
print(2, "\t", 25==x, get()+5==30)


Y = func(f){
    return func(x){
        return f(f(f(f(f(f(f(f)))))))(x)
    }
}
fact_helper = func(rec){
    return func(n){
        return 1 if n == 0 else n*rec(n-1)
    }
}
fact = Y(fact_helper)
print(1, "\t", 120==fact(5))


x = [1, 2, 3, 4, 5]
a = x[:2]
b = x[2:]
c = x[1:2]
d = x[-2:]
e = x[-2:-1]
f = x[::-1]
print(18, "\t", len(a)==2, a[0]==1, a[1]==2, len(b)==3, b[0]==3, b[1]==4,
      b[2]==5, len(c)==1, c[0]==2, len(d)==2, d[0]==4, d[1]==5, len(e)==1,
      e[0]==4, len(f)==5, f[0]==5, f[-1]==1, f[1]==4)


add = /? + ?/
add80 = /80 + 2*?/
add120 = /? + 120/
isodd = func(x) {
    if (x == 0) {
        return false
    }
    return iseven(x-1)
}
iseven = func(x) {
    if (x == 0) {
        return true
    }
    return isodd(x-1)
}

print(3, "\t", 200==add(120,80), 200==add120(80), 200==add80(60))
print(6, "\t", isodd(101), iseven(246), not isodd(246), not iseven(101),
      isodd(1), iseven(0))
"""[1:-1], False

    TEST2 = """
fib = func(x) ->
    if x < 1 ->
        return 1
    return fib(x-2) + fib(x-1)
print(fib(15), "should be", 1597)
print(fib(30), "should be", 2178309)
"""[1:-1], True

    TEST3 = """
i = 0
while i < 10_000_000 ->
    i += 1
print(i)
"""[1:-1], True

    TEST4 = """
max = 10_000
primes = [2]
i = 2
while (i < max){
    i += 1
    j = 2
    while (j < i){
        if (i%j){
            j += 1
        }else{
            continue 2
        }
    }
    append(primes, i)
}
print("the number of primes bellow", max, "is:", len(primes),
      "which should be", 1229)
print("the last prime is:", primes[-1], "which should be", 9973)
"""[1:-1], False

    TEST5 = """
f = func(x){
    if (x){
        return f(x-1) + 1
    }
    return x
}
print(f(100_000), "should be", 100_000)
"""[1:-1], False

    TEST6 = """
t = 0
A = class {
    nonlocal t
    X = 5
    f = func() {
        nonlocal t
        print(A.X, "should be", 6)
        print(A, "should be a class")
        t = A.X + 1
        A.X = 0
    }
    t = X
    if X {a=0}
}
print(t, "should be", 5)
A.X = 6
A.f()
print(t, "should be", 7)
print(A.X, "should be", 0)
"""[1:-1], False

    TEST7 = """
A = class {
    X = 0
}

i = 0
while i < 10000000 {
    i += 1
    A.X += 1
}
print(A.X)
"""[1:-1], False

    # DEBUG_RAISE:bool = True
    # 1:all, 2:fib, 3:while++, 4:primes, 5:rec++, 6:cls
    TEST = TEST1
    assert not isinstance(TEST, str), "TEST should be tuple[str,bool]"
    ast:Body = Parser(Tokeniser(StringIO(TEST[0])), colon=TEST[1]).read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        flags = FeatureFlags()
        flags.set("ENV_IS_LIST")
        flags.set("CLEAR_AFTER_USE")

        bytecoder:ByteCoder = ByteCoder(ast, flags)
        frame_size, env_size, attrs, bytecode_blocks = bytecoder.to_bytecode()
        bytecode:Block = reduce(iconcat, bytecode_blocks, [])
        if bytecode:
            print(bytecode_list_to_str(bytecode))

            raw_bytecode:bytes = bytecoder.serialise()
            with open("code-examples/example.clizz", "wb") as file:
                file.write(raw_bytecode)

            data = derialise(raw_bytecode)
            dec_flags, dec_frame_size, dec_env_size, dec_attrs, dec_bast = data
            print(f"{frame_size=}, {flags=}")
            assert bytecode_list_to_str(bytecode) == \
                   bytecode_list_to_str(dec_bast), "AssertionError"
            assert frame_size == dec_frame_size, "AssertionError"
            assert flags.issame(dec_flags), "AssertionError"
            if flags.is_set("ENV_IS_LIST"):
                assert env_size == dec_env_size, "AssertionError"
                assert attrs == dec_attrs, "AssertionError"

    # fib = lambda x: 1 if x < 1 else fib(x-2)+fib(x-1)
    # print(fib(30))