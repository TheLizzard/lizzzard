from __future__ import annotations
from contextlib import contextmanager
from functools import reduce
from operator import iconcat
from sys import stderr, argv
from io import StringIO
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
        self.max_reg:int = 2

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

    def free_reg(self, reg:int) -> None:
        assert 0 <= reg < len(self._taken), "ValueError"
        assert self._taken[reg], "reg not taken"
        self.mark_as_free_reg(reg)

    def mark_as_free_reg(self, reg:int) -> None:
        assert 0 <= reg < len(self._taken), "ValueError"
        if reg > 1:
            self._taken[reg] = False

    def clear_reg(self, reg:int, instructions:Block) -> None:
        if reg > 1:
            instructions.append(BLiteral(EMPTY_ERR, reg, BNONE,
                                         BLiteral.UNDEFINED_T))

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
        self.must_ret:bool = False
        self.master:State = master
        self.block:Block = block
        self.regs:Regs = regs
        self.env:Env = env
        if self.master is None:
            self._attrs:list[str] = SPECIAL_ATTRS.copy()

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
        return self._add_lists(self.env, sorted(list(self.not_env)))

    def _add_lists(self, *arrays:tuple[list[object]]) -> list[object]:
        # Joins the lists into 1, while removing duplicates
        output:list[object] = []
        seen:set[object] = set()
        for array in arrays:
            for element in array:
                if element not in seen:
                    output.append(element)
                    seen.add(element)
        return output

    # Copy helpers
    def _for_block(self) -> tuple[Nonlocals,State]:
        while self.class_state:
            self:State = self.master
        new_nl:NonLocals = {name:link+1 for name,link in self.nonlocals.items()}
        return new_nl, self

    def copy_for_func(self) -> State:
        new_nl, self = self._for_block()
        state:State = State(blocks=self.blocks, block=Block(), loop_labels=[],
                            nonlocals=new_nl, regs=Regs(self.regs), env=Env(),
                            not_env=self.not_env.copy(), flags=self.flags,
                            master=self, uses={})
        state.blocks.append(state.block)
        return state

    def copy_for_class(self) -> State:
        _, self = self._for_block()
        state:State = State(blocks=self.blocks, block=Block(), loop_labels=[],
                            nonlocals=Nonlocals(), regs=Regs(self.regs),
                            env=Env(), not_env=self.not_env.copy(),
                            flags=self.flags, master=self, uses={})
        state.blocks.append(state.block)
        state.class_state:bool = True
        return state

    def copy_for_branch(self) -> State:
        return State(blocks=self.blocks, loop_labels=self.loop_labels,
                     block=self.block, nonlocals=self.nonlocals,
                     regs=self.regs, env=self.env.copy(), flags=self.flags,
                     not_env=self.not_env.copy(), master=self.master,
                     uses=self.uses.copy())

    @staticmethod
    def new(flags:FeatureFlags) -> State:
        block:Block = Block()
        env:Env = Env(REAL_BUILTINS)
        uses:dict[str:tuple[Token,bool]] = {n:(None,True) for n in BUILTINS}
        return State(block=block, blocks=[block], nonlocals=Nonlocals(),
                     loop_labels=[], regs=Regs(), env=env, uses=uses,
                     not_env=EnvReason(), flags=flags)

    # Env helpers
    def _merge_branch(self, branch:Branch, other:State) -> None:
        self.not_env |= other.not_env
        if self.must_ret or self.must_end_loop:
            self.env:Env = self._add_lists(self.env, other.env)
            return
        if other.must_ret or other.must_end_loop:
            self.env:Env = self._add_lists(self.env, other.env)
            return
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

    def assert_read(self, op:Cmd, name_token:Token, reg:int) -> None:
        assert isinstance(name_token, Token), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert isinstance(op, Cmd), "TypeError"
        assert reg > 1, "ValueError"
        if self.must_end_loop or self.must_ret: return
        state:State = self
        while state is not None:
            name:str = state._guard_env(name_token)
            if name in state.env:
                if name not in self.uses:
                    self.uses[name] = (name_token, False)
                self.append_bast(BStoreLoadDict(op_to_err(op), name, reg,
                                                False))
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
        master:State = self.master
        if name not in master.env:
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
                msg1:str = "This refers to a nonlocal variable [mode=read]"
                msg2:str = f"But this refers to a nonlocal variable " \
                           f"[mode=write].\nPerhaps you forgot " \
                           f"`nonlocal {name}`?"
                used_token.double_throw(msg1, msg2, token)
            self.uses[name] = (token, True)

    def get_class_chain(self) -> int:
        state:State = self
        link:int = 0
        while state.class_state:
            state:State = state.master
            link += 1
        return link

    def is_global_scope(self) -> bool:
        state:State = self
        while state.class_state:
            state:State = state.master
        return state.master is None

    # Reg helpers
    def free_reg(self, reg:int) -> None:
        assert isinstance(reg, int), "TypeError"
        self.regs.free_reg(reg)
        self.clear_reg(reg)

    def mark_as_free_reg(self, reg:int) -> None:
        assert isinstance(reg, int), "TypeError"
        self.regs.mark_as_free_reg(reg)
        self.clear_reg(reg)

    def get_free_reg(self) -> int:
        return self.regs.get_free_reg()

    def clear_reg(self, reg:int) -> None:
        assert isinstance(reg, int), "TypeError"
        if self.must_ret: return
        if self.flags.is_set("CLEAR_AFTER_USE"):
            self.regs.clear_reg(reg, self.block)

    def get_none_reg(self) -> int:
        reg:int = self.get_free_reg()
        self.append_bast(BLiteral(EMPTY_ERR, reg, BNONE, BLiteral.NONE_T))
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
        fransformed_block:Block = Block()
        for bt in self.block:
            if isinstance(bt, BLoadLink):
                continue
            elif isinstance(bt, BDotDict) and self.flags.is_set("ENV_IS_LIST"):
                self.add_attr(bt.attr)
                attr_idx:int = self.attrs.index(bt.attr)
                bt:Bast = BDotList(bt.err, bt.obj_reg, attr_idx, bt.reg,
                                   bt.storing)
            elif isinstance(bt, BStoreLoadDict):
                scope, link = self, 0
                if bt.name in self.nonlocals:
                    for _ in range(self.nonlocals[bt.name]):
                        scope, link = scope.master, link+1
                else:
                    while True:
                        while scope.class_state:
                            scope:State = scope.master
                        if bt.name in scope.full_env:
                            break
                        link += 1
                        scope:State = scope.master
                        if scope is None:
                            raise NotImplementedError("Impossible")
                    link += self.class_state
                if scope.master is None:
                    link:int = -1
                idx:int = scope.full_env.index(bt.name)
                bt:Bast = BStoreLoadList(bt.err, link, idx, bt.reg, bt.storing)
            fransformed_block.append(bt)
        self.block[:] = fransformed_block

    def fransform_class(self) -> None:
        fransformed_block:Block = Block()
        for bt in self.block:
            if isinstance(bt, BStoreLoadDict):
                if bt.name in self.full_env:
                    self.add_attr(bt.name)
                    if self.flags.is_set("ENV_IS_LIST"):
                        attr_idx:int = self.attrs.index(bt.name)
                        bt:Bast = BDotList(bt.err, CLS_REG, attr_idx, bt.reg,
                                           bt.storing)
                    else:
                        bt:Bast = BDotDict(bt.err, CLS_REG, bt.name, bt.reg,
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

    def fransform_block(self) -> None:
        fransformed_block:Block = Block()
        must_ret:bool = False
        for bt in self.block:
            if isinstance(bt, BRet):
                if not must_ret:
                    fransformed_block.append(bt)
                must_ret:bool = True
            elif isinstance(bt, Bable):
                must_ret:bool = False
            if not must_ret:
                fransformed_block.append(bt)
        self.block[:] = fransformed_block


class Block(list[Bast]):
    __slots__ = ()

class SemanticError(Exception): ...

RegsSize = EnvSize = int


class ByteCoder:
    __slots__ = "ast", "source_code", "_labels", "_flags", "_states", "_todo"

    def __init__(self, ast:Body, source_code:str, flags:FeatureFlags):
        self._todo:list[Callable[None]] = []
        self.source_code:str = source_code
        self._flags:FeatureFlags = flags
        self._labels:Labels = Labels()
        self._states:list[State] = []
        self.ast:Body = ast

    def _to_bytecode(self) -> tuple[int,int,list[str],list[Block]]:
        main_state:State = State.new(self._flags)
        self._states:list[State] = [main_state]
        self._labels.reset()
        for cmd in self.ast:
            reg:int = self._convert(cmd, main_state)
            main_state.free_reg(reg)
        with main_state.get_free_reg_wrapper() as tmp_reg:
            main_state.append_bast(BLiteral(EMPTY_ERR, tmp_reg, BLiteralInt(0),
                                            BLiteral.INT_T))
            main_state.append_bast(BRet(EMPTY_ERR, tmp_reg))
        main_state.fransform_func()
        while self._todo:
            self._todo.pop(0)()
        for state in self._states:
            state.fransform_block()
        return main_state.regs.max_reg+1, len(main_state.full_env), \
               main_state.attrs, main_state.blocks, self.source_code

    def to_bytecode(self) -> tuple[RegsSize,EnvSize,list[str],list[Block],str]:
        try:
            return self._to_bytecode()
        except (SemanticError, FinishedWithError) as error:
            if DEBUG_RAISE and isinstance(error, FinishedWithError):
                print(traceback.format_exc(), end="")
            print(f"\x1b[91mSemanticError: {error.msg}\x1b[0m", file=stderr)
            return 3, 0, [], [], ""

    def _serialise(self, frame_size:int, env_size:int, attrs:list[str],
                   bytecode:Block, source_code:str) -> bytes:
        bast:bytearray = bytearray()
        for instruction in bytecode:
            bast += instruction.serialise()
        return serialise(self._flags, frame_size, env_size, attrs, bytes(bast),
                         source_code)

    def serialise(self) -> bytes:
        frame_size, env_size, attrs, blocks, source_code = self.to_bytecode()
        bytecode:Block = reduce(iconcat, blocks, [])
        return self._serialise(frame_size, env_size, attrs, bytecode,
                               source_code)

    def _convert_body(self, body:Body, state:State) -> None:
        for cmd in body:
            reg:int = self._convert(cmd, state)
            if reg > 1:
                state.free_reg(reg)

    def _convert_assign(self, state:State, target:Assignable, reg:int) -> None:
        if isinstance(target, Var):
            token:Token = target.identifier
            state.write_env(token)
            state.append_bast(BStoreLoadDict(op_to_err(target),
                                             token.token, reg, True))
        elif isinstance(target, Op):
            # res_reg:int = state.get_free_reg()
            if target.op == "simple_idx":
                args:list[int] = []
                for exp in target.args:
                    args.append(self._convert(exp, state))
                with state.get_free_reg_wrapper() as idx_reg:
                    state.append_bast(BStoreLoadDict(op_to_err(target),
                                                     "$simple_idx=",
                                                     idx_reg, False))
                    state.append_bast(BCall(op_to_err(target),
                                            [0,idx_reg]+args+[reg],
                                            clear=args))
                for _reg in args:
                    state.mark_as_free_reg(_reg)
            elif target.op == ".":
                if not isinstance(target.args[1], Var):
                    raise NotImplementedError("Impossible")
                attr:str = target.args[1].identifier.token
                obj_reg:int = self._convert(target.args[0], state)
                state.append_bast(BDotDict(op_to_err(target), obj_reg,
                                           attr, reg, True))
                state.free_reg(obj_reg)
            elif target.op == "·,·":
                with state.get_free_reg_wrapper() as idx_reg:
                    state.append_bast(BStoreLoadDict(op_to_err(target),
                                                     "$simple_idx",
                                                     idx_reg, False))
                    for i, subtarget in enumerate(target.args):
                        subvalue:int = state.get_free_reg()
                        lit_reg:int = state.get_free_reg()
                        state.append_bast(BLiteral(op_to_err(target), lit_reg,
                                                   BLiteralInt(i),
                                                   BLiteral.INT_T))
                        args:list[int] = [subvalue, idx_reg, reg, lit_reg]
                        state.append_bast(BCall(op_to_err(target), args,
                                                clear=[lit_reg]))
                        state.mark_as_free_reg(lit_reg)
                        self._convert_assign(state, subtarget, subvalue)
                        state.mark_as_free_reg(subvalue)
            else:
                raise NotImplementedError("Impossible")
        else:
            raise NotImplementedError("Impossible")

    def _convert(self, cmd:Cmd, state:State, name:str="*anonymous*") -> int:
        assert isinstance(cmd, Cmd), "TypeError"
        if not isinstance(cmd, NonLocal):
            state.first_inst:bool = False
        res_reg:int = 0
        if isinstance(cmd, Assign):
            for target in cmd.targets:
                if isinstance(target, Var):
                    name:str = target.identifier.token
                    if name == CONSTRUCTOR_NAME:
                        break
            reg:int = self._convert(cmd.value, state, name=name)
            for target in cmd.targets:
                self._convert_assign(state, target, reg)
            state.free_reg(reg)

        elif isinstance(cmd, Literal):
            value:Token = cmd.literal
            if value.isint():
                literal:int = int(value.token)
                if literal in (0, 1):
                    res_reg:int = literal
                else:
                    res_reg:int = state.get_free_reg()
                    state.append_bast(BLiteral(op_to_err(cmd), res_reg,
                                               BLiteralInt(literal),
                                               BLiteral.INT_T))
            elif value.isstring():
                res_reg:int = state.get_free_reg()
                val:BLiteralStr = BLiteralStr(value.token[1:])
                state.append_bast(BLiteral(op_to_err(cmd), res_reg, val,
                                           BLiteral.STR_T))
            elif value in ("true", "false"):
                res_reg:int = state.get_free_reg()
                state.append_bast(BLiteral(op_to_err(cmd), res_reg,
                                           BLiteralBool(value == "true"),
                                           BLiteral.BOOL_T))
            elif value == "none":
                res_reg:int = state.get_none_reg()
            elif value.isfloat():
                res_reg:int = state.get_free_reg()
                val:BLiteralFloat = BLiteralFloat(float(value.token))
                state.append_bast(BLiteral(op_to_err(cmd), res_reg, val,
                                           BLiteral.FLOAT_T))
            else:
                raise NotImplementedError("Impossible")

        elif isinstance(cmd, Var):
            res_reg:int = state.get_free_reg()
            state.assert_read(cmd, cmd.identifier, res_reg)

        elif isinstance(cmd, Op):
            if cmd.op == "if":
                # Set up
                if_id:int = self._labels.get_fl()
                label_true:Bable = Bable("if_true_"+str(if_id))
                label_end:Bable = Bable("if_end_"+str(if_id))
                # Condition
                reg:int = self._convert(cmd.args[0], state)
                state.append_bast(BJump(op_to_err(cmd), label_true.id, reg,
                                        False, clear=[reg]))
                res_reg:int = state.get_free_reg()
                state.mark_as_free_reg(reg)
                # If-false
                tmp_reg:int = self._convert(cmd.args[2], state)
                state.append_bast(BRegMove(res_reg, tmp_reg))
                state.free_reg(tmp_reg)
                state.append_bast(BJump(EMPTY_ERR, label_end.id, 1, False,
                                        clear=[]))
                # If-true
                state.append_bast(label_true)
                state.mark_as_free_reg(reg)
                tmp_reg:int = self._convert(cmd.args[1], state)
                state.append_bast(BRegMove(res_reg, tmp_reg))
                state.free_reg(tmp_reg)
                state.append_bast(label_end)
            elif cmd.op == "·,·":
                new_cmd:Cmd = Op(cmd.ft, cmd.lt, cmd.op.name_as("[]"),
                                 *cmd.args)
                return self._convert(new_cmd, state, name)
            elif cmd.op == ".":
                if not isinstance(cmd.args[1], Var):
                    raise NotImplementedError("Impossible")
                res_reg:int = state.get_free_reg()
                attr:str = cmd.args[1].identifier.token
                obj_reg:int = self._convert(cmd.args[0], state)
                state.append_bast(BDotDict(op_to_err(cmd), obj_reg, attr,
                                           res_reg, False))
                state.free_reg(obj_reg)
            elif cmd.op in ("and", "or"):
                # Set up
                res_reg:int = state.get_free_reg()
                or_id:int = self._labels.get_fl()
                negate_if:bool = cmd.op == "and"
                label_or:Bable = Bable(f"or_{or_id}")
                # Arg 1
                tmp_reg:int = self._convert(cmd.args[0], state)
                state.append_bast(BRegMove(res_reg, tmp_reg))
                state.append_bast(BJump(op_to_err(cmd), label_or.id, tmp_reg,
                                        negate_if, [tmp_reg]))
                state.mark_as_free_reg(tmp_reg)
                # Arg 2
                tmp_reg:int = self._convert(cmd.args[1], state)
                state.append_bast(BRegMove(res_reg, tmp_reg))
                state.free_reg(tmp_reg)
                # End
                state.append_bast(label_or)
            else:
                res_reg:int = state.get_free_reg()
                used_regs:list[int] = [0]
                if cmd.op in RENAME_BUILTIN_FUNCS:
                    op = cmd.op.name_as(RENAME_BUILTIN_FUNCS[cmd.op])
                else:
                    op = cmd.op
                if op == "call":
                    for arg in cmd.args[1:]:
                        reg:int = self._convert(arg, state)
                        used_regs.append(reg)
                    reg:int = self._convert(cmd.args[0], state)
                    used_regs[0] = reg
                else:
                    for arg in cmd.args:
                        reg:int = self._convert(arg, state)
                        used_regs.append(reg)
                    func:int = state.get_free_reg()
                    state.assert_read(Var(op), op, func)
                    used_regs[0] = func
                state.append_bast(BCall(op_to_err(cmd), [res_reg]+used_regs,
                                        clear=used_regs))
                for reg in used_regs:
                    state.mark_as_free_reg(reg)

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
            state.append_bast(BJump(op_to_err(cmd), label_true.id, reg, False,
                                    [reg]))
            state.mark_as_free_reg(reg)
            other_state:State = state.copy_for_branch()
            for subcmd in cmd.false:
                tmp_reg:int = self._convert(subcmd, state)
                state.free_reg(tmp_reg)
            state.append_bast(BJump(op_to_err(cmd), label_end.id, 1, False,
                                    clear=[]))
            state.append_bast(label_true)
            state.mark_as_free_reg(reg)
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
            state.append_bast(BJump(op_to_err(cmd), label_end.id, reg, True,
                                    [reg]))
            state.mark_as_free_reg(reg)
            state.loop_labels.append((label_start.id, label_end.id))
            other_state:State = state.copy_for_branch()
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(subcmd, other_state)
                other_state.free_reg(tmp_reg)
            state.loop_labels.pop()
            state.append_bast(BJump(EMPTY_ERR, label_start.id, 1, False,
                                    clear=[]))
            state.append_bast(label_end)
            state.mark_as_free_reg(reg)
            state.merge_branch(cmd, other_state)

        elif isinstance(cmd, Func):
            func_id:int = self._labels.get_fl()
            label:str = f"func_{name.lower()}_{func_id}"
            # <INSTRUCTIONS>:
            #   res_reg := func_label
            res_reg:int = state.get_free_reg()
            link:int = state.get_class_chain()
            defaults:list[int] = []
            for arg in cmd.args:
                if arg.default is None:
                    if defaults:
                        arg.throw("Argument with no default follows one " \
                                  "with default")
                else:
                    defaults.append(self._convert(arg.default, state))
            max_args:int = len(cmd.args)
            min_args:int = max_args - len(defaults)
            record:bool = (name != CONSTRUCTOR_NAME) and state.is_global_scope()
            func_literal:BLiteralFunc = BLiteralFunc(0, label, min_args,
                                                     max_args, name, link,
                                                     defaults, record=record)
            link:int = state.get_class_chain()
            state.append_bast(BLiteral(op_to_err(cmd), res_reg, func_literal,
                                       BLiteral.FUNC_T))
            for arg in defaults:
                state.free_reg(arg)
            is_constructor:bool = state.class_state and \
                                  (name == CONSTRUCTOR_NAME)
            def todo() -> None:
                # <NEW BODY>:
                # label_start:
                #   <copy args>
                #   <function-body>
                label_start:Bable = Bable(label)
                nstate:State = state.copy_for_func()
                self._states.append(nstate)
                nstate.append_bast(label_start)
                for i, arg in enumerate(cmd.args, start=2):
                    token:Token = arg.identifier
                    nstate.write_env(token)
                    nstate.append_bast(BStoreLoadDict(op_to_err(arg),
                                                      token.token, i, True))
                for i, subcmd in enumerate(cmd.body):
                    tmp_reg:int = self._convert(subcmd, nstate)
                    if (i == len(cmd.body)-1) and isinstance(subcmd, Expr):
                        # If last cmd is an Expr and not constructor, return it
                        if not is_constructor:
                            nstate.append_bast(BRet(EMPTY_ERR, tmp_reg))
                    nstate.free_reg(tmp_reg)
                if not nstate.must_ret: # return `none` or `self`
                    if is_constructor:
                        reg:int = nstate.get_free_reg()
                        arg:Var = cmd.args[0]
                        nstate.append_bast(BStoreLoadDict(op_to_err(arg),
                                                          arg.identifier.token,
                                                          reg, False))
                    else:
                        reg:int = nstate.get_none_reg()
                    nstate.append_bast(BRet(EMPTY_ERR, reg))
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
            # if state.class_state:
            #     cmd.ft.throw("class scope is the same as the outter scope " \
            #                  "and the nonlocal keyword is disallowed")
            for identifier_token in cmd.identifiers:
                identifier:str = identifier_token.token
                if identifier not in state.nonlocals:
                    state.assert_can_nonlocal(identifier_token)
                    state.nonlocals[identifier] = 1
                link:int = state.nonlocals[identifier]
                state.append_bast(BLoadLink(op_to_err(cmd), identifier, link))

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
                state.append_bast(BRet(op_to_err(cmd), reg))
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
            state.append_bast(BJump(EMPTY_ERR, label, 1, False,
                                    clear=[]))
            state.must_end_loop:bool = True # Must be at the end

        elif isinstance(cmd, Class):
            res_reg:int = state.get_free_reg()
            cls_init:int = state.get_free_reg()
            cls_label_id:int = self._labels.get_fl()
            label:str = f"cls_{name.lower()}_{cls_label_id}"
            bases:list[Reg] = []
            for base in cmd.bases:
                bases.append(self._convert(base, state))
            link:int = state.get_class_chain()
            cls_literal:BLiteralClass = BLiteralClass(bases, name)
            cls_init_lit:BLiteralFunc = BLiteralFunc(0, label, 1, 1, label,
                                                     link, [], record=False)
            state.append_bast(BLiteral(op_to_err(cmd), res_reg, cls_literal,
                                       BLiteral.CLASS_T))
            state.append_bast(BLiteral(op_to_err(cmd), cls_init, cls_init_lit,
                                       BLiteral.FUNC_T))
            state.append_bast(BCall(op_to_err(cmd), [0,cls_init,res_reg],
                                    clear=[cls_init]))
            state.mark_as_free_reg(cls_init)
            for base in bases:
                state.free_reg(base)
            def todo() -> None:
                nstate:State = state.copy_for_class()
                nstate.append_bast(Bable(label))
                cls_obj_reg:int = nstate.get_free_reg()
                assert cls_obj_reg == CLS_REG, "InternalError"
                for assignment in cmd.body:
                    self._convert(assignment, nstate)
                nstate.free_reg(cls_obj_reg)
                nstate.append_bast(BRet(op_to_err(cmd), 0))
                nstate.fransform_class()
            # self._todo.append(todo)
            todo()

        else:
            raise NotImplementedError(f"Not implemented {cmd!r}")

        return res_reg

    def to_file(self, filepath:str) -> None:
        assert isinstance(filepath, str), "TypeError"
        with open(filepath, "wb") as file:
            file.write(self.serialise())


def token_to_idx_pair(token:Token, *, last:bool) -> tuple[int,int]:
    return (token.stamp.line,
            token.stamp.char + token.size*last)

def op_to_err(op:Cmd) -> ErrorIdx:
    err = ErrorIdx(token_to_idx_pair(get_first_token(op), last=False),
                    token_to_idx_pair(get_last_token(op), last=True))
    return err


RENAME_BUILTIN_FUNCS:dict[str:str] = {
                                       "idx": "$idx",
                                       "idx=": "$idx=",
                                       "simple_idx": "$simple_idx",
                                       "simple_idx=": "$simple_idx=",
                                     }


if __name__ == "__main__":
    TEST1 = r"""
io.print("####### General stuff ########")
io.print(2, "\t", int(0.1+0.2==0.3), int(0.3==0.1+0.2))

f = func(x){func(){x}}
c = func(f){
    x = y = g = 0
    f()
}
x = 5
y = 2
io.print(1, "\t", int(7==c(f(x+y))))


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
io.print(1, "\t", int(5==z))


x = [5, 10, 15]
io.print(1, "\t", int(10==x[1]))
x[1] = 15
x.append(20)
io.print(2, "\t", int(15==x[1]), int(20==x[3]))
io.print(2, "\t", int(str.join("", x)=="5151520"),
                  int(", ".join(x)=="5, 15, 15, 20"))
io.print(4, "\t", int("abcdef".index("cd")==2), int("abcdef".index("cf")==-1),
                  int(x.index(15)==1), int(x.index(200)==-1))


io.print("########## Closures ##########")
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
    return [g, (func(){nonlocal x; x})]
}()
add = tmp[0]
get = tmp[1]
add()
add()
io.print(2, "\t", int(25==x), int(get()+5==30))


# Similar to the one above
f = func(x){[func(){x}, func(){nonlocal x;x+=1}]}
tmp = f(1)
get = tmp[0]
add = tmp[1]
io.print(2, "\t", int(get()*15==15), int(get()*20==20))
add()
io.print(1, "\t", int(get()*12+1==25))
add()
io.print(1, "\t", int(get()*10==30))


Y = func(f){
    return func(x){ f(f(f(f(f(f(f(f)))))))(x) }
}
fact_helper = func(rec){
    return func(n){
        1 if n == 0 else n*rec(n-1)
    }
}
fact = Y(fact_helper)
io.print(1, "\t", int(120==fact(5)))


io.print("########### Lists ############")
x = [1, 2, 3, 4, 5]
a = x[:2]
b = x[2:]
c = x[1:2]
d = x[-2:]
e = x[-2:-1]
f = x[::-1]
i = 1
æ, ß = [2, 4, func(){nonlocal i; i+=1}()]
io.print(5, "\t", int(a.len()==2), int(a[0]==1), int(a[1]==2), int(b.len()==3),
                  int(b[0]==3))
io.print(5, "\t", int(b[1]==4), int(b[2]==5), int(c.len()==1), int(c[0]==2),
                  int(d.len()==2))
io.print(5, "\t", int(d[0]==4), int(d[1]==5), int(e.len()==1), int(e[0]==4),
                  int(f.len()==5))
io.print(5, "\t", int(f[0]==5), int(f[-1]==1), int(f[1]==4),
                  int("abcd"=="dcba"[::-1]), int("abcdef"[1:-1:2]=="bd"))
io.print(3, "\t", int(æ==2), int(ß==4), int(i==2))


io.print("####### Partial funcs ########")
add = /? + ?/
add80 = /80 + 2*?/
add120 = /? + 120/
io.print(3, "\t", int(200==add(120,80)), int(200==add120(80)),
                  int(200==add80(60)))


io.print("###### Mutual recursion ######")
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
io.print(5, "\t", int(isodd(101)), int(iseven(246)), int(not isodd(246)),
                  int(not iseven(101)), int(isodd(1)))
io.print(1, "\t", int(iseven(0)))


io.print("#### And/Or short-circuit ####")
⊥ = func(){⊥()}
io.print(5, "\t", int((5 or ⊥())==5), int((0 or 1)==1), int((1 and 2)==2),
                  int((0 and ⊥())==0), int((1 and 0)==0))


io.print("######## Object model ########")
unbounded_funcs = []
bounded_funcs = []
classes = []
objects = []

t = 0
A = class {
    nonlocal t
    X = 5
    f = func(self) {
        nonlocal t
        if (self != 7) {
            objects.append(self)
        } else {
            classes.append(A)
        }
        io.print(1, "\t", int(A.X==6))
        t = A.X + 2
        A.X = 0
    }
    t = X
    if X {a=1}
    Y = class {
        Z = class {
            Δ = 13
        }
    }
}
io.print(2, "\t", int(t==5), int(A.a==1))
A.X = 6
A.f(7)
io.print(3, "\t", int(t==8), int(A.X==0), int(A.Y.Z.Δ==13))

a = A()
objects.append(a)
A.X = 6
a.f()

A = class {
    X = 0
}
B = class(A) {
    Y = 0
    f = func() {}
}
unbounded_funcs.append(B.f)
bounded_funcs.append(B().f)

io.print(3, "\t", int(A.X==0), int(B.X==0), int(B.Y==0))
B.X = 1
io.print(3, "\t", int(A.X==0), int(B.X==1), int(B.Y==0))
A.X = 2
io.print(3, "\t", int(A.X==2), int(B.X==1), int(B.Y==0))


io.print("######## Is instance #########")
A = class {}
a = A()
io.print(5, "\t", int(isinstance(a, A)), int(not isinstance(a, int)),
                  int(isinstance(1, float)), int(not isinstance(1, A)),
                  int(isinstance("a", str)))
io.print(2, "\t", int(isinstance(true, int)), int(isinstance(false, bool)))


io.print("############ Math ############")
io.print(5, "\t", int(math.round(-0.1691,6)==-0.1691),
                  int(math.round(-0.1691,5)==-0.1691),
                  int(math.round(-0.1691,4)==-0.1691),
                  int(math.round(-0.1691,3)==-0.169),
                  int(math.round(-0.1691,2)==-0.17))
io.print(5, "\t", int(math.round(-0.1691,1)==-0.2),
                  int(math.round(-0.1691,0)==0),
                  int(math.round(-0.1691,-1)==0),
                  int(math.round(-0.1691,-2)==0),
                  int(math.round(-0.1691,-3)==0))
io.print(5, "\t", int(math.round(12,0)==12), int(math.round(12,1)==12),
                  int(math.round(12,2)==12), int(math.round(12,-1)==10),
		  int(math.round(15,-1)==20))
io.print(5, "\t", int(math.round(19,-1)==20), int(math.round(false,0)==0),
                  int(math.round(false,1)==0), int(math.round(false,-1)==0),
                  int(math.round(true,0)==1))
io.print(5, "\t", int(math.round(true,1)==1), int(math.round(true,-1)==0),
                  int(math.round(1.2,1)==1.2), int(math.round(1.2,2)==1.2),
                  int(math.round(1.2,0)==1))
io.print(5, "\t", int(math.round(1.2,-1)==0), int(math.round(1.5,1)==1.5),
                  int(math.round(1.5,2)==1.5), int(math.round(1.5,0)==2),
                  int(math.round(1.5,-1)==0))
io.print(5, "\t", int(math.round(0.05,1)==0.1), int(math.round(-0.05,1)==-0.1),
                  int(math.round(0.5,0)==1), int(math.round(-0.5,0)==-1),
                  int(math.round(5,-1)==10))
io.print(3, "\t", int(math.round(-5,-1)==-10), int(math.round(-4,-1)==0),
                  int(math.round(4,-1)==0))


io.print("########## Round str #########")
io.print(5, "\t", int(math.str_round(7135.8642,-6)=="0"),
                  int(math.str_round(7135.8642,-5)=="0"),
                  int(math.str_round(7135.8642,-4)=="10000"),
                  int(math.str_round(7135.8642,-3)=="7000"),
                  int(math.str_round(7135.8642,-2)=="7100"))
io.print(5, "\t", int(math.str_round(7135.8642,-1)=="7140"),
                  int(math.str_round(7135.8642,0)=="7136"),
                  int(math.str_round(7135.8642,1)=="7135.9"),
                  int(math.str_round(7135.8642,2)=="7135.86"),
                  int(math.str_round(7135.8642,3)=="7135.864"))
io.print(5, "\t", int(math.str_round(7135.8642,4)=="7135.8642"),
                  int(math.str_round(7135.8642,5)=="7135.86420"),
                  int(math.str_round(7135.8642,6)=="7135.864200"),
                  int(math.str_round(0.0999,-5)=="0"),
                  int(math.str_round(0.0999,-4)=="0"))
io.print(5, "\t", int(math.str_round(0.0999,-3)=="0"),
                  int(math.str_round(0.0999,-2)=="0"),
                  int(math.str_round(0.0999,-1)=="0"),
                  int(math.str_round(0.0999,0)=="0"),
                  int(math.str_round(0.0999,1)=="0.1"))
io.print(5, "\t", int(math.str_round(0.0999,2)=="0.10"),
                  int(math.str_round(0.0999,3)=="0.100"),
                  int(math.str_round(0.0999,4)=="0.0999"),
                  int(math.str_round(0.0999,5)=="0.09990"),
                  int(math.str_round(0.0999,6)=="0.099900"))
io.print(5, "\t", int(math.str_round(5,-1)=="10"),
                  int(math.str_round(0.5,0)=="1"),
                  int(math.str_round(0.05,1)=="0.1"),
                  int(math.str_round(0.005,2)=="0.01"),
                  int(math.str_round(true,-2)=="0"))
io.print(5, "\t", int(math.str_round(true,-1)=="0"),
                  int(math.str_round(true,0)=="1"),
                  int(math.str_round(true,1)=="1.0"),
                  int(math.str_round(true,2)=="1.00"),
                  int(math.str_round(true,3)=="1.000"))
io.print(5, "\t", int(math.str_round(-7135.8642,-6)=="0"),
                  int(math.str_round(-7135.8642,-5)=="0"),
                  int(math.str_round(-7135.8642,-4)=="-10000"),
                  int(math.str_round(-7135.8642,-3)=="-7000"),
                  int(math.str_round(-7135.8642,-2)=="-7100"))
io.print(5, "\t", int(math.str_round(-7135.8642,-1)=="-7140"),
                  int(math.str_round(-7135.8642,0)=="-7136"),
                  int(math.str_round(-7135.8642,1)=="-7135.9"),
                  int(math.str_round(-7135.8642,2)=="-7135.86"),
                  int(math.str_round(-7135.8642,3)=="-7135.864"))
io.print(5, "\t", int(math.str_round(-7135.8642,4)=="-7135.8642"),
                  int(math.str_round(-7135.8642,5)=="-7135.86420"),
                  int(math.str_round(-7135.8642,6)=="-7135.864200"),
                  int(math.str_round(-0.0999,-5)=="0"),
                  int(math.str_round(-0.0999,-4)=="0"))
io.print(5, "\t", int(math.str_round(-0.0999,-3)=="0"),
                  int(math.str_round(-0.0999,-2)=="0"),
                  int(math.str_round(-0.0999,-1)=="0"),
                  int(math.str_round(-0.0999,0)=="0"),
                  int(math.str_round(-0.0999,1)=="-0.1"))
io.print(5, "\t", int(math.str_round(-0.0999,2)=="-0.10"),
                  int(math.str_round(-0.0999,3)=="-0.100"),
                  int(math.str_round(-0.0999,4)=="-0.0999"),
                  int(math.str_round(-0.0999,5)=="-0.09990"),
                  int(math.str_round(-0.0999,6)=="-0.099900"))
io.print(5, "\t", int(math.str_round(-5,-1)=="-10"),
                  int(math.str_round(-0.5,0)=="-1"),
                  int(math.str_round(-0.05,1)=="-0.1"),
                  int(math.str_round(-0.005,2)=="-0.01"),
                  int(math.str_round(-true,-2)=="0"))
io.print(5, "\t", int(math.str_round(-true,-1)=="0"),
                  int(math.str_round(-true,0)=="-1"),
                  int(math.str_round(-true,1)=="-1.0"),
                  int(math.str_round(-true,2)=="-1.00"),
                  int(math.str_round(-true,3)=="-1.000"))


io.print("###### Default func args #####")
f = func(a=5, b=8) { a+b }
io.print(3, "\t", f(0,0)+1, f(0)-7, f()-12)


io.print("#### String split/replace ####")
io.print(2, "\t", int("abc\"deafa".split("a")==["","bc\"de","f",""]),
                  int("abc\"deafa".split("a", 2)==["","bc\"de","fa"]))
io.print(3, "\t", int("acbcdecf".replace("c","#")=="a#b#de#f"),
                  int("acbcdecf".replace("c","")=="abdef"),
                  int("acbcdecf".replace("c","", 1)=="abcdecf"))


io.print("These should be classes:\t\t", classes)
io.print("These should be instances:\t\t", objects)
io.print("These should be unbounded functions:\t", unbounded_funcs)
io.print("These should be bounded functions:\t", bounded_funcs)
"""[1:-1], False

    TEST2 = """
fib = func(x) ->
    if x < 1 ->
        return 1
    return fib(x-2) + fib(x-1)
io.print(fib(15), "should be", 1597)
io.print(fib(30), "should be", 2178309)
io.print(fib(36), "should be", 39088169)
io.print("cpython takes 3.474 sec")
io.print("pypy takes 0.187 sec")
io.print("lizzzard takes 0.833 sec")
"""[1:-1], True

    TEST3 = """
max = 100_000
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
    primes.append(i)
}
io.print("the number of primes bellow", max, "is:", primes.len(),
         "which should be", 9592)
io.print("the last prime is:", primes[-1], "which should be", 99991)
io.print("cpython takes 61.300 sec")
io.print("pypy takes 1.689 sec")
io.print("lizzzard takes 1.728 sec")
"""[1:-1], False

    TEST4 = """
i = 0
while i < 100_000_000 ->
    i += 1
io.print("while++", i)
io.print("cpython takes 6.320 sec")
io.print("pypy takes 0.092 sec")
io.print("lizzzard takes 0.234 sec")
"""[1:-1], True

    TEST5 = """
f = func(x) {
    if (x) {
        return f(x-1) + 1
    }
    return x
}
io.print("rec++", f(1_000_000))
io.print("cpython takes 0.177 sec")
io.print("pypy seg faults")
io.print("lizzzard takes 0.309 sec")
"""[1:-1], False

    TEST6 = """
# class {}
A = class {
    A = B = 0
}
A.X = 1
while (A.X < 100_000_000) {
    A.X += 1
}
io.print("attr++", A.X)
io.print("cpython takes 13.463 sec")
io.print("pypy takes 0.113 sec")
io.print("lizzzard takes 0.231 sec")
"""[1:-1], False

    TEST7 = """
f = func(x) {
    if (x) {
        return f(x-1) + 1
    }
    return x + ""
}

f(2)
"""[1:-1], False

    TEST9 = """
assert = func(boolean, msg) {
    if (not boolean) {
        io.print(msg)
        io.print(1/0)
    }
}

A = class {
    __init__ = func(this, x) {
        assert(isinstance(x, int), "TypeError")
        this.x = x
    }

    add = func(this, other) {
        assert(isinstance(other, A), "TypeError")
        return A((this.x + other.x) % 100)
    }
}

a = A(1)

i = 0
while (i < 5_000) {
    i += 1
    if (i % 2) {
        a = A(i) # a.add(a)
    }
}
"""[1:-1], False

    TEST9 = """
A = class {}
f = func() { a = A() }

i = 0
while (i < 100_000_000) {
    i += 1
    if (i&1 == 0) { ... }
    f()
}
"""[1:-1], False

    TEST10 = """
A = class {
    __init__ = func(this, x) {
        this.x = x
    }
    add = func(this, other) {
        if (not other.x) {
            return A(this.x)
        } else {
            new = this.add(A(other.x - 1))
            new.x += 1
            return new
        }
    }
}

a = A(0).add(A(1_000_000))
io.print("wrapped++", a.x)
"""[1:-1], False

    TEST11 = """
WrappedInt = class {
    __init__ = func(this, x) {
        this.x = x
    }
    add = func(this, other) { WrappedInt(this.x+other.x) }
}

ONE = WrappedInt(1)
a = WrappedInt(0)
while (a.x < 10_000_000) {
    a = a.add(ONE)
}
io.print("obj_creat++", a.x)
"""[1:-1], False

    TEST12 = """
A = class {y=0}
A = class {}

a = A()
a.x = 0
while (a.x < 1000) {
    a.x += 1
}
"""[1:-1], False

    TEST13 = r"""
"""[1:-1], False

    if len(argv) == 13:
        with open("code-examples/raytracer.lizz", "r") as file:
            TEST8 = (file.read(), False)

    # DEBUG_RAISE:bool = True
    # 1:all, 2:fib, 3:primes, 4:while++, 5:rec++, 6:attr++, 7:err, 8:raytracer
    # 9:problem
    TEST = TEST1
    if len(argv) > 1:
        with open(argv[1], "r") as file:
            TEST = (file.read(), False)
    assert not isinstance(TEST, str), "TEST should be tuple[str,bool]"
    parser:Parser = Parser(Tokeniser(StringIO(TEST[0])), colon=TEST[1])
    ast:Body = parser.read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        flags = FeatureFlags()
        flags.set("ENV_IS_LIST")
        flags.set("CLEAR_AFTER_USE")

        bytecoder:ByteCoder = ByteCoder(ast, parser.get_all_text(), flags)
        (frame_size, env_size, attrs,
         bytecode_blocks, code) = bytecoder.to_bytecode()
        bytecode:Block = reduce(iconcat, bytecode_blocks, [])
        if bytecode:
            if len(bytecode) > 500:
                print(f"<{len(bytecode)} bytecode instructions>")
            else:
                print(bytecode_list_to_str(bytecode))

            raw_bytecode:bytes = bytecoder.serialise()
            filepath:str = "code-examples/example.clizz"
            if len(argv) > 1:
                filepath:str = argv[1].removesuffix(".lizz") + ".clizz"
            with open(filepath, "wb") as file:
                file.write(b".clizz file" + raw_bytecode)

            data = derialise(raw_bytecode)
            (dec_flags, dec_frame_size,
             dec_env_size, dec_attrs, dec_bast, dec_code) = data
            print(f"{frame_size=}, {flags=}")
            assert bytecode_list_to_str(bytecode) == \
                   bytecode_list_to_str(dec_bast), "AssertionError"
            assert frame_size == dec_frame_size, "AssertionError"
            assert flags.issame(dec_flags), "AssertionError"
            if flags.is_set("ENV_IS_LIST"):
                assert env_size == dec_env_size, "AssertionError"
                assert attrs == dec_attrs, "AssertionError"
            assert dec_code == code

    # from time import perf_counter
    # fib = lambda x: 1 if x < 1 else fib(x-2)+fib(x-1)
    # s = perf_counter(); fib(15); fib(30); perf_counter()-s