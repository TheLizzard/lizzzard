from __future__ import annotations
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
        self._taken:list[bool] = [True,True,True] # 0,1,return
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
            instructions.append(BLiteral(reg, BNONE, BLiteral.NONE_T))

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
                "flags"

    def __init__(self, *, blocks:list[Block], nonlocals:Nonlocals,
                 block:Block, loop_labels:list[LoopLabel], regs:Regs,
                 env:Env, not_env:EnvReason, flags:FeatureFlags,
                 master:State=None) -> State:
        self.loop_labels:list[LoopLabel] = loop_labels
        self.nonlocals:Nonlocals = nonlocals
        self.blocks:list[Block] = blocks
        self.not_env:EnvReason = not_env
        self.flags:FeatureFlags = flags
        self.first_inst:bool = True
        self.must_ret:bool = False
        self.master:State = master
        self.block:Block = block
        self.regs:Regs = regs
        self.env:Env = env

    # Copy helpers
    def copy_for_func(self) -> State:
        new_nl:NonLocals = {name:link+1 for name,link in self.nonlocals.items()}
        state:State = State(blocks=self.blocks, loop_labels=[], block=Block(),
                            nonlocals=new_nl, regs=Regs(self.regs), env=Env(),
                            not_env=self.not_env.copy(), flags=self.flags,
                            master=self)
        state.blocks.append(state.block)
        return state

    def copy_for_branch(self) -> State:
        return State(blocks=self.blocks, loop_labels=self.loop_labels,
                     block=self.block, nonlocals=self.nonlocals,
                     regs=self.regs, env=self.env.copy(), flags=self.flags,
                     not_env=self.not_env.copy(), master=self.master)

    @staticmethod
    def new(flags:FeatureFlags) -> State:
        block:Block = Block()
        return State(block=block, blocks=[block], nonlocals=Nonlocals(),
                     loop_labels=[], regs=Regs(), env=Env(BUILTINS[1:]),
                     not_env=EnvReason(), flags=flags)

    # Env helpers
    def merge_branch(self, branch:Branch, other:State) -> None:
        if self.must_ret:
            self.not_env:EnvReason = other.not_env
            self.env:Env = other.env
        if other.must_ret:
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
                self.env.pop(name)
                if name in other.not_env:
                    self.not_env[name] = other.not_env[name]
                else:
                    self.not_env[name] = branch

    def assert_read(self, name_token:Token, reg:int) -> None:
        assert isinstance(name_token, Token), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert reg > 1, "ValueError"
        state:State = self
        while state is not None:
            name:str = state.guard_env(name_token)
            if name in state.env:
                self.append_bast(BStoreLoadDictEnv(name, reg, False))
                return
            state:State = state.master
        name_token.throw(f"variable {name!r} was not defined")

    def guard_env(self, name_token:Token) -> None:
        name:str = name_token.token
        if name in self.not_env:
            branch_token:Token = self.not_env[name].ft
            name_token.double_throw(f"Variable {name!r} might be undefined " \
                                    f"because of:", branch_token)
        return name

    def assert_can_nonlocal(self, name_token:Token) -> None:
        assert isinstance(name_token, Token), "TypeError"
        name:str = name_token.token
        if name not in self.master.env:
            name_token.throw(f"variable {name!r} not defined in parent scope")

    def write_env(self, name:str) -> None:
        assert isinstance(name, str), "TypeError"
        if name in self.nonlocals:
            # TODO: merge name with nonlocals
            pass
        else:
            if name in self.not_env:
                self.not_env.pop(name)
            self.env.append(name)

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
        if not self.flags.is_set("CLEAR_AFTER_USE"):
            self.append_bast(BLiteral(reg, BLiteralEmpty(), BLiteral.NONE_T))
        return reg

    # Block helpers
    def append_bast(self, instruction:Bast) -> None:
        assert isinstance(instruction, Bast), "TypeError"
        self.block.append(instruction)

    # Fransform
    def fransform(self) -> None:
        if not self.flags.is_set("ENV_IS_LIST"):
            return None
        nonlocals:Nonlocals = Nonlocals()
        fransformed_block:Block = Block()
        for i, bt in enumerate(self.block):
            if isinstance(bt, BLoadLink):
                nonlocals[bt.name] = bt.link
                continue
            elif isinstance(bt, BStoreLoadDictEnv):
                scope, link = self, 0
                if bt.name in nonlocals:
                    for i in range(nonlocals[bt.name]):
                        scope, link = scope.master, link+1
                else:
                    while bt.name not in scope.full_env:
                        scope, link = scope.master, link+1
                idx:int = scope.full_env.index(bt.name) + 1
                bt:Bast = BStoreLoadListEnv(link, idx, bt.reg, bt.storing)
            fransformed_block.append(bt)
        self.block.clear()
        self.block.extend(fransformed_block)

    @property
    def full_env(self) -> Env:
        return self.env + sorted(list(self.not_env))


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

    def _to_bytecode(self) -> tuple[int,int,list[Block]]:
        state:State = State.new(self._flags)
        self._states:list[State] = [state]
        self._labels.reset()
        for cmd in self.ast:
            reg:int = self._convert(cmd, state)
            state.free_reg(reg)
        tmp_reg:int = state.get_free_reg()
        state.append_bast(BLiteral(tmp_reg, BLiteralInt(0), BLiteral.INT_T))
        state.append_bast(BRegMove(2, tmp_reg))
        state.free_reg(tmp_reg)
        state.fransform()
        while self._todo:
            self._todo.pop(0)()
        return state.regs.max_reg+1, len(state.full_env)+1, state.blocks

    def to_bytecode(self) -> tuple[RegsSize,EnvSize,list[Block]]:
        try:
            return self._to_bytecode()
        except (SemanticError, FinishedWithError) as error:
            if DEBUG_RAISE and isinstance(error, FinishedWithError):
                print(traceback.format_exc(), end="")
            print(f"\x1b[91mSemanticError: {error.msg}\x1b[0m", file=stderr)
            return 3, 0, []

    def _serialise(self, frame_size:int, env_size:int, bytecode:Block) -> bytes:
        bast:bytearray = bytearray()
        for instruction in bytecode:
            bast += instruction.serialise()
        return serialise_int(VERSION, VERSION_SIZE) + \
               self._flags.serialise() + \
               serialise_int(frame_size, FRAME_SIZE) + \
               serialise_int(env_size, ENV_SIZE_SIZE) + \
               bytes(bast)

    def serialise(self) -> bytes:
        frame_size, env_size, bytecode_blocks = self.to_bytecode()
        bytecode:Block = reduce(iconcat, bytecode_blocks, [])
        return self._serialise(frame_size, env_size, bytecode)

    def _convert(self, cmd:Cmd, state:State) -> int:
        if not isinstance(cmd, NonLocal):
            state.first_inst:bool = False
        res_reg:int = 0
        if isinstance(cmd, Assign):
            for target in cmd.targets:
                if isinstance(target, Var):
                    name:str = target.identifier.token
                    state.write_env(name)
            reg:int = self._convert(cmd.value, state)
            for target in cmd.targets:
                if isinstance(target, Var):
                    state.append_bast(BStoreLoadDictEnv(target.identifier.token,
                                                        reg, True))
                elif isinstance(target, Op):
                    # res_reg:int = state.get_free_reg()
                    if target.op == "simple_idx":
                        args:list[int] = []
                        for exp in target.args:
                            args.append(self._convert(exp, state))
                        tmp_reg:int = state.get_free_reg()
                        state.append_bast(BStoreLoadDictEnv("simple_idx=",
                                                            tmp_reg, False))
                        state.append_bast(BCall([0,tmp_reg] + args + [reg]))
                        state.free_reg(tmp_reg)
                        for tmp_reg in args:
                            state.free_reg(tmp_reg)
                    elif target.op == ".":
                        raise NotImplementedError("TODO")
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
                res_reg:int = value == "true"
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
            else:
                res_reg:int = state.get_free_reg()
                used_regs:list[int] = [res_reg]
                if cmd.op != "call":
                    func:int = state.get_free_reg()
                    state.assert_read(cmd.op, func)
                    used_regs.append(func)
                for arg in cmd.args:
                    reg:int = self._convert(arg, state)
                    used_regs.append(reg)
                state.append_bast(BCall(used_regs))
                for reg in used_regs[1:]:
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
                    name:str = arg.identifier.token
                    assert isinstance(name, str), "TypeError"
                    nstate.write_env(name)
                    nstate.append_bast(BStoreLoadDictEnv(name, i, True))
                for subcmd in cmd.body:
                    tmp_reg:int = self._convert(subcmd, nstate)
                    nstate.free_reg(tmp_reg)
                reg:int = nstate.get_free_reg()
                if not self._flags.is_set("CLEAR_AFTER_USE"):
                    nstate.append_bast(BLiteral(reg, BLiteralEmpty(),
                                                BLiteral.NONE_T))
                nstate.append_bast(BRegMove(2, reg))
                nstate.fransform()
                func_literal.env_size = len(nstate.full_env) + 1
            # Convert the Func to bytecode after the rest of the code in the
            #   currect scope. This allows for mutual recursion
            self._todo.append(todo)

        elif isinstance(cmd, NonLocal):
            if not state.first_inst:
                cmd.ft.throw("nonlocal directives must be at the top of " \
                             "the scope")
            if state.master is None:
                cmd.ft.throw("variables in the global scope can't be nonlocal")
            for identifier_token in cmd.identifiers:
                identifier:str = identifier_token.token
                if identifier not in state.nonlocals:
                    state.assert_can_nonlocal(identifier_token)
                    state.nonlocals[identifier] = 1
                link:int = state.nonlocals[identifier]
                state.append_bast(BLoadLink(identifier, link))

        elif isinstance(cmd, ReturnYield):
            if cmd.isreturn:
                if state.master is None:
                    # Returning from global scope
                    pass
                state.must_ret:bool = True
                if cmd.exp is None:
                    reg:int = state.get_none_reg()
                else:
                    reg:int = self._convert(cmd.exp, state)
                state.append_bast(BRegMove(2, reg))
                state.free_reg(reg)
            else:
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

        else:
            raise NotImplementedError(f"Not implemented {cmd!r}")

        return res_reg

    def to_file(self, filepath:str) -> None:
        assert isinstance(filepath, str), "TypeError"
        with open(filepath, "wb") as file:
            file.write(self.serialise())


if __name__ == "__main__":
    TEST1 = """
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
print(0, "should be", c(f(x+y))-7)


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
print(5, "should be", z)


x = [5, 10, 15]
print(10, "should be", x[1])
x[1] = 15
print(15, "should be", x[1])
append(x, 20)
print(20, "should be", x[3])


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
print(25, "should be", x)
print(30, "should be", get()+5)


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
print(120, "should be", fact(5))


print("The following should be all 1s/`true`s")
x = [1, 2, 3, 4, 5]
a = x[:2]
b = x[2:]
c = x[1:2]
d = x[-2:]
e = x[-2:-1]
f = x[::-1]
print("\t", len(a)==2, a[0]==1, a[1]==2, len(b)==3, b[0]==3, b[1]==4, b[2]==5,
      len(c)==1, c[0]==2, len(d)==2, d[0]==4, d[1]==5, len(e)==1, e[0]==4,
      len(f)==5, f[0]==5, f[-1]==1, f[1]==4)


add = /? + ?/
add80 = /80 + 2*?/
add120 = /? + 120/
print("\t", 200==add(120,80), 200==add120(80), 200==add80(60))
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
odd = func(x){
    if x == 0{
        return 0
    }
    return 1 - even(x-1)
}
even = func(x){
    if x == 0{
        return 1
    }
    return 1 - odd(x-1)
}

print(1, "should be", even(100))
print(0, "should be", odd(100))
"""[1:-1], False

    # DEBUG_RAISE:bool = True
    # TEST1:test_all, TEST2:fib, TEST3:while++, TEST4:primes, TEST5:rec++
    TEST = TEST2
    assert not isinstance(TEST, str), "TEST should be tuple[str,bool]"
    ast:Body = Parser(Tokeniser(StringIO(TEST[0])), colon=TEST[1]).read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        flags = FeatureFlags()
        flags.set("ENV_IS_LIST")
        flags.set("CLEAR_AFTER_USE")

        bytecoder:ByteCoder = ByteCoder(ast, flags)
        frame_size, env_size, bytecode_blocks = bytecoder.to_bytecode()
        bytecode:Block = reduce(iconcat, bytecode_blocks, [])
        if bytecode:
            print(bytecode_list_to_str(bytecode))

            raw_bytecode:bytes = bytecoder.serialise()
            with open("code-examples/example.clizz", "wb") as file:
                file.write(raw_bytecode)

            data = derialise(raw_bytecode)
            dec_flags, dec_frame_size, dec_env_size, dec_bast = data
            print(f"{frame_size=}, {flags=}")
            assert bytecode_list_to_str(bytecode) == \
                   bytecode_list_to_str(dec_bast), "AssertionError"
            assert frame_size == dec_frame_size, "AssertionError"
            assert env_size == dec_env_size, "AssertionError"
            assert flags.issame(dec_flags), "AssertionError"

    # fib = lambda x: 1 if x < 1 else fib(x-2)+fib(x-1)
    # print(fib(30))