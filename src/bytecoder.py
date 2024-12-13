from __future__ import annotations
from functools import reduce
from operator import iconcat
from io import StringIO
from sys import stderr

from frontend.bcast import *
from asts.ccast import *
from parser import *
from lexer import *

# Should we clear regs after use so they are always none?
CLEAR_REGS:bool = True


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

    def free_reg(self, reg:int, instructions:list[Bast]) -> None:
        assert 0 <= reg < len(self._taken), "ValueError"
        if reg in (0, 1):
            return None
        assert self._taken[reg], "reg not taken"
        self._taken[reg] = False
        self.clear_reg(reg, instructions)

    def clear_reg(self, reg:int, instructions:list[Bast]) -> None:
        if CLEAR_REGS:
            instructions.append(BLiteral(reg, BNONE, BLiteral.NONE_T))

    def _notify_max_reg(self, max_reg:int) -> None:
        self.max_reg:int = max(self.max_reg, max_reg)
        if self._parent is not None:
            self._parent._notify_max_reg(self.max_reg)


class State:
    __slots__ = "block", "blocks", "regs", "nonlocals", "loop_labels"

    def __init__(self, *, blocks:list[list[Bast]], nonlocals:dict[str:int],
                 block:list[Bast], loop_labels:list[tuple[str,str]], regs:Regs):
        self.loop_labels:list[tuple[str,str]] = loop_labels
        self.nonlocals:dict[str:int] = nonlocals
        self.blocks:list[list[Bast]] = blocks
        self.block:list[Bast] = block
        self.regs:Regs = regs

    def copy(self) -> State:
        return State(blocks=self.blocks, loop_labels=self.loop_labels,
                     block=self.block, nonlocals=self.nonlocals,
                     regs=self.regs)


class Block(list[Bast]):
    __slots__ = ()

class SemanticError(Exception): ...


class ByteCoder:
    __slots__ = "ast", "_labels"

    def __init__(self, ast:Body) -> ByteCoder:
        self._labels:Labels = Labels()
        self.ast:Body = ast

    def to_bytecode(self) -> tuple[int,list[list[Bast]]]:
        try:
            return self._to_bytecode()
        except SemanticError as error:
            print(f"\x1b[91mSemanticError: {error.msg}\x1b[0m", file=stderr)
            return 3, []

    def _to_bytecode(self) -> tuple[int,list[list[Bast]]]:
        state:State = State(block=Block(), blocks=[], nonlocals={},
                            loop_labels=[], regs=Regs())
        state.blocks.append(state.block)
        self._labels.reset()
        for cmd in self.ast:
            reg:int = self._convert(cmd, state)
            state.regs.free_reg(reg, state.block)
        tmp_reg:int = state.regs.get_free_reg()
        state.block.append(BLiteral(tmp_reg, BLiteralInt(0), BLiteral.INT_T))
        state.block.append(BRegMove(2, tmp_reg))
        state.regs.free_reg(tmp_reg, state.block)
        return state.regs.max_reg, state.blocks

    def serialise(self) -> bytes:
        frame_size, bytecode_blocks = self.to_bytecode()
        bytecode:list[Bast] = reduce(iconcat, bytecode_blocks, [])
        return self._serialise(frame_size, bytecode)

    def _serialise(self, frame_size:int, bytecode:list[Bast]) -> bytes:
        bast:bytearray = bytearray()
        for instruction in bytecode:
            bast += instruction.serialise()
        return serialise_int(VERSION, VERSION_SIZE) + \
               serialise_int(frame_size+1, FRAME_SIZE) + \
               bytes(bast)

    def _convert(self, cmd:Cmd, state:State) -> int:
        res_reg:int = 0
        if isinstance(cmd, Assign):
            reg:int = self._convert(cmd.value, state)
            for target in cmd.targets:
                if isinstance(target, Var):
                    name:str = target.identifier.token
                    state.block.append(BStoreLoad(name, reg, True))
                elif isinstance(target, Op):
                    # res_reg:int = state.regs.get_free_reg()
                    if target.op == "simple_idx":
                        args:list[int] = []
                        for exp in target.args:
                            args.append(self._convert(exp, state))
                        tmp_reg:int = state.regs.get_free_reg()
                        state.block.append(BStoreLoad("simple_idx=", tmp_reg,
                                                      False))
                        state.block.append(BCall([0,tmp_reg] + args + [reg]))
                        state.regs.free_reg(tmp_reg, state.block)
                        for tmp_reg in args:
                            state.regs.free_reg(tmp_reg, state.block)
                    elif target.op == ".":
                        raise NotImplementedError("TODO")
                    elif target.op == "·,·":
                        raise NotImplementedError("TODO")
                    else:
                        raise NotImplementedError("Impossible")
                else:
                    raise NotImplementedError("Impossible")
            state.regs.free_reg(reg, state.block)

        elif isinstance(cmd, Literal):
            value:Token = cmd.literal
            if value.isint():
                literal:int = int(value.token)
                if literal in (0, 1):
                    res_reg:int = literal
                else:
                    res_reg:int = state.regs.get_free_reg()
                    state.block.append(BLiteral(res_reg, BLiteralInt(literal),
                                                BLiteral.INT_T))
            elif value.isstring():
                res_reg:int = state.regs.get_free_reg()
                state.block.append(BLiteral(res_reg, BLiteralStr(value.token),
                                            BLiteral.STR_T))
            elif value in ("true", "false"):
                res_reg:int = value == "true"
            elif value == "none":
                res_reg:int = state.regs.get_free_reg()
                if not CLEAR_REGS:
                    state.block.append(BLiteral(res_reg, BLiteralEmpty(),
                                                BLiteral.NONE_T))
            elif value.isfloat():
                raise NotImplementedError("TODO")
            else:
                raise NotImplementedError("Impossible")

        elif isinstance(cmd, Var):
            name:str = cmd.identifier.token
            assert isinstance(name, str), "TypeError"
            res_reg:int = state.regs.get_free_reg()
            state.block.append(BStoreLoad(name, res_reg, False))

        elif isinstance(cmd, Op):
            if cmd.op == "if":
                # Set up
                if_id:int = self._labels.get_fl()
                label_true:Bable = Bable("if_true_"+str(if_id))
                label_end:Bable = Bable("if_end_"+str(if_id))
                # Condition
                reg:int = self._convert(cmd.args[0], state)
                state.block.append(BJump(label_true.id, reg, False))
                state.regs.free_reg(reg, state.block)
                res_reg:int = state.regs.get_free_reg()
                # If-false
                tmp_reg:int = self._convert(cmd.args[2], state)
                state.block.append(BRegMove(res_reg, tmp_reg))
                state.regs.free_reg(tmp_reg, state.block)
                state.block.append(BJump(label_end.id, 1, False))
                # If-true
                state.block.append(label_true)
                state.regs.clear_reg(reg, state.block)
                tmp_reg:int = self._convert(cmd.args[1], state)
                state.block.append(BRegMove(res_reg, tmp_reg))
                state.regs.free_reg(tmp_reg, state.block)
                state.block.append(label_end)
            else:
                res_reg:int = state.regs.get_free_reg()
                used_regs:list[int] = [res_reg]
                if cmd.op != "call":
                    func:int = state.regs.get_free_reg()
                    state.block.append(BStoreLoad(cmd.op.token, func, False))
                    used_regs.append(func)
                for arg in cmd.args:
                    reg:int = self._convert(arg, state)
                    used_regs.append(reg)
                state.block.append(BCall(used_regs))
                for reg in used_regs[1:]:
                    state.regs.free_reg(reg, state.block)

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
            state.block.append(BJump(label_true.id, reg, False))
            state.regs.free_reg(reg, state.block)
            for subcmd in cmd.false:
                tmp_reg:int = self._convert(subcmd, state)
                state.regs.free_reg(tmp_reg, state.block)
            state.block.append(BJump(label_end.id, 1, False))
            state.block.append(label_true)
            state.regs.clear_reg(reg, state.block)
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(subcmd, state)
                state.regs.free_reg(tmp_reg, state.block)
            state.block.append(label_end)

        elif isinstance(cmd, While):
            # while_start:
            #   reg := <condition>
            #   reg_neg := not(reg)
            #   jmpif (!reg) to while_end
            #   <while-body>
            #   jmp while_start
            # while_end:
            while_id:int = self._labels.get_fl()
            label_start:Bable = Bable("while_"+str(while_id)+"_start")
            label_end:Bable = Bable("while_"+str(while_id)+"_end")
            state.block.append(label_start)
            reg:int = self._convert(cmd.exp, state)
            state.block.append(BJump(label_end.id, reg, True))
            state.regs.free_reg(reg, state.block)
            state.loop_labels.append((label_start.id, label_end.id))
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(subcmd, state)
                state.regs.free_reg(tmp_reg, state.block)
            state.loop_labels.pop()
            state.block.append(BJump(label_start.id, 1, False))
            state.block.append(label_end)
            state.regs.clear_reg(reg, state.block)

        elif isinstance(cmd, Func):
            # <INSTRUCTIONS>:
            #   res_reg := func_label
            # <NEW BODY>:
            # label_start:
            #   <copy args>
            #   <function-body>
            func_id:int = self._labels.get_fl()
            res_reg:int = state.regs.get_free_reg()

            label_start:Bable = Bable(f"func_{func_id}_start")
            label_end:Bable = Bable(f"func_{func_id}_end")
            nstate:State = state.copy()
            nstate.nonlocals = {name:link+1
                                for name,link in state.nonlocals.items()}
            nstate.regs = Regs(state.regs)
            nstate.loop_labels = []
            nstate.block = Block()
            nstate.block.append(label_start)
            nstate.blocks.append(nstate.block)
            for i, arg in enumerate(cmd.args, start=3):
                name:str = arg.identifier.token
                assert isinstance(name, str), "TypeError"
                nstate.block.append(BStoreLoad(name, i, True))
            for subcmd in cmd.body:
                tmp_reg:int = self._convert(subcmd, nstate)
                nstate.regs.free_reg(tmp_reg, nstate.block)
            reg:int = nstate.regs.get_free_reg()
            if not CLEAR_REGS:
                nstate.block.append(BLiteral(reg, BLiteralEmpty(),
                                             BLiteral.NONE_T))
            nstate.block.append(BRegMove(2, reg))
            state.block.append(BLiteral(res_reg, BLiteralStr(label_start.id),
                                        BLiteral.FUNC_T))

        elif isinstance(cmd, NonLocal):
            for identifier in cmd.identifiers:
                if identifier not in state.nonlocals:
                    state.nonlocals[identifier] = 1
                link:int = state.nonlocals[identifier]
                state.block.append(BLoadLink(identifier.token, link))

        elif isinstance(cmd, ReturnYield):
            if cmd.isreturn:
                reg:int = self._convert(cmd.exp, state)
                state.block.append(BRegMove(2, reg))
                state.regs.free_reg(reg, state.block)
            else:
                raise NotImplementedError("TODO")

        elif isinstance(cmd, BreakContinue):
            if cmd.n > len(state.loop_labels):
                action:str = "Break" if cmd.isbreak else "Continue"
                if len(state.loop_labels) == 0:
                    msg:str = "No loops to " + action.lower()
                else:
                    msg:str = action + " number too high"
                raise_error_token(msg, cmd.ft, SemanticError())
            start_label, end_label = state.loop_labels[-cmd.n]
            label:str = end_label if cmd.isbreak else start_label
            state.block.append(BJump(label, 1, False))

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
print(c(f(x+y)), "should be", 7)


f = func(x, y){
    return 0 if y == 0 else x + f(x, y-1)
}
x = 5
y = 10
z = f(x, y)
while (z > 8){
    z -= 1
}
while (z > 0){
    if (z == 8){
        break
    }
}
print(z, "should be", 8)


x = 7
func(){
    nonlocal x
    g = func(){
        nonlocal x
        x += 1
    }
    func(h){
            func(a){a()}(h)
           }(g)
    func(h){h()}(g)
}()
print(x, "should be", 9)


x = [5, 10, 15]
print(x[1], "should be", 10)
x[1] = 15
print(x[1], "should be", 15)
append(x, 20)
print(x[3], "should be", 20)


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
print(fact(5), "should be", 120)


print("The following should be all 1s/`true`s")
x = [1, 2, 3, 4, 5]
a = x[:2]
b = x[2:]
c = x[1:2]
d = x[-2:]
e = x[-2:-1]
f = x[::-1]
print(len(a)==2, a[0]==1, a[1]==2, len(b)==3, b[0]==3, b[1]==4, b[2]==5)
print(len(c)==1, c[0]==2, len(d)==2, d[0]==4, d[1]==5, len(e)==1, e[0]==4)
print(len(f)==5, f[0]==5, f[-1]==1, f[1]==4)
""", False

    TEST2 = """
fib = func(x) ->
    if x < 1 ->
        return 1
    return fib(x-2) + fib(x-1)
print(fib(15), "should be", 1597)
print(fib(30), "should be", 2178309)
""", True

    TEST3 = """
i = 0
while i < 10_000_000 ->
    i += 1
print(i)
""", True

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
print("the number of primes bellow", max, "is:", len(primes))
print("the last prime is:", primes[-1])
""", False

    TEST5 = """
f = func(x){
    if (x){
        return f(x-1) + 1
    }
    return x
}
print(f(100_000), "should be", 100_000)
""", False

    TEST6 = """
""", True

    # TEST1:test_all, TEST2:fib, TEST3:while++, TEST4:primes, TEST5:rec++
    TEST = TEST1
    assert not isinstance(TEST, str), "TEST should be tuple[str,bool]"
    ast:Body = Parser(Tokeniser(StringIO(TEST[0])), colon=TEST[1]).read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        bytecoder:ByteCoder = ByteCoder(ast)
        frame_size, bytecode_blocks = bytecoder.to_bytecode()
        bytecode:list[Bast] = reduce(iconcat, bytecode_blocks, [])
        if bytecode:
            print(bytecode_list_to_str(bytecode))

            raw_bytecode:bytes = bytecoder._serialise(frame_size, bytecode)
            with open("code-examples/example.clizz", "wb") as file:
                file.write(raw_bytecode)

            frame_size, decoded_bast = derialise(raw_bytecode)
            print(f"{frame_size=}")
            assert bytecode_list_to_str(bytecode) == \
                   bytecode_list_to_str(decoded_bast), "AssertionError"

    # fib = lambda x: 1 if x < 1 else fib(x-2)+fib(x-1)
    # print(fib(30))