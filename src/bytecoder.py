from __future__ import annotations
from functools import reduce
from operator import iconcat
from io import StringIO
from sys import stderr

from frontend.bcast import *
from asts.ccast import *
from parser import *
from lexer import *


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

    def free_reg(self, reg:int, instructions:list[Bast],
                 jump_reg:bool=False) -> None:
        assert 0 <= reg < len(self._taken), "ValueError"
        if reg in (0, 1):
            return None
        self._taken[reg] = False
        if not jump_reg:
            instructions.append(BLiteral(reg, BNONE, BLiteral.NONE_T))

    def _notify_max_reg(self, max_reg:int) -> None:
        self.max_reg:int = max(self.max_reg, max_reg)
        if self._parent is not None:
            self._parent._notify_max_reg(self.max_reg)


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
        instructions:Block = Block()
        blocks:list[Block] = [instructions]
        self._labels.reset()
        regs:Regs = Regs()
        for cmd in self.ast:
            reg:int = self._convert(instructions, cmd, regs, blocks, {}, [])
            regs.free_reg(reg, instructions)
        tmp_reg:int = regs.get_free_reg()
        instructions.append(BLiteral(tmp_reg, BLiteralInt(0), BLiteral.INT_T))
        instructions.append(BRegMove(2, tmp_reg))
        regs.free_reg(tmp_reg, instructions, jump_reg=True)
        return regs.max_reg, blocks

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

    def _convert(self, instructions:list[Bast], cmd:Cmd, regs:Regs,
                 blocks:list[Block], nonlocals:dict[str:int],
                 loop_labels:list[tuple[str,str]]) -> int:
        res_reg:int = 0
        if isinstance(cmd, Assign):
            reg:int = self._convert(instructions, cmd.value, regs, blocks,
                                    nonlocals, loop_labels)
            for target in cmd.targets:
                assert isinstance(target, Var), "NotImplementedError"
                name:str = target.identifier.token
                instructions.append(BStoreLoad(name, reg, True))
            regs.free_reg(reg, instructions)
        elif isinstance(cmd, Literal):
            value:Token = cmd.literal
            if value.isint():
                literal:int = int(value.token)
                if literal in (0, 1):
                    res_reg:int = literal
                else:
                    res_reg:int = regs.get_free_reg()
                    instructions.append(BLiteral(res_reg, BLiteralInt(literal),
                                                 BLiteral.INT_T))
            elif value.isstring():
                res_reg:int = regs.get_free_reg()
                instructions.append(BLiteral(res_reg, BLiteralStr(value.token),
                                             BLiteral.STR_T))
            elif value in ("true", "false"):
                res_reg:int = value == "true"
            elif value == "none":
                res_reg:int = regs.get_free_reg()
                instructions.append(BLiteral(res_reg, BLiteralEmpty(),
                                             BLiteral.NONE_T))
            else:
                raise NotImplementedError()
        elif isinstance(cmd, Var):
            name:str = cmd.identifier.token
            assert isinstance(name, str), "TypeError"
            res_reg:int = regs.get_free_reg()
            instructions.append(BStoreLoad(name, res_reg, False))
        elif isinstance(cmd, Op):
            if cmd.op == "if":
                # Set up
                if_id:int = self._labels.get_fl()
                label_true:Bable = Bable("if_true_"+str(if_id))
                label_end:Bable = Bable("if_end_"+str(if_id))
                # Condition
                reg:int = self._convert(instructions, cmd.args[0], regs, blocks,
                                        nonlocals, loop_labels)
                instructions.append(BJump(label_true.id, reg, False))
                regs.free_reg(reg, instructions, jump_reg=True)
                res_reg:int = regs.get_free_reg()
                # If-false
                tmp_reg:int = self._convert(instructions, cmd.args[2], regs,
                                            blocks, nonlocals, loop_labels)
                instructions.append(BRegMove(res_reg, tmp_reg))
                regs.free_reg(tmp_reg, instructions)
                instructions.append(BJump(label_end.id, 1, False))
                # If-true
                instructions.append(label_true)
                tmp_reg:int = self._convert(instructions, cmd.args[1], regs,
                                            blocks, nonlocals, loop_labels)
                instructions.append(BRegMove(res_reg, tmp_reg))
                regs.free_reg(tmp_reg, instructions)
                instructions.append(label_end)
            else:
                res_reg:int = regs.get_free_reg()
                used_regs:list[int] = [res_reg]
                if cmd.op != "call":
                    func:int = regs.get_free_reg()
                    instructions.append(BStoreLoad(cmd.op.token, func, False))
                    used_regs.append(func)
                for arg in cmd.args:
                    reg:int = self._convert(instructions, arg, regs, blocks,
                                            nonlocals, loop_labels)
                    used_regs.append(reg)
                instructions.append(BCall(used_regs))
                for reg in used_regs[1:]:
                    regs.free_reg(reg, instructions)
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
            reg:int = self._convert(instructions, cmd.exp, regs, blocks,
                                    nonlocals, loop_labels)
            instructions.append(BJump(label_true.id, reg, False))
            regs.free_reg(reg, instructions, jump_reg=True)
            for subcmd in cmd.false:
                tmp_reg:int = self._convert(instructions, subcmd, regs, blocks,
                                            nonlocals, loop_labels)
                regs.free_reg(tmp_reg, instructions)
            instructions.append(BJump(label_end.id, 1, False))
            instructions.append(label_true)
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(instructions, subcmd, regs, blocks,
                                            nonlocals, loop_labels)
                regs.free_reg(tmp_reg, instructions)
            instructions.append(label_end)
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
            instructions.append(label_start)
            reg:int = self._convert(instructions, cmd.exp, regs, blocks,
                                    nonlocals, loop_labels)
            instructions.append(BJump(label_end.id, reg, True))
            regs.free_reg(reg, instructions, jump_reg=True)
            loop_labels.append((label_start.id, label_end.id))
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(instructions, subcmd, regs, blocks,
                                            nonlocals, loop_labels)
                regs.free_reg(tmp_reg, instructions)
            loop_labels.pop()
            instructions.append(BJump(label_start.id, 1, False))
            instructions.append(label_end)
        elif isinstance(cmd, Func):
            # <INSTRUCTIONS>:
            #   res_reg := func_label
            # <NEW BODY>:
            # label_start:
            #   <copy args>
            #   <function-body>
            func_id:int = self._labels.get_fl()
            res_reg:int = regs.get_free_reg()

            label_start:Bable = Bable(f"func_{func_id}_start")
            label_end:Bable = Bable(f"func_{func_id}_end")
            block:Block = Block()
            blocks.append(block)
            block.append(label_start)
            frame_regs:Regs = Regs(regs)
            for i, arg in enumerate(cmd.args, start=3):
                name:str = arg.identifier.token
                assert isinstance(name, str), "TypeError"
                block.append(BStoreLoad(name, i, True))
            subnonlocals = {name:link+1 for name,link in nonlocals.items()}
            for subcmd in cmd.body:
                tmp_reg:int = self._convert(block, subcmd, frame_regs, blocks,
                                            subnonlocals, [])
                frame_regs.free_reg(tmp_reg, block)
            reg:int = frame_regs.get_free_reg()
            block.append(BLiteral(reg, BLiteralEmpty(), BLiteral.NONE_T))
            block.append(BRegMove(2, reg))
            instructions.append(BLiteral(res_reg, BLiteralStr(label_start.id),
                                         BLiteral.FUNC_T))
        elif isinstance(cmd, NonLocal):
            for identifier in cmd.identifiers:
                if identifier not in nonlocals:
                    nonlocals[identifier] = 1
                link:int = nonlocals[identifier]
                instructions.append(BLoadLink(identifier.token, link))
        elif isinstance(cmd, ReturnYield):
            assert cmd.isreturn, "Haven't implemented yield yet"
            reg:int = self._convert(instructions, cmd.exp, regs, blocks,
                                    nonlocals, loop_labels)
            instructions.append(BRegMove(2, reg))
            regs.free_reg(reg, instructions)
        elif isinstance(cmd, BreakContinue):
            if cmd.n > len(loop_labels):
                action:str = "Break" if cmd.isbreak else "Continue"
                if len(loop_labels) == 0:
                    msg:str = "No loops to " + action.lower()
                else:
                    msg:str = action + " number too high"
                raise_error_token(msg, cmd.ft, SemanticError())
            start_label, end_label = loop_labels[-cmd.n]
            label:str = end_label if cmd.isbreak else start_label
            instructions.append(BJump(label, 1, False))
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
while z > 8{
    z -= 1
}
print(z, "should be", 8)


x = 0
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
print(x, "should be", 2)


x = [5, 10, 15]
print(x[1], "should be", 10)
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
i = 0
o = 0
while i < 10_000_000 ->
    i += 1
    if i%2 ->
        continue
    o += 1
print(i)
""", True

    TEST5 = """
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

    TEST6 = """
f = func(x){
    if (x){
        return f(x-1)+1
    }
    return x
}
print(f(10_000))
""", False

    TEST = TEST6
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