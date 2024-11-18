from __future__ import annotations
from io import StringIO
from sys import stderr

from frontend.bcast import *
from asts.ccast import *
from parser import *
from lexer import *


def get_fl() -> int:
    global _free_label
    current, _free_label = _free_label, _free_label+1
    return current

def new_reg_frame() -> list[bool]:
    #       0    1   result
    return [True,True,True] + [False]*_max_frame_size
def get_free_reg(regs:list[bool]) -> int:
    global _max_frame_size
    assert isinstance(regs, list), "TypeError"
    for i in range(len(regs)):
        if not regs[i]:
            regs[i] = True
            return i
    regs.append(True)
    _max_frame_size = max(_max_frame_size, len(regs))
    if _max_frame_size > (1<<(FRAME_SIZE*8)):
        raise RuntimeError("What are you doing that needs so many registers")
    return len(regs)-1
def free_reg(regs:list[bool], reg:int) -> None:
    assert isinstance(regs, list), "TypeError"
    assert isinstance(reg, int), "TypeError"
    assert 0 <= reg < len(regs), "ValueError"
    if reg > 1:
        regs[reg] = False

def reset() -> None:
    global _max_frame_size, _free_label
    _max_frame_size = 5 # Doesn't include reg0 and reg1
    _free_label = 0
reset()


class ByteCoder:
    __slots__ = "ast"

    def __init__(self, ast:Body) -> ByteCoder:
        self.ast:Body = ast

    def to_bytecode(self) -> list[Bast]:
        instructions:list[Bast] = []
        registers:list[bool] = new_reg_frame()
        for cmd in self.ast:
            free_reg(registers, self._convert(instructions, cmd, registers))
        return instructions

    def serialise(self) -> bytes:
        return serialise(self.to_bytecode())

    def _convert(self, instructions:list[Bast], cmd:Cmd,
                 registers:list[bool]) -> int:
        res_reg:int = 0
        if isinstance(cmd, Assign):
            reg:int = self._convert(instructions, cmd.value, registers)
            for target in cmd.targets:
                assert isinstance(target, Var), "NotImplementedError"
                name:str = target.identifier.token
                instructions.append(BStoreLoad(name, reg, True))
            free_reg(registers, reg)
        elif isinstance(cmd, Literal):
            value:Token = cmd.literal
            if value.isint():
                literal:int = int(value.token)
                if literal in (0, 1):
                    res_reg:int = literal
                else:
                    res_reg:int = get_free_reg(registers)
                    instructions.append(BLiteral(res_reg, BLiteralInt(literal),
                                                 BLiteral.INT_T))
            elif value.isstring():
                res_reg:int = get_free_reg(registers)
                instructions.append(BLiteral(res_reg, BLiteralStr(value.token),
                                             BLiteral.STR_T))
            else:
                raise NotImplementedError()
        elif isinstance(cmd, Var):
            name:str = cmd.identifier.token
            assert isinstance(name, str), "TypeError"
            res_reg:int = get_free_reg(registers)
            instructions.append(BStoreLoad(name, res_reg, False))
        elif isinstance(cmd, Op):
            if cmd.op == "if":
                # Set up
                if_id:int = get_fl()
                label_true:Bable = Bable("if_true_"+str(if_id))
                label_end:Bable = Bable("if_end_"+str(if_id))
                # Condition
                reg:int = self._convert(instructions, cmd.args[0], registers)
                instructions.append(BJump(label_true, reg, False))
                free_reg(registers, reg)
                res_reg:int = get_free_reg(registers)
                # If-false
                tmp_reg:int = self._convert(instructions, cmd.args[2], registers)
                instructions.append(BRegMove(res_reg, tmp_reg))
                free_reg(registers, tmp_reg)
                instructions.append(BJump(label_end, 1, False))
                # If-true
                instructions.append(label_true)
                tmp_reg:int = self._convert(instructions, cmd.args[1], registers)
                instructions.append(BRegMove(res_reg, tmp_reg))
                free_reg(registers, tmp_reg)
                instructions.append(label_end)
            else:
                res_reg:int = get_free_reg(registers)
                regs:list[int] = [res_reg]
                if cmd.op != "call":
                    func:int = get_free_reg(registers)
                    instructions.append(BStoreLoad(cmd.op.token, func, False))
                    regs.append(func)
                for arg in cmd.args:
                    regs.append(self._convert(instructions, arg, registers))
                instructions.append(BCall(regs))
                for reg in regs[1:]:
                    free_reg(registers, reg)
        elif isinstance(cmd, If):
            #   reg := condition
            #   jmpif reg if_true
            #   <if-false>
            #   jmp if_end
            # if_true:
            #   <if-true>
            # if_end:
            if_id:int = get_fl()
            label_true:Bable = Bable("if_true_"+str(if_id))
            label_end:Bable = Bable("if_end_"+str(if_id))
            reg:int = self._convert(instructions, cmd.exp, registers)
            instructions.append(BJump(label_true, reg, False))
            free_reg(registers, reg)
            for subcmd in cmd.false:
                tmp_reg:int = self._convert(instructions, subcmd, registers)
                free_reg(registers, tmp_reg)
            instructions.append(BJump(label_end, 1, False))
            instructions.append(label_true)
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(instructions, subcmd, registers)
                free_reg(registers, tmp_reg)
            instructions.append(label_end)
        elif isinstance(cmd, While):
            # while_start:
            #   reg := <condition>
            #   reg_neg := not(reg)
            #   jmpif (!reg) to while_end
            #   <while-body>
            #   jmp while_start
            # while_end:
            while_id:int = get_fl()
            label_start:Bable = Bable("while_"+str(while_id)+"_start")
            label_end:Bable = Bable("while_"+str(while_id)+"_end")
            instructions.append(label_start)
            reg:int = self._convert(instructions, cmd.exp, registers)
            instructions.append(BJump(label_end, reg, True))
            free_reg(registers, reg)
            for subcmd in cmd.true:
                tmp_reg:int = self._convert(instructions, subcmd, registers)
                free_reg(registers, tmp_reg)
            instructions.append(BJump(label_start, 1, False))
            instructions.append(label_end)
        elif isinstance(cmd, Func):
            #   jmp label_end
            # label_start:
            #   <copy args>
            #   <function-body>
            # label_end:
            #   res_reg := func_label
            func_id:int = get_fl()
            res_reg:int = get_free_reg(registers)

            label_start:Bable = Bable("func_" + str(func_id)+"_start")
            label_end:Bable = Bable("func_" + str(func_id)+"_end")
            instructions.append(BJump(label_end, 1, False))
            instructions.append(label_start)
            frame_regs:list[int] = new_reg_frame()
            for i, arg in enumerate(cmd.args, start=3):
                name:str = arg.identifier.token
                assert isinstance(name, str), "TypeError"
                instructions.append(BStoreLoad(name, i, True))
            for subcmd in cmd.body:
                self._convert(instructions, subcmd, frame_regs)
            reg:int = get_free_reg(frame_regs)
            instructions.append(BLiteral(reg, BLiteralInt(999), BLiteral.INT_T))
            instructions.append(BRegMove(2, reg))
            instructions.append(label_end)
            instructions.append(BLiteral(res_reg, BLiteralInt(func_id),
                                         BLiteral.FUNC_T))
        elif isinstance(cmd, ReturnYield):
            assert cmd.isreturn, "Haven't implemented yield yet"
            reg:int = self._convert(instructions, cmd.exp, registers)
            instructions.append(BRegMove(2, reg))
            free_reg(registers, reg)
        else:
            raise NotImplementedError(f"Not implemented {cmd!r}")
        return res_reg

    def to_file(self, filepath:str) -> None:
        assert isinstance(filepath, str), "TypeError"
        with open(filepath, "wb") as file:
            file.write(serialise(self.to_bytecode()))


def serialise(bytecode:list[Bast]) -> bytes:
    bast:bytes = b"".join(instruction.serialise() for instruction in bytecode)
    return serialise_int(VERSION, VERSION_SIZE) + \
           serialise_int(_max_frame_size, FRAME_SIZE) + \
           bast


if __name__ == "__main__":
    TEST = """
f = func(x) ->
    g = func() -> return x
    return g
c = func(f) ->
    x = y = g = 0
    return f()

x = 5
y = 2
print(c(f(x+y)), "should be", 7)

f = func(x, y) ->
    return 0 if y == 0 else x + f(x, y-1)

x = 5
y = 10
z = f(x, y)
while z > 8 ->
    z -= 1
print(z, "should be", 8)

Y = func(f) ->
    return func(x) ->
        return f(f(f(f(f(f(f(f)))))))(x)
fact_helper = func(rec) ->
    return func(n) ->
        return 1 if n == 0 else n*rec(n-1)
fact = Y(fact_helper)
print(fact(5), "should be", 120)
"""
    TEST = """
fib = func(x) ->
    if x < 1 ->
        return 1
    return fib(x-2) + fib(x-1)
print(fib(30))
"""
    ast:Body = Parser(Tokeniser(StringIO(TEST)), colon=True).read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        bytecoder:ByteCoder = ByteCoder(ast)
        bytecode:list[Bast] = bytecoder.to_bytecode()
        print(bytecode_list_to_str(bytecode))

        bytecoder.to_file("code-examples/example.clizz")
        raw_bytecode:bytes = serialise(bytecode)
        # print(raw_bytecode)

        frame_size, decoded_bast = derialise(raw_bytecode)
        print(f"{frame_size=}")
        assert bytecode_list_to_str(bytecode) == \
               bytecode_list_to_str(decoded_bast), "AssertionError"

    fib = lambda x: 1 if x < 1 else fib(x-2)+fib(x-1)
    print(fib(30))