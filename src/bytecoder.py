from __future__ import annotations
from io import StringIO
from sys import stderr

from asts.ccast import *
from asts.bcast import *
from parser import *
from lexer import *

from python3.bytes_compat import bytes_join_empty

FRAME_SIZE:int = 1 # number of bytes that hold the max frame size
VERSION_SIZE:int = 4 # number of bytes to store the version
VERSION:int = 1

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

    def serialise_bytecode(self, bytecode:list[Bast]) -> bytes:
        bast:bytes = bytes_join_empty(instruction.serialise() \
                                      for instruction in bytecode)
        return serialise_int(VERSION, VERSION_SIZE) + \
               serialise_int(_max_frame_size, FRAME_SIZE) + \
               bast

    def serialise(self) -> bytes:
        return self.serialise_bytecode(self.to_bytecode())

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
                    instructions.append(BLiteral(res_reg, literal,
                                                 BLiteral.INT_T))
            elif value.isstring():
                res_reg:int = get_free_reg(registers)
                instructions.append(BLiteral(res_reg, value.token,
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
            instructions.append(BLiteral(reg, 999, BLiteral.INT_T))
            instructions.append(BRegMove(2, reg))
            instructions.append(label_end)
            instructions.append(BLiteral(res_reg, func_id, BLiteral.FUNC_T))
        elif isinstance(cmd, ReturnYield):
            assert cmd.isreturn, "Haven't implemented yield yet"
            reg:int = self._convert(instructions, cmd.exp, registers)
            instructions.append(BRegMove(2, reg))
            free_reg(registers, reg)
        else:
            raise NotImplementedError(f"Not implemented {cmd!r}")
        return res_reg

    @staticmethod
    def derialise(data:bytes) -> tuple[int,list[Bast]]:
        version, data = derialise_int(data, VERSION_SIZE)
        frame_size, data = derialise_int(data, FRAME_SIZE)
        output:list[Bast] = []
        while data:
            ast_t_id, _ = derialise_ast_t_id(data)
            bast, data = TABLE[ast_t_id].derialise(data)
            output.append(bast)
        return frame_size, output


if __name__ == "__main__":
    TEST = """
f = func(x, y){
    return 0 if y == 0 else x + f(x, y-1)
}

x = 5
y = 10
z = f(x, y)

while z > 8{
    z--
}
"""
    ast:Body = Parser(Tokeniser(StringIO(TEST)), colon=False).read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        bytecoder:ByteCoder = ByteCoder(ast)
        bytecode:list[Bast] = bytecoder.to_bytecode()
        print(bytecode_list_to_str(bytecode))

        raw_bytecode:bytes = bytecoder.serialise_bytecode(bytecode)
        print(raw_bytecode)

        frame_size, decoded_bast = ByteCoder.derialise(raw_bytecode)
        print(f"{frame_size=}")
        assert bytecode_list_to_str(bytecode) == \
               bytecode_list_to_str(decoded_bast), "AssertionError"


"""
        jumpif (1!=0) to func_0_end       goto func_0_end
func_0_start:                           func_0_start:
        Name[x] := 3
        Name[y] := 4                      copy args [regs => env]
        4 := Name[if]
        6 := Name[==]
        7 := Name[y]
        5 := func-from-reg-6(7, 0)        register5  := (y == 0)
        7 := Name[+]
        8 := Name[x]
        10 := Name[f]
        11 := Name[x]
        13 := Name[-]
        14 := Name[y]
        12 := func-from-reg-13(14, 1)     register12 := (y - 1)
        9 := func-from-reg-10(11, 12)     register9  := f(x, register12)
        6 := func-from-reg-7(8, 9)        register6  := (x + register9)
        3 := func-from-reg-4(5, 0, 6)     return 0 if register5 else register6
        2 := 3
        3 := Literal[999]                 return 999
        2 := 3

func_0_end:                             func_0_end
        3 := Literal[0]
        Name[f] := 3                      f := function0
        3 := Literal[5]
        Name[x] := 3                      x := 5
        3 := Literal[10]
        Name[y] := 3                      y := 10
        4 := Name[f]
        5 := Name[x]
        6 := Name[y]
        3 := func-from-reg-4(5, 6)
        Name[z] := 3                      z := f(x, y)

while_1_start:                          while_1_start:
        4 := Name[>]
        5 := Name[z]
        6 := Literal[8]
        3 := func-from-reg-4(5, 6)        register3 := (z > 8)
        jumpif (3==0) to while_1_end      jmpif (!register3) to while_1_end
        4 := Name[--]
        5 := Name[z]
        3 := func-from-reg-4(5)           register3 := (z - 1)
        jumpif (1!=0) to while_1_start    goto while_1_start
while_1_end:                            while_1_end:
""" # bug in --, reg not being stored after calc