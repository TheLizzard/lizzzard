from __future__ import annotations
from io import StringIO
from sys import stderr
from time import sleep

from bytecoder import ByteCoder
from asts.bcast import *
from parser import *
from lexer import *


class IntValue:
    __slots__ = "value"

    def __init__(self, value:int) -> IntValue:
        assert isinstance(value, int), "TypeError"
        self.value:int = value

    def __repr__(self) -> str:
        return f"IntValue[{self.value}]"


class StrValue:
    __slots__ = "value"

    def __init__(self, value:str) -> StrValue:
        assert isinstance(value, str), "TypeError"
        self.value:str = value

    def __repr__(self) -> str:
        return f"StrValue[{repr(self.value)[1:-1]}]"


class FuncValue:
    __slots__ = "tp_idx", "master"

    def __init__(self, tp_idx:int, master:Env) -> IntValue:
        assert isinstance(master, dict), "TypeError"
        assert isinstance(tp_idx, int), "TypeError"
        self.master:Env = master
        self.tp_idx:int = tp_idx

    def __repr__(self) -> str:
        return f"FuncValue[tp={self.tp_idx}]" # captured={repr_env(self.master)}


class NoneValue:
    __slots__ = ()

    def __repr__(self) -> str:
        return "NoneValue"


Value:type = IntValue|FuncValue|NoneValue

Env:type = dict[str:Value]
Regs:type = list[int]
PC:type = int


BUILT_INS = [
              ["+",  lambda x,y: IntValue(x.value+y.value)],
              ["-",  lambda x,y: IntValue(x.value-y.value)],
              ["*",  lambda x,y: IntValue(x.value*y.value)],
              ["<",  lambda x,y: IntValue(x.value<y.value)],
              [">",  lambda x,y: IntValue(x.value>y.value)],
              ["or", lambda x,y: IntValue(x.value|y.value)],
              ["!=", lambda x,y: IntValue(x.value!=y.value)],
              [">=", lambda x,y: IntValue(x.value>=y.value)],
              ["<=", lambda x,y: IntValue(x.value<=y.value)],
              ["==", lambda x,y: IntValue(x.value==y.value)],
              ["print", lambda *args: [print(*args),999][1]], # TERRIBLE FIXME
            ]


def new_regs() -> Regs:
    return [IntValue(0), IntValue(1)] + [NoneValue()]*frame_size

def new_env(start:int) -> Env:
    env:Env = {}
    for i, (op,_) in enumerate(BUILT_INS):
        env[op] = FuncValue(start+i, {})
    return env

def repr_env(env:Env) -> str:
    env:Env = env.copy()
    for op, _ in BUILT_INS:
        env.pop(op, None)
    return repr(env).replace("'", '"').replace(": ", ":")


class Interpreter:
    __slots__ = "bytecode", "stack", "pc"

    def __init__(self, bytecode:list[Bast]) -> Interpreter:
        assert isinstance(bytecode, list), "TypeError"
        self.bytecode:list[Bast] = bytecode

    def interpret(self) -> None:
        teleports:dict[str:int] = {}
        for i, bt in enumerate(self.bytecode):
            if isinstance(bt, Bable):
                teleports[bt.id] = i
        start_built_ins:int = len(teleports)+100
        pc:PC = -1 # current instruction being executed
        stack:list[tuple[Env,Regs,PC,int]] = []
        env, regs = new_env(start_built_ins), new_regs()
        while True:
            pc += 1
            if pc == len(self.bytecode):
                break
            bt:Bast = self.bytecode[pc]
            if isinstance(bt, Bable):
                pass
            elif isinstance(bt, BStoreLoad):
                if bt.storing:
                    value:Value = regs[bt.reg]
                    assert isinstance(value, Value), "TypeError"
                    env[bt.name] = value
                else:
                    regs[bt.reg] = env[bt.name]
            elif isinstance(bt, BLiteral):
                if bt.type == BLiteral.INT_T:
                    regs[bt.reg] = IntValue(bt.literal)
                elif bt.type == BLiteral.FUNC_T:
                    regs[bt.reg] = FuncValue(bt.literal, env)
                elif bt.type == BLiteral.STR_T:
                    regs[bt.reg] = StrValue(bt.literal)
                else:
                    raise NotImplementedError()
            elif isinstance(bt, BJump):
                value:Value = regs[bt.condition_reg]
                condition:bool = bool(value.value) ^ bt.negated
                if condition:
                    pc:PC = teleports[bt.label.id]
            elif isinstance(bt, BRegMove):
                if bt.reg1 == 2:
                    ret_val:IntValue = regs[bt.reg2]
                    env, regs, pc, res_reg = stack.pop()
                    regs[res_reg] = ret_val
                else:
                    regs[bt.reg1] = regs[bt.reg2]
            elif isinstance(bt, BCall):
                func:FuncValue = regs[bt.regs[1]]
                assert isinstance(func, FuncValue), "InnerTypeError"
                if func.tp_idx < len(teleports):
                    new_pc:int = teleports["func_"+str(func.tp_idx)+"_start"]
                    stack.append((env,regs,pc,bt.regs[0]))
                    env, pc = func.master.copy(), new_pc
                    old_regs, regs = regs, new_regs()
                    for i in range(2, len(bt.regs)):
                        regs[i+1] = old_regs[bt.regs[i]]
                else:
                    # Built-ins
                    op, func = BUILT_INS[func.tp_idx-start_built_ins]
                    args = (regs[bt.regs[i]] for i in range(2,len(bt.regs)))
                    regs[bt.regs[0]] = func(*args)
        print(f"\n\nFinalResult: {repr_env(env)}")


if __name__ == "__main__":
    TEST:str = """

f = func(x):
    g = func(): return x
    return g

x = 5
y = 2
print(f(x+y)())

print("Wow! A string")
print(brf"Wow! Another string")

Y = func(f):
    return func(x):
        return f(f(f(f(f(f(f(f)))))))(x)
fact_helper = func(rec):
    return func(n):
        return 1 if n == 0 else n*rec(n-1)
fact = Y(fact_helper)
print(fact(5))
"""
    ast:Body = Parser(Tokeniser(StringIO(TEST))).read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    else:
        raw_bytecode:bytes = ByteCoder(ast).serialise()
        frame_size, bytecode = ByteCoder.derialise(raw_bytecode)
        # print(bytecode_list_to_str(bytecode))
        Interpreter(bytecode).interpret()

# Strong type checing
# Reftypes if needed
# Lizzzard
# Fix brackets instead of colon