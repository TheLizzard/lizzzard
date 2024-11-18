import sys

from python3.rpython_compat import JitDriver, USE_JIT
from python3.dict_compat import *
from python3.int_compat import *
from python3.star import *
from bcast import *


class Value:
    __slots__ = ()

    def __repr__(self):
        return u"Value[???]"


class IntValue(Value):
    __slots__ = "value"

    def __init__(self, value):
        assert isinstance(value, int), "TypeError"
        self.value = value

    def __repr__(self):
        return u"IntValue[" + str(self.value) + u"]"


class StrValue(Value):
    __slots__ = "value"

    def __init__(self, value):
        assert isinstance(value, str), "TypeError"
        self.value = value

    def __repr__(self):
        return u"StrValue[" + str(repr(self.value)[1:-1]) + u"]"


class FuncValue(Value):
    __slots__ = "tp_idx", "master"

    def __init__(self, tp_idx, master):
        assert isinstance(master, dict), "TypeError"
        assert isinstance(tp_idx, int), "TypeError"
        self.master = master
        self.tp_idx = tp_idx

    def __repr__(self):
        return u"FuncValue[tp=" + str(self.tp_idx) + u"]"


class NoneValue(Value):
    __slots__ = ()

    def __repr__(self):
        return u"none"
NONE = NoneValue()


BUILT_INS = ["+", "-", "*", "//", "==", "!=", "<", ">", "<=", ">=", "or",
             "print"]
BUILT_INS = list(map(str, BUILT_INS))

def new_regs(frame_size):
    return [IntValue(0), IntValue(1)] + [NONE]*frame_size

def new_env(start):
    env = dict()
    for i, op in enumerate(BUILT_INS):
        env[op] = FuncValue(start+i, dict())
    return env

def repr_env(env):
    env = env.copy()
    for op in BUILT_INS:
        env.pop(op, None)
    return repr(env).replace("'", '"').replace(": ", ":")


if USE_JIT:
    greens = ["pc", "bytecode"]
    jitdriver = JitDriver(greens=greens, reds=["stack", "env", "regs"])


class Interpreter:
    __slots__ = "bytecode", "frame_size", "stack", "pc"

    def __init__(self, frame_size, bytecode):
        assert isinstance(frame_size, int), "TypeError"
        assert isinstance(bytecode, list), "TypeError"
        assert frame_size >= 0, "ValueError"
        self.frame_size = frame_size
        self.bytecode = bytecode

    def interpret(self):
        self._interpret(self.bytecode)

    def _interpret(self, bytecode):
        teleports = {}
        for i, bt in enumerate(self.bytecode):
            if isinstance(bt, Bable):
                teleports[bt.id] = i
        start_built_ins = len(teleports)+100
        pc = -1 # current instruction being executed
        stack = []
        env, regs = new_env(start_built_ins), new_regs(self.frame_size)

        while True:
            if USE_JIT:
                jitdriver.jit_merge_point(stack=stack, env=env, regs=regs,
                                          pc=pc, bytecode=bytecode)
            pc += 1
            if pc == len(self.bytecode):
                break
            bt = self.bytecode[pc]
            if isinstance(bt, Bable):
                pass
            elif isinstance(bt, BStoreLoad):
                if bt.storing:
                    env[bt.name] = regs[bt.reg]
                else:
                    assert bt.name in env, "InnerNameError"
                    regs[bt.reg] = env[bt.name]
            elif isinstance(bt, BLiteral):
                literal = bt.literal
                if bt.type == BLiteral.INT_T:
                    assert isinstance(literal, BLiteralInt), "TypeError"
                    regs[bt.reg] = IntValue(literal.int_value)
                elif bt.type == BLiteral.FUNC_T:
                    assert isinstance(literal, BLiteralInt), "TypeError"
                    regs[bt.reg] = FuncValue(literal.int_value, env)
                elif bt.type == BLiteral.STR_T:
                    assert isinstance(literal, BLiteralStr), "TypeError"
                    regs[bt.reg] = StrValue(literal.str_value)
                else:
                    raise NotImplementedError()
            elif isinstance(bt, BJump):
                value = regs[bt.condition_reg]
                assert isinstance(value, IntValue), "TypeError"
                condition = bool(value.value) ^ bt.negated
                if condition:
                    pc = teleports[bt.label.id]
            elif isinstance(bt, BRegMove):
                if bt.reg1 == 2:
                    ret_val = regs[bt.reg2]
                    env, regs, pc, res_reg = stack.pop()
                    regs[res_reg] = ret_val
                else:
                    regs[bt.reg1] = regs[bt.reg2]
            elif isinstance(bt, BCall):
                func = regs[bt.regs[1]]
                assert isinstance(func, FuncValue), "InnerTypeError"
                if func.tp_idx < len(teleports):
                    teleport = str("func_") + int_to_str(func.tp_idx) + \
                               str("_start")
                    new_pc = teleports[teleport]
                    can_merge = new_pc < pc
                    stack.append((env,regs,pc,bt.regs[0]))
                    env, pc = func.master.copy(), new_pc
                    old_regs, regs = regs, new_regs(self.frame_size)
                    if can_merge and USE_JIT:
                        jitdriver.can_enter_jit(stack=stack, regs=regs, pc=pc,
                                                env=env, bytecode=bytecode)
                    for i in range(2, len(bt.regs)):
                        regs[i+1] = old_regs[bt.regs[i]]
                else: # Built-ins
                    op = BUILT_INS[func.tp_idx-start_built_ins]
                    args = [regs[bt.regs[i]] for i in range(2,len(bt.regs))]
                    regs[bt.regs[0]] = self.builtin(op, args)
        if PYTHON == 2:
            return
        print("\n\nFinalResult: " + repr_env(env))

    def builtin(self, op, args):
        bin_int_op = bin_str_op = mul_str_op = False
        if op == u"+":
            assert len(args) == 2, "InnerTypeError"
            if isinstance(args[0], IntValue):
                bin_int_op = True
            elif isinstance(args[0], StrValue):
                bin_str_op = True
            else:
                raise RuntimeError("InnerTypeError")
        elif op == u"*":
            assert len(args) == 2, "InnerTypeError"
            arg0, arg1 = args
            if isinstance(arg0, IntValue) and isinstance(arg1, StrValue):
                mul_str_op = True
            elif isinstance(arg0, StrValue) and isinstance(arg1, IntValue):
                mul_str_op = True
            elif isinstance(arg0, IntValue) and isinstance(arg1, IntValue):
                bin_int_op = True
            else:
                raise RuntimeError("InnerTypeError")
        elif op in (u"-", u"//"):
            assert len(args) == 2, "InnerTypeError"
            bin_int_op = True
        elif op in (u"==", u"!=", u">=", u"<=", u"<", u">", u"or", u"and"):
            assert len(args) == 2, "InnerTypeError"
            if isinstance(args[0], IntValue):
                bin_int_op = True
            elif isinstance(args[0], StrValue):
                bin_str_op = True
            else:
                raise RuntimeError("InnerTypeError")
        elif op in (u"print", u"not"):
            pass
        else:
            raise NotImplementedError()

        if bin_int_op:
            assert len(args) == 2, "Unreachable"
            arg0, arg1 = args[0], args[1]
            assert isinstance(arg0, IntValue), "InnerTypeError"
            assert isinstance(arg1, IntValue), "InnerTypeError"
            if op == u"+":
                res = arg0.value + arg1.value
            elif op == u"-":
                res = arg0.value - arg1.value
            elif op == u"*":
                res = arg0.value * arg1.value
            elif op == u"//":
                res = arg0.value // arg1.value
            elif op == u"==":
                res = arg0.value == arg1.value
            elif op == u"!=":
                res = arg0.value != arg1.value
            elif op == u"<":
                res = arg0.value < arg1.value
            elif op == u"<=":
                res = arg0.value <= arg1.value
            elif op == u">":
                res = arg0.value > arg1.value
            elif op == u">=":
                res = arg0.value >= arg1.value
            else:
                raise NotImplementedError()
            return IntValue(res)
        elif bin_str_op:
            assert len(args) == 2, "Unreachable"
            arg0, arg1 = args[0], args[1]
            assert isinstance(arg0, StrValue), "InnerTypeError"
            assert isinstance(arg1, StrValue), "InnerTypeError"
            if op == u"+":
                return StrValue(arg0.value + arg1.value)
            elif op == u"==":
                res = arg0.value == arg1.value
            elif op == u"!=":
                res = arg0.value != arg1.value
            elif op == u"<":
                res = arg0.value < arg1.value
            elif op == u"<=":
                res = arg0.value <= arg1.value
            elif op == u">":
                res = arg0.value > arg1.value
            elif op == u">=":
                res = arg0.value >= arg1.value
            else:
                raise NotImplementedError()
            return IntValue(res)
        elif op == u"not":
            assert len(args) == 1, "InnerTypeError"
            arg = args[0]
            if isinstance(arg, StrValue):
                res = len(arg.value) == 0
            elif isinstance(arg, IntValue):
                res = arg.value == 0
            elif isinstance(arg, NoneValue):
                res = 1
            else:
                res = 0
            return IntValue(res)
        elif op == u"print":
            string = u""
            for i, arg in enumerate(args):
                if isinstance(arg, IntValue):
                    string += int_to_str(arg.value)
                elif isinstance(arg, StrValue):
                    string += arg.value[1:]
                elif isinstance(arg, NoneValue):
                    string += u"none"
                else:
                    raise NotImplementedError("Not implemented in print")
                if i != len(args)-1:
                    string += u" "
            print(u"STDOUT: " + string)
            return NONE
        elif mul_str_op:
            assert len(args) == 2, "Unreachable"
            arg0, arg1 = args[0], args[1]
            if isinstance(arg0, StrValue):
                assert isinstance(arg1, IntValue), "InnerTypeError"
                res = arg0.value * arg1.value
            elif isinstance(arg0, IntValue):
                assert isinstance(arg1, StrValue), "InnerTypeError"
                res = arg0.value * arg1.value
            else:
                raise NotImplementedError("Unreachable")
            return StrValue(res)
        raise NotImplementedError()


def _main(raw_bytecode):
    assert isinstance(raw_bytecode, bytes), "TypeError"
    Interpreter(*derialise(raw_bytecode)).interpret()

def main(filepath):
    with open(filepath, "rb") as file:
        _main(file.read())


if __name__ == "__main__":
    main("../code-examples/example.clizz")


# Reftypes if needed
# Write fib(x) in C/lizzzzard/python check performance