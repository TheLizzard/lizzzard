import sys

from python3.rpython_compat import *
# from python3.dict_compat import *
from python3.int_compat import *
from python3.star import *
from bcast import *

from debugger import debug, DEBUG_LEVEL


class Value:
    __slots__ = ()

    def __repr__(self):
        return u"Value[???]"


class IntValue(Value):
    __slots__ = "value"
    _immutable_fields_ = ["value"]

    def __init__(self, value):
        assert isinstance(value, int), "TypeError"
        self.value = const(value)

    def __repr__(self):
        return u"IntValue[" + int_to_str(self.value) + u"]"


class StrValue(Value):
    __slots__ = "value"
    _immutable_fields_ = ["value"]

    def __init__(self, value):
        assert isinstance(value, str), "TypeError"
        self.value = const(const_str(value))

    def __repr__(self):
        return u"StrValue[" + self.value + u"]"


class FuncValue(Value):
    __slots__ = "tp", "master"
    _immutable_fields_ = ["tp", "master"]

    def __init__(self, tp, master):
        # assert isinstance(master, dict), "TypeError"
        assert isinstance(tp, str), "TypeError"
        self.master = const(master)
        self.tp = const(const_str(tp))

    def __repr__(self):
        return u"FuncValue[tp=" + tp + u"]"


class NoneValue(Value):
    __slots__ = ()

    def __repr__(self):
        return u"none"
NONE = NoneValue()


def get_type(value):
    if isinstance(value, IntValue):
        return u"int"
    if isinstance(value, StrValue):
        return u"str"
    if value == NONE:
        return u"none"
    return u"unknown"


def reprr(value):
    return value.__repr__()


BUILTIN_OPS = ["+", "-", "*", "%", "//", "==", "!=", "<", ">", "<=", ">=", "or"]
BUILTIN_SIDES = ["print"]
BUILTIN_OPS = list(map(str, BUILTIN_OPS))
BUILTIN_SIDES = list(map(str, BUILTIN_SIDES))

def new_regs(frame_size):
    return [const(IntValue(0)), const(IntValue(1))] + [NONE]*frame_size

def repr_env(env):
    env = env.copy()
    for op in BUILTIN_OPS+BUILTIN_SIDES:
        env.pop(op, None)
    return repr(env).replace("'", '"').replace(": ", ":")


def bytecode_debug_str(pc, bt):
    data_unicode = int_to_str(pc,zfill=2) + u"| " + bytecode_list_to_str([bt],mini=True)
    data = bytes2(data_unicode)
    data = data.replace(".", ",")
    while data[-1] == "\n":
        data = data[:-1]
    return data


if USE_JIT:
    def get_location(pc, bytecode, _=None):
        return bytecode_debug_str(pc, bytecode[pc])
        return "Instruction[%s]" % bytes2(int_to_str(pc,zfill=2))
    greens = ["pc", "bytecode", "teleports"]
    jitdriver = JitDriver(greens=greens, reds=["stack", "env", "regs"],
                          get_printable_location=get_location)


class Interpreter:
    __slots__ = "bytecode", "frame_size"

    def __init__(self, frame_size, bytecode):
        assert isinstance(frame_size, int), "TypeError"
        assert isinstance(bytecode, list), "TypeError"
        assert frame_size >= 0, "ValueError"
        self.frame_size = frame_size
        self.bytecode = bytecode

    def interpret(self):
        teleports = {}
        for i, bt in enumerate(self.bytecode):
            if isinstance(bt, Bable):
                teleports[const(const_str(bt.id))] = const(IntValue(const(i)))
        for i, op in enumerate(BUILTIN_OPS+BUILTIN_SIDES):
            teleports[const(const_str(op))] = const(IntValue(len(self.bytecode)+i))
        regs = new_regs(self.frame_size)
        Interpreter._interpret(self.bytecode, teleports, regs)

    @staticmethod
    def _interpret(bytecode, teleports, regs):
        bytecode = const([const(bt) for bt in bytecode])
        teleports = const(teleports)
        pc = 0 # current instruction being executed
        stack = [] # list[tuple[Env,Regs,Pc,Reg]]
        env = {}
        for i, op in enumerate(BUILTIN_OPS+BUILTIN_SIDES):
            assert isinstance(op, str), "TypeError"
            env[op] = FuncValue(op, env)

        while pc < len(bytecode):
            if USE_JIT:
                jitdriver.jit_merge_point(stack=stack, env=env, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports)
            bt = const(bytecode[pc])
            if (DEBUG_LEVEL >= 3) or DEBUG_JIT:
                debug_str = bytecode_debug_str(pc, bt)
            if DEBUG_LEVEL >= 3:
                debug(debug_str, 3)

            if isinstance(bt, Bable):
                if USE_JIT and DEBUG_JIT:
                    jit_debug(str(debug_str))
            elif isinstance(bt, BStoreLoad):
                bt_name, bt_storing, bt_reg = const(const_str(bt.name)), const(bt.storing), const(bt.reg)
                assert isinstance(bt_name, str), "TypeError"
                if bt_storing:
                    env[bt_name] = regs[bt_reg]
                else:
                    if bt_name not in env:
                        raise_name_error(bt_name)
                    else:
                        regs[bt_reg] = env[bt_name]
            elif isinstance(bt, BLiteral):
                bt_literal, bt_type, bt_reg = const(bt.literal), const(bt.type), const(bt.reg)
                if bt_type == BLiteral.INT_T:
                    assert isinstance(bt_literal, BLiteralInt), "TypeError"
                    regs[bt_reg] = IntValue(const(bt_literal.int_value))
                elif bt_type == BLiteral.FUNC_T:
                    assert isinstance(bt_literal, BLiteralStr), "TypeError"
                    regs[bt_reg] = FuncValue(const(bt_literal.str_value), env)
                elif bt_type == BLiteral.STR_T:
                    assert isinstance(bt_literal, BLiteralStr), "TypeError"
                    regs[bt_reg] = StrValue(const(const_str(bt_literal.str_value)))
                else:
                    raise NotImplementedError()
            elif isinstance(bt, BJump):
                bt_condition_reg, bt_negated, bt_label_id = const(bt.condition_reg), const(bt.negated), const(bt.label.id)
                value = regs[bt_condition_reg]
                assert isinstance(value, IntValue), "TypeError"
                condition = bool(value.value) ^ bt_negated
                if condition:
                    tp = teleports[bt_label_id]
                    assert isinstance(tp, IntValue), "TypeError"
                    old_pc, pc = pc, tp.value
                    if USE_JIT and DEBUG_JIT:
                        jit_debug(str(debug_str))
                    # if USE_JIT and (pc < old_pc):
                    #     jitdriver.can_enter_jit(stack=stack, regs=regs, pc=pc, env=env, bytecode=bytecode, teleports=teleports)
            elif isinstance(bt, BRegMove):
                bt_reg1, bt_reg2 = const(bt.reg1), const(bt.reg2)
                if bt_reg1 == 2:
                    ret_val = regs[bt_reg2]
                    env, regs, pc, res_reg = stack.pop()
                    regs[res_reg] = ret_val
                else:
                    regs[bt_reg1] = regs[bt_reg2]
            elif isinstance(bt, BCall):
                bt_regs = [const(reg) for reg in bt.regs]
                func = const(regs[bt_regs[1]])
                if not isinstance(func, FuncValue):
                    raise_type_error(get_type(func) + u" is not callable")
                assert isinstance(func, FuncValue), "TypeError"
                func_tp, func_master = const(func.tp), const(func.master)
                if func_tp not in teleports:
                    raise_name_error(func_tp)
                tp = teleports[func_tp] # Get teleport
                assert isinstance(tp, IntValue), "TypeError"
                tp_value = const(tp.value)
                if tp_value < len(bytecode):
                    pc, old_pc = tp_value, pc
                    stack.append((env,regs,old_pc,bt_regs[0]))
                    old_regs, regs = regs, new_regs(len(regs))
                    env = func_master.copy()
                    for i in range(2, len(bt_regs)):
                        regs[i+1] = old_regs[bt_regs[i]]
                    if USE_JIT and (pc < old_pc):
                        jitdriver.can_enter_jit(stack=stack, regs=regs, pc=pc, env=env, bytecode=bytecode, teleports=teleports)
                else: # Built-ins
                    op_idx = tp_value - len(bytecode)
                    pure_op = op_idx < len(BUILTIN_OPS)
                    if pure_op:
                        op = BUILTIN_OPS[op_idx]
                    else:
                        op = BUILTIN_SIDES[op_idx-len(BUILTIN_OPS)]
                    args = [regs[bt_regs[i]] for i in range(2,len(bt_regs))]
                    if pure_op:
                        regs[bt_regs[0]] = builtin_pure(op, args)
                    else:
                        regs[bt_regs[0]] = builtin_side(op, args)
            pc += 1
        if PYTHON == 2:
            return
        debug(int_to_str(pc,zfill=2) + u":\tFinal result: \x1b[0m" + repr_env(env), level=2)

def builtin_pure(op, args):
    bin_int_op = bin_str_op = mul_str_op = False
    if op == u"+":
        if len(args) != 2:
            raise_type_error(op + u" expected 2 arguments")
        assert len(args) == 2, "Unreachable"
        if isinstance(args[0], IntValue):
            bin_int_op = True
        elif isinstance(args[0], StrValue):
            bin_str_op = True
        else:
            raise_type_error(get_type(args[0]) + u" doesn't support " + op)
            assert False, "Unreachable"
    elif op == u"*":
        if len(args) != 2:
            raise_type_error(op + u" expected 2 arguments")
        assert len(args) == 2, "Unreachable"
        arg0, arg1 = args
        if isinstance(arg0, IntValue) and isinstance(arg1, StrValue):
            mul_str_op = True
        elif isinstance(arg0, StrValue) and isinstance(arg1, IntValue):
            mul_str_op = True
        elif isinstance(arg0, IntValue) and isinstance(arg1, IntValue):
            bin_int_op = True
        else:
            raise_type_error(get_type(arg0) + u" doesn't support " + op + u" with " + get_type(arg1))
            assert False, "Unreachable"
    elif op in (u"-", u"//", u"%"):
        if len(args) != 2:
            raise_type_error(op + u" expected 2 arguments")
        assert len(args) == 2, "Unreachable"
        bin_int_op = True
    elif op in (u"==", u"!=", u">=", u"<=", u"<", u">", u"or", u"and"):
        if len(args) != 2:
            raise_type_error(op + u" expected 2 arguments")
        assert len(args) == 2, "Unreachable"
        if isinstance(args[0], IntValue):
            bin_int_op = True
        elif isinstance(args[0], StrValue):
            bin_str_op = True
        else:
            raise_type_error(get_type(args[0]) + u" doesn't support " + op)
            assert False, "Unreachable"
    elif op in (u"not",):
        pass
    else:
        raise NotImplementedError()

    if bin_int_op:
        assert len(args) == 2, "Unreachable"
        arg0, arg1 = args[0], args[1]
        if not isinstance(arg0, IntValue):
            raise_type_error(get_type(arg0) + u" doesn't support " + op)
        if not isinstance(arg1, IntValue):
            raise_type_error(u"can't " + op + u" int with " + get_type(arg1))
        assert isinstance(arg0, IntValue), "Unreachable"
        assert isinstance(arg1, IntValue), "Unreachable"
        if op == u"+":
            res = arg0.value + arg1.value
        elif op == u"-":
            res = arg0.value - arg1.value
        elif op == u"*":
            res = arg0.value * arg1.value
        elif op == u"//":
            res = arg0.value // arg1.value
        elif op == u"%":
            res = arg0.value % arg1.value
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
        if not isinstance(arg1, StrValue):
            raise_type_error(u"can't " + op + u" str with " + get_type(arg1))
        assert isinstance(arg0, StrValue), "Unreachable"
        assert isinstance(arg1, StrValue), "Unreachable"
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
        if len(args) != 1:
            raise_type_error(op + u" expected 1 argument")
        assert len(args) == 1, "Unreachable"
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
    elif mul_str_op:
        assert len(args) == 2, "Unreachable"
        arg0, arg1 = args[0], args[1]
        if isinstance(arg0, StrValue):
            if not isinstance(arg1, IntValue):
                raise_type_error(u"can't " + op + u" str " + u"with " + get_type(arg1))
            assert isinstance(arg1, IntValue), "Unreachable"
            res = arg0.value * arg1.value
        elif isinstance(arg0, IntValue):
            if not isinstance(arg1, StrValue):
                raise_type_error(u"can't " + op + u" int " + u"with " + get_type(arg1))
            assert isinstance(arg1, StrValue), "Unreachable"
            res = arg0.value * arg1.value
        else:
            raise NotImplementedError("Unreachable")
        return StrValue(res)
    raise_unreachable_error(u"Builtin " + op + u" not implemented")
    assert False, "Unreachable"

def builtin_side(op, args):
    if op == u"print":
        string = u""
        for i, arg in enumerate(args):
            if isinstance(arg, IntValue):
                string += int_to_str(arg.value)
            elif isinstance(arg, StrValue):
                string += arg.value[1:]
            elif isinstance(arg, NoneValue):
                string += u"none"
            else:
                raise_type_error(get_type(arg) + u" doesn't support repr")
            if i != len(args)-1:
                string += u" "
        print(u"[STDOUT]: " + string)
        return NONE
    raise_unreachable_error(u"Builtin " + op + u" not implemented")
    assert False, "Unreachable"

def raise_name_error(name):
    print(u"\x1b[91mNameError: " + name + u" was not been defined\x1b[0m")
    raise_error()

def raise_type_error(msg):
    print(u"\x1b[91mTypeError: " + msg + u"\x1b[0m")
    raise_error()

def raise_unreachable_error(msg):
    print(u"\x1b[91mUnreachable reached: " + msg + u"\x1b[0m")
    raise_error()

def raise_error():
    raise Exception("\x1b[91m[ERROR]\x1b[0m")


def _main(raw_bytecode):
    assert isinstance(raw_bytecode, bytes), "TypeError"
    debug(u"Derialising bytecode...", 2)
    data = derialise(raw_bytecode)
    debug(u"Creatng interpreter...", 2)
    interpreter = Interpreter(*data)
    debug(u"Starting interpreter...", 1)
    interpreter.interpret()

def main(filepath):
    debug(u"Reading file...", 2)
    with open(filepath, "rb") as file:
        data = file.read()
    _main(data)


if __name__ == "__main__":
    main("../code-examples/example.clizz")


# Reftypes if needed
# Write fib(x) in C/lizzzzard/python check performance
# https://github.com/aheui/rpaheui/blob/main/LOG.md
# https://pypy.org/posts/2011/04/tutorial-part-2-adding-jit-8121732841568309472.html
# https://pypy.org/posts/2011/03/controlling-tracing-of-interpreter-with_15-3281215865169782921.html
# https://web.archive.org/web/20170929153251/https://bitbucket.org/brownan/pypy-tutorial/src/tip/example4.py

"""
00       jumpif (1!=0) to func_0_end
01 func_0_start:
02       Name[x] := 3
03       4 := Name[<]
04       5 := Name[x]
05       3 := bcall<(5, 1)
06       jumpif (3!=0) to if_true_1
07       jumpif (1!=0) to if_end_1
08 if_true_1:
09       2 := 1
10 if_end_1:
11      4 := Name[+]
12      6 := Name[fib]
13      8 := Name[-]
14      9 := Name[x]
15      10 := Literal[2]
16      7 := bcall-(9, 10)
17      5 := func-from-reg-6(7)
18      7 := Name[fib]
19      9 := Name[-]
20      10 := Name[x]
21      8 := bcall-(10, 1)
22      6 := func-from-reg-7(8)
23      3 := bcall+(5, 6)
24      2 := 3
25      3 := Literal[999]
26      2 := 3
27 func_0_end:
28      3 := Literal[0]
29      Name[fib] := 3
30      4 := Name[print]
31      6 := Name[fib]
32      7 := Literal[30]
33      5 := func-from-reg-6(7)
34      3 := func-from-reg-4(5)
"""

"""
00      3 := Literal[500009]
01      Name[n] := 3
02      Name[is_prime] := 1
03      3 := Literal[2]
04      Name[i] := 3
05 while_0_start:
06      4 := Name[<]
07      5 := Name[i]
08      6 := Name[n]
09      3 := func-from-reg-4(5, 6)
10      jumpif (3==0) to while_0_end
11      4 := Name[==]
12      6 := Name[%]
13      7 := Name[n]
14      8 := Name[i]
15      5 := func-from-reg-6(7, 8)
16      3 := func-from-reg-4(5, 0)
17      jumpif (3!=0) to if_true_1
18      jumpif (1!=0) to if_end_1
19 if_true_1:
20      Name[is_prime] := 0
21 if_end_1:
22      jumpif (1!=0) to while_0_start
23 while_0_end:
24      4 := Name[print]
25      5 := Name[is_prime]
26      3 := func-from-reg-4(5)
"""

"""
00      Name[i]:=Reg[0]
01 while_0_start:
02      Reg[4]:=Name[<]
03      Reg[5]:=Name[i]
04      Reg[6]:=Literal[10000000]
05      Reg[3]:=Reg[4](Reg[5],Reg[6])
06      jumpif(Reg[3]==0)=>while_0_end
07      Reg[4]:=Name[+]
08      Reg[5]:=Name[i]
09      Reg[3]:=Reg[4](Reg[5],Reg[1])
10      Name[i]:=Reg[3]
11      jumpif(Reg[1]!=0)=>while_0_start
12 while_0_end:
13      Reg[4]:=Name[print]
14      Reg[5]:=Name[i]
15      Reg[3]:=Reg[4](Reg[5])
"""