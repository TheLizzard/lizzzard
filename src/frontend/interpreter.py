import sys

from python3.rpython_compat import *
from python3.dict_compat import *
from python3.int_compat import *
from python3.star import *
from bcast import *

from debugger import debug, DEBUG_LEVEL


class Value:
    _immutable_fields_ = []
    __slots__ = ()

    def __repr__(self):
        return u"Value[???]"


class IntValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"

    def __init__(self, value):
        assert isinstance(value, int), "TypeError"
        self.value = value

    def __repr__(self):
        return u"IntValue[" + int_to_str(self.value) + u"]"

BoolValue = IntValue


class StrValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"

    def __init__(self, value):
        assert isinstance(value, str), "TypeError"
        self.value = value

    def __repr__(self):
        return u"StrValue[" + self.value + u"]"


class FuncValue(Value):
    _immutable_fields_ = ["tp", "master", "env_size"]
    __slots__ = "tp", "master", "env_size"

    def __init__(self, tp, master, env_size):
        assert isinstance(env_size, int), "TypeError"
        if ENV_IS_LIST:
            assert isinstance(master, list), "TypeError"
        else:
            assert isinstance(master, Dict), "TypeError"
        assert isinstance(tp, int), "TypeError"
        self.env_size = env_size
        self.master = master
        self.tp = tp

    def __repr__(self):
        return u"FuncValue[tp=" + int_to_str(self.tp) + u"]"


class LinkValue(Value):
    _immutable_fields_ = ["link"]
    __slots__ = "link"

    def __init__(self, link):
        assert isinstance(link, int), "TypeError"
        assert link > 0, "ValueError"
        self.link = link

    def __repr__(self):
        return u"link[" + int_to_str(self.link) + "]"


class ListValue(Value):
    _immutable_fields_ = ["array"]
    __slots__ = "array"

    def __init__(self):
        self.array = []

    @look_inside
    def append(self, value):
        assert isinstance(value, Value), "TypeError"
        self.array.append(value)

    @look_inside
    @elidable
    def index(self, idx):
        assert isinstance(idx, int), "TypeError"
        return self.array[idx]

    @look_inside
    def index_set(self, idx, value):
        assert isinstance(idx, int), "TypeError"
        self.array[idx] = value

    @look_inside
    @elidable
    def len(self):
        return len(self.array)

    def __repr__(self):
        return u"array[size=" + int_to_str(self.len()) + u"]"


ZERO = const(IntValue(0))
ONE = const(IntValue(1))
FALSE = const(BoolValue(0))
TRUE = const(BoolValue(1))


@look_inside
@elidable
def get_type(value):
    if value is None:
        return u"none"
    if isinstance(value, IntValue):
        return u"int"
    if isinstance(value, StrValue):
        return u"str"
    if isinstance(value, LinkValue):
        return u"link"
    if isinstance(value, ListValue):
        return u"list"
    if isinstance(value, FuncValue):
        return u"func"
    return u"unknown"

@look_inside
@elidable
def force_bool(val):
    if val is None:
        return False
    if isinstance(val, IntValue):
        return bool(val.value)
    if isinstance(val, StrValue):
        return bool(val.value)
    if isinstance(val, ListValue):
        return bool(val.len())
    if isinstance(val, LinkValue):
        assert False, "InternalError: LinkValue in Regs"
    return True


@look_inside
@elidable
def reg_index(regs, idx):
    if idx == 0:
        return ZERO
    elif idx == 1:
        return ONE
    else:
        return regs[idx]


def bytecode_debug_str(pc, bt):
    data_unicode = int_to_str(pc,zfill=2) + u"| " + bytecode_list_to_str([bt],mini=True)
    data = bytes2(data_unicode)
    data = data.replace(".", ",")
    while data[-1] == "\n":
        data = data[:-1]
    return data


if USE_JIT:
    def get_location(pc, bytecode, *_):
        return bytecode_debug_str(pc, bytecode[pc])
        return "Instruction[%s]" % bytes2(int_to_str(pc,zfill=2))
    jitdriver = JitDriver(greens=["pc","bytecode","teleports"], reds=["CLEAR_AFTER_USE","stack","env","regs"], get_printable_location=get_location)


def interpret(flags, frame_size, env_size, bytecode):
    if ENV_IS_LIST != flags.is_set("ENV_IS_LIST"):
        if ENV_IS_LIST:
            print("\x1b[91m[ERROR]: The interpreter was compiled with ENV_IS_LIST but the clizz file wasn't\x1b[0m")
        else:
            print("\x1b[91m[ERROR]: The interpreter was compiled with ENV_IS_LIST==false but the clizz file uses ENV_IS_LIST\x1b[0m")
        raise SystemExit()
    # Create teleports
    teleports = Dict()
    for i, bt in enumerate(bytecode):
        if isinstance(bt, Bable):
            teleports[const(const_str(bt.id))] = const(IntValue(const(i)))
    for i, op in enumerate(BUILTINS):
        tp = len(bytecode) + i
        teleports[const(const_str(int_to_str(tp)))] = const(IntValue(tp))
    # Create regs
    regs = [None]*(frame_size+2)
    # Create env
    if ENV_IS_LIST:
        env = [None]*env_size
        for i, op in enumerate(BUILTINS):
            assert isinstance(op, str), "TypeError"
            env[i] = FuncValue(i+len(bytecode), env, 0)
    else:
        env = Dict()
        for i, op in enumerate(BUILTINS):
            assert isinstance(op, str), "TypeError"
            env[op] = FuncValue(i+len(bytecode), env, 0)
    # Start actual interpreter
    _interpret(bytecode, teleports, regs, env, flags)


ENV_IS_LIST = True
if ENV_IS_LIST:
    PREV_ENV = 0
else:
    PREV_ENV = u"$prev_env"

def _interpret(bytecode, teleports, regs, env, flags):
    CLEAR_AFTER_USE = const(flags.is_set("CLEAR_AFTER_USE"))

    bytecode = [const(bt) for bt in bytecode]
    pc = 0 # current instruction being executed
    stack = [] # list[tuple[Env,Regs,Pc,RetReg]]

    while pc < len(bytecode):
        if USE_JIT:
            jitdriver.jit_merge_point(stack=stack, env=env, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE)
        bt = bytecode[pc]
        if DEBUG_LEVEL >= 3:
            debug(str(bytecode_debug_str(pc, bt)), 3)
        pc += 1

        if isinstance(bt, Bable):
            pass

        elif isinstance(bt, BLoadLink):
            assert not ENV_IS_LIST, "Invalid flag/bytecode"
            env[bt.name] = LinkValue(bt.link)

        elif isinstance(bt, BStoreLoadDictEnv):
            assert not ENV_IS_LIST, "Invalid flag/bytecode"
            # Get the correct scope
            scope, value = env, env.get(bt.name, None)
            if isinstance(value, LinkValue):
                for i in range(value.link):
                    scope_holder = scope.get(PREV_ENV, None)
                    assert isinstance(scope_holder, FuncValue), "InternalNonlocalError"
                    scope = scope_holder.master
                if not bt.storing:
                    value = scope.get(bt.name, None)
            # Store/Load variable
            if bt.storing:
                scope[bt.name] = reg_index(regs, bt.reg)
            else:
                if value is None:
                    raise_name_error(bt.name)
                regs[bt.reg] = value

        elif isinstance(bt, BStoreLoadListEnv):
            assert ENV_IS_LIST, "Invalid flag/bytecode"
            # Get the correct scope
            scope = env
            for i in range(bt.link):
                scope_holder = scope[PREV_ENV]
                assert isinstance(scope_holder, FuncValue), "InternalNonlocalError"
                scope = scope_holder.master
            # Store/Load variable
            if bt.storing:
                scope[bt.name] = reg_index(regs, bt.reg)
            else:
                regs[bt.reg] = scope[bt.name]

        elif isinstance(bt, BLiteral):
            bt_literal = bt.literal
            if bt.type == BLiteral.INT_T:
                assert isinstance(bt_literal, BLiteralInt), "TypeError"
                literal = IntValue(bt_literal.value)
            elif bt.type == BLiteral.FUNC_T:
                assert isinstance(bt_literal, BLiteralFunc), "TypeError"
                literal = FuncValue(bt_literal.value, env, bt_literal.env_size)
            elif bt.type == BLiteral.STR_T:
                assert isinstance(bt_literal, BLiteralStr), "TypeError"
                literal = StrValue(bt_literal.value)
            elif bt.type == BLiteral.NONE_T:
                literal = None
            elif bt.type == BLiteral.LIST_T:
                literal = ListValue()
            else:
                raise NotImplementedError()
            regs[bt.reg] = literal
            del literal, bt_literal

        elif isinstance(bt, BJump):
            value = reg_index(regs, bt.condition_reg)
            if CLEAR_AFTER_USE and (bt.condition_reg > 1):
                regs[bt.condition_reg] = None
            condition = force_bool(value)
            # if condition != bt.negated: # RPython's JIT can't constant fold bt.negated in this form :/
            if (condition and (not bt.negated)) or ((not condition) and bt.negated):
                tp = teleports.get(bt.label, None)
                assert isinstance(tp, IntValue), "TypeError"
                pc = tp.value
                assert isinstance(bytecode[pc], Bable), "InternalError"
                if USE_JIT:
                    jitdriver.can_enter_jit(stack=stack, env=env, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE)
                del tp
            del value

        elif isinstance(bt, BRegMove):
            if bt.reg1 == 2:
                ret_val = reg_index(regs, bt.reg2)
                if len(stack) == 0:
                    if not isinstance(ret_val, IntValue):
                        raise_type_error(u"exit value should be an int not " + get_type(ret_val))
                    print(u"[EXIT]: " + int_to_str(ret_val.value))
                    break
                env, regs, pc, res_reg = stack.pop()
                regs[res_reg] = ret_val
                del ret_val
            else:
                regs[bt.reg1] = reg_index(regs, bt.reg2)

        elif isinstance(bt, BCall):
            func = const(reg_index(regs, bt.regs[1]))
            if not isinstance(func, FuncValue):
                raise_type_error(get_type(func) + u" is not callable")
            assert isinstance(func, FuncValue), "TypeError"
            tp = teleports.get(int_to_str(func.tp), None)
            if tp is None:
                raise_name_error(int_to_str(func.tp))
            assert isinstance(tp, IntValue), "TypeError"
            tp_value = const(tp.value)
            if tp_value < len(bytecode):
                pc, old_pc = tp_value, pc
                stack.append((env,regs,old_pc,bt.regs[0]))
                old_regs, regs = regs, list(regs)
                assert len(old_regs) == len(regs), "InternalError"
                if ENV_IS_LIST:
                    env = [None]*func.env_size
                else:
                    env = func.master.copy()
                env[PREV_ENV] = func
                for i in range(2, len(bt.regs)):
                    regs[i+1] = old_regs[bt.regs[i]]
                assert isinstance(bytecode[pc], Bable), "InternalError"
                if USE_JIT:
                    jitdriver.can_enter_jit(stack=stack, env=env, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE)
                del old_regs
            else: # Built-ins
                op_idx = tp_value - len(bytecode) - 1
                pure_op = op_idx < len(BUILTIN_OPS)
                if pure_op:
                    op = BUILTIN_OPS[op_idx]
                else:
                    op = BUILTIN_SIDES[op_idx-len(BUILTIN_OPS)]
                args = [reg_index(regs, bt.regs[i]) for i in range(2,len(bt.regs))]
                if pure_op:
                    value = builtin_pure(op, args, op)
                else:
                    value = builtin_side(op, args)
                if bt.regs[0] > 1: # Don't store values inside ZERO,ONE
                    regs[bt.regs[0]] = value
                del args, pure_op, op_idx, op, value
            del tp, tp_value, func
        del bt


@look_inside
def builtin_pure(op, args, op_print):
    if op in (u"+", u"*", u"==", u"len", u"idx", u"simple_idx"):
        if (len(args) > 0) and isinstance(args[0], ListValue):
            return builtin_pure_list(op, args, op)
        elif (len(args) == 2) and isinstance(args[1], ListValue):
            return builtin_pure_list(op, args, op)
    elif op in (u"[]", u"simple_idx="):
        return builtin_pure_list(op, args, op)
    return builtin_pure_nonlist(op, args, op)


@look_inside
def builtin_pure_list(op, args, op_print):
    assert len(args) > 0, "InternalError"
    if op == u"+":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(get_type(arg0) + u" doesn't support " + op_print + u" with " + get_type(arg1))
        if not isinstance(arg1, ListValue):
            raise_type_error(u"can't " + op_print + u" list with " + get_type(arg1))
        new = ListValue()
        new.array = arg0.array + arg1.array
        return new

    elif op == u"*":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, ListValue):
            if not isinstance(arg1, IntValue):
                raise_type_error(u"can't " + op_print + u" list with " + get_type(arg1))
            new = ListValue()
            for _ in range(arg1.value):
                new.array += arg0.array
            return new
        elif isinstance(arg0, IntValue):
            assert isinstance(arg1, ListValue), "InternalError"
            new = ListValue()
            for _ in range(arg0.value):
                new.array += arg1.array
            return new
        else:
            raise_type_error(u"can't " + op_print + u" " + get_type(arg0) + u" with " + get_type(arg1))

    elif op == u"==":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, ListValue):
            if not isinstance(arg1, ListValue):
                raise_type_error(u"can't " + op_print + u" list with " + get_type(arg1))
            if arg0.len() != arg1.len():
                return FALSE
            for i in range(arg0.len()):
                v = builtin_pure(u"==", [arg0.index(i),arg1.index(i)], op_print)
                assert isinstance(v, BoolValue), "InternalError"
                if not v.value:
                    return FALSE
            return TRUE
        return FALSE

    elif op == u"len":
        if len(args) != 1:
            raise_type_error(op_print + u" expects 1 argument")
        arg = args[0]
        if not isinstance(arg, ListValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg))
        return IntValue(arg.len())

    elif op == u"idx":
        if len(args) != 4:
            raise_type_error(op_print + u" expects 4 arguments (list, start, stop, step)")
        arg0, arg1, arg2, arg3 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg0) + u" for the 1st arg")
        length = arg0.len()
        if arg3 is None: # step
            step = 1
        elif isinstance(arg3, IntValue):
            step = arg3.value
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg3) + u" for the step arg")
            return None
        if step == 0:
            raise_type_error(op_print + u" doesn't support 0 for the step arg")
            return None
        if arg1 is None: # start
            start = 0 if step > 0 else length
        elif isinstance(arg1, IntValue):
            start = arg1.value
            if start < 0:
                start += length
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the start arg")
            return None
        if arg2 is None: # stop
            stop = -1 if step < 0 else length
        elif isinstance(arg2, IntValue):
            stop = arg2.value
            if stop < 0:
                stop += length
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg2) + u" for the stop arg")
            return None
        new = ListValue()
        for idx in range(start, stop, step):
            if 0 <= idx < length:
                new.append(arg0.index(idx))
        return new

    elif op == u"simple_idx":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments (list, idx)")
        arg0, arg1 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg0) + u" for the 1st arg")
        if not isinstance(arg1, IntValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the index")
        length = arg0.len()
        idx = arg1.value
        if idx < 0:
            idx += length
        if not (0 <= idx < length):
            raise_index_error(u"when indexing idx=" + int_to_str(idx))
        return arg0.index(idx)

    elif op == u"simple_idx=":
        if len(args) != 3:
            raise_type_error(op_print + u" expects 3 arguments (list, idx, value)")
        arg0, arg1, arg2 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg0) + u" for the 1st arg")
        if not isinstance(arg1, IntValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the index")
        length = arg0.len()
        idx = arg1.value
        if idx < 0:
            idx += length
        if not (0 <= idx < length):
            raise_index_error(u"when setting idx=" + int_to_str(idx))
        arg0.index_set(idx, arg2)
        return None

    elif op == u"[]":
        new = ListValue()
        new.array = args
        return new

    else:
        raise_unreachable_error(u"builtin_pure_list " + op + u" not implemented (needed for " + op_print + u")")


@look_inside
def builtin_pure_nonlist(op, args, op_print):
    if op == u"+":
        if len(args) == 1:
            arg = args[0]
            if not isinstance(arg, IntValue):
                raise_type_error(op_print + u" expects an int if used with 1 argument")
            return arg
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, IntValue):
                if not isinstance(arg1, IntValue):
                    raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
                return IntValue(arg0.value + arg1.value)
            elif isinstance(arg0, StrValue):
                if not isinstance(arg1, StrValue):
                    raise_type_error(u"can't " + op_print + u" str with " + get_type(arg1))
                return StrValue(arg0.value + arg1.value)
            else:
                raise_type_error(get_type(arg0) + u" doesn't support " + op_print)
        else:
            raise_type_error(op_print + u" expects 1 or 2 arguments")

    elif op == u"-":
        if len(args) == 1:
            arg = args[0]
            if not isinstance(arg, IntValue):
                raise_type_error(op_print + u" expects an int if used with 1 argument")
            return IntValue(-arg.value)
        elif len(args) == 2:
            arg0, arg1 = args
            if not isinstance(arg0, IntValue):
                raise_type_error(get_type(arg0) + u" doesn't support " + op_print)
            if not isinstance(arg1, IntValue):
                raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
            return IntValue(arg0.value - arg1.value)

    elif op == u"*":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value * arg1.value)
            elif isinstance(arg1, StrValue):
                return StrValue(arg0.value * arg1.value)
            else:
                raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, IntValue):
                raise_type_error(u"can't " + op_print + u" str with " + get_type(arg1))
            return StrValue(arg0.value * arg1.value)
        else:
            raise_type_error(get_type(arg0) + u" doesn't support " + op_print)

    elif op == u"%":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, IntValue):
            raise_type_error(get_type(arg0) + u" doesn't support " + op_print)
        if not isinstance(arg1, IntValue):
            raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
        return IntValue(arg0.value % arg1.value)

    elif op == u"//":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, IntValue):
            raise_type_error(get_type(arg0) + u" doesn't support " + op_print)
        if not isinstance(arg1, IntValue):
            raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
        return IntValue(arg0.value // arg1.value)

    elif op == u"==":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if not isinstance(arg1, IntValue):
                return FALSE
            if arg0.value == arg1.value:
                return TRUE
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, StrValue):
                return FALSE
            if arg0.value == arg1.value:
                return TRUE
        return FALSE

    elif op == u"!=":
        v = builtin_pure(u"==", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return FALSE if v.value else TRUE

    elif op == u"<":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if not isinstance(arg1, IntValue):
                raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
            return BoolValue(arg0.value < arg1.value)
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, StrValue):
                raise_type_error(u"can't " + op_print + u" str with " + get_type(arg1))
            return BoolValue(arg0.value < arg1.value)
        else:
            raise_type_error(get_type(arg0) + u" doesn't support " + op_print)

    elif op == u">":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if not isinstance(arg1, IntValue):
                raise_type_error(u"can't " + op_print + u" int with " + get_type(arg1))
            return BoolValue(arg0.value > arg1.value)
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, StrValue):
                raise_type_error(u"can't " + op_print + u" str with " + get_type(arg1))
            return BoolValue(arg0.value > arg1.value)
        else:
            raise_type_error(get_type(arg0) + u" doesn't support " + op_print)

    elif op == u"<=":
        v = builtin_pure(u">", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return FALSE if v.value else TRUE

    elif op == u">=":
        v = builtin_pure(u"<", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return FALSE if v.value else TRUE

    elif op == u"or":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        return arg0 if force_bool(arg0) else arg1

    elif op == u"len":
        if len(args) != 1:
            raise_type_error(op_print + u" expects 1 argument")
        arg0 = args[0]
        if not isinstance(arg0, StrValue):
            raise_type_error(op_print + u" not supported on " + get_type(args[0]))
        return IntValue(len(arg0.value))

    elif op == u"simple_idx":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, StrValue):
            raise_type_error(op_print + u" not supported on " + get_type(args[0]))
        if not isinstance(arg1, IntValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the index")
        return StrValue(arg0.value[arg1.value])

    elif op == u"idx":
        if len(args) != 4:
            raise_type_error(op_print + u" expects 4 arguments")
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    else:
        raise_unreachable_error(u"builtin_pure " + op + u" not implemented (needed for " + op_print + u")")


@look_inside
@elidable
def inner_repr(obj):
    if obj is None:
        return u"none"
    elif isinstance(obj, IntValue):
        return int_to_str(obj.value)
    elif isinstance(obj, StrValue):
        return obj.value[1:]
    elif isinstance(obj, FuncValue):
        return u"Func"
    elif isinstance(obj, ListValue):
        string = u"["
        for i, element in enumerate(obj.array):
            string += inner_repr(element)
            if i != obj.len()-1:
                string += u", "
        return string + u"]"
    else:
        return get_type(obj) + u" doesn't support repr"

@look_inside
def builtin_side(op, args):
    if op == u"print":
        string = u""
        for i, arg in enumerate(args):
            string += inner_repr(arg)
            if i != len(args)-1:
                string += u" "
        print(u"[STDOUT]: " + string)
        return None
    elif op == u"append":
        if len(args) != 2:
            raise_type_error(op + u" expected 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op + u" expected a list as its 1st argument")
        arg0.append(arg1)
        return None
    raise_unreachable_error(u"Builtin " + op + u" not implemented")
    assert False, "Unreachable"


def raise_name_error(name):
    print(u"\x1b[91mNameError: " + name + u" was not been defined\x1b[0m")
    raise_error()

def raise_index_error(msg):
    print(u"\x1b[91mIndexError: " + msg + u"\x1b[0m")
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
    flags, frame_size, env_size, bytecode = derialise(raw_bytecode)
    debug(u"Parsing flags...", 2)
    debug(u"Starting interpreter...", 1)
    interpret(flags, frame_size, env_size, bytecode)

def main(filepath):
    debug(u"Reading file...", 2)
    with open(filepath, "rb") as file:
        data = file.read()
    _main(data)


if __name__ == "__main__":
    main("../code-examples/example.clizz")


# https://github.com/aheui/rpaheui/blob/main/LOG.md
# https://pypy.org/posts/2011/04/tutorial-part-2-adding-jit-8121732841568309472.html
# https://pypy.org/posts/2011/03/controlling-tracing-of-interpreter-with_15-3281215865169782921.html
# https://web.archive.org/web/20170929153251/https://bitbucket.org/brownan/pypy-tutorial/src/tip/example4.py
# https://doi.org/10.1016/j.entcs.2016.12.012
# /media/thelizzard/TheLizzardOS-SD/rootfs/home/thelizzard/honours/lizzzard/src/frontend/rpython/rlib/jit.py
# https://eprints.gla.ac.uk/113615/