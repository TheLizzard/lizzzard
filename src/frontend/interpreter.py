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
    _immutable_fields_ = ["tp", "master", "env_size", "nargs"]
    __slots__ = "tp", "master", "env_size", "nargs"

    def __init__(self, tp, master, env_size, nargs):
        assert isinstance(env_size, int), "TypeError"
        assert isinstance(master, ENV_TYPE), "TypeError"
        assert isinstance(nargs, int), "TypeError"
        assert isinstance(tp, int), "TypeError"
        self.env_size = env_size
        self.master = master
        self.nargs = nargs
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


class NoneValue(Value):
    _immutable_fields_ = []
    __slots__ = ()

    def __init__(self): pass

    def __repr__(self):
        return u"none"


class ClassValue(Value):
    _immutable_fields_ = ["bases", "master"]
    __slots__ = "bases", "master" # "metaclass" is unused

    def __init__(self, bases, master):
        assert isinstance(bases, ListValue), "TypeError"
        assert isinstance(master, ENV_TYPE), "TypeError"
        self.master = master
        self.bases = bases
        if len(bases.array) > 0:
            raise NotImplementedError("Inheritance not implemented yet")

    def __repr__(self):
        return u"class"


NONE = const(NoneValue())
ZERO = const(IntValue(0))
ONE = const(IntValue(1))
FALSE = const(BoolValue(0))
TRUE = const(BoolValue(1))


@look_inside
@elidable
def get_type(value):
    if value is None:
        return u"undefined" # This should never be used
    if value is NONE:
        return u"none"
    if isinstance(value, IntValue):
        return u"Int"
    if isinstance(value, StrValue):
        return u"Str"
    if isinstance(value, LinkValue):
        return u"Link"
    if isinstance(value, ListValue):
        return u"List"
    if isinstance(value, FuncValue):
        return u"Func"
    if isinstance(value, ClassValue):
        return u"Class"
    return u"unknown"

@look_inside
def force_bool(val):
    if val is None:
        assert False, "InternalError: undefined in regs"
    if val is NONE:
        return False
    if isinstance(val, LinkValue):
        assert False, "InternalError: LinkValue in regs"
    if isinstance(val, IntValue):
        return val.value != 0
    if isinstance(val, StrValue):
        return len(val.value) != 0
    if isinstance(val, ListValue):
        return val.len() != 0
    return True

@look_inside
@elidable
def to_bool_value(boolean):
    assert isinstance(boolean, bool), "TypeError"
    return TRUE if boolean else FALSE

@look_inside
def reg_index(regs, idx):
    if idx == 0:
        return ZERO
    elif idx == 1:
        return ONE
    else:
        return regs[idx]

@look_inside
def reg_store(regs, reg, value):
    assert value is not None, "trying to store undefined inside a register"
    if reg < 1:
        return # Don't store in regs 0 or 1
    regs[reg] = value

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

ENV_IS_LIST = True
if ENV_IS_LIST:
    PREV_ENV_IDX = 0
    BASES_IDX = 1
    ENV_TYPE = list
else:
    PREV_ENV_IDX = u"$prev_env"
    BASES_IDX = u"$cls_bases"
    ENV_TYPE = Dict


def interpret(flags, frame_size, env_size, bytecode):
    global ENV_IS_LIST
    if ENV_IS_LIST != flags.is_set("ENV_IS_LIST"):
        if PYTHON == 3:
            global PREV_ENV_IDX, BASES_IDX, ENV_TYPE
            ENV_IS_LIST = flags.is_set("ENV_IS_LIST")
            PREV_ENV_IDX = 0 if ENV_IS_LIST else "$prev_env"
            BASES_IDX = 0 if ENV_IS_LIST else "$cls_bases"
            ENV_TYPE = list if ENV_IS_LIST else Dict
        else:
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
            env[i] = FuncValue(i+len(bytecode), env, 0, 0)
    else:
        env = Dict()
        for i, op in enumerate(BUILTINS):
            assert isinstance(op, str), "TypeError"
            env[op] = FuncValue(i+len(bytecode), env, 0, 0)
    # Start actual interpreter
    _interpret(bytecode, teleports, regs, env, flags)


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
                    scope_holder = scope[PREV_ENV_IDX]
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
                reg_store(regs, bt.reg, value)

        elif isinstance(bt, BStoreLoadListEnv):
            assert ENV_IS_LIST, "Invalid flag/bytecode"
            # Get the correct scope
            scope = env
            for i in range(bt.link):
                scope_holder = scope[PREV_ENV_IDX]
                assert isinstance(scope_holder, FuncValue), "InternalNonlocalError"
                scope = scope_holder.master
            # Store/Load variable
            if bt.storing:
                scope[bt.name] = reg_index(regs, bt.reg)
            else:
                reg_store(regs, bt.reg, scope[bt.name])

        elif isinstance(bt, BLiteral):
            bt_literal = bt.literal
            if bt.type == BLiteral.INT_T:
                assert isinstance(bt_literal, BLiteralInt), "TypeError"
                literal = IntValue(bt_literal.value)
            elif bt.type == BLiteral.FUNC_T:
                assert isinstance(bt_literal, BLiteralFunc), "TypeError"
                literal = FuncValue(bt_literal.value, env, bt_literal.env_size, bt_literal.nargs)
            elif bt.type == BLiteral.STR_T:
                assert isinstance(bt_literal, BLiteralStr), "TypeError"
                literal = StrValue(bt_literal.value)
            elif bt.type == BLiteral.NONE_T:
                literal = NONE
            elif bt.type == BLiteral.LIST_T:
                literal = ListValue()
            elif bt.type == BLiteral.CLASS_T:
                assert isinstance(bt_literal, BLiteralClass), "TypeError"
                bases = ListValue()
                for base in bt_literal.bases:
                    bases.append(reg_index(regs, base))
                stack.append((env,regs,pc,bt.reg))
                if ENV_IS_LIST:
                    env = [None]*bt_literal.env_size
                else:
                    env = env.copy()
                env[BASES_IDX], env[PREV_ENV_IDX] = bases, FuncValue(-1,env,-1,-1)
                literal = NONE
                tp = teleports.get(bt_literal.label, None)
                assert isinstance(tp, IntValue), "TypeError"
                pc, regs = const(tp.value), list(regs)
            else:
                raise NotImplementedError()
            reg_store(regs, bt.reg, literal)

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

        elif isinstance(bt, BRegMove):
            reg_store(regs, bt.reg1, reg_index(regs, bt.reg2))

        elif isinstance(bt, BRet):
            if bt.capture_env:
                bases = env[BASES_IDX]
                if not isinstance(bases, ListValue):
                    raise_type_error(u"bases is not a list")
                for base in bases.array:
                    if not isinstance(base, ClassValue):
                        raise_type_error(u"bases is not a list of classes")
                value = ClassValue(bases, env)
            else:
                value = reg_index(regs, bt.reg)
                if len(stack) == 0:
                    if not isinstance(value, IntValue):
                        raise_type_error(u"exit value should be an int not " + get_type(value))
                    print(u"[EXIT]: " + int_to_str(value.value))
                    break
            env, regs, pc, ret_reg = stack.pop()
            reg_store(regs, ret_reg, value)

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
                stack.append((env,regs,pc,bt.regs[0]))
                # Set the pc/regs/env
                pc, old_regs, regs = tp_value, regs, list(regs)
                if ENV_IS_LIST:
                    env = [None]*func.env_size
                else:
                    env = func.master.copy()
                env[PREV_ENV_IDX] = func
                # Check number of args
                if len(bt.regs)-2 > func.nargs:
                    raise_type_error(u"too many arguments")
                elif len(bt.regs)-2 < func.nargs:
                    raise_type_error(u"too few arguments")
                # Copy arguments values into new regs
                for i in range(2, len(bt.regs)):
                    reg_store(regs, i+1, reg_index(old_regs, bt.regs[i]))
                # Tell the JIT compiler about the jump
                if USE_JIT:
                    jitdriver.can_enter_jit(stack=stack, env=env, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE)
            else: # Built-ins
                op_idx = tp_value - len(bytecode) - len(BUILTIN_HELPERS)
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
                reg_store(regs, bt.regs[0], value)
        else:
            raise NotImplementedError("Haven't implemented this bytecode yet")


@look_inside
def builtin_pure(op, args, op_print):
    if op in (u"+", u"*", u"==", u"!=", u"len", u"idx", u"simple_idx"):
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

    elif op == u"!=":
        v = builtin_pure_list(u"==", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return to_bool_value(not v.value)

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
        if arg3 is NONE: # step
            step = 1
        elif isinstance(arg3, IntValue):
            step = arg3.value
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg3) + u" for the step arg")
            return NONE
        if step == 0:
            raise_type_error(op_print + u" doesn't support 0 for the step arg")
            return NONE
        if arg1 is NONE: # start
            start = 0 if step > 0 else length
        elif isinstance(arg1, IntValue):
            start = arg1.value
            if start < 0:
                start += length
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the start arg")
            return NONE
        if arg2 is NONE: # stop
            stop = -1 if step < 0 else length
        elif isinstance(arg2, IntValue):
            stop = arg2.value
            if stop < 0:
                stop += length
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg2) + u" for the stop arg")
            return NONE
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
        return NONE

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
        return to_bool_value(not v.value)

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
        return to_bool_value(not v.value)

    elif op == u">=":
        v = builtin_pure(u"<", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return to_bool_value(not v.value)

    elif op == u"or":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        return arg0 if force_bool(arg0) else arg1

    elif op == u"not":
        if len(args) != 1:
            raise_type_error(op_print + u" expects 1 argument")
        return to_bool_value(not force_bool(args[0]))

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
        return u"undefined"
    if obj is NONE:
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
        return NONE
    elif op == u"append":
        if len(args) != 2:
            raise_type_error(op + u" expected 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op + u" expected a list as its 1st argument")
        arg0.append(arg1)
        return NONE
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