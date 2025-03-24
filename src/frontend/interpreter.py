# coding=utf-8
import signal
import math
import os

from python3.rpython_compat import *
from python3.dict_compat import *
from python3.int_compat import *
from python3.str_compat import *
from python3.star import *
from bcast import *

from debugger import debug, DEBUG_LEVEL


# Value classes
class Value:
    _immutable_fields_ = []
    __slots__ = ()

class IntValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    @look_inside
    def __init__(self, value):
        assert isinstance(value, int), "TypeError"
        self.value = value

class BoolValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    @look_inside
    def __init__(self, value):
        assert isinstance(value, bool), "TypeError"
        self.value = value

class StrValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    @look_inside
    def __init__(self, value):
        assert isinstance(value, str), "TypeError"
        self.value = value

class FloatValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    @look_inside
    def __init__(self, value):
        assert isinstance(value, float), "TypeError"
        self.value = value

class FuncValue(Value):
    _immutable_fields_ = ["func_info", "masters", "defaults[*]"]
    __slots__ = "func_info", "masters", "defaults"
    @unroll_safe
    @look_inside
    def __init__(self, func_info, masters, defaults):
        assert isinstance(defaults, list), "TypeError"
        assert isinstance(masters, list), "TypeError"
        for default in defaults:
            assert isinstance(default, Value), "TypeError"
        for master in masters:
            assert isinstance(master, ENV_TYPE), "TypeError"
        assert isinstance(func_info, FuncInfo), "TypeError"
        self.func_info = func_info
        self.defaults = defaults
        self.masters = masters

class FuncInfo:
    _immutable_fields_ = ["tp", "env_size", "min_nargs", "max_nargs", "name", "canjit"]
    __slots__ = "tp", "env_size", "min_nargs", "max_nargs", "name", "canjit"
    @look_inside
    def __init__(self, tp, env_size, min_nargs, max_nargs, name, canjit):
        assert isinstance(min_nargs, int), "TypeError"
        assert isinstance(max_nargs, int), "TypeError"
        assert isinstance(env_size, int), "TypeError"
        assert isinstance(canjit, bool), "TypeError"
        assert isinstance(name, str), "TypeError"
        assert isinstance(tp, int), "TypeError"
        self.min_nargs = min_nargs
        self.max_nargs = max_nargs
        self.env_size = env_size
        self.canjit = canjit
        self.name = name
        self.tp = tp

class BoundFuncValue(Value):
    _immutable_fields_ = ["func", "bound_obj"]
    __slots__ = "func", "bound_obj"
    @look_inside
    def __init__(self, func, bound_obj):
        assert isinstance(bound_obj, Value), "TypeError"
        assert isinstance(func, FuncValue), "TypeError"
        self.bound_obj = bound_obj
        self.func = func

class SpecialValue(Value):
    _immutable_fields_ = ["type", "str_value", "int_value"]
    __slots__ = "type", "str_value", "int_value"
    @look_inside
    def __init__(self, type, str_value=u"", int_value=0):
        assert isinstance(str_value, str), "TypeError"
        assert isinstance(int_value, int), "TypeError"
        assert isinstance(type, str), "TypeError"
        self.str_value = str_value
        self.int_value = int_value
        self.type = type

class NoneValue(Value):
    _immutable_fields_ = []
    __slots__ = ()
    @look_inside
    def __init__(self): pass

class ObjectValue(Value):
    _immutable_fields_ = ["attr_vals", "obj_info", "is_obj"]
    __slots__ = "attr_vals", "obj_info", "is_obj"
    @look_inside
    def __init__(self, obj_info, is_obj):
        assert isinstance(obj_info, ObjectInfo), "TypeError"
        assert isinstance(is_obj, bool), "TypeError"
        self.attr_vals = [None]
        self.obj_info = obj_info
        self.is_obj = is_obj

class ObjectInfo:
    _immutable_fields_ = ["type", "name", "mro", "obj_info"]
    __slots__ = "type", "name", "mro", "obj_info"
    @unroll_safe
    @look_inside
    def __init__(self, type, name, mro, obj_info):
        if obj_info is not None:
            assert isinstance(obj_info, ObjectInfo), "TypeError"
        assert isinstance(type, int), "TypeError" # even types for classes and odds for objects of that type
        assert isinstance(name, str), "TypeError"
        assert isinstance(mro, list), "TypeError"
        for cls in mro:
            assert isinstance(cls, ObjectValue), "TypeError"
            assert not cls.is_obj, "TypeError"
        self.obj_info = obj_info
        self.type = type
        self.name = name
        self.mro = mro

class ListValue(Value):
    _immutable_fields_ = ["array"]
    __slots__ = "array"
    @look_inside
    def __init__(self):
        self.array = []


NONE = const(NoneValue())
ZERO = const(IntValue(0))
ONE = const(IntValue(1))
FALSE = const(BoolValue(False))
TRUE = const(BoolValue(True))
PI = const(FloatValue(math.pi))
EPSILON = const(FloatValue(6e-8))
EMPTY_OBJ = const(ObjectValue(ObjectInfo(98, u"*unreachable*", [], ObjectInfo(99, u"*unreachable*", [], None)), False))


# General helpers
@look_inside
@elidable
def get_type(value):
    if value is None:
        return u"undefined" # This should never be used
    if value is NONE:
        return u"none"
    if isinstance(value, IntValue):
        return u"int"
    if isinstance(value, StrValue):
        return u"str"
    if isinstance(value, ListValue):
        return u"list"
    if isinstance(value, FuncValue):
        return u"func"
    if isinstance(value, ObjectValue):
        return u"class"
    if isinstance(value, FloatValue):
        return u"float"
    if isinstance(value, BoolValue):
        return u"bool"
    if isinstance(value, SpecialValue):
        if value.type == u"module":
            return u"module"
        elif value.type == u"file":
            return u"file"
        else:
            return u"unknown-special-type"
    return u"unknown"

@look_inside
def force_bool(val):
    if val is NONE:
        return False
    if isinstance(val, IntValue):
        return val.value != 0
    if isinstance(val, StrValue):
        return len(val.value) != 0
    if isinstance(val, ListValue):
        return len(val.array) != 0
    if isinstance(val, BoolValue):
        return val.value
    if isinstance(val, FloatValue):
        return not (-EPSILON.value < val.value < EPSILON.value)
    return True

@look_inside
@elidable
def to_bool_value(boolean):
    assert isinstance(boolean, bool), "TypeError"
    return TRUE if boolean else FALSE

@look_inside
def env_load(env, idx):
    assert isinstance(env, ENV_TYPE), "TypeError"
    if idx == 0:
        return ZERO
    if idx == 1:
        return ONE
    assert idx < const(len(env)), "InternalError"
    value = env[idx]
    if value is None:
        raise_error(u"InternalError: trying to load undefined from env")
    return value

@look_inside
def env_store(env, idx, value, chk=True):
    assert isinstance(env, ENV_TYPE), "TypeError"
    if (value is None) and chk:
        raise_error(u"InternalError: trying to store undefined inside env")
    assert idx < const(len(env)), "InternalError"
    env[idx] = value

@look_inside
def attr_vals_load(attr_vals, idx, chk=True):
    assert isinstance(attr_vals, list), "TypeError"
    assert idx < len(attr_vals), "InternalError"
    value = attr_vals[idx]
    if (value is None) and chk:
        raise_error(u"InternalError: trying to load undefined from attrs")
    return value

@look_inside
def attr_vals_store(attr_vals, idx, value):
    if value is None:
        raise_error(u"InternalError: trying to store undefined inside attrs")
    assert idx < len(attr_vals), "InternalError"
    attr_vals[idx] = value

@unroll_safe
@look_inside
def attr_vals_extend_until_len(attr_vals, new_length):
    assert isinstance(attr_vals, list), "TypeError"
    while len(attr_vals) < new_length:
        attr_vals.append(None)

@unroll_safe
@look_inside
def attr_access(attr_matrix, lens, obj, attr, storing):
    # Use the mro to figure out where attr is supposed to be
    mro = hint(const(obj.obj_info).mro, promote=True)
    if obj.is_obj:
        mro = [obj] + mro
    for cls in mro:
        if (cls is not obj) and storing:
            continue
        cls_type = const(const(cls.obj_info).type)
        row = hint(attr_matrix[cls_type], promote=True)
        soft, attr_idx = hint(row[attr], promote=True)
        soft, attr_idx = const(soft), const(attr_idx)
        if attr_idx != -1:
            if soft and cls.is_obj:
                if (attr_idx >= len(cls.attr_vals)) or (cls.attr_vals[attr_idx] is None):
                    continue
            break
    # Create the attr space in cls if attr_idx is -1
    else:
        cls = obj
        cls_type = const(const(cls.obj_info).type)
        attr_idx, lens[cls_type] = lens[cls_type], lens[cls_type]+1
        row = attr_matrix[cls_type]
        row[attr] = (False, attr_idx)
    if len(cls.attr_vals) <= attr_idx:
        attr_vals_extend_until_len(cls.attr_vals, attr_idx+1)
    return cls, attr_idx

@look_inside
@elidable # Since teleports is constant we can mark this as elidable
def teleports_get(teleports, label):
    teleports = hint(teleports, promote=True)
    label = const_str(label)
    result = teleports.get(label, None)
    result = const(result)
    return result

@look_inside
def _c3_merge(mros):
    # Stolen from: https://stackoverflow.com/a/54261655/11106801
    if len(mros) == 0:
        return []
    for mro in mros:
        assert isinstance(mro, list), "TypeError"
        for cls in mro:
            assert isinstance(cls, ObjectValue), "TypeError"
    for mro in mros:
        candidate = mro[0]
        failed = False
        for mro in mros:
            if failed: break
            for tail in mro[1:]:
                if tail == candidate:
                    failed = True
                    break
        if not failed:
            tails = []
            for mro in mros:
                if mro[0] is candidate:
                    if len(mro) > 1:
                        tails.append(mro[1:])
                    continue
                tails.append(mro)
            return [candidate] + _c3_merge(tails)
            # return [candidate] + _c3_merge([tail if head is candidate else [head, *tail] for head, *tail in mros])
    raise_type_error(u"No legal mro")

@look_inside
def print_err(text):
    assert isinstance(text, str), "TypeError"
    text += u"\n"
    os.write(2, text.encode("utf-8"))

@elidable
@look_inside
def fast_pow(base, power):
    # https://en.wikipedia.org/wiki/Exponentiation_by_squaring#With_constant_auxiliary_memory
    assert isinstance(base, int), "TypeError"
    assert isinstance(power, int), "TypeError"
    assert power >= 0, "ValueError"
    if power == 0: return 1 # assumes base != 0
    y = 1
    while power > 1:
        if power & 1:
            y *= base
            power -= 1
        base *= base
        power >>= 1
    return base*y

def _round_int(number, round_to):
    assert isinstance(round_to, int), "TypeError"
    assert isinstance(number, int), "TypeError"
    assert round_to <= 0, "ValueError"
    if round_to == 0:
        return number
    tmp = fast_pow(10, -round_to)
    mod = number % tmp
    number += tmp * bool((2*mod > tmp) + (number > 0)*(2*mod == tmp)) # stop rpython guards
    return number-mod

def _round_float_str(number, round_to):
    assert isinstance(round_to, int), "TypeError"
    assert isinstance(number, float), "TypeError"

    if not ((-15 < round_to) and (round_to < 15)):
        raise_value_error(u"Precision too high")

    if number < 0:
        output = _round_float_str(-number, round_to)
        for chr in output:
            if chr not in (u"0", u"."):
                return u"-" + output
        return output

    int_part = int(number)
    fra_part = number - int_part
    if round_to <= 0:
        bit = fast_pow(10, -round_to)
        tmp = (int_part // bit) * bit
        if 2*(int_part-tmp+fra_part) >= bit:
            return int_to_str(tmp + bit)
        else:
            return int_to_str(tmp)
    else:
        bit = fast_pow(10, round_to)
        mul_fra_part = fra_part * bit
        mul_fra_part = int(mul_fra_part+0.5)
        if mul_fra_part >= bit:
            int_part += 1
            mul_fra_part = 0
        str_mul_fra_part = int_to_str(mul_fra_part)
        zeros = len(int_to_str(bit)) - len(str_mul_fra_part) - 1
        str_mul_fra_part = u"0"*zeros + str_mul_fra_part
        return int_to_str(int_part) + u"." + str_mul_fra_part

@look_inside
def divmod(a, b):
    div = a // b
    mod = a - div*b
    return div, mod

@look_inside
def bytecode_debug_str(pc, bt):
    data_unicode = int_to_str(pc,zfill=2) + u"| " + bytecode_list_to_str([bt],mini=True)
    data = bytes2(data_unicode)
    while data[-1] == "\n":
        data = data[:-1]
    return data


class VirtualisableArray:
    # With help from @cfbolz <https://github.com/pypy/pypy/issues/5166>
    # Doesn't work with recursion
    _immutable_fields_ = ["array"]
    _virtualizable_ = ["array[*]"]
    __slots__ = "array"

    @look_inside
    def __init__(self, size):
        self = hint(self, access_directly=True, fresh_virtualizable=True)
        self.array = [None]*const(size)

    @look_inside
    def __getitem__(self, key):
        assert key >= 0, "ValueError"
        return self.array[key]

    @look_inside
    def __setitem__(self, key, value):
        assert key >= 0, "ValueError"
        self.array[key] = value

    @look_inside
    @elidable
    def __len__(self):
        return len(self.array)


ENV_IS_LIST = True
STACK_IS_LIST = False # if TRUE: super slow on raytracer
ENV_IS_VIRTUALISABLE = False

if not ENV_IS_LIST:
    assert not ENV_IS_VIRTUALISABLE, "Invalid compile settings"
    raise NotImplementedError("No longer supported")
if ENV_IS_VIRTUALISABLE:
    if ENTER_JIT_FUNC_CALL or ENTER_JIT_FUNC_RET:
        raise NotImplementedError("ENV_IS_VIRTUALISABLE breaks the interpreter if either ENTER_JIT_FUNC_CALL or ENTER_JIT_FUNC_RET")
    raise NotImplementedError("This doesn't work no matter what settings I provide :/")
    ENV_TYPE = VirtualisableArray
else:
    ENV_TYPE = list

if USE_JIT:
    def get_location(_, pc, bytecode, *__):
        # Got a rpython type union error. Took me 20min to figure out why so I am adding type checks here for future me
        assert isinstance(bytecode, list), "TypeError"
        assert isinstance(pc, int), "TypeError"
        for bt in bytecode:
            assert isinstance(bt, Bast), "TypeError"
        assert 0 <= pc < len(bytecode), "ValueError"
        return bytecode_debug_str(pc, bytecode[pc])
        return "Instruction[%s]" % bytes2(int_to_str(pc,zfill=2))
    virtualizables = ["env"] if ENV_IS_VIRTUALISABLE else []
    jitdriver = JitDriver(greens=["frame_size","pc","bytecode","teleports","SOURCE_CODE"],
                          reds=["next_cls_type","stack","env","func","attr_matrix","attrs","lens","global_scope","func_infos"],
                          virtualizables=virtualizables, get_printable_location=get_location)


# Stack linked list
class StackFrame:
    _immutable_fields_ = ["env", "pc", "func", "ret_reg", "prev_stack"]
    __slots__ = "env", "pc", "func", "ret_reg", "prev_stack"

    # @look_inside
    def __init__(self, env, func, pc, ret_reg, prev_stack):
        assert isinstance(func, FuncValue), "TypeError"
        assert isinstance(env, ENV_TYPE), "TypeError"
        assert isinstance(ret_reg, int), "TypeError"
        assert isinstance(pc, int), "TypeError"
        if prev_stack is not None:
            assert isinstance(prev_stack, StackFrame), "TypeError"
        self.env = env
        self.pc = pc
        self.func = func
        self.ret_reg = ret_reg
        self.prev_stack = prev_stack


# Error traceback helpers
@look_inside
def str_count_prefix(string, substring):
    count = 0
    while count < len(string):
        if not string[count*len(substring)].startswith(substring):
            break
        count += 1
    return count

@look_inside
def str_find(string, substring, from_idx):
    while from_idx < len(string):
        if string[from_idx:].startswith(substring):
            return from_idx
        from_idx += 1
    return len(string)

@look_inside
def _line_from_string(string, line_number):
    last_line = u""
    last_idx = 0
    for _ in range(line_number):
        idx = str_find(string, u"\n", last_idx)
        last_line = string[last_idx:idx]
        last_idx = idx+1
    return last_line

@look_inside
def _remove_repeating(stack, max_pc):
    # stack is of type list[FuncInfo,Pc]
    # We want to remove repeating patterns from the stack
    #   test = [1, 2,3, 2,3, 2,3, 2,3, 2,3, 4]
    #   test = [1, 2,3,2, 2,3,2, 2,3,2, 4]
    #   test = [1, 2,3,2,3,4, 2,3,2,3,4, 2,3,2,3,4, 4]
    #   test = [1, 2,2,2,2, 4]
    #   test = [1,1,1,1,1,1,1,1,1,1]
    new = []
    i = 0
    while i < len(stack):
        for j in range(i+1, len(stack)):
            # Detect loops
            looped = 0 # The period of the loop
            for looped in range(len(stack)-j):
                if stack[i+looped][1] != stack[j+looped][1]:
                    break
                if i+looped == j:
                    break
            # If no loop
            if not looped:
                continue
            # Figure out the total number of frames in all of the loops
            loopeds = 0
            for loopeds in range(len(stack)-j):
                if stack[i+loopeds][1] != stack[j+loopeds][1]:
                    break
            if j+loopeds+1 == len(stack): # edge case literally
                loopeds += 1
            # loops = (total frames in all loops) // (period of a loop)
            loops = loopeds//looped
            if loops < 3: # If not enough loops, skip
                continue
            # Add the first looped block
            for k in range(looped):
                new.append(stack[i+k%looped])
            # Add a tag showing the period and number of loops
            new.append((None, loops*max_pc+looped))
            # new.append(f"last {looped} items repeated {loops} more times")
            i += looped*loops # Skip the correct number of stack frames
            break
        else:
            # If no loop, add the current stack frame
            new.append(stack[i])
            i += 1
    return new

@look_inside
def data_from_err_idx(code, err):
    if err.start[0] == err.start[1] == err.end[0] == err.end[1] == 0:
        return -1, u"<code missing>", 0, 14, False
    line_str = _line_from_string(code, err.start[0])
    start = err.start[1]
    if err.start[0] == err.end[0]:
        size = err.end[1] - start
    else:
        size = len(line_str)-start+1
    assert start >= 0, "InternalError"
    assert size > 0, "InternalError"
    return err.start[0], line_str, start, size, True

@unroll_safe
@look_inside
def mul_isinstance(obj, classes):
    assert isinstance(classes, list), "TypeError"
    for Cls in classes:
        if isinstance(obj, Cls):
            return True
    return False

@look_inside
def get_err_idx_from_bt(bt):
    assert mul_isinstance(bt, [BCall,BStoreLoadDict,BStoreLoadList,BLiteral,BJump,BLoadLink,BRet,BDotDict,BDotList]), "InternalError"
    if isinstance(bt, BCall): return bt.err
    elif isinstance(bt, BStoreLoadDict): return bt.err
    elif isinstance(bt, BStoreLoadList): return bt.err
    elif isinstance(bt, BLiteral): return bt.err
    elif isinstance(bt, BJump): return bt.err
    elif isinstance(bt, BLoadLink): return bt.err
    elif isinstance(bt, BRet): return bt.err
    elif isinstance(bt, BDotDict): return bt.err
    elif isinstance(bt, BDotList): return bt.err
    else:
        assert False, "Impossible"

class InterpreterError(Exception):
    _immutable_fields_ = ["msg"]
    __slots__ = "msg"

    def __init__(self, msg):
        self.msg = msg


# Main interpreter
@unroll_safe
@look_inside
def interpret(flags, frame_size, env_size, attrs, bytecode, SOURCE_CODE, cmd_args):
    if not flags.is_set("ENV_IS_LIST"):
        print("\x1b[91m[ERROR]: ENV_IS_LIST==false is not longer supported\x1b[0m")
        raise SystemExit()
    if frame_size <= 0:
        print("\x1b[91m[ERROR]: Invalid frame_size in bytecode\x1b[0m")
        raise SystemExit()
    # Create teleports
    teleports = Dict()
    for i, bt in enumerate(bytecode):
        if isinstance(bt, Bable):
            teleports[const_str(bt.id)] = IntValue(i)
    # Create env
    # Note that regs is env[:frame_size]
    if ENV_IS_VIRTUALISABLE:
        env = VirtualisableArray(env_size + frame_size + 2)
    else:
        env = [None]*(env_size + frame_size + 2)
    envs = hint([env], promote=True)
    for i, op in enumerate(BUILTINS):
        assert isinstance(op, str), "TypeError"
        pure_op = op not in BUILTIN_MODULE_SIDES
        env[i+frame_size] = FuncValue(FuncInfo(i+len(bytecode), pure_op, 0, 0, op, False), envs, [])
    for i, op in enumerate(BUILTIN_MODULES):
        assert isinstance(op, str), "TypeError"
        env[i+frame_size+len(MODULE_ATTRS)] = SpecialValue(u"module", str_value=op)
    inner_cmd_args = ListValue()
    for cmd_arg in cmd_args:
        inner_cmd_args.array.append(StrValue(str(cmd_arg)))
    env[CMD_ARGS_IDX+frame_size] = inner_cmd_args
    # Start actual interpreter
    return _interpret(bytecode, teleports, env, attrs, SOURCE_CODE, frame_size)

@unroll_safe
@look_inside
def _interpret(bytecode, teleports, env, attrs, SOURCE_CODE, frame_size):
    SOURCE_CODE = const_str(SOURCE_CODE)
    pc = 0 # current instruction being executed
    stack = [] if STACK_IS_LIST else None
    next_cls_type = 0
    attr_matrix = hint([], promote=True)
    lens = hint([], promote=True)
    func = FuncValue(FuncInfo(0, 0, 0, 0, u"main-scope", False), [], [])
    global_scope = hint(env, promote=True)
    frame_size = const(frame_size)
    func_infos = hint([None]*len(bytecode), promote=True)

    while pc < len(bytecode):
        if USE_JIT:
            jitdriver.jit_merge_point(stack=stack, env=env, func=func, pc=pc, bytecode=bytecode, teleports=teleports, frame_size=frame_size, func_infos=func_infos,
                                      attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)
        bt = bytecode[pc]
        if DEBUG_LEVEL >= 3:
            debug(str(bytecode_debug_str(pc, bt)), 3)
        pc += 1

        try:
            if isinstance(bt, Bable):
                pass

            elif isinstance(bt, BStoreLoadList):
                # Get the correct scope (bt.link==0 and bt.link==1 are shortcuts to the top/bottom of envs)
                if bt.link == 0:
                    scope = env
                elif bt.link == -1:
                    if ENV_IS_VIRTUALISABLE:
                        if len(func.masters) == 0:
                            scope = env
                        else:
                            scope = func.masters[0]
                    else:
                        scope = global_scope
                else:
                    idx = const(len(func.masters))-bt.link
                    if idx == 0:
                        scope = global_scope
                    else:
                        scope = func.masters[idx]
                # Store/Load variable
                if bt.storing:
                    env_store(scope, bt.name+frame_size, env_load(env, bt.reg))
                else:
                    env_store(env, bt.reg, env_load(scope, bt.name+frame_size))

            elif isinstance(bt, BDotList):
                obj = env_load(env, bt.obj_reg)
                # List
                if isinstance(obj, ListValue):
                    if bt.storing:
                        raise_name_error(u"cannot change builtin attribute")
                    if bt.attr not in (LEN_IDX, APPEND_IDX, INDEX_IDX):
                        raise_name_error(u"Unknown attribute")
                    env_store(env, bt.reg, BoundFuncValue(global_scope[bt.attr+frame_size], obj))
                # String
                elif isinstance(obj, StrValue):
                    if bt.storing:
                        raise_name_error(u"cannot change builtin attribute")
                    if bt.attr not in (LEN_IDX, JOIN_IDX, INDEX_IDX, SPLIT_IDX, REPLACE_IDX):
                        raise_name_error(u"Unknown attribute")
                    env_store(env, bt.reg, BoundFuncValue(global_scope[bt.attr+frame_size], obj))
                # Special
                elif isinstance(obj, SpecialValue):
                    obj_type = const_str(obj.type)
                    # math.ε can be modified
                    if bt.storing:
                        if (obj_type == u"module") and (obj.str_value == u"math") and (bt.attr == EPSILON_IDX):
                            value = env_load(env, bt.reg)
                            new_epsilon = 0.0
                            if isinstance(value, FloatValue):
                                new_epsilon = value.value
                            elif isinstance(value, IntValue):
                                new_epsilon = value.value
                            EPSILON.value = new_epsilon
                        else:
                            raise_name_error(u"cannot change builtin attribute")
                    # Libraries
                    elif obj_type == u"module":
                        value = None
                        if obj.str_value == u"io":
                            if bt.attr not in (PRINT_IDX, OPEN_IDX):
                                raise_name_error(u"this")
                            value = global_scope[bt.attr+frame_size]
                        elif obj.str_value == u"math":
                            if bt.attr == PI_IDX:
                                value = PI
                            elif bt.attr == EPSILON_IDX:
                                value = EPSILON
                            else:
                                if bt.attr not in (SQRT_IDX, SIN_IDX, COS_IDX, TAN_IDX, POW_IDX, ROUND_IDX, STR_ROUND_IDX):
                                    raise_name_error(u"this")
                                value = global_scope[bt.attr+frame_size]
                        else:
                            raise_name_error(u"this")
                        env_store(env, bt.reg, value)
                    # File
                    elif obj_type == u"file":
                        if bt.attr not in (READ_IDX, WRITE_IDX, CLOSE_IDX):
                            raise_name_error(u"this")
                        env_store(env, bt.reg, BoundFuncValue(global_scope[bt.attr+frame_size], obj))
                    else:
                        raise_unreachable_error(u"TODO . operator on SpecialValue with obj.type=" + obj.type)
                # Function
                elif isinstance(obj, FuncValue):
                    if bt.storing:
                        raise_name_error(u"cannot change builtin attribute")
                    if obj is global_scope[STR_IDX+frame_size]:
                        if bt.attr not in (LEN_IDX, JOIN_IDX):
                            raise_name_error(u"Unknown attribute")
                    elif obj is global_scope[LIST_IDX+frame_size]:
                        if bt.attr not in (LEN_IDX, APPEND_IDX):
                            raise_name_error(u"Unknown attribute")
                    else:
                        raise_name_error(u"cannot access attributes of " + inner_repr(obj))
                    env_store(env, bt.reg, global_scope[bt.attr+frame_size])
                # Object/Class
                elif isinstance(obj, ObjectValue):
                    # Get cls (the object storing attr) and attr_idx (the idx into cls.attr_vals)
                    cls, attr_idx = attr_access(attr_matrix, lens, obj, bt.attr, bt.storing)
                    # Use the information above to execute the bytecode
                    attr_idx = const(attr_idx)
                    assert 0 <= attr_idx < len(cls.attr_vals), "InternalError"
                    if bt.storing:
                        attr_vals_store(cls.attr_vals, attr_idx, env_load(env, bt.reg))
                    else:
                        value = attr_vals_load(cls.attr_vals, attr_idx)
                        obj_type = const(const(obj.obj_info).type)
                        if isinstance(value, FuncValue) and obj.is_obj:
                            value = BoundFuncValue(value, obj)
                            _, attr_idx = attr_matrix[obj_type][bt.attr]
                            if attr_idx == -1:
                                attr_idx, lens[obj_type] = lens[obj_type], lens[obj_type]+1
                                attr_matrix[obj_type][bt.attr] = (True, attr_idx)
                            attr_vals_extend_until_len(obj.attr_vals, attr_idx+1)
                            attr_vals_store(obj.attr_vals, attr_idx, value)
                        env_store(env, bt.reg, value)
                else:
                    raise_type_error(u". operator expects object got " + get_type(obj) + u" instead")

            elif isinstance(bt, BLiteral):
                bt_literal = bt.literal
                if bt.type == BLiteral.INT_T:
                    assert isinstance(bt_literal, BLiteralInt), "TypeError"
                    literal = IntValue(bt_literal.value)
                elif bt.type == BLiteral.BOOL_T:
                    assert isinstance(bt_literal, BLiteralBool), "TypeError"
                    literal = BoolValue(bt_literal.value)
                elif bt.type == BLiteral.FUNC_T:
                    assert isinstance(bt_literal, BLiteralFunc), "TypeError"
                    if bt_literal.link == 0:
                        master = env
                    else:
                        master = func.masters[len(func.masters)-bt_literal.link]
                    if func_infos[pc] is None:
                        tp = teleports[bt_literal.tp_label]
                        assert isinstance(tp, IntValue), "InternalError"
                        func_infos[pc] = FuncInfo(tp.value, bt_literal.env_size, bt_literal.min_nargs, bt_literal.max_nargs, bt_literal.name, bt_literal.record)
                    func_info = const(func_infos[pc])
                    defaults = [env_load(env, reg) for reg in bt_literal.defaults]
                    literal = FuncValue(func_info, func.masters+[master], defaults)
                elif bt.type == BLiteral.STR_T:
                    assert isinstance(bt_literal, BLiteralStr), "TypeError"
                    literal = StrValue(bt_literal.value)
                elif bt.type == BLiteral.FLOAT_T:
                    assert isinstance(bt_literal, BLiteralFloat), "TypeError"
                    literal = FloatValue(bt_literal.value)
                elif bt.type == BLiteral.NONE_T:
                    literal = NONE
                elif bt.type == BLiteral.UNDEFINED_T:
                    env_store(env, bt.reg, None, chk=False)
                    continue
                elif bt.type == BLiteral.LIST_T:
                    literal = ListValue()
                elif bt.type == BLiteral.CLASS_T:
                    assert isinstance(bt_literal, BLiteralClass), "TypeError"
                    # Create ObjectValue type
                    _mros = []
                    for base_reg in bt_literal.bases:
                        base = env_load(env, base_reg)
                        if not isinstance(base, ObjectValue):
                            raise_type_error(u"can't inherit from " + get_type(base))
                        if base.is_obj:
                            raise_type_error(u"can't inherit from " + inner_repr(base))
                        _mros.append(base.obj_info.mro)
                    # Register new class
                    cls_type, next_cls_type = next_cls_type, next_cls_type+2 # even types for classes and odds for objects of that type
                    for _ in range(2):
                        row = [(False, -1) for _ in range(len(attrs))]
                        row = hint(row, promote=True)
                        row[0] = hint((False, 0), promote=True)
                        attr_matrix.append(row)
                    lens.extend([1,1])
                    mro = [EMPTY_OBJ]+_c3_merge(_mros)
                    literal = ObjectValue(ObjectInfo(cls_type, bt_literal.name, mro, ObjectInfo(cls_type+1, bt_literal.name, mro, None)), False)
                    mro[0] = literal
                else:
                    raise NotImplementedError()
                env_store(env, bt.reg, literal)

            elif isinstance(bt, BJump):
                value = env_load(env, bt.condition_reg)
                # Clear the regs in bt.clear
                for reg in bt.clear:
                    if reg > 1:
                        env_store(env, reg, None, chk=False)
                condition = force_bool(value)
                # if condition != bt.negated: # RPython's JIT can't constant fold in this form :/
                if (condition and (not bt.negated)) or ((not condition) and bt.negated):
                    tp = teleports_get(teleports, bt.label)
                    assert isinstance(tp, IntValue), "TypeError"
                    old_pc, pc = pc, tp.value
                    assert isinstance(bytecode[pc], Bable), "InternalError"
                    if USE_JIT and (pc < old_pc): # Tell the JIT about the jump
                        jitdriver.can_enter_jit(stack=stack, env=env, func=func, pc=pc, bytecode=bytecode, teleports=teleports, frame_size=frame_size, func_infos=func_infos,
                                                attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)

            elif isinstance(bt, BRegMove):
                env_store(env, bt.reg1, env_load(env, bt.reg2))

            elif isinstance(bt, BRet):
                old_pc = pc
                value = env_load(env, bt.reg)
                enter_jit = ENTER_JIT_FUNC_RET and const(func.func_info.canjit)
                env_store(env, bt.reg, None, chk=False)
                if not stack:
                    exit_value = 0
                    if isinstance(value, IntValue):
                        exit_value = value.value
                    elif isinstance(value, BoolValue):
                        exit_value = int(value.value)
                    else:
                        raise_type_error(u"exit value should be an int not " + get_type(value))
                    return exit_value
                if ENV_IS_VIRTUALISABLE:
                    hint(env, force_virtualizable=True)
                if STACK_IS_LIST:
                    env, func, pc, ret_reg = stack.pop()
                else:
                    env, func, pc, ret_reg = stack.env, stack.func, stack.pc, stack.ret_reg
                    stack = stack.prev_stack
                env_store(env, const(ret_reg), value)
                if USE_JIT and enter_jit: # Tell the JIT about the jump
                    jitdriver.can_enter_jit(stack=stack, env=env, func=func, pc=pc, bytecode=bytecode, teleports=teleports, frame_size=frame_size, func_infos=func_infos,
                                            attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)
                func = hint(func, promote=True)

            elif isinstance(bt, BCall):
                args = []
                enter_jit = True
                _func = env_load(env, bt.regs[1])
                if isinstance(_func, BoundFuncValue):
                    args.append(_func.bound_obj)
                    _func = _func.func
                elif isinstance(_func, FuncValue):
                    pass
                elif isinstance(_func, ObjectValue):
                    if not _func.is_obj:
                        # Create new object and call constructor
                        _func = const(_func)
                        obj_info = const(const(_func.obj_info).obj_info)
                        self = ObjectValue(obj_info, True)
                        args = [self]
                        _func = const(_func.attr_vals[CONSTRUCTOR_IDX])
                        if _func is None:
                            # Default constructor
                            env_store(env, bt.regs[0], self)
                            continue
                        if not isinstance(_func, FuncValue):
                            raise_type_error(u"constructor should be a function not " + get_type(_func))
                    else:
                        raise_type_error(inner_repr(_func) + u" is not callable")
                else:
                    raise_type_error(get_type(_func) + u" is not callable")

                func_info = const(_func.func_info)
                enter_jit = func_info.canjit
                if func_info.tp < len(bytecode):
                    # Check number of args
                    if len(bt.regs)-2+len(args) > func_info.max_nargs:
                        raise_type_error(u"too many arguments")
                    elif len(bt.regs)-2+len(args) < func_info.min_nargs:
                        raise_type_error(u"too few arguments")
                    # Create a new stack frame
                    if STACK_IS_LIST:
                        stack.append((env, func, pc, bt.regs[0]))
                    else:
                        stack = StackFrame(env, func, pc, bt.regs[0], stack)
                    # Build the arguments
                    tmp = [None]*func_info.max_nargs
                    for i in range(len(_func.defaults)):
                        tmp[i+func_info.min_nargs] = _func.defaults[i]
                    for i in range(len(args)):
                        tmp[i] = args[i]
                    for i in range(len(bt.regs)-2):
                        tmp[i+len(args)] = env_load(env, bt.regs[i+2])
                    # Clear the regs in bt.clear
                    for reg in bt.clear:
                        if reg > 1:
                            env_store(env, reg, None, chk=False)
                    # Set the pc/env/func
                    func, old_pc, pc = _func, pc, func_info.tp
                    if ENV_IS_VIRTUALISABLE:
                        hint(env, force_virtualizable=True)
                        env = VirtualisableArray(func_info.env_size+frame_size)
                    else:
                        env = [None]*(func_info.env_size+frame_size)
                    # Copy arguments values into new env
                    for i in range(len(tmp)):
                        env_store(env, i+2, tmp[i])
                    enter_jit = enter_jit and ENTER_JIT_FUNC_CALL
                else: # Built-ins
                    op_idx = func_info.tp - len(bytecode)
                    pure_op = bool(func_info.env_size)
                    op = BUILTINS[op_idx]
                    args += [env_load(env, bt.regs[i]) for i in range(2,len(bt.regs))]
                    if op_idx == ISINSTANCE_IDX:
                        value = inner_isinstance(global_scope, frame_size, args)
                    elif pure_op:
                        value = builtin_pure(op, args, op)
                    else:
                        value = builtin_side(op, args)
                    env_store(env, bt.regs[0], value)
                    # Clear the regs in bt.clear
                    for reg in bt.clear:
                        if reg > 1:
                            env_store(env, reg, None, chk=False)
                if USE_JIT and enter_jit: # Tell the JIT about the jump
                    jitdriver.can_enter_jit(stack=stack, env=env, func=func, pc=pc, bytecode=bytecode, teleports=teleports, frame_size=frame_size, func_infos=func_infos,
                                            attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)

            elif isinstance(bt, BLoadLink) or isinstance(bt, BStoreLoadDict) or isinstance(bt, BDotDict):
                raise_unreachable_error(u"ENV_IS_LIST==false is not longer supported so this bytecode instruction is illegal")

            else:
                raise NotImplementedError("Haven't implemented this bytecode yet")

        except InterpreterError as error:
            funcs = [(func.func_info, pc)]
            if STACK_IS_LIST:
                for _, stack_func, _, stack_pc, _ in stack:
                    funcs.append((stack_func.func_info, stack_pc))
            else:
                s = stack
                while s is not None:
                    funcs.append((s.func.func_info, s.pc))
                    s = s.prev_stack
            new_funcs = []
            for i in range(len(funcs)):
                new_funcs.append(funcs[len(funcs)-i-1])
            new_funcs = _remove_repeating(new_funcs, len(bytecode))
            print_err(u"Traceback (most recent call last):")
            for i, (stack_func,stack_pc) in enumerate(new_funcs):
                if stack_func is None:
                    loops = stack_pc // len(bytecode)
                    looped = stack_pc - loops*len(bytecode)
                    if looped > 1:
                        text = u"   [[\x1b[95mLast " + int_to_str(looped) + u" frames repeated "
                    else:
                        text = u"   [[\x1b[95mLast frame repeated "
                    if loops > 1:
                        text += int_to_str(loops) + u" more times\x1b[0m]]"
                    else:
                        text += u"1 more time\x1b[0m]]"
                    print_err(text)
                else:
                    bt = bytecode[stack_pc-1]
                    err_idx = get_err_idx_from_bt(bt)
                    line, line_str, start, size, success = data_from_err_idx(SOURCE_CODE, err_idx)
                    indent = str_count_prefix(line_str, u" ") + str_count_prefix(line_str, u"\t")
                    line_str, start = line_str[indent:], start-indent
                    assert start >= 0, "InternalError"
                    print_err(u"   File \x1b[95m<???>\x1b[0m in \x1b[95m<" + stack_func.name + u">\x1b[0m on line \x1b[95m" + int_to_str(line) + u"\x1b[0m:")
                    print_err(u" "*6 + line_str[:start] + u"\x1b[91m" + line_str[start:start+size] + u"\x1b[0m" + line_str[start+size:])
                    print_err(u" "*(start+6) + u"\x1b[91m" + u"^"*size + u"\x1b[0m")
            print_err(u"\x1b[93m" + error.msg + u"\x1b[0m")
            return 1
    return 0


@unroll_safe
@look_inside
def inner_isinstance(global_scope, frame_size, args):
    if len(args) != 2:
        raise_type_error(u"isinstance takes 2 arguments")
    instance, cls = args
    cls = const(cls)
    if isinstance(instance, ObjectValue):
        if not instance.is_obj:
            return FALSE
    if isinstance(cls, FuncValue):
        output = False
        if cls is const(global_scope[INT_IDX+frame_size]):
            output = isinstance(instance, IntValue) or isinstance(instance, BoolValue)
        elif cls is const(global_scope[FLOAT_IDX+frame_size]):
            output = isinstance(instance, IntValue) or isinstance(instance, BoolValue) or isinstance(instance, FloatValue)
        elif cls is const(global_scope[STR_IDX+frame_size]):
            output = isinstance(instance, StrValue)
        elif cls is const(global_scope[LIST_IDX+frame_size]):
            output = isinstance(instance, ListValue)
        elif cls is const(global_scope[BOOL_IDX+frame_size]):
            output = isinstance(instance, BoolValue)
        else:
            raise_type_error(u"the 2nd argument of isinstance should be a type or a list of types")
        return to_bool_value(output)
    elif isinstance(cls, ObjectValue):
        if not isinstance(instance, ObjectValue):
            return FALSE
        if cls.is_obj:
            return FALSE
        cls_obj_info = const(cls.obj_info)
        instance_obj_info = const(instance.obj_info)
        if cls_obj_info.obj_info is const(instance_obj_info):
            return TRUE
        for super_cls in hint(cls_obj_info.mro, promote=True):
            if super_cls.obj_info == instance_obj_info:
                return TRUE
        return FALSE
    elif isinstance(cls, ListValue):
        for subcls in cls.array:
            if inner_isinstance(global_scope, frame_size, [instance, subcls]):
                return TRUE
        return FALSE
    else:
        raise_type_error(u"the 2nd argument of isinstance should be a type")


# Builtin helpers
@look_inside
def builtin_pure(op, args, op_print):
    if op in (u"*", u"==", u"!="):
        if (len(args) > 0) and isinstance(args[0], ListValue):
            return builtin_pure_rollable(op, args, op)
    elif op in (u"[]", u"$idx", u"join", u"split", u"replace"):
        return builtin_pure_rollable(op, args, op)
    return builtin_pure_unrollable(op, args, op)

# This can't have @unroll_safe
@look_inside
def builtin_pure_rollable(op, args, op_print):
    if op == u"*":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, ListValue):
            mult = 0
            if isinstance(arg1, IntValue):
                mult = arg1.value
            elif isinstance(arg1, BoolValue):
                mult = int(arg1.value)
            else:
                raise_type_error(u"can't " + op_print + u" list with " + get_type(arg1))
            new = ListValue()
            for _ in range(mult):
                new.array += arg0.array
            return new
        elif isinstance(arg0, IntValue):
            assert isinstance(arg1, ListValue), "InternalError"
            new = ListValue()
            for _ in range(arg0.value):
                new.array += arg1.array
            return new
        elif isinstance(arg0, BoolValue):
            assert isinstance(arg1, ListValue), "InternalError"
            new = ListValue()
            for _ in range(int(arg0.value)):
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
            if len(arg0.array) != len(arg1.array):
                return FALSE
            for i in range(len(arg0.array)):
                v = builtin_pure(u"==", [arg0.array[i],arg1.array[i]], op_print)
                assert isinstance(v, BoolValue), "InternalError"
                if not v.value:
                    return FALSE
            return TRUE
        return FALSE

    elif op == u"!=":
        v = builtin_pure_rollable(u"==", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return to_bool_value(not v.value)

    elif op == u"$idx":
        if len(args) != 4:
            raise_type_error(op_print + u" expects 4 arguments (list, start, stop, step)")
        arg0, arg1, arg2, arg3 = args
        if isinstance(arg0, ListValue):
            length = len(arg0.array)
            start, stop, step = _get_full_idx_args(op_print, length, arg1, arg2, arg3)
            new = ListValue()
            for idx in range(start, stop, step):
                if 0 <= idx < length:
                    new.array.append(arg0.array[idx])
            return new
        elif isinstance(arg0, StrValue):
            length = len(arg0.value)
            start, stop, step = _get_full_idx_args(op_print, length, arg1, arg2, arg3)
            new = []
            for idx in range(start, stop, step):
                if 0 <= idx < length:
                    new.append(arg0.value[idx])
            return StrValue(u"".join(new))
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg0))

    elif op == u"[]":
        new = ListValue()
        new.array = args
        return new

    elif op == u"join":
        if len(args) != 2:
            raise_type_error(u"list." + op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, StrValue):
            raise_type_error(u"list." + op_print + u" doesn't support " + get_type(arg0) + u" for the 1st arg")
        if not isinstance(arg1, ListValue):
            raise_type_error(u"list." + op_print + u" doesn't support " + get_type(arg1) + u" for the 2nd arg")
        sep = arg0.value
        array = [u""]*len(arg1.array) # pre-allocate space
        for i in range(len(arg1.array)):
            array[i] = inner_repr(arg1.array[i])
        return StrValue(sep.join(array))

    elif op == u"split":
        if len(args) == 2:
            arg0, arg1 = args
            if not isinstance(arg0, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u" with " + get_type(arg1))
            if not isinstance(arg1, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u" with " + get_type(arg1))
            # RPython can't split python2-unicode but can split python2-str
            array = str_split(arg0.value, arg1.value)
            new = ListValue()
            new.array = [NONE]*len(array)
            for i in range(len(array)):
                new.array[i] = StrValue(array[i])
            return new
        elif len(args) == 3:
            arg0, arg1, arg2 = args
            if not isinstance(arg0, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2))
            if not isinstance(arg1, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2))
            if not isinstance(arg2, IntValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2))
            number_split = arg2.value
            if number_split == -1:
                return builtin_pure_rollable(op, [arg0,arg1], op_print)
            if number_split <= 0:
                raise_value_error(u"the number passed in str." + op_print + u" must be > 0")
            array = str_split_n(arg0.value, arg1.value, number_split)
            new = ListValue()
            new.array = [NONE]*len(array)
            for i in range(len(array)):
                new.array[i] = StrValue(array[i])
            return new
        else:
            raise_type_error(u"str." + op_print + u" expects 2 or 3 arguments")

    elif op == u"replace":
        if len(args) == 3:
            arg0, arg1, arg2 = args
            if not isinstance(arg0, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2))
            if not isinstance(arg1, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2))
            if not isinstance(arg2, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2))
            array = str_split(arg0.value, arg1.value)
            return StrValue(arg2.value.join(array))
        elif len(args) == 4:
            arg0, arg1, arg2, arg3 = args
            if not isinstance(arg0, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2) + u", " + get_type(arg3))
            if not isinstance(arg1, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2) + u", " + get_type(arg3))
            if not isinstance(arg2, StrValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2) + u", " + get_type(arg3))
            if not isinstance(arg3, IntValue):
                raise_type_error(u"str." + op_print + u" not supported on " + get_type(arg0) + u", " + get_type(arg1) + u", " + get_type(arg2) + u", " + get_type(arg3))
            number_split = arg3.value
            if number_split == -1:
                return builtin_pure_rollable(op, [arg0,arg1,arg2], op_print)
            if number_split <= 0:
                raise_value_error(u"the number passed in str." + op_print + u" must be > 0")
            array = str_split_n(arg0.value, arg1.value, number_split)
            return StrValue(arg2.value.join(array))
        else:
            raise_type_error(u"str." + op_print + u" expects 3 arguments")

    else:
        if op == op_print:
            raise_unreachable_error(u"builtin-pure-list " + op + u" not implemented (needed for " + op_print + u")")
        else:
            raise_unreachable_error(u"builtin-pure-list " + op + u" not implemented")


@unroll_safe
@look_inside
def builtin_pure_unrollable(op, args, op_print):
    if op == u"+":
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, IntValue):
                return arg
            elif isinstance(arg, BoolValue):
                return IntValue(int(arg.value))
            elif isinstance(arg, FloatValue):
                return arg
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, IntValue):
                if isinstance(arg1, IntValue):
                    return IntValue(arg0.value + arg1.value)
                elif isinstance(arg1, BoolValue):
                    return IntValue(arg0.value + arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value + arg1.value)
            elif isinstance(arg0, StrValue):
                if isinstance(arg1, StrValue):
                    return StrValue(arg0.value + arg1.value)
            elif isinstance(arg0, FloatValue):
                if isinstance(arg1, IntValue):
                    return FloatValue(arg0.value + arg1.value)
                elif isinstance(arg1, BoolValue):
                    return FloatValue(arg0.value + arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value + arg1.value)
            elif isinstance(arg0, ListValue):
                if isinstance(arg1, ListValue):
                    new = ListValue()
                    new.array = arg0.array + arg1.array
                    return new
            elif isinstance(arg0, BoolValue):
                if isinstance(arg1, IntValue):
                    return IntValue(arg0.value + arg1.value)
                elif isinstance(arg1, BoolValue):
                    return IntValue(arg0.value + arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value + arg1.value)
        else:
            raise_type_error(op_print + u" expects 1 or 2 arguments")

    elif op == u"-":
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, IntValue):
                return IntValue(-arg.value)
            elif isinstance(arg, BoolValue):
                return IntValue(-int(arg.value))
            elif isinstance(arg, FloatValue):
                return FloatValue(-arg.value)
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, IntValue):
                if isinstance(arg1, IntValue):
                    return IntValue(arg0.value - arg1.value)
                elif isinstance(arg1, BoolValue):
                    return IntValue(arg0.value - arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value - arg1.value)
            elif isinstance(arg0, FloatValue):
                if isinstance(arg1, IntValue):
                    return FloatValue(arg0.value - arg1.value)
                elif isinstance(arg1, BoolValue):
                    return FloatValue(arg0.value - arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value - arg1.value)
            elif isinstance(arg0, BoolValue):
                if isinstance(arg1, IntValue):
                    return IntValue(arg0.value - arg1.value)
                elif isinstance(arg1, BoolValue):
                    return IntValue(arg0.value - arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value - arg1.value)

    elif op == u"*":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value * arg1.value)
            elif isinstance(arg1, BoolValue):
                return IntValue(arg0.value * arg1.value)
            elif isinstance(arg1, FloatValue):
                return FloatValue(arg0.value * arg1.value)
            elif isinstance(arg1, StrValue):
                return StrValue(arg0.value * arg1.value)
        elif isinstance(arg0, StrValue):
            if isinstance(arg1, IntValue):
                return StrValue(arg0.value * arg1.value)
            elif isinstance(arg1, BoolValue):
                return arg0 if arg1.value else StrValue(u"")
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, IntValue):
                return FloatValue(arg0.value * arg1.value)
            elif isinstance(arg1, BoolValue):
                return FloatValue(arg0.value * arg1.value)
            elif isinstance(arg1, FloatValue):
                return FloatValue(arg0.value * arg1.value)
        elif isinstance(arg0, BoolValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value * arg1.value)
            elif isinstance(arg1, BoolValue):
                return IntValue(arg0.value * arg1.value)
            elif isinstance(arg1, FloatValue):
                return FloatValue(arg0.value * arg1.value)
            elif isinstance(arg1, StrValue):
                return arg1 if arg0.value else StrValue(u"")

    elif op == u"%":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        div0 = div1 = 0
        if isinstance(arg0, IntValue):
            div0 = arg0.value
        elif isinstance(arg0, BoolValue):
            div0 = arg0.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg0) + u" and " + get_type(arg1))
        if isinstance(arg1, IntValue):
            div1 = arg1.value
        elif isinstance(arg1, BoolValue):
            div1 = arg1.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg1) + u" and " + get_type(arg1))
        try:
            return IntValue(div0 % div1)
        except ZeroDivisionError:
            raise_value_error(u"division by zero")

    elif op == u"//":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        div0 = div1 = 0
        if isinstance(arg0, IntValue):
            div0 = arg0.value
        elif isinstance(arg0, BoolValue):
            div0 = arg0.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg0) + u" and " + get_type(arg1))
        if isinstance(arg1, IntValue):
            div1 = arg1.value
        elif isinstance(arg1, BoolValue):
            div1 = arg1.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg1) + u" and " + get_type(arg1))
        try:
            return IntValue(div0 // div1)
        except ZeroDivisionError:
            raise_value_error(u"division by zero")

    elif op == u"<<":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        val0 = val1 = 0
        if isinstance(arg0, IntValue):
            val0 = arg0.value
        elif isinstance(arg0, BoolValue):
            val0 = arg0.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg0) + u" and " + get_type(arg1))
        if isinstance(arg1, IntValue):
            val1 = arg1.value
        elif isinstance(arg1, BoolValue):
            val1 = arg1.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg1) + u" and " + get_type(arg1))
        return IntValue(val0 << val1)

    elif op == u">>":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        val0 = val1 = 0
        if isinstance(arg0, IntValue):
            val0 = arg0.value
        elif isinstance(arg0, BoolValue):
            val0 = arg0.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg0) + u" and " + get_type(arg1))
        if isinstance(arg1, IntValue):
            val1 = arg1.value
        elif isinstance(arg1, BoolValue):
            val1 = arg1.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg1) + u" and " + get_type(arg1))
        return IntValue(val0 >> val1)

    elif op == u"&":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        val0 = val1 = 0
        if isinstance(arg0, IntValue):
            val0 = arg0.value
        elif isinstance(arg0, BoolValue):
            val0 = arg0.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg0) + u" and " + get_type(arg1))
        if isinstance(arg1, IntValue):
            val1 = arg1.value
        elif isinstance(arg1, BoolValue):
            val1 = arg1.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg1) + u" and " + get_type(arg1))
        return IntValue(val0 & val1)

    elif op == u"|":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        val0 = val1 = 0
        if isinstance(arg0, IntValue):
            val0 = arg0.value
        elif isinstance(arg0, BoolValue):
            val0 = arg0.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg0) + u" and " + get_type(arg1))
        if isinstance(arg1, IntValue):
            val1 = arg1.value
        elif isinstance(arg1, BoolValue):
            val1 = arg1.value
        else:
            raise_type_error(op_print + u" unsupported on " + get_type(arg1) + u" and " + get_type(arg1))
        return IntValue(val0 | val1)

    elif op == u"==":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                if arg0.value == arg1.value:
                    return TRUE
            elif isinstance(arg1, BoolValue):
                if arg0.value == arg1.value:
                    return TRUE
            elif isinstance(arg1, FloatValue):
                if arg1.value-EPSILON.value < arg0.value < arg1.value+EPSILON.value:
                    return TRUE
            return FALSE
        elif isinstance(arg0, BoolValue):
            if isinstance(arg1, IntValue):
                if arg0.value == arg1.value:
                    return TRUE
            elif isinstance(arg1, BoolValue):
                if arg0.value == arg1.value:
                    return TRUE
            elif isinstance(arg1, FloatValue):
                if arg1.value-EPSILON.value < arg0.value < arg1.value+EPSILON.value:
                    return TRUE
            return FALSE
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, StrValue):
                return FALSE
            if arg0.value == arg1.value:
                return TRUE
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, IntValue):
                if arg0.value-EPSILON.value < arg1.value < arg0.value+EPSILON.value:
                    return TRUE
            elif isinstance(arg1, BoolValue):
                if arg0.value-EPSILON.value < arg1.value < arg0.value+EPSILON.value:
                    return TRUE
            elif isinstance(arg1, FloatValue):
                if arg0.value-EPSILON.value < arg1.value < arg0.value+EPSILON.value:
                    return TRUE
            return FALSE
        elif arg0 is arg1:
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
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, IntValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, BoolValue):
                return BoolValue(arg0.value < arg1.value)
        elif isinstance(arg0, StrValue):
            if isinstance(arg1, StrValue):
                return BoolValue(arg0.value < arg1.value)
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, IntValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, BoolValue):
                return BoolValue(arg0.value < arg1.value)
        elif isinstance(arg0, BoolValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, IntValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, BoolValue):
                return BoolValue(arg0.value < arg1.value)

    elif op == u">":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, IntValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, BoolValue):
                return BoolValue(arg0.value > arg1.value)
        elif isinstance(arg0, StrValue):
            if isinstance(arg1, StrValue):
                return BoolValue(arg0.value > arg1.value)
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, IntValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, BoolValue):
                return BoolValue(arg0.value > arg1.value)
        elif isinstance(arg0, BoolValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, IntValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, BoolValue):
                return BoolValue(arg0.value > arg1.value)

    elif op == u"<=":
        v = builtin_pure(u">", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return to_bool_value(not v.value)

    elif op == u">=":
        v = builtin_pure(u"<", args, op_print)
        assert isinstance(v, BoolValue), "InternalError"
        return to_bool_value(not v.value)

    elif op == u"/":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        try:
            if isinstance(arg0, IntValue):
                if isinstance(arg1, IntValue):
                    if arg1.value == 0: raise_value_error(u"division by zero")
                    return FloatValue(float(arg0.value)/float(arg1.value))
                elif isinstance(arg1, BoolValue):
                    if not arg1.value: raise_value_error(u"division by zero")
                    return FloatValue(float(arg0.value)/float(arg1.value))
                elif isinstance(arg1, FloatValue):
                    if arg1.value == 0: raise_value_error(u"division by zero")
                    return FloatValue(float(arg0.value)/arg1.value)
            elif isinstance(arg0, FloatValue):
                if isinstance(arg1, IntValue):
                    if arg1.value == 0: raise_value_error(u"division by zero")
                    return FloatValue(arg0.value/float(arg1.value))
                elif isinstance(arg1, BoolValue):
                    if not arg1.value: raise_value_error(u"division by zero")
                    return FloatValue(arg0.value/float(arg1.value))
                elif isinstance(arg1, FloatValue):
                    if arg1.value == 0: raise_value_error(u"division by zero")
                    return FloatValue(arg0.value/arg1.value)
            elif isinstance(arg0, BoolValue):
                if isinstance(arg1, IntValue):
                    if arg1.value == 0: raise_value_error(u"division by zero")
                    return FloatValue(float(arg0.value)/float(arg1.value))
                elif isinstance(arg1, BoolValue):
                    if not arg1.value: raise_value_error(u"division by zero")
                    return FloatValue(float(arg0.value)/float(arg1.value))
                elif isinstance(arg1, FloatValue):
                    if arg1.value == 0: raise_value_error(u"division by zero")
                    return FloatValue(float(arg0.value)/arg1.value)
        except ZeroDivisionError:
            raise_value_error(u"division by zero")

    elif op == u"sqrt":
        if len(args) != 1:
            raise_type_error(u"math." + op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.sqrt(arg.value))
        elif isinstance(arg, BoolValue):
            return FloatValue(math.sqrt(arg.value))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.sqrt(arg.value))
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"round":
        if len(args) != 2:
            raise_type_error(u"math." + op_print + u" expects only 2 arguments")
        arg0, arg1 = args
        round_to = 0
        if isinstance(arg1, BoolValue):
            round_to = arg1.value
        elif isinstance(arg1, IntValue):
            round_to = arg1.value
        else:
            raise_type_error(u"math." + op_print + u" expects an int for the 2nd argument")
        if isinstance(arg0, IntValue):
            if round_to <= 0:
                return IntValue(_round_int(arg0.value, round_to))
            else:
                return FloatValue(float(arg0.value))
        elif isinstance(arg0, BoolValue):
            if round_to == 0:
                return IntValue(arg0.value)
            elif round_to > 0:
                return FloatValue(1.0 * arg0.value)
            else:
                return ZERO
        elif isinstance(arg0, FloatValue):
            tmp = math.pow(10.0, -float(round_to))
            value = arg0.value
            add = 0.5 * (2*(value > 0)-1) # stop guards
            return FloatValue(tmp * int(arg0.value/tmp + add))
        else:
            raise_type_error(u"math." + op_print + u" expects a number for the 1st argument")
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"str_round":
        if len(args) != 2:
            raise_type_error(u"math." + op_print + u" expects only 2 arguments")
        arg0, arg1 = args
        round_to = 0
        if isinstance(arg1, BoolValue):
            round_to = arg1.value
        elif isinstance(arg1, IntValue):
            round_to = arg1.value
        else:
            raise_type_error(u"math." + op_print + u" expects an int for the 2nd argument")
        if isinstance(arg0, IntValue):
            if round_to <= 0:
                return StrValue(int_to_str(_round_int(arg0.value, round_to)))
            else:
                return StrValue(_round_float_str(float(arg0.value), round_to))
        elif isinstance(arg0, BoolValue):
            if round_to == 0:
                return StrValue(u"1")
            elif round_to > 0:
                return StrValue(_round_float_str(float(arg0.value), round_to))
            else:
                return StrValue(u"0")
        elif isinstance(arg0, FloatValue):
            return StrValue(_round_float_str(arg0.value, round_to))
        else:
            raise_type_error(u"math." + op_print + u" expects a number for the 1st argument")
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"pow":
        if len(args) != 2:
            raise_type_error(u"math." + op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(int(math.pow(arg0.value, arg1.value)))
            elif isinstance(arg1, BoolValue):
                return IntValue(int(math.pow(arg0.value, arg1.value)))
            elif isinstance(arg1, FloatValue):
                return FloatValue(math.pow(float(arg0.value), arg1.value))
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, IntValue):
                return FloatValue(math.pow(arg0.value, float(arg1.value)))
            elif isinstance(arg1, BoolValue):
                return FloatValue(math.pow(arg0.value, float(arg1.value)))
            elif isinstance(arg1, FloatValue):
                return FloatValue(math.pow(arg0.value, arg1.value))
        elif isinstance(arg0, BoolValue):
            if isinstance(arg1, IntValue):
                return IntValue(int(math.pow(arg0.value, arg1.value)))
            elif isinstance(arg1, BoolValue):
                return IntValue(int(math.pow(arg0.value, arg1.value)))
            elif isinstance(arg1, FloatValue):
                return FloatValue(math.pow(float(arg0.value), arg1.value))
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"tan":
        if len(args) != 1:
            raise_type_error(u"math." + op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.tan(arg.value*math.pi/180))
        elif isinstance(arg, BoolValue):
            return FloatValue(math.tan(arg.value*math.pi/180))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.tan(arg.value*math.pi/180))
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"sin":
        if len(args) != 1:
            raise_type_error(u"math." + op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.sin(arg.value*math.pi/180))
        elif isinstance(arg, BoolValue):
            return FloatValue(math.sin(arg.value*math.pi/180))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.sin(arg.value*math.pi/180))
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"cos":
        if len(args) != 1:
            raise_type_error(u"math." + op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.cos(arg.value*math.pi/180))
        elif isinstance(arg, BoolValue):
            return FloatValue(math.cos(arg.value*math.pi/180))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.cos(arg.value*math.pi/180))
        raise_type_error(u"math." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"str":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        return StrValue(inner_repr(args[0]))

    elif op == u"int":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return arg
        elif isinstance(arg, FloatValue):
            return IntValue(int(arg.value))
        elif isinstance(arg, BoolValue):
            return IntValue(int(arg.value))
        elif isinstance(arg, StrValue):
            try:
                return IntValue(int(arg.value))
            except ValueError:
                raise_value_error(u"cannot convert " + arg.value + u" into an int")
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"float":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(float(arg.value))
        elif isinstance(arg, BoolValue):
            return FloatValue(float(arg.value))
        elif isinstance(arg, FloatValue):
            if arg is EPSILON:
                return FloatValue(arg.value)
            else:
                return arg
        elif isinstance(arg, StrValue):
            return FloatValue(str_to_float(arg.value))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"or":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        return arg0 if force_bool(arg0) else arg1

    elif op == u"not":
        if len(args) != 1:
            raise_type_error(op_print + u" expects 1 argument")
        return to_bool_value(not force_bool(args[0]))

    elif op == u"is":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        return to_bool_value(arg0 is arg1)

    elif op == u"len":
        if len(args) != 1:
            raise_type_error(u"[list|str]." + op_print + u" expects 1 argument")
        arg0 = args[0]
        if isinstance(arg0, ListValue):
            return IntValue(len(arg0.array))
        elif isinstance(arg0, StrValue):
            return IntValue(len(arg0.value))
        raise_type_error(u"[list|str]." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"index":
        if len(args) != 2:
            raise_type_error(u"[list|str]." + op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, ListValue):
            for i in range(len(arg0.array)):
                val = builtin_pure(u"==", [arg0.array[i],arg1], u"==")
                assert isinstance(val, BoolValue), "InternalError"
                if val.value:
                    return IntValue(i)
            return IntValue(-1)
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, StrValue):
                raise_type_error(u"[list|str]." + op_print + u" not supported between " + get_type(arg0) + u" and " + get_type(arg1))
            val = arg1.value
            len_val = len(val)
            for i in range(len(arg0.value)-len_val+1):
                if arg0.value[i:i+len_val] == val:
                    return IntValue(i)
            return IntValue(-1)
        raise_type_error(u"[list|str]." + op_print + u" not supported on " + get_type(args[0]))

    elif op == u"$simple_idx":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        idx = 0
        if isinstance(arg1, IntValue):
            idx = arg1.value
        elif isinstance(arg1, BoolValue):
            idx = arg1.value
        else:
            raise_type_error(op_print + u" expects an int for the index")
        if isinstance(arg0, StrValue):
            if idx < 0:
                idx += len(arg0.value)
            if not (0 <= idx < len(arg0.value)):
                raise_index_error(u"$idx=" + int_to_str(idx))
            return StrValue(arg0.value[idx])
        elif isinstance(arg0, ListValue):
            if idx < 0:
                idx += len(arg0.array)
            if not (0 <= idx < len(arg0.array)):
                raise_index_error(u"$idx=" + int_to_str(idx))
            return arg0.array[idx]
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"$simple_idx=":
        if len(args) != 3:
            raise_type_error(op_print + u" expects 3 arguments (list, idx, value)")
        arg0, arg1, arg2 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op_print + u" doesn't support " + get_type(arg0) + u" for the 1st arg")
        idx = 0
        if isinstance(arg1, IntValue):
            idx = arg1.value
        elif isinstance(arg1, BoolValue):
            idx = arg1.value
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the index")
        length = len(arg0.array)
        if idx < 0:
            idx += length
        if not (0 <= idx < length):
            raise_index_error(u"when setting idx=" + int_to_str(idx))
        arg0.array[idx] = arg2
        return NONE

    else:
        if op == op_print:
            raise_unreachable_error(u"builtin-pure " + op + u" not implemented (needed for " + op_print + u")")
        else:
            raise_unreachable_error(u"builtin-pure " + op + u" not implemented")

    if len(args) == 1:
        arg = args[0]
        raise_type_error(u"can't use unary " + op_print + u" on " + get_type(arg))
    elif len(args) == 2:
        arg0, arg1 = args
        raise_type_error(u"can't " + op_print + u" " + get_type(arg0) + u" with " + get_type(arg1))
    else:
        raise_unreachable_error(u"builtin-pure " + op + u" doesn't handle type error explicitly even though it has > 2 arguments")

@look_inside
def _get_full_idx_args(op_print, length, arg1, arg2, arg3):
    start = stop = step = 0 # To make rpython's flowspace happy
    if arg3 is NONE: # step
        step = 1
    elif isinstance(arg3, IntValue):
        step = arg3.value
    elif isinstance(arg3, BoolValue):
        step = int(arg3.value)
    else:
        raise_type_error(op_print + u" doesn't support " + get_type(arg3) + u" for the step arg")
    if step == 0:
        raise_type_error(op_print + u" doesn't support 0 for the step arg")
    if arg1 is NONE: # start
        start = 0 if step > 0 else length
    elif isinstance(arg1, IntValue):
        start = arg1.value
        if start < 0: start += length
    elif isinstance(arg1, BoolValue):
        start = int(arg1.value)
        if start < 0: start += length
    else:
        raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the start arg")
    if arg2 is NONE: # stop
        stop = -1 if step < 0 else length
    elif isinstance(arg2, IntValue):
        stop = arg2.value
        if stop < 0: stop += length
    elif isinstance(arg2, BoolValue):
        stop = int(arg2.value)
        if stop < 0: stop += length
    else:
        raise_type_error(op_print + u" doesn't support " + get_type(arg2) + u" for the stop arg")
    return start, stop, step

@look_inside
def inner_repr(obj, repr=False):
    if obj is None:
        return u"undefined"
    if obj is NONE:
        return u"none"
    elif isinstance(obj, IntValue):
        return int_to_str(obj.value)
    elif isinstance(obj, BoolValue):
        return u"true" if obj.value else u"false"
    elif isinstance(obj, FloatValue):
        return _round_float_str(obj.value, 8)
    elif isinstance(obj, StrValue):
        if repr:
            return str_repr(obj.value)
        else:
            return obj.value
    elif isinstance(obj, FuncValue):
        return u"Func<" + obj.func_info.name + u">"
    elif isinstance(obj, BoundFuncValue):
        string = u"Func<"
        bound_obj = obj.bound_obj
        if isinstance(bound_obj, ObjectValue):
            string += bound_obj.obj_info.name + u"."
        else:
            string += get_type(bound_obj) + u"<object>."
        string += obj.func.func_info.name + u">"
        return string
    elif isinstance(obj, ListValue):
        string = u"["
        for i, element in enumerate(obj.array):
            string += inner_repr(element, repr=True)
            if i != len(obj.array)-1:
                string += u", "
        return string + u"]"
    elif isinstance(obj, ObjectValue):
        if obj.is_obj:
            return u"<Object " + obj.obj_info.name + u">"
        else:
            return u"<Class " + obj.obj_info.name + u">"
    elif isinstance(obj, SpecialValue):
        if obj.type == u"module":
            return u"<Module " + obj.str_value + u">"
        elif obj.type == u"file":
            return u"<File fd=" + int_to_str(obj.int_value) + u">"
        else:
            return u"<Unknown special value>"
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
        print(string)
        return NONE

    elif op == u"append":
        if len(args) != 2:
            raise_type_error(op + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op + u" expects list for the 1st arg")
        arg0.array.append(arg1)
        return NONE

    elif op == u"open":
        if len(args) != 2:
            raise_type_error(op + u" expects 2 arguments")
        path, mode = args
        if not isinstance(path, StrValue):
            raise_type_error(op + u" expects str for the 1st arg")
        if not isinstance(mode, StrValue):
            raise_type_error(op + u" expects str for the 2nd arg")
        r = w = a = b = False
        for char in mode.value:
            if char == u"r":
                if r:
                    raise_value_error(u"invalid file open mode")
                r = True
            elif char == u"w":
                if w:
                    raise_value_error(u"invalid file open mode")
                w = True
            elif char == u"b":
                if b:
                    raise_value_error(u"invalid file open mode")
                b = True
            elif char == u"a":
                if a:
                    raise_value_error(u"invalid file open mode")
                a = True
            else:
                raise_value_error(u"invalid file open mode")
        if (r and a) or (w and a) or ((not r) and (not w) and (not a)):
            raise_value_error(u"invalid file open mode")
        int_flags = 0
        if r and w:
            int_flags |= os.O_RDWR | os.O_CREAT | os.O_TRUNC
        elif r:
            int_flags |= os.O_RDONLY
        elif w:
            int_flags |= os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        elif a:
            int_flags |= os.O_APPEND
        fd = os.open(bytes2(path.value), int_flags, 0o777) | (b<<10)
        return SpecialValue(u"file", int_value=fd)

    elif op == u"write":
        if len(args) != 2:
            raise_type_error(op + u" expects 2 arguments")
        file, data = args
        if not isinstance(file, SpecialValue):
            raise_type_error(op + u" expects a file for the 1st arg")
        fd = file.int_value
        if fd == -1:
            raise_runtime_error(u"file already closed")
        binary = False
        if fd & (1<<10):
            binary = True
            fd ^= 1<<10 # turn off the 1<<10 bit
        if binary:
            raise_type_error(u"bytes objects have not been implemented")
        else:
            if not isinstance(data, StrValue):
                raise_type_error(op + u" expects a str, got " + get_type(data) + u" instead")
            os.write(fd, bytes(data.value))
        return NONE

    elif op == u"read":
        if len(args) != 2:
            raise_type_error(op + u" expects 2 arguments")
        file, length = args
        if not isinstance(file, SpecialValue):
            raise_type_error(op + u" expects a file for the 1st arg")
        if not isinstance(length, IntValue):
            raise_type_error(op + u" expects an int, got " + get_type(length) + u" instead")
        fd = file.int_value
        if fd == -1:
            raise_runtime_error(u"file already closed")
        binary = False
        if fd & (1<<10):
            binary = True
            fd ^= 1<<10 # turn off the 1<<10 bit
        data = os.read(fd, length.value)
        ret = NONE # to make rpython's flowspace happy
        if binary:
            raise_type_error(u"bytes objects have not been implemented")
        else:
            ret = StrValue(str(data))
        return ret

    elif op == u"close":
        if len(args) != 1:
            raise_type_error(op + u" expects only 1 argument")
        file = args[0]
        if not isinstance(file, SpecialValue):
            raise_type_error(op + u" expects a file for the 1st arg")
        fd = file.int_value
        os.close((fd^(1<<10)) if fd & (1<<10) else fd)
        file.int_value = -1 # mark it as closed
        return NONE

    elif op == u"read":
        if len(args) != 1:
            raise_type_error(op + u" expects only 1 argument")
        file, length = args
        if not isinstance(file, SpecialValue):
            raise_type_error(op + u" expects a file for the 1st arg")
        if not isinstance(length, IntValue):
            raise_type_error(op + u" expects an int for the 2nd arg")
        fd = file.int_value
        os.close((fd^(1<<10)) if fd & (1<<10) else fd)
        return NONE

    else:
        raise_unreachable_error(u"builtin-side " + op + u" not implemented (needed for " + op + u")")

    raise_unreachable_error(u"builtin-side invalid type error raising")


# Error helpers (note they all print traceback)
def raise_name_error(name):
    assert isinstance(name, str), "TypeError"
    raise_error(u"NameError: " + name + u" has not been defined")

def raise_index_error(msg):
    assert isinstance(msg, str), "TypeError"
    raise_error(u"IndexError: " + msg)

def raise_type_error(msg):
    assert isinstance(msg, str), "TypeError"
    raise_error(u"TypeError: " + msg)

def raise_unreachable_error(msg):
    assert isinstance(msg, str), "TypeError"
    raise_error(u"Unreachable reached: " + msg)

def raise_value_error(msg):
    assert isinstance(msg, str), "TypeError"
    raise_error(u"ValueError: " + msg)

def raise_runtime_error(msg):
    assert isinstance(msg, str), "TypeError"
    raise_error(u"RuntimeError: " + msg)

def raise_error(msg):
    assert isinstance(msg, str), "TypeError"
    raise InterpreterError(msg)


# Main code that reads and deserialises the bytecode
def _main(raw_bytecode, cmd_args, filepath="*unknown*"):
    assert isinstance(raw_bytecode, bytes), "TypeError"
    debug(u"Derialising bytecode...", 2)
    try:
        flags, frame_size, env_size, attrs, bytecode, SOURCE_CODE = derialise(raw_bytecode)
    except UnicodeDecodeError:
        print_err(u"\x1b[91m[DecodeError]: Invalid bytecode file \x1b[92m" + str(filepath) + u"\x1b[0m")
        return 2
    debug(u"Parsing flags...", 2)
    debug(u"Starting interpreter...", 1)
    return interpret(flags, frame_size, env_size, attrs, bytecode, SOURCE_CODE, cmd_args)

def main(filepath, cmd_args=[]):
    debug(u"Reading file...", 2)
    with open(filepath, "rb") as file:
        if file.read(11) != b".clizz file":
            print_err(u"\x1b[91m[DecodeError]: Invalid bytecode file \x1b[92m" + str(filepath) + u"\x1b[0m")
            return 2
        data = file.read()
    return _main(data, cmd_args, filepath)


# For testing (note runs with PYTHON == 3)
if __name__ == "__main__":
    if PYTHON == 3:
        from time import perf_counter
        start = perf_counter()
    import sys
    if len(sys.argv) <= 1:
        exit_code = main("../code-examples/example.clizz")
    else:
        exit_code = main(sys.argv[1], sys.argv[2:])
    if PYTHON == 3:
        print("Time taken: {:.3f}".format(perf_counter()-start))


# https://github.com/aheui/rpaheui/blob/main/LOG.md
# https://pypy.org/posts/2011/04/tutorial-part-2-adding-jit-8121732841568309472.html
# https://pypy.org/posts/2011/03/controlling-tracing-of-interpreter-with_15-3281215865169782921.html
# https://web.archive.org/web/20170929153251/https://bitbucket.org/brownan/pypy-tutorial/src/tip/example4.py
# https://doi.org/10.1016/j.entcs.2016.12.012
# https://eprints.gla.ac.uk/113615/
# https://www.hpi.uni-potsdam.de/hirschfeld/publications/media/Pape_2021_EfficientCompoundValuesInVirtualMachines_Dissertation.pdf
# /media/thelizzard/TheLizzardOS-SD/rootfs/home/thelizzard/honours/lizzzard/src/frontend/rpython/rlib/jit.py
# https://github.com/pypy/pypy/issues/5166
# https://github.com/pypy/pypy/issues/5181
# https://github.com/pypy/pypy/issues/5184
# https://rpython.readthedocs.io/en/latest/jit/optimizer.html
# /media/thelizzard/TheLizzardOS-SD/rootfs/home/thelizzard/honours/lizzzard/src/frontend/rpython/jit/metainterp/optimizeopt
# https://github.com/hanabi1224/Programming-Language-Benchmarks/tree/main
# https://www.csl.cornell.edu/~cbatten/pdfs/cheng-type-freezing-slides-cgo2020.pdf
# https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/253137/618488_FULLTEXT01.pdf
# https://en.wikipedia.org/wiki/Comparison_of_functional_programming_languages