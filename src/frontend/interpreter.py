import math
import os

from python3.rpython_compat import *
from python3.dict_compat import *
from python3.int_compat import *
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
    def __init__(self, value):
        assert isinstance(value, int), "TypeError"
        self.value = value

BoolValue = IntValue # TODO: implement BoolValue

class StrValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    def __init__(self, value):
        assert isinstance(value, str), "TypeError"
        self.value = value

class FloatValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    def __init__(self, value):
        assert isinstance(value, float), "TypeError"
        self.value = value

class FuncValue(Value):
    _immutable_fields_ = ["tp", "masters", "env_size", "nargs", "name", "bound_obj"]
    __slots__ = "tp", "masters", "env_size", "nargs", "name", "bound_obj"
    def __init__(self, tp, masters, env_size, nargs, name, bound_obj):
        if bound_obj is not None:
            assert isinstance(bound_obj, Value), "TypeError"
        assert isinstance(masters, list), "TypeError"
        for master in masters:
            assert isinstance(master, ENV_TYPE), "TypeError"
        assert isinstance(env_size, int), "TypeError"
        assert isinstance(nargs, int), "TypeError"
        assert isinstance(name, str), "TypeError"
        assert isinstance(tp, int), "TypeError"
        self.bound_obj = bound_obj
        self.env_size = env_size
        self.masters = masters
        self.nargs = nargs
        self.name = name
        self.tp = tp

class SpecialValue(Value):
    _immutable_fields_ = ["type", "str_value", "int_value"]
    __slots__ = "type", "str_value", "int_value"
    def __init__(self, type, str_value=u"", int_value=0):
        assert isinstance(str_value, str), "TypeError"
        assert isinstance(int_value, int), "TypeError"
        assert isinstance(type, str), "TypeError"
        self.str_value = str_value
        self.int_value = int_value
        self.type = type

class LinkValue(Value):
    _immutable_fields_ = ["link"]
    __slots__ = "link"
    def __init__(self, link):
        assert isinstance(link, int), "TypeError"
        assert link > 0, "ValueError"
        self.link = link

class NoneValue(Value):
    _immutable_fields_ = []
    __slots__ = ()
    def __init__(self): pass

class ObjectValue(Value):
    _immutable_fields_ = ["attr_vals", "type", "name"]
    __slots__ = "attr_vals", "type", "name"
    @look_inside
    def __init__(self, type, name):
        assert isinstance(type, int), "TypeError" # even types for classes and odds for objects of that type
        assert isinstance(name, str), "TypeError"
        if ENV_IS_LIST:
            self.attr_vals = [None]*len(SPECIAL_ATTRS)
        else:
            self.attr_vals = Dict()
            for name in SPECIAL_ATTRS:
                self.attr_vals[name] = None
        self.type = type
        self.name = name


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
        assert isinstance(value, Value), "TypeError"
        assert isinstance(idx, int), "TypeError"
        self.array[idx] = value

    @look_inside
    def len(self):
        return len(self.array)

    def __repr__(self):
        return u"array[size=" + int_to_str(self.len()) + u"]"


epsilon = 6e-8
NONE = const(NoneValue())
ZERO = const(IntValue(0))
ONE = const(IntValue(1))
FALSE = const(BoolValue(0))
TRUE = const(BoolValue(1))
PI = const(FloatValue(math.pi))
EPSILON = const(FloatValue(epsilon))


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
    if isinstance(value, LinkValue):
        return u"link"
    if isinstance(value, ListValue):
        return u"list"
    if isinstance(value, FuncValue):
        return u"func"
    if isinstance(value, ObjectValue):
        return u"class"
    if isinstance(value, FloatValue):
        return u"float"
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
    if isinstance(val, FloatValue):
        return not (-epsilon < val.value < epsilon)
    return True

@look_inside
@elidable
def to_bool_value(boolean):
    assert isinstance(boolean, bool), "TypeError"
    return TRUE if boolean else FALSE

@look_inside
def regs_load(regs, reg):
    if reg == 0:
        return ZERO
    elif reg == 1:
        return ONE
    else:
        value = regs[reg]
        if value is None:
            raise_error(u"InternalError: trying to load undefined from regs")
        return value

@look_inside
def regs_store(regs, reg, value):
    if value is None:
        raise_error(u"InternalError: trying to store undefined inside regs")
    if reg < 1:
        return # Don't store in regs 0 or 1
    assert reg < const(len(regs)), "InternalError"
    regs[reg] = value

@look_inside
def env_load(env, idx):
    assert isinstance(env, ENV_TYPE), "TypeError"
    if ENV_IS_LIST:
        assert idx < const(len(env)), "InternalError"
    value = env[idx]
    if value is None:
        raise_error(u"InternalError: trying to load undefined from env")
    return value

@look_inside
def env_store(env, idx, value):
    assert isinstance(env, ENV_TYPE), "TypeError"
    if value is None:
        raise_error(u"InternalError: trying to store undefined inside env")
    if ENV_IS_LIST:
        assert idx < const(len(env)), "InternalError"
    env[idx] = value

@look_inside
def attr_vals_load(attr_vals, idx):
    assert isinstance(attr_vals, list), "TypeError"
    assert idx < len(attr_vals), "InternalError"
    value = attr_vals[idx]
    if value is None:
        raise_error(u"InternalError: trying to load undefined from attrs")
    return value

@look_inside
def attr_vals_store(attr_vals, idx, value):
    if value is None:
        raise_error(u"InternalError: trying to store undefined inside attrs")
    assert idx < len(attr_vals), "InternalError"
    attr_vals[idx] = value

@look_inside
def attr_vals_extend_until_len(attr_vals, new_length):
    assert isinstance(attr_vals, list), "TypeError"
    while len(attr_vals) < new_length:
        attr_vals.append(None)

@unroll_safe
@look_inside
def attr_access(mros, attr_matrix, lens, obj, attr, storing):
    # Use the mro to figure out where attr is supposed to be
    mro = hint(mros[const(obj.type)], promote=True)
    if obj.type&1:
        mro = [obj] + mro
    for cls in mro:
        if (cls is not obj) and storing:
            continue
        assert 0 <= cls.type < len(attr_matrix), "InternalError"
        row = attr_matrix[cls.type]
        assert 0 <= attr < len(row), "InternalError"
        soft, attr_idx = hint(row[attr], promote=True)
        soft, attr_idx = const(soft), const(attr_idx)
        if attr_idx != -1:
            if soft and (cls.type&1):
                if (attr_idx >= len(cls.attr_vals)) or (cls.attr_vals[attr_idx] is None):
                    continue
            break
    # Create the attr space in cls if attr_idx is -1
    else:
        cls = obj
        attr_idx, lens[cls.type] = lens[cls.type], lens[cls.type]+1
        assert 0 <= attr_idx, "InternalError"
        assert 0 <= cls.type < len(attr_matrix), "InternalError"
        row = attr_matrix[cls.type]
        assert 0 <= attr < len(row), "InternalError"
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
def copy_and_bind_func(func, obj):
    assert isinstance(func, FuncValue), "TypeError"
    assert isinstance(obj, Value), "TypeError"
    return FuncValue(func.tp, func.masters, func.env_size, func.nargs, func.name, obj)

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
STACK_IS_LIST = False
ENV_IS_VIRTUALISABLE = False

if ENV_IS_LIST:
    if ENV_IS_VIRTUALISABLE:
        raise NotImplementedError("Very unsafe - `fib` breaks if VirtualisableArray is used. More info in version 1.3.5 commit message")
        ENV_TYPE = VirtualisableArray
    else:
        ENV_TYPE = list
else:
    assert not ENV_IS_VIRTUALISABLE, "Invalid compile settings"
    ENV_TYPE = Dict

if USE_JIT:
    def get_location(_, pc, bytecode, *__):
        return bytecode_debug_str(pc, bytecode[pc])
        return "Instruction[%s]" % bytes2(int_to_str(pc,zfill=2))
    virtualizables = ["env"] if ENV_IS_VIRTUALISABLE else []
    jitdriver = JitDriver(greens=["CLEAR_AFTER_USE", "pc","bytecode","teleports","SOURCE_CODE"],
                          reds=["next_cls_type","stack","env","func","regs","attr_matrix","attrs","lens","mros","global_scope"],
                          virtualizables=virtualizables, get_printable_location=get_location)


# Stack linked list
class StackFrame:
    _immutable_fields_ = ["env", "regs", "pc", "func", "ret_reg", "prev_stack"]
    __slots__ = "env", "regs", "pc", "func", "ret_reg", "prev_stack"

    # @look_inside
    def __init__(self, env, func, regs, pc, ret_reg, prev_stack):
        assert isinstance(func, FuncValue), "TypeError"
        assert isinstance(env, ENV_TYPE), "TypeError"
        assert isinstance(ret_reg, int), "TypeError"
        assert isinstance(regs, list), "TypeError"
        assert isinstance(pc, int), "TypeError"
        if prev_stack is not None:
            assert isinstance(prev_stack, StackFrame), "TypeError"
        self.env = env
        self.regs = regs
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

@look_inside
@unroll_safe
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
def interpret(flags, frame_size, env_size, attrs, bytecode, SOURCE_CODE):
    global ENV_IS_LIST
    if ENV_IS_LIST != flags.is_set("ENV_IS_LIST"):
        if PYTHON == 3:
            global ENV_TYPE
            ENV_IS_LIST = flags.is_set("ENV_IS_LIST")
            if ENV_IS_LIST:
                if ENV_IS_VIRTUALISABLE:
                    ENV_TYPE = VirtualisableArray
                else:
                    ENV_TYPE = list
            else:
                ENV_TYPE = Dict
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
            teleports[const_str(bt.id)] = IntValue(i)
    for i, op in enumerate(BUILTINS):
        i += len(bytecode)
        teleports[const_str(int_to_str(i))] = IntValue(i)
    # Create regs
    regs = [None]*(frame_size+2)
    # Create env
    if ENV_IS_LIST:
        if ENV_IS_VIRTUALISABLE:
            env = VirtualisableArray(env_size)
        else:
            env = [None]*env_size
        envs = hint([env], promote=True)
        for i, op in enumerate(BUILTINS):
            assert isinstance(op, str), "TypeError"
            pure_op = op not in BUILTIN_MODULE_SIDES
            env[i] = FuncValue(i+len(bytecode), envs, pure_op, 0, op, None)
        for i, op in enumerate(BUILTIN_MODULES):
            assert isinstance(op, str), "TypeError"
            env[i+len(MODULE_ATTRS)] = SpecialValue(u"module", str_value=op)
    else:
        raise NotImplementedError("TODO")
    # Start actual interpreter
    return _interpret(bytecode, teleports, regs, env, attrs, flags, SOURCE_CODE)

def _interpret(bytecode, teleports, regs, env, attrs, flags, SOURCE_CODE):
    CLEAR_AFTER_USE = const(flags.is_set("CLEAR_AFTER_USE"))
    SOURCE_CODE = const_str(SOURCE_CODE)
    pc = 0 # current instruction being executed
    stack = [] if STACK_IS_LIST else None
    next_cls_type = 0
    attr_matrix = hint([], promote=True)
    lens = hint([], promote=True)
    mros = hint([], promote=True)
    func = FuncValue(0, [], 0, 0, u"main-scope", None)
    global_scope = hint(env, promote=True)

    while pc < len(bytecode):
        if USE_JIT:
            jitdriver.jit_merge_point(stack=stack, env=env, func=func, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE,
                                      attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, mros=mros, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)
        bt = bytecode[pc]
        if DEBUG_LEVEL >= 3:
            debug(str(bytecode_debug_str(pc, bt)), 3)
        pc += 1

        try:
            if isinstance(bt, Bable):
                pass

            elif isinstance(bt, BLoadLink):
                assert not ENV_IS_LIST, "Invalid flag/bytecode"
                raise NotImplementedError("TODO")

            elif isinstance(bt, BStoreLoadDict):
                assert not ENV_IS_LIST, "Invalid flag/bytecode"
                raise NotImplementedError("TODO")

            elif isinstance(bt, BDotDict):
                assert not ENV_IS_LIST, "Invalid flag/bytecode"
                raise NotImplementedError("TODO")

            elif isinstance(bt, BStoreLoadList):
                assert ENV_IS_LIST, "Invalid flag/bytecode"
                # Get the correct scope
                if bt.link == 0:
                    scope = env
                else:
                    scope = func.masters[len(func.masters)-bt.link]
                # Store/Load variable
                if bt.storing:
                    env_store(scope, bt.name, regs_load(regs, bt.reg))
                else:
                    regs_store(regs, bt.reg, env_load(scope, bt.name))

            elif isinstance(bt, BDotList):
                assert ENV_IS_LIST, "Invalid flag/bytecode"
                obj = regs_load(regs, bt.obj_reg)
                if isinstance(obj, ListValue):
                    if bt.storing:
                        raise_name_error(u"cannot change builtin attribute")
                    if bt.attr not in (LEN_IDX, APPEND_IDX):
                        raise_name_error(u"Unknown attribute")
                    regs_store(regs, bt.reg, copy_and_bind_func(global_scope[bt.attr], obj))
                elif isinstance(obj, StrValue):
                    if bt.storing:
                        raise_name_error(u"cannot change builtin attribute")
                    if bt.attr not in (LEN_IDX,):
                        raise_name_error(u"Unknown attribute")
                    regs_store(regs, bt.reg, copy_and_bind_func(global_scope[bt.attr], obj))
                elif isinstance(obj, SpecialValue):
                    if obj.type == u"module":
                        value = None
                        if obj.str_value == u"io":
                            if bt.attr not in (PRINT_IDX, OPEN_IDX):
                                raise_name_error(u"this")
                            value = global_scope[bt.attr]
                        elif obj.str_value == u"math":
                            if bt.attr == PI_IDX:
                                value = PI
                            elif bt.attr == EPSILON_IDX:
                                value = EPSILON
                            else:
                                if bt.attr not in (SQRT_IDX, SIN_IDX, COS_IDX, TAN_IDX, POW_IDX):
                                    raise_name_error(u"this")
                                value = global_scope[bt.attr]
                        else:
                            raise_name_error(u"this")
                        regs_store(regs, bt.reg, value)
                    elif obj.type == u"file":
                        if bt.attr not in (READ_IDX, WRITE_IDX, CLOSE_IDX):
                            raise_name_error(u"this")
                        regs_store(regs, bt.reg, copy_and_bind_func(global_scope[bt.attr], obj))
                    else:
                        raise_unreachable_error(u"TODO . operator on SpecialValue with obj.type=" + obj.type)
                elif isinstance(obj, ObjectValue):
                    # Get cls (the object storing attr) and attr_idx (the idx into cls.attr_vals)
                    cls, attr_idx = attr_access(mros, attr_matrix, lens, obj, bt.attr, bt.storing)
                    # Use the information above to execute the bytecode
                    attr_idx = const(attr_idx)
                    assert 0 <= attr_idx < len(cls.attr_vals), "InternalError"
                    if bt.storing:
                        attr_vals_store(cls.attr_vals, attr_idx, regs_load(regs, bt.reg))
                    else:
                        value = attr_vals_load(cls.attr_vals, attr_idx)
                        if isinstance(value, FuncValue) and (obj.type&1) and (value.bound_obj is None):
                            value = copy_and_bind_func(value, obj)
                            _, attr_idx = attr_matrix[obj.type][bt.attr]
                            if attr_idx == -1:
                                attr_idx, lens[obj.type] = lens[obj.type], lens[obj.type]+1
                                attr_matrix[obj.type][bt.attr] = (True, attr_idx)
                            attr_vals_extend_until_len(obj.attr_vals, attr_idx+1)
                            attr_vals_store(obj.attr_vals, attr_idx, value)
                        regs_store(regs, bt.reg, value)
                else:
                    raise_type_error(u". operator expects object got " + get_type(obj) + u" instead")

            elif isinstance(bt, BLiteral):
                bt_literal = bt.literal
                if bt.type == BLiteral.INT_T:
                    assert isinstance(bt_literal, BLiteralInt), "TypeError"
                    literal = IntValue(bt_literal.value)
                elif bt.type == BLiteral.FUNC_T:
                    assert isinstance(bt_literal, BLiteralFunc), "TypeError"
                    envs_copy = hint(func.masters+[env], promote=True)
                    literal = FuncValue(bt_literal.value, envs_copy, bt_literal.env_size, bt_literal.nargs, bt_literal.name, None)
                elif bt.type == BLiteral.STR_T:
                    assert isinstance(bt_literal, BLiteralStr), "TypeError"
                    literal = StrValue(bt_literal.value)
                elif bt.type == BLiteral.FLOAT_T:
                    assert isinstance(bt_literal, BLiteralFloat), "TypeError"
                    literal = FloatValue(bt_literal.value)
                elif bt.type == BLiteral.NONE_T:
                    literal = NONE
                elif bt.type == BLiteral.UNDEFINED_T:
                    regs[bt.reg] = None
                    continue
                elif bt.type == BLiteral.LIST_T:
                    literal = ListValue()
                elif bt.type == BLiteral.CLASS_T:
                    assert isinstance(bt_literal, BLiteralClass), "TypeError"
                    # Create ObjectValue type
                    _mros = []
                    for base_reg in bt_literal.bases:
                        base = regs_load(regs, base_reg)
                        if not isinstance(base, ObjectValue):
                            raise_type_error(u"can't inherit from " + get_type(base))
                        _mros.append(mros[base.type])
                    # Register new class
                    cls_type, next_cls_type = next_cls_type, next_cls_type+2 # even types for classes and odds for objects of that type
                    for _ in range(2):
                        row = [(False, -1) for _ in range(len(attrs))]
                        row = hint(row, promote=True)
                        row[0] = hint((False, 0), promote=True)
                        attr_matrix.append(row)
                    lens.extend([len(SPECIAL_ATTRS),len(SPECIAL_ATTRS)])
                    literal = ObjectValue(cls_type, bt_literal.name)
                    while len(mros) <= literal.type: mros.append([])
                    mros[literal.type] = hint([literal]+_c3_merge(_mros), promote=True)
                    regs_store(regs, bt.reg, literal)
                    # Append to stack
                    if STACK_IS_LIST:
                        stack.append((env, func, regs, pc, bt.reg))
                    else:
                        stack = StackFrame(env, func, regs, pc, bt.reg, stack)
                    tp = teleports_get(teleports, bt_literal.label)
                    assert isinstance(tp, IntValue), "TypeError"
                    pc, regs = tp.value, list(regs)
                    regs[CLS_REG] = literal
                    continue
                else:
                    raise NotImplementedError()
                regs_store(regs, bt.reg, literal)

            elif isinstance(bt, BJump):
                value = regs_load(regs, bt.condition_reg)
                if CLEAR_AFTER_USE and (bt.condition_reg > 1):
                    regs[bt.condition_reg] = None
                condition = force_bool(value)
                # if condition != bt.negated: # RPython's JIT can't constant fold in this form :/
                if (condition and (not bt.negated)) or ((not condition) and bt.negated):
                    tp = teleports_get(teleports, bt.label)
                    assert isinstance(tp, IntValue), "TypeError"
                    old_pc, pc = pc, tp.value
                    assert isinstance(bytecode[pc], Bable), "InternalError"
                    if USE_JIT and (pc < old_pc):
                        jitdriver.can_enter_jit(stack=stack, env=env, func=func, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE,
                                                attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, mros=mros, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)

            elif isinstance(bt, BRegMove):
                regs_store(regs, bt.reg1, regs_load(regs, bt.reg2))

            elif isinstance(bt, BRet):
                old_pc = pc
                if bt.capture_env:
                    if not stack:
                        raise NotImplementedError("Impossible")
                    if STACK_IS_LIST:
                        _, _, regs, pc, ret_reg = stack.pop()
                    else:
                        regs, pc, stack = stack.regs, stack.pc, stack.prev_stack
                else:
                    value = regs_load(regs, bt.reg)
                    if not stack:
                        if not isinstance(value, IntValue):
                            raise_type_error(u"exit value should be an int not " + get_type(value))
                        print(u"[EXIT]: " + int_to_str(value.value))
                        break
                    if STACK_IS_LIST:
                        env, func, regs, pc, ret_reg = stack.pop()
                        regs_store(regs, ret_reg, value)
                    else:
                        env, func, regs, pc = stack.env, stack.func, stack.regs, stack.pc
                        regs_store(regs, stack.ret_reg, value)
                        stack = stack.prev_stack
                # Tell the JIT compiler about the jump
                if USE_JIT and ENTER_JIT_FUNC_RET:
                    jitdriver.can_enter_jit(stack=stack, env=env, func=func, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE,
                                            attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, mros=mros, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)
                # env, func, regs, stack = hint(env, promote=True), hint(func, promote=True), hint(regs, promote=True), hint(stack, promote=True)

            elif isinstance(bt, BCall):
                args = []
                _func = const(regs_load(regs, bt.regs[1]))
                if isinstance(_func, ObjectValue) and (_func.type&1 == 0):
                    self = ObjectValue(_func.type+1, _func.name)
                    while len(mros) <= self.type: mros.append([])
                    mros[self.type] = hint(mros[_func.type], promote=True)
                    args = [self]
                    _func = _func.attr_vals[CONSTRUCTOR_IDX]
                    if _func is None:
                        # Default constructor
                        regs_store(regs, bt.regs[0], self)
                        continue
                    if not isinstance(_func, FuncValue):
                        raise_type_error(u"constructor should be a function not " + get_type(_func))
                elif isinstance(_func, FuncValue):
                    if _func.bound_obj is not None:
                        args.append(_func.bound_obj)
                else:
                    raise_type_error(get_type(_func) + u" is not callable")

                tp = const(teleports[int_to_str(_func.tp)])
                if tp is None:
                    raise_name_error(_func.name)
                assert isinstance(tp, IntValue), "TypeError"
                tp_value = const(tp.value)
                if tp_value < len(bytecode):
                    # Create a new stack frame
                    if STACK_IS_LIST:
                        stack.append((env, func, regs, pc, bt.regs[0]))
                    else:
                        stack = StackFrame(env, func, regs, pc, bt.regs[0], stack)
                    # Set the pc/regs/env/func
                    func, old_pc, pc, old_regs, regs = _func, pc, tp_value, regs, [None]*const(len(regs))
                    if ENV_IS_LIST:
                        if ENV_IS_VIRTUALISABLE:
                            env = VirtualisableArray(func.env_size)
                        else:
                            env = [None]*func.env_size
                    else:
                        raise NotImplementedError("TODO")
                    # Check number of args
                    if len(bt.regs)-2+len(args) > func.nargs:
                        raise_type_error(u"too many arguments")
                    elif len(bt.regs)-2+len(args) < func.nargs:
                        raise_type_error(u"too few arguments")
                    # Copy arguments values into new regs
                    for i in range(len(args)):
                        regs_store(regs, i+2, args[i])
                    for i in range(2, len(bt.regs)):
                        regs_store(regs, i+len(args), regs_load(old_regs, bt.regs[i]))
                    # Tell the JIT compiler about the jump
                    if USE_JIT and ENTER_JIT_FUNC_CALL:
                        jitdriver.can_enter_jit(stack=stack, env=env, func=func, regs=regs, pc=pc, bytecode=bytecode, teleports=teleports, CLEAR_AFTER_USE=CLEAR_AFTER_USE,
                                                attr_matrix=attr_matrix, next_cls_type=next_cls_type, attrs=attrs, lens=lens, mros=mros, SOURCE_CODE=SOURCE_CODE, global_scope=global_scope)
                else: # Built-ins
                    op_idx = tp_value - len(bytecode)
                    pure_op = bool(const(_func.env_size))
                    op = BUILTINS[op_idx]
                    args += [regs_load(regs, bt.regs[i]) for i in range(2,len(bt.regs))]
                    if op_idx == ISINSTANCE_IDX:
                        value = inner_isinstance(mros, global_scope, args)
                    elif pure_op:
                        value = builtin_pure(op, args, op)
                    else:
                        value = builtin_side(op, args)
                    regs_store(regs, bt.regs[0], value)

            else:
                raise NotImplementedError("Haven't implemented this bytecode yet")

        except InterpreterError as error:
            funcs = [(func, pc)]
            if STACK_IS_LIST:
                for _, stack_func, _, stack_pc, _ in stack:
                    funcs.append((stack_func, stack_pc))
            else:
                s = stack
                while s is not None:
                    funcs.append((s.func, s.pc))
                    s = s.prev_stack
            print(u"Traceback (most recent call last):")
            for i in range(len(funcs)):
                stack_func, stack_pc = funcs[len(funcs)-i-1]
                bt = bytecode[stack_pc-1]
                err_idx = get_err_idx_from_bt(bt)
                line, line_str, start, size, success = data_from_err_idx(SOURCE_CODE, err_idx)
                indent = str_count_prefix(line_str, u" ") + str_count_prefix(line_str, u"\t")
                line_str, start = line_str[indent:], start-indent
                assert start >= 0, "InternalError"
                if not success:
                    continue
                print(u"   File \x1b[95m<???>\x1b[0m in \x1b[95m<" + stack_func.name + u">\x1b[0m on line \x1b[95m" + int_to_str(line) + u"\x1b[0m:")
                print(u" "*6 + line_str[:start] + u"\x1b[91m" + line_str[start:start+size] + u"\x1b[0m" + line_str[start+size:])
                print(u" "*(start+6) + u"\x1b[91m" + u"^"*size + u"\x1b[0m")
            print(u"\x1b[93m" + error.msg + u"\x1b[0m")
            return 1
    return 0


@look_inside
def inner_isinstance(mros, global_scope, args):
    if len(args) != 2:
        raise_type_error(u"isinstance takes 2 arguments")
    instance, cls = args
    if isinstance(cls, FuncValue):
        output = False
        if cls is const(global_scope[INT_IDX]):
            output = isinstance(instance, IntValue)
        elif cls is const(global_scope[FLOAT_IDX]):
            output = isinstance(instance, IntValue) or isinstance(instance, FloatValue)
        elif cls is const(global_scope[STR_IDX]):
            output = isinstance(instance, StrValue)
        elif cls is const(global_scope[LIST_IDX]):
            output = isinstance(instance, ListValue)
        elif cls is const(global_scope[BOOL_IDX]):
            output = isinstance(instance, BoolValue)
        else:
            raise_type_error(u"the 2nd argument of isinstance should be a type or a list of types")
        return to_bool_value(output)
    elif isinstance(cls, ObjectValue):
        if not isinstance(instance, ObjectValue):
            return FALSE
        if cls.type == instance.type:
            return TRUE
        for super_cls in hint(mros[const(cls.type)], promote=True):
            if super_cls.type == instance.type:
                return TRUE
            if instance.type&1:
                if super_cls.type == instance.type-1:
                    return TRUE
        return FALSE
    elif isinstance(cls, ListValue):
        for subcls in cls.array:
            if inner_isinstance(mros, global_scope, [instance, subcls]):
                return TRUE
        return FALSE
    else:
        raise_type_error(u"the 2nd argument of isinstance should be a type")


# Builtin helpers
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
        assert isinstance(arg0, ListValue), "InternalError" # builtin*_list should only be called with a list arg
        length = arg0.len()
        start = stop = step = 0 # To make rpython's flowspace happy
        if arg3 is NONE: # step
            step = 1
        elif isinstance(arg3, IntValue):
            step = arg3.value
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg3) + u" for the step arg")
        if step == 0:
            raise_type_error(op_print + u" doesn't support 0 for the step arg")
        if arg1 is NONE: # start
            start = 0 if step > 0 else length
        elif isinstance(arg1, IntValue):
            start = arg1.value
            if start < 0:
                start += length
            if start < 0:
                raise_index_error(u"start-idx = " + int_to_str(start))
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the start arg")
        if arg2 is NONE: # stop
            stop = -1 if step < 0 else length
        elif isinstance(arg2, IntValue):
            stop = arg2.value
            if stop < 0:
                stop += length
            if stop < 0:
                raise_index_error(u"stop-idx = " + int_to_str(stop))
        else:
            raise_type_error(op_print + u" doesn't support " + get_type(arg2) + u" for the stop arg")
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
        if op == op_print:
            raise_unreachable_error(u"builtin-pure-list " + op + u" not implemented (needed for " + op_print + u")")
        else:
            raise_unreachable_error(u"builtin-pure-list " + op + u" not implemented")


@look_inside
def builtin_pure_nonlist(op, args, op_print):
    if op == u"+":
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, IntValue):
                return arg
            elif isinstance(arg, FloatValue):
                return arg
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, IntValue):
                if isinstance(arg1, IntValue):
                    return IntValue(arg0.value + arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value + arg1.value)
            elif isinstance(arg0, StrValue):
                if isinstance(arg1, StrValue):
                    return StrValue(arg0.value + arg1.value)
            elif isinstance(arg0, FloatValue):
                if isinstance(arg1, IntValue):
                    return FloatValue(arg0.value + arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value + arg1.value)
        else:
            raise_type_error(op_print + u" expects 1 or 2 arguments")

    elif op == u"-":
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, IntValue):
                return IntValue(-arg.value)
            elif isinstance(arg, FloatValue):
                return FloatValue(-arg.value)
        elif len(args) == 2:
            arg0, arg1 = args
            if isinstance(arg0, IntValue):
                if isinstance(arg1, IntValue):
                    return IntValue(arg0.value - arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value - arg1.value)
            elif isinstance(arg0, FloatValue):
                if isinstance(arg1, IntValue):
                    return FloatValue(arg0.value - arg1.value)
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value - arg1.value)

    elif op == u"*":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value * arg1.value)
            elif isinstance(arg1, FloatValue):
                return FloatValue(arg0.value * arg1.value)
            elif isinstance(arg1, StrValue):
                return StrValue(arg0.value * arg1.value)
        elif isinstance(arg0, StrValue):
            if isinstance(arg1, IntValue):
                return StrValue(arg0.value * arg1.value)
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, IntValue):
                return FloatValue(arg0.value * arg1.value)
            elif isinstance(arg1, FloatValue):
                return FloatValue(arg0.value * arg1.value)

    elif op == u"%":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                try:
                    return IntValue(arg0.value % arg1.value)
                except ZeroDivisionError:
                    raise_value_error(u"division by zero")

    elif op == u"//":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                try:
                    return IntValue(arg0.value // arg1.value)
                except ZeroDivisionError:
                    raise_value_error(u"division by zero")

    elif op == u"<<":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value << arg1.value)

    elif op == u">>":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value >> arg1.value)

    elif op == u"&":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value & arg1.value)

    elif op == u"|":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(arg0.value | arg1.value)

    elif op == u"==":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                if arg0.value == arg1.value:
                    return TRUE
            elif isinstance(arg1, FloatValue):
                if arg1.value-epsilon < arg0.value < arg1.value+epsilon:
                    return TRUE
            return FALSE
        elif isinstance(arg0, StrValue):
            if not isinstance(arg1, StrValue):
                return FALSE
            if arg0.value == arg1.value:
                return TRUE
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, IntValue):
                if arg0.value-epsilon < arg1.value < arg0.value+epsilon:
                    return TRUE
            elif isinstance(arg1, FloatValue):
                if arg0.value-epsilon < arg1.value < arg0.value+epsilon:
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
        elif isinstance(arg0, StrValue):
            if isinstance(arg1, StrValue):
                return BoolValue(arg0.value < arg1.value)
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value < arg1.value)
            elif isinstance(arg1, IntValue):
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
        elif isinstance(arg0, StrValue):
            if isinstance(arg1, StrValue):
                return BoolValue(arg0.value > arg1.value)
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, FloatValue):
                return BoolValue(arg0.value > arg1.value)
            elif isinstance(arg1, IntValue):
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
                    return FloatValue(float(arg0.value)/float(arg1.value))
                elif isinstance(arg1, FloatValue):
                    return FloatValue(float(arg0.value)/arg1.value)
            elif isinstance(arg0, FloatValue):
                if isinstance(arg1, IntValue):
                    return FloatValue(arg0.value/float(arg1.value))
                elif isinstance(arg1, FloatValue):
                    return FloatValue(arg0.value/arg1.value)
        except ZeroDivisionError:
            raise_value_error(u"division by zero")

    elif op == u"sqrt":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.sqrt(arg.value))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.sqrt(arg.value))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"pow":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, IntValue):
            if isinstance(arg1, IntValue):
                return IntValue(int(math.pow(arg0.value, arg1.value)))
            elif isinstance(arg1, FloatValue):
                return FloatValue(math.pow(float(arg0.value), arg1.value))
        elif isinstance(arg0, FloatValue):
            if isinstance(arg1, IntValue):
                return FloatValue(math.pow(arg0.value, float(arg1.value)))
            elif isinstance(arg1, FloatValue):
                return FloatValue(math.pow(arg0.value, arg1.value))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"tan":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.tan(arg.value*math.pi/180))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.tan(arg.value*math.pi/180))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"sin":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(math.sin(arg.value*math.pi/180))
        elif isinstance(arg, FloatValue):
            return FloatValue(math.sin(arg.value*math.pi/180))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

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
        elif isinstance(arg, StrValue):
            return IntValue(int(arg.value))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"float":
        if len(args) != 1:
            raise_type_error(op_print + u" expects only 1 argument")
        arg = args[0]
        if isinstance(arg, IntValue):
            return FloatValue(float(arg.value))
        elif isinstance(arg, FloatValue):
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

    elif op == u"len":
        if len(args) != 1:
            raise_type_error(op_print + u" expects 1 argument")
        arg0 = args[0]
        if isinstance(arg0, StrValue):
            return IntValue(len(arg0.value))
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"simple_idx":
        if len(args) != 2:
            raise_type_error(op_print + u" expects 2 arguments")
        arg0, arg1 = args
        if isinstance(arg0, StrValue):
            if isinstance(arg1, IntValue):
                return StrValue(arg0.value[arg1.value])
            raise_type_error(op_print + u" expects int for the index, got " + get_type(arg1) + u" instead")
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

    elif op == u"idx":
        if len(args) != 4:
            raise_type_error(op_print + u" expects 4 arguments")
        arg0, arg1, arg2, arg3 = args
        if isinstance(arg0, StrValue):
            length = len(arg0.value)
            start = stop = 0 # To make rpython's flowspace happy
            if arg3 is not NONE:
                raise_type_error(op_print + u" doesn't support the step arg on str")
            if arg1 is NONE: # start
                start = 0
            elif isinstance(arg1, IntValue):
                start = arg1.value
                if start < 0:
                    start += length
                if start < 0:
                    raise_index_error(u"start-idx = " + int_to_str(start))
            else:
                raise_type_error(op_print + u" doesn't support " + get_type(arg1) + u" for the start arg")
            if arg2 is NONE: # stop
                stop = length
            elif isinstance(arg2, IntValue):
                stop = arg2.value
                if stop < 0:
                    stop += length
                if stop < 0:
                    raise_index_error(u"stop-idx = " + int_to_str(stop))
            else:
                raise_type_error(op_print + u" doesn't support " + get_type(arg2) + u" for the stop arg")
            return StrValue(arg0.value[start:stop])
        raise_type_error(op_print + u" not supported on " + get_type(args[0]))

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
def inner_repr(obj):
    if obj is None:
        return u"undefined"
    if obj is NONE:
        return u"none"
    elif isinstance(obj, IntValue):
        return int_to_str(obj.value)
    elif isinstance(obj, FloatValue):
        return str(bytes2(obj.value))
    elif isinstance(obj, StrValue):
        return obj.value
    elif isinstance(obj, FuncValue):
        string = u"Func<"
        bound_obj = obj.bound_obj
        if bound_obj is not None:
            if isinstance(bound_obj, ObjectValue):
                string += bound_obj.name + u"."
            else:
                string += get_type(bound_obj) + u"<object>."
        string += obj.name + u">"
        return string
    elif isinstance(obj, ListValue):
        string = u"["
        for i, element in enumerate(obj.array):
            string += inner_repr(element)
            if i != obj.len()-1:
                string += u", "
        return string + u"]"
    elif isinstance(obj, ObjectValue):
        if obj.type & 1:
            return u"<Object " + obj.name + u">"
        else:
            return u"<Class " + obj.name + u">"
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
        print(u"[STDOUT]: " + string)
        return NONE

    elif op == u"append":
        if len(args) != 2:
            raise_type_error(op + u" expects 2 arguments")
        arg0, arg1 = args
        if not isinstance(arg0, ListValue):
            raise_type_error(op + u" expects list for the 1st arg")
        arg0.append(arg1)
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
def _main(raw_bytecode):
    assert isinstance(raw_bytecode, bytes), "TypeError"
    debug(u"Derialising bytecode...", 2)
    flags, frame_size, env_size, attrs, bytecode, SOURCE_CODE = derialise(raw_bytecode)
    debug(u"Parsing flags...", 2)
    debug(u"Starting interpreter...", 1)
    return interpret(flags, frame_size, env_size, attrs, bytecode, SOURCE_CODE)

def main(filepath):
    debug(u"Reading file...", 2)
    with open(filepath, "rb") as file:
        data = file.read()
    return _main(data)


# For testing (note runs with PYTHON == 3)
if __name__ == "__main__":
    if PYTHON == 3:
        from time import perf_counter
        start = perf_counter()
    main("../code-examples/example.clizz")
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
# https://rpython.readthedocs.io/en/latest/jit/optimizer.html
# /media/thelizzard/TheLizzardOS-SD/rootfs/home/thelizzard/honours/lizzzard/src/frontend/rpython/jit/metainterp/optimizeopt