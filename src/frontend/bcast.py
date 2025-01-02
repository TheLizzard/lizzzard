try:
    from python3.rpython_compat import *
    from python3.int_compat import *
    from python3.star import *
except ImportError:
    from .python3.rpython_compat import *
    from .python3.int_compat import *
    from .python3.star import *

def serialise_int(integer, size):
    return int_to_bytes(integer, size)
def derialise_int(data, size):
    int_bytes, data = data[0:size], data[size:]
    return int_from_bytes(int_bytes), data

def serialise_str(string, size_size):
    assert isinstance(size_size, int), "TypeError"
    assert isinstance(string, str), "TypeError"
    encoded = string.encode("utf-8")
    return int_to_bytes(len(encoded), size_size) + encoded
def derialise_str(data, size_size):
    assert len(data) >= size_size, "ValueError"
    length, data = derialise_int(data, size_size)
    assert length >= 0, "ValueError"
    assert len(data) >= length, "ValueError"
    string_bytes, data = data[0:length], data[length:]
    string = string_bytes.decode("utf-8")
    return string, data

def serialise_list_int(array, size_size, size_item):
    assert isinstance(size_size, int), "TypeError"
    assert isinstance(array, list), "TypeError"
    encoded = int_to_bytes(len(array), size_size)
    for item in array:
        assert isinstance(item, int), "TypeError"
        encoded += int_to_bytes(item, size_item)
    return encoded
def derialise_list_int(data, size_size, size_item):
    assert isinstance(size_size, int), "TypeError"
    assert isinstance(size_item, int), "TypeError"
    assert isinstance(data, bytes), "TypeError"
    assert len(data) >= size_size, "ValueError"
    length, data = derialise_int(data, size_size)
    decoded = []
    for i in range(length):
        item, data = derialise_int(data, size_item)
        decoded.append(item)
    return decoded, data


_free_ast_t_id = 0
def free_ast_t_id():
    global _free_ast_t_id
    ast_t_id, _free_ast_t_id = _free_ast_t_id, _free_ast_t_id+1
    return ast_t_id
def assert_ast_t_id(data, ast_t_id):
    assert isinstance(data, bytes), "TypeError"
    assert isinstance(ast_t_id, int), "TypeErrror"
    assert 0 <= ast_t_id <= 1<<(AST_T_ID_SIZE*8), "ValueError"
    assert len(data) >= AST_T_ID_SIZE, "ValueError"
    ast_t_id_read, data = derialise_int(data, AST_T_ID_SIZE)
    assert ast_t_id_read == ast_t_id, "AssertionError"
    return data
def serialise_ast_t_id(ast_t_id):
    return serialise_int(ast_t_id, AST_T_ID_SIZE)
def derialise_ast_t_id(data):
    return derialise_int(data, AST_T_ID_SIZE)

REG_SIZE = 1 # number of bytes needed to store a register id
INT_LITERAL_SIZE = 8 # number of bytes needed to store a literal ints
STR_LITERAL_SIZE = 4 # number of bytes to store a literal string size
AST_T_ID_SIZE = 1 # Size of free_ast_id
NAME_SIZE = 1 # Variable name/bable size

LINK_SIZE = 1 # number of bytes used to store the link
ENV_SIZE_SIZE = 2 # number of bytes used to store the env_size of a func
FUNC_ID_SIZE = 2 # number of bytes used to store the func label

ATTR_SIZE_SIZE = 2 # number of bytes to store the length of the set of attrs
ATTR_SIZE = 2 # number of bytes to store the attr id instead of the attr str

MAX_REG_VALUE = (1<<(REG_SIZE<<3)) - 1
MAX_LINK_VALUE = (1<<(LINK_SIZE<<3)) - 1


class Bast:
    __slots__ = "err_idx"


class Bable(Bast):
    _immutable_fields_ = ["id"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "id"

    def __init__(self, id):
        assert isinstance(id, str), "TypeError"
        self.id = const_str(id)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_str(self.id, NAME_SIZE)

    def derialise(data):
        data = assert_ast_t_id(data, Bable.AST_T_ID)
        id, data = derialise_str(data, NAME_SIZE)
        return Bable(id), data


class BCall(Bast):
    _immutable_fields_ = ["regs"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "regs"

    def __init__(self, regs):
        assert isinstance(regs, list), "TypeError"
        for reg in regs:
            assert isinstance(reg, int), "TypeError"
            assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.regs = [const(reg) for reg in regs] # regs[0]:=result, regs[1]:=func

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_list_int(self.regs, 1, REG_SIZE)

    def derialise(data):
        data = assert_ast_t_id(data, BCall.AST_T_ID)
        regs, data = derialise_list_int(data, 1, REG_SIZE)
        return BCall(regs), data


class BStoreLoadDict(Bast):
    _immutable_fields_ = ["name", "reg", "storing"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "name", "reg", "storing"

    def __init__(self, name, reg, storing):
        assert isinstance(storing, bool), "TypeError"
        assert isinstance(name, str), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.storing = const(storing)
        self.name = const_str(name)
        self.reg = const(reg)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_str(self.name, NAME_SIZE) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.storing, 1)

    def derialise(data):
        data = assert_ast_t_id(data, BStoreLoadDict.AST_T_ID)
        name, data = derialise_str(data, NAME_SIZE)
        reg, data = derialise_int(data, REG_SIZE)
        storing, data = derialise_int(data, 1)
        assert 0 <= storing <= 1, "ValueError"
        return BStoreLoadDict(name, reg, bool(storing)), data


class BStoreLoadList(Bast):
    _immutable_fields_ = ["link", "name", "reg", "storing"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "link", "name", "reg", "storing"

    def __init__(self, link, name, reg, storing):
        assert isinstance(storing, bool), "TypeError"
        assert isinstance(link, int), "TypeError"
        assert isinstance(name, int), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        assert 0 <= link < MAX_LINK_VALUE, "ValueError"
        self.storing = const(storing)
        self.link = const(link)
        self.name = const(name)
        self.reg = const(reg)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.link, LINK_SIZE) + \
               serialise_int(self.name, ENV_SIZE_SIZE) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.storing, 1)

    def derialise(data):
        data = assert_ast_t_id(data, BStoreLoadList.AST_T_ID)
        link, data = derialise_int(data, LINK_SIZE)
        name, data = derialise_int(data, ENV_SIZE_SIZE)
        reg, data = derialise_int(data, REG_SIZE)
        storing, data = derialise_int(data, 1)
        assert 0 <= storing <= 1, "ValueError"
        return BStoreLoadList(link, name, reg, bool(storing)), data


class BLiteralHolder:
    _immutable_fields_ = []
    __slots__ = ()

class BLiteralInt(BLiteralHolder):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    def __init__(self, value):
        assert isinstance(value, int), "TypeError"
        self.value = const(value)
    def serialise(self):
        return serialise_int(self.value, INT_LITERAL_SIZE)
    def derialise(data):
        value, data = derialise_int(data, INT_LITERAL_SIZE)
        return BLiteralInt(value), data

class BLiteralStr(BLiteralHolder):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    def __init__(self, value):
        assert isinstance(value, str), "TypeError"
        self.value = const_str(value)
    def serialise(self):
        return serialise_str(self.value, STR_LITERAL_SIZE)
    def derialise(data):
        value, data = derialise_str(data, STR_LITERAL_SIZE)
        return BLiteralStr(value), data

class BLiteralFunc(BLiteralHolder):
    _immutable_fields_ = ["env_size", "value", "nargs", "name"]
    __slots__ = "env_size", "value", "nargs", "name"
    def __init__(self, env_size, value, nargs, name):
        assert isinstance(env_size, int), "TypeError"
        assert isinstance(value, int), "TypeError"
        assert isinstance(nargs, int), "TypeError"
        assert isinstance(name, str), "TypeError"
        self.env_size = const(env_size)
        self.name = const_str(name)
        self.value = const(value)
        self.nargs = const(nargs)
    def serialise(self):
        data = serialise_int(self.env_size, ENV_SIZE_SIZE)
        data += serialise_int(self.value, FUNC_ID_SIZE)
        data += serialise_str(self.name, NAME_SIZE)
        data += serialise_int(self.nargs, REG_SIZE)
        return data
    def derialise(data):
        env_size, data = derialise_int(data, ENV_SIZE_SIZE)
        value, data = derialise_int(data, FUNC_ID_SIZE)
        name, data = derialise_str(data, NAME_SIZE)
        nargs, data = derialise_int(data, REG_SIZE)
        return BLiteralFunc(env_size, value, nargs, name), data

class BLiteralClass(BLiteralHolder):
    _immutable_fields_ = ["bases", "label", "name"]
    __slots__ = "bases", "label", "name"
    def __init__(self, bases, label, name):
        assert isinstance(bases, list), "TypeError"
        assert isinstance(label, str), "TypeError"
        assert isinstance(name, str), "TypeError"
        for reg in bases:
            assert isinstance(reg, int), "TypeError"
            assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.bases = [const(reg) for reg in bases]
        self.label = const_str(label)
        self.name = const_str(name)
    def serialise(self):
        data = serialise_str(self.label, NAME_SIZE)
        data += serialise_str(self.name, NAME_SIZE)
        data += serialise_int(len(self.bases), REG_SIZE)
        for base in self.bases:
            data += serialise_int(base, REG_SIZE)
        return data
    def derialise(data):
        label, data = derialise_str(data, NAME_SIZE)
        name, data = derialise_str(data, NAME_SIZE)
        nbases, data = derialise_int(data, REG_SIZE)
        bases = []
        for _ in range(nbases):
            base, data = derialise_int(data, REG_SIZE)
            bases.append(base)
        return BLiteralClass(bases, label, name), data

class _BLiteralEmpty(BLiteralHolder):
    _immutable_fields_ = []
    __slots__ = ()
    def __init__(self): pass
    def serialise(self): return b""
    def derialise(data): return BNONE, data

BNONE = _BLiteralEmpty()


class BLiteral(Bast):
    _immutable_fields_ = ["reg", "literal", "type"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "reg", "literal", "type"
    UNDEFINED_T = 0
    CLASS_T = 1
    FUNC_T = 2
    PROC_T = 3
    NONE_T = 4
    LIST_T = 5
    INT_T = 6
    STR_T = 7
    EMPTY_TS = (UNDEFINED_T, NONE_T, LIST_T)

    def __init__(self, reg, literal, type):
        assert isinstance(literal, BLiteralHolder), "TypeError"
        assert isinstance(type, int), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.literal = const(literal)
        self.type = const(type)
        self.reg = const(reg)

    def serialise(self):
        if self.type == BLiteral.INT_T:
            assert isinstance(self.literal, BLiteralInt), "TypeError"
            literal = self.literal.serialise()
        elif self.type in (BLiteral.PROC_T, BLiteral.FUNC_T):
            assert isinstance(self.literal, BLiteralFunc), "TypeError"
            literal = self.literal.serialise()
        elif self.type == BLiteral.STR_T:
            assert isinstance(self.literal, BLiteralStr), "TypeError"
            literal = self.literal.serialise()
        elif self.type in BLiteral.EMPTY_TS:
            assert self.literal is BNONE, "TypeError"
            literal = self.literal.serialise()
        elif self.type == BLiteral.CLASS_T:
            assert isinstance(self.literal, BLiteralClass), "TypeError"
            literal = self.literal.serialise()
        else:
            raise ValueError("InvalidType")
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.type, REG_SIZE) + \
               literal

    def derialise(data):
        data = assert_ast_t_id(data, BLiteral.AST_T_ID)
        reg, data = derialise_int(data, REG_SIZE)
        type, data = derialise_int(data, REG_SIZE)
        if type == BLiteral.INT_T:
            literal, data = BLiteralInt.derialise(data)
        elif type in (BLiteral.PROC_T, BLiteral.FUNC_T):
            literal, data = BLiteralFunc.derialise(data)
        elif type == BLiteral.STR_T:
            literal, data = BLiteralStr.derialise(data)
        elif type in BLiteral.EMPTY_TS:
            literal = BNONE
        elif type == BLiteral.CLASS_T:
            literal, data = BLiteralClass.derialise(data)
        else:
            raise ValueError("InvalidType")
        assert isinstance(literal, BLiteralHolder), "TypeError"
        return BLiteral(reg, literal, type), data


class BJump(Bast):
    ### WARNING: BJump clears condition_reg no mater what!!!
    _immutable_fields_ = ["label", "negated", "condition_reg"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "label", "negated", "condition_reg"

    def __init__(self, label, condition_reg, negated):
        assert isinstance(condition_reg, int), "TypeError"
        assert isinstance(negated, bool), "TypeError"
        assert isinstance(label, str), "TypeError"
        assert 0 <= condition_reg < MAX_REG_VALUE, "ValueError"
        self.condition_reg = const(condition_reg)
        self.negated = const(negated)
        self.label = const_str(label)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_str(self.label, NAME_SIZE) + \
               serialise_int(self.negated, 1) + \
               serialise_int(self.condition_reg, REG_SIZE)

    def derialise(data):
        data = assert_ast_t_id(data, BJump.AST_T_ID)
        label, data = derialise_str(data, NAME_SIZE)
        negated, data = derialise_int(data, REG_SIZE)
        condition_reg, data = derialise_int(data, REG_SIZE)
        assert 0 <= negated <= 1, "ValueError"
        return BJump(label, condition_reg, bool(negated)), data


class BRegMove(Bast):
    _immutable_fields_ = ["reg1", "reg2"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "reg1", "reg2"

    def __init__(self, reg1, reg2):
        assert isinstance(reg1, int), "TypeError"
        assert isinstance(reg2, int), "TypeError"
        assert 0 <= reg1 < MAX_REG_VALUE, "ValueError"
        assert 0 <= reg2 < MAX_REG_VALUE, "ValueError"
        # Currently only used as the phi in single assignment form (ifexpr)
        # reg1 := reg2
        self.reg1 = const(reg1)
        self.reg2 = const(reg2)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.reg1, REG_SIZE) + \
               serialise_int(self.reg2, REG_SIZE)

    def derialise(data):
        data = assert_ast_t_id(data, BRegMove.AST_T_ID)
        reg1, data = derialise_int(data, REG_SIZE)
        reg2, data = derialise_int(data, REG_SIZE)
        return BRegMove(reg1, reg2), data


class BLoadLink(Bast):
    _immutable_fields_ = ["name", "link"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "name", "link"

    def __init__(self, name, link):
        assert isinstance(name, str), "TypeError"
        assert isinstance(link, int), "TypeError"
        self.name = const_str(name)
        self.link = const(link)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.link, 1) + \
               serialise_str(self.name, NAME_SIZE)

    def derialise(data):
        data = assert_ast_t_id(data, BLoadLink.AST_T_ID)
        link, data = derialise_int(data, 1)
        name, data = derialise_str(data, NAME_SIZE)
        return BLoadLink(name, link), data


class BRet(Bast):
    _immutable_fields_ = ["reg", "capture_env"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "reg", "capture_env"

    def __init__(self, reg, capture_env):
        assert isinstance(capture_env, bool), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.capture_env = const(capture_env)
        self.reg = const(reg)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.capture_env, 1) + \
               serialise_int(self.reg, REG_SIZE)

    def derialise(data):
        data = assert_ast_t_id(data, BRet.AST_T_ID)
        capture_env, data = derialise_int(data, 1)
        reg, data = derialise_int(data, REG_SIZE)
        assert 0 <= capture_env <= 1, "ValueError"
        return BRet(reg, bool(capture_env)), data


class BDotDict(Bast):
    _immutable_fields_ = ["reg", "obj_reg", "attr", "storing"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "reg", "obj_reg", "attr", "storing"

    def __init__(self, obj_reg, attr, reg, storing):
        assert isinstance(storing, bool), "TypeError"
        assert isinstance(obj_reg, int), "TypeError"
        assert isinstance(attr, str), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert 0 <= obj_reg < MAX_REG_VALUE, "ValueError"
        assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.attr = const_str(attr)
        self.storing = const(storing)
        self.obj_reg = const(obj_reg)
        self.reg = const(reg)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_str(self.attr, NAME_SIZE) + \
               serialise_int(self.obj_reg, REG_SIZE) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.storing, 1)

    def derialise(data):
        data = assert_ast_t_id(data, BDotDict.AST_T_ID)
        attr, data = derialise_str(data, NAME_SIZE)
        obj_reg, data = derialise_int(data, REG_SIZE)
        reg, data = derialise_int(data, REG_SIZE)
        storing, data = derialise_int(data, 1)
        assert 0 <= storing <= 1, "ValueError"
        return BDotDict(obj_reg, attr, reg, bool(storing)), data


class BDotList(Bast):
    _immutable_fields_ = ["reg", "obj_reg", "attr", "storing"]
    AST_T_ID = free_ast_t_id()
    __slots__ = "reg", "obj_reg", "attr", "storing"

    def __init__(self, obj_reg, attr, reg, storing):
        assert isinstance(storing, bool), "TypeError"
        assert isinstance(obj_reg, int), "TypeError"
        assert isinstance(attr, int), "TypeError"
        assert isinstance(reg, int), "TypeError"
        assert 0 <= obj_reg < MAX_REG_VALUE, "ValueError"
        assert 0 <= reg < MAX_REG_VALUE, "ValueError"
        self.storing = const(storing)
        self.obj_reg = const(obj_reg)
        self.attr = const(attr)
        self.reg = const(reg)

    def serialise(self):
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.obj_reg, REG_SIZE) + \
               serialise_int(self.attr, ATTR_SIZE) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.storing, 1)

    def derialise(data):
        data = assert_ast_t_id(data, BDotList.AST_T_ID)
        obj_reg, data = derialise_int(data, REG_SIZE)
        attr, data = derialise_int(data, ATTR_SIZE)
        reg, data = derialise_int(data, REG_SIZE)
        storing, data = derialise_int(data, 1)
        assert 0 <= storing <= 1, "ValueError"
        return BDotList(obj_reg, attr, reg, bool(storing)), data


def reg_to_str(reg):
    return u"reg[" + int_to_str(reg) + u"]"

def env_list_to_str(name, link):
    if link == 0:
        return u"nameidx[" + int_to_str(name) + u"]"
    return u"linked" + int_to_str(link) + u"[" + int_to_str(name) + u"]"

def bytecode_list_to_str(bytecodes, mini=False):
    tab = u"" if mini else u"\t"
    output = u""
    for i, bt in enumerate(bytecodes):
        assert isinstance(bt, Bast), "TypeError"
        if not mini:
            output += int_to_str(i, zfill=3) + u" "
        if isinstance(bt, Bable):
            output += bt.id + u":"
        elif isinstance(bt, BLoadLink):
            output += tab + u"name[" + bt.name + u"]:=link[" + \
                      int_to_str(bt.link) + u"]"
        elif isinstance(bt, BCall):
            output += tab + reg_to_str(bt.regs[0]) + u":=" + \
                      reg_to_str(bt.regs[1]) + u"(" + \
                      u",".join([reg_to_str(i) for i in bt.regs[2:]]) + \
                      u")"
        elif isinstance(bt, BStoreLoadDict):
            if bt.storing:
                output += tab + u"name[" + bt.name + u"]:=" + \
                          reg_to_str(int(bt.reg))
            else:
                output += tab + reg_to_str(bt.reg) + u":=name[" + bt.name + \
                          u"]"
        elif isinstance(bt, BStoreLoadList):
            if bt.storing:
                output += tab + env_list_to_str(bt.name, bt.link) + u":=" + \
                          reg_to_str(int(bt.reg))
            else:
                output += tab + reg_to_str(bt.reg) + u":=" + \
                          env_list_to_str(bt.name, bt.link)
        elif isinstance(bt, BLiteral):
            bt_literal = bt.literal
            if bt.type in (BLiteral.FUNC_T, BLiteral.PROC_T):
                if bt.type == BLiteral.FUNC_T:
                    t = u"func"
                else:
                    t = u"proc"
                assert isinstance(bt_literal, BLiteralFunc), "TypeError"
                literal = t + u"[" + int_to_str(bt_literal.value) + u", " + \
                          u"env_size=" + int_to_str(bt_literal.env_size) + \
                          u",nargs=" + int_to_str(bt_literal.nargs) + u"]"
            elif bt.type == BLiteral.NONE_T:
                literal = u"none"
            elif bt.type == BLiteral.UNDEFINED_T:
                literal = u"undefined"
            elif bt.type == BLiteral.LIST_T:
                literal = u"list"
            elif bt.type == BLiteral.INT_T:
                assert isinstance(bt_literal, BLiteralInt), "TypeError"
                literal = int_to_str(bt_literal.value)
            elif bt.type == BLiteral.STR_T:
                assert isinstance(bt_literal, BLiteralStr), "TypeError"
                literal = u'"' + bt_literal.value[1:] + u'"'
            elif bt.type == BLiteral.CLASS_T:
                assert isinstance(bt_literal, BLiteralClass), "TypeError"
                literal = u"new_class[" + bt_literal.label
                # literal += u", env_size=" + int_to_str(bt_literal.env_size)
                if len(bt_literal.bases) != 0:
                    literal += u", [bases="
                    for i in range(len(bt_literal.bases)):
                        literal += reg_to_str(bt_literal.bases[i])
                        if i != len(bt_literal.bases)-1:
                            literal += u","
                    literal += u"]"
                literal += u"]"
            else:
                literal = u"unknown"
            output += tab + reg_to_str(bt.reg) + u":=" + literal
        elif isinstance(bt, BJump):
            output += tab + u"jumpif(" + (u"!" if bt.negated else u"") + \
                      reg_to_str(bt.condition_reg) + u")=>" + bt.label
        elif isinstance(bt, BRegMove):
            output += tab + reg_to_str(bt.reg1) + u":=" + reg_to_str(bt.reg2)
        elif isinstance(bt, BRet):
            if bt.capture_env:
                output += tab + u"capture_env"
            else:
                output += tab + u"return[" + reg_to_str(bt.reg) + u"]"
        elif isinstance(bt, BDotDict):
            tmp = reg_to_str(bt.obj_reg) + u".attr[" + bt.attr + u"]"
            if bt.storing:
                output += tab + tmp + u":=" + reg_to_str(bt.reg)
            else:
                output += tab + reg_to_str(bt.reg) + u":=" + tmp
        elif isinstance(bt, BDotList):
            tmp = reg_to_str(bt.obj_reg) + u".attr[" + int_to_str(bt.attr) + \
                  u"]"
            if bt.storing:
                output += tab + tmp + u":=" + reg_to_str(bt.reg)
            else:
                output += tab + reg_to_str(bt.reg) + u":=" + tmp
        else:
            output += u"UnknownInstruction"
        if i != len(bytecodes)-1:
            output += u"\n"
    return output


"""
if ENV_IS_LIST:
    BAST_TYPES = [Bable, BCall, BStoreLoadList, BLiteral,
                  BJump, BRegMove, BRet, BDotList]
else:
    BAST_TYPES = [Bable, BCall, BStoreLoadDict, BLiteral,
                  BJump, BRegMove, BLoadLink, BRet, BDotDict]
"""
BAST_TYPES = [Bable, BCall, BStoreLoadDict, BStoreLoadList, BLiteral, BJump,
              BRegMove, BLoadLink, BRet, BDotDict, BDotList]
TABLE = {T.AST_T_ID:T.derialise for T in BAST_TYPES}


VERSION = 1
FRAME_SIZE = 1 # number of bytes that hold the max frame size
VERSION_SIZE = 4 # number of bytes to store the version
BASIC_FLAG_SIZE = 16 # bits for basic flags
assert BASIC_FLAG_SIZE&7 == 0, "ValueError"


class FeatureFlags:
    _immutable_fields_ = ["flag_names"]
    __slots__ = "flag_names"

    ENV_IS_LIST = 1
    CLEAR_AFTER_USE = 2
    EXTENDED = BASIC_FLAG_SIZE
    ALL_FLAGS = {"ENV_IS_LIST":ENV_IS_LIST,
                 "CLEAR_AFTER_USE":CLEAR_AFTER_USE,
                 "EXTENDED":EXTENDED}

    def __init__(self):
        self.clear()

    def clear(self):
        self.flag_names = {}

    def is_set(self, flag_name):
        assert isinstance(flag_name, bytes2)
        assert flag_name in FeatureFlags.ALL_FLAGS, "Unknown flag"
        return flag_name in self.flag_names

    def set(self, flag_name):
        assert isinstance(flag_name, bytes2)
        assert flag_name in FeatureFlags.ALL_FLAGS, "Unknown flag"
        self.flag_names[flag_name] = None

    def serialise(self):
        output = 0
        for flag_name in self.flag_names:
            assert flag_name in FeatureFlags.ALL_FLAGS, "Unknown flag"
            flag = FeatureFlags.ALL_FLAGS[flag_name]
            assert 0 < flag <= BASIC_FLAG_SIZE, "ValueError"
            output |= 1<<(flag-1)
        return serialise_int(output, BASIC_FLAG_SIZE>>3)

    def derialise(self, serialised_flags):
        assert isinstance(serialised_flags, int), "TypeError"
        assert 0 <= serialised_flags < 1<<FeatureFlags.EXTENDED, "ValueError"
        if serialised_flags & (1<<(FeatureFlags.EXTENDED-1)):
            raise NotImplementedError("Extended flags not implemented")
        self.clear()
        for flag_name, flag in FeatureFlags.ALL_FLAGS.items():
            if serialised_flags & (1<<(flag-1)):
                self.set(flag_name)

    def issame(self, other):
        assert isinstance(other, FeatureFlags)
        return list(self.flag_names) == list(other.flag_names)

    def __repr__(self):
        output = "Flags["
        for i, flag_name in enumerate(self.flag_names):
            output += flag_name
            if i != len(self.flag_names)-1:
                output += ", "
        return output + "]"


def serialise(flags, frame_size, env_size, attrs, bytecode):
    output = serialise_int(VERSION, VERSION_SIZE) + \
             flags.serialise() + \
             serialise_int(frame_size, FRAME_SIZE)
    if flags.is_set("ENV_IS_LIST"):
        output += serialise_int(env_size, ENV_SIZE_SIZE)
        output += serialise_int(len(attrs), ATTR_SIZE_SIZE)
        for attr in attrs:
            output += serialise_str(attr, NAME_SIZE)
    return output + bytecode

def derialise(data):
    version, data = derialise_int(data, VERSION_SIZE)
    raw_flags, data = derialise_int(data, BASIC_FLAG_SIZE>>3)
    frame_size, data = derialise_int(data, FRAME_SIZE)
    flags = FeatureFlags()
    flags.derialise(raw_flags)
    attrs = []
    if flags.is_set("ENV_IS_LIST"):
        env_size, data = derialise_int(data, ENV_SIZE_SIZE)
        attrs_size, data = derialise_int(data, ATTR_SIZE_SIZE)
        for _ in range(attrs_size):
            attr, data = derialise_str(data, NAME_SIZE)
            attrs.append(attr)
    else:
        env_size, attrs = 0, []
    bytecode = []
    while data:
        ast_t_id, _ = derialise_ast_t_id(data)
        bast, data = TABLE[ast_t_id](data)
        assert isinstance(bast, Bast), "Impossible"
        bytecode.append(bast)
    return flags, frame_size, env_size, attrs, bytecode


"""
Special:
    Calling `simple_idx=` with args=(array, idx, value)
    Calling `.=` with args=(obj, ?, value)
Not implemented:
    Calling `idx=` with args=(array, start, stop, step, value)
"""

BUILTIN_OPS = ["+", "-", "*", "%", "//", "==", "!=", "<", ">", "<=", ">=",
               "or", "not", "len", ".", ".=", "idx", "simple_idx",
               "simple_idx=", "[]"]
BUILTIN_SIDES = ["print", "append"]
BUILTIN_HELPERS = ["$prev_env"]

# "__class__" was in BULTIN_HELPERS but class scope no longer gets an env
CLS_REG = 2

BUILTIN_OPS = list(map(str, BUILTIN_OPS))
BUILTIN_SIDES = list(map(str, BUILTIN_SIDES))
BUILTIN_HELPERS = list(map(str, BUILTIN_HELPERS))
BUILTINS = BUILTIN_HELPERS + BUILTIN_OPS + BUILTIN_SIDES