from __future__ import annotations
from python3.bytes_compat import bytes_startswith, bytes_decode
from python3.int_compat import int_to_bytes, int_from_bytes
from python3.str_compat import str_encode


def serialise_int(integer:int, size:int) -> bytes:
    return int_to_bytes(integer, size)
def derialise_int(data:bytes, size:int) -> tuple[int,bytes]:
    int_bytes, data = data[0:size], data[size:]
    return int_from_bytes(int_bytes), data

def serialise_str(string:str, size_size:int) -> bytes:
    assert isinstance(size_size, int), "TypeError"
    assert isinstance(string, str), "TypeError"
    return int_to_bytes(len(string), size_size) + str_encode(string)
def derialise_str(data:bytes, size_size:int) -> tuple[str,bytes]:
    assert len(data) >= size_size, "ValueError"
    length, data = derialise_int(data, size_size)
    assert len(data) >= length, "ValueError"
    string_bytes, data = data[0:length], data[length:]
    string:str = bytes_decode(string_bytes)
    return string, data

def serialise_list_int(array:list[int], size_size:int, size_item:int) -> bytes:
    assert isinstance(size_size, int), "TypeError"
    assert isinstance(array, list), "TypeError"
    encoded:bytes = int_to_bytes(len(array), size_size)
    for item in array:
        assert isinstance(item, int), "TypeError"
        encoded += int_to_bytes(item, size_item)
    return encoded
def derialise_list_int(data:bytes, size_size:int, size_item:int):
    assert isinstance(size_size, int), "TypeError"
    assert isinstance(size_item, int), "TypeError"
    assert isinstance(data, bytes), "TypeError"
    assert len(data) >= size_size, "ValueError"
    length, data = derialise_int(data, size_size)
    decoded:list[int] = []
    for i in range(length):
        item, data = derialise_int(data, size_item)
        decoded.append(item)
    return decoded, data


_free_ast_t_id:int = 0
def free_ast_t_id() -> int:
    global _free_ast_t_id
    ast_t_id, _free_ast_t_id = _free_ast_t_id, _free_ast_t_id+1
    return ast_t_id
def assert_ast_t_id(data:bytes, ast_t_id:int) -> bytes:
    assert isinstance(data, bytes), "TypeError"
    assert isinstance(ast_t_id, int), "TypeErrror"
    assert 0 <= ast_t_id <= 1<<(AST_T_ID_SIZE*8), "ValueError"
    assert len(data) >= AST_T_ID_SIZE, "ValueError"
    ast_t_id_read, data = derialise_int(data, AST_T_ID_SIZE)
    assert ast_t_id_read == ast_t_id, "AssertionError"
    return data
def serialise_ast_t_id(ast_t_id:int) -> bytes:
    return serialise_int(ast_t_id, AST_T_ID_SIZE)
def derialise_ast_t_id(data:bytes) -> tuple[int,bytes]:
    return derialise_int(data, AST_T_ID_SIZE)

REG_SIZE:int = 1 # number of bytes needed to store a register id
INT_LITERAL_SIZE:int = 8 # number of bytes needed to store a literal ints
STR_LITERAL_SIZE:int = 4 # number of bytes to store a literal string size
AST_T_ID_SIZE:int = 1 # Size of free_ast_id


class Bast:
    __slots__ = ()


class Bable(Bast):
    AST_T_ID:bytes = free_ast_t_id()
    __slots__ = "id"

    def __init__(self, id:str) -> Bable:
        assert isinstance(id, str), "TypeError"
        self.id:str = id

    def serialise(self) -> bytes:
        return serialise_ast_t_id(self.AST_T_ID) + serialise_str(self.id, 1)

    @classmethod
    def derialise(Cls:type, data:bytes) -> tuple[Bable,bytes]:
        data:bytes = assert_ast_t_id(data, Cls.AST_T_ID)
        id, data = derialise_str(data, 1)
        return Bable(id), data


class BCall(Bast):
    AST_T_ID:int = free_ast_t_id()
    __slots__ = "regs"

    def __init__(self, regs:list[int]) -> BCall:
        assert isinstance(regs, list), "TypeError"
        for reg in regs:
            assert isinstance(reg, int), "TypeError"
        self.regs:list[int] = regs # 1st-reg := result, 2nd-reg := func

    def serialise(self) -> bytes:
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_list_int(self.regs, 1, REG_SIZE)

    @classmethod
    def derialise(Cls:type, data:bytes) -> tuple[BCall,bytes]:
        data:bytes = assert_ast_t_id(data, Cls.AST_T_ID)
        regs, data = derialise_list_int(data, 1, REG_SIZE)
        return BCall(regs), data


class BStoreLoad(Bast):
    AST_T_ID:int = free_ast_t_id()
    __slots__ = "name", "reg", "storing"

    def __init__(self, name:str, reg:int, storing:bool) -> BLoad:
        assert isinstance(storing, bool), "TypeError"
        assert isinstance(name, str), "TypeError"
        assert isinstance(reg, int), "TypeError"
        self.storing:bool = storing
        self.name:str = name
        self.reg:int = reg

    def serialise(self) -> bytes:
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_str(self.name, 2) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.storing, 1)

    @classmethod
    def derialise(Cls:type, data:bytes) -> tuple[BStoreLoad,bytes]:
        data:bytes = assert_ast_t_id(data, Cls.AST_T_ID)
        name, data = derialise_str(data, 2)
        reg, data = derialise_int(data, REG_SIZE)
        storing, data = derialise_int(data, 1)
        assert 0 <= storing <= 1, "ValueError"
        return BStoreLoad(name, reg, bool(storing)), data


class BLiteral(Bast):
    AST_T_ID:int = free_ast_t_id()
    __slots__ = "reg", "literal", "type"
    FUNC_T:int = 0
    PROC_T:int = 1
    INT_T:int = 2
    STR_T:int = 3

    def __init__(self, reg:int, literal:int, type:int) -> BLiteral:
        if type in (BLiteral.FUNC_T, BLiteral.PROC_T, BLiteral.INT_T):
            assert isinstance(literal, int), "TypeError"
        elif type == BLiteral.STR_T:
            assert isinstance(literal, str), "TypeError"
        else:
            raise ValueError("InvalidType")
        assert isinstance(type, int), "TypeError"
        assert isinstance(reg, int), "TypeError"
        self.literal:object = literal
        self.type:int = type
        self.reg:int = reg

    def serialise(self) -> bytes:
        if self.type in (BLiteral.FUNC_T, BLiteral.PROC_T, BLiteral.INT_T):
            literal:bytes = serialise_int(self.literal, INT_LITERAL_SIZE)
        elif self.type == BLiteral.STR_T:
            literal:bytes = serialise_str(self.literal, STR_LITERAL_SIZE)
        else:
            raise ValueError("InvalidType")
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.reg, REG_SIZE) + \
               serialise_int(self.type, REG_SIZE) + \
               literal

    @classmethod
    def derialise(Cls:type, data:bytes) -> tuple[BLiteral,bytes]:
        data:bytes = assert_ast_t_id(data, Cls.AST_T_ID)
        reg, data = derialise_int(data, REG_SIZE)
        type, data = derialise_int(data, REG_SIZE)
        if type in (BLiteral.FUNC_T, BLiteral.PROC_T, BLiteral.INT_T):
            literal, data = derialise_int(data, INT_LITERAL_SIZE)
        elif type == BLiteral.STR_T:
            literal, data = derialise_str(data, STR_LITERAL_SIZE)
        else:
            raise ValueError("InvalidType")
        return BLiteral(reg, literal, type), data


class BJump(Bast):
    AST_T_ID:int = free_ast_t_id()
    __slots__ = "label", "negated", "condition_reg"

    def __init__(self, label:Bable, condition_reg:int, negated:bool) -> BJump:
        assert isinstance(condition_reg, int), "TypeError"
        assert isinstance(negated, bool), "TypeError"
        assert isinstance(label, Bable), "TypeError"
        self.condition_reg:int = condition_reg
        self.negated:bool = negated
        self.label:Bable = label

    def serialise(self) -> bytes:
        return serialise_ast_t_id(self.AST_T_ID) + \
               self.label.serialise() + \
               serialise_int(self.negated, 1) + \
               serialise_int(self.condition_reg, REG_SIZE)

    @classmethod
    def derialise(Cls:type, data:bytes) -> tuple[BJump,bytes]:
        data:bytes = assert_ast_t_id(data, Cls.AST_T_ID)
        bable, data = Bable.derialise(data)
        negated, data = derialise_int(data, REG_SIZE)
        condition_reg, data = derialise_int(data, REG_SIZE)
        assert 0 <= negated <= 1, "ValueError"
        return BJump(bable, condition_reg, bool(negated)), data


class BRegMove(Bast):
    AST_T_ID:int = free_ast_t_id()
    __slots__ = "reg1", "reg2"

    def __init__(self, reg1:int, reg2:int) -> BRegMove:
        assert isinstance(reg1, int), "TypeError"
        assert isinstance(reg2, int), "TypeError"
        # reg1 := reg2
        # Writing to reg 2 means return from function
        self.reg1:int = reg1
        self.reg2:int = reg2

    def serialise(self) -> bytes:
        return serialise_ast_t_id(self.AST_T_ID) + \
               serialise_int(self.reg1, REG_SIZE) + \
               serialise_int(self.reg2, REG_SIZE)

    @classmethod
    def derialise(Cls:type, data:bytes) -> tuple[BRegMove,bytes]:
        data:bytes = assert_ast_t_id(data, Cls.AST_T_ID)
        reg1, data = derialise_int(data, REG_SIZE)
        reg2, data = derialise_int(data, REG_SIZE)
        return BRegMove(reg1, reg2), data


def bytecode_list_to_str(bytecodes:list[Bast]) -> str:
    tab:str = "\t"
    output:str = ""
    for bt in bytecodes:
        assert isinstance(bt, Bast), "TypeError"
        if isinstance(bt, Bable):
            output += bt.id + ":"
        elif isinstance(bt, BCall):
            output += tab + str(bt.regs[0]) + " := " + \
                      "func-from-reg-" + str(bt.regs[1]) + \
                      "(" + ", ".join(map(str,bt.regs[2:])) + ")"
        elif isinstance(bt, BStoreLoad):
            if bt.storing:
                output += tab + "Name[" + bt.name + "] := " + str(bt.reg)
            else:
                output += tab + str(bt.reg) + " := Name[" + bt.name + "]"
        elif isinstance(bt, BLiteral):
            output += tab + str(bt.reg) + " := Literal[" + str(bt.literal) + "]"
        elif isinstance(bt, BJump):
            output += tab + "jumpif (" + str(bt.condition_reg) + \
                      ("=" if bt.negated else "!") + \
                      "=0) to " + bt.label.id
        elif isinstance(bt, BRegMove):
            output += tab + str(bt.reg1) + " := " + str(bt.reg2)
        else:
            raise NotImplementedError(f"Haven't implemented {bt!r}")
        output += "\n"
    return output[:-1]


BAST_TYPES:list[type[Bast]] = [Bable, BCall, BStoreLoad, BLiteral, BJump,
                               BRegMove]
TABLE:dict[int,type[Bast]] = {T.AST_T_ID:T for T in BAST_TYPES}