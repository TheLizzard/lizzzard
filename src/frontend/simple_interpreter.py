from python3.rpython_compat import JitDriver
from python3.dict_compat import * # Can be replaced with `Dict = lambda:{}`
from bcast import *
BLabel, BListStore = Bable, BDotList # Renaming so it makes more sense


class Value:
    _immutable_fields_ = []
    __slots__ = ()

class IntValue(Value):
    _immutable_fields_ = ["value"]
    __slots__ = "value"
    def __init__(self, value):
        self.value = value

class ListValue(Value):
    _immutable_fields_ = ["array"]
    __slots__ = "array"
    def __init__(self):
        self.array = []


jitdriver = JitDriver(greens=["pc","bytecode","jump_table"], reds=["env"])

def interpret(bytecode):
    # Create a jump table from all of the labels
    jump_table = hint(Dict(), promote=True)
    for i, bt in enumerate(bytecode):
        if isinstance(bt, BLabel):
            jump_table[promote_unicode(bt.id)] = hint(IntValue(i), promote=True)

    # Main interpreter loop
    env = [None]*8
    pc = 0
    while pc < len(bytecode):
        jitdriver.jit_merge_point(env=env, pc=pc, bytecode=bytecode, jump_table=jump_table)
        instruction = bytecode[pc]
        pc += 1

        if isinstance(instruction, BListStore):
            # Bytecode to store a value in a list
            obj, idx = env[instruction.obj_reg], instruction.attr
            assert isinstance(obj, ListValue), "TypeError"
            # Tell optimiser that vals and env don't occupy the same space
            assert obj.array is not env # has no effect on the optimised trace
            while len(obj.array) <= idx: # extend the list so that we can store in it
                obj.array.append(None)
            assert instruction.reg < len(env), "BoundsCheck"
            assert idx < hint(len(obj.array), promote=True), "BoundsCheck"
            obj.array[idx] = env[instruction.reg] # store into array

        elif isinstance(instruction, BLiteral):
            # Loads a literal/new list into env
            data = instruction.literal
            assert instruction.reg < len(env), "BoundsCheck"
            if instruction.type == BLiteral.CLASS_T:
                env[instruction.reg] = ListValue() # create a new ListValue
            elif instruction.type == BLiteral.INT_T:
                assert isinstance(data, BLiteralInt), "TypeError"
                env[instruction.reg] = IntValue(data.value) # create a new IntValue

        elif isinstance(instruction, BJump):
            # If bool(value) -> jump to label
            value = env[instruction.condition_reg]
            assert isinstance(value, IntValue), "TypeError"
            if value.value:
                pc = hint(jump_table[instruction.label].value, promote=True)
                jitdriver.can_enter_jit(env=env, pc=pc, bytecode=bytecode, jump_table=jump_table)

def main(filepath=None):
    idx = 1 # If this is 0, the optimised trace is much longer than if it's 1
    bytecode = [
                 BLiteral(EMPTY_ERR, 0, BLiteralInt(1), BLiteral.INT_T), # env[0] = IntValue(1)
                 BLiteral(EMPTY_ERR, 5, BLiteralClass([],u"",u""), BLiteral.CLASS_T), # env[5] = new list
                 BLabel(u"while_start"),            # label
                 BListStore(EMPTY_ERR, 5, idx, 1, True),       # env[5].store(idx, val=env[0])
                 BListStore(EMPTY_ERR, 5, idx, 2, True),       # env[5].store(idx, value=NULL_PTR)
                 BJump(EMPTY_ERR, u"while_start", 0, False),   # if env[7] -> jump to label
               ]
    interpret(bytecode)