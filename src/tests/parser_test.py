from __future__ import annotations
from io import StringIO
import sys
import os
import re

ROOT:str = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from parser import Parser
from asts.ccast import *
import lexer


TEST_LITERAL_PRINT:str = """
-1
y = (0,)
x = []
x = [none]*6
x[0] = 5
x[1] = "hello world"
x[2] = (
    # comment
            5.5+0
       )
x[3] = [[1, 2],
        [3, 4]]
(x[4], x[5]) = (5,
                6)
x[4], x[5] = (5, 6)
x[4:6] = 5,6
print(x)

{}
{"x":5}
{5.5:"10"}
{y:5, 10:z}
{1}
{1,}
{1,2}
{1,2,3}
{1,2,3,4,}
{1,2,3,4, # comment
}

a, b = [5, 6]
"""
RESULT_LITERAL_PRINT:str = """
Literal[-1]
Var[y] = Op(·,· Literal[0])
Var[x] = Op([] )
Var[x] = Op(* Op([] Literal[none]), Literal[6])
Op(idx Var[x], Literal[none], Literal[0], Literal[none]) = Literal[5]
Op(idx Var[x], Literal[none], Literal[1], Literal[none]) = Literal[0'hello world']
Op(idx Var[x], Literal[none], Literal[2], Literal[none]) = Op(+ Literal[5.5], Literal[0])
Op(idx Var[x], Literal[none], Literal[3], Literal[none]) = Op([] Op([] Literal[1], Literal[2]), Op([] Literal[3], Literal[4]))
Op(·,· Op(idx Var[x], Literal[none], Literal[4], Literal[none]), Op(idx Var[x], Literal[none], Literal[5], Literal[none])) = Op(·,· Literal[5], Literal[6])
Op(·,· Op(idx Var[x], Literal[none], Literal[4], Literal[none]), Op(idx Var[x], Literal[none], Literal[5], Literal[none])) = Op(·,· Literal[5], Literal[6])
Op(idx Var[x], Literal[4], Literal[6], Literal[none]) = Op(·,· Literal[5], Literal[6])
Op(call Var[print], Var[x])
Op({:} )
Op({:} Literal[0'x']:Literal[5])
Op({:} Literal[5.5]:Literal[0'10'])
Op({:} Var[y]:Literal[5], Literal[10]:Var[z])
Op({,} Literal[1])
Op({,} Literal[1])
Op({,} Literal[1], Literal[2])
Op({,} Literal[1], Literal[2], Literal[3])
Op({,} Literal[1], Literal[2], Literal[3], Literal[4])
Op({,} Literal[1], Literal[2], Literal[3], Literal[4])
Op(·,· Var[a], Var[b]) = Op([] Literal[5], Literal[6])
"""


TEST_FUNC:str = """
f = func(x) ->
    return x+1
print(f(4))

f = proc(x) ->
    return x+1

print((func(x) -> return x+1)(4))

f = func(x:int, y:str): int ->
    nonlocal y
    return x+1
print(f(4, y="ignored arg"))

x = (func(x, y="") ->
    nonlocal y
    x += 1
    print(x))(4, "ignored arg")
print(x)

f = func(x) -> print(x+1)
f(4)

f = func(x) ->
    x += 1
    print(x)
    return
f(4)
"""
RESULT_FUNC:str = """
Var[f] = ([Var[x]] => [Return(Op(+ Var[x], Literal[1]))])
Op(call Var[print], Op(call Var[f], Literal[4]))
Var[f] = ([Var[x]] => [Return(Op(+ Var[x], Literal[1]))])
Op(call Var[print], Op(call ([Var[x]] => [Return(Op(+ Var[x], Literal[1]))]), Literal[4]))
Var[f] = ([Var[x], Var[y]] => [NonLocal(y), Return(Op(+ Var[x], Literal[1]))])
Op(call Var[print], Op(call Var[f], Literal[4], Var[y default=Literal[0'ignored arg']]))
Var[x] = Op(call ([Var[x], Var[y default=Literal[0'']]] => [NonLocal(y), Var[x] = Op(+ Var[x], Literal[1]), Op(call Var[print], Var[x])]), Literal[4], Literal[0'ignored arg'])
Op(call Var[print], Var[x])
Var[f] = ([Var[x]] => [Op(call Var[print], Op(+ Var[x], Literal[1]))])
Op(call Var[f], Literal[4])
Var[f] = ([Var[x]] => [Var[x] = Op(+ Var[x], Literal[1]), Op(call Var[print], Var[x]), Return])
Op(call Var[f], Literal[4])
"""


TEST_EXPR:str = """
x = 4+5*6+7**2 # bodmas + comment
print(5 if true else 0)
print(1 if false else 5)
print(0 if 10 if 15 else 20 else 1 if false else 2)
# print(5++--++)
print(not not (not 5) + 10)
"""
RESULT_EXPR:str = """
Var[x] = Op(+ Op(+ Literal[4], Op(* Literal[5], Literal[6])), Op(** Literal[7], Literal[2]))
Op(call Var[print], Op(if Literal[true], Literal[5], Literal[0]))
Op(call Var[print], Op(if Literal[false], Literal[1], Literal[5]))
Op(call Var[print], Op(if Op(if Literal[15], Literal[10], Literal[20]), Literal[0], Op(if Literal[false], Literal[1], Literal[2])))
# Op(call Var[print], Op(++ Op(-- Op(++ Literal[5]))))
Op(call Var[print], Op(not Op(not Op(+ Op(not Literal[5]), Literal[10]))))
"""


TEST_IF:str = """
if x      -> x = 5
else if y -> x = 8
else      -> x = 10
print(x)

if x ->
    x = 5
else if y ->
    x = 8
else ->
    x = 10
print(x)

if x ->
    x = 5
else ->
    if y ->
        x = 8
    else ->
        x = 10
print(x)
"""
RESULT_IF:str = """
If(Var[x])%
    [Var[x] = Literal[5]]%
    [If(Var[y])%
        [Var[x] = Literal[8]]%
        [Var[x] = Literal[10]%
    ]%
]
Op(call Var[print], Var[x])%

If(Var[x])%
    [Var[x] = Literal[5]]%
    [If(Var[y])%
        [Var[x] = Literal[8]]%
        [Var[x] = Literal[10]%
    ]%
]
Op(call Var[print], Var[x])%

If(Var[x])%
    [Var[x] = Literal[5]]%
    [If(Var[y])%
        [Var[x] = Literal[8]]%
        [Var[x] = Literal[10]%
    ]%
]
Op(call Var[print], Var[x])
"""


TEST_WHILE:str = """
x = 0
while x < 5 ->
    x += 1
while true ->
    break
    break 5
    continue
    continue 10
print(x)
"""
RESULT_WHILE:str = """
Var[x] = Literal[0]
While(Op(< Var[x], Literal[5]))[Var[x] = Op(+ Var[x], Literal[1])]
While(Literal[true])[Break(1), Break(5), Continue(1), Continue(10)]
Op(call Var[print], Var[x])
"""


TEST_WITH:str = """
with open(r"/path/to/file.ext", "r") as file ->
    print(0)
except FileNotFoundError|Exception as error ->
    print(1)
except BaseException ->
    print(2)
except ->
    print(3)
finally ->
    print(4)
else ->
    print(5)
"""
RESULT_WITH:str = """
With%
    (Var[file] = Op(call Var[open], Literal[1'/path/to/file.ext'], Literal[0'r']))%
        [Op(call Var[print], Literal[0])]%
    [%
        ([Var[FileNotFoundError], Var[Exception]], Var[error], [%
            Op(call Var[print], Literal[1])%
        ]), %
        ([Var[BaseException]], None, [%
            Op(call Var[print], Literal[2])%
        ]), %
        ([], None, [%
            Op(call Var[print], Literal[3])%
        ])%
    ]%
    [Op(call Var[print], Literal[4])]%
    [Op(call Var[print], Literal[5])]
"""


TEST_MACRO:str = """
hell_word = $import("__hello__")
"""


TEST_GENERATOR:str = """
f = func(x:int) ->
    while true ->
        x += 1
        yield x
        yield 0

for i in f(5) ->
    print(i)
    if i == 7 ->
        break
        break 1
"""
RESULT_GENERATOR:str = """
Var[f] = ([Var[x]] => [While(Literal[true])[Var[x] = Op(+ Var[x], Literal[1]), Yield(Var[x]), Yield(Literal[0])]])
For(Var[i] in Op(call Var[f], Literal[5]))[Op(call Var[print], Var[i]), If(Op(== Var[i], Literal[7]))[Break(1), Break(1)][]]
"""


TEST_CLASS:str = r"""
A = class ->
    __init__ = func(self) ->
        self.x = 1
        self.bar()

    foo = func() ->
        print("called static method foo")

    bar = func(self) ->
        msg = "called method bar"
        print(msg)

B = class(A) -> ...

__add__ = func(x:A, y:A): B ->
    return x.x + y.x
"""
RESULT_CLASS:str = """
Var[A] = Class()[%
    Var[__init__] = ([Var[self]] => [%
        Op(. Var[self], Var[x]) = Literal[1], %
        Op(call Op(. Var[self], Var[bar]))%
                                    ]), %
    Var[foo] = ([] => [%
        Op(call Var[print], Literal[0'called static method foo'])%
                      ]), %
    Var[bar] = ([Var[self]] => [%
        Var[msg] = Literal[0'called method bar'], %
        Op(call Var[print], Var[msg])%
                               ])%
]
Var[B] = Class(Var[A])[]
Var[__add__] = ([Var[x], Var[y]] => [%
    Return(Op(+ Op(. Var[x], Var[x]), Op(. Var[y], Var[x])))%
                                    ])
"""


TEST_MULTI_ASSIGN:str = """
a = b = c = 5
x = y = 1
"""
RESULT_MULTI_ASSIGN:str = """
[Var[a], Var[b], Var[c]] = Literal[5]
[Var[x], Var[y]] = Literal[1]
"""


if __name__ == "__main__":
    # lexer.DEBUG_THROW:bool = True
    codes = (TEST_LITERAL_PRINT, TEST_EXPR, TEST_IF, TEST_WHILE,
             TEST_FUNC, TEST_WITH, TEST_GENERATOR, TEST_CLASS,
             TEST_MULTI_ASSIGN)
    expecteds = (RESULT_LITERAL_PRINT, RESULT_EXPR, RESULT_IF, RESULT_WHILE,
                 RESULT_FUNC, RESULT_WITH, RESULT_GENERATOR, RESULT_CLASS,
                 RESULT_MULTI_ASSIGN)
    # codes = ()
    for code, expected in zip(codes, expecteds):
        ast:Body = Parser(lexer.Tokeniser(StringIO(code))).read()
        if ast is None:
            raise RuntimeError("Test failed: parsing program")
        result:str = "\n".join(map(repr, ast))
        expected, result = expected.strip("\n"), result.strip("\n")
        expected:str = expected.rstrip("%")
        while "%\n" in expected:
            expected:str = re.sub("%\n *", "", expected)
        while re.match(".*\n#[^\n]*\n.*", expected, re.DOTALL):
            expected:str = re.sub("\n#[^\n]*\n", "\n", expected)
        if result != expected:
            raise RuntimeError("Test failed: ast not correct")


if __name__ == "__main__":
    # lexer.USING_COLON:bool = False
    # lexer.DEBUG_THROW:bool = True
    # lexer.DEBUG_READ:bool = True
    TEST:str = r"""
CONST = 10

match 11 ->
    case CONST ->
        5
    case Int(?x) ->
        10
    case ? ->
        15
id = ?
odd = 2 * ? + 1
partial = f(?:Int, 5:Int):List[Int]
""".removeprefix("\n").removesuffix("\n")
    t = lexer.Tokeniser(StringIO(TEST))
    p:Parser = Parser(t)
    ast:Body = p.read()
    if ast is None:
        print("\x1b[96mNo ast due to previous error\x1b[0m")
    elif ast:
        for line in ast:
            print(line)
    else:
        print("\x1b[96mEmpty ast\x1b[0m")