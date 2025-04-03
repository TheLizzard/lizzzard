# lizzzard
Functional language with rpython's tracing JIT compiler

# How to run
1. Download the [RPython](https://github.com/pypy/pypy/tree/main/rpython) directory from PyPy's GitHub into `src/frontend/rpython`
2. `cd` into `src/frontend` and run `./compile.run`
3. Use `python3 src/bytecoder.py <file.lizz>` to compile a source file into `.clizz` bytecode
4. Run `src/frontend/lizzzard <file.clizz>` (executable should have been created by `compile.run`)

# Implemented
| Feature        | Lexer | Parser | IR1 | Semantic analyser | IR2 | Interpreter |
| :------------- | ----: | -----: | --: | ----------------: | --: | ----------: |
| BasicOp        |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Func           |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Class          |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Assginment     |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Variables      |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| IfExpr         |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| If             |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| While          |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Return         |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Break/Continue |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| MatchCase      |     ✔ |      ✔ |     |                   |     |             |
| NonLocal       |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| For            |     ✔ |      ✔ |     |                   |   ✔ |           ✔ |
| Yield          |     ✔ |      ✔ |     |                   |     |             |
| With           |     ✔ |      ✔ |     |                   |     |             |
| Exceptions     |     ✔ |      ✔ |     |                   |     |             |
| Comprehension  |     ✔ |        |     |                   |     |             |
| Partial funcs  |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |


| Type     | Lexer | Parser | IR | Interpreter |
| :------- | ----: | -----: | -: | ----------: |
| Int64    |     ✔ |      ✔ |  ✔ |           ✔ |
| Int      |     ✔ |      ✔ |    |             |
| Str      |     ✔ |      ✔ |  ✔ |           ✔ |
| None     |     ✔ |      ✔ |  ✔ |           ✔ |
| List     |     ✔ |      ✔ |  ✔ |           ✔ |
| Tuples   |     ✔ |      ✔ |    |             |
| Class    |     ✔ |      ✔ |  ✔ |           ✔ |
| Dict/Set |     ✔ |      ✔ |    |             |


# Missing from interpreter
* match/case (requires records)
* for loops (requires more designing)
* dict/set

# Missing from interpreter (can be implemented later on)
* yield (still thinking about it)
* with/raise (still thinking about it)
* comprehension (needs to be implemented in the parser)
* arbitrary size int

# Interpreter speed improvements
* a bytecode optimiser should be able to cut the number of instructions by ~15% but the speed increase might be negligible
* an environment optimiser that clears environment variables after their last use. This might produce a 2x speed up depending on the benchmark
* optimise away useless operations after JIT (pypy issue #5166)