# lizzzard
Functional language with rpython's tracing JIT compiler

# Implemented
| Feature        | Lexer | Parser | IR1 | Semantic analyser | IR2 | Interpreter |
| :------------- | ----: | -----: | --: | ----------------: | --: | ----------: |
| BasicOp        |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ? |
| Func           |     ✔ |      ✔ |     |                   |   ✔ |           - |
| Proc           |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Class          |     ✔ |      ✔ |     |                   |     |             |
| Record         |     ✔ |      ✔ |     |                   |     |             |
| Assginment     |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Variables      |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| IfExpr         |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| If             |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| While          |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Return         |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| Break/Continue |     ✔ |      ✔ |   ✔ |                   |   ✔ |           ✔ |
| MatchCase      |     ✔ |      ✔ |     |                   |     |             |
| NonLocal       |     ✔ |      ✔ |   ✔ |                 ✔ |   ✔ |           ✔ |
| For            |     ✔ |      ✔ |     |                   |   - |           - |
| Yield          |     ✔ |      ✔ |     |                   |     |             |
| With           |     ✔ |      ✔ |     |                   |     |             |
| Exceptions     |     ✔ |      ✔ |     |                   |     |             |
| Comprehension  |     ✔ |        |     |                   |   - |           - |
| Partial funcs  |     ✔ |      ✔ |   - |                   |   - |           - |


| Type     | Lexer | Parser | IR | Interpreter |
| :------- | ----: | -----: | -: | ----------: |
| Int64    |     ✔ |      ✔ |  ✔ |           ✔ |
| Int      |     ✔ |      ✔ |    |             |
| Str      |     ✔ |      ✔ |  ✔ |           ✔ |
| None     |     ✔ |      ✔ |  ✔ |           ✔ |
| List     |     ✔ |      ✔ |  ✔ |           ✔ |
| Tuples   |     ✔ |      ✔ |    |             |
| Class    |     ✔ |      ✔ |    |             |
| Dict/Set |     ✔ |      ✔ |    |             |


# Missing from interpreter
* classes/records (still thinking about the object model)
* match/case (requires records)
* destructuring assignment (requires object model)
* booleans (instead of reusing `IntValue`)
* for loops (requires object model)
* dict/set (requires object model - will be implemented inside lizzzard from scratch)

# Missing from interpreter (can be implemented later on)
* yield (still thinking about it)
* with/raise (still thinking about it)
* arbitrary size int (requires object model)
* comprehension (needs to be implemented in the parser+bytecoder)

# Interpreter speed improvements
* a bytecode optimiser should be able to cut the number of instructions by ~15% but the speed increase might be negligible
* optimise away useless operations after JIT (requires more experiments/rpython stackoverflow question)