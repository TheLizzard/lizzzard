# lizzzard
Functional language with rpython's tracing JIT compiler

# Implemented
| Feature        | Lexer | Parser | IR1 | Semantic analyser | Type checker | IR2 | Interpreter |
| :------------- | ----: | -----: | --: | ----------------: | -----------: | --: | ----------: |
| BasicOp        |     ✔ |      ✔ |   ✔ |                   |              |     |      mostly |
| Func           |     ✔ |      ✔ |     |                   |              |     |           ✔ |
| Proc           |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| Class          |     ✔ |      ✔ |     |                   |              |     |             |
| Record         |     ✔ |      ✔ |     |                   |              |     |             |
| Assginment     |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| Variables      |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| IfExpr         |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| If             |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| While          |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| Return         |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| Break/Continue |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| MatchCase      |     ✔ |      ✔ |     |                   |              |     |             |
| NonLocal       |     ✔ |      ✔ |   ✔ |                   |              |     |           ✔ |
| For            |     ✔ |      ✔ |     |                   |              |     |           - |
| Yield          |     ✔ |      ✔ |     |                   |              |     |             |
| With           |     ✔ |      ✔ |     |                   |              |     |             |
| Exceptions     |     ✔ |      ✔ |     |                   |              |     |             |
| Comprehension  |     ✔ |        |     |                   |              |     |           - |
| Partial funcs  |     ✔ |      ✔ |     |                   |              |     |           - |


| Type     | Lexer | Parser | IR1 | Interpreter |
| :------- | ----: | -----: | --: | ----------: |
| Int64    |     ✔ |      ✔ |   ✔ |           ✔ |
| Int      |     ✔ |      ✔ |     |             |
| Str      |     ✔ |      ✔ |   ✔ |           ✔ |
| None     |     ✔ |      ✔ |   ✔ |           ✔ |
| List     |     ✔ |      ✔ |   ✔ |           ✔ |
| Tuples   |     ✔ |      ✔ |     |             |
| Class    |     ✔ |      ✔ |     |             |
| Dict/Set |     ✔ |      ✔ |     |             |


# Missing from interpreter
* classes/records (still thinking about the object model)
* match/case (requires records)
* partial func syntax desugaring (needs to be implemented in the parser/bytecoder/semantic analyser)
* destructuring assignment (needs to be implemented in the bytecoder)
* booleans (instead of reusing `IntValue`)
* for loops (still thinking)
* dict/set (still thinking)

# Missing from interpreter (can be implemented later on)
* yield (still thinking about it)
* with/raise (still thinking about it)
* arbitrary size int (requires object model)
* comprehension (needs to be implemented in the parser+bytecoder)

# Interpreter speed improvements
* optimise away env and only use regs (requires semantic analyser)
* optimise away useless operations after JIT (requires more experiments/rpython stackoverflow question)