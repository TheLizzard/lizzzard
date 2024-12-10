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


| Type  | Lexer | Parser | IR1 | Interpreter |
| :---- | ----: | -----: | --: | ----------: |
| Int64 |     ✔ |      ✔ |   ✔ |           ✔ |
| Int   |     ✔ |      ✔ |     |             |
| Str   |     ✔ |      ✔ |   ✔ |           ✔ |
| None  |     ✔ |      ✔ |   ✔ |           ✔ |
| List  |     ✔ |      ✔ |   ✔ |           ✔ |
| Class |     ✔ |      ✔ |     |             |