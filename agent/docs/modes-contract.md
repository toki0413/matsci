# Mode/Phase 契约 (prompt 面)

自动生成: `python -m huginn.cli.config_audit --modes --out docs/modes-contract.md`.
登记 prompt 面的 mode 系统: **mode** (MODE_INSTRUCTIONS, agent 顶层行为) 与 **phase** (PHASE_PROMPTS + PHASE_BUDGETS, 研究流程阶段)。`budget` 是该 phase 的 工具调用预算 (max_calls); `head` 是 phase 提示头首行; `g51` 是 v6 结构关系语义对齐补充所覆盖的 phase。

### Mode

| mode | 行为说明 |
|---|---|
| chat | Conversational assistance. Answer directly; avoid heavy simulation |
| research | Systematic research mode. Cite literature for claims, quantify |
| extreme | Extreme mode. Long-horizon task at maximum capability: unlock all |
| code | Code-act mode. Solve tasks by writing and executing code in the |
| fusion | Fusion mode. Integrate evidence across simulation, experiment, and |

### Phase

| phase | 枚举名 | 预算 | 提示头首行 |
|---|---|---|---|
| literature | LITERATURE | 50 | ## Current phase: Literature Review |
| hypothesis | HYPOTHESIS | 30 | ## Current phase: Hypothesis Formation |
| planning | PLANNING | 30 | ## Current phase: Experiment Planning |
| execution | EXECUTION | 300 | ## Current phase: Execution |
| validation | VALIDATION | 100 | ## Current phase: Validation & Analysis |
| reporting | REPORTING | 20 | ## Current phase: Reporting |
| open | OPEN | 500 |  |


G51 结构关系补充覆盖 phase: `hypothesis`, `validation`
