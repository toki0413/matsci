# Brooks-Lint — Full Sweep Report

Mode: Full Sweep | Scope: `/workspace/agent`（huginn 1,057 + tests 593 个 Python 文件）
Config: 无 `.brooks-lint.yaml`，未禁用任何风险项，未忽略任何路径

**Health Score:** 0/100（地板值）

本次 sweep 自动修复了 7 处 Critical 级代码衰减（长方法 / 深嵌套）、9 组重复实现、2 处 BOM 语法污染、14 条测试命名/断言问题，并新增 11 条行为护栏测试；剩余 1,568 条 Critical 集中在 1,149 行的 `streaming.chat`、2,088 行的 `rcb_step2._step2_execute` 等研究编排单体上，属于必须逐函数人工重构、无法机械改写的结构性问题。

---

## Dimension Summary

| Dimension | Scanned | Safe Applied | Extended Applied | Reverted | Residual |
|-----------|---------|--------------|------------------|----------|----------|
| Review (R1–R6) | 5,061 | 20 | 0 | 0 | 5,041 |
| Test (T1–T6) | 374 | 14 | 0 | 0 | 360 |
| Debt | 8 | 0 | 0 | 0 | 0（8 条全部判定假阳性） |
| Audit | 949 条模块级 import 边 | 0 | 0 | 0 | 3 条待人工判定 |

> 扫描口径：R1 4,893（长方法 / 深嵌套）、R3 5 组重复、R4 1、R5 160、R6 2（人工复核的超时语义）；T1 374（Assertion Roulette / Ambiguous test name）。R5 报出的 40 条“依赖环”经严格模块级 import 图复核为 **0 环**，全部为假阳性，不计入 Residual。

---

## Iteration History

Round 1: mixed — 5,437 条候选（Critical 1,580 / Warning 467 / Suggestion 3,390），其中 1,676 个文件里可安全机械改写的有 36 条。
Round 2: 重扫全部 `huginn/` + `tests/`（含改动文件、同模块、静态消费者）— 5,433 条（Critical 1,568 / Warning 461 / Suggestion 3,404），净减 4 条；新增的 14 条 Suggestion 来自抽取出的辅助函数本身，属预期。
Stopped at: no outstanding automatable criticals（剩余 Critical 全部是跨 100+ 行的编排单体，改写需要产品语义判断，非本轮 Safe/Extended-Safe 可覆盖）。

---

## Fix Log

| # | File | Lines | Risk | Outcome | Change |
|---|------|-------|------|---------|--------|
| 1 | huginn/tools/code_tool.py | 超时分支 | R6 | applied | 补 `timed_out` / `timeout` / `_timeout_error`，让超时不再伪装成空 stderr 的硬失败 |
| 2 | huginn/tools/adapter.py | `_timeout_failure_hint` | R6 | applied | 新增 `ErrorKind.TIMEOUT` 分流与超时提示 |
| 3 | huginn/tools/lean_tool.py | 342–436 | R1 | applied | `_do_auto_verify` 163 行 / nest 15 → 32 行 / nest 3（12 个 `elif` 分支折叠为 `_SIMPLE_VERIFIERS` 表 + 2 个专属方法） |
| 4 | huginn/validation/kg_feedback.py | 22–120 | R1 | applied | `write_validation_to_kg` nest 7 → 2（拆 `_write_result_constraints` / `_link_if_present`，`continue` 早退替代嵌套 if） |
| 5 | huginn/metacog/self_model.py | 403–516 | R1 | applied | `extract_step_result_from_audit` nest 7 → 2（拆 4 个 helper，FIFO 配对语义不变） |
| 6 | huginn/agent/core.py | 884–1000 | R1 | applied | `_pre_model_hook` nest 8 / 97 行 → 21 行 / nest 2（4 个模块级 hint 收集器） |
| 7 | huginn/tools/sim/lammps_tool.py | 1118–1342 | R1 | applied | `_parse_trajectory_python` nest 7 / 225 行 → nest 4 / 54 行（dump 解析 + 5 个分析器） |
| 8 | huginn/tools/sim/lammps_tool.py | 748–831 | R1 | applied | `_parse_log` nest 6 / 78 行 → nest 3（列 / 行 / 转置 / 能量 / 诊断 5 个 helper） |
| 9 | huginn/tools/sim/lammps_tool.py | 312–670 | R1 | applied | `call` 295 行 → 40 行编排器 + 9 个 helper（派发 / 可执行解析 / 输入准备 / 命令拼装 / 重试循环 / 尝试评估 / 结果组装 / 审计兜底） |
| 10 | huginn/research/_scalars.py | 新建 | R3 | applied | 收敛 3 处重复的标量强转 |
| 11 | huginn/tools/sim/_packing.py | 新建 | R3 | applied | 收敛 3 处重复的 packing 数据加载 |
| 12 | huginn/tools/sim/_async_job.py | 新建 | R3 | applied | 收敛 2 处重复的后台作业提交 |
| 13 | huginn/tools/_llm.py | 新建 | R3 | applied | 收敛 2 处重复的 LLM 调用封装 |
| 14 | huginn/metacog/_json.py | 新建 | R3 | applied | 收敛 2 处重复的 JSON 解析 |
| 15 | huginn/tools/_embed.py | 新建 | R3 | applied | 收敛 2 处重复的惰性 embedding |
| 16 | huginn/utils/numeric_targets.py | 新建 | R3 | applied | 建立唯一数值目标事实源，`hint_coordinator` / `rcb_utils` 改为 re-export（保留公开 API） |
| 17 | huginn/tools/sim/_validate.py | 新建 | R3 | applied | 收敛 2 处重复的输入校验 |
| 18 | huginn/runtime/trace_context.py | 1 | R4 | applied | 剥离 U+FEFF BOM，修复 `ast.parse` 失败 |
| 19 | tests/test_rcb_path_rules.py | 1 | R4 | applied | 剥离 U+FEFF BOM |
| 20 | tests/test_toolchain_e2e.py | 7 个类 | T1 | applied | 7 个同名 `test_chain` 改为描述性名称（Ambiguous test name ×7、Assertion Roulette ×7） |
| 21 | tests/test_kg_feedback.py | 新建 | T4 | applied | 4 条行为护栏覆盖抽取后的 KG 写入语义（节点去重 / 边集合 / 空取） |
| 22 | tests/test_self_model_audit.py | 新建 | T4 | applied | 7 条行为护栏覆盖 call/result/error 配对、duration、孤儿调用、坏行容错 |

**验证**：`ruff check` 全部改动文件 `All checks passed!`；`151 passed, 5 skipped`（code_tool / kg_feedback / self_model_audit / toolchain_e2e / prompt_cache / engine_decomposed / residual_risks）；`119 passed, 18 skipped`（lammps + trajectory）；lean 系列 `134 passed`。唯一失败 `test_mock_lean_hpc.py::test_connect_paramiko` 为环境缺 `paramiko`，与本轮改动无关；`test_provenance.py::test_run_persists_provenance_with_tool_chain` 经 `git worktree` 在未改动 HEAD 上复现，判定为既有失败。

---

## Health Score Delta

Before: 0/100 → After: 0/100

评分公式为 100 − 15×Critical − 5×Warning − 1×Suggestion（下限 0）。1,568 条 Critical 使分数在修复前后都触地板，因此该指标对本次改动不敏感；更有意义的增量是：Critical 1,580 → 1,568、nest>5 的函数 64 → 59、重复实现组 10 → 5、未覆盖模块 87 → 84、函数总数 11,528 → 11,561（抽取 helper 带来的自然增长）。

---

## Residual Items (5,401 not applied)

### 🔴 Critical

**Long Method / Deep Nesting — `streaming.chat` 单体**
Symptom: `huginn/agent/streaming.py:1316`，1,145 行，nest 9。
Source: *Refactoring* — Long Method；*Clean Code* — Functions Should Do One Thing。
Consequence: 任何一处 SSE 事件格式调整都要在千行函数里定位，回归面覆盖全部流式输出路径。
Remedy: 按事件类型（token / tool_call / tool_result / error / done）拆成独立生成器，主函数只做 dispatch。
Not applied because: 公开流式 API，无针对分支级行为的测试兜底，blast radius 覆盖所有前端会话。

**Long Method — `rcb_step2._step2_execute`**
Symptom: `huginn/cli/rcb_step2.py:91`，2,088 行，nest 8。
Source: *Working Effectively with Legacy Code* — God Class 内的巨函数。
Consequence: 6-step 研究流程的第二步无法单独测试，任何改动只能靠端到端跑通验证。
Remedy: 按 pipeline 阶段（准备 / 执行 / 校验 / 汇总）拆成 4–6 个方法，中间产物显式传递。
Not applied because: 无单元测试覆盖，且是书生实战链路的主入口，重构需配套录制回放。

**Deep Nesting — `ws_helpers._stream_agent_response`**
Symptom: `huginn/routes/ws_helpers.py:245`，510 行，nest 8。
Source: *A Philosophy of Software Design* — 深嵌套掩盖控制流。
Consequence: WebSocket 错误分支相互遮蔽，异常路径难以审计（安全相关）。
Remedy: 用 early-return 取代 8 层 if，把消息序列化抽出。
Not applied because: 鉴权 / 心跳 / 背压耦合在嵌套里，需安全评审。

**Deep Nesting — `literature._do_citation_graph`**
Symptom: `huginn/tools/literature/tool.py:1503`，235 行，nest 8。
Source: *Refactoring* — Extract Method。
Consequence: 引用图构建的容错分支不可测，数据源变化时容易静默吞错。
Remedy: 拆 `_collect_nodes` / `_dedupe_edges` / `_render_graph`。
Not applied because: 该文件本轮已在 R3 改动中，避免同文件二次大改叠加风险。

**Deep Nesting — `eval.eval_run`**
Symptom: `huginn/routes/eval.py:49`，179 行，nest 8。
Source: *Clean Code* — 控制流扁平化。
Consequence: 评测失败原因难以定位到具体子步骤。
Remedy: 抽出 metric 计算与结果持久化两段。
Not applied because: 与 HTTP 契约耦合，需对照 `contract_audit` 快照。

**Deep Nesting — `symmetry_tool._verify_group`**
Symptom: `huginn/tools/sci/symmetry_tool.py:262`，150 行，nest 8。
Source: *Refactoring* — Decompose Conditional。
Consequence: 点群校验的多级 fallback 互斥关系不可见。
Remedy: 把每类群检验注册成表，主函数遍历。
Not applied because: 科学正确性依赖校验顺序，需领域测试确认。

**Long Method — `rcb_runner.run` / `contract_audit.build_http_field_contract` / `orchestrator.run`**
Symptom: 1,090 行 nest 7 / 170 行 nest 7 / 88 行 nest 7。
Source: *The Mythical Man-Month* — 概念完整性被单体侵蚀。
Consequence: 研究主循环与契约快照生成无法独立演进。
Remedy: 各自按阶段拆分（runner：6 step；contract：schema 收集 / 校验 / 落盘；orchestrator：任务派发 / 聚合）。
Not applied because: 三者都是对外入口，重构需同步更新快照与集成测试。

**Long Method — `engine_reflect._learn` / `code_act_loop.run_code_act_turn` / `skills.base.execute` / `vasp_tool._run_vasp` / `rcb/fallback.generate_fallback_figures`**
Symptom: 738 / 347 / 184 / 179 / 180 行，nest 6。
Source: *Clean Code* — 函数应只做一层抽象。
Consequence: 认知闭环的学习阶段与执行后端细节混在一起，调试需全量阅读。
Remedy: 按“输入归一化 / 决策 / 执行 / 记录”四段拆分。
Not applied because: 单一职责边界需产品语义确认，且均无细粒度测试。

**Assertion Roulette / Ambiguous test name — 360 条**
Symptom: 210 个测试文件中共 360 个测试函数包含多条无 message 的断言，或存在同名测试。
Source: *xUnit Test Patterns* — Assertion Roulette；*The Art of Unit Testing* — 测试可诊断性。
Consequence: 断言失败时只能看到行号，无法从输出判断破坏的是哪条契约。
Remedy: 给每条断言补 message，或按断言主题拆分测试。
Not applied because: 纯 message 级机械改写会产出约 2,000 行无行为收益的 diff，且跨 210 个文件，超过 Extended-Safe 的 ≤5 文件上限；保留供人工按模块批量处理。

**Duplicated Implementation — 5 组**
Symptom: 测试内 `_install_auditor` / `_stub_heavy_calls` / `_make_engine` / `_serialize` / `_no_rate_limit` 等替身构造重复；`workflows/mcp_export` 与 `capabilities/mcp_export` 的 `_http_app` 为不同后端；`autoloop` 5 处 `__setattr__` 是组合式转发协议；`security` 3 处 `__init__` 属不同物理模型。
Source: *The Pragmatic Programmer* — DRY（但需区分“知识重复”与“形状相同”）。
Consequence: 若强行合并会把测试夹具耦合到生产模块。
Remedy: 仅对同包内断言完全一致的替身抽取共享 fixture。
Not applied because: 逐组复核后均为形状相似而非知识重复，合并属过度抽象。

### 🟡 Warning

**Module Dependency — 3 条可疑边**
Symptom: `tools/recall_context_tool.py` 模块级 `from huginn import metacog`（`tools → 包根`）；`runtime → autoloop`、`validation → autoloop` 两条上行依赖。
Source: *A Philosophy of Software Design* — 依赖应指向稳定抽象。
Consequence: 底层包反向依赖高层编排，未来拆分包时会出现循环。
Remedy: `recall_context_tool` 改为惰性导入；`runtime/checkpoint` 与 `validation/research` 通过 Protocol 注入。
Not applied because: 实测无环、无分层倒置（严格 import 图 949 条边 0 环），属于可接受的局部反向调用，改动收益低于风险。

### 🟢 Suggestion

**Asset Inventory — 87 → 84 个无直接测试模块**
Symptom: `huginn.academic.*`、`huginn.plugins.*`、`huginn.workflows.templates_*`、`huginn.snapshot.selfcheck` 等模块无同名测试。
Source: *Working Effectively with Legacy Code* — 无测试的代码是遗留代码。
Consequence: 这些模块的行为变更不会被 CI 捕获。
Remedy: 优先为 `plugins` / `workflows.templates_*` 补契约测试。
Not applied because: 新建测试超出“修复既有发现”的范围，且需要领域输入样例。

---

## Summary

- Total findings detected: 5,437（Round 1）→ 5,433（Round 2 重扫）
- Fixed this sweep: 34（20 Review + 14 Test）
- Residual (needs human review): 5,401
- Unresolvable (3-retry exhausted): 0

自动化可覆盖的衰减已处理完：深嵌套的 3 个最大单体函数（`lean_tool` 15 层、`lammps` 7 层 ×2）、4 处 Critical 嵌套、9 组重复实现、2 处 BOM 语法污染、7 个同名测试已全部清零，且全部通过对应测试批次与 `ruff`。剩余 1,568 条 Critical 并非“未发现”，而是 1,100 个文件的研究框架里大量 50–2,000 行的编排函数——它们需要按产品语义逐个拆分并配套回放测试，属于必须由人主导的重构，不应由 sweep 机械改写。建议下一步以 `streaming.chat`、`rcb_step2._step2_execute`、`ws_helpers._stream_agent_response` 三个入口为起点，先补分支级测试再拆函数。
