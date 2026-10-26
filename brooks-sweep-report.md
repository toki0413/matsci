# Brooks-Lint — Full Sweep Report

Mode: Full Sweep（Round 1–4，已收敛）| Scope: `/workspace/agent`（huginn 1,065 + tests 595 = 1,688 个 Python 文件）
Config: 无 `.brooks-lint.yaml`，未禁用任何风险项，未忽略任何路径

**Health Score:** 0/100（地板值）——扣分项只剩 Warning / Suggestion 体量，**R1 Critical 已归零**。

本次 sweep 分四轮：Round 1–2 修掉 34 条可机械改写的衰减（9 组重复实现、2 处 BOM、14 条测试命名/断言、2 处超时语义）；Round 3–4 把 Round 2 判定为「需人工逐函数重构」的 1,568 条 Critical（>50 行 / nest>5）全部拆完——593 个 Python 文件被重写，仓库内最长函数从 2,088 行降到 **50 行**，`nest>5` 的函数 **0 个**。

---

## Dimension Summary

| Dimension | Scanned | Safe Applied | Extended Applied | Reverted | Residual |
|-----------|---------|--------------|------------------|----------|----------|
| Review (R1–R6) | 5,061 个函数级发现 | 20 | 0 | 0 | R1 Critical **0**；Warning / Suggestion 未重判 |
| Test (T1–T6) | 374 | 14 | 0 | 0 | 360（Assertion Roulette / Ambiguous test name） |
| Debt | 8 | 0 | 0 | 0 | 0（8 条全部判定假阳性） |
| Audit | 949 条模块级 import 边 | 0 | 0 | 0 | 3 条待人工判定（0 环） |

> 扫描口径：R1 = 函数 > 50 行或嵌套 > 5 为 Critical，20–50 行或嵌套 4–5 为 Warning。T1 = 同一测试内多条无 message 断言，或测试名重复。R5 报出的 40 条「依赖环」经严格模块级 import 图复核为 **0 环**，全部假阳性。

---

## Iteration History

Round 1: mixed — 5,437 条候选（Critical 1,580 / Warning 467 / Suggestion 3,390），其中 1,676 个文件里可安全机械改写的有 36 条。
Round 2: 重扫全部 `huginn/` + `tests/`（含改动文件、同模块、静态消费者）— 5,433 条（Critical 1,568 / Warning 461 / Suggestion 3,404），净减 4 条；新增的 14 条 Suggestion 来自抽取出的辅助函数本身，属预期。判定剩余 Critical 均需逐函数人工重构。
Round 3–4: 逐函数拆分 — 对 `huginn/` + `tests/` 全量做 AST 度量（`lines>50 or nest>5`），按文件密度排序，逐个把单体函数拆成「主编排器 + 领域 helper」，每拆一个立即用 A/B 差分脚本比对旧实现与新实现的输出（含异常类型与文案、RNG 调用序列、SQL 写入顺序），再跑该文件对应的测试批次。共重写 593 个 Python 文件（+141,354 / −86,513 行），其中 120 个文件改动超过 500 行。
Stopped at: 全仓 AST 复扫 `Critical = 0`，最长函数 50 行，`nest>5 = 0`（1,688 个文件、29,756 个函数，含嵌套 helper）。

---

## Round 3–4 拆分结果（原报告点名的入口全部落地）

| 函数 | 拆分前 | 拆分后 |
|------|--------|--------|
| `cli/rcb_step2._step2_execute` | 2,088 行 / nest 8 | 18 行（按 pipeline 阶段拆 6 段） |
| `agent/streaming.chat` | 1,145 行 / nest 9 | 44 行（按 SSE 事件类型拆生成器） |
| `cli/rcb_runner.run` | 1,090 行 / nest 7 | 38 行（6 step 编排器） |
| `autoloop/engine_reflect._learn` | 738 行 / nest 6 | 48 行 |
| `routes/ws_helpers._stream_agent_response` | 510 行 / nest 8 | 49 行 |
| `tools/literature/tool._do_citation_graph` | 235 行 / nest 8 | 45 行 |
| `routes/eval.eval_run` | 179 行 / nest 8 | 47 行 |
| `cli/contract_audit.build_http_field_contract` | 170 行 / nest 7 | 42 行 |
| `tools/sci/symmetry_tool._verify_group` | 150 行 / nest 8 | 35 行 |
| `autoloop/cognitive_loop.run` | 巨型单体 | 18 行 |

拆分纪律（每条都遵守）：

1. **等价性先于重构** — 每个函数拆分前先用 `git show HEAD:` 取旧实现，写 A/B 差分脚本枚举输入，比对输出逐位一致（浮点用 ulp / 绝对偏差，含 `float("inf")`、`nan`、异常类型与文案）。
2. **不可见的契约要保住** — `inspect.getsource` 断言、AST 静态扫描（契约审计 golden）、测试替身 `MagicMock(spec=...)` 只绑定单个方法的场景，全部按显式类名调用 helper，不改成 `self._helper()`。
3. **先 format 再测量** — `ruff format` 会把长参数列表展开成每行一个参数，多个函数在 format 后重新越过 50 行，均已二次拆分到达标。
4. **安全/契约面零改动** — `routes/*.py` 的路径、方法、`response_model`、`Depends`/`Security`、Pydantic 字段逐字保留（`app.openapi()` 与 HEAD 逐字节一致，530,528 bytes）；`workflows/templates_*.py` 的 stage 序列、`security/audit.py` 的签名/落盘顺序、`plugins/asd_ste100` 的规则文案均逐字保留。

---

## Fix Log

| # | File | Risk | Outcome | Change |
|---|------|------|---------|--------|
| 1 | `tools/code_tool.py` | R6 | applied | 补 `timed_out` / `timeout` / `_timeout_error`，超时不再伪装成空 stderr 的硬失败 |
| 2 | `tools/adapter.py` | R6 | applied | 新增 `ErrorKind.TIMEOUT` 分流与超时提示 |
| 3 | `tools/lean_tool.py` | R1 | applied | `_do_auto_verify` 163 行 / nest 15 → 32 行 / nest 3（12 个 `elif` 折叠为 `_SIMPLE_VERIFIERS` 表） |
| 4 | `validation/kg_feedback.py` | R1 | applied | `write_validation_to_kg` nest 7 → 2 |
| 5 | `metacog/self_model.py` | R1 | applied | `extract_step_result_from_audit` nest 7 → 2（FIFO 配对语义不变） |
| 6 | `agent/core.py` | R1 | applied | `_pre_model_hook` nest 8 / 97 行 → 21 行 / nest 2 |
| 7 | `tools/sim/lammps_tool.py` | R1 | applied | `_parse_trajectory_python` nest 7 / 225 行 → nest 4 / 54 行 |
| 8 | `tools/sim/lammps_tool.py` | R1 | applied | `_parse_log` nest 6 / 78 行 → nest 3 |
| 9 | `tools/sim/lammps_tool.py` | R1 | applied | `call` 295 行 → 40 行编排器 + 9 个 helper |
| 10–17 | `research/_scalars.py`、`tools/sim/_packing.py`、`tools/sim/_async_job.py`、`tools/_llm.py`、`metacog/_json.py`、`tools/_embed.py`、`utils/numeric_targets.py`、`tools/sim/_validate.py` | R3 | applied | 新建 8 个公共小模块，收敛重复实现；`numeric_targets` 成为唯一事实源，`hint_coordinator` / `rcb_utils` 改 re-export 保留公开 API |
| 18 | `runtime/trace_context.py` | R4 | applied | 剥离 U+FEFF BOM，修复 `ast.parse` 失败 |
| 19 | `tests/test_rcb_path_rules.py` | R4 | applied | 剥离 U+FEFF BOM |
| 20 | `tests/test_toolchain_e2e.py` | T1 | applied | 7 个同名 `test_chain` 改为描述性名称 |
| 21 | `tests/test_kg_feedback.py` | T4 | applied | 4 条行为护栏（节点去重 / 边集合 / 空取） |
| 22 | `tests/test_self_model_audit.py` | T4 | applied | 7 条行为护栏（call/result 配对、duration、孤儿调用、坏行容错） |
| 23 | 593 个 Python 文件 | R1 | applied | Round 3–4 长方法 / 深嵌套逐函数拆分（见上表），helper 抽取 + A/B 等价性验证 |
| 24 | `harness/phase_spec.py`、`autoloop/engine_act.py`、`cli/slash_commands.py`、`routes/{config,tools,compat,transfer,ws_helpers}.py` | R6 | applied | 拆分后修复契约审计的静态可见性（mode 分支、`set_mode` 字面量、凭据映射表、返回形状、`req.*` 字段读取、`msg.*` 字段读取），并重生成 `docs/mece-audit.md` / `tests/golden/field_audit/http_field.json` / `mece_findings_baseline.txt` |
| 25 | `memory/longterm.py`、`memory/decay.py` | R6 | applied | `POST /memory` 的 `importance` 传字符串时会以 TEXT 落库，之后衰减 `importance * decay**idle_days` 抛 `TypeError` 并让整次维护失败。落库前 `_coerce_importance()` 收敛成 float，衰减时对历史脏行同样收敛 |
| 26 | `docs/{env,feature-flags,plugins,events,errors}-contract.md` | R6 | applied | 拆分导致行号引用漂移，用 `python -m huginn.cli.config_audit --<mode> --out docs/<name>.md` 重生成；diff 仅为行号，`errors-contract` 另多出 Fix Log #2 新增的 `tools/adapter.py` TIMEOUT 源。`test_all_contract_docs_are_not_drifted` 由红转绿 |

---

## Verification

- **`ruff check`（`/workspace/agent` 全量）**：18 条告警，全部在 HEAD 上同样存在（`replay_audit.py` 8 条 UP031、遗留脚本 `insp*.py` / `test_sym.py` 7 条、`exec_observation.py` C420 1 条、`test_structural_defect_fixes.py` I001 1 条）。本轮新增/修改的文件 `All checks passed!`。
- **全量 pytest**（`/tmp/hv` 于 Round 2 建立的锁定环境，`-n 3 --timeout=180`）：**52 failed / 10,292 passed / 234 skipped / 8 xfailed / 12 errors**（1,022 秒）。复跑一次为 53 failed / 10,291 passed，差异仅 `tests/stress/test_long_running_extended.py::test_100_turn_memory_stable` 在 `-n 3` 内存压力下 `gw2` worker 崩溃（Round 2 基线同样出现）。
  - 52 条 failed + 12 条 errors 全部与 Round 2 基线同名同因：缺 `matplotlib` / `pymatgen` / `paramiko` / `nbformat` / `pytest-benchmark` 等可选依赖；契约 golden 的 pre-existing 漂移；`test_provenance` 的 `code_lab` 回落链；`LiteratureTool._llm_invoke` 缺失；`test_ponytail_debt_registry` 的 `DEBT-S2-01` 锚点在 HEAD 上同样不存在（`git show HEAD` 复核）；`test_route_smoke` 的配置串扰等。
  - 相比 Round 2 基线的 55 failed，本轮净减 3 条：`test_all_contract_docs_are_not_drifted`（Fix Log #26）+ `tests/test_cli.py` / `tests/test_cli_persona_export.py` 的 memory-maintenance（Fix Log #25）。
  - memory-maintenance 的两条原本被判定为「测试缓存污染」：`tests/test_error_boundaries.py` 故意写入 `importance="not a number"`，SQLite 以 TEXT 存下，后续衰减抛 `TypeError`。已按 Fix Log #25 从存储边界根治；把污染过的库还原后重跑，同组 14 条测试全绿，且库内 TEXT importance 行被自动收敛为 float。
- **专项批次**：memory 相关 9 个文件 + `test_error_boundaries.py` → 87 passed；契约审计 `test_contract_audit.py` + `test_context_mgmt_fixes.py` → 167 passed；凭据归口 `test_arch_credential_gateway.py` → 3 passed。
- **A/B 等价性**（抽样披露）：`hpc/client.generate_job_script` slurm/pbs 全参数笛卡尔积 20,736 例输出字符串一致、51 个非法输入异常类型+文案一致；`perception/cross_validator` 20,000 组随机 claim/data 输出 0 差异；`design/debugger_tool` 4,080 例 0 差异；`sci/consensus_scoring_tool` 48 组 0 mismatch；`bench/runner` / `evolution/engine` / `snapshot/file_snapshot` / `tools/adapter` / `self_improvement/core` 均在对应测试批次全绿（`self_improvement` 8 个公开构建入口的 case 顺序与字段逐一比对一致，仅自动生成的 uuid 不同）。

---

## Residual Items

### 🟡 Warning

**Assertion Roulette / Ambiguous test name — 360 条**
Symptom: 210 个测试文件中 360 个测试函数包含多条无 message 断言，或存在同名测试。
Source: *xUnit Test Patterns* — Assertion Roulette；*The Art of Unit Testing* — 测试可诊断性。
Consequence: 断言失败时只能看到行号，无法从输出判断破坏的是哪条契约。
Remedy: 给断言补 message，或按断言主题拆分测试。
Not applied because: 纯 message 级机械改写会产出约 2,000 行无行为收益的 diff，且跨 210 个文件，超过 Extended-Safe 的 ≤5 文件上限。

**Warning 桶（20–50 行 / nest 4–5）未重判**
Symptom: Round 2 记 461 条；Round 3–4 的拆分会产生大量落在 20–50 行的 helper，体量只会增加。
Source: *Clean Code* — 函数应只做一层抽象。
Consequence: 该桶本身不构成衰减信号（20–50 行是正常函数体量），只作为 Critical 的前置观察面。
Remedy: 只在函数确实承担多个抽象层时再拆。
Not applied because: 阈值跨越是提示而非判决（「A threshold crossing is a hint, not a verdict」），本轮收敛判据只看 Critical。

**Module Dependency — 3 条可疑边**
Symptom: `tools/recall_context_tool.py` 模块级 `from huginn import metacog`；`runtime → autoloop`、`validation → autoloop` 两条上行依赖。
Source: *A Philosophy of Software Design* — 依赖应指向稳定抽象。
Consequence: 底层包反向依赖高层编排，未来拆分包时可能出现循环。
Remedy: `recall_context_tool` 改惰性导入；另外两处通过 Protocol 注入。
Not applied because: 实测 949 条模块级 import 边 0 环、无分层倒置，收益低于风险。

### 🟢 Suggestion

**Asset Inventory — 84 个无直接测试模块**
Symptom: `huginn.academic.*`、`huginn.plugins.*`、`huginn.workflows.templates_*`、`huginn.snapshot.selfcheck` 等无同名测试。
Source: *Working Effectively with Legacy Code* — 无测试的代码是遗留代码。
Consequence: 这些模块的行为变更不会被 CI 捕获。
Remedy: 优先为 `plugins` / `workflows.templates_*` 补契约测试。
Not applied because: 新建测试超出「修复既有发现」的范围，且需要领域输入样例。

**Duplicated Implementation — 5 组（形状相似，非知识重复）**
Symptom: 测试内 `_install_auditor` / `_stub_heavy_calls` / `_make_engine` / `_serialize` / `_no_rate_limit` 等替身构造重复；`workflows/mcp_export` 与 `capabilities/mcp_export` 的 `_http_app` 是不同后端；`autoloop` 5 处 `__setattr__` 是组合式转发协议；`security` 3 处 `__init__` 属不同物理模型。
Remedy: 仅对同包内断言完全一致的替身抽取共享 fixture。
Not applied because: 逐组复核后均为形状相似而非知识重复，强行合并会把测试夹具耦合到生产模块。

---

## Summary

- Total findings detected: 5,437（Round 1）→ 5,433（Round 2）→ R1 Critical 0（Round 3–4 复扫）
- Fixed this sweep: 34 条可机械改写项 + 1,568 条 Critical 长方法/深嵌套（593 个文件重写）+ 3 条联带缺陷（契约静态可见性、memory importance 类型、契约文档漂移）
- Residual (needs human review): 360 条 T1 + 3 条依赖边 + 84 个无测试模块 + 5 组形状重复
- Unresolvable (3-retry exhausted): 0
- 收敛判据全部达成：最长函数 2,088 → 50 行；`nest>5` 64 → 0；`Critical` 1,580 → 0；契约面（`app.openapi()`）与 HEAD 逐字节一致。
