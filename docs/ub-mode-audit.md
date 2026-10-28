# Huginn Agent 框架审计（ub-mode 透镜）

- 审计对象：`/workspace/agent/huginn`，重点 `huginn/autoloop`（核心自主循环）
- 审计透镜：`.claude/skills/ub-mode/SKILL.md`（Uncle Bob 的 First Make it Work / Then Make it Right / Practice）
- 审计日期：2026-10-28
- 结论性质：**结构性**，非逐行 code review。全部结论附可复现证据。

---

## 0. 方法与可复现证据

| 证据 | 命令 | 结果 |
|---|---|---|
| 架构静态门 | `pytest tests/test_arch_no_import_cycles.py tests/test_arch_cleanliness.py -o addopts=""` | ✅ 12 passed |
| 契约审计基线 | `python -m huginn.cli.contract_audit`（产物见 `agent/docs/mece-audit.md`、`agent/tests/golden/mece_findings_baseline.txt`） | 已生成，缺口见该文档 |
| 复杂度 | `radon cc -s -n C huginn/autoloop` | 最高 C(17)，无 D/E |
| 克隆检测 | AST 归一化后匹配 ≥8 条连续语句 | 7 组，最大 4 处同源 |
| 规模 | `wc -l` | autoloop 38,572 行 / 43 个模块 |

**未安装的配套工具**：`crapper` / `dryer` / `mutator`（Uncle Bob 的独立仓库）。因此 `CRAP<10`、`重复率<0.8`、变异测试覆盖率三项**未按原工具口径执行**，本文用 radon 与 AST 克隆检测作替代近似，已在下文标注。

---

## 1. 一句话结论

Huginn 的**静态架构是干净的**（无导入环、分层检查通过、契约文档与审计工具成体系），`PlanStore`、`CognitiveLoop` 内核这类局部设计达到 ub-mode 标准；**真正的结构性缺陷集中在一点：整个 Agent 只有"一个可变的 engine 大对象"，11 个协作模块只是它的转发外壳**。由此派生出非法状态可写、失败被降级成"合法行为"、持久化两套约定、边界解析散落四类问题。

---

## 2. 缺陷清单

### A. 领域建模：非法状态不是"不可能"，而是"没人管"

#### A1【严重 / 结构性】✅ 已修复（§6）协作对象是假分解：一个可变对象 + 11 个转发壳

- `huginn/autoloop/engine.py:366-410` `_init_collaborators()`：`MathValidator/EngineAct/EngineControl/PlanCheck/EngineObserve/EngineReflect/HypothesisLoop/CognitiveRunner/EnginePerceive/VisualInspect/RecallModes` 全部以 `Xxx(self)` 构造，**共享同一个 engine 实例**。
- 每个协作对象用 `__getattr__` + `__setattr__` 把**所有属性读写转发回 engine**：
  - `huginn/autoloop/cognitive_loop.py:768`
  - `huginn/autoloop/engine_observe.py:59`
  - `huginn/autoloop/engine_reflect.py:909`
  - `huginn/autoloop/hypothesis_loop.py:2738`
  
  四处**逐字重复**同一段转发逻辑 + 各自手工维护一份 `_OWN_ATTRS` 名单。
- 度量：`AutoloopEngine` 共 **253 个方法，其中 114 个是单行委托**（`return self._engine_actor._x(...)`）；`engine.py` 内可变的 `self.<attr>` 赋值点 **115 个**，再加协作对象转发写入的字段，实际可变状态远超 200 个。
- 拆前拆后：`去 mixin` 注释出现 **34 处**、`ponytail` 标记 **1221 处**。说明已做过约 10 轮"拆分"，但每轮都在**搬方法**，没有切状态。

**为什么这是根因而不是风格问题**：
ub-mode 的 `Separate Before Serializing Shared State` 要求"需要同一份可变状态时，只允许一个写入者，并在结构上强制"。这里写入者是 N 个，且没有任何结构阻止 `EngineObserve` 写 `_darwin_best_score`。`_OWN_ATTRS` 是**手工同步的名单**——漏一个方法名，赋值就会静默转发回 engine，产生"看起来改了、其实改在别处"的隐形状态漂移。这类 bug 只能靠测试撞见，无法靠类型或结构排除。

**处方（分三步，可独立验收）**：
1. 把 `_OWN_ATTRS` 从手工 frozenset 改为**自动推导**（`inspect.getmembers(type(self), predicate=isfunction)`），四处转发逻辑收敛为一个 `_EngineFacade` 基类 → 一处决策、一处实现；
2. 给 engine 的可变字段做**归属清单**（ownership map）：每个字段声明唯一 owner 模块，其余模块只读；用测试断言"非 owner 写入即失败"（在 `__setattr__` 里加断言即可）；
3. 只为真正跨阶段共享的少数字段（iteration/phase/hypothesis 指针）保留 engine，其余下沉为协作对象**自有**状态。

#### A2【严重 / 行为】✅ 已修复（§6）核心循环把异常降级成"另一种合法行为"

- `huginn/autoloop/cognitive_loop.py:415-538`：
  - observe 抛错 → `observation = {}`（:419-421）
  - decide 抛错 → **`state.should_stop = True`**，返回 None（:431-434）
  - execute 抛错/超时 → `last_action_result = None`，仅标 redirect（:486-501）
  - reflect 抛错 → `ReflectionResult()` 默认"继续"（:512-514）
  - writer 抛错 → `logger.debug`（:537-538）
- 全量：`huginn/autoloop` 内 `except Exception` **523 处**；`engine_reflect.py` 的 114 处中约 **80 处**只记 debug/pass（`engine_reflect.py`）。

**为什么是缺陷**：这正是 ub-mode `Fix Root Causes` 里说的"只掩盖崩溃的 guard 不是修复"。后果不止是丢日志——**同一个崩溃在统计上表现为"正常停机"**：decide 挂了会让整轮 run 以 `should_stop=True` 正常收尾，报告里看不出失败，测试也不会红。这是把"失败"塞进合法状态空间，属于 A1 在行为层的同一个根因。

**处方**：区分**可降级**（事件派发、遥测、缓存写入）与**不可降级**（observe/decide/reflect 的返回值契约）两类；后者失败必须以 `PhaseStatus=failed` + 显式异常传播，让上层用 `success=False` 参与判定。降级白名单写在一处常量里。

#### A3【中 / 一致性】✅ 已修复（§6）模式与阶段词表 6 套且互不一致，mode 是裸 `str`

- 证据（已提交基线）：`agent/docs/mece-audit.md:64-80`
  - `prompt_builder._MODE_INSTRUCTIONS` = `chat/code/extreme/fusion/research`
  - session 恢复白名单 = `chat/extreme/plan/research`
  - `critique._VALID_MODES` = `chat/plan/research`
  - `core._LONG_HORIZON_MODES` = `extreme/research`
  - `set_mode()` 运行时实参 = `chat/plan/research`
- 有 prompt 段却无生产者：`code/extreme/fusion`；有生产者却无 prompt 段：`plan`。
- 类型层面无约束：`self._mode = mode` 直赋字符串（如 `huginn/agent/session.py:139`、`huginn/budget_pause.py:193`）。

**ub-mode 对应**：`Make the illegal value impossible to write`。应有一个 `Literal`/`StrEnum` 单一真值源，六处词表从中派生（或直接删除冗余词表）。

#### A4【中 / 契约】✅ 已修复（§6）契约审计只产出"报告"，没变成"门"

`agent/docs/mece-audit.md` 已抓出：dispatch 支持但 planner 未教的 4 个 mode（`code_lab/dynamic_workflow/frame_inspect/trace_inspect`，:62）、未登记撞名、事件常量撞值等。基线在 `agent/tests/golden/mece_findings_baseline.txt`，但缺一条"**新增缺口即 CI 红**"的断言——目前是快照，不是门。

**处方**：把 baseline 比对升级为断言（只允许减少，不允许新增），一条测试即可。

---

### B. 边界纪律：同一份外部数据被解析多次

#### B1【严重】✅ 已修复（§6）环境变量没有单一入口

- 实测：`os.environ` 裸读 **556 处，涉及 389 个变量名**；其中 **82 个变量名未在 `huginn/env_schema.py` 声明**，而 `env_schema.py:4-24` 明确宣称"每个 `HUGINN_*` 只声明一次、必须经 `env_access` 读取"。
- 规范访问器 `huginn/env_access.py` 实际只有 **7 个模块**采用（`permissions.py:22`、`cognitive_loop.py:69`、`engine_control.py:40` 等）。
- 同一变量两处解析，互不知情：
  - `VASP_EXECUTABLE`：`huginn/config.py:758` 与 `huginn/tools/sim/vasp_tool.py:248`
  - `LAMMPS_EXECUTABLE`：`config.py:759` 与 `tools/sim/lammps_tool.py:815`
  - `MP_API_KEY`：`config.py:760` 与 `tools/materials_database_tool.py:665`
  - `HPC_HOST/HPC_USERNAME/HPC_KEY_PATH`：`config.py:786-788` 与 `tools/job_tool.py:269-272`
  - `ABAQUS_MCP_SERVER_PATH`：`config.py:816` 与 `cli/context.py:68`

**ub-mode 对应**：`Parse outside data once, where it comes in` + `One decision lives in one place`。当前"边界"存在于三处（`config.py` / `env_defaults.py` / 各模块裸读），三者都不是唯一入口。

**处方**：先做**不改变行为的收敛**——`env_access` 增加 `env_str/env_path/env_csv` 三个访问器，按上面的重复清单逐条替换（每条可独立提交、独立验证），`env_schema` 补 82 个声明；再用 `test_env_surface_governance.py` 把"裸读"变成 lint 级失败。

#### B2【中】✅ 已修复（§6）LLM 输出解析有 5 套独立实现，策略互不相同

| 位置 | 策略 |
|---|---|
| `huginn/autoloop/plan_check.py:512` `_extract_json_object` | 返回 `(text, rest)` 二元组 |
| `huginn/metacog/text_to_json.py:76` `extract_json_object` | `prefer="last"`，返回 dict |
| `huginn/utils/text_types.py:127` `_json_from_brackets` | 括号配对扫描 |
| `huginn/plugins/autoresearch.py:444` `_extract_json` | 返回 str |
| `huginn/tools/vision_locate_tool.py:35` `_extract_json` | 返回 `Any \| None` |

同名不同契约、不同修复策略，同一个模型输出在不同路径上可能得到不同结果。**处方**：抽一个 `huginn/utils/json_extract.py` 单一实现，其余退化为薄封装（先补等价性测试再替换）。

#### B3【中】✅ 已修复（§6）内层延迟导入 424 处

`huginn/autoloop` 内函数体内的 `from huginn.x import ...` 共 **424 处**；`engine.py` 有 **25 处 `# noqa: E402`**（`engine.py:118/136/142/...`）。导入环被静态门挡住了，但依赖方向仍靠**在每个函数里延后导入**维持——这是"用位置换依赖方向"，不是结构保证。**处方**：先按 `E402` 的 25 处逐个消除（每次一处、跑全量测试），剩下的内层导入按"跨层"与"同层懒加载"分类，前者上移，后者保留。

---

### C. 类型纪律：核心状态不是类型，是字典

> ✅ 已按 C1 完成第一层收口（§6）：`cog` 已定 `TypedDict`，64 处下标获得静态键检查；`Any` 与 dict→dataclass 仍为后续项。

- `huginn/autoloop` 内 `Any` 出现 **1302 处**（审计时为 1325；C1 未动 `Any`，差异来自同期其他改动）。
- 核心运行状态是裸 `dict`：`cog["..."]` 直接下标 **67 处**（审计时 64）；`run_cognitive(cog=...)` 全链路传 dict 并就地 mutate。现已统一标注为 `CognitiveState`（`huginn/autoloop/types.py`），字面量键由 `tests/test_cognitive_state_contract.py` 静态校验 ⊆ 声明字段。
- 相比之下 `LoopState`（`cognitive_loop.py:262`）和 `AutoloopResult`（`types.py:24-55`）是正经 dataclass —— **同一系统里两套状态表示**，内核用类型、业务用字典。

**ub-mode 对应**：`Get the data shape right before you write the logic`。**处方**：先给 `cog` 定 `TypedDict`（零运行时成本），让 64 处下标获得静态检查；再逐步升级为 dataclass。这是"先拿静态收益、不动运行时"的最小步。

---

### D. 幂等性与持久化：两套约定并存

#### D1【严重】✅ 已修复（§6）快照写盘不是原子的，且 `.in_progress` 无人清理

- `huginn/autoloop/types.py:108-111`：`path.write_text(json.dumps(...))` —— **无锁、无临时文件、无 rename**。写到一半崩溃 → 截断 JSON → `load`（`types.py:126-133`）捕获异常返回 `None` → **整轮 run 静默丢失**。
- `types.py:83-107` 的 `.in_progress.json` 约定"由调用方清理"，但全仓 grep 不到任何清理代码 → 陈旧文件永久堆积（非法状态可存在）。

**对照组（已达成范式）**：`huginn/autoloop/plan_store.py:216-237` 用 `_file_lock` + `atomic_write_json` + 损坏文件备份改名。**同一仓库、同一个"存 JSON"的活儿，两套标准。** 这是 ub-mode `Make Operations Idempotent` 的直接违例：run 两遍、或半途崩溃后重跑，落盘结果不同。

**处方**：`save_autoloop_snapshot` 直接改用 `atomic_write_json`；`in_progress` 文件在 `_finalize_run` 里删除；补一条"写一半被杀 → 重跑后状态一致"的测试（可注入一个在第 N 字节抛 IOError 的 fake writer，无需真杀进程）。

#### D2【中】autoloop 内 64 处文件写入点分散

`write_text / json.dump / .write(` 共 **64 处**，分属 `PlanStore/GoalStore/HypothesisEventStore/FrameStore/cognitive_persist/types.py` 等。没有统一 `Store` 接口，故没有统一的事务/锁/损坏处理语义。

---

### E. 死代码与测试

#### E1【中高】✅ 已修复（§6）生产模块里躺着 10 个测试替身

`huginn/autoloop/cognitive_loop.py:4978-5060`：`_sc_observe/_sc_decide/_sc_execute/_sc_reflect/_sc_stuck_decide/_sc_bad_decide/_MockWriter/_sc_new_loop/_sc_basic_flow/_sc_stuck_loop/_sc_invalid_action/_sc_run_all` —— **全仓零外部引用**（grep 确认）。它们本该在 `tests/`。后果：生产模块 +120 行噪声、`__all__`/覆盖率统计被测试代码污染、未来读代码的人分不清哪条路径是真的。

**处方**：整体搬到 `tests/test_cognitive_loop_kernel.py`（文件已有同名测试更好，合并即可），生产文件删除。单次可验证。

#### E2【中】测试体量与 mock 密度

- 测试函数 **8,614 个** / 548 文件，对 43.5 万行生产代码 —— **数量上是健康的**（ub-mode 的 TDD 精神已具备）。
- 但 `monkeypatch.setattr` / `patch(` 出现 **989 处**。这不是缺陷本身，却是 `Test Behavior, Not Implementation` 的风险面：mock 装配越密，测试越可能在"函数都返回 None 时仍然通过"。**建议**：抽样 30 个 autoloop 测试做"空实现测验"（把所有被 mock 的方法体清空，看测试是否仍绿），作为一次性诊断，不必全量。

---

## 3. 已经做对的（不要在重构中改坏）

| 项 | 证据 | ub-mode 对应 |
|---|---|---|
| 无导入环 + 分层检查通过 | `tests/test_arch_no_import_cycles.py`、`test_arch_cleanliness.py`（12 passed） | Architecture / Boundary |
| 契约审计自成体系且有基线 | `agent/docs/mece-audit.md`、`tests/golden/mece_findings_baseline.txt` | Keep a Log / Build the Lever |
| `PlanStore` 原子写 + 文件锁 + 损坏备份 | `plan_store.py:216-237` | Make Operations Idempotent |
| `CognitiveLoop` 内核是纯 hook 编排、不含业务 | `cognitive_loop.py:315-420` | Keep business logic in a pure function |
| 域词汇表与契约文档齐备 | `agent/docs/*-contract.md`（events/tools/permission/modes/errors/env/routes） | Maintain a glossary |
| 单函数复杂度尚可控 | radon 最高 C(17)，无 D/E | —— 说明问题**不在函数粒度**，在对象与状态粒度 |

---

## 4. 修复次序（按"根因 × 可验证单元"排，不按文件排）

> 原则：先断根因（A1/A2 同源），再做单向收敛（B1/D1），最后做类型化（C1/A3）。每步都要能独立验证、独立提交。

**第 0 批 · 只加门，不改行为（风险最低）** — ✅ 全部完成
1. ✅ `mece_findings_baseline` 从快照升级为断言门（只减不增）。
2. ✅ 新增"裸读 env" lint 测试（`tests/test_env_surface_governance.py`，棘轮 464/42，只禁新增）。
3. ✅ 新增 autoloop except 卫生棘轮（`tests/loops/test_structural_guards.py`）：AST except 处理器 575、宽捕获静默 473，只减不增。

**第 1 批 · 断根因（A1 + A2）** — ✅ 全部完成
4. ✅ 四处 `__setattr__` 转发收敛为 `huginn/autoloop/_facade.py:EngineFacade`，`_OWN_ATTRS` 由 `dir(type(self))` 自动推导。
5. ✅ 建立 engine 字段 ownership 清单（`_facade.SINGLE_WRITER` 101 项 + `SHARED_FIELDS` 27 项）+ "非 owner 写入即 `EngineOwnershipViolation`"断言。
6. ✅ observe/decide/reflect 的异常从"降级"改为"显式失败"，降级白名单 `_DEGRADABLE_HOOKS` 集中一处常量。

**第 2 批 · 单向收敛** — ✅ 全部完成
7. ✅ `save_autoloop_snapshot` 改 `atomic_write_json` + `in_progress` 清理 + 崩溃重跑测试（D1）。
8. ✅ 五套 JSON 提取合并为 `huginn/utils/json_extract.py`，其余退化为薄封装（B2），含等价性测试。
9. ✅ `engine.py` 25 处 `E402` 消除（现 0 处），import 块整体上移到模块顶部（B3）。

**第 3 批 · 类型与词表** — ✅ 全部完成
10. ✅ `cog` dict → `CognitiveState`（`TypedDict`，C1），5 个模块注解替换 + 字面量键门禁测试。
11. ✅ mode 词表收敛为单一 `Literal`（`huginn/agent_modes.py`），六处派生（A3）。
12. ✅ 搬走 `cognitive_loop.py` 的测试替身到 `tests/test_cognitive_loop_kernel.py`（E1）。

---

## 5. 局限

- `crapper/dryer/mutator` 未安装，`CRAP<10`、`重复率<0.8`、变异测试三项未按原口径跑；本文的复杂度与重复检测是近似替代，**结论不依赖这三项的精确阈值**。
- 未运行时序/并发审计：A1 的"多写入者"风险只做了静态证据，没有实际并发竞态复现。
- 契约面的数字（词表、撞名）来自已提交的 `mece-audit.md` 基线；本次未重新全量生成（依赖较重，CLI 导入链需要完整运行时）。
- 当前所有改动仍在工作区，**未提交、未推送**。
- 回归环境有两个既存缺失依赖：`starlette`（23 个 FastAPI/WS 测试模块无法收集）与 `mcp_types`；这些模块的失败与本轮改动无关，已在验证口径中单列。

---

## 6. 修复记录（"都做吧"执行结果）

审计第 4 节 12 项全部落地，逐项证据如下。

| 项 | 交付物 | 验证 |
|---|---|---|
| A1（4/5） | `huginn/autoloop/_facade.py`：`EngineFacade` 统一 `__getattr__`/`__setattr__`；自有名字由 `dir(type(self))` 自动推导（删除四处手工 `_OWN_ATTRS`）；`SINGLE_WRITER` 101 字段单写者清单 + `SHARED_FIELDS` 27 字段显式共享声明；非 owner 写真实引擎字段即 `EngineOwnershipViolation` | `tests/test_engine_decomposed.py`、`test_autoloop_engine.py` 等 296 passed |
| A2（6） | `cognitive_loop.py`：`_DEGRADABLE_HOOKS = {output_writer, workflow_event}` 单一降级白名单；其余钩子失败 → `StepFailure` 进 `LoopState.failures`（有界 `_MAX_LOOP_FAILURES`）+ ERROR 日志，最终 `AutoloopResult.success` 增加 `not loop_failures` 项 | `tests/loops/test_structural_guards.py`（A2 段）、`test_cognitive_loop_kernel.py` |
| A3（11） | 新建 `huginn/agent_modes.py`：`AgentMode = Literal[...]` + `ALL/SETTABLE/LONG_HORIZON/CRITIQUE/PROMPT_MODES`；`core.py`/`session.py`/`prompt_builder.py`/`critique.py` 六处派生；`_MODE_INSTRUCTIONS` 键集由模块级 assert 对齐 | `tests/test_agent_mode_vocabulary.py`（9 用例）、`contract_audit` 模式面零新发现 |
| A4（1） | `mece_findings_baseline` 升级为棘轮断言（基线内退出 0，基线外新增即红） | `tests/test_contract_audit.py::test_findings_baseline_covers_current_findings`、`contract_audit --check` |
| B1（2） | `env_access` 增加 `env_str/env_path/env_csv`；`VASP_EXECUTABLE`/`LAMMPS_EXECUTABLE`/`MP_API_KEY`/`HPC_*`/`ABAQUS_MCP_SERVER_PATH` 等重复变量收敛到单一读点；`env_schema.SCHEMA` 补至 405 条 | `tests/test_env_surface_governance.py`（6 条不变量） |
| B2（8） | 新建 `huginn/utils/json_extract.py`；`plan_check`/`text_to_json`/`text_types`/`autoresearch`/`vision_locate_tool` 五处退化为薄封装，原契约不变 | `tests/test_json_extract.py`（等价性测试） |
| B3（9） | `engine.py` 延迟 import 块整体上移到模块顶部，`# noqa: E402` 25 → **0**；`BenchmarkRunner` 保留 `noqa: F401 # monkeypatch` | `python -c "import huginn.autoloop"`、autoloop 全套 296 passed |
| C1（10） | `huginn/autoloop/types.py: CognitiveState(TypedDict, total=False)`；`cognitive_loop/engine/engine_reflect/cognitive_checks/three_cabin_reflector` 注解替换 | `tests/test_cognitive_state_contract.py`（字面量键 ⊆ 声明字段） |
| D1（7） | `save_autoloop_snapshot` 改 `_file_lock` + `atomic_write_json`；非 in_progress 写成功后清理同名 `.in_progress.json` | `tests/test_autoloop_engine.py`（崩溃/重跑一致性） |
| E1（12） | `cognitive_loop.py` 的 `_sc_*`/`_MockWriter` 测试替身整体迁至 `tests/test_cognitive_loop_kernel.py`，生产文件零残留 | `grep -n "_sc_observe\|_MockWriter" huginn/autoloop/cognitive_loop.py` → 空 |

**环境面实测（`config_audit.build_inventory()`）**：引用变量 325 个 / 声明 405 个 / **未声明 0**；读点 478，其中裸读 464（= `RAW_READ_BASELINE`，无新增余量可透支）、同模块重复 42（= `RAW_DUP_READ_BASELINE`）。审计原文的 556/389/82 未声明已归零。

**本轮新增门禁**
- `tests/test_agent_mode_vocabulary.py` — 模式词表单一来源 + 不可设置模式拒绝。
- `tests/test_cognitive_state_contract.py` — `cog` 字面量键不得越出 `CognitiveState`。
- `tests/loops/test_structural_guards.py` — autoloop except 总数与"宽捕获静默"双棘轮。
- `tests/test_env_surface_governance.py` — 声明完整性 / 死声明 / 默认值漂移 / 裸读与重复读棘轮。
- `tests/test_autoloop_engine.py` — 快照原子写与 in_progress 清理。

**仍未做（不属于本次 12 项，审计原文的后续项）**
- A1 处方第 3 步：跨阶段共享字段下沉为协作对象自有状态（当前用 `SHARED_FIELDS` 显式登记，只解决了"隐形"，未解决"共享"本身）。
- D2（64 处写盘点统一 Store 接口）、E2（mock 密度抽样"空实现测验"）。
- `huginn/autoloop` 仍有 11 条 ruff 既存告警（`replay_audit.py` UP031 ×8、`plan_check.py` E741、`hypothesis_loop.py` 兼容再导出 E402、`exec_observation.py` C420），不在本轮范围。
