# Autoloop 控制面审计

审计对象：`agent/huginn/autoloop/`（含被它调用的 `research/`、`constraints/` 门）。
目的：回答"控制流是不是过强"，并把每个硬决策点标成 **删 / 降 / 留**。
本文只出判断，不改代码。

---

## 0. 一句话结论

**控制流已经是 run 失败的首要原因，而不是安全带。**

最近 6 轮长程 run 里，**3 轮是被框架自己的控制机制杀死的**，且只跑到预算的 17%–25%：

| run | 实际时长 / 预算 | 终止原因 | 是谁干的 |
|-----|----------------|---------|---------|
| run50 | 903s / 3600s (25%) | exec convergence | **A1 硬控**（且是假收敛） |
| run51 | 607s / 3600s (17%) | rename debt 越限 | **A2 硬控** |
| run52 | 905s / 3600s (25%) | exec convergence | **A1 硬控**（假收敛） |
| run53 | 3606s / 3600s (满) | 挂钟耗尽 | 正常 |
| run54 | 3612s / 3600s (满) | 挂钟耗尽 | 正常 |
| run55 | 3604s / 3600s (满) | 挂钟耗尽 | 正常 |

两处关键证据：

1. **硬控会互相打架。** run52 里 pivot 硬指令（B1）已经在 streak=2/3/4 **触发了 3 次**，
   但 run 仍被另一个硬控 A1 杀死。一个硬控存在，并不能阻止另一个硬控误杀——
   因为杀它的不是"书生没进展"，而是"预算门跳过了 execute，循环复用旧结果"。
2. **控制流生产出需要更多控制流的故障。** run50/52 的假收敛，根因是 A6（预算门
   跳过 execute）。修它的办法又是加控制流——`_long_horizon_keep_going` 让位。
   目前该方法在 `cognitive_loop.py` 里被特判了 **9 处**，这 9 处本身就是新的控制面。

> 更正：我此前口头说过"pivot 硬约束三轮满预算 0 次触发"——**这是错的**。
> run52 触发了 3 次，run51 达 streak=2，run53 达 streak=1。正分支在野外是生效的。

---

## 1. 控制面体量

| 指标 | 数值 |
|------|------|
| `autoloop/` 下 `HUGINN_*` 环境旋钮 | **59 个** |
| `cognitive_loop.py` 里 `environ.get` 读取点 | 16 |
| `engine_reflect.py` 里 `environ.get` 读取点 | 14 |
| `_long_horizon_keep_going()` 特判点 | 9 → **1**（A5 `cognitive_loop.py:2587`；A6 已按 run56 数据收掉） |
| 硬决策状态量 | `_exec_converged` `_repeat_exec_streak` `_rename_debt` `_fp_result_ref` `_exec_fp_history` `_darwin_stagnation` …（`_force_exec_variation` 已随 B1 删除） |

项目自己写的红线是"**低熵红线：平台内核不为单个命题累积例外**"
（[code_lab.py](../agent/huginn/research/code_lab.py) 等多处）。控制面增长走的是这条路线的反面。

---

## 2. 决策点清单

分类口径：
- **A 终止类**——能直接结束 run。风险最高（误杀即丢失整轮实验）。
- **B 强制/引导类**——不改终止，但替书生决定"下一步该怎么做"。
- **C 诚实/证据门**——只判"这条证据算不算数"，与科学判断无关。**应当硬**。
- **D 观测类**——只记录，不干预。**应当留**。

### A. 终止类（建议逐条降级）

| # | 机制 | 位置 | 实测触发 | 对书生的架空 | 建议 |
|---|------|------|---------|-------------|------|
| A1 | exec convergence → conclude+stop | `engine_reflect.py:1355` 置标志；`cognitive_loop.py:3538-3545` 终止 | run50、run52（均为**假收敛**） | 高：直接结题 | ✅ **已降**：只写 trace + 提示，不自动终止 |
| A2 | rename debt 越限 → conclude+stop | `hypothesis_loop.py:2315` 累加；`cognitive_loop.py:3561-3569` 终止 | run51（607s 早停） | 高：直接结题 | ✅ **已降**：转观测计数（已用于审计），不终止 |
| A3 | darwin stagnation → stop | `cognitive_loop.py:1153-1174` | 长程下被让位，未触发 | 中 | ✅ **已降**：提示 + trace（原 else 分支的 stop） |
| A4 | belief σ² 收敛 → stop | `cognitive_loop.py:1181-1202` | 未触发 | 中 | ✅ **已降**：提示 + trace |
| A5 | LLM decider "stop" | `cognitive_loop.py:2587`（长程拦截） | run50 曾因此提前收起（已修） | 中 | ✅ **已降**：显式提示 + trace，让书生自己决定何时收结 |
| A6 | 预算门跳过 execute | `cognitive_loop.py:2780` | run52（**假收敛根因**）；run56 长程 7/7 bypass | 高：静默跳过阶段 | ✅ **已收掉**：run56 实测长程下 7/7 直接 bypass = 零信息，故长程整段不走预算门、不留 bypass trace；短程保留原门 |
| A7 | 挂钟预算 → stop | `cognitive_loop.py:961` | run53/54/55 正常耗尽 | 低：用户给定的边界 | **留**（唯一的合法自动出口）；⚠️ 见 §5：迭代内不查预算，曾被修复循环架空 |
| A8 | max_consecutive_failures / by_type | `cognitive_loop.py:3072-3139` | 未知 | 中 | **留 + 观测**：已接 `campaign.control_trace`（`name=failure_budget`, `action=stop`），待数据后定 |

### B. 强制/引导类（建议统一降为"写进 trace 的提示"）

| # | 机制 | 位置 | 实测触发 | 建议 |
|---|------|------|---------|------|
| B1 | pivot 硬指令 `_force_exec_variation` | `engine_reflect.py` 置；`engine_act.py:_build_codelab_focus` 注入作者提示 | run50/51/52 多次 | ✅ **已降**：并入提示层（按 `_repeat_exec_streak` 现算），去掉独立标志状态机 |
| B2 | repeat 软提示 `_speculator_hint` | `engine_reflect.py:1306-1320` | run50-53 | 留（本就是提示） |
| B3 | 换名重定向 hint | `hypothesis_loop.py` | run50-55（5–20 次/轮） | 留（提示） |
| B4 | effort floor hint | `engine_reflect.py:589-606` | 未知 | **留 + 观测**：已接 `campaign.control_trace`（`name=effort_floor`） |
| B5 | 非有限数值 hint | `engine_reflect.py:533` | run54（1 次） | 留（属 C 的提示面） |
| B6 | coldstart / prompt guards | `research/coldstart_guards.py` | 每轮 | 留（防退化） |
| B7 | curiosity hint / PMK / MCMC 注入 | `engine_observe.py:912-928` 等 | 未知 | **留 + 观测**：curiosity 已接 `campaign.control_trace`（`name=curiosity_hint`）；PMK/MCMC 待观测 |

### C. 诚实/证据门（**留，且是唯一该硬的**）

| # | 机制 | 位置 | 说明 |
|---|------|------|------|
| C1 | 非有限数值不算证据 | `engine_reflect.py:71, 1381` | `_non_finite_objective_keys` 接进 `_is_code_lab_solved` |
| C2 | claim grounding 门 | `engine_reflect.py` `_report` / `_citation_gap` | ✅ **已实现**：报告面 citation 门（见 §5.4） |
| C3 | 沙箱白名单 / 禁 assert / 禁 IO | `research/code_lab.py` | 安全与诚实 |
| C4 | provenance 溯源 | `.huginn/provenance.jsonl` | 审计基础 |

### D. 观测类（**留**）

| # | 机制 | 说明 |
|---|------|------|
| D1 | 语义等价/换名审计 | 只告警，不干预；本轮 5–20 次/轮 |
| D2 | replay_audit | 离线重放判据 |
| D3 | `_fp_result_ref` 证据新鲜度 | 属证据链修正（防复用旧结果充数） |

---

## 3. 建议的判断原则（"控制面预算"）

新增任何硬机制前必须回答两问，答不上来就不加：

1. **它替书生做了哪一步判断？** ——科学判断（要不要换族、有没有进展、何时收结）
   一律下沉给书生；只有"这条证据算不算数"可以硬。
2. **能不能改成提示或观测？** ——能改就改。硬终止只保留两个出口：
   **挂钟预算耗尽** 与 **目标达成**。

配套：每个硬控触发时写一条结构化 trace（`name / iteration / evidence / action / advisory`），
双通道落盘：`campaign.control_trace` 事件（装了 audit 订阅器时进 `audit.jsonl`）
+ 一条带固定 tag 的 WARNING 日志 `control_trace name=...`（**CLI autoloop 路径不装
订阅器，故离线触发率统计以 `run.log` 的该行为准**）。
便于统计触发率；**长期 0 触发或长期误杀的，删或降**。

聚合命令（按 `name` 计数）：

```bash
grep -o 'control_trace name=[a-z_]*' run.log | sort | uniq -c | sort -rn
```

---

## 4. 建议动作顺序

1. **立原则** —— ✅ 已写进 [`docs/architecture.md`](agent/docs/architecture.md)
   「控制面预算（autoloop 硬决策准入）」，后续机制按 §3 过闸。
2. **A1 + A2 降级** —— ✅ 已降为"提示 + `campaign.control_trace`"，不再 `should_stop`
   （[`cognitive_loop.py`](agent/huginn/autoloop/cognitive_loop.py) 的 F5 / v11 两处）。
   唯一自动终止交给 A7。
3. **A3/A4/A5/A6 清理** —— ✅ 已完成。A3/A4/A5 降为"提示 + trace"；A6 先收敛为
   长程单一开关，run56 数据（长程 7/7 bypass）出来后**整段收掉**（短程保留原门）。
   `_long_horizon_keep_going` 特判点 **9 → 1**（仅剩 A5）。
4. **B1 并入提示层** —— ✅ 已完成。去掉 `_force_exec_variation` / `_repeat_exec_last_result`
   独立状态机；`_build_codelab_focus` 直接按 `_repeat_exec_streak` 现算硬指令。
5. **加触发率观测** —— ✅ 观测面已补齐：`campaign.control_trace` 统一 schema
   （`name / iteration / evidence / action / advisory`），落 `audit.jsonl`。
   除 A1–A6/B1 外，待定项 **A8（failure_budget）/ B4（effort_floor）/ B7（curiosity）**
   也已接线。
6. **跑观测 run 采数据** —— 🔄 已跑 **run56**（见 §5）：首批触发率数据到手；
   顺手修掉它暴露的 A7 漏洞（§5.2）。**下一步**：再跑 3–4 轮同类 run 凑样本
   （含新的 `report_citation` 口径），再按 §5.3 定第二轮删减。A6 长程特判已按数据收掉。
7. **报告 citation 门（C2）** —— ✅ 已完成（见 §5.4）：证据台账 + 提示硬口径 +
   `report_citation` 观测 + 条件标注（只标注不终止）。属 §2 C 族（允许硬）。

优先级：**第 2 步 > 第 3 步 > 第 4 步 > 第 5 步**。
第 2 步风险最低、收益最大：把两个会误杀整轮实验的硬终止降级，不损失任何诚实边界。

### 观测口径（第二轮删减的数据来源）

`campaign.control_trace` 每条含 `name`，按 `name` 计数即得各机制触发率：

| `name` | 机制 | 期望判读 |
|--------|------|---------|
| `exec_convergence` | A1 | 每轮出现 = 长期误杀风险高，降级后仍应观察其是否伴随真停滞 |
| `rename_debt` | A2 | 同上 |
| `darwin_stagnation` / `belief_convergence` | A3/A4 | 长期 0 触发 → 删 |
| `decider_stop` | A5 | 每轮出现 = 书生常想收结但被拦，需回看是否目标真未达成 |
| `execute_budget_gate` | A6 | `action=bypassed` 占比高 = 阶段门在长程下基本无意义 |
| `failure_budget` | A8 | 触发即硬终止；若常在非真失败时触发 → 降 |
| `effort_floor` | B4 | 长期 0 触发 → 删；高频 = 门设计或命题不匹配 |
| `curiosity_hint` | B7 | 默认 off；若开启后频繁注入但不改行为 → 删 |
| `pivot_directive` | B1 | 越阈值注入强制变异令的次数 |
| `report_citation` | C2 | 报告 Results 未溯源数值数/总数；长期低触发 = 提示面已够，长期高 = 需收紧 |
| `goal_judge` | 出口 | 每次到期完成判定（每 3 轮）一条；`action=stop_candidate` 占比 = judge 认为达成的频率 |
| `goal_acceptance` | 验收门·证据 | `action=block` = 台账无有限数值被拦；长期 100% block = judge 提前宣布达成 |
| `goal_metacog_audit` | 验收门·元认知 | 完成度审计拦截次数 |
| `goal_skeptic` | 验收门·对抗 | `verdict` 分布；`block` 高频 = judge 过于乐观或常识证伪未过 |
| `collab_blind_reconstruct` | 协作·盲重建 | `action=skip` 携带跳过原因；**长期全 skip 或 0 条 = 该协作没通电/没生效**（见 §7.5） |
| `collab_branch_incubator` | 协作·N 路孵化 | 同上；`empty:` = 跑了但 N 路子 agent 全空手 |
| `collab_failure_invert` | 协作·失败反推 | 同上 |
| `code_lab_timeout` | 证据·算力 | 被沙箱超时饿死的修复尝试数；高频 = 作者提示仍产出过重扫描（见 §7.5） |

---

## 5. run56 观测结果（触发率第一批数据）

run56 = §4 第 5 步的触发率观测 run（3600s 预算、默认旋钮、与 run53 可比；
`.huginn/evolved_skills.json` 自 run55 继承）。**已完成**（`loop_195e53a4`）。

### 5.1 触发率（`grep -o 'control_trace name=[a-z_]*' run.log | sort | uniq -c`）

| `name` | 机制 | run56 次数 | 判读 |
|--------|------|-----------|------|
| `execute_budget_gate` | A6 | **7**（全部 `action=bypassed`） | 长程下 7/7 直接放行 → 该门在长程路径上是**纯 no-op** |
| `darwin_stagnation` | A3 | 1（`action=advisory_hint`，iter 29，未终止） | 降级生效：真停滞但**不误杀**，只提示 |
| `exec_convergence` / `rename_debt` / `belief_convergence` / `decider_stop` / `failure_budget` / `effort_floor` / `curiosity_hint` / `pivot_directive` | A1/A2/A4/A5/A8/B4/B7/B1 | **0** | 见 5.3 |

run56 共 6 个完整迭代，语义换名审计命中 3 次，无任何框架硬终止。

### 5.2 关键收获：run56 暴露了 A7（唯一合法硬出口）被架空 —— 已修

- **现象**：预算 3600s，实际跑了 **4903s**（越限 36%）。日志停在 iteration 33，
  反复 `code_lab 执行超时 (> 900.0s)` → `code-lab-author` 重生 → 再超时，
  该迭代 execute 单阶段烧掉 **3634.7s**。
- **根因**：[`engine_act.py`](agent/huginn/autoloop/engine_act.py) `_execute_code_lab`
  的**自修复循环**（`HUGINN_CODELAB_REPAIR_ATTEMPTS` 默认 3 → 最多 4 次尝试）每次
  `_run_code_lab` 最长烧满 `HUGINN_CODELAB_TIMEOUT_S`（默认 900s）。单轮最坏可越
  挂钟上限近 1 小时，而挂钟只在**每轮 observe 开头**查（`cognitive_loop.py:2396`），
  迭代内不查 → 出口被架空。
- **定性**：这不是"控制流过强"，而是**该硬的边界没硬起来**。按 §3 两问过闸：
  它不替书生做任何科学判断（纯时间边界）→ 允许硬；也没有"改成提示"的空间。
  故直接修，而不是降级。
- **修法**（最小改动，复用既有写法）：`_execute_code_lab` 每次尝试**前**查一次
  `goal_store.wall_clock_expired`，已耗尽就不再起新尝试（最多再多跑一个在飞的
  尝试）。仅长程模式生效，短程行为不变；fail-open。单测
  `test_codelab_repair_loop_stops_on_wall_clock_expiry` /
  `test_codelab_repair_loop_runs_when_no_long_horizon` 锁定。

> run56 本身用的是**改前**代码（进程已加载旧模块），故它仍跑完了整轮自修复循环；
> 触发率数据不受影响。

### 5.3 第二轮删减判断（数据不足，暂不删）

审计口径要求"**长期** 0 触发或长期误杀者删/降"。**一轮不足以定"长期"**：
run56 里 8 个机制记 0，但其中 `exec_convergence`/`rename_debt` 已被降级（0 是**期望**，
非删信号）、`curiosity_hint` 默认 off（0 无信息量）。真正可作为删候选的是
`belief_convergence`/`effort_floor`/`decider_stop`/`failure_budget`，样本量仅 1 轮。

**结论：先不删。** 建议再跑 3–4 轮同类观测 run，凑齐样本后再按同一个 `grep` 口径
定"长期 0"。

**已做**：A6 —— run56 里 7/7 bypass 说明长程路径上这道门零信息，已把它的长程特判
整段收掉（连 trace 都不留），`_long_horizon_keep_going` 特判点 2→1。这是本条"按数据
删减"原则的第一次实际兑现。

### 5.4 报告面「诚实」风险 → 已实现 citation 门（C2）

run56 的 `goal_judgment`（引擎自己的 F2/F17 判据）给出 **score=0.2、achieved=false**，
gaps 直指"**缺乏真实运行的数值实验证据……仅呈现了经过理想化处理的摘要表格，无法验证
可复现性**"，reasoning 定性为"**基于假设的模拟推演，而非真实数值实验**"。即诚实门
（C2）在**终点**抓住了它。但报告 `Results` 里那张干净的 N_c(w) 表（12/15/20/25/32/45）
与实际 per-iteration 证据（nobj 多为 1–2，末轮 execute 全是超时、零证据）明显不匹配。

**根因**：`_report` 组装作者提示时只给 `_last_execution_result`（**末轮**），中间轮的真实
数值全丢 → 书生凭印象编表（run56 末轮零证据，报告照样满表）。

**已实现（C 族，按 §3 过闸：只判"证据算不算数"，不替书生做科学判断）**：

1. **证据台账**（`engine_act.py::_append_execution_ledger`，随 `_record_provenance`
   每次真实 execute 追加）：留 `objectives/summary` 等数值面，丢脚本体，容量/长度封顶；
   随 run 在 `cognitive_loop._setup_run_state` 重置（`engine.py` __init__ 初始化）。
2. **提示面**（`_build_science_report_prompt`）：把**本轮每次** execute 的台账注入
   `## Execution Evidence Ledger`，并加一条硬性 `CITATION RULE` —— Results 数值只能取自
   台账，无出处的必须显式声明"no execution evidence"，不许编/插值/理想化。
3. **观测面**（`_report`）：`_citation_gap` 取 Results 节里 |值|≥10 的数字（忽略章节号等
   结构小整数），与台账比对，落 `report_citation` trace（`untraceable=gap/total`）。
4. **标注门**（唯一"硬"动作，只标注不改结论、不终止）：未溯源数 ≥3 且占比 ≥50% 时，
   报告末尾附 `## Citation Audit` 告示（与既有 P0-5「零执行」同一风格）；报告同时自包含
   `## Execution Evidence Ledger` 供读者核对 `[ev#]` 引用。

单测：`test_execution_ledger_appends_drops_script_and_caps` /
`test_citation_gap_flags_untraceable_numbers` /
`test_citation_gap_zero_when_numbers_traceable` /
`test_science_report_prompt_includes_citation_rule_with_ledger` /
`test_report_flags_untraceable_numbers_and_annotates`。

> 与 §3 原则一致：这是**证据门**（不是科学判断门），且只在"查无出处"时标注，不越权替
> 书生下结论；`report_citation` 触发率进同一观测口径，供后续按数据决定是否收紧/放宽。

---

## 6. 目标达成硬出口：合并 + 验收门（v24）

§3 原则说"硬终止只保留两个出口：挂钟预算耗尽 与 **目标达成**"。A1–A6 降级后，
"目标达成"成了唯一保留的**语义硬出口** —— 但它的实现本身是散的，必须先收干净，
否则原则只落了一半。

### 6.1 问题：目标达成判定有三套并行实现，且默认口径不可信

循环内判"是否完成"此前有 **三条各自独立** 的实现，各自重算、各自写 `should_stop`：

| 路径 | 位置 | 口径 | 问题 |
|------|------|------|------|
| 默认散装 F2 + v10-F17 | `cognitive_loop.py`（旧） | **规则版** GoalJudge（关键词覆盖） | 与出口路径的 LLM 版不一致 → 循环内误判 |
| Unified + Arbiter | `HUGINN_USE_UNIFIED_DECISION=1` | 统一评估器 | 独立重判 goal，与默认路径结论可能打架 |
| CompletionGate | `HUGINN_USE_COMPLETION_GATE=1` | 完成门 | 同上，且第三条 stop 来源 |

三套并存 = 控制面分叉（§1 的病灶在同一处复发）。且默认路径用规则版判据，
是**唯一合法自动出口**上最不该省的地方。

### 6.2 动作（用户选定四项，组合实施）

1. **修目标达成硬出口** —— 循环内改用 **LLM 版 GoalJudge**（`verification_model`
   → `model` 降级；无模型才退规则），判据是**证据台账 + 本轮产出**，不再是中间摘要的
   关键词覆盖。与出口路径口径对齐。
2. **用证据台账做验收门** —— "完成"与"验收"分离：judge 说达成后，先过
   `_ledger_has_finite_evidence`（台账须至少一条**含有限数值**的真实执行证据；
   `inf/nan`、纯文本错误、exec 结构字段如 `exit_code` 都不算）。
3. **加独立对抗验收 SKEPTIC** —— 证据门过后，用**独立** LLM 走
   `adversarial_critique` 尝试**证伪**声明（核对每个数值能否在台账查到出处；
   优于合理基线的数值一律 red flag），未过则拦回、反例回灌为 hint。
4. **合并三条停止路径** —— 收敛为单一出口 `_evaluate_completion`：
   `GoalJudge → (可选变体消费同一 judge 结果) → 验收门`，默认路径不再各自 `should_stop`；
   Unified / CompletionGate 退化为方法内两个**变体**，只决定 stop 之外的
   switch_tool/requery，stop 只有一个来源。`_last_completion_iter` 去重同迭代重复判定。

判定顺序（`_evaluate_completion`，每 3 轮到期一次）：

```
GoalJudge(LLM)  →  achieved?  ── 否 ──→  hint=gaps, 返回 (不停)
                      │是
                      ▼
        _accept_completion(验收门)
          ├─ 证据门: 台账无有限数值 → block (goal_acceptance)
          ├─ 元认知完成度审计       → block (goal_metacog_audit)
          └─ SKEPTIC 对抗审查       → 未过 block (goal_skeptic)
                      │全过
                      ▼
              stop=True  →  调用方写**一次** should_stop
```

### 6.3 定性（按 §3 过闸）

- **验收门属于 C 族（诚实/证据门）**：只判"这条证据算不算数"，不替书生做科学判断
  → 允许硬。
- **合并是减控制面**：3 条 stop 来源 → 1 处写 `should_stop`；2 条可选变体降为方法内分支，
  不再是独立判停路径。
- **唯一语义硬出口**：目标达成 + 证据 + 独立对抗三者齐备才收结；其余一律提示。

### 6.4 单测

`test_ledger_has_finite_evidence_accepts_finite_objectives` /
`..._rejects_nonfinite_and_text` / `test_ledger_evidence_text_renders_indexed_lines`
（证据口径）；`test_evaluate_completion_stops_only_with_evidence` /
`..._blocks_without_execution_evidence` / `..._skeptic_can_block` /
`..._dedups_same_iteration`（端到端）；`test_cognitive_loop_has_single_completion_exit`
（回归守卫：旧散装 F2/F17 与 `_use_unified_decision` / `_use_gate` 不得复活）。

### 6.5 待办：起观测 run 采新出口触发率（✅ 已起 run58）

run57 已于 `2026-10-01 08:26` 结束（55 条 trace：`darwin_stagnation` 26 /
`belief_convergence` 22 / `effort_floor` 6 / `report_citation` 1），但它是**改前**代码
（v24 改动落盘于 09:24），故 **`goal_judge` / `goal_acceptance` / `goal_skeptic` 全为 0**，
新出口尚无野外样本。**下一步**：起一轮同参数 run，确认这四条新 trace 能被点亮，
并把触发率并入 §5.3 的第二轮删减决策。

**run58 已起**（同 objective / 同旋钮 / 3600s），开跑即点亮 `goal_judge`（前 7 轮已 2 次）
→ v24 单一出口在野外生效。

---

## 7. 多智能体协作：意图写了，线从没接（v25）

起因：讨论"是否需要更多智能体协作"。查证后发现，**生产路径上协作角色是 0 个** ——
不是"要不要更多"，而是已有的全部没通电。

### 7.1 断电链路

| 环节 | 事实 |
|------|------|
| CLI 构造引擎 | `AutoloopEngine(workspace=obj.workspace)`，**不传 `agent_factory`**（[`autoloop.py`](agent/huginn/cli/commands/autoloop.py)） |
| 引擎默认 | `agent_factory: Any = None`（[`engine.py`](agent/huginn/autoloop/engine.py) `__init__`） |
| 盲重建 | `if self._agent_factory is None: return` —— 静默空转 |
| failure_inverter | 同上 |
| BranchIncubator | `if self._agent_factory is None: return None` —— 静默回退 2 路 |

`engine.py` 注释写着"由 RCBench runner / CLI 在需要 N=3 隔离采样时注入"——
**意图存在，线从没接**。故 runs 50–58 全是纯单 agent，是漏接，不是判断。

### 7.2 已写好的协作资产（无需新造）

- [`subagent.py`](agent/huginn/agents/subagent.py) 内置 6 spec：`explore / coder / analyst /
  support / blind_reconstructor / failure_inverter`，含递归深度保护、token 摘要。
- [`branch_incubator.py`](agent/huginn/metacog/branch_incubator.py)：N 路隔离探索
  （`ContextBundle` 裁上下文，族间互不可见），并发 `asyncio.gather`。
- [`personas.py`](agent/huginn/personas.py) 8 persona + [`templates.py`](agent/huginn/workflows/templates.py)
  `reviewer_workflow` 4 阶段流水线（库，主循环未用）。

### 7.3 动作：接线并门控（默认关）

`HUGINN_ENABLE_AGENT_COLLAB=1` 时，CLI 才 `get_agent_factory()` 注入；
默认 `None` → 默认行为与成本**零变化**（符合 §3：先证明挣得到成本，再谈开）。
工厂构造失败 fail-open 回退单 agent。单测锁定：
`test_default_off_returns_none` / `test_flag_on_builds_factory` /
`test_factory_failure_falls_back_to_single_agent`。

### 7.4 A/B 测量（run59/60/61 已完成，结论：无显著增量）

接线只是前置。真正要回答的是"增加协作有没有增量"。**已跑 A/B 三组**
（同 objective / 同旋钮 / 同 3600s / 同继承起点 run57 的 `evolved_skills.json`）：

| run | 协作配置 | 时长 | `control_trace` | 换名审计命中 | `report_citation` | 达成 |
|-----|---------|------|-----------------|------------|-------------------|------|
| run59 | collab ON + `BLIND_RECONSTRUCTION=1` | 满预算耗尽 | `effort_floor`×1 | 2 | — | 未达成 |
| run60 | collab ON + `USE_BRANCH_INCUBATOR=1` | 满预算耗尽 | `effort_floor`×1, `report_citation` 4/6 | 3 | 4/6（`annotate`） | 未达成 |
| run61 | **baseline**（collab off） | 满预算耗尽 | `effort_floor`×1, `report_citation` 3/8, `goal_judge`×1 | 4 | 3/8 | `achieved=False score=0.0` |

**判读**：

1. **三组均满预算耗尽、均未达成目标**，差异不显著（每组 n=1，换名命中 2/3/4 无趋势）。
   单轮 A/B 不足以判定"协作有效/无效"——**先不下结论**。
2. **决定性发现：三组 run.log 里 `collab_*` trace 全为 0。** 但 run59/60 明明开了
   `HUGINN_ENABLE_AGENT_COLLAB=1`。查证：该批 run 跑的是**加 trace 之前**的代码，
   协作跳过路径当时全是**静默 return** → 野外根本分不清"没通电"与"跑了但没结果"。
   这是本轮真正的产出：**协作的 A/B 此前无法测量，因为观测面是黑的**（见 §7.5）。

---

## 8. 协作可观测性 + code_lab 超时（v26）

两大类问题一起修：**协作机制静默空转看不见**，**code_lab 超时被当成语法 bug 反复喂同样的重代码**。

### 8.1 协作跳过路径补 trace（D 观测类，按 §3 过闸：只观不判）

此前盲重建 / failure_inverter / BranchIncubator 的每个"不适用"分支都是**静默 return**。
野外只看到上层"returned None, fallback"，分不清是**没通电**、**跑了没结果**、还是**条件不满足**。
现给每条跳过路径补 `campaign.control_trace`（`action=skip`，`evidence` 携带原因）：

| 机制 | 位置 | 新增 skip 原因 |
|------|------|---------------|
| 盲重建 | [`engine_reflect.py`](agent/huginn/autoloop/engine_reflect.py) `_blind_reconstruct_verify` | 无 `_current_hyp_id_for_plan` / 节点非 `untested` / 无 `agent_factory`（协作未开）/ dispatch 抛异常 / 返回空 |
| 失败反推 | 同上 `_invert_failure_trace` | 跳过原因 |
| BranchIncubator | [`hypothesis_loop.py`](agent/huginn/autoloop/hypothesis_loop.py) `_hypothesize_via_branch_incubator` | 无 `agent_factory` / 导入失败 / `run_round` 抛异常 / `empty:`（跑了但 N 路全空手，带 `branches`/`ok` 计数） |

统一入口：`EngineReflect._emit_control_trace`（经 `engine.py::_emit_control_trace` 供协作对象
`__getattr__` 转发）。三者均为**纯观测**，不改任何决策路径。

### 8.2 control_trace 增加 OTel 遥测面（对齐 Langfuse）

`_control_trace` 除既有 `campaign.control_trace` 事件 + WARNING 日志外，新增
`get_telemetry_collector().add_event("control_trace", name=..., iteration=..., evidence=..., action=...)`
→ 配了 `HUGINN_OTEL_ENDPOINT`（Langfuse: `https://cloud.langfuse.com/api/public/otel/v1/traces`）
即可在 Langfuse 按 `name` 检索。双通道互补：**日志离线可 grep，遥测在线可查**；fail-open。

> **修了一个静默 bug**：`TelemetryCollector.add_event(self, name, **metadata)` 的位置参数名
> 恰是 `name`，而调用方传 `add_event("control_trace", name=<机制名>, ...)` → 位置 + 关键字
> 撞名 `TypeError`，被 fail-open 的 `except` 吞掉 → **遥测面实际从未落过一条事件**。
> 修法：位置参数改名 `event_name`，让 metadata 可自由含 `name`。单测
> `test_control_trace_emits_telemetry_event` 锁定（此前红，现已绿）。

### 8.3 code_lab 超时：对症提示 + 观测

**现象**（run59/60/61 实测）：三组 run.log 各有 **5 次** `code_lab 执行超时`，是最高频失败模式；
run61 iteration 14 的 execute 单阶段烧掉 **3973s**、修复循环被"挂钟预算耗尽"中止。

**根因**：[`code_lab.py`](agent/huginn/research/code_lab.py) `build_author_prompt` 对**超时**仍套用
NameError/形状那套**语法类对症提示** → 书生以为代码写错，重生同样重的扫描 → 反复超时。
超时是**算力预算**问题，跟"代码写错"是两类失败，必须分开。

**修法**（命题无关，只谈算力、不碰科学判断）：
`_timeout_like` 命中时单独回灌**可执行的降算力**指令（砍扫描组合数 / 降 `mlp_fit.maxiter` /
先跑最小可判配置），不再提语法。并新增 `code_lab_timeout` trace（`action=advisory_hint`，
带 `attempt=x/y`）量化"被超时饿死的迭代占比"。单测
`test_build_author_prompt_timeout_hint_reduces_compute` /
`..._bug_hint_keeps_syntax_advice` 锁定两条分支不串。

### 8.4 复验 run（进行中）

run59/60/61 是**改前**代码（无 `collab_*`/`code_lab_timeout` trace，且无超时提示修复）。
故起 **run62（collab ON + 盲重建）/ run63（baseline off）**，同 objective / 同旋钮 /
同 3600s / 同 run57 继承点，用新代码复验：

**开跑即点亮，观测面立刻抓到两条真问题**（run62）：

| iteration | `collab_blind_reconstruct` evidence | 含义 |
|-----------|-------------------------------------|------|
| 4 | `skip: no current_hyp_id_for_plan` | **接线缺口**：verify 时 `_current_hyp_id_for_plan` 未置位 → 协作即使通电也常被跳过 |
| 9 | `skip: dispatch returned nothing success=True summary_len=0` | **跑了但空手**：盲重建子 agent 派发成功、却返回空 summary → 零产出 |

run63（baseline）`collab_*` 为 **0**，符合预期：调用点由 `HUGINN_BLIND_RECONSTRUCTION`
门控（[`engine_reflect.py:782`](agent/huginn/autoloop/engine_reflect.py)），未开则连方法都不进。

> 这正是补 trace 的价值：此前只能说"协作 returned None, fallback"，现在能精确定位到
> **是没通电 / 没置位 / 跑了空手**。两条都指向下一步要修的是**接线**（`_current_hyp_id_for_plan`
> 的置位时机 + 子 agent 空 summary），而不是加更多协作角色。

- `code_lab_timeout` 待本轮 execute 阶段出现后并入统计；
- 数据并入 §5.3 第二轮删减决策。

单测全绿：`tests/test_engine_decomposed.py` **83 passed**（含 §8.1–§8.3 新增 5 例）。