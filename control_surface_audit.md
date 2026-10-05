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
2. **能不能改成提示或观测？** ——能改就改。**科学**硬终止只保留两个出口：
   **挂钟预算耗尽** 与 **目标达成**。此外只允许一类**资源熔断**：它不替书生做
   任何科学判断、只防烧钱 —— 即 `failure_budget`（同类失败连续触顶即停）。
   它不属科学出口, 故不受"两个出口"约束; 但须记 §11.4 ① 的实测: 21 个已接线
   run 零触发。区分标准 = **"停的是不是科学结论"**: 是 → 禁硬; 否 (只是别再烧
   资源) → 可硬。

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
| `convergence_advisory` | A1/A2/A3 合流 (v34) | `kind=surprise/exec/rename`；同模式三合为一，野外计数不再被摊薄。每轮出现 = 长期误杀风险高，降级后仍应观察是否伴随真停滞 |
| `darwin_stagnation` / `belief_convergence` | A3/A4 | 长期 0 触发 → 删 |
| `decider_stop` | A5 | 每轮出现 = 书生常想收结但被拦，需回看是否目标真未达成 |
| `execute_budget_gate` | A6 | `action=bypassed` 占比高 = 阶段门在长程下基本无意义 |
| `failure_budget` | A8 | **资源熔断** (非科学出口, 见 §3 澄清)；触发即停；21 个已接线 run 零触发 (§11.4 ①) |
| `effort_floor` | B4 | 长期 0 触发 → 删；高频 = 门设计或命题不匹配 |
| `curiosity_hint` | B7 | 默认 off；若开启后频繁注入但不改行为 → 删 |
| `pivot_directive` | B1 | 越阈值/等价族打转时注入强制变异令的次数；`cycling=` 见 §11.4 ② |
| `report_citation` | C2 | 报告 Results 未溯源数值数/总数；长期低触发 = 提示面已够，长期高 = 需收紧 |
| `goal_judge` | 出口 | 每次到期完成判定（每 3 轮）一条；`action=stop_candidate` 占比 = judge 认为达成的频率 |
| `goal_acceptance` | 验收门·证据 | `action=block` = 台账无有限数值被拦；长期 100% block = judge 提前宣布达成 |
| `goal_metacog_audit` | 验收门·元认知 | 完成度审计拦截次数 |
| `goal_skeptic` | 验收门·对抗 | `verdict` 分布；`block` 高频 = judge 过于乐观或常识证伪未过 |
| `collab_blind_reconstruct` | 协作·盲重建 | `action=skip` 携带跳过原因；**长期全 skip 或 0 条 = 该协作没通电/没生效**（见 §7.5） |
| `collab_branch_incubator` | 协作·N 路孵化 | 同上；`empty:` = 跑了但 N 路子 agent 全空手 |
| `collab_failure_inverter` | 协作·失败反推 | 同上 |
| `code_lab_timeout` | 证据·算力 | 被沙箱超时饿死的修复尝试数；高频 = 作者提示仍产出过重扫描（见 §7.5） |
| `branch_slice_skip` | D5 预算切片 | hypothesize 侧预算不足跳过可选 slice（PRM 打分 / layer2）；`budget<Ns` = 被拦门槛。高频 = 单 slice 成本逼近挂钟，需查树宽/超时 |
| `code_lab_slice_skip` | D6 预算切片 | execute 侧预算不足跳过 code_lab 修复重写 slice；与 `branch_slice_skip` 同族，分两 phase 便于定位超支源 |
| `progress_invariant` | P3.2 进展不变量 | `action=force_route` + `tail=`；连续 window 轮停在执行前阶段 → 强制推进。高频 = 长期空转（见 run72/73/74） |
| `llm_unavailable` | P1/P2 瞬时故障 | `action=retry_in_place`；瞬时 LLM 故障（限流/过载）拦破坏性 redirect。高频 = provider 不稳，非科学停滞 |

> **机制清单的代码权威源**：上表是人工可读视图；机器可读的**单一权威登记**在
> [`agent/huginn/autoloop/control_mechanisms.py`](agent/huginn/autoloop/control_mechanisms.py)
> 的 `MECHANISMS`，计数受 `CONTROL_MECHANISM_BUDGET`（only-shrink）约束，由
> `tests/test_control_mechanism_governance.py` 强制：**发射未登记 / 登记无发射 /
> 越预算 / 非白名单的 `effect=stop`** 均测试红。增删机制先改代码登记表，再同步本表。

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

### 8.5 修 run62 暴露的两处接线（v27）

§8.4 的两条 trace 各自指向一处**接线**缺陷，已按最小改动修掉：

| 现象（run62） | 根因 | 修法 |
|---------------|------|------|
| `skip: no current_hyp_id_for_plan`（iter 4/14/19） | [`cognitive_loop.py`](agent/huginn/autoloop/cognitive_loop.py) `add_hypothesis` 对**空壳/精确重复**陈述返回 `None`，旧代码把它直接写回 `_current_hyp_id_for_plan` → **清掉上一轮的有效 id**，下游盲重建/plan_check 误判"无当前假设"而跳过 | 置位前加非空守卫：只有 `cog["current_hyp_id"]` 为真才覆盖 |
| `skip: dispatch returned nothing success=True summary_len=0`（iter 9/24/29） | [`subagent.py`](agent/huginn/agents/subagent.py) `_extract_output` 只取**最后一条**消息；子 agent 用尽 tool 预算时末条是 `content=""` → 整个产出被判为 `""` | 反向遍历取**最后一条非空** content；全空才返回 `""` |

两处均为**接线修复**，不改任何科学判断，也不新增控制面硬点（按 §3 过闸）。

验证：
- `_extract_output` 反向取非空 — 新增回归守卫
  `TestSubagentDispatch::test_extract_output_takes_last_non_empty_message`（绿）。
- 协作跳过路径 trace 未被破坏：
  `test_blind_reconstruct_skip_emits_control_trace` / `test_branch_incubator_empty_candidates_emits_trace`
  等 `tests/test_engine_decomposed.py` 相关 6 例全绿。

> 注：本轮验证在无 `.venv` 的沙箱内进行，仅装了 import 链最小依赖
> （pydantic/numpy/cryptography/langchain*/langgraph/networkx/pyyaml）后跑**定向**用例；
> 全量 `83 passed` 的结论仍来自 §8.4 的运行环境，不在本沙箱复核。
> 下一步：起 run64（同 run62 参数、含 v27 接线修复）复验两条 skip 是否转成真产出。

### 8.6 run64 复验：id 修复生效，但 `summary_len=0` 另有真因（v28）

**id 修复已生效**。run64（collab ON，含 v27）实测：iteration 4 仍是
`skip: no current_hyp_id_for_plan`（此时确实还没成功加过假设，**合法**），
而 run62 里同因跳过的 iteration 14 / 19 在 run64 中**不再因 id 跳过**，改为进入派发。

**但 `summary_len=0` 未解决**（run64 iter 9/14/19 仍是
`skip: dispatch returned nothing success=True summary_len=0`）→ §8.5 的
`_extract_output` 反向取非空**不是真因**。

**真因**（单次派发探针 `/tmp/probe_blind_recon.py` 直接观测到流事件序列）：

```
has_msgs=False  keys=['exec_mode','flags','trace_id','type','user_mode']   # mode_banner
has_msgs=False  keys=['material','raw','reason','suggestion','type']       # clarification_request
has_msgs=True   keys=['files','messages']                                  # ← 真状态
has_msgs=False  keys=['_token']                                            # 流式 token
has_msgs=True   keys=['files','messages']                                  # ← 真状态
has_msgs=False  keys=['_auto_continue']                                    # ← 末条! 无 messages
```

[`streaming.py`](agent/huginn/agent/streaming.py) 的 `chat()` 除 langgraph state 外还会
yield 一批**控制事件**（`_token` / `_reasoning` / `_compacted` / `_auto_continue` /
`tool_break`）。子 agent 用尽 tool 预算时会触发 synthetic continue，于是**末条 yield 是
`{"_auto_continue": True}`**。而 [`subagent.py`](agent/huginn/agents/subagent.py) `dispatch`
旧实现 `final_state = state` 取最后一条 → 该 state **没有 `messages` 键** →
`_extract_output` / `_extract_tool_calls` / `_estimate_tokens` 全读到空 →
`success=True` 但 `summary_len=0`、`tool_calls=0`。

**修法**（[`subagent.py`](agent/huginn/agents/subagent.py) `dispatch` 的流循环）：只把
**带非空 `messages`** 的 state 认作 `final_state`（嵌套的 `{"state": {...}}` 解开一层），
从未见过 messages 时才兜底留首条。§8.5 的"反向取非空"仍保留（同 state 内末条常空）。

**探针实测**（同一 statement，修复后）：`success=True, summary_len=26, tool_calls=0`，
且事件序列确认末条是 `_auto_continue`。即：修复前该派发必为 `summary_len=0`。

新增回归守卫：`TestSubagentDispatch::test_dispatch_skips_control_events_when_picking_final_state`
（构造 yield 控制事件的假 agent，断言 summary 取到真状态内容；绿）。
`collab_blind_reconstruct` 空返回 trace 追加 `full_len=` / `tool_calls=` 两个字段，
便于下一轮区分"没拿到状态" vs "拿到了但真为空"。

> 下一步：起 run65（同 run62 参数、含 v28）——预期 `summary_len=0` 消失，
> `collab_blind_reconstruct` 出现真产出（holds/derivation）或至少进入解析分支。

### 8.7 run65 复验：v28 生效，`summary_len=0` 归零，但成功路径漏观测（v29）

run65（collab ON + 盲重建，含 v28）实测：`collab_blind_reconstruct` 的
`summary_len=0` **彻底消失**。但 run65 的 4 条 trace（iter 9/14/24/34）**全是**
`skip: node status=refuted` —— 这是 §8.5 的 `_current_hyp_id_for_plan` 修复**生效后**
的正常结果：当前假设已经被盲重建自己反证过，所以后续轮次在 node-status 闸门就跳过。

**真正证据在 FAILED.md**（成功路径不落 trace，只能从 durable state 反查）：

| run | 盲重建反证条数（FAILED.md `Modality: blind_reconstruction`） |
|-----|------|
| run62（改前） | 0（无 FAILED.md） |
| run64（v27） | 0（无 FAILED.md） |
| run65（v28） | **3** |

三条均为 `blind_holds=False vs orig_holds=True mismatch`，时间戳 15:05 / 15:11 / 15:14；
`hypothesis_graph_loop_*.json` 同步显示 24 节点中 **3 个 refuted**，且每条 refuted 节点的
`evidence` 都带齐 `blind_holds / blind_confidence / blind_derivation / orig_holds /
orig_reasoning_excerpt / derivation_consistent` —— 说明 `dispatch` 确实拿到了**非空**
子 agent 产出并走完了 JSON 解析 + 三档判定。**即 v28 修复经端到端验证成立**：
`summary_len=0` 从 run64 的 3/3 次派发，降到 run65 的 0 次，且产出真结论。

**副作用暴露的观测缺口**：`_blind_reconstruct_verify` 只在 **skip 路径**落
control_trace，成功路径仅 `logger.info`（run.log 只收 WARNING，看不到）。于是
控制面上 run65 只有"全 skip"，若不去翻 FAILED.md / 假设图，会**误判机制空转**。
这正是 §8 要解决的"开没开、跑没跑"看不清的问题。

**修法（v29，纯观测，按 §3 过闸：只观不判）**：
- 反证分支落 `action="refute"` trace（含 `blind_holds` vs `orig_holds` / `summary_len`）；
- 支持分支落 `action="support"`（legacy / strong 两档）；
- derivation 冲突档落 `action="weak"`；
- `statement` 过短（原**静默 return**）也补一条 `skip: statement too short` trace。

至此 `collab_blind_reconstruct` 在控制面上覆盖 **skip / refute / support / weak** 全分支，
触发率统计不再系统性低估。

> 下一步：起 run66（同参数、含 v29）——预期 run.log 直接出现
> `control_trace ... action=refute`，不必再翻 FAILED.md。

### 8.8 run65 终态 + replay_audit 判据对齐 A2（v30）

**run65 终态**（iteration 148）：假设图 **128 节点**（122 untested / **6 refuted**），
无 PROVED.md；`FAILED.md` 6 条全是 `Modality: blind_reconstruction` 反证。即 v28 后
盲重建产出 **6 次真反证**（run62/64 均为 0）。但 run.log 里 23 条
`collab_blind_reconstruct` **仍全是 `skip: node status=refuted`** —— 6 次成功一条都没
上控制面，正是 §8.7 要补的那条 trace；v29 修的就是它。

**replay_audit 与 A2 脱节**（`replay_audit` 即 §4 D2 "离线重放判据"）。用它重放 run65，
发现判据还停在 A2 降级**前**：

- v11 结论文案写"单调债务越界 ⇒ 第 9 次换名触发 `conclude+stop` ⇒ **必然终止**"，
  而 §A2 早已把该出口降级为 advisory（[cognitive_loop.py](file:///workspace/agent/huginn/autoloop/cognitive_loop.py#L3709-L3736)
  注释明写"不再自动终止 run"）。run65 实测 `可终止出口=0`，与其自相矛盾。
- `scan_runlog` 把带 `advisory only ... no stop` 的行也计入 `terminal`（`rename debt`
  命中 `TERMINAL_MARKERS`）⇒ 出口体检假阳性风险。
- v12 verdict 把"旧码记录 11 次换名"当单值，但 11 是**离线反推**值；run.log 实际只记
  了 **4** 次 streak 事件。用不同口径算，差集是 0 还是 6，结论差很大。

**动作（v30，纯观测口径修正，不改任何终止行为）**：

- 模块/`replay_rename_debt` 口径改为 **A2 前反事实**，新增 `stops_live=False` 显式标注；
- `scan_runlog` 排除 `advisory only` 行，`terminal` 不再把提示当出口；
- 出口体检由"是否终止"改判 **「无进展是否可观测」**，并新增 `control_traces` 统计
  （`control_trace name=...` 落盘计数）作为实测证据；
- v12 verdict 改成给**区间**（记录事件口径 / 反推口径两个差集），不再伪装单值。

**重放复验（run62/64/65）**：出口体检现在直接给出落盘 trace ——

```
run62: collab_blind_reconstruct×6, effort_floor×2, goal_judge×2, code_lab_timeout×2,
       darwin_stagnation×1, report_citation×1
run65: collab_blind_reconstruct×23, belief_convergence×20, darwin_stagnation×14,
       goal_judge×9, effort_floor×5, report_citation×1
```

run65 **无 `rename_debt` trace** ⇒ 线上债务从未越限（与离线反推的 11 次换名不符），
这正说明离线反推的口径偏大 —— 故 v12 差集给区间是必要的。另 run65 仍有
`surprise 恒定 [1.0]`（路由退化）与 `JEPA structure_desc 全 0`（结构通道无信息）两条
待查。

### 8.9 两条待查的根因（v31，含一处纯观测修正）

**① `surprise 恒定 [1.0]`（"路由信号死"）——根因是三层回落叠加，不是相对秩本身**

用 run65 实测数据逐层坐实（不是推断）：

1. `sentence_transformers` 未安装，且 [store.py](file:///workspace/agent/huginn/knowledge/store.py#L244-L269)
   的 `_EmbeddingModel._st` 需**别处先调 RAG embed 才会被懒加载**；而
   [engine_reflect.py](file:///workspace/agent/huginn/autoloop/engine_reflect.py#L2146-L2157)
   的 `_try_embed_text` 只读 `_st`、**刻意不触发加载** ⇒ `_semantic_distance` 恒 `None`。
2. JEPA predictor 权重（`jepa_predictor.json` / `jepa_span_predictor.json`）在 run 的
   runtime home 下**不存在** ⇒ `_predictor_surprise` 恒 `None`。
3. 于是 source 落到 `"jaccard"`，`surprise = robust["worst"] = max(4 路扰动 jaccard)`。
   预测文本（计划里的定性预测）与实际输出（数值倾倒）token/bigram 集多半不相交 ⇒
   bigram 那路 jaccard = 1.0 ⇒ **`worst` 饱和在 1.0**。

实测证据：`engine_state` 里 `_surprise_history` 的取值**只有 0.0 与 1.0**（连续语义
距离绝不会恰为 0/1）⇒ 确证走的是 jaccard；run65 `run.log` 30 条 `exec-route` 里 25 条
`explore`，其中 7 条是 `[auto-routed: surprise=1.00]` 强制改道。

相对秩（`_relative_surprise`）**不是**路由输入（路由读 `_last_surprise` 的**原始值**），
但它自己也有一个数学缺陷：旧码把当前样本并入直方图后算 `le/len`，于是"当前值为历史
最大（含并列）"一律 = 1.0 ⇒ 恒定信号恒 1.0。episodic shard 实测 49 轮里 **34 轮 = 1.0**。

**修（v31，纯观测，只动相对秩）**：`_relative_surprise` 改**中位秩、排除自身**：
首个样本 → 0.5（中性），恒定 → 0.5，递增 → 1.0，递减 → 0。单调性与 `[0,1]` 契约不变，
影响面仅 episodic 的 `surprise` 字段（`_relative_surprise` 只此一个调用点）。

**修（v31，行为变更，已确认）**：原始饱和值才是触发源。新增**单一自由函数**
[signals.routing_surprise()](file:///workspace/agent/huginn/autoloop/signals.py#L116-L133)
—— 统一返回**秩归一** `_last_surprise_rel`（秩未产出时回落原始值保旧行为），并把
**全部 9 个行为消费点**收口到它（除赋值 / 持久化 / late-binding 外，无一处再直接读原始
`_last_surprise`）：

- **路由**：[plan_check.py:372](file:///workspace/agent/huginn/autoloop/plan_check.py#L372)（软提示 `>0.5`）、
  [plan_check.py:436](file:///workspace/agent/huginn/autoloop/plan_check.py#L436)（硬改道 `>0.9`）、
  [hypothesis_loop.py:2776](file:///workspace/agent/huginn/autoloop/hypothesis_loop.py#L2776)（`should_imaginate` `>0.5`）、
  [hypothesis_loop.py:3046](file:///workspace/agent/huginn/autoloop/hypothesis_loop.py#L3046)（`reviewer` persona `>0.6`）
- **记忆**：[cognitive_loop.py:3784](file:///workspace/agent/huginn/autoloop/cognitive_loop.py#L3784)（episodic 快照）、
  [engine_observe.py:693](file:///workspace/agent/huginn/autoloop/engine_observe.py#L693)（replay cue）、
  [engine_observe.py:833](file:///workspace/agent/huginn/autoloop/engine_observe.py#L833)（PMK 冲突落盘）
- **展示**：[engine_reflect.py:3758](file:///workspace/agent/huginn/autoloop/engine_reflect.py#L3758)（报告 prompt "Surprise score"）、
  [slash_commands.py:975](file:///workspace/agent/huginn/cli/slash_commands.py#L975)（`/status` 表）
- **触发（flag off）**：[engine_reflect.py:3011](file:///workspace/agent/huginn/autoloop/engine_reflect.py#L3011)（桥 A）——
  旧阈值 `2.0` 在任何口径下都**不可达**（surprise ∈ [0,1]），该 bridge 实际恒不触发；
  改按秩刻度取 `0.9`（与硬改道一致），flag 仍默认 off，默认行为不变。

用 `getattr` 容错：协作对象经 `__getattr__` 转发到 engine，stub 缺字段拾默认值、不做硬依赖
（`tests/test_engine_decomposed.py` 的 `_StubEngine` 即靠此通过）。

语义：**"当前值相对历史分布异常"**取代**"绝对饱和"** —— 恒定信号 → ≈0.5 不触发，
只有真正相对异常才逼近 1.0。回归：`test_lucid_prereqs` / `test_engine_decomposed` /
`test_engine_signals` / `test_cross_scale_invariance` / `test_hypothesis_loop` /
`test_loop_inspired` / `test_reflection` / `test_structure_tool` / `test_cognitive_engine`
等 **431 passed, 3 skipped**（旧测试设 `_last_surprise=0.95` 仍如期强制 explore ——
因秩未就绪时回落原始值）。

**② `JEPA structure_desc 全 0`（"结构通道无信息"）——多数是误报，少数是接线未完成**

- run65 的目标是**纯机器学习/数学**（解空间刚性），全程 0 次 `cognitive_map` 调用，
  `structure_cognitive_map_tool._MAPS` 为空 ⇒ 无活跃 `StructureCognitiveMap` ⇒ 全 0 是
  **正确行为**（该通道本就只编码晶体/材料结构）。replay_audit 的"结构编码恒零"对它
  是**误报**：审计器不知道目标类型。
- 接线本身确实**未完成**：[cognitive_checks.py](file:///workspace/agent/huginn/autoloop/cognitive_checks.py#L66-L83)
  的 `_snapshot_structure_desc` 只从 `cog` 里找 `structure_cognitive_map/cmap/structure`
  三个键，而 `cog` **从不携带**这些键（映射实际存在
  `structure_cognitive_map_tool._MAPS`，引擎已有 `_get_active_cognitive_map()` 可取其最近者）。
  即使跑结构类目标也会恒 0。

**修（v31，接线补全）**：`_snapshot_structure_desc(cog, cmap=None)` 增加显式入参；
调用点 [cognitive_loop.py:3775](file:///workspace/agent/huginn/autoloop/cognitive_loop.py#L3775)
传入 `self._get_active_cognitive_map()`。验证：给一个带 `lattice` 的假 map 后
`[1.0, 0, 0, 0, 0, 0.978…]`（配位数 / 空间群 225 归一）—— 通道在结构类目标下可用；
非结构目标仍全 0（正确）。

---

## 9. 协作"跑了但空手"的真因：internlm 消息序（v32）

### 9.1 现象：§8 所有"空手"都指向同一串

run69（含 v31）实测控制面：

| iteration | trace | 含义 |
|-----------|-------|------|
| 24 | `collab_branch_incubator ... evidence=empty: branches=3 ok=0` | 3 路孵化全空手 |
| 4 | `collab_blind_reconstruct ... refute: blind_holds=False vs orig_holds=True summary_len=26` | 反证依据只有 26 字 |
| 9/14/19 | `collab_blind_reconstruct ... skip: node status=refuted` | 后续因节点已 refuted 跳过 |

26 字正是 `in prompt processing error` 的长度 —— 即**子智能体整轮返回的是错误串**，
不是空产出。§8.5/§8.6 修的 `_extract_output` / `skip control event` 都是**必要但不充分**：
它们解决"拿到真状态却读到空"，这次是"模型压根没产出"。

### 9.2 真因：端点只收「至多一条、且在首条」的 system 消息

三层坐实（非推断）：

1. **合成探针** `_scratch/test_msg_order.py` 对 `intern-s2-preview`：

   | 消息序 | 结果 |
   |--------|------|
   | `[S,H]` / `[S,H,H]` | `HELLO_RIGIDITY` |
   | `[H,S,H,S]` / `[S,S,H]` / `[S,S,S,H]` / `[H,S,A,H,S]` | `in prompt processing error` |

   即：多条 system，或 system 不在首条，一律报错。

2. **真实 agent 路径** `_scratch/test_subagent_chat.py` 抓到实际消息序：
   `[H, S(风格指令), H(inner_state), S(预算)]` —— 与探针命中的形状完全一致，
   模型回复 `in prompt processing error`。

3. **链路成因**：图框架把顶层 `system_prompt` 单独前置（langgraph 的
   `_get_prompt_runnable`：`[prompt] + llm_input_messages`），而 huginn 又在会话中间
   注入风格 / 预算 / context hints 等 system → 最终 prompt 天然是 `[S, H, S, …]`。
   这是**所有** OpenAI 兼容端点的公共路径，不是 internlm 独有写法。

### 9.3 修法：模型边界归一（不碰图拓扑）

按 §3 过闸：这不是科学判断，也不是新控制面 —— 是让「该通的线」通起来。

- [`registry._merge_system_messages`](file:///workspace/agent/huginn/models/registry.py)：
  把全部 `SystemMessage` 正文按出现顺序合并为**一条**置于队首，其余消息保持相对顺序；
  未命中（无需改动）时原样返回同一列表对象。
- `_normalizing_chat_openai_cls()`：`ChatOpenAI` 子类重写 `_get_request_payload`
  —— 它是 `invoke/ainvoke/stream/astream/bind_tools` 的**唯一出口**，一处覆盖
  deepagents / react fallback / 子智能体全部路径。**每次现建子类、import 在函数内**，
  以便测试 monkeypatch `langchain_openai.ChatOpenAI` 时取到被替换的基类。
- 所有 OpenAI 兼容构造点统一改用该子类：`openai/vllm/local`、`deepseek`、`openrouter`、
  `_LOCAL_PRESETS`、`_DOMESTIC_OPENAI_COMPATIBLE`（[registry.py](file:///workspace/agent/huginn/models/registry.py)）。

### 9.4 验证

| 探针 / 用例 | 改前 | 改后 |
|-------------|------|------|
| `_scratch/test_msg_order.py` | 4/6 报错 | **6/6 全绿** |
| `_scratch/test_subagent_chat.py`（真实 chat） | 错误串 | `HELLO_RIGIDITY`（tools 开/关均） |
| `_scratch/test_branch_dispatch.py`（dispatch） | `summary_len=0` | `success=True summary_len=1182 tool_calls=2` |
| `tests/test_domestic_llm.py::TestSystemMessageNormalization` | — | 5 例绿 |
| 模型 / 引擎回归（model_tier / model_router / model_thinking / engine_decomposed / engine_signals） | — | **131 passed** |

> 下一步：run70（同 run69 参数、含 v32）复验
> `collab_branch_incubator` 是否 `ok>0`、`collab_blind_reconstruct` 是否出现
> `action=refute/support` 真产出（而非 26 字错误串）。

## 10. run70 暴露的两处「通而未电」（v33）

run70 复验确认 §9 修复生效：`collab_branch_incubator ... branches=3 ok=2`
（错误串已消除，子智能体真产出）。但同一轮暴露两处「线接上了、电没到」：

### 10.1 PRM rollout value 恒空（valued=0）

- **现象**：`collab_branch_incubator ... valued=0 winner_value=None` —— 有分支、
  有候选，但树搜索的剪枝价值全空，退化成按 token 剪枝。
- **根因**：[step_verifier.make_default_llm_chat_fn](file:///workspace/agent/huginn/runtime/step_verifier.py)
  硬编码 `provider="deepseek", model_name="deepseek-chat"`。run70 用
  `HUGINN_PROVIDER=internlm` 且未配 `DEEPSEEK_API_KEY` → 构造抛错 → 返回
  `None` → `make_branch_value_fn` 恒 `None` → `valued=0`。
- **修法**：改为跟随当前配置 —— `huginn.llm.get_model()` 读
  `get_config()`（`HUGINN_PROVIDER/MODEL/BASE_URL`），与主 agent 同源；
  另设 `HUGINN_PRM_PROVIDER / HUGINN_PRM_MODEL` 显式覆盖（verification 与
  main 异槽，保留「不用同一模型自评」的设计意图）。

### 10.2 S7 meta-critique 跨事件循环（bound to a different event loop）

- **现象**：`meta_critique failed: <asyncio.locks.Event ...> is bound to a
  different event loop`。S7 自修改恒走 except → 全 reject → 死循环。
- **根因**：reflection 的 S7 分支用
  [async_bridge.run_async](file:///workspace/agent/huginn/utils/async_bridge.py)
  在**独立线程开新 loop** 跑 `_handle_s7_self_modify`；该 handler 复用
  `self.model`，其 httpx `AsyncClient` 绑定在创建它的**主 loop** 上 → 换 loop
  必炸。（此前「改 await」的修法未落到调用点，`run_async` 仍在。）
- **修法**：新增
  [`_schedule_s7_self_modify`](file:///workspace/agent/huginn/agent/reflection.py)：
  有 running loop 时 `loop.create_task` 挂**主循环**（fire-and-forget，S7 副作用
  是写 stable_principle / rejection 并回 S1，不阻塞本轮反思）；纯 sync CLI
  无 loop 才退回 `asyncio.run`。持强引用防 task 被 GC。

### 10.3 验证

| 探针 | 改前 | 改后 |
|------|------|------|
| `step_verifier.py` 自检 | — | All passed |
| `make_default_llm_chat_fn`（internlm 环境） | 抛错→None | `model_name=intern-s2-preview` |
| `_schedule_s7_self_modify`（in-loop / no-loop 探针） | — | 主 loop 命中 / fallback OK |
| **run71**（同 run70 参数） | `valued=0 winner_value=None` | **`valued=3 winner_value=1.0`** |
| **run71** 错误扫描 | `bound to a different event loop` | **NONE（clean）** |

> run71：`control_trace name=collab_branch_incubator iteration=1
> evidence=use: branches=3 ok=3 valued=3 winner_value=1.0 winner_tokens=13819
> depth=2 action=use`.

---

## 11. 通电清单审计：版本感知版（v34）

### 11.1 v1 的口径错误

§3 定的删减口径是"**长期** 0 触发 → 删或降"。第一次审计（v1，跨 79 个 run.log 直接
`grep -o 'control_trace name=[a-z_]*' | uniq -c`）给出 23 个登记机制里 **12 个 0 命中**，
看起来是一条现成的删除清单。**它不能直接用**：

79 个 run 横跨多个代码版本，`control_trace` 本身是**逐步接线**的（§6.5 run58 才点亮
`goal_judge`，§8.4 run62 才有 `collab_*`/`code_lab_timeout`，v29 才补
`collab_blind_reconstruct` 的 refute/support 分支）。于是 0 命中里混着大量
"**当时那条线还没接上**"，与"接了但从不触发"无法区分 —— 照 v1 删会误杀。

run 目录的 mtime 不可用（全部是复制进 `research_outputs` 的时间，清一色 10-04）；
版本轴只能取自 `git log -S`（机制名的首次引入）与 run 目录内嵌日期。

### 11.2 v2 方法（`/tmp/ct_audit_v2.py`）

- 机制名 → `git log --reverse -S'"<name>"' --date=short` 取**首次引入日期**；
- run → 目录内嵌 `2026-*` 日期取 **min**（保守：开跑时线就得在了）；
- 只在 `run_min_date >= intro_date` 的 run 上统计命中；`elig` = 已接线 run 数。

### 11.3 结果（79 run / 23 机制）

| name | intro | hit | elig | e_hit | 判读 |
|------|-------|----:|----:|----:|------|
| collab_blind_reconstruct | 10-01 | 96 | 21 | 8 | 活 |
| darwin_stagnation | 10-01 | 92 | 21 | 10 | 活 |
| belief_convergence | 10-01 | 85 | 21 | 7 | 活 |
| goal_judge | 09-24 | 43 | 79 | 11 | 稀有 |
| effort_floor | 10-01 | 39 | 21 | 14 | 活 |
| report_citation | 10-01 | 13 | 21 | 13 | 活 |
| code_lab_timeout | 10-01 | 7 | 21 | 6 | 活 |
| collab_branch_incubator | 10-01 | 7 | 21 | 6 | 活 |
| execute_budget_gate | 10-01 | 7 | 21 | 1 | 稀有（长程已收，见 §4.3） |
| stall_as_action | 10-03 | 3 | 5 | 1 | 活 |
| hypothesis_status_writeback | 10-03 | 2 | 5 | 1 | 活 |
| **failure_budget** | 10-01 | 0 | 21 | 0 | 见 §11.4 ① |
| **pivot_directive** | 10-01 | 0 | 21 | 0 | 见 §11.4 ② |
| **collab_failure_inverter** | 10-01 | 0 | 21 | 0 | 真死候选 |
| **decider_stop** | 10-01 | 0 | 21 | 0 | 已降级，0 为期望 |
| **surprise_convergence** | 10-01 | 0 | 21 | 0 | 见 §11.4 ③ |
| **exec_convergence** | 09-30 | 0 | 25 | 0 | 已降级，0 为期望 |
| **rename_debt** | 09-30 | 0 | 25 | 0 | 已降级，0 为期望 |
| goal_acceptance | 10-01 | 0 | 21 | 0 | **条件式**：judge 从未判达成 |
| goal_metacog_audit | 10-01 | 0 | 21 | 0 | 同上 |
| goal_skeptic | 10-01 | 0 | 21 | 0 | 同上 |
| report_discrimination | 10-04 | 0 | 1 | 0 | **未测**（语料只有 1 个 run 在引入后） |
| report_decisive | 10-04 | 0 | 1 | 0 | **未测** |

**关键**：`goal_acceptance` / `goal_metacog_audit` / `goal_skeptic` 三件套的 0
**不是死**——它们只在 `GoalJudge` 判 `achieved=True` 后才进入。`goal_judge` 野外
43 次（11 个 run）里没有一次判达成（run56 `score=0.2`、run61 `achieved=False`），
故验收门本就到不了。这是**条件式**，删掉等于删掉"如果哪天判达成时的诚实门"。

### 11.4 v2 顺带坐实的三处

**① `failure_budget` 是**政策**违规项，不只是 0 命中。**
它是 §2 A 族唯一没被降级的**硬终止**（[cognitive_loop.py](agent/huginn/autoloop/cognitive_loop.py)
`action="stop"`，`return should_stop=True`）。而 §3 明写"硬终止只保留两个出口：
挂钟预算耗尽 与 目标达成"。它 0/21，说明野外从未触发 —— 即它既**违反准入原则**，
又**没有实测价值**。按 §3 应降为提示 + trace，而不是删（保底仍需一个失败熔断）。

**② `pivot_directive` 0/21 是**结构性**失配，不是"重复真的消失了"。**
`_repeat_exec_streak` 只有**连续两轮指纹完全相同**才自增，否则清零
（[engine_reflect.py](agent/huginn/autoloop/engine_reflect.py#L1650-L1686)）。而**同一文件自己**
在 1689-1691 行写明："书生常在两种等价 family 间来回换 (A,B,A,B…), **连续相同 streak
反复被重置, 永远到不了阈值**" —— 正因如此旁边的收敛检测器才改用**窗口去重**口径。
但 B1 改造把 pivot 硬指令的触发仍挂在 `_repeat_exec_streak` 上
（[engine_act.py](agent/huginn/autoloop/engine_act.py#L371) 与
[engine_reflect.py](agent/huginn/autoloop/engine_reflect.py#L1672) 均按 `>= _REPEAT_HARD_STREAK=2`）→
**沿用了文件自己已判定为不可达的口径** ⇒ 21 个已接线 run 全 0。

定性：这是**接线错配**，不是"该不该加控制流"的问题。修法应让 pivot 与收敛检测器
共用**同一口径**（窗口去重），而不是各挂一条；§3 过闸：它不替书生做科学判断、
只是把"在等价实验间打转"翻译成作者面的强制变异令，属 B 族提示，可保留。

**③ `surprise_convergence` 读的是**饱和的原始信号**（v31 未收口）。**
[cognitive_loop.py](agent/huginn/autoloop/cognitive_loop.py#L3769-L3775) 拿
`_surprise_history[-3:]` 的 `w` 与阈值 `[0.08, 0.20]` 比；而
[engine_reflect.py](agent/huginn/autoloop/engine_reflect.py#L692) 写入的是
`(surprise_exposed, std)` —— 即 **`surprise_exposed`**，在"语义 embedder + JEPA
predictor 双缺失"环境下正是 §8.9 判定为**饱和在 1.0** 的 jaccard 原值。v31 把 9 个
**行为**消费点收口到 `signals.routing_surprise()`，但这条早停路径漏了。
**注意修法不是换成 `routing_surprise()`**：秩信号按构造是均匀分布，"最近 3 轮秩 <
阈值"是**水平量**测试，用秩会退化成按概率随机触发，语义相反。要么删（advisory-only、
0/21、源盲），要么等 embedder/predictor 恢复后它自然可用。

### 11.5 命名漂移（通信契约应拦未拦）

§4 观测口径表把该机制写作 `collab_failure_invert`，代码实际发的是
`collab_failure_inverter`（[engine_reflect.py](agent/huginn/autoloop/engine_reflect.py#L1236)）。
一词两拼正是 §通信契约审计要拦的类型，已按下表口径更正。

### 11.6 执行结果（动手后修正了三处判读）

**已改 (4 项)**

1. **`pivot_directive` — 修接线错配 (真 bug)**。新增 `_exec_cycling` (窗口去重≤2
   **且** min 计数≥2 = 真在 A,B,A,B 打转), 与连续 streak 并列驱动硬指令, 并把 trace
   移到**唯一**观测点 ([engine_reflect.py](agent/huginn/autoloop/engine_reflect.py#L1696-L1724)、
   [engine_act.py](agent/huginn/autoloop/engine_act.py#L371-L390))。原判"先查回归" →
   查实为**结构性铁证** (§11.4 ②)。新增回归测试
   `test_repeat_chain_alternating_families_lights_directive`。
2. **`collab_failure_inverter` — 原判"真死候选"是错的**。它只在**失败**分支发 trace,
   成功面没有观测点 → 一旦正常产出, 统计侧仍计 0。**这是观测缺口, 不是死代码**;
   已补成功面 trace。
3. **三条收敛提示合流** — `surprise_convergence` / `exec_convergence` / `rename_debt`
   合并为单一机制 `convergence_advisory` (`kind` 区分来源), 行为不变 (仍 advisory-only)。
   **名义机制数 -2**, 且野外计数不再被同类机制摊薄 —— 这才是"降熵不降能力"。
4. **§3 政策澄清** — `failure_budget` **不降级**: 它不替书生做任何科学判断, 是**资源
   熔断**, §3 单列一类合法硬出口 (原"只留两出口"与它自相矛盾, 已改口径)。代码不动。

**未删 (原候选, 动手时被代码/测试反驳)**

- `decider_stop` —— **不是死代码, 它本身就是 run50 的修复** (拦"LLM 一句 stop 被静默
  丢弃")。0 命中 = decider 从未选过 stop。删它 = 把 run50 的病放回去。
- `goal_acceptance` / `goal_metacog_audit` / `goal_skeptic` —— 条件式诚实门, 上游
  (judge 判达成) 从未发生, 不是不通电。
- `surprise_convergence` 的**整体删除** —— 其源 `_surprise_history` 是 env 受限
  (embedder/JEPA 双缺失 → jaccard 饱和), 恢复环境后即有效; 且字段已落盘, 全删会级联
  到 signals 持久化与 `SIGNAL_NAMES >= 28` 断言。故只合流, 不减字段。

**对"系统是否太高熵"的正面回答**

野外看到的"死代码", 多数是**条件式守卫**与**观测缺口**, 不是可删冗余 —— 拿 0 命中
清单直接删会误杀。真正的问题只在两处**接线**: ① pivot 口径错配 (已修) ② 观测面只记
失败不记成功 (已补)。加三合一, 控制面机制 **23 → 21** 名, 且每一名都可达。
配置面 (390 个 `HUGINN_*`) 仍是最大一笔账, 属下一阶段。

### 11.7 补登记: 治理盲区 + 上界债偿还 (2026-10-05, 21 → 26 上限 22 → 26)

本轮 P1/P2 (`llm_unavailable`)、P3.2 (`progress_invariant`)、D5 (`branch_slice_skip`)、
D6 (`code_lab_slice_skip`) 四类机制**已落代码并发射**, 但漏登记 —— `pytest
tests/test_control_mechanism_governance.py` 报红。四者均为"长程空转 / 预算超支"修复链
的组成, 各自带回归测试, 属**上界债偿还而非扩张**; 预算上限一次性 22 → 26, 自此继续
only-shrink。

同时修一处**扫描盲区**: D5 的 `branch_slice_skip` 经调用方注入的 `trace_fn("name", …)`
回调上报, 此前 `scan_emitted_names` 只认 `_control_trace(` / 直发 `campaign.control_trace`
两种形态, 故该 trace 对治理面**不可见**(= 可静默新增机制的缺口)。已补 `_TRACE_FN_CALL`
正则, 把注入式回调纳入扫描 —— 这也是 §通信契约审计要拦的"绕开单一发射点"类型。

改代码登记表后须重跑治理测试; 新增 `emit` 点仍须先过 §3 两问再登记。