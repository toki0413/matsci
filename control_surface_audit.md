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
| `_long_horizon_keep_going()` 特判点 | 9（`cognitive_loop.py`） |
| 硬决策状态量 | `_force_exec_variation` `_exec_converged` `_repeat_exec_streak` `_rename_debt` `_fp_result_ref` `_exec_fp_history` `_darwin_stagnation` … |

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
| A1 | exec convergence → conclude+stop | `engine_reflect.py:1355` 置标志；`cognitive_loop.py:3538-3545` 终止 | run50、run52（均为**假收敛**） | 高：直接结题 | **降**：只写 trace + 提示，不自动终止 |
| A2 | rename debt 越限 → conclude+stop | `hypothesis_loop.py:2315` 累加；`cognitive_loop.py:3561-3569` 终止 | run51（607s 早停） | 高：直接结题 | **降**：转观测计数（已用于审计），不终止 |
| A3 | darwin stagnation → stop | `cognitive_loop.py:1164-1168` | 长程下被让位，未触发 | 中 | **删**：长程已有挂钟出口，这是死代码 |
| A4 | belief σ² 收敛 → stop | `cognitive_loop.py:1190` | 未触发 | 中 | **删**：同上 |
| A5 | LLM decider "stop" | `cognitive_loop.py:2545`（长程拦截） | run50 曾因此提前收起（已修） | 中 | **降**：转提示，让书生自己决定何时收结 |
| A6 | 预算门跳过 execute | `cognitive_loop.py:2720-2726` | run52（**假收敛根因**） | 高：静默跳过阶段 | **删**：长程模式下不该有阶段门 |
| A7 | 挂钟预算 → stop | `cognitive_loop.py:961` | run53/54/55 正常耗尽 | 低：用户给定的边界 | **留**（唯一的合法自动出口） |
| A8 | max_consecutive_failures / by_type | `HUGINN_MAX_FAILURES_BY_TYPE` 等 | 未知 | 中 | 待观测后再定 |

### B. 强制/引导类（建议统一降为"写进 trace 的提示"）

| # | 机制 | 位置 | 实测触发 | 建议 |
|---|------|------|---------|------|
| B1 | pivot 硬指令 `_force_exec_variation` | `engine_reflect.py:1327` 置；`engine_act.py:363` 注入作者提示 | run50/51/52 多次 | **降**：并入提示层，去掉独立状态机 |
| B2 | repeat 软提示 `_speculator_hint` | `engine_reflect.py:1306-1320` | run50-53 | 留（本就是提示） |
| B3 | 换名重定向 hint | `hypothesis_loop.py` | run50-55（5–20 次/轮） | 留（提示） |
| B4 | effort floor hint | `engine_reflect.py:588-596` | 未知 | 待观测 |
| B5 | 非有限数值 hint | `engine_reflect.py:533` | run54（1 次） | 留（属 C 的提示面） |
| B6 | coldstart / prompt guards | `research/coldstart_guards.py` | 每轮 | 留（防退化） |
| B7 | curiosity hint / PMK / MCMC 注入 | 多处 | 未知 | 待观测；疑似可删 |

### C. 诚实/证据门（**留，且是唯一该硬的**）

| # | 机制 | 位置 | 说明 |
|---|------|------|------|
| C1 | 非有限数值不算证据 | `engine_reflect.py:71, 1381` | `_non_finite_objective_keys` 接进 `_is_code_lab_solved` |
| C2 | claim grounding 门 | 门控层 | 结论必须溯源到真实数值 |
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

配套：每个硬控触发时写一条结构化 trace（`name / iteration / evidence`），
便于统计触发率；**长期 0 触发或长期误杀的，删或降**。

---

## 4. 建议动作顺序

1. **立原则** —— ✅ 已写进 [`docs/architecture.md`](agent/docs/architecture.md)
   「控制面预算（autoloop 硬决策准入）」，后续机制按 §3 过闸。
2. **A1 + A2 降级** —— ✅ 已降为"提示 + `campaign.control_trace`"，不再 `should_stop`
   （[`cognitive_loop.py`](agent/huginn/autoloop/cognitive_loop.py) 的 F5 / v11 两处）。
   唯一自动终止交给 A7。
3. **A3/A4/A5/A6 清理**——长程模式下它们已是 no-op 或被让位；9 处
   `_long_horizon_keep_going` 特判应收敛成**一个**"长程 = 只有挂钟+目标两出口"的开关。
4. **B1 并入提示层**，去掉 `_force_exec_variation` 独立状态机。
5. **加触发率观测**，跑若干轮后按数据做第二轮删减（B4/B7/A8 待定项）。

优先级：**第 2 步 > 第 3 步 > 第 4 步 > 第 5 步**。
第 2 步风险最低、收益最大：把两个会误杀整轮实验的硬终止降级，不损失任何诚实边界。