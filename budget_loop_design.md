# 长程研究闭环与预算控制：矛盾分析与设计说明

> 状态：**P1–P3 已实现**（D1+D4 / D2 / D3 / D5），各带回滚开关，单测已覆盖。范围：`agent/huginn/autoloop/`、`agent/huginn/metacog/branch_incubator.py` 与 `agent/huginn/agent/streaming.py`。
> 目标：把「闭环要不要继续探索（科学判据）」与「还能不能继续（资源判据）」解耦到正确的层。
>
> 落地清单：D1 `_budget_remaining_s`/`_budget_exhausted`（`engine_control.py`）+ streaming 降级阈值封顶（`streaming.py`）；D2 `for_remaining`/`stricter_tier`/`_resolve_budget_tier`（`budget.py`/`engine_control.py`）；D3 `_long_horizon_stall_action` + `_force_stall_redirect` + `decide_fn` 强制 pivot（`cognitive_loop.py`，默认关）；D4 `GoalStore.expire`（`goal_store.py`）；D5 D-slice 细粒度预算切片（`branch_incubator.py`）+ budget contextvar 刷新（`_run_single_branch`）。

> **D3 实测（run78，同命题混元 endpoint，精简资源配置，35 步 / 7 个认知环）**：
> - `darwin_stagnation` advisory 触发 3 次（步 14/24/34，均 `stagnation=2, best=2.50`）；
> - `stall_as_action` 强制 pivot 触发 3 次（步 15/25/35）→ **触发率 3/3 = 100%**：每次停滞 advisory 都被升级为有向 pivot；
> - 循环未崩、未提前停：pivot 后照常回到 hypothesize 继续推进；
> - **但 best 分始终 2.50 未改善**——强制 pivot 机制正确且安全，本轮未观测到它打破停滞；瓶颈在假设生成策略本身（产不出更高分候选），不在"该不该转向"。
> - 结论：D3 保持默认关；是否默认开是"禁止静默空转"的政策选择，不能称其为经证据验证的收益。

---

## 1. 问题陈述

矛盾的表象是"闭环想一直迭代 vs 预算要求收口"，但本质是**两套判据装在了不同的层、且粒度不匹配**：

- 科学判据（有没有进展）住在 `reflect`，在长程模式下被**让位**，几乎不终止任何 run。
- 资源判据（还剩多少）住在 `observe` 的**步边界**，管不住迭代内的长动作。

结果：长程模式下闭环既不会因"无进展"收口，也无法在"预算将被单步烧穿"前踩刹车 —— 表现为静默空转与预算旁路。

---

## 2. 现状机制清单

| 机制 | 位置 | 触发/读取 | 行为 | 与预算关系 |
|---|---|---|---|---|
| 主循环 | `cognitive_loop.py:196-344` | `while iteration < max_iterations and not should_stop` | observe→decide→execute→reflect 四段编排 | 步粒度；`max_iterations` 为步数上限 |
| 迭代终止判据汇总 | `cognitive_loop.py:211`（循环条件）、`:250-266`（重复动作 2× 上限）、`:327-336`（stop action）、`:310`（reflect.should_stop） | — | 多出口 | 均为步边界 |
| 长程让位 | `cognitive_loop.py:919-945` | `HUGINN_PERSISTENT_GOAL_MODE=1` + active goal + 挂钟未耗尽 | 启发式早停（darwin stagnation / belief / surprise）**让位**给挂钟 | 科学判据被资源判据取代 |
| 步数上限抬高 | `cognitive_loop.py:947-982` | 同上 | `cap = max(max_iterations, wall_clock/10)`，并回写 `goal.max_iterations` | 把步数上限绑定"预算/10s" |
| 挂钟硬停 | `cognitive_loop.py:2666-2703` | 每步 observe 开头 `wall_clock_expired()` | `complete(goal)` + `campaign.budget_exhausted(reason="wall_clock")` + `should_stop` | **步边界**检查；耗尽标记为 completed |
| 挂钟判据 | `goal_store.py:345-366` | `>0` 且 `started_at` 非空 | `(now-started) >= budget` | 单一时间口径 |
| 迭代内挂钟检查（仅 code_lab） | `engine_act.py:415-441` | 每次修复尝试前 | 已耗尽则不再起新尝试（run56 修复） | 唯一一处迭代内检查 |
| 档位预算 | `budget.py:29-82`、`engine_control.py:473-512` | `for_iteration(iteration)` | 按**迭代序号**限 mode（open/medium/light），拒绝额度用尽则降级放行 | 与剩余预算**脱钩** |
| token/cost 硬刹车 | `budget.py:85-175` | 每次 LLM 调用 `update()` | 超 `HUGINN_TOKEN_BUDGET`/`HUGINN_COST_BUDGET` 抛 `BudgetExhausted` | 独立于挂钟；触点是 LLM 调用后 |
| 流式降级 watchdog | `streaming.py:64-109`、`:1959-2060` | 主流空闲超时后 | 降级并把空闲阈值放宽到 `total_timeout`（≈300s） | **不看** goal 剩余预算 |

---

## 3. 冲突点（代码级，逐条可指）

1. **科学判据缺出口**：长程模式下 `_long_horizon_keep_going` 让位启发式早停，唯一硬出口是挂钟 + 目标达成（`cognitive_loop.py:2674-2703`）。无进展轮在长程模式下没有终止或转向出口 → 静默空转。
2. **粒度错配**：预算按"步"查（observe 开头一次），消耗按"动作"烧。code_lab 修复循环已补迭代内检查（`engine_act.py:415-441`），但 streaming 降级路径（`streaming.py:94-109`）把空闲阈值放宽到 `total_timeout`，与 goal 剩余预算无关 —— run72/run75 即在此吞掉整轮预算。
3. **档位与资源方向相反**：`ProgressiveBudget` 用 `iteration` 序号分档（`budget.py:57-82`，`engine_control.py:485`）。快速省钱的 run 被过早限档；慢而烧钱的 run 在第 1 轮仍是 `open` 无限档。
4. **"耗尽"冒充"达成"**：挂钟到点调用 `_gs.complete(goal)`（`cognitive_loop.py:2683`），把预算耗尽记成 completed，污染下游"目标达成"信号。

---

## 4. 设计目标 / 非目标

**目标**
- 资源判据下沉为**单一 deadline 原语**，所有可能长阻塞的动作在**启动前**查询一次：不启动一个必然超时的动作。
- 科学判据从"终止器"降级为"**动作选择器**"：无进展 + 有预算 → 转向，而非停止。
- 档位与实际剩余预算一致。
- 预算耗尽与目标达成在语义与状态上分离。

**非目标**
- 不改动科学判据本身（darwin 评分 / belief / surprise）的算法与阈值。
- 不引入新框架、不改 `/goal`、`serve` 的对外接口。
- 不改 token/cost 硬刹车的既有语义（仍为 LLM 调用后的硬上限）。

---

## 5. 设计

### D1 统一 deadline 原语（止血核心）

**接口**（`Engine`，落 `engine_control.py`；`self._run_goal_id` 已存在）
```python
def _budget_remaining_s(self) -> float | None:
    """返回挂钟剩余秒数；无挂钟限制（非长程/无 goal）返回 None。"""

def _budget_exhausted(self) -> bool:
    """单一判据：长程模式且挂钟耗尽 → True；否则 False。"""
```
实现复用 `GoalStore.wall_clock_expired` 与 `self._run_goal_id`（语义对齐 `_long_horizon_keep_going` 的"本 run goal 优先"防串台逻辑）。

**改动点**
- `engine_act.py:415-441`：删除私有 `_wall_clock_expired()`，改调 `self._budget_exhausted()`（行为等价，去重）。
- `streaming.py:94-109`：`_fallback_stream_idle(total_timeout, budget_left=None)` 取 `min(放宽后的阈值, budget_left)`；调用点 `streaming.py:2025` 附近读取剩余预算。
  - 传递方式：用 `contextvar`（例如 `huginn.agent.streaming.remaining_budget_s`）由 autoloop 每步设置；未设置 = `None`，完全保持现行为。**避免 streaming 反向依赖 autoloop**。
- 其余长阻塞动作（DFT/MD、BranchIncubator 多路多轮、dynamic_workflow 并行子任务）在起新尝试前统一查 `_budget_exhausted()`。

**回滚**：`HUGINN_BUDGET_DEADLINE_UNIFIED=0` 时保持"各写各的"旧行为。
**行为变化**：仅在长程模式且预算将耗尽时更早停止启动新动作；非长程路径零变化。

### D2 档位预算改按剩余预算

**接口**（`budget.py`）
```python
def for_remaining(self, fraction: float) -> IterationBudget:
    """按剩余预算比例取档：fraction∈[0,1]，None 语义见调用方回退。"""
```
档位示例（按比例，非序号）：`>0.6 open / 0.3–0.6 medium / <0.3 light`；`iteration` 仍作**上界兜底**。

**改动点**：`engine_control.py:485` 由 `for_iteration(iteration)` 改为：能拿到挂钟预算时用 `for_remaining(remaining/budget)`，否则回退 `for_iteration`。

**回滚**：`HUGINN_PROGRESSIVE_BUDGET_BY_REMAINING=0`（默认开）。
**风险**：会改变"哪些 mode 何时被允许"，需重跑 end-to-end 对照。

### D3 无进展 → 动作选择器（非终止器）

**位置**：`cognitive_loop.py:1147` 附近的 stagnation 分支 + `_long_horizon_keep_going` 的判定。

**行为**：长程模式 + 预算未耗尽 + stagnation 达阈值时，**不** stop，而是设 `state.should_redirect=True` 并注入强制 meta 动作（`pivot` / `method-swap` / `review`），把"空转"转成有向动作。

**接口**：新增 `_long_horizon_stall_action() -> str`；复用现有 `should_redirect`/`redirect_reason`。

**前置依赖（已核实）**：核心循环**没有**"外部注入强制动作"的通道 —— `ActionDecision.force`（`cognitive_loop.py:127`）的语义是"跳过 reflect 的 redirect 建议"，不是"强制某动作"；`decide` 的产物只由钩子产出。因此 D3 应实现于 **autoloop 自己的 `_decide` 钩子**：当 `_long_horizon_stall_action()` 触发时直接返回 `ActionDecision(action="pivot", force=True, ...)`，无需改核心循环。这条路可行，但"pivot 是否真能换方向"仍依赖 pivot 分支的实际效果，需实证。

**回滚**：`HUGINN_STALL_AS_ACTION=0`（默认关，先观测再默认开）。

### D4 耗尽语义分离

**接口**（`goal_store.py`，镜像现有 `complete()` 于 `:310`）
```python
def expire(self, goal_id: str, reason: str = "wall_clock") -> Goal:
    """预算耗尽：status='expired'，metadata 记 reason；不冒充 completed。"""
```
**改动点**：`cognitive_loop.py:2682-2685` 的 `_gs.complete(...)` 改为 `_gs.expire(..., reason="wall_clock")`；`campaign.budget_exhausted` 事件保持（`reason` 字段已存在）。
**回滚**：`HUGINN_BUDGET_EXPIRE_SEMANTICS=0` 时回退 `complete`。

### D5 细粒度预算切片（D-slice，针对单阶段超支）

**问题（run72/73）**：D1–D4 把预算判据下沉到**步边界**，但一次 `hypothesize` 阶段的 `BranchIncubator` 树状探索（depth=2：3 layer1 + 6 layer2 = 9 个子智能体）在**阶段内部**串行/并发烧时间。run72/73 里 9 个子智能体各触发 ~300s 流式超时，单阶段累计 887s，远超 420s 挂钟预算；步边界的 `_budget_exhausted()` 覆盖不到阶段内部。

**方案（"少给硬时长、多切细粒度"，非硬 kill）**：不引入整阶段的硬超时，而是把一轮探索拆成**有序、各自独立有产出**的 slice，每个**可选** slice 启动前查一次剩余预算，不足即用已产出结果收尾：

| slice | 内容 | 是否可选 |
|---|---|---|
| S1 | layer1 扇出（核心，产出 N 条可用假设） | 必跑（调用方启动前已查 `_budget_exhausted`） |
| S2 | layer1 PRM 打分（rollout value） | 可选 |
| S3 | layer2 精修扇出（逐 parent） | 可选 |
| S4 | layer2 PRM 打分（逐 sub-branch） | 可选 |
| S5 | prune（廉价） | 总是执行 |

- **自校准门槛**：固定下限 `slice_min_s`（`HUGINN_BRANCH_SLICE_MIN_S`，默认 60s）只作下界；上界用**上一片实测挂钟**估计 —— `_need = max(_min_s, 上一片实测)`。run72 场景"layer1 烧 300s、只剩 120s"→ 门槛 300s → 拒绝启动同样要 ~300s 的 layer2，而不是拿拍脑袋固定阈值。
- **子智能体自限**：`_run_single_branch` 在起 Subagent **前**把"此刻剩余预算"刷进 `remaining_budget_s` contextvar（随 task 传播），streaming 降级阈值据此封顶（D1 那条）——所以连**在飞的 S1** 也不会无界烧穿。
- **不 kill 在跑的**：只拒绝启动"注定越预算的下一片"，已启动的照常收。故单阶段超支被限制在**一个 slice** 以内，而非整棵树。
- **可观测**：跳过任一可选 slice 时上报 `branch_slice_skip` 控制面 trace（`name/evidence/action`，见 `campaign.control_trace`）。

**回滚/等价**：`budget_remaining_fn=None`（非长程/无 goal）或查询返回 `None` → 所有 slice 照跑，行为 100% 不变；查询异常 fail-open。

---

## 6. 分期与依赖

- **P1（止血，低风险）：D1 + D4**。行为面最窄，直接堵住"迭代内烧穿预算"与"耗尽冒充达成"。
- **P2（对齐，中风险）：D2**。改变档位时序，需对照 run。
- **P3（行为，需实证）：D3**。改变探索行为，先默认关、观测后再默认开。
- **P3.2（阶段内止血）：D5 D-slice**。不改科学行为，只把阶段内的可选动作按细粒度 slice 门控；非长程 100% 等价。run72/73 单阶段超支的针对性修复。
依赖：D2/D3 均依赖 D1 的剩余预算原语；D3 的硬强制依赖 `decide` 是否支持 forced action（需先核）；D5 依赖 D1 的 `_budget_remaining_s` 与 streaming 的 `remaining_budget_s` contextvar。

---

## 7. 验收判据

1. **单测**：`_budget_exhausted()` 在无 goal / 非长程 / 未耗尽 / 已耗尽四种组合下的返回值；`_fallback_stream_idle` 在给定 `budget_left` 时不超过剩余预算；`for_remaining` 的分档边界；`expire()` 后 `status=="expired"`。
2. **replay_audit**：历史轨迹重放，确认"无进展轮"仍触发可观测 advisory trace，且不再出现在预算耗尽后仍启动新动作的轮次。
3. **end-to-end**：一次长程 run 对照（同命题、同预算），确认（a）总挂钟不超预算超出量收敛到"至多一个在飞动作"；（b）goal 终态在"耗尽"下为 `expired`、在"达成"下为 `completed`。
4. **D5 单测**（`branch_incubator._selfcheck` 22–27）：`_slice_affordable` 五态；预算不足时 depth=2 仍只跑 layer1（3 dispatch）；预算充足/无约束时行为不变（9 dispatch）；`_assign_values` 逐 branch 预算门；`branch_slice_skip` trace 上报；自校准门槛（layer1 实测 → 决定 layer2 是否负担得起）。
5. **D5 end-to-end**：长程 run 的 run.log 应在预算将尽时出现 `skip slice=layer2 (return layer1)` / `branch_slice_skip`，且单次 hypothesize 阶段耗时不再显著超出一个 layer1 slice 的实测成本。

---

## 8. 风险与回滚汇总

| 项 | 回滚开关 | 默认 | 主要风险 |
|---|---|---|---|
| D1 统一原语 | `HUGINN_BUDGET_DEADLINE_UNIFIED=0` | 开 | contextvar 未设置时须严格等价旧行为 |
| D2 预算比档位 | `HUGINN_PROGRESSIVE_BUDGET_BY_REMAINING=0` | 开 | 档位时序变化影响 mode 可用性 |
| D3 无进展转动作 | `HUGINN_STALL_AS_ACTION=0` | **关** | 强制动作可能与 decide 冲突 |
| D4 耗尽语义 | `HUGINN_BUDGET_EXPIRE_SEMANTICS=0` | 开 | 下游若依赖 `completed` 需同步 |
| D5 D-slice | `budget_remaining_fn=None`（非长程自动等价） | 开（仅长程生效） | 门槛自校准依赖"上一片实测"，首片无参考时用固定下限 |

---

## 9. 明确不做

- 不新增第二套预算/闭环框架（一词一义：预算是预算，科学判据是科学判据）。
- 不改 `GoalScheduler`（`goal_scheduler.py`）与 `GoalStore` 的既有对外方法签名。
- 不为每种 plan mode 单独做放大/缩减系数（沿用现有离散档位）。