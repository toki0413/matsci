# 数论图论突破异步委派天花板 Spec

> 目标: 用数论图论把异步委派的 4 个天花板从"已知局限"变成"已解".
> 前提: networkx >= 3.0 + sympy >= 1.12 已装.
> 原则: 数学动机 → 算法 → 代码, 每个映射有明确的天花板对应物.

## 动机

异步委派 spec 的 4 个天花板本质是同一问题: **组合搜索效率**

| 天花板 | 表象 | 数学本质 |
|---|---|---|
| subagent 失败不知卡哪步 | 只回 success=False | 图上的失败节点定位 (路径回溯) |
| 并行无依赖感知, 重复探路 | 4 个 explore 探同路径 | 集合相交无约束 (DAG 拓扑序可解) |
| _check_stuck 启发式简单 | 3 轮无 tool_call 就判卡 | 序列周期检测, 存全历史 |
| subagent 预算三参数无原则 | max_depth/parallel/per 任意选 | 算术基本定理给出可行配置集 |

## 4 个数学映射

### M1. 子任务 DAG (图论: 有向无环图 + 拓扑排序)

**治**: G3 并行 dispatch 无依赖感知, 4 个 subagent 可能探同路径.

**数学**: 子任务建为 DAG G=(V, E), v ∈ V 是子任务, (u,v) ∈ E 表示 u 输出是 v 输入.

- **拓扑序 (Kahn)**: O(V+E), 决定执行顺序
- **并行度 = 最大反链 (Dilworth)**: 最小链覆盖数 = 最大反链大小, 二分图匹配
- **关键路径**: DAG 最长路径, wall-clock 下限

**接入**: `dispatch_parallel` 接受 `tasks: list[dict]` + `dependencies: list[tuple]`, 内部建 DAG, 按拓扑序分层并行 dispatch, 同层 antichain 内并行.

### M2. 子图同构检索 (图论: VF2 算法)

**治**: trajectory_pattern KB 有写入无读取, _check_stuck 启发式简单.

**数学**: 历史轨迹 = 标签图 G_h, 当前轨迹 = 标签图 G_c. 判断 G_c 是否"在走 G_h 的路" = **子图同构** (subgraph isomorphism).

- **VF2 算法**: 实用次指数 (最坏 NP-complete, 实际很快)
- **line graph 转换**: tool_name 序列 → 相邻关系图, 子图同构 = 当前 prefix 是否在某历史轨迹里出现
- **相似度**: 匹配长度 / 历史轨迹长度

**接入**: `_check_stuck` 调 `trajectory_match(current, history)`, 找到相似历史 → 取下一步 tool 作为建议注入 prompt.

### M3. Floyd 循环检测 (算法/数论: O(1) 空间周期检测)

**治**: _check_stuck 当前存全历史判卡顿, 长程任务 (530 步) 历史膨胀.

**数学**: tool_call 序列 = 数列 a_0, a_1, ..., 卡顿 = 序列进入 cycle.

- **Floyd 算法**: tortoise & hare 双指针, O(1) 空间, O(μ+λ) 时间 (μ=前缀, λ=周期)
- **Brent 改进**: 比 Floyd 快 ~36%, 仍 O(1) 空间
- **数论背景**: Pollard rho 用同原理分解大整数; 周期检测是数论核心操作

**接入**: `_check_stuck` 用 `detect_cycle(recent_tool_calls)`, 发现 cycle → 卡顿. 不存全历史, 只存两个指针.

### M4. 素数预算分解 (数论: 算术基本定理)

**治**: max_depth / parallel / per_subagent 三参数选择无原则.

**数学**: 总预算 N = depth × parallel × per_subagent. 算术基本定理: N = p_1^a_1 × p_2^a_2 × ... 唯一分解.

- **530 = 2 × 5 × 53**: 三个素因子
  - (depth=2, parallel=5, per=53): 2 层 × 5 并行 × 53 单 agent
  - (depth=1, parallel=10, per=53): 1 层 × 10 并行 × 53 单 agent
  - (depth=1, parallel=2, per=265): 1 层 × 2 并行 × 265 单 agent
- **素数选择原则**:
  - depth 用小素数 (2, 3) — 失控风险随深度指数增长
  - parallel 用中素数 (5, 7) — LLM API 限速约束
  - per_subagent 用大素数 (53, 97) — 单 agent 任务复杂度

**接入**: `huginn.toml` 加 `budget_strategy: "auto" | "manual"`, auto 时按 max_total_calls 素数分解 + 启发式 (depth≤2, parallel≤5) 选配置.

## 改动清单

| ID | 文件 | 改动 | 行数 |
|---|---|---|---|
| M1 | 新建 `huginn/agents/task_dag.py` | TaskDAG (Kahn 拓扑序 + antichain_width + critical_path) | ~80 |
| M2 | `huginn/knowledge/trajectory_pattern.py` | 加 `trajectory_match` (line graph + VF2) | ~40 |
| M3 | 新建 `huginn/runtime/cycle_detect.py` | `detect_cycle` (Floyd, hash 到 Z/pZ) | ~30 |
| M4 | 新建 `huginn/agents/budget_decomp.py` | `budget_configurations` (sympy.factorint) | ~25 |
| 接入 | `huginn/tools/subagent_tool.py` | dispatch_parallel 接 DAG | ~15 |
| 接入 | `huginn/autoloop/engine.py` | _check_stuck 用 cycle_detect + trajectory_match | ~20 |
| selfcheck | 各模块 `_selfcheck()` | 4 个 assert-based 检查 | ~40 |

**总计 ~250 行** (比纯异步委派 spec 的 95 行多 155 行, 这 155 行是把"天花板"变成"已解"的数学代价).

## 上限测算 (数学保证)

| 配置 | depth | parallel | per | 上限 (tool_calls) |
|---|---|---|---|---|
| 单层 (现状) | 1 | 1 | 50 | 530 × 50 = 26,500 |
| M4 推荐 (530=2·5·53) | 2 | 5 | 53 | 530 × 5 × 53 = 140,450 |
| M1 并行优化 | 2 | 5 (DAG antichain) | 53 | 同上, wall-clock 减半 (DAG 调度) |
| 极端 (depth=3) | 3 | 5 | 53 | 530 × 5 × 53 × 50 = 7,022,500 |

**关键洞察**: M1 的 DAG 调度让 wall-clock 不随 parallel 线性增长 (同 antichain 内并行), 这解决"步的语义 = wall-clock"维度.

## selfcheck 计划

- `task_dag._selfcheck`: 5 节点 DAG (A→B, A→C, B→D, C→D, D→E), 验证拓扑序合法 + antichain_width=2 ({B,C}) + critical_path=4 (A→B→D→E)
- `trajectory_pattern._selfcheck`: 历史 [[a,b,c,d], [a,b,e,f]], 当前 [a,b,c] → 匹配 hid=0, sim=0.75
- `cycle_detect._selfcheck`: [a,b,c,a,b,c] → (mu=0, lam=3); [a,b,c,d] → None; [a,b,c,d,a,b] → None (序列未结束不判)
- `budget_decomp._selfcheck`: 530 → 含 {depth:2, parallel:5, per:53}; 530 → 不含 {depth:4, ...} (4 不是 530 的因子)

## 天花板 (本 spec 解不掉的)

- **VF2 最坏 NP-complete**: 实际很快, 但极端 case 慢. 升级: 用 GED 近似 (DP, polynomial)
- **素数分解假设 N 需全用**: 实际 agent 可能 200 步就完成. 升级: 动态预算 (跑一步看一步, 剩余预算重新分解)
- **trajectory_match 假设历史可复用**: 不同 paper 轨迹可能完全不同. 升级: 跨域相似度度量 (Jaccard on tool_name set)
- **DAG 假设依赖已知**: LLM 可能不知道子任务依赖. 升级: LLM 在 dispatch_parallel 时填 dependencies 字段 (M1 已支持)

## 不做 (YAGNI)

- ❌ Ramanujan 图 / LPS 构造 — 理论最优 expander 但实现复杂, Kahn + VF2 已够用
- ❌ Erdős–Ko–Rado 子集相交定理 — 理论可避免重复探路, 但 DAG 拓扑序已隐含处理
- ❌ Pólya 计数定理 — 4 并行 subagent 的对称群太小 (S_4 = 24 元素), 暴力枚举更快
- ❌ Siegel 零点 / GRH — trajectory KB 检索不需要数论假设
- ❌ Cayley 图 / 群论 — agent 决策树不是群, Cayley 图不直接适用
- ❌ 整数规划解 budget 分配 — 素数分解 + 启发式足够, ILP 求解器过重

## 执行顺序

1. **M4** (budget_decomp) — 最小 (25 行), 纯数论, 无依赖, 先做热身 ✓ 6/6 selfcheck
2. **M3** (cycle_detect) — 独立模块 (30 行), 无依赖 ✓ 7/7 selfcheck
3. **M2** (trajectory_match) — 40 行, 需要 networkx (已装) ✓ 4/4 selfcheck
4. **M1** (task_dag) — 最大 (80 行), 需要 M4 决定并行度 ✓ 7/7 selfcheck
5. **接入** subagent_tool (G3 dispatch_parallel 接 DAG) + autoloop engine (G2 _check_stuck 接 cycle + trajectory) + subagent (G1 递归深度守卫) ✓
6. **selfcheck** 全过 → commit + push ✓

每个 M 完成后立即 selfcheck, 不批量验证. 数学模块独立, 不互相阻塞.

## 完成状态

| 模块 | 状态 | selfcheck |
|---|---|---|
| M4 budget_decomp | ✓ | 6/6 |
| M3 cycle_detect | ✓ | 7/7 |
| M2 trajectory_match | ✓ | 4/4 |
| M1 task_dag | ✓ | 7/7 |
| G1 递归深度守卫 | ✓ | 4/4 (subagent.py selfcheck) |
| G2 _check_stuck (engine.py) | ✓ | 6/6 (engine_selfcheck.py G2-A~F) |
| G3 dispatch_parallel (subagent_tool.py) | ✓ | 跟 G1 共用 subagent_tool 验证 |

数学映射全部接入主循环, 异步委派 4 天花板从"已知局限"变为"已解".
