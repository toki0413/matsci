# 成本-剪枝参与感契约 (Cost & Pruning Participation)

> 状态: **staging**（设计稿，待评审后定稿移入 active）
> 定位: 人机协作参与感的设计契约。登记两类机制——**决策点对话**（agent 主动召回用户决策，主动参与）与**成本叙事面板**（用户被动感知，读懂 agent 在干嘛、为什么、花在哪、值不值）。
> 上游: [cost_ledger.py](../huginn/cost_ledger.py) / [value_budget.py](../huginn/value_budget.py) / [budget_pause.py](../huginn/budget_pause.py) / [permissions.py](../huginn/permissions.py)。本契约定义"如何把这些后端机制转化为用户可感知、可参与的东西"。

## 1. 设计原则

参与感 ≠ 审批流。把人叫进来点"允许/拒绝"是**打断式参与**，是负担，不是协作。本契约遵循四条原则：

1. **读懂优先**：用户要先能看懂 agent 在干嘛、为什么，才谈得上参与。成本必须"叙事化"（数字 + 意图 + 预测），不是裸数字。
2. **关口召回，而非处处打断**：只在**关键决策点**（剪枝/降档/休眠）召回用户，且召回成本低、可丢弃。
3. **用户握最终判断权**：涉及"是不是真金"的价值误判风险时，决策权交还用户，agent 只给建议和证据。
4. **可逆且留痕**：每个决策点可撤销、状态可观测，用户信任来自"永远能改回来"。

## 2. 两类机制总览

| 机制 | 方向 | 触发方 | 用户角色 | 后端底座 |
|---|---|---|---|---|
| 决策点对话 (DecisionPoint) | 主动参与 | agent 在关口发起 | 选择/裁决/改方向 | BranchPolicy(UCB+休眠)、ValueBudget、BudgetPause |
| 成本叙事 (CostNarrative) | 被动感知 | 持续流式 | 读懂/被说服/形成判断 | CostLedger(by_tool/by_phase/by_dimension) |

两者正交：叙事负责"持续可读"，决策点负责"关键处可参与"。长程任务靠叙事建立全程握感，靠决策点在高风险关口取得话语权。

## 3. 决策点对话 (DecisionPoint)

### 3.1 触发关口

agent 在以下关口**主动发起**决策点（而非自动执行）：

| 关口 | 触发条件 | 默认动作若用户不响应 |
|---|---|---|
| `prune` 剪枝 | UCB 判定"低均值 + 低不确定性"分支该砍 | 降级为休眠（不硬砍） |
| `hibernate` 休眠 | "低均值 + 高不确定性"（可能是真金） | 执行休眠，保留 lifeline |
| `degrade` 降档 | 成本压力高，需降 ThinkingIntensity / ModelTier | 执行降一档 |
| `pause` 预算边缘 | 成本达阶段预算边缘 | 软停止 + checkpoint（BudgetPause） |
| `resume` 续投 | pause 后决定是否续投 | 保持暂停 |

每个决策点**必须可撤销、可留痕**。

### 3.2 决策点消息格式（契约）

一个决策点 = 结构化 payload，前端据此渲染为"状态 + 我的判断 + 请你选"：

```jsonc
{
  "type": "decision_point",
  "id": "dp_<uuid>",
  "kind": "hibernate",              // prune | hibernate | degrade | pause | resume
  "created_at": "<iso8601>",
  "session_id": "s1",
  "branch_id": "b_7",               // 关联的假设分支
  "status": "pending",              // pending | approved | edited | denied | expired
  "narrative": {                    // 成本叙事片段（见 §4）
    "phase": "converge",
    "cost_usd": 12.40,
    "by_dimension": { "llm": 2.1, "compute": 9.8, "other": 0.5 },
    "by_phase": { "explore": 8.0, "converge": 4.4 },
    "predicted_cost_to_converge_usd": 5.0
  },
  "agent_judgment": {
    "mean_value": 0.3,              // 归一化已实现价值
    "uncertainty": 0.8,             // 探测少/方差大/分歧高 → 高不确定性
    "recommendation": "hibernate",  // agent 建议, 非定论
    "reason": "分支 b_7 探测 6 次、方差高、尚未试探透, 可能藏真金; 建议休眠保留 lifeline 低成本盯着"
  },
  "options": [
    { "id": "hibernate_lifeline", "label": "休眠(保留 lifeline)", "risk": "low" },
    { "id": "continue_invest",     "label": "继续投",             "risk": "medium" },
    { "id": "abandon",             "label": "放弃",               "risk": "low" }
  ],
  "response": null                  // 用户选择后回填
}
```

### 3.3 裁决语义

- `approved`：采纳 agent 建议。
- `edited`：用户改动作（例如把"放弃"改成"休眠"），agent 按用户选择执行。
- `denied`：用户否决某个选项，agent 记录并重新提议或回退。
- `expired`：用户逾时不响应，走 §3.1 的"默认动作"（保守方向——休眠/软停，不硬砍）。

### 3.4 后端契约（建议承载）

新增 `huginn/branch_policy.py`（骨架，本契约只定义接口语义，不实现）：

- `BranchScore`: `mean_value` / `uncertainty` / `ucb = α·mean + β·uncertainty`。
- `BranchState`: `active` | `hibernating` | `abandoned`；hibernating 必带 `lifeline` 与 `revive_conditions`。
- `DecisionPointRegistry`: 登记/查询/过期决策点，emit 事件供前端订阅。
- 与 `BudgetPauseHandler` 复用"软停止 + 可续投"哲学，作用对象从"任务"泛化为"分支"。

## 4. 成本叙事 (CostNarrative)

### 4.1 原则

成本不是一行数字，是**"花在哪 + 值不值 + 接下来怎么办"**的可读叙事。持续流式，不打断。

### 4.2 叙事 = 数字 + 意图 + 预测

| 要素 | 说明 | 数据来源 |
|---|---|---|
| **数字** | 当前累计 USD、按维度/工具/阶段分解 | CostLedger.by_dimension / by_tool / by_phase |
| **意图** | 为什么花这么多（当前在做什么、什么阶段） | 会话 phase + branch 上下文 |
| **预测** | 距收敛还需多少（预测成本 + 置信） | ValueBudget.effective_budget + 历史费率 |
| **价值判断** | 当前 ROI / 不确定性，值不值得继续 | ValueBudget.value_ok / BranchScore |

### 4.3 降档透明

降 ThinkingIntensity / ModelTier 必须叙事化："成本压力高，我降一档省算力，当前档位 X"。让用户感知 agent 在**为自己的开销负责**，而非默默变蠢。

### 4.4 前端承载（建议）

- 强化 `MetricsBar`：从裸 `$12.4` → 可展开的"成本叙事"行（维度分解 + 意图 + 预测）。
- 新增决策点卡片渲染（基于 §3.2 payload），可展开/可操作/可撤销。
- 复用 `AutoloopProgress` 的阶段叙事，把成本挂在对应阶段上。

## 5. 用户旅程（示例）

1. agent 进入 explore，`CostNarrative` 持续显示"在搜 VASP 相空间，已花 $8，预计还需 ~$5"。
2. 分支 b_7 高不确定性，agent 发起 `hibernate` 决策点："这条线看不出值但没试透，可能藏真金"。
3. 用户选"休眠(保留 lifeline)" → agent 执行休眠，b_7 留廉价 lifeline 盯着。
4. 成本压力升高，agent 推送 `degrade` 叙事"降一档省算力"，用户看到透明度，可撤销。
5. 达预算边缘 → `pause` 软停 + checkpoint → 用户过几天 `resume` 续投。

## 6. 可观测与审计

- 所有决策点/降档/休眠/续投记录进 `CostLedger`（挂 `label`/`phase`），可回溯"为什么当时砍了/留了这条线"。
- 每条决策点 `id` 唯一、`status` 可查，前端可针对单条请求撤销。

## 7. 待办（评审通过后）

- [ ] 新增 `huginn/branch_policy.py`（BranchScore / BranchState / DecisionPointRegistry 骨架）。
- [ ] 新增成本叙事事件类型，接入统一事件总线（events-contract）。
- [ ] 前端：MetricsBar 成本叙事增强 + 决策点卡片。
- [ ] 单元测试：决策点裁决语义、默认动作、可撤销、UI 渲染 payload。