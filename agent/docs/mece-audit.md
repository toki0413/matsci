# MECE 契约审计 (奖励面 + 授权面)

自动生成: `python -m huginn.cli.contract_audit --out docs/mece-audit.md`.
以 MECE 两原则审计 agent 的**奖励面 / 授权面**: **collectively exhaustive** 抓「宣称维度零调用者」; **mutually exclusive** 抓「同轴惩罚叠加」与「跨模块同名重复实现」. 纯静态扫描, 只提示候选, 不判死.

## 奖励面: 宣称项 vs 调用者 (collectively exhaustive)

来源: `claim_reward.py::__all__` 的宣称面. `状态 dead` = 宣称但零调用者; `internal-only` = 仅模块内被组合复用 (非死, 但无独立接线).

| 奖励项 | 生产调用 | 测试引用 | 模块内引用 | 状态 | 备注 |
|---|---|---|---|---|---|
| `numeric_accuracy_reward` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `grounding_source_reward` | 1 | 1 | 1 | `wired` |  |
| `grounded_accuracy_reward` | 1 | 0 | 0 | `wired` |  |
| `strict_scope_reward` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `efficiency_discount` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `idle_turn_penalty` | 0 | 0 | 1 | `internal-only` | 仅模块内被组合调用 (经 anti_hacking_reward 等) |
| `anti_hacking_reward` | 1 | 0 | 0 | `wired` |  |
| `reconcile_r_phys` | 0 | 0 | 0 | `dead` | 零调用者 —— 宣称但未接线; 同名跨模块, 归属已按模块限定隔离 |

### 惩罚轴重叠 (mutually exclusive)

| 惩罚轴 | 惩罚项 (触发信号) | 是否重叠 |
|---|---|---|
| 授权越界 | `strict_scope_reward`(authorized_ratio) | 否 |
| 轮次 | `efficiency_discount`(solved_at_turn), `idle_turn_penalty`(extra_turns) | ⚠️ 是 |

### 跨模块同名 (mutually exclusive)

| 名称 | 其它模块也定义 |
|---|---|
| `reconcile_r_phys` | `huginn/security/world_state.py` |

## 授权面: 口径源 vs 消费点

来源: `scope_authority.py::__all__` (S1 合规口径 / S2 意图口径). 两口径声明为**独立开关**, 消费点在 `autoloop/engine_reflect.py::_apply_strict_scope`.

| 口径源 | 生产消费点 | 状态 | 说明 |
|---|---|---|---|
| `compute_authorized_ratio` | 1 | `wired` | S1 合规口径 |
| `compute_intent_ratio` | 1 | `wired` | S2 意图口径 |

### 独立开关

| 开关 | 已注册 | 默认 | 消费点 |
|---|---|---|---|
| `anti_hacking_reward` | True | `False` | huginn/autoloop/engine_reflect.py:2322 |
| `intent_scope_reward` | True | `False` | huginn/autoloop/engine_reflect.py:2323 |

## 发现汇总

- 奖励项零调用者: reconcile_r_phys
- 同轴惩罚候选 (轮次): efficiency_discount, idle_turn_penalty
- 跨模块同名: reconcile_r_phys @ huginn/security/world_state.py
