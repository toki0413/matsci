# 世界表征闭环 · 现状对照盘点（已存在 vs 真缺 vs 需接线）

> 状态：report（审计快照）
> 触发：核实"机器能否从感知形成可泛化、可交互、支持行动的世界表征"这一认知链在
> Huginn 的真实落地程度。**结论先行：项目已"做过"相当多，多数部件已接入某种形态；
> 真正缺的不是再造一个世界模型，而是"状态估计 → 前向预测"这对主循环内闭环，
> 以及它与记忆/奖励的打通。**
> 全部结论基于 src 代码亲验（Grep 符号级），非文档自述。引用路径由
> `tests/test_world_inventory.py` 实测存在，漂移即红。

---

## 1. 盘点方法

- 对四层认知链的每个部件，用 `Grep` 在运行时代码（`huginn/**/*.py`）里按符号级匹配核对"存在性"与"是否被主循环引用"。
- "接主循环"指被 autoloop 引擎（`engine_perceive` / `engine_observe` / `cognitive_loop`）或执行链（`BehaviorLifecycle` / `ExecutionGuard`）直接 import/调用。
- 只列实证；拿不准的标 ⚠️ 并给证据行。

## 2. 四层 × 部件 对照表

### ① 看见什么（感知）
| 部件 | 位置 | 证据 | 接主循环 | 结论 |
|---|---|---|---|---|
| 感知层 L1-L4 / `PerceptionSnapshot` | `huginn/perception/__init__.py`(L65/L153) | `get_cognitive_state()` 存在 | ✅ `engine_perceive.py:55` import、`:447` 调 `get_cognitive_state()` | 已接入 |
| 视觉路由/模态/CV 编码 | `vision/router.py`、`modality_router.py`、`visual_encoder.py` | `VisionRoute{NATIVE_LLM,CV_TOOLS,TEXT_ONLY}` | ✅（供 perceive 消费） | 已接入 |
| OCR/知识入库 | `knowledge/ocr_loader.py` | `extract_text_with_ocr` | ⚠️（知识线路，非实时传感环） | 已存在/部分接入 |

### ② 正在发生什么（状态）
| 部件 | 位置 | 证据 | 接主循环 | 结论 |
|---|---|---|---|---|
| 共享观测契约+零填充 | `security/physics_schema.py`(L257-268) | `OP_PARAM_SLOTS`/`canonical_params`/`SHARED_CONTRACT_VERSION` | ⚠️ 仅动作参数槽 | 已存在，范围窄 |
| 可观测空间声明 | `security/tool_registry.py` | `schema.space{state,action}` + `observables` | ⚠️ 消费方未完全 | 岛状 |
| 观测最小表征(σ) | `metacog/hypothesis_manifold.py` | `Observation(name,value,sigma)` | ⚠️ | 岛状 |
| 可辨识性档位 | `validation/identifiability.py` | `identifiability_ceiling`/`lambda_min` | ⚠️ | 岛状 |
| **`StateEstimator` / `StateSnapshot` / `ObsVector`** | — | Grep **0 命中** | — | **真缺** |

### ③ 接下来会发生什么（预测）
| 部件 | 位置 | 证据 | 接主循环 | 结论 |
|---|---|---|---|---|
| 懒路世界模型块 | `autoloop/engine_observe.py`(L355) | `_build_world_model_block` | ✅ 已注入 `[WORLD MODEL]`，注释"无外推" | 已接入·简化 |
| 解析前向真值 | `security/thermo_system.py`(`IdealGasWorldModel.predict`)、`compute_adapter.py`(`ShellComputeWorldModel.predict`) | `predict` 存在 | ⚠️ 按工具调用，非主动逐轮 | 岛状 |
| 数据驱动预测 | `tools/sci/dynamics_discovery_tool.py`(SINDy)、`msm_tool.py`、`tools/neural_proxy.py` | 均可 predict | ⚠️ 需被调用 | 岛状 |
| **`ForwardPredictor`** | — | Grep **0 命中** | — | **真缺** |

### ④ 为了目标应该做什么（行动）
| 部件 | 位置 | 证据 | 接主循环 | 结论 |
|---|---|---|---|---|
| 逆模型/回滚 | `security/world_model.py` | `WorldModel.infer_inverse`/`check_constraints` | ✅ | 已接入 |
| 传感器前向+sensor_view+标定 | `security/actuator_model.py` | `SensorModelExecutor` | ✅ | 已接入 |
| 真实计算工具(=仪表) | `security/compute_adapter.py` | `ExternalComputeExecutor` | ✅ | 已接入 |
| 分层策略+死手+仲裁 | `security/control_safety.py`、`control_authority.py` | `GatePolicy`/`Deadman`/优先级 | ✅ | 已接入 |
| 启发式奖励(UCB/MDP/MC/PRM/物理分) | `agent/bandit_controller.py`、`runtime/step_verifier.py`、`tools/validate_tool.py` | `r_phys`/PRM/bandit | ✅ | 已接入(无梯度) |
| **预测命中 → 奖励/记忆回流** | — | Grep 无"预测命中率入 reward" | — | **半缺** |

## 3. 关键实证更正（相对先前讨论）

- **感知层确实已进主循环**：由 `engine_perceive.py`（autoloop 的 perceive 阶段）import 并调用
  `PerceptionLayer` 与 `get_cognitive_state()`——并非孤岛。
- **主循环已有一个世界模型块**：`engine_observe._build_world_model_block` 用 `longterm.predict_via_analogy`
  检索历史实验注入 `[WORLD MODEL]`，但它标注"**无外推**"——即"记忆类比"，不是"前向数值预测"。
- **与 rewards 类似**：奖励也不是空的——是 bandit/PRM/`r_phys` 的启发式链（无梯度训练）。

## 4. 真缺清单（唯一需要做的事）

1) **无 `StateEstimator`/`StateSnapshot`/`ObsVector`**：物理状态仍以 `dict` 存在，没有
    "带不确定性（均值±σ）的状态分布"统一进入主循环。
2) **无 `ForwardPredictor`**：主循环没有逐轮"当前状态 → 前向数值预测"，只有类比检索块。
3) **预测不回流**：预测命中率未作为 reward/记忆特征送入 bandit 与 episodic。
4) （可选）**深度 RL 奖励未落地**：`docs/reward_design.md` 为理论稿，无梯度训练，无
    "预测命中→过程奖励"的稀疏回流。

## 5. 结论与建议方向

- **"做过"= 真**：感知、观测契约、解析前向真值、逆模型/tsim2real、分层控制、启发式 reward
  都已存在且多数已挂到主循环/执行链上。
- **"再造"= 不必**：不要再从零写一个世界模型。
- **该做 = 收口/接线**：以 `physics_schema` 的契约 + `ToolSpec.observables` 为锚，把
  `IdealGas/ShellCompute` 等解析真值 + `identifiability` 收成一个 `StateEstimator(+Snapshot)`，
  再补一个逐轮 `ForwardPredictor`，让预测命中回流到 bandit/episodic。这才是当前唯一真缺，
  且改动量远小于"重造"。

---

## 6. 相关文件索引（供追踪）
`huginn/autoloop/{engine_perceive,engine_observe}.py` · `huginn/perception/__init__.py` ·
`huginn/security/{physics_schema,tool_registry,world_model,actuator_model,compute_adapter,control_safety,control_authority,thermo_system}.py` ·
`huginn/validation/identifiability.py` · `huginn/tools/{sci/dynamics_discovery_tool,sci/msm_tool,neural_proxy,validate_tool}.py` ·
`huginn/agent/bandit_controller.py` · `huginn/runtime/step_verifier.py` · `docs/research-notes/physical-rsi-and-world-model-interpretability.md`