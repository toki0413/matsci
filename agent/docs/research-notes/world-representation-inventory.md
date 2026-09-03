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
| **`StateEstimator` / `StateSnapshot` / `ObsVector`** | `security/world_state.py` | 已实现(阶段0) | 未接主循环 | 已实现·未接线 |

### ③ 接下来会发生什么（预测）
| 部件 | 位置 | 证据 | 接主循环 | 结论 |
|---|---|---|---|---|
| 懒路世界模型块 | `autoloop/engine_observe.py`(L355) | `_build_world_model_block` | ✅ 已注入 `[WORLD MODEL]`，注释"无外推" | 已接入·简化 |
| 解析前向真值 | `security/thermo_system.py`(`IdealGasWorldModel.predict`)、`compute_adapter.py`(`ShellComputeWorldModel.predict`) | `predict` 存在 | ⚠️ 按工具调用，非主动逐轮 | 岛状 |
| 数据驱动预测 | `tools/sci/dynamics_discovery_tool.py`(SINDy)、`msm_tool.py`、`tools/neural_proxy.py` | 均可 predict | ⚠️ 需被调用 | 岛状 |
| **`ForwardPredictor`** | `security/world_state.py` | 已实现(阶段0) | 未接主循环 | 已实现·未接线 |

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

## 4. 真缺清单（更新：阶段 0/1/2/3 已落地）

> 阶段 0（收口核心件）**已于 `security/world_state.py` 实现**：`ObsVector`（定长槽零填充+契约握手）、
> `StateSnapshot`（状态+不确定度+可辨识档位）、`StateEstimator`（由 `space.state/observables` 生成状态分布）、
> `ForwardPredictor`（逐轮前向投影 + `[WORLD PREDICTION]`，命中回填 `predicted`）。
>
> 阶段 1（接线，已落地）：`WorldStateTracker`（`StateEstimator`+`ForwardPredictor`+解析前向真值）以可选 `schema=`
> 接入 `PhysicalWorkspace.execute`，每动作记录含 `predicted` 的快照并算预测 vs 实测的相对 RMS 误差。
> 全部 advisory、失败静默不阻塞。
>
> 阶段 2（奖励回流，已落地）：`prediction_error_to_reward(error)` 把误差映射为 [0,1] 命中奖励；
> tracker/workspace 暴露 `last_prediction_reward()`；端到端测试喂进 `WorkflowBandit(r_phys=reward)` 与
> `ProjectKnowledgeGraph.add_episode_node`（bandit+episodic）。
>
> 阶段 3（runner 默认消费，已落地）：tracker 累计运行期奖励并暴露 `avg_reward()`，workspace 出
> `prediction_reward_avg()` 与 `reconcile_r_phys(base)`（base 与平均预测奖励各 0.5 加权；无世界跟踪时
> 原样返回 base 不扣分）；生产 runner `build_pipette_workflow` 新增可选 `schema=` 透传给工作台，使
> 物理工具 runner 可把 `last_prediction_reward` 并进其 `r_phys` 聚合。
>
> 尚未做（真缺）：
1) **autoloop LLM prompt 注入未接**：`StateEstimator`/`ForwardPredictor` 尚未进入 autoloop 的
   `_build_world_model_block`（现注的是"历史类比"），只因该路径缺稳定的物理工具 schema/state。
2) **各物理 tool 未逐个接 `reconcile_r_phys`**：`reconcile_r_phys` 是权威出口，但上游各物理 tool
   尚未逐个把其工作台结果调到它再喂 bandit。
3) （可选）**深度 RL 奖励未落地**：`docs/reward_design.md` 为理论稿，无梯度训练，无
   "预测命中→过程奖励"的稀疏回流。

## 5. 结论与建议方向

- **"做过"= 真**：感知、观测契约、解析前向真值、逆模型/tsim2real、分层控制、启发式 reward
  都已存在且多数已挂到主循环/执行链上。
- **"再造"= 不必**：不要再从零写一个世界模型。
- **该做 = 接线/回流（阶段 0/1/2/3 已完成）**：以 `physics_schema` 契约 + `ToolSpec.observables`
  + `ToolSpec.build_world_model` 为锚，`WorldStateTracker` 已在 `PhysicalWorkspace` 产出逐轮快照、
  预测命中奖励，并经 `reconcile_r_phys` 可作为物理 runner 的 r_phys 贡献；奖励已能喂进 bandit 与
  episodic。剩余是把 `reconcile_r_phys` 逐个接进各物理 tool（阶段 4），改动量远小于"重造"。

---

## 6. 相关文件索引（供追踪）
`huginn/autoloop/{engine_perceive,engine_observe}.py` · `huginn/perception/__init__.py` ·
`huginn/security/{physics_schema,tool_registry,world_model,actuator_model,compute_adapter,control_safety,control_authority,thermo_system,world_state}.py` ·
`huginn/validation/identifiability.py` · `huginn/tools/{sci/dynamics_discovery_tool,sci/msm_tool,neural_proxy,validate_tool}.py` ·
`huginn/agent/bandit_controller.py` · `huginn/runtime/step_verifier.py` · `docs/research-notes/physical-rsi-and-world-model-interpretability.md`