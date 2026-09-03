# Research Note: Life Operators 与 Huginn 的范式对照

> 状态: **report**（研究参考，非实现契约；据此驱动的设计变更需另立 spec/plan）
> 日期: 2026-09-03
> 来源: Wang Shuo, Yike Guo. *Life Operators: a self-evolving framework for multiscale
>       life modelling*. Preprint (2026-09-01), doi: 10.5281/zenodo.22210275
> 关联: `security/world_state.py` · `security/actuator_model.py` · `security/compute_adapter.py`
>      · `security/experiment_protocol.py` · `security/tool_registry.py`
>      · `security/behavior_lifecycle.py` · `autoloop/engine_observe.py`
>      与此前笔记的关系: 这是 `world-representation-inventory.md` 的"外部参照系"补充,
>      提纲挈领地把 project 已有的物理执行/世界表征/自演化词汇与论文算子语言对齐。

---

## TL;DR

**结论先行：不是"类似"，而是同一工程范式的两种词汇表。** 论文把可执行的"科学角色"
拆成 Perception / Evolution / Generation 三类算子 + 桥 + 任务图 + 科学契约，并以
"AI 提议 + 独立证据门"做版本化自演化；Huginn 用 microduck 学来的"执行器/传感器 + 前向
真值 + 生命周期"词汇，已在代码层面把其中绝大多数环节落地。两者共享同一条原则：
**self-evolution = 累积的版本化修订，不是无约束自改写；advisory，不硬编造。**

---

## 1. 论文核心框架

- **Perception (P)**: 观测 → 状态**后验** `p(s_t | x_{1:t}, a_{1:t}, u, c)`，避免把单一
  还原当 ground truth。提供其余算子使用的状态表示与其不确定性。
- **Evolution (E)**: `(s_t, u_t, c_t) ↦ p(s_{t+Δt} | s_t, u_t, c_t)`——自然进展与干预条件
  轨迹共用同一动力学框架。
- **Generation (G)**: `(s_t, a_t) ↦ p(x_t | s_t, a_t)`——内部状态 → 可测信号 + 信号采集/
  记录过程。
- **Bridge operator** (Eq.5): 跨变量、尺度、时间步的显式变换，传递**带不确定性的类型化
  消息** `r_{i→j,t}`；Scale Bridge 的上/下映射不等价、不互为逆。
- **Operator Graph Γ_Q**: 由问题决定的最小充分状态+机制集合；图本身是一个**可证伪假设**
  （缺后端/尺度 → 补桥/子图；未改善预测的组件被移除）。
- **科学契约**: 每算子声明 角色/输入输出/变量单位/假设/有效域/证据/版本/失败条件。
- **证据门控自演化** (Eq.6): `M_k=(S_k, Ω_k, B_k, Γ_{Q,k})`；AI co-scientist 提议候选
  `C_{k+1}`，**独立证据**（hold-out / 扰动实验 / 前瞻校验）决定保留/限制/退役。

---

## 2. 逐点映射到 Huginn

| Life Operators | Huginn 现状 | 位置 |
|---|---|---|
| **P** 观测 → 状态后验 | `StateEstimator` + `ObsVector`（共享观测契约、定长槽零填充）+ `PerceptionLayer` | `world_state.py` · `physics_schema.py` |
| **E** 状态+干预 → 未来 | `ForwardPredictor` + 解析前向真值（`IdealGasWorldModel`/`ShellComputeWorldModel.predict`）+ 可学代理 `LearnableForwardModel` | `world_state.py` · `thermo_system.py` · `compute_adapter.py` |
| **G** 状态 → 可测信号 | `SensorModelExecutor.sensor_view` + `_apply_bias`（systematic gauge 偏置 = 测量过程） | `actuator_model.py` |
| **Bridge · Scale Bridge**（跨变量/尺度/时间步，带不确定性的类型化消息） | `compute_adapter` 三形态后端（本地/远程HPC/HTTP）、ToolSpec schema、"世界模型是快代理 / 执行器是真实计算" 的 view-consistency 契约 | `compute_adapter.py` · `tool_registry.py` |
| **Operator Graph**（任务最小充分图，可 degrade） | `build_pipette_workflow` / `PhysicalWorkspace` 空间可组合依赖链（declare/provides/requires + CoEffectRegistry + 后端缺失自动停用不臆造） | `experiment_protocol.py` · `workspace.py` |
| **科学契约**（域/单位/假设/证据/版本/失败） | `ToolSpec` + `contract_version` + 工具级契约握手 + `physics_schema.SHARED_CONTRACT_VERSION` | `tool_registry.py` · `physics_schema.py` |
| **证据门控自演化** `M_k=(S,Ω,B,Γ)` | `BehaviorLifecycle` + `BehaviorArtifact` + 整体换目录 + 健康门控回滚 + 版本化 registry | `behavior_lifecycle.py` |
| **AI co-scientist 提议候选** | autoloop `_build_world_catalog_block`（planner 得知可预演工具）+ `dynamics_discovery_tool`（SINDy）+ learnable surrogate | `autoloop/engine_observe.py` · `tools/sci/` |
| **独立证据门 select（保留/限制/退役）** | `bandit/PRM/物理校验 r_phys` + 预测命中奖励 + `reconcile_r_phys` + `SparseRewardTuner`（从 (预测,实测) 学 sensor 偏置） | `world_state.py` · `bandit_controller.py` |
| **"逐层自洽、不冒充"**（当前模型保留/收窄域/带不确定性前进） | "advisory、不硬编造" 整体原则 | 贯穿 |

**最显著的对应**在自演化：论文的"self-evolution 是累积的版本化修订"，即 `BehaviorLifecycle`
那套"整目录换 + 健康门 + 回滚 + 版本化"的设计意图。我们按 microduck 学来的词汇、他们按
临床算子词汇，底下的结构是同一个。

---

## 3. 论文点破、我们偏弱的三处真缺

1. **Perception 应是"状态后验"，不是单一估计**。
   论文 Eq.4 = 显式贝叶斯滤波递归（Evolution 供转移先验、Generation 供观测似然、Perception
   做后验更新，且把"患者本身的变化"与"关于患者知识的变化"分离，不确定性不折叠成单一置信度）。
   Huginn 的 `StateEstimator` 只给启发式 σ（`_OBSERVED_SIGMA=0.01`），没有把"演化预测 + 新观测"
   逐轮折叠成后验。**衔接点**：可与已有的 `SparseRewardTuner`（学 sensor 偏置）和
   `validation/identifiability.identifiability_ceiling`（可辨识上限）组成一次最小后验更新。
   可落地性: 需设计（阶段7候选）。
2. **Bridge 要显式传输"带不确定性的类型化消息"**。
   论文 Eq.5 的 `r_{i→j,t}` 携带 有效参数 / 源项 / 边界条件 / 分布 + 其不确定性。Huginn 的
   `ToolSpec` 契约含 uncertainty，但桥路（`compute_adapter` 三后端）没有把 σ 沿连接显式传给
   接收组件。可落地性: 需设计（简单：桥契约加 uncertainty 透传）。
3. **"独立数据 + 冻结版本 + 前瞻校验"的选择门**。
   论文 Eq.6 要求候选与当前版本 + 更简单基线，用**独立证据**（hold-out / 扰动 / 前瞻）比对；
   同一数据不得既生成候选又最终确认。Huginn 的 `r_phys`/预测命中奖励是启发式聚合，缺真正
   "独立数据选候选"一步——`BehaviorLifecycle` 健康门 + `reconcile_r_phys` 已有雏形。可落地性: 需设计。

---

## 4. 反向：我们能补他（论文停在愿景处）

- 论文给的是小样板（HCM 药物响应，Box 1）与"medical ASI"愿景；Huginn 已有**可运行**的
  执行器/传感器基类（`SensorModelExecutor`）、事务回滚（`RevertibleContext`）、分层安全仲裁
  （`control_safety`/`control_authority`）、数据驱动前向代理（`LearnableForwardModel`）。
- 这些正是论文设想的、处于不同实现形态（方程 / 统计 / 神经网络 / 混合）的 P/E/G 算子实例；
  可作为"算子实现库"直接反哺其框架。

---

## 5. 结论与建议

- **"同构"= 真**：P-E-G ↔ PerceptionLayer/StateEstimator + ForwardPredictor/前向真值 +
  SensorModelExecutor；桥 ↔ compute_adapter；算子图 ↔ PhysicalWorkspace 依赖链；科学契约 ↔
  ToolSpec/contract_version；证据门控自演化 ↔ BehaviorLifecycle。
- **下一步候选**（均需另立 spec/plan）：A. `StateEstimator` 升级为带后验传播的最小滤波；
  B. compute_adapter 桥补不确定度透传；C. 显式"独立数据选候选"门。

> 引用需遵守原文 CC-BY-SA 4.0（注明作者与许可）。本备忘不杜撰论文未声明内容。