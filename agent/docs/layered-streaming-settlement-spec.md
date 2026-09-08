# Spec: 分层流式结算（缺陷一/二的三阶段方案：P-A · P-B · P-C）

> 状态: **P-A / P-B / P-C 已落地**（`huginn/research/` 下 `program.py` + `replan_gate.py` +
> `early_stop_gate.py` + `aggregation_head.py`），串行/并发/压力测试全绿。
> 目标: 破除 batch-synchronous（BSP）结算的两个装配错误 —— 证据时序被抹平、DAG 边被
> 当作一次性编译的先验信念。方案是统一的一条循环：
> `interleave(plan, execute_layer, observe, replan, synthesize)`。
> 红线（贯穿三阶段，不可逾越）: **层间决策只做预算再分配** —— 不改任何已执行结果、
> 被跳过/终止的实验永不产生证据、全部决策记录在案（可证伪）。

## 1. 问题背景（BSP 的两个装配错误）

- **缺陷一**：batch-synchronous 结算让"决策上下文"只在**全部实验跑完后一次性结算**
  → 层间证据的时间顺序被抹平（先看到的证据不再先参与决策），且"到底要等多少层"是盲的，
  默认等完整个计划。
- **缺陷二**：计划期把依赖边编成 DAG 后照单执行 → 即使前序证据已表明某方向冗余/被证伪，
  后序实验照样排队（预算空转）；DAG 边被当作**一次性的先验信念编译物**，而非可修订的工作假说。
- **统一破法**：把"全跑完 → 结算 → 报告"拆成 `interleave(plan, execute_layer, observe,
  replan, synthesize)` 循环 —— 每层结算后先观察、再决定下一步怎么走。

## 2. 三阶段语义（映射表）

| 阶段 | 破解的缺陷 | 机制 | 开关（默认关） | 审计出口（唯一治理面） | 模块 |
|---|---|---|---|---|---|
| P-A | 一：证据时序抹平 | 分层**流式结算**：每完成真实实验即按层压缩记入有界 `stream_view` | `layer_epochs` | `out.consolidated.epochs` / `stream_view` | `program.py` |
| P-B | 二：DAG 边冻结 | **层间重规划门**：前序层结算后按证据跳过冗余/被证伪方向的实验 | `replan_gate` | `out.consolidated.replan` + `gate.replan` | `replan_gate.py` |
| P-C | 一：让"等"变得必要 | **证据驱动提前终止**：连续层 top-1 分数进入高原 → 终止剩余层 | `early_stop_gate` | `out.consolidated.early_stop` + `gate.early_stop` | `early_stop_gate.py` |

三阶段均可独立开关、可任意组合；都要求 planner 提供 `layers`（拓扑层映射），并共享
同一个 `_layers_map`，无独立副本。

## 3. 诚实红线（不可逾越）

1. **证据只进不出**：判定只基于 `cache` 里真实执行过的证据 —— 绝不基于未执行实验的猜测；
2. **保守放行**：前序层未全部结算（P-B）、观测层数不足（P-C 防早停）→ 一律放行/不判稳定；
3. **跳过 = 预算决策**：被跳过/终止的实验**不写 cache / stream_view / trace** ——
   "没做"就是"没做"，绝不装作做了；
4. **决策在案**：每条跳过/终止记录 {name, layer, reasons, evidence_from}，可反向审计、可证伪；
5. **默认零行为变化**：三个开关默认 `False` —— 不启用即完全等同原 BSP 管线。

## 4. P-A · 分层流式结算（evidence 时序不再抹平）

- **触发**：`layer_epochs=True` 且 planner 给出 `layers`。
- **机制**：executor 每完成一个真实实验，把 `summary` 经漏B 门控压缩（`distill_tool_output`，
  单条上界 1200 字符）后按拓扑层增量记入 `_stream_rows`；全部执行后按层分组为有界的
  `stream_view`（层序即时间序）。
- **无损性**：流式只改"表达"，不改"执行" —— `cache`/`verdict`/`pareto_front`/确定性报告
  与全 BSP 完全一致（测试锚定 `test_pa_lossless_preserves_bsp_numerics`）。
- **决策上下文有界**：LLM 综合阶段只读 `stream_view`（各层增量摘要之和，有上界）；
  全量可证伪证据始终留在 `trace` 供 grounding 门禁核对 —— 决策不再被"全跑完"绑架。

## 5. P-B · 层间重规划门（DAG 边在证据后可修订）

- **前置守卫**：前序层**全部结算**（执行 ∪ 已被重规划跳过）才启用判定；乱序调度一律放行。
- **判定规则**（纯函数，确定性，全在 `replan_gate.py`）：
  - `redundant_direction`：实验假说与某**已执行前序实验**假说的 Jaccard 重叠 ≥ 0.55
    → 该方向已被覆盖采样，继续执行是重复采样（预算空转）；
  - `falsified_direction`：某前序实验带 `predicted` 且对账相对误差 > 3%（方向被真实执行
    证伪）且假说重叠 → 死磕一个已被证伪的方向。
- **执行**：判定跳过的实验直接返回 `replanned`，不产生证据；理由记入
  `out.consolidated["replan"]["log"]`。
- **报告**：报告如实陈述被跳过实验与规则，证据以 `{"type": "replan_gate", ...}` 进 trace
  供 grounding 门禁核对（不引用任何不存在的实验结果）。

## 6. P-C · 证据驱动提前终止（稳定后不等全层）

- **稳定判据**（纯函数 `early_stop_gate.py`）：结算的**总层数** ≥ 2，且最后两层各取
  top-1（score 大者胜；平分取名字典序小者）后，得分相对变化 ≤ 2% → **分数高原** → 稳定。
  相对变化大（无论上升/下降 = 探索未饱和）→ 继续。
- **防早停门**：观测层数 < 2 一律判不稳定 —— 防止"第一层成绩就当占优"。
- **执行**：稳定且还有剩余层 → 激活 `early_stop`，剩余实验全部终止（预算回收，不产生
  证据）；终止名单与判据记入 `out.consolidated["early_stop"]`。
- **最终一致性**：并行（`max_parallel > 1`）下终止边界最终一致 —— 被终止层实验要么早已
  执行，要么必出现在 `early_stop["log"]`（测试锚定 `test_pabc_parallel_early_stop_final_consistent`）。

## 7. 三阶段协同

- 共享 `_layers_map`；P-C 的层结算判定**互认** P-B 的 skip 名单（重规划跳过的成员算
  "已决策"），不会因缺证据误判；
- 执行顺序：`early_stop 激活判定 → replan 逐实验判定 → 真实执行`（P-C 全局终止优先于
  P-B 局部修订）；
- **单一治理出口**：三阶段元数据统一从 `out.consolidated`（聚合头）读取 ——
  `gate.replan` / `gate.early_stop` 头 + `epochs` / `stream_view` / `replan` / `early_stop`
  字段；harness 六维报告的 `learning_capture` 投影两门。遵守"新视角只许注册进聚合头，
  不再加 out.* 字段"的架构纪律（`test_research_outcome_fields_are_frozen` 执法）。

## 8. 验收（测试锚定）

| 性质 | 测试 |
|---|---|
| P-A 无损性（流式不改执行） | `test_pa_lossless_preserves_bsp_numerics` |
| P-A 上下文有界 / 层有序 | `test_pa_stream_view_is_bounded_and_epoch_labelled` |
| P-B 冗余方向跳过 | `test_pb_replan_skips_redundant_direction` |
| P-B 证伪方向（world_model 对账） | `test_pb_falsified_direction_via_world_model` |
| P-B 前序未结算保守放行 | `test_pb_gate_guards_unsettled_prior_layer` |
| P-C 分数高原终止 | `test_pc_early_stop_on_score_plateau` |
| P-C top 移动不终止 / 防早停 | `test_pc_no_stop_when_top_moves` / `test_pc_single_layer_never_stops` |
| 无伪造（跳过/终止不入 cache/stream_view/pareto） | `test_pb_skip_never_fabricates_evidence` 等 |
| 三阶段正交组合 | `test_pabc_interleave_cooperative` |
| 并行最终一致（无静默丢失不变式） | `test_pabc_parallel_no_loss_invariant` 等 |
| 大压力组合（宽 DAG+并行+world_model+变异×3 轮） | `test_pabc_combined_stress_wide_dag` |
| 唯一治理出口纪律 | `test_research_outcome_fields_are_frozen` |

## 9. 里程碑

- [x] P-A: 分层流式结算 + 有界 stream_view（`_STREAM_SUMMARY_CHARS=1200`）。
- [x] P-B: 层间重规划门（redundant/falsified 双规则 + 前序结算守卫）。
- [x] P-C: 证据驱动提前终止（分数高原判据 + 防早停门）。
- [x] 三阶段组合 + 并发 + 大压力验证（`tests/test_aggregation_head.py`）。
- [x] 聚合头统一出口 + harness 投影（learning_capture）。
- [x] 生产化: 阈值已参数化(`stream_summary_chars` / `replan_similarity` /
     `early_stop_min_layers` / `early_stop_margin`)；稳定度先验已沉淀
     (`prior_store.extract_prior` → `run_research_program(prior=...)` 保守注入,
     min_layers 单调不减, prior_used 入聚合视图)。
- [x] 跨 run 先验 goal 归一化匹配: `prior_store.goal_slug`(与 program._slug_goal
     同口径) —— 异域先验(goal_slug ≠ 当前 goal)被 `resolve_early_stop_args` 拒绝
     套用(note=goal_mismatch), 先验只对同域生效, 绝不张冠李戴。
- [x] 先验时间衰减(A5): `prior["age"]` 按半衰期=1 的指数衰减(`0.5 ** age`,
     age=2 → 权重 0.25) —— 越旧先验权重越低, 领域漂移后旧经验自然淡出;
     诚实红线不变: 衰减只降权, min_layers 永不低于默认。
- [x] goal 模糊匹配(A6): `prior_store.goal_match_level` 用 token Jaccard 重叠分档
     (exact / related ≥0.5 / foreign / unknown) —— related 按重叠比例注入(半权起步),
     foreign 拒绝套用(文本也可证伪), unknown 向后兼容只受衰减影响。
     实施计划: `docs/superpowers/plans/2026-09-09-settlement-productionize.md`。

## 10. 非目标

- 不改动任何真实实验结果的科学性 —— 三阶段只做**调度/预算**决策；
- 不做黑盒"该方向值不值得研究"的质量判断 —— 只抓**冗余采样**与**分数高原**这两个
  确定性信号，其余交给人类的下一步决策；
- 不与具体 ExplorationStrategy（Pareto/变异/HITL）耦合 —— 三阶段在 executor 接缝处
  生效，对上游策略透明。