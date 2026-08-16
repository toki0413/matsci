# 独立审计：metacog 头接线核查 与 blind_spot_mapper 去孤岛

**状态**: report · **日期**: 2026-08-16 · **性质**: 第三方只读审计 + 一处落地

## 背景

对 HUGINN 主认知循环（autoloop/engine）做独立 seed→use 追踪，回答"metacog 视觉/时空/拓扑/自我认知头是真被消费，还是装饰/孤岛"。

## 结论

1. **8 个 head 确认主环真实接线**，非装饰：
   - **硬门控**（改变控制流）：`cognitive_heat_engine.should_imaginate`（[hypothesis_loop.py L2547](huginn/autoloop/hypothesis_loop.py)）、`decision_arbiter`（[cognitive_loop.py](huginn/autoloop/cognitive_loop.py)）。
   - **advisory 注入**（结构化提示、走优先级+预算裁剪 `_trim_to_budget`）：拓扑（simplicial/sheaf/Hodge）、category_functor、episodic_replay、mental_imagery。聚合是"优先级排序 + 预算裁剪"的真 selection，不是无脑拼接。
2. **两大"主环孤岛"被审计暴露**：`trace_topology`（主环被 simplicial_homology 取代，仅旧 rcb 层在用）与 **`blind_spot_mapper`（生产消费点只在旧 rcb_step2，主环零接线）**。
3. **blind_spot_mapper 已去孤岛**：接入 [engine_observe.py](huginn/autoloop/engine_observe.py) 的 `_build_hypothesis_prompt`，作为优先级列表（`topo` 之后）的 advisory block。

## blind_spot 接入的关键取舍

- autoloop 的 self-model 是 `memory.longterm.get_self_model()` 的 **cluster 汇总 dict**（persona 粒度 `rate/n`），没有 per-skill 三档（blind/uncertain/capable）。per-skill `SelfModel` 只在 rcb_step2 实例化并喂 `update_from_step`。
- 因此当前判据用 **`rate==0 && n>=3`（确认从未成功）当作 "blind" 档**，与 `[CURIOSITY]`（`rate<0.4`，主张探索）互补不重复。
- 诚实 ceiling（已写入代码注释）：盲点粒度=persona，`_DEFAULT_WORKAROUNDS` 按 skill 名匹配大概率 miss → 走通用"绕过"建议而非 `vasp→少 k 点/转 QE` 这类 skill 级建议。
- 测试：`tests/test_math_prompt_injection.py::TestBlindSpotBlockWiring`（4 条：注入 / 弱簇不注入 / 无 memory 静默 / block 序在 topo 后）。无回归（159 passed）。

## 结论：为何不做 per-skill `SelfModel` 升级（保持现状）

**不需要升级**。理由：

1. 当前接线已满足目标——盲点从"旧 rcb 孤岛"成为主环真能用、能出数据、静默安全的 advisory 块；升级只改*质量*（workaround 更精准、可联动 imagination 种子），不修 bug。
2. 升级有反直觉坑：若只调 `infer_blind_spots(SelfModel)` 而不在同一处喂 `update_from_step(工具结果)`，`SelfModel` 为空 → 返回空 → 盲点块反而比现在的壳更弱。而"每轮工具结果喂 SelfModel"需要一个干净的 tool-outcome hook，autoloop 主循环没有现成落点，硬接有回归风险。
3. 当以下任一成立时才值得升级：
   - 需要按能力名给**具体绕过建议**（`vasp→少 k 点`、`rdkit→查 SMILES`）而非通用提示；
   - 需要 **"盲点→imagination 重构"完整闭环**（`pick_imagination_seed` → `imagine_from_blind_spot`），而不只是注入一句话；
   - 产品把"agent 自我认知"从 cluster 级明确提升到 **skill 级**。

升级路径（记录，不主动做）：①在 autoloop 持有并持久化 `SelfModel`（task_local + cross_task 累积）②在工具结果落地处喂 `update_from_step` ③盲点块改调 `infer_blind_spots` ④（可选）接 imagination 种子。