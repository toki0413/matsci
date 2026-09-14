# Recursive Compounding — 策略器可演化 + 复合护栏（轨道 A1）

> 状态：design-approved（2026-09-13）。前置：M-R1 recursive (meta-)improver 已落地（`huginn/harness/meta_improver.py`）。目标：补「递归自我改进 + 复合加速」——让改进器自身的改进策略可被改进，并用复合护栏验收、防自举发散。
> 关联：`docs/staging/specs/2026-09-13-recursive-improver.md`（M-R1，本设计的直接上层）。

## 动机（audit 结论 + M-R1 现状）

- M-R1 已是"单层递归"：`MetaImprover.maybe_propose` 用硬编码 `_META_IMPROVE_TEMPLATE` 生成改进器模板候选，经 `SignificanceGate`（显著）+ `OODHoldoutValidator`（防背题）GREEN 后换 champion。**但改进器如何提方案（strategy）本身是写死的** → 改进带来的改进不随"改进能力的改进"加速，即无复合（compounding）。
- 距 AGI 的自举缺口正是：I_t = I_{t-1}(I_{t-1}) 的递归深度 + 「改进使改进本身更快/更省」的复合验证。
- 本设计加**一层**递归（strategist 可演化）+ **复合护栏**（滚动窗口验收 + 发散回退）。不做无限递归（YAGNI）。

## contract: StrategistImprover（新增，扩展 `huginn/harness/meta_improver.py`）

```
StrategistConfig  dataclass          # level-1: 生成改进器候选的"策略"模板
    config_id: str
    strategist_prompt: str           # 覆盖 _META_IMPROVE_TEMPLATE 的模板
    active: bool = False
    created_at: float

CompoundingTracker                    # 复合验收（滚动窗口）
    window: int = 8
    slope_tolerance: float = 0.05     # 质量斜率容差 — 只判"不退化", 不苛求严格上升
    hysteresis_band: float = 0.03     # revert 滞回带 — 抑制 A→B→A 抖动
    record(epoch, config_id, champion_quality_mean, fidelity_anchor,
           proposals_to_promotion, generations_to_promotion, win_rate) -> None
    is_compounding() -> bool          # (质量斜率≥-tol) 且 (每次换件成本不持续升)
    would_degrade(new_cid, incumbent_cid) -> bool
    deadlock_timeout: int = 5         # strategist 连续 N 次 YELLOW → 降低门槛/改 advisory
    stats() -> dict                   # {slope, cost_trend, window, best_quality, ongoing, deadlock_n}

BehavioralFidelity  (#1 行为级奖励回流 — Goodhart 锚)
    record_acceptance(candidate_id, accepted: bool)      # 真实 apply_patches 是否采纳
    fidelity_score(candidate_id) -> float                 # 采纳率 ETF 平滑，作复合指标的保真锚
    anchor_in(candidate_id, p_quality, p_fidelity) -> float  # 加权合成: quality↔fidelity 平衡

RandomizedControl  (#3 对照验收)
    run_pair(strategist_id, baseline_id, n=3) -> dict    # 随机化 + 真实 r_phys 差分
    requires: HUGINN_META_ABLATION=1 (默认 off, 仅显式评估时开)

StrategistImprover (MetaImprover 内部扩展, 同单例)
    strategist_champion() -> StrategistConfig | None
    maybe_propose_strategist(llm_chat_fn) -> str | None   # meta² 固定模板生成候选
    evaluate_strategist(cid, llm_chat_fn) -> dict         # 复用众 + CompoundingTracker
    maybe_promote_strategist(cid) -> bool                 # GREEN 且 !would_degrade
    revert_strategist(cid) -> bool                        # 复合退化时回退上一 champion
```

契约要点：
- `maybe_propose`（level-0）改用 `strategist_champion().strategist_prompt`（无则回落 `_META_IMPROVE_TEMPLATE`）→ 改进方式可被改进。
- **只做一层递归**：strategist 自身的 meta² 模板固定，不再递归（A1 边界）。
- `maybe_promote_strategist` 门控 = `SignificanceGate` GREEN **且** `OODHoldoutValidator` 通过 **且** `CompoundingTracker.would_degrade(new, incumbent)` 为 False **且** 复合指标用 `BehavioralFidelity.anchor_in`（质量⊕真实采纳保真）合成（#1）。
- **Goodhart 抑制（#4）**：evaluate 用一份**保留的真实结果子集**（holdout）做最终判断，该子集对 strategist 不可见 —— 防止 strategist 只优化可见代理分。
- 复合退化即 `revert_strategist` 回上一 champion（元层级 darwin-ratchet），**带滞回带**（#2）防 A→B→A 抖动。
- **死锁检测（#2）**：strategist 连续 `deadlock_timeout` 次 YELLOW（样本不足/不显著）→ 自动降级门槛（降 `min_samples`）或改 advisory，避免"演化近惰性"。
- **随机化对照（#3）**：显式评估时（`HUGINN_META_ABLATION=1`）用 champion vs 固定 baseline 各跑 N 次，比较**真实 r_phys 差分**验收（比单点代理分可靠）。默认 off。

## data shape

- `.huginn/harness/meta_improver/strategist/<id>.json`: `StrategistConfig.to_dict()` + `config.json` 记录 `{active_strategist_id, strategies_history}`。
- `.huginn/harness/meta_improver/compounding.json`: `{window: [...], slope, cost_trend, best_quality, deadlock_n}`。
- `.huginn/harness/meta_improver/fidelity.json`: `{candidate_id: {accepted, applied, fidelity_score}}`（BehavioralFidelity 采纳统计，LRU 上限 50）。
- `meta_trace.jsonl` 追加事件类型 `strategy_propose / strategy_evaluate / strategy_promote / strategy_revert / fidelity_accept`。
- 注意：strategist 与 improver 两组 champion 独立持久化，互不覆盖；`strategist/` 与 `fidelity.json` 均设 LRU 上限防无限增长（#2 资源）。

## 鲁棒性加固（并入 A1，四项）

1. **行为级奖励回流（BehavioralFidelity）**：复合指标不只依赖代理分，加权融合**真实采纳率**（`apply_patches` 是否真的采纳该 patch）——代理分可能饱和/被优化，真实行为采纳是保真锚，治 Goodhart。
2. **死锁检测 + 滞回带**：strategist 连续 `deadlock_timeout=5` 次 YELLOW → 自动降 `min_samples` 或改 advisory（防"演化近惰性"）；`revert` 带 `hysteresis_band=0.03`，抑制代理分噪声下的 A→B→A 抖动。
3. **随机化对照（RandomizedControl）**：显式评估（`HUGINN_META_ABLATION=1`）时，champion vs 固定 baseline 各跑 N=3，比较**真实 r_phys 差分**验收——比单点代理分可靠，作为疑虑时的仲裁手段。
4. **Goodhart 抑制（保留真实结果子集）**：evaluate 用一份对 strategist **不可见的 holdout 真实结果子集**做最终判断；strategist 无法只优化可见代理分。

## 验收（test）

新增到 `tests/test_meta_improver.py` + `python -m huginn.harness.meta_improver` selfcheck 扩展：

1. **零回归**：无 strategist champion → `maybe_propose` 回落默认模板。
2. **好 strategist**（候选在重放集上提升改进器产出 rate 且 GREEN）→ `maybe_promote_strategist` True，champion 切换。
3. **差/退化 strategist** → `would_degrade` True → 不换件。
4. **发散回退**：切换后复合下滑 → `revert_strategist` 回到上一 champion（带回滞带验证）。
5. **CompoundingTracker 数学**：斜率/效率/窗口/`is_compounding` 边界 + 滞回带/死锁计数。
6. **BehavioralFidelity 锚**：采纳率高则复合指标上修、低则下修；`fidelity.json` 持久化。
7. **随机化对照**：`run_pair` 在寓真 r_phys 差分下判定 champion vs baseline（`HUGINN_META_ABLATION=1` 分支）。
8. **Goodhart holdout**：strategist 看不到保留子集 → 不因可见代理分过度贴合。

## failure modes / deferred

1. 二次滚动的样本稀释：strategist 层也要攒够 sig+OOD 样本 → 设定 `min_samples` 与窗口 W 平衡，样本不足即 YELLOW 不换件（不影响 level-0）。
2. 复合指标噪声：用容差 + 中位数而非瞬时值；`is_compounding` 只判"不退化"，不苛求严格上升。仍不达 → 走死锁降级（#2）。
3. Goodhart 残余：代理分 + 采纳回流的融合不能完全排除"策略器为讨好采纳率而行为固化"。缓解：保留真实结果 holdout 子集 + 随机化对照仲裁；根治在接真实 r_phys 端到端（明确列为后续轨道）。
4. 递归过深导致开销：A1 明确只一层，strategist 的 meta² 模板写死，绝不叠第三层。
5. **不做**：直接自改源码（A3）、无限递归、让 strategist 自己递归改自己（留作后续轨道，先验证一层）。
6. model 维度不纳入（沿用 H3 P8 限制）。
7. 真 r_phys 端到端验收是根治 Goodhart 的路径，但成本高、需 compute-heavy objective；本设计以 `BehavioralFidelity` + holdout + 随机化对照为近似，端到端列为独立后续阶段。

## 与现有工作关系

- `MetaImprover`（M-R1）：被扩展为 level-0（improver）与 level-1（strategist）两层，复用其 gate 数学。
- `SignificanceGate` / `OODHoldoutValidator` / `AdoptionGate`：复用，不重写。
- `_enabled.py`：`harness_meta_improver` 开启后两层才生效（默认关）。
- ROADMAP 独立轨道；设计文档 `2026-09-13-recursive-compounding-design.md`。