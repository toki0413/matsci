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
    record_acceptance(candidate_id, accepted: bool, verified: bool | None = None, # 论文: 是否经状态化仿真验证
                      ) -> None                                # verified=True → 变换类型化对象且前置/约束通过
    fidelity_score(candidate_id) -> float                 # 采纳率(或验证率) ETF 平滑，作复合指标的保真锚
    anchor_in(candidate_id, p_quality, p_fidelity) -> float  # 加权合成: quality↔fidelity 平衡

VerifiableGate  (#论文 2609.03621: 可计算实验室表示 → 可验证工作流)
    # 策略候选被判"值得换件"当且仅当其改进器产出的实验/计划, 在状态化仿真下
    # 验证通过: 变换了类型化研究对象、op 前置条件满足、不违反实验室约束.
    verify_outcome(plan_or_experiment: dict) -> dict      # 尽力接入 world_model.apply_forward/check_constraints
    enabled() -> bool                                     # 机器可用且显式开启才硬 gate; 否则 advisory(不阻塞)

RandomizedControl  (#3 对照验收)
    run_pair(strategist_id, baseline_id, n=3) -> dict    # 随机化 + 真实 r_phys 差分
    requires: HUGINN_META_ABLATION=1 (默认 off, 仅显式评估时开)

RPhysTrack  (#6 端到端 r_phys 真实验收 — 主板优化代理分的根治通道)
    # 把 note_generation 每次真实产 patch 的 r_phys 归因到「当时代际驱动该 patch
    # 的改进器配置」, 形成按配置的真实 r_phys 代际序列, 用不成对两样本检验
    # (Mann-Whitney U, 单侧, 纯 stdlib) 判定:
    #     H0: median(r_phys|config) <= median(r_phys|其余配置池)
    # 显著上行 + 样本充足 → rphys_green. 默认 advisory 不阻塞; 显式开
    # harness_rphys_gate 且数据可判时作硬闸 (挡「代理分说好但真实 r_phys 没跟上」).
    record(config_id, r_phys) -> None                    # note_generation 归因 + 落盘 rphys.json
    series(config_id) -> list[float]                     # 该配置的真实 r_phys 代际
    verdict(config_id) -> dict                           # {green, n, pool, p, median_*, reason}
```
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
- **时空可组合（Cordis，并入）**：
  - **时间**：`maybe_promote_strategist` 在 `RevertibleContext.transaction()` 内执行；换件时 `ctx.compensate("strategist_swap", {old_cid, new_cid, store})` 并注册 `_compensate_strategist_swap`。复合护栏触发 → 对本 scope `revert_all()` LIFO 撤销 → 回滚到上一 champion，并**与同一 run 内其他副作用（git_commit / memory）扭结恢复**。该 op 可落 journal → `recover_from` 跨崩溃重放恢复正确 champion。
  - **空间**：用 `CoEffectRegistry.declare` 声明依赖 `strategist(provides=improvement_strategy, requires=gate/fidelity)`、`improver(requires=improvement_strategy)`、`gate(provides=gate)`、`fidelity(provides=fidelity)`；strategist 缺失/被 degrade → improver 自动退化到默认模板（`degrade 不停跑无依据步骤`），与 `PhysicalWorkspace` 同一立场。

## data shape

- `.huginn/harness/meta_improver/strategist/<id>.json`: `StrategistConfig.to_dict()` + `config.json` 记录 `{active_strategist_id, strategies_history}`。
- `.huginn/harness/meta_improver/compounding.json`: `{window: [...], slope, cost_trend, best_quality, deadlock_n}`。
- `.huginn/harness/meta_improver/fidelity.json`: `{candidate_id: {accepted, applied, fidelity_score}}`（BehavioralFidelity 采纳统计，LRU 上限 50）。
- `.huginn/harness/meta_improver/revertible_journal.json`: strategist 换件的补偿逆（`OP_COMPENSATE`）journal，供 `recover_from` 跨崩溃重放。
- `.huginn/harness/meta_improver/rphys.json`: `{series: {config_id: [真实 r_phys 代际...]}}`（RPhysTrack 端到端真实归因，LRU 上限 10/配置）。
- `meta_trace.jsonl` 追加事件类型 `strategy_propose / strategy_evaluate / strategy_promote / strategy_revert / fidelity_accept / evaluate{rphys_green,rphys_n} / reject{rphys_fail}`。
- 注意：strategist 与 improver 两组 champion 独立持久化，互不覆盖；`strategist/` 与 `fidelity.json` 均设 LRU 上限防无限增长（#2 资源）。

## 鲁棒性加固（并入 A1，六项）

1. **行为级奖励回流（BehavioralFidelity）**：复合指标不只依赖代理分，加权融合**真实采纳率**（`apply_patches` 是否真的采纳该 patch）——代理分可能饱和/被优化，真实行为采纳是保真锚，治 Goodhart。
2. **死锁检测 + 滞回带**：strategist 连续 `deadlock_timeout=5` 次 YELLOW → 自动降 `min_samples` 或改 advisory（防"演化近惰性"）；`revert` 带 `hysteresis_band=0.03`，抑制代理分噪声下的 A→B→A 抖动。
3. **随机化对照（RandomizedControl）**：显式评估（`HUGINN_META_ABLATION=1`）时，champion vs 固定 baseline 各跑 N=3，比较**真实 r_phys 差分**验收——比单点代理分可靠，作为疑虑时的仲裁手段。
4. **Goodhart 抑制（保留真实结果子集）**：evaluate 用一份对 strategist **不可见的 holdout 真实结果子集**做最终判断；strategist 无法只优化可见代理分。
5. **可验证工作流验收门控（VerifiableGate，结合论文 arXiv:2609.03621）**：把 strategist 换件锚定从"代理分+采纳率"再上一级到**可验证的研究状态变换**——候选只有当其改进器产出的实验/计划在状态化仿真（优先接 `world_model.apply_forward` / `check_constraints`）下**验证通过**（变换类型化研究对象 + 前置满足 + 不违反约束）才被换件。机器可用且显式开启时硬 gate；不可用/未开时 advisory（不阻塞推进，回落 BehavioralFidelity 锚）。这按该文主张，把验收从"优化标量代理"根治为"验证能力受限变换"，是 Goodhart 的最终闸。
6. **端到端 r_phys 真实验收（RPhysTrack）**：把 `note_generation` 每次真实 r_phys 归因到当时代际实际驱动 patch 的改进器配置，形成按配置的真实 r_phys 代际序列，用**不成对两样本检验（Mann-Whitney U，纯 stdlib，无 scipy 依赖）**判定该配置是否伴随真实 r_phys 显著上行。默认 advisory（样本不足 `green=None` 不阻塞，回落既有代理分门控）；显式开 `harness_rphys_gate` 且数据可判时作**硬闸**，把「LLM 代理分说好、但真实物理验证分没跟上」的候选挡在 champion 之外——这是 LLM 代理分之外的最终地面真值通道，直接落实上述论文"从优化标量代理根治为验证真实变换"。

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
9. **时空可组合**：`maybe_promote_strategist` 在 `transaction()` 内；切换后在 `revert_all()` 下恢复上一 champion；`strategist_swap` 补偿器经 `compensate` 注册、journal 可重放；`CoEffectRegistry.declare` 使 improver 在 strategist 缺席时 `is_active=False` → 退化默认。
10. **端到端 r_phys**：`note_generation` 把真实 r_phys 归因到 active 配置（无 champion → `_base`）；`RPhysTrack.verdict` 对上行配置判 green、下行/不足判非 green/None；`harness_rphys_gate` 开启且候选真实 r_phys 未上行 → `maybe_promote` 拒绝（默认 off 不拦）；`rphys.json` 持久化 + reload 保留。

## failure modes / deferred

1. 二次滚动的样本稀释：strategist 层也要攒够 sig+OOD 样本 → 设定 `min_samples` 与窗口 W 平衡，样本不足即 YELLOW 不换件（不影响 level-0）。
2. 复合指标噪声：用容差 + 中位数而非瞬时值；`is_compounding` 只判"不退化"，不苛求严格上升。仍不达 → 走死锁降级（#2）。
3. Goodhart 残余：代理分 + 采纳回流 + holdout + 随机化对照共同缓解，并由 **RPhysTrack 端到端真实 r_phys 验收**（#6）提供最终地面真值通道——默认 advisory，`harness_rphys_gate` 开启即硬闸。仍残余的：真实 r_phys 也是下游 grader 可 gaming 的标量，根治需能力受限的实验室级验证（见 VerifiableGate 与后续改造）。
4. 递归过深导致开销：A1 明确只一层，strategist 的 meta² 模板写死，绝不叠第三层。
5. **不做**：直接自改源码（A3）、无限递归、让 strategist 自己递归改自己（留作后续轨道，先验证一层）。
6. model 维度不纳入（沿用 H3 P8 限制）。
7. 真 r_phys 端到端验收（`RPhysTrack`，#6）已纳入 A1：把真实 r_phys 归因到配置并以不成对两样本检验给出地面真值验收。仍未做的：compute-heavy 的 ablation 全跑（`HUGINN_META_ABLATION` 之下）+ 能力受限实验室级验证（VerifiableGate），留作后续独立阶段。

## 与现有工作关系

- `MetaImprover`（M-R1）：被扩展为 level-0（improver）与 level-1（strategist）两层，复用其 gate 数学。
- `SignificanceGate` / `OODHoldoutValidator` / `AdoptionGate`：复用，不重写。
- `RevertibleContext`（`huginn/security/revertible.py`）+ `CoEffectRegistry`（`huginn/security/coeffect.py`）：**时空可组合** —— strategist 换件做为可逆补偿效应（时间），strategist/improver/gate/fidelity 依赖图用 provides/requires 声明（空间）。不改 revertible/coeffect 本身，只是接入（同 `PhysicalWorkspace` 立场）。
- `_enabled.py`：`harness_meta_improver` 开启后两层才生效（默认关）。
- ROADMAP 独立轨道；设计文档 `2026-09-13-recursive-compounding-design.md`。