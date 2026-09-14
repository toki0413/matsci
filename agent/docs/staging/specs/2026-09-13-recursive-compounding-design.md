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
    record(epoch, config_id, champion_quality_mean, proposals_to_promotion,
           generations_to_promotion, win_rate) -> None
    is_compounding() -> bool          # 质量趋势↑/持平(容差内) 且 每次换件成本不持续升
    would_degrade(new_cid, incumbent_cid) -> bool
    stats() -> dict                   # {slope, cost_trend, window, best_quality, ongoing}

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
- `maybe_promote_strategist` 门控 = `SignificanceGate` GREEN **且** `OODHoldoutValidator` 通过 **且** `CompoundingTracker.would_degrade(new, incumbent)` 为 False。
- 复合退化即 `revert_strategist` 回上一 champion（元层级 darwin-ratchet），发散防死。

## data shape

- `.huginn/harness/meta_improver/strategist/<id>.json`: `StrategistConfig.to_dict()` + `config.json` 记录 `{active_strategist_id, strategies_history}`。
- `.huginn/harness/meta_improver/compounding.json`: `{window: [...], slope, cost_trend, best_quality}`。
- `meta_trace.jsonl` 追加事件类型 `strategy_propose / strategy_evaluate / strategy_promote / strategy_revert`。
- 注意：strategist 与 improver 两组 champion 独立持久化，互不覆盖。

## 验收（test）

新增到 `tests/test_meta_improver.py` + `python -m huginn.harness.meta_improver` selfcheck 扩展：

1. **零回归**：无 strategist champion → `maybe_propose` 回落默认模板。
2. **好 strategist**（候选在重放集上提升改进器产出 rate 且 GREEN）→ `maybe_promote_strategist` True，champion 切换。
3. **差/退化 strategist** → `would_degrade` True → 不换件。
4. **发散回退**：切换后复合下滑 → `revert_strategist` 回到上一 champion。
5. **CompoundingTracker 数学**：斜率/效率/窗口/`is_compounding` 边界。

## failure modes / deferred

1. 二次滚动的样本稀释：strategist 层也要攒够 sig+OOD 样本 → 设定 `min_samples` 与窗口 W 平衡，样本不足即 YELLOW 不换件（不影响 level-0）。
2. 复合指标噪声：用容差 + 中位数而非瞬时值；`is_compounding` 只判"不退化"，不苛求严格上升。
3. 递归过深导致开销：A1 明确只一层，strategist 的 meta² 模板写死，绝不叠第三层。
4. **不做**：直接自改源码（A3）、无限递归、让 strategist 自己递归改自己（留作后续轨道，先验证一层）。
5. model 维度不纳入（沿用 H3 P8 限制）。

## 与现有工作关系

- `MetaImprover`（M-R1）：被扩展为 level-0（improver）与 level-1（strategist）两层，复用其 gate 数学。
- `SignificanceGate` / `OODHoldoutValidator` / `AdoptionGate`：复用，不重写。
- `_enabled.py`：`harness_meta_improver` 开启后两层才生效（默认关）。
- ROADMAP 独立轨道；设计文档 `2026-09-13-recursive-compounding-design.md`。