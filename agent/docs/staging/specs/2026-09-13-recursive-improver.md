# Recursive (Meta-)Improver — 让「改进器如何改进」也是可改进、可验收的对象

> 状态：proposed。里程碑 M-R1（见 ROADMAP 独立轨道）。前置：harness H1 prompt_patch 已落地、significance/ood/adoption 门控已落地。本 spec 是 audit 后对「真正 RSI 缺口」的最小落地。

## 动机（audit 结论）

审计现状（2026-09-13）确认：
- H1–H4（prompt_patch / workflow bandit / joint_optimizer / phase_spec）**代码已接入生产**，但全部被 `feature_flags.harness_*` 门控且**默认 off**（`_enabled.py`，`feature_flags.py` 无任何 harness_*=true）。
- 即便打开，改进器也是**单层非递归**：`prompt_patch.generate_patch(phase, blocks, r_phys, directive, llm_chat_fn)` 的 improver_prompt 是函数内**硬编码字符串**（[prompt_patch.py:304-320](file:///workspace/agent/huginn/harness/prompt_patch.py#L304-L320)），SPEC 明确「我们不递归（不递归改 patch 生成器），只做单层」，STOP 递归改进器被列为 YAGNI。
- 已具备完整验收地基，可直接复用于递归层：`SignificanceGate`（显著性）、`OODHoldoutValidator`（确定性留出防背题）、`AdoptionGate`（GREEN/YELLOW/RED 三级软门控）——全部按 `config_id` 记录 outcome。

**结论**：真正还缺的 RSI 能力是「改进器自身如何改进」也可被改进、且被 gate 验收（STOP 的退化单步版）。本 spec 只做 H1 single-layer 之上的**一层** meta，不做多层递归。

## contract: MetaImprover

新文件 `huginn/harness/meta_improver.py`：

```
ImproverConfig  dataclass
    config_id: str
    improver_prompt: str          # 会被 generate_patch 用作改进器 prompt
    r_phys_gate: float = 0.7      # 只有 r_phys<=gate 才生成 patch
    min_beta_mean: float = 0.5    # apply 门槛（下游 apply_patches 用它）
    task_route: str = "summarize"
    active: bool = False          # 显式 active 才覆盖硬编码默认
    created_at: float

MetaImprover  (单例, 懒加载, 同 PromptPatchStore 模式)
    champion() -> ImproverConfig                         # active config（无则返回"默认=硬编码"哨兵）
    maybe_propose(llm_chat_fn) -> str|None               # 每 N 次 improve() 触发, 生成候选
    evaluate(candidate_id) -> AdoptionDecision           # 冻结重放集离线打分 → 注册进门控
    maybe_promote(candidate_id) -> bool                  # 仅 GREEN 才提升为 champion
    compounding_trace() -> dict                          # meta 赢率 + 换代历史
```

契约要点：
- `generate_patch` 改进器 prompt **改从 `MetaImprover.champion().improver_prompt` 取**；champion 不存在或 `active=False` 时**回落现有硬编码字符串**——即默认零行为变更。
- 候选变为 champion 仅当 `AdoptionGate.should_adopt(candidate_id)` 为 GREEN（显著 + OOD 未退化 + 样本足够）。不再重复"默认 advisory 全放行"——对改进器自升级这一高风险切换，门控必须是硬闸。
- 门控永不删除数据：候选无论提升与否都保留，champion 换代后旧 config 冻结可回退。

## data shape

- `.huginn/harness/meta_improver/config.json`: `{"active_config_id": str|null, "history": [ids]}`
- `.huginn/harness/meta_improver/candidates/<id>.json`: `ImproverConfig.to_dict()`
- `.huginn/harness/meta_improver/meta_trace.jsonl`: 每行 `{ts, type: propose|evaluate|promote|reject|champion_switch, candidate_id, score, delta, reason}`
- 冻结重放集：从 `_last_hypothesis_blocks`（engine_reflect 已记录）+ 最近该 phase 的 `directive`/`r_phys` 维护一条**环形缓冲 ~K=10**（`MetaImprover._replay`），供离线评估候选，不额外触发真实 rollout。

## 验收（test）

新增 `tests/test_meta_improver.py` + `python -m huginn.harness.meta_improver` selfcheck 对齐现有 harness selfcheck：

1. **toggle off** → `MetaImprover` 不起作用，`generate_patch` 用硬编码 prompt（regression，行为不变）。
2. **候选更好**：冻结重放集上，候选 improver_prompt 产出率/有效性/与 directive 对齐明显优于 champion → 积累样本到 ≥min_samples 且 显著 + OOD 通过 → `maybe_promote` 返回 True，champion 切换。
3. **候选更差**：产出率低于 champion / RED → 不 swap，champion 不变。
4. **背题拦截**：候选在 OOD holdout 上显著退化 → `OODHoldoutValidator` RED → 不 promote。
5. **换代历史**：`meta_trace.jsonl` 追加 propose/evaluate/promote/champion_switch 事件；旧 champion 冻结可回退。

## 接入点

- `generate_patch`（[prompt_patch.py:304](file:///workspace/agent/huginn/harness/prompt_patch.py#L304)）：prompt 源改为 champion 覆盖（默认回落硬编码）；成功产 patch 后计数，每 N（默认 5）次调 `MetaImprover.get_instance().maybe_propose(self._llm_chat)`。
- toggle key：`harness_meta_improver`，经 `_harness_enabled`，**且要求 `harness_prompt_patch` 同时 on** 才生效。
- 复用 `SignificanceGate` / `OODHoldoutValidator` / `AdoptionGate`，不重写数学。

## invariant

- `champion().active=False` 或缺省 → `generate_patch` 行为 = 现有硬编码，零回归。
- 门控只决定"是否全局换用改进器配置"，永不删数据（同 AdoptionGate 原则）。
- meta 层默认 off；非 GREEN 不换 champion。
- `r_phys_gate` 候选扰动限制在 `[0.5, 0.8]`；`min_beta_mean` 固定 0.5 或实测中位数（不开放极端值）。

## failure modes

1. 候选 improver_prompt 是垃圾（JSON/改写失败）→ 静默丢弃，不阻塞主循环。
2. 代理分（冻结重放集的产出率）≠ 真实 r_phys 增益 → 诚实声明：代理分只是**首道闸**；真实增益由下游 patch 的 Beta 接受度兜底，champion 学偏时可手动回退（换代替换冻结保留）。
3. 冻结重放集数据陈旧 → 环形缓冲滚动，旧样本被新样本覆盖。
4. 候选始终不 GREEN → 永不换 champion，improver 保持默认，无副作用。

## deferred / non-goals

- 不做完整多层 STOP（$I_t=I_{t-1}(I_{t-1})$ 叠加）——先做单步 meta 层。
- 不改 meta-improver 自己的 meta prompt（不递归层中加层）。
- 不做改进器 prompt 的权重微调（无 RLHF）。
- 不做真实端到端 r_phys 的昂贵 A/B rollout——以离线代理分 + 下游 Beta 兜底。
- 不纳入 model 维度、不接前端面板（meta_trace.jsonl 足够观测，前端留 H 系列升级）。

## 与现有工作关系

- `prompt_patch.generate_patch`：被 meta 层唯一覆盖的对象（改进器的 prompt/阈值）。
- `adoption_gate` / `significance_gate` / `ood_holdout`：直接复用验收，不重写。
- `_enabled.py`: 注册 `harness_meta_improver` 开关。
- ROADMAP M-R1 追踪。