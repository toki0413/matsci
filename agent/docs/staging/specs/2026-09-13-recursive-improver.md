# Recursive (Meta-)Improver — 让「改进器如何改进」也是可改进、可验收的对象

> 状态：implemented ✅（M-R1 完成 + 真实 r_phys 回填改造）。里程碑 M-R1（见 ROADMAP 独立轨道）。前置：harness H1 prompt_patch 已落地、significance/ood/adoption 门控已落地。本 spec 是 audit 后对「真正 RSI 缺口」的最小落地。
> 验证（2026-09-13）：`meta_improver selfcheck 5/5`、`prompt_patch selfcheck 5/5`（零回归）、`tests/test_meta_improver.py 7 passed`、`autoloop engine / harness 全量 import OK`、note_generation 第 5 次触发自动 promote 闭环跑通。
> 改造（2026-09-23，真实 r_phys 回填）：evaluate 不再用离线格式代理分，改用真实迭代回填的 r_phys。引入 **Canary 交替臂**（`select_generation_arm`）让候选模板以 `_CANARY_P=0.3` 概率接管 patch 生成，从而在真实迭代里产出可归因的 patch，拿到真实 reward。验证：`meta_improver selfcheck 6/6`、`tests/test_meta_improver.py 8 passed`。

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
    active: bool = False          # 显式 active 才覆盖默认模板
    created_at: float

MetaImprover  (单例, 懒加载, 同 PromptPatchStore 模式)
    champion_cfg() -> ImproverConfig|None                # active config（无则 None → 回落默认模板）
    candidate_ids() -> list[str]                         # 全部候选，供回填后逐个 evaluate
    select_generation_arm() -> (arm_id, template)        # 选本轮产 patch 的臂: champion/canary/baseline
    record_patch_arm(patch_id, arm_id) -> None           # patch → 生成臂 记账
    arm_for_patch(patch_id) -> str|None                  # 回填时反查臂
    record_real_outcome(arm_id, task_id, r_phys) -> None # 真实迭代 r_phys 按臂回填 ledger
    maybe_propose(llm_chat_fn) -> str|None               # 每 N 次 patch 生成触发, 生成候选
    evaluate(candidate_id) -> dict                       # 用真实 r_phys 分桶配对 → 注册 sig+OOD
    maybe_promote(candidate_id) -> bool                  # 仅 GREEN 才提升为 champion
    compounding_trace() -> dict                          # meta 赢率 + 换代历史 + real_outcomes_n
```

契约要点：
- `generate_patch` 改进器 prompt **改从 `MetaImprover.select_generation_arm()` 取**（champion 覆盖 / canary 候选 / baseline 默认模板）；未开启或无 champion/候选时回落默认模板——即默认零行为变更。
- **Canary 交替臂**：无 champion 且有候选时，候选以 `_CANARY_P=0.3` 概率接管本轮的 patch 生成（arm=candidate），其余轮走 baseline。这样候选模板能在真实迭代里产 patch，其产出仍需过 Beta 门（`α>=β`）才真正 apply，对 explore 的扰动被 Beta 闸兜住。
- **真实 r_phys 归因**：`engine_reflect._learn` 在每轮把真实 `r_phys` 按「生成本轮 apply 过的 patch 的臂」回填进 ledger（`arm_for_patch` 反查）。`task_id` 用 `_run_id`——同 run 内两臂共享一桶，跨 run 累积 `>=_MIN_SAMPLES(5)` 桶才判定。
- `evaluate` 用真实 r_phys 按**确定性任务分桶配对**（`_bucket_pairs`，同 OODHoldout 分桶思路）给候选 vs baseline 打分，注册进 `SignificanceGate` + `OODHoldoutValidator`；**幂等**（每次重建 derived 视图，避免重复 record 膨胀样本导致假性显著）。
- 候选变为 champion 仅当显著性 + OOD 均通过（GREEN）。样本不足时**不判定、不换件**（宁可保持默认也不靠噪声换件）。
- 门控永不删除数据：候选无论提升与否都保留，champion 换代后旧 config 冻结可回退。

## data shape

- `.huginn/harness/meta_improver/config.json`: `{"active_config_id": str|null, "history": [ids]}`
- `.huginn/harness/meta_improver/candidates/<id>.json`: `ImproverConfig.to_dict()`
- `.huginn/harness/meta_improver/meta_trace.jsonl`: 每行 `{ts, type: propose|evaluate|promote|reject|generation|real_outcome, candidate_id, arm_id, r_phys, reason}`
- `.huginn/harness/meta_improver/arm_ledger.json`: `{patch_id: arm_id}`，patch → 生成臂的记账。
- `.huginn/harness/meta_improver/real_outcomes.json`: 环形缓冲（`_REAL_OUTCOME_MAX=500`），每项 `{arm_id, task_id, r_phys, ts}`，是 evaluate 的**唯一真源**。

## 验收（test）

新增 `tests/test_meta_improver.py` + `python -m huginn.harness.meta_improver` selfcheck 对齐现有 harness selfcheck：

1. **toggle off** → `MetaImprover` 不起作用，`generate_patch` 用默认模板（regression，行为不变）。
2. **canary 选臂 + 归因**：有候选时以 `_CANARY_P` 概率接管，否则 baseline；`record_patch_arm` 后 `arm_for_patch` 可回查。
3. **候选更好（真实 r_phys）**：回填真实 r_phys 到两臂，分桶配对积累到 ≥min_samples 且 显著 + OOD 通过 → `evaluate.green` 为真、`maybe_promote` 返回 True，champion 切换；样本不足时 `insufficient_real_outcomes` 不判定。
4. **候选更差**：真实 r_phys 显著更低 → 不 swap，champion 不变。
5. **背题拦截**：候选在 OOD holdout 上显著退化 → `OODHoldoutValidator` RED → 不 promote。
6. **evaluate 幂等**：重复 evaluate 不膨胀 sig 配对（否则 Wilcoxon 假性显著）。
7. **换代历史**：`meta_trace.jsonl` 追加 propose/evaluate/promote/generation/real_outcome 事件；旧 champion 冻结可回退。

## 接入点

- `generate_patch`（[prompt_patch.py:304](file:///workspace/agent/huginn/harness/prompt_patch.py#L304)）：prompt 源改为 `select_generation_arm()`（默认回落默认模板）；成功产 patch 后记账 patch→臂，每 N（默认 5）次调 `MetaImprover.get_instance().maybe_propose(self._llm_chat)`。
- `engine_reflect._learn`（[engine_reflect.py:2475](file:///workspace/agent/huginn/autoloop/engine_reflect.py#L2475)）：每轮把真实 r_phys 按 `_last_applied_patches` 的生成臂回填 ledger，并对候选跑 `evaluate` + 仅 GREEN `maybe_promote`。
- toggle key：`harness_meta_improver`，经 `_harness_enabled`，**且要求 `harness_prompt_patch` 同时 on** 才生效。
- 复用 `SignificanceGate` / `OODHoldoutValidator`，不重写数学。

## invariant

- champion 缺省 → `generate_patch` 行为 = 默认模板，零回归。
- 门控只决定"是否全局换用改进器配置"，永不删数据（同 AdoptionGate 原则）。
- meta 层默认 off；非 GREEN 不换 champion；样本不足不判定。
- `r_phys_gate` 候选扰动限制在 `[0.5, 0.8]`；`min_beta_mean` 固定 0.5 或实测中位数（不开放极端值）。
- 回填只归因到本轮**真正 apply 过 patch** 的臂（因果链完整），不污染 baseline。

## failure modes

1. 候选 improver_prompt 是垃圾（JSON/改写失败）→ 静默丢弃，不阻塞主循环。
2. **真实 A/B 无法同 task 同时跑两臂** → 用确定性分桶把「同类任务」配对；桶内两臂各有观测才成对，样本不足不判定。
3. canary 探索扰动 explore → 候选产出的 patch 仍需过 Beta 门（`α>=β`）才 apply，扰动被 Beta 闸兜住；explore 端不受影响。
4. `task_id` 粒度过粗（如 `_run_id` 单桶）→ 样本累积慢，判定被推迟；宁可慢也不靠噪声换件。
5. 候选始终不 GREEN → 永不换 champion，improver 保持默认，无副作用。

## deferred / non-goals

- 不做完整多层 STOP（$I_t=I_{t-1}(I_{t-1})$ 叠加）——先做单步 meta 层。
- 不改 meta-improver 自己的 meta prompt（不递归层中加层）。
- 不做改进器 prompt 的权重微调（无 RLHF）。
- 不做同一 task 上两臂并行的昂贵 A/B rollout——以 canary 交替臂 + 分桶配对拿真实 r_phys。
- 不纳入 model 维度、不接前端面板（meta_trace.jsonl 足够观测，前端留 H 系列升级）。

## 与现有工作关系

- `prompt_patch.generate_patch`：被 meta 层唯一覆盖的对象（改进器的 prompt/阈值）。
- `adoption_gate` / `significance_gate` / `ood_holdout`：直接复用验收，不重写。
- `_enabled.py`: 注册 `harness_meta_improver` 开关。
- ROADMAP M-R1 追踪。