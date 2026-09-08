# Spec: Self-Harness 报告（对齐 Qoder/Better Harness 五维 + 组织层任务实录）

> 状态: **Design + 部分已落地**（`huginn/research/harness.py` 提供 `HarnessReport`）。
> 目标: 不复制 Better Harness 的实现，采纳它的**方法论**——用一个证据绑定的五维评分，
> 把我们的散装机械门禁收敛成"能横跨项目/会话看"的治理面。
> 红线（与 Better Harness 同）：**配置存在 ≠ 能力可用**；每个维度都必须"链到真实任务实录
> 的证据"，evidence 三态 Observed / Unobserved / Missing。

## 1. 五维 ↔ 我们的机制（映射表）
| 维度 | Better Harness 问什么 | 我们的机制 | 证据来源 |
|---|---|---|---|
| Task Understanding | 知道项目/从哪开始/改多大 | `run_research_program(planner)` → ResearchPlan + SubResearch DAG | `out.plan_summary` |
| Controlled Execution | 能起项目/在权限内/不越界 | `security/` 沙箱/权限/revertible + HITL `supervision_log` + `plan.max_parallel` 预算 | `out.supervision_log`、plan budget |
| Change Validation | 每次改都过 lint/测试并复检 | `test_arch_cleanliness` 架构门 + `claim_grounding` + `structural_audit` | `out.verdict`、`out.structural_gate`、`out.report_source` |
| Reliable Delivery | 结果有证据验收/有回滚恢复 | `reconcile`(数值对账) + 工作区广播门 + 真实执行 cache | `out.cache`、`out.workspace_verified` |
| Learning Capture | 下个任务是否受益 | 能力自省 §1–§6 缺口→提案→实现→回滚 + C-Space promote + 蒸馏 | self_audit 提案、`out.structural_gate` 等审计产出 |
维度禁 4 存在性评分: 每格都必须能指向**这条任务实录里真实触发过**的检查。

## 2. evidence 三态语义
- **Observed**: 该检查在本任务实录中真实执行并留下结果（如 `structural_gate` 非空）。
- **Unobserved**: 机制存在（配置/接线在），但本任务没走到（如 planner 未传 → plan_summary 空）。
- **Missing**: 无对应机制，或无从推断 → 诚实标缺失并计入缺口。

> 这逼着打分"跟证据走": 一个 run 没跑 planner，就说 Task Understanding=Unobserved，而不是
> 因为代码里有 `planning.py` 就给它满分。

## 3. 计分（透明、可解释）
- 每维 = 若干检查项；每项结果映射: `passed=1 / unobserved=0.5 / missing=0`（observed 但 failed=0）。
- 维度分 = 该维检查项均值；保留各检查明细，供 reading 用 "这条实录到底触发过哪些门"。
- 报告输出 JSON（含 `task_episode`），可供跨 run 聚合（组织层账本输入）。

## 4. 组织层: Task Episode 统一身份（种子）
- `make_task_episode_id(goal)` 生成稳定 episode id（goal 的规范化 slug + 时间/非突变种子）。
- 输出带 `{task_episode, agent, machine}` 维度，使同一 intent 的实录可在**跨 Agent/项目/机器**
  聚合（呼应 Qoder 1.0 的"跨项目/团队知识"方向；本 spec 只落单机编排器，跨机器账本后续）。
- `HarnessReport` 一旦可灌入同一 task_episode，即成为"治理账本"的一行——属于
  上一轮"任务实录 + 共享治理账本"的第一步。

## 5. 非目标
- 不造一个独立评分器副本；复用 `ResearchOutcome` 已记录的 gate 结果，不重复计算。
- 不自动改仓库；报告只点名断点，改进仍走"人工批准"（对齐 Better Harness no-auto-apply）。

## 6. 里程碑
- M1: `huginn/research/harness.py` 的 `HarnessReport` + `make_task_episode_id`（纯 stdlib）。
- M2: 接入 `run_research_program` 返回值（可选 `out.harness`），demo 可一键出报告。
- M3: 跨 run 按 task_episode 聚合（治理账本行）——组织层。