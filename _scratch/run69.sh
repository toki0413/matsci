#!/usr/bin/env bash
# run68 — 短程对照: 验证 BranchIncubator 树搜索 + step_verifier rollout value 是否真的通电.
#
# 相对 run67 的差异: HUGINN_USE_BRANCH_INCUBATOR=1 (run67 缺此门 → 树 0 次触发).
# v31 追加: 补开 3 个"默认关且此前脚本从未开"的低风险门 (见下, fail-open).
# 短程 (-i 15) 只为对照: 树是否产生 value、winner 是否与 token 贪心不同、停滞是否改善.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=150
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py
# --- 协作 + 盲重建 (v28) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=1
# --- 本次对照的关键开关: 树搜索 ---
export HUGINN_USE_BRANCH_INCUBATOR=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2
# --- Gramian 预条件 + MCMC ---
export HUGINN_MCMC_GRAMIAN=1
export HUGINN_MCMC_GRAMIAN_K=1
# --- v31 补通电: 默认关且此前 run 脚本从未开的门 (低风险, 全 fail-open) ---
export HUGINN_EPISODIC_REPLAY=1              # 情景重放: 按 cue 召回历史情境 advice → decider
export HUGINN_PER_HYP_BUDGET=1               # 单假设盲重建预算上限, 防一个假设独占协作预算
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=1   # 桥A: 高秩 surprise(>0.9) 触发解释差异的新假设
# 注: FAILURE_INVERSION / SKILL_ABSTRACTION / SELF_MODEL / USE_UNIFIED_DECISION /
# COMPLETION_GATE / TRAJECTORY_PATTERN / PMK_INJECT / SKILL_CONTEXT / CROSS_DOMAIN /
# SELF_GOAL_SYNTHESIS 仍保持关: 要么改决策拓扑、要么每轮加 LLM 成本, 属需单独对照的 opt-in.

cd /workspace/research_outputs/shusheng_rsi_run69
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 15 --wall-clock-budget 900 > run.log 2>&1