#!/usr/bin/env bash
# run68 — 短程对照: 验证 BranchIncubator 树搜索 + step_verifier rollout value 是否真的通电.
#
# 相对 run67 的唯一差异: HUGINN_USE_BRANCH_INCUBATOR=1 (run67 缺此门 → 树 0 次触发).
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

cd /workspace/research_outputs/shusheng_rsi_run69
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 15 --wall-clock-budget 900 > run.log 2>&1