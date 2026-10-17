#!/usr/bin/env bash
# run70 — 验证 system 消息归一修复 (internlm "in prompt processing error"):
# run69 里 branch_incubator=empty ok=0、blind_reconstruct summary_len=26 (=错误串),
# 本轮应看到 branch ok>0 且 blind_reconstruct 出现真实 evidence (非 26 字错误串).
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
# --- 协作 + 盲重建 ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=1
# --- 树搜索 ---
export HUGINN_USE_BRANCH_INCUBATOR=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2
# --- Gramian 预条件 + MCMC ---
export HUGINN_MCMC_GRAMIAN=1
export HUGINN_MCMC_GRAMIAN_K=1
# --- v31 补通电 ---
export HUGINN_EPISODIC_REPLAY=1
export HUGINN_PER_HYP_BUDGET=1
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=1

cd /workspace/research_outputs/shusheng_rsi_run70
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 25 --wall-clock-budget 1500 > run.log 2>&1