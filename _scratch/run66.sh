#!/usr/bin/env bash
# run66 — 全新长程科研任务 (v28 修复 + Gramian 预条件 + branch rollout value 全量启用).
#
# 相对 run65 的差异:
#   - 全新工作目录 run66 (无继承), 干净长程起点;
#   - 启用 Gramian 谱预条件 (HUGINN_MCMC_GRAMIAN=1, 沿高可控/高信息主轴提议);
#   - 启用 BranchIncubator 的 step_verifier rollout value 剪枝 (HUGINN_BRANCH_VALUE_PRM=1);
#   - 协作开 + 盲重建 (v28: 只认带非空 messages 的真实状态, 消除 summary_len=0).
#
# 目标: 同 run65 的"小型前馈网络泛化行为作为解空间刚性探针"命题 (纯 ML/数学).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py
# --- 协作 + 盲重建 (v28) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=1
# --- 新增机制: Gramian 预条件 + 树搜索 rollout value ---
export HUGINN_MCMC_GRAMIAN=1
export HUGINN_MCMC_GRAMIAN_K=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2

cd /workspace/research_outputs/shusheng_rsi_run66
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1