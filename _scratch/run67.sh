#!/usr/bin/env bash
# run67 — 全新长程科研任务: 验证 N_c 判据一致性修复 (zero_violation 原语 + 报告 provenance 规则).
#
# 相对 run66 的差异:
#   - 全新工作目录 run67 (无继承), 干净长程起点;
#   - 脚手架新增 zero_violation 原语 (判据钉死 1e-3, 禁止书生重定义) —— 修复 run66
#     中"报 N_c=8 而留出误差 1.7e-3 > 1e-3"的自相矛盾 (同一命题不同轮判据互斥);
#   - 报告期新增 PROVENANCE RULE: 跨 [ev#] 协议数值不得并成同一可比序列, 未验证的
#     阈值判定不得断言。
#
# 目标: 同 run66 的"小型前馈网络泛化行为作为解空间刚性探针"命题 (纯 ML/数学).
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
# --- Gramian 预条件 + 树搜索 rollout value ---
export HUGINN_MCMC_GRAMIAN=1
export HUGINN_MCMC_GRAMIAN_K=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2

cd /workspace/research_outputs/shusheng_rsi_run67
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1