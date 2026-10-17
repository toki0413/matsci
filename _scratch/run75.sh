#!/usr/bin/env bash
# run75 — 与 run73 同命题、同环境, 唯一变量: autoloop CLI 现在会加载 Star 插件.
#
# 验证目标 (b): autoloop 命令入口挂载 Star 插件后, asd_ste100 的 STE prompt 段
# 与 ste_lint/comms_lint 工具在**长程 run 里真实生效**; comms_contract 事件审计
# 也在同一进程内运行. run73 是插件缺席的对照 (log 里没有 [plugins] / mounted).
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
# --- 通信规范化层 ---
export HUGINN_FEATURE_COMMS_CONTRACT=1
export HUGINN_FEATURE_ASD_STE100=1

cd /workspace/research_outputs/shusheng_rsi_run75
exec python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 1 --wall-clock-budget 240 > run.log 2>&1