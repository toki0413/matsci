#!/usr/bin/env bash
# run73 — 测试新增的「通信规范化」层:
#   1) comms_contract 事件总线审计 (每个 publish 做信封/词表/载荷/别名/结果自洽检查,
#      只报告不拦截; 违规按 rule 累积 + 首次 hard 告警)
#   2) asd_ste100 受控语言 (prompt 段注入 + ste_lint/comms_lint 工具)
# 与 run72 同命题, 便于对照: 只换"通信层", 不改研究目标.
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
# --- 协作 + 盲重建 (让 agent 间消息 / 子 agent 结果面被真正走过) ---
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
# --- 新: 通信规范化层 (显式置位, 便于日志取证) ---
export HUGINN_FEATURE_COMMS_CONTRACT=1
export HUGINN_FEATURE_ASD_STE100=1

cd /workspace/research_outputs/shusheng_rsi_run73
exec python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 3 --wall-clock-budget 420 > run.log 2>&1