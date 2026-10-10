#!/usr/bin/env bash
# run79 — 验证「受控独立观察者」+「假设图评分修复 A/B/C/D」.
#
# 背景:
#   run78 (精简版) 观测到 darwin 评分恒定 2.50 —— supported/testable/topology
#   三维被结构性钉死为 0. 已修:
#     A 存 testable_prediction   B 回写节点状态   C 建派生边   D 并入任务性能
#   run78 同时关掉了协作/观察者, 无法验证「受控独立观察者」.
#
# 本轮验证:
#   1) darwin 评分是否脱离 2.50 常量 (A/B/C/D 是否真的通电);
#   2) control_trace 是否记录 collab_blind_reconstruct 的 holds/mismatch 差分;
#   3) 观察者分歧 → hypothesis_strength 是否随之下降 (转探索).
#
# 减负: 关掉与本次验证无关的重活 (树搜索/Gramian/重放/通信层), 保留
#       协作 + 观察者(auto) + 认知环 + darwin ratchet + 强度调度.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负: 单轮更短, 让轮次跑得动 ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- 本次验证: 协作 + 受控独立观察者(auto 档位门控) + 强度调度 ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- 关掉与本次验证无关的重子系统 ---
export HUGINN_USE_BRANCH_INCUBATOR=0
export HUGINN_BRANCH_VALUE_PRM=0
export HUGINN_MCMC_GRAMIAN=0
export HUGINN_EPISODIC_REPLAY=0
export HUGINN_PER_HYP_BUDGET=0
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=0
export HUGINN_FEATURE_COMMS_CONTRACT=0
export HUGINN_FEATURE_ASD_STE100=0

cd /workspace/research_outputs/shusheng_rsi_run79
exec python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 20 --wall-clock-budget 3000 > run.log 2>&1