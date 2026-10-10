#!/usr/bin/env bash
# run80 — 真实验: 书生(internlm) + 复刻 run72/73 的 420s 预算, 验证 D-slice
# 是否把 BranchIncubator 单阶段超支堵住.
#
# 背景: run72/73 里 depth=2 的 9 个子智能体 (3 layer1 + 6 layer2) 各触发 ~300s
# 流式超时, 单 hypothesize 阶段累计 887s, 远超 420s 挂钟预算. D-slice 把一轮拆
# 成 S1..S5, 每个可选 slice 启动前查剩余预算, 不足则用已产出的 layer1 收尾.
#
# 本轮观察点:
#   1) run.log 是否出现 `skip slice=layer2 (return layer1)` / branch_slice_skip;
#   2) 单次 hypothesize 阶段耗时是否被限在"一个 layer1 slice"量级, 不再 887s;
#   3) 总耗时是否收敛到 预算 + 至多一个在飞 slice.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负: 让轮次跑得动 (与 run79 一致) ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- 协作 ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- D-slice 目标: 树搜索 (depth=2) + PRM rollout value ---
export HUGINN_USE_BRANCH_INCUBATOR=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2

# --- 关掉与本次验证无关的重子系统 (减少混淆) ---
export HUGINN_MCMC_GRAMIAN=0
export HUGINN_EPISODIC_REPLAY=0
export HUGINN_PER_HYP_BUDGET=0
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=0
export HUGINN_FEATURE_COMMS_CONTRACT=0
export HUGINN_FEATURE_ASD_STE100=0

cd /workspace/research_outputs/shusheng_dslice_run80
exec /workspace/agent/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 6 --wall-clock-budget 420 > run.log 2>&1