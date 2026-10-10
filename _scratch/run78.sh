#!/usr/bin/env bash
# run78 — D3 触发率观测 run (精简资源版, 同命题同 endpoint).
#
# 目的: 单轮在本 endpoint 上 >15min, 长程 20 轮不可达 → 砍资源让轮次跑起来,
# 观测 D3 stall→action 的触发率与强制 pivot 后的行为.
# D3 开 (HUGINN_STALL_AS_ACTION=1), stagnation 阈值固定为 2 以便在有限窗口内可观测.
#
# 砍掉的都是与 decide_fn/ratchet 控制流无关的重活 (协作/树搜索/Gramian/重放),
# 唯一保留的控制面是 认知环 + darwin ratchet + D3.
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
export HUGINN_CODELAB_TIMEOUT_S=20
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- 关掉与 D3 无关的重子系统 ---
export HUGINN_ENABLE_AGENT_COLLAB=0
export HUGINN_BLIND_RECONSTRUCTION=0
export HUGINN_USE_BRANCH_INCUBATOR=0
export HUGINN_BRANCH_VALUE_PRM=0
export HUGINN_MCMC_GRAMIAN=0
export HUGINN_EPISODIC_REPLAY=0
export HUGINN_PER_HYP_BUDGET=0
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=0
export HUGINN_FEATURE_COMMS_CONTRACT=0
export HUGINN_FEATURE_ASD_STE100=0

# --- D3 观测 ---
export HUGINN_STRENGTH_SCHEDULE=0
export HUGINN_DARWIN_STAGNATION_LIMIT=2
export HUGINN_STALL_AS_ACTION=1

cd /workspace/research_outputs/shusheng_rsi_run78
exec python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 40 --wall-clock-budget 3000 > run.log 2>&1