#!/usr/bin/env bash
# run86 — 真实验: 野外点亮观测面新增 episodic 结构化字段
#   (prompt_len / obj_len / nobj).
#
# 背景: run85 的 episodic 快照里这三项全空 —— run85 (10:20) 早于字段落地
# (cognitive_loop.py 11:57 / engine_act.py 11:44). 本 run 与 run85 配置**完全一致**
# (可比), 唯一差别 = 代码已含新字段 + consume-once 清空.
#
# 与 run85 相比复核的观测面 (硬判据):
#   1) 至少一条 episodic entry 带 `prompt_len` (>0) 与 `obj_len` (>=0);
#      execute 轮的 `nobj` 非 None;
#   2) surprise 分布非恒定 (run85 已 {0.0:3, 0.5:4}, 期望复现);
#   3) replay_audit 不再对"输入冻结/执行输出恒同"静默失明 (有样本即可判);
#   4) 主进程退出后无 `ppid==1` 的 python 孤儿 (进程组回收回归).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负 (与 run83/84/85 一致) ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- D6 (与 run85 一致) ---
export HUGINN_CODELAB_SLICE=1
export HUGINN_CODELAB_SLICE_MIN_S=400

# --- P3.2 进展不变量窗口 (与 run85 一致) ---
export HUGINN_PROGRESS_INVARIANT_WINDOW=3

# --- 协作 (与 run85 一致) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- D-slice (hypothesize 侧, 与 run85 一致) ---
export HUGINN_USE_BRANCH_INCUBATOR=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2

# --- 关掉与本次验证无关的重子系统 ---
export HUGINN_MCMC_GRAMIAN=0
export HUGINN_EPISODIC_REPLAY=0
export HUGINN_PER_HYP_BUDGET=0
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=0
export HUGINN_FEATURE_COMMS_CONTRACT=0
export HUGINN_FEATURE_ASD_STE100=0

cd /workspace/research_outputs/shusheng_dslice_run86
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 6 --wall-clock-budget 700 > run.log 2>&1