#!/usr/bin/env bash
# run81 — 真实验: 验证 D6 (execute 侧 code_lab 切片) 的 `code_lab_slice_skip`
# 控制面 trace 能被点亮, 并给 P3.2 `progress_invariant` 一次机会.
#
# 与 run80 的差别只在"逼出 D6 门":
#   run80 里 execute 是经 `_budget_exhausted()` (硬 0 门) 收尾的, 没走到 D6② 的
#   "修复重写可负担性"门 → code_lab_slice_skip 未发. 本轮把
#   HUGINN_CODELAB_SLICE_MIN_S 抬高 (自校准门槛的下限), 使 attempt0 跑完后剩余
#   预算 < 下限 → 修复重写 slice 被判不可负担 → 发 code_lab_slice_skip.
#
# 观察点:
#   1) run.log 出现 control_trace name=code_lab_slice_skip (D6 点亮);
#   2) replay_audit 的 exits.control_traces 能计到该名 (正确计数);
#   3) 可选: control_trace name=progress_invariant (P3.2).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负 (与 run80 一致) ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- D6 目标: 抬高修复重写 slice 的可负担门槛, 逼出 code_lab_slice_skip ---
export HUGINN_CODELAB_SLICE=1
export HUGINN_CODELAB_SLICE_MIN_S=250

# --- P3.2 目标: 缩短进展不变量窗口, 提高被点亮概率 ---
export HUGINN_PROGRESS_INVARIANT_WINDOW=3

# --- 协作 (与 run80 一致) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- D-slice (hypothesize 侧, 与 run80 一致) ---
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

cd /workspace/research_outputs/shusheng_dslice_run82
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 6 --wall-clock-budget 700 > run.log 2>&1