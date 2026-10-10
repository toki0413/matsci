#!/usr/bin/env bash
# run87 — 真实验: 验证 D7 续 (流式涓流总时长封顶) 是否止住预算压穿.
#
# 背景: run86 (与 run85 同配置) 预算 700s, 实跑 22min+ (≈1300s) 仍未触发挂钟硬停,
# 人工中止 —— 根因是主流只有**空闲**封顶, 一条慢而不死的 token 涓流每 chunk 重置
# idle 计时, 永不触发 (design doc §7.4). 修复: `_astream_with_watchdog` 新增
# `total_timeout` (绝对 deadline, 到点即抛), 主流挂实时剩余预算.
#
# run87 与 run86 配置**完全一致** (可比), 唯一差别 = 代码已含 D7 续涓流封顶.
#
# 复核的硬判据:
#   1) 总挂钟不再压穿预算: 主进程在 ≤ 预算+一个在飞动作 内退出 (run86 为 1300s+);
#   2) execute 阶段可达 (不再被 hypothesize 整段饿死), code_lab_slice_skip 有野外样本;
#   3) 主进程退出后无 `ppid==1` 的 python 孤儿 (进程组回收回归);
#   4) run.log 出现 `stream total timeout` (涓流被总时长截断的直接证据).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负 (与 run83/84/85/86 一致) ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- D6 (与 run86 一致) ---
export HUGINN_CODELAB_SLICE=1
export HUGINN_CODELAB_SLICE_MIN_S=400

# --- P3.2 进展不变量窗口 (与 run86 一致) ---
export HUGINN_PROGRESS_INVARIANT_WINDOW=3

# --- 协作 (与 run86 一致) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- D-slice (hypothesize 侧, 与 run86 一致) ---
export HUGINN_USE_BRANCH_INCUBATOR=1
export HUGINN_BRANCH_VALUE_PRM=1
export HUGINN_BRANCH_INCUBATOR_DEPTH=2

# --- 关掉与本次验证无关的重子系统 (与 run86 一致) ---
export HUGINN_MCMC_GRAMIAN=0
export HUGINN_EPISODIC_REPLAY=0
export HUGINN_PER_HYP_BUDGET=0
export HUGINN_ALIGNMENT_SURPRISE_TRIGGER=0
export HUGINN_FEATURE_COMMS_CONTRACT=0
export HUGINN_FEATURE_ASD_STE100=0

cd /workspace/research_outputs/shusheng_dslice_run87
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 6 --wall-clock-budget 700 > run.log 2>&1