#!/usr/bin/env bash
# run84 — 真实验: 在 run83 同一命题/配置下, 验证 D7 (实时预算封顶) + 进程组回收
# 两个修复是否奏效.
#
# 与 run83 的唯一差别 = 修复已进入代码 (配置完全一致, 保证可比):
#   run83 病征: hypothesize 566.2s/700s (81%), execute 从未跑; 4 次
#   `fallback stream collect timed out (idle=30s, total=300s)`; 退出后遗留 2 个
#   `python3 experiment.py` 孤儿 (PPID→1, 烧 1h35m CPU).
#
# 观察点:
#   1) 不再出现 `total=300s` 的固定总超时 — 降级收集被实时预算封顶 (D7);
#   2) execute 阶段能跑起来 (不被 hypothesize 饿死);
#   3) 进程退出后 `ppid==1` 下无 `python3 experiment.py` 孤儿 (进程组回收);
#   4) 可选: control_trace name=code_lab_slice_skip (D6).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负 (与 run83 一致) ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- D6 目标: 抬高修复重写 slice 的可负担门槛, 逼出 code_lab_slice_skip ---
export HUGINN_CODELAB_SLICE=1
export HUGINN_CODELAB_SLICE_MIN_S=400

# --- P3.2 目标: 缩短进展不变量窗口 ---
export HUGINN_PROGRESS_INVARIANT_WINDOW=3

# --- 协作 (与 run83 一致) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- D-slice (hypothesize 侧, 与 run83 一致) ---
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

cd /workspace/research_outputs/shusheng_dslice_run84
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 6 --wall-clock-budget 700 > run.log 2>&1