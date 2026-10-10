#!/usr/bin/env bash
# run85 — 真实验: 验证"超时孤儿泄漏"修复 (进程组整组回收).
#
# 与 run84 配置完全一致 (可比), 唯一差别 = 进程组回收已进入代码:
#   huginn/utils/process.py           新增 new_group_popen_kwargs / kill_process_group
#   huginn/tools/persistent_terminal   session 子进程自立新组, kill 整组回收
#   huginn/security/sandbox.py         _kill_process_tree 整组回收
#   huginn/security/compute_adapter    run_job Popen + 超时整组回收
#
# run83/84 病征: 退出后遗留 PPID=1 的 `python experiment.py` 孤儿 (烧 CPU 1h+).
#
# 观察点 (硬判据):
#   1) 主进程退出后, `ppid==1` 下不再有 `python* experiment` 孤儿;
#   2) execute 阶段仍能跑起来 (D7 未被回退);
#   3) 可选: control_trace name=code_lab_slice_skip (D6).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent

# --- 减负 (与 run83/84 一致) ---
export HUGINN_MAX_TOOL_CALLS=4
export HUGINN_MAX_TOOL_CALLS_PER_TOOL=1
export HUGINN_CODELAB_TIMEOUT_S=60
export HUGINN_FALLBACK_STREAM_IDLE=30
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

# --- D6 目标: 逼出 code_lab_slice_skip ---
export HUGINN_CODELAB_SLICE=1
export HUGINN_CODELAB_SLICE_MIN_S=400

# --- P3.2 目标: 缩短进展不变量窗口 ---
export HUGINN_PROGRESS_INVARIANT_WINDOW=3

# --- 协作 (与 run83/84 一致) ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=auto
export HUGINN_STRENGTH_SCHEDULE=1

# --- D-slice (hypothesize 侧, 与 run83/84 一致) ---
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

cd /workspace/research_outputs/shusheng_dslice_run85
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 6 --wall-clock-budget 700 > run.log 2>&1