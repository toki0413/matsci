#!/usr/bin/env bash
# run62 — 协作可观测性复验 (实验组 A: 盲重建).
#
# 目的: 上一轮 A/B (run59/60/61) 的 run.log 里看不到任何 `collab_*` trace ——
#   因为那批 run 跑的是**加 trace 之前**的代码. 本次用同一 objective / 同旋钮 /
#   同 3600s 预算, 唯一目的是让新加的协作跳过路径 trace (collab_blind_reconstruct
#   / collab_branch_incubator) 与新 code_lab_timeout trace 在野外点亮.
#
# 与 run59 严格对齐, 唯一差异: 代码已含控制面观测 (OTel + run.log WARNING).
# 继承: .huginn/evolved_skills.json 自 run57 拷贝 (与 run59/60 同一起点).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py
# --- 实验组 A: 协作开 + 盲重建 ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=1

cd /workspace/research_outputs/shusheng_rsi_run62
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1