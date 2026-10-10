#!/usr/bin/env bash
# run60 — A/B 实验组 B: BranchIncubator (N=3 隔离假设采样, 提多样性).
#
# 与 run58 严格对齐 (baseline): 同 objective / 同旋钮 / 同 3600s 预算 /
#   同 HUGINN_PERSISTENT_GOAL_MODE=1. 唯一差异:
#     HUGINN_ENABLE_AGENT_COLLAB=1   ← 通电
#     HUGINN_USE_BRANCH_INCUBATOR=1  ← 开 N=3 隔离探索采样
#
# 观测: 真进展率 / 换名打转率 / report_citation 未溯源数 / control_trace 分布.
# 继承: .huginn/evolved_skills.json 自 run57 拷贝 (与 run58 同一起点).
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
# --- A/B 差异 ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_USE_BRANCH_INCUBATOR=1

cd /workspace/research_outputs/shusheng_rsi_run60
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1