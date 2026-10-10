#!/usr/bin/env bash
# run61 — A/B baseline (单 agent). run58 被沙箱重启打断, 这里重跑一份完整 baseline.
#
# 与 run59/run60 严格对齐: 同 objective / 同旋钮 / 同 3600s / 同继承起点.
# 唯一差异: **不设** HUGINN_ENABLE_AGENT_COLLAB → _agent_factory=None, 纯单 agent.
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
# --- baseline: 协作关 (默认) ---

cd /workspace/research_outputs/shusheng_rsi_run61
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1