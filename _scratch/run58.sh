#!/usr/bin/env bash
# run58 launcher — v24 单一完成出口 + 验收门 的野外点亮观测 (长程).
#
# 目的: control_surface_audit.md §6.5 —— run57 (改前代码) 里
#   goal_judge / goal_acceptance / goal_skeptic / goal_metacog_audit 全为 0,
#   本轮验证这四条新 trace 能被点亮, 并采集触发率供 §5.3 第二轮删减。
#
# 与 run57 严格对齐 (可比): 同 objective / 同旋钮 / 同 3600s 预算 / 同
#   HUGINN_PERSISTENT_GOAL_MODE=1 / 不设 REPEAT_EXEC_HARD_STREAK。
#
# 观测口径 (run.log):
#   grep -o 'control_trace name=[a-z_]*' run.log | sort | uniq -c | sort -rn
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run57 拷贝 (只带演化技能
#   前行, 不带旧循环状态/goals).
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

cd /workspace/research_outputs/shusheng_rsi_run58
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1