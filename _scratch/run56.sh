#!/usr/bin/env bash
# run56 launcher — 控制面触发率观测 (长程, 书生自主推进, 人只监控).
#
# 目的: 审计动作第 5 步 —— 跑一轮长程, 采 `campaign.control_trace` /
#   run.log 的 `control_trace name=...` 触发率, 据此做第二轮删减
#   (B4 effort_floor / B7 curiosity_hint / A8 failure_budget 等长期 0 触发
#   或长期误杀者 → 删或降).
#
# 与 run53/54 对齐 (可比): 默认旋钮, 不复用 run55 的 REPEAT_EXEC_HARD_STREAK=1
#   (那是为逼出 pivot 正分支做的专项压测, 会污染"机制自然触发率"口径).
#
# 观测口径 (run.log):
#   grep -o 'control_trace name=[a-z_]*' run.log | sort | uniq -c | sort -rn
#   关注: exec_convergence / rename_debt / darwin_stagnation / belief_convergence /
#         decider_stop / execute_budget_gate / failure_budget / effort_floor /
#         curiosity_hint / pivot_directive
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run55 拷贝 (与既有约定一致:
#   只带演化技能前行, 不带旧循环状态/goals).
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

cd /workspace/research_outputs/shusheng_rsi_run56
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1