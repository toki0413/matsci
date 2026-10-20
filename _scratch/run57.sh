#!/usr/bin/env bash
# run57 launcher — 控制面触发率观测 (长程, 书生自主推进, 人只监控).
#
# 目的: §5.3 —— 再跑同类观测 run 凑样本 (run56 之后第 2 轮), 除既有
#   control_trace 口径外, 本轮新增观测 `report_citation` (报告 citation 门 C2:
#   Results 未溯源数值数/总数).
#
# 与 run56 严格对齐 (可比): 同 objective / 同旋钮 / 同 3600s 预算 / 不设
#   REPEAT_EXEC_HARD_STREAK (那是专项压测, 会污染"自然触发率"口径).
#
# 观测口径 (run.log):
#   grep -o 'control_trace name=[a-z_]*' run.log | sort | uniq -c | sort -rn
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run56 拷贝 (只带演化技能
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

cd /workspace/research_outputs/shusheng_rsi_run57
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1