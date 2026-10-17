#!/usr/bin/env bash
# run52 launcher — 长程确认 (long-horizon), 书生自主推进, 人只监控.
#
# 目的: 验证 v12 "pivot 硬约束". run50(B) 实测: repeat-execution 的软提示只进
#   _speculator_hint(假设生成提示), 而真正写实验的是 code_lab 作者提示
#   (build_author_prompt) —— 它不读该提示, 于是 streak 1-4 指纹恒同, 循环撞
#   6 窗口 exec convergence 提前离场 (1409.7s/3600s, 39% 预算). 本轮相对 run50
#   只改一处: streak>=HUGINN_REPEAT_EXEC_HARD_STREAK 即置 _force_exec_variation,
#   由 engine_act._build_codelab_focus 把"强制变异"令直接注入作者提示, 并回灌上
#   一轮真实结果 → 逼出不同的实验族/参数, 指纹应随之改变.
#
# 观测口径: ① 是否仍撞 exec convergence 早停; ② repeat streak 越阈值后下一轮
#   指纹是否改变/实验族是否更换; ③ 换名债务是否累积到触发重定向.
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run51 拷贝 (与
#   run49→run50/run51 同一约定: 只带演化技能前行, 不带旧循环状态/goals).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export HUGINN_REPEAT_EXEC_HARD_STREAK=2
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run52
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1