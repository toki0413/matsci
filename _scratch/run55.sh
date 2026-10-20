#!/usr/bin/env bash
# run55 launcher — 野外观测 pivot 硬约束 (长程, 书生自主推进, 人只监控).
#
# 目的: run53/run54 均跑满挂钟预算, 但日志里 **0 次** "repeat execution" —— 书生
#   每轮都换了实验族/参数, pivot 硬约束的**正分支**(检测到重复→注入强制变异令)
#   在野外从未被点亮. 本轮专门提高正分支被观测到的概率:
#
#   ① HUGINN_REPEAT_EXEC_HARD_STREAK=1 (默认 2): 只要出现**一次**同指纹重跑,
#      即置 _force_exec_variation, 把强制变异令注入 code_lab 作者提示. 这是机制
#      自带的配置旋钮, 不改代码; 默认 2 的语义在 config 里保留.
#   ② 挂钟预算 3600s / 迭代上限 60, 与 run53/54 对齐 (可比).
#
# 观测口径:
#   ① 日志出现 "repeat execution detected (streak=1): force experiment variation";
#   ② 下一轮 code_lab 作者提示里带 "【强制变异·硬约束】" 且含上一轮真实结果;
#   ③ 下一轮实验代码的**结构指纹**相对重复轮改变 (AST 归一后不同);
#   ④ run 仍跑到挂钟/目标, 早停只应来自目标达成.
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run54 拷贝 (与
#   run52→53/54 同一约定: 只带演化技能前行, 不带旧循环状态/goals).
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export HUGINN_REPEAT_EXEC_HARD_STREAK=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run55
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1