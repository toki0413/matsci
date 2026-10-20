#!/usr/bin/env bash
# run51 launcher — 长程确认 (long-horizon), 书生自主推进, 人只监控.
#
# 目的: 在源头A (输入冻结) + 源头B (脚手架彻底解绑) 均已落地后, 用**更长程**
#   配置复核"空转已消除": run50 的离线审计已证 prompt_len 不再冻结、nobj 不再
#   恒定、换名归约闭环消失、并主动经 exec convergence → conclude+stop 终止;
#   run51 把迭代上限与墙钟预算抬高, 检验循环在更长时间尺度上仍**有进展或按出口
#   停止**, 而不是重新退化成"跑一次修一次"。
#
# 继承 (RSI bootstrap): .huginn/evolved_skills.json 自 run50 拷贝 (与 run49→run50
#   同一约定: 只带演化技能前行, 不带旧循环状态)。
#
# 保留: v11 进展不变量 (单调换名债务 _rename_debt → 无进展轮必被逼到终止出口)、
#   执行收敛指纹硬终止、提交闸 (harness 未过测试不进提交)。
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

cd /workspace/research_outputs/shusheng_rsi_run51
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 7200 > run.log 2>&1