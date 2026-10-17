#!/usr/bin/env bash
# run43 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 相对 run42 的两处根因修复 (都在"检测到了但不行动"的执行断点上):
#   1. engine_observe._build_hypothesis_prompt: _speculator_hint 截断
#      [:500] → [-500:]. _speculator_hint 每 run 只 reset 一次, 30+ 处 append,
#      所有纠偏指令 (反例搜索 / [强制重定向]) 都 append 到尾部; 取首部 = 把
#      最新最该执行的纠偏最先丢掉 → 信号生成了却从没进 prompt.
#   2. cognitive_loop._trigger_counterexample_hunt: 把关键 directive 的 append
#      从"方法中段"移到"方法末尾", 避免被随后的 failure traces / verifier
#      weakness 顶出 [-500:] 窗口.
#   另: run42 已加 hint block 提到 body 之后 (高优先级), 本次一并带上.
#   RSI 产物回流: 继承 run42 的 evolved_skills.json 作为 bootstrap.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run43
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600