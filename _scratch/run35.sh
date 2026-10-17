#!/usr/bin/env bash
# run35 launcher — 长程探索模式 (long-horizon), Huginn-native autoloop.
#
# 与 run34 的唯一差别: 打开长程模式.
#   --wall-clock-budget 3600  → CLI 创建 active goal + 开启
#   HUGINN_PERSISTENT_GOAL_MODE=1, 于是 darwin 停滞 / 信念收敛 / surprise
#   收敛这类"启发式早停"在挂钟预算未耗尽时不再终止整个 run; 循环自主推进
#   到目标达成 (裁判 F2/F17/arbiter) 或预算/迭代上限耗尽.
# RSI 修复(run34 之后): evolve 触发词改为执行内容派生 + 产物镜像到本轮 .huginn.
# 平台/脚手架/目标/书生(InternLM) 自主决策 均不变.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
# 任务脚手架 = 命题资产(非平台内核): NN 容量扫描的模板/提示/原语都在这里。
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run35
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600