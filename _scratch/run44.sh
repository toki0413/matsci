#!/usr/bin/env bash
# run44 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 相对 run43 的根因修复 (平台不得绑定具体命题):
#   engine_observe._build_hypothesis_prompt 的 body 块原先硬编码
#   "You are an autonomous material science research agent" + 材料维度表
#   (composition/temperature/defect/structure/transport) + "优先写成 PDE/
#   变分/守恒律". 结果: 任何非材料命题 (纯数学/机器学习) 都被拽进材料/PDE
#   语言 → run43 里 31 条假设全落在材料 5 维度, plan_check 连续判 misalign,
#   循环空转. 现改为领域无关: 维度由命题自身推导, 结构用命题自然语言描述.
# 继承 RSI 产物: 复用 run43 的 evolved_skills.json 作为 bootstrap.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run44
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600