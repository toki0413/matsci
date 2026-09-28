#!/usr/bin/env bash
# run32 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 与 run31 的差别: 修掉收尾判官的 bug —— cognitive_loop 之前把报告**文件路径**当
# final_output 喂给 GoalJudge, 判官只见路径不见正文, 于是 run31 明明产出了真实
# N_c 表(dataset.jsonld 里 36 条 neg_heldout_* 证据)却被判"无数值证据". 现在改为
# 读入报告正文再判. 平台/脚手架/目标均不变(架构纠偏后命题已外置).
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

cd /workspace/research_outputs/shusheng_rsi_run32
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24