#!/usr/bin/env bash
# run33 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# run32 之后修掉的判官链路 bug (全部在平台侧, 与命题无关):
#   1) cognitive_loop 把报告**文件路径**当 final_output → 改读正文;
#   2) 报告开头 autoloop 头部(~3.5k字)挤爆判官窗口 → 只取 "## Research Report" 起;
#   3) GoalJudge 截断 4000→12000; JSON 解析加围栏/前后文剥离, 避免掉进恒 false 的
#      关键字兜底. 平台/脚手架/目标不变; 书生(InternLM)自主决策执行.
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

cd /workspace/research_outputs/shusheng_rsi_run33
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24