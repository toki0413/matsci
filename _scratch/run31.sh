#!/usr/bin/env bash
# run31 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 与 run30 的差别: 架构纠偏后, NN 脚手架已从平台内核外置为任务资产
# (examples/codelab_scaffolds/network_rigidity.py), 由 HUGINN_CODELAB_SCAFFOLD 声明;
# 平台 code_lab 只保留命题无关的沙箱与注入点。轻引导: objective.txt 只给研究问题 +
# 判别口径; 书生 (InternLM) 自主决策执行。
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

cd /workspace/research_outputs/shusheng_rsi_run31
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24