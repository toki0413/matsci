#!/usr/bin/env bash
# run34 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# run33 之后修掉的 RSI 平台缺陷 (与命题无关):
#   1) evolve_from_rewards 原按"组名"去重 → 首条技能后该组永久冻结, RSI 一次性.
#      改为按"工作流内容签名"(同组工具集)去重: 签名相同→原地刷新(不增殖), 不同→新增.
#   2) 可观测: 进化产物另镜像一份到本轮 workspace/.huginn/, 便于逐轮看产出.
# 平台/脚手架/目标不变; 书生(InternLM)自主决策执行.
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

cd /workspace/research_outputs/shusheng_rsi_run34
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24