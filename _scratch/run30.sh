#!/usr/bin/env bash
# run30 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 轻引导: objective.txt 只给研究问题 + 判别口径; 书生 (InternLM) 自主决策执行.
#
# 相对 run29 的三处修复:
#  1) 环境: 本机 numpy/scipy 曾被清空, 已按 requirements.lock 装回
#     numpy==2.5.2 / scipy==1.18.0 (此前 code_lab 会 ModuleNotFoundError).
#  2) harness: mlp_fit 加输入/输出标准化 (mlp_predict 原样还原), 大尺度目标
#     不再因量纲失衡而爆炸。
#  3) 口径: 判别式改为二值 —— 刚性=有限小 N_c 且不随 w 增长;
#     胖=扫描内 N_c 不可达 (脚手架新增 trend='unreachable')。
#     另新增"尺度纪律": y/yv 必须归一到 O(1), 否则绝对 1e-3 判据失效。
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent

cd /workspace/research_outputs/shusheng_rsi_run30
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24