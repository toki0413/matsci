#!/usr/bin/env bash
# run28 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 轻引导: objective.txt 只给研究问题 + 验收口径; 书生 (InternLM) 自主决策执行.
#
# 相对 run27 的修复:
#  1) code_lab 新增 _check_label_shapes 硬校验: y/yv 必须是 (N,1)。
#     run27 的 fat 臂犯了广播 bug (X[:,0] + standard_normal((w,1)) -> (w,w)),
#     留出误差被钉在 ~0.51 的假值却不报错, 污染整张 N_c 表 → 现已硬失败,
#     由修复循环把精确形状问题回灌给书生。
#  2) objective 新增"run27 伪结果诊断"节: 要求 rigid/fat 是真正互斥的解析族
#     (fat 必须在扫描上限内留出点不可达), 且报告不得把与数值矛盾的趋势写成结论。
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent

cd /workspace/research_outputs/shusheng_rsi_run28
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24