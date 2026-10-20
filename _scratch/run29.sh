#!/usr/bin/env bash
# run29 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 轻引导: objective.txt 只给研究问题 + 验收口径; 书生 (InternLM) 自主决策执行.
#
# 相对 run28 的验收收紧 (只改边界/口径, 不替书生做设计):
#  - fat 族不得是"与 x 无关的 i.i.d. 纯噪声"(否则测不到 N_c 递增趋势, 命题无法验证);
#    要求为 x 的、高维但可部分学习的函数族, 其留出违规来自欠定/欠拟合。
#  - 新增"阈值标定"条目: 若留出误差压在 1e-3 判据附近来回跳, 说明 N_c 由噪底决定,
#    须按 anchor 标定 ho_tol 并交代, 否则抖动不构成趋势证据。
#  - 报告纪律硬项: 结论须被数值支持; train/heldout 两列不得互相复制; 禁材料类比。
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent

cd /workspace/research_outputs/shusheng_rsi_run29
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24