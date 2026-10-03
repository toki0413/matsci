#!/usr/bin/env bash
# run26 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 轻引导: objective.txt 只给研究问题 + 验收口径; 书生 (InternLM) 自主决策执行.
# 规则版定序器 (LLM decider off): hypothesize→plan→execute→validate→learn→循环.
#
# 与 run25 的差异 (两处修复, 针对 run25 的两个根因):
#  1) 去掉 `-s tests_passed`. run25 里 Code Lab 成功即 tests_passed=True,
#     check_completion 立刻判"目标完成"→ 循环只跑 2 个 execute 周期就停,
#     learn 只跑 1 次 → 奖励记录 <2 条 → RSI (evolve_from_rewards) 不产出.
#     去掉完成标准后循环按 5 步序列持续 cycle, 同一 run 内积累多条 learn 奖励.
#  2) code_lab 的 capacity_scan 新增"留出集有效性守卫": Xv 与 X 重合时判
#     trend='invalid_heldout' 并记 warnings —— run25 伪结果 (heldout==train) 的根治.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent

cd /workspace/research_outputs/shusheng_rsi_run26
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24