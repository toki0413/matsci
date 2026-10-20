#!/usr/bin/env bash
# run27 launcher — Huginn-native autoloop for the NN-generalization rigidity probe.
# 轻引导: objective.txt 只给研究问题 + 验收口径; 书生 (InternLM) 自主决策执行.
#
# 相对 run26 的两处修复:
#  1) G2 周期重定向: 规则版定序器本为确定性周期序列, 却被 _check_stuck 当"卡住"
#     强制 pivot, 触发 CognitiveLoop "no hyp to pivot from" → 2 个 cycle 就早停.
#     修复: LLM 决策器关闭时跳过周期重定向 (engine_reflect._check_stuck).
#  2) RSI 分组键: _learn 里 reward 记账用稳定的 calculation_type='autoloop'
#     / software='huginn', 不再用漂移的 plan['mode'] —— 否则同 run 的记录被切进
#     单条组, evolve_from_rewards 的 ≥2 门槛永远达不到, RSI 不产出.
# 去掉 `-s tests_passed`: 否则 Code Lab 成功即判目标完成, 首轮 validate 后即停.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent

cd /workspace/research_outputs/shusheng_rsi_run27
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" -i 24