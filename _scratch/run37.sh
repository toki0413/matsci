#!/usr/bin/env bash
# run37 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 平台侧长程能力已就绪(run36 实测 25 轮, wall-clock 收口):
#   - 步数上限按挂钟预算反推 (预算/10s), 同步抬 goal.max_iterations
#   - observe 每步查 wall_clock_expired, 耗尽即落 report
#   - 启发式早停(darwin/belief/surprise) 在挂钟未耗尽时让位 (_long_horizon_keep_going)
#   - RSI 技能触发词改为执行内容派生, 能在 plan 阶段作 "Learned skills" 回流
# 本轮挂钟放宽到 3600s (60min), -i 30 仅作步数下限; 目标达成或挂钟耗尽才收口.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run37
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600