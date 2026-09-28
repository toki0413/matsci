#!/usr/bin/env bash
# run39 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 相对 run38 的修复 (接线"检测到了但不行动"的执行断点):
#   1. evolution/engine.py get_relevant_skills 的 tuple 排序崩溃 → 修 (run38 已验证 0 崩溃)
#   2. 连续换名归约 → 复用已有反例搜索 (hypothesis_loop._metacog_audit_hypothesis):
#      advisory 仍不阻断当前假设, 但连续 3 轮被判"换名归约"即触发
#      _trigger_counterexample_hunt (设 _force_imaginate + 注入 counterexample hint),
#      逼下一轮 hypothesize 换方向. 此前该 verdict 只写进 _metacog_last_audit 无人消费.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run41
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600