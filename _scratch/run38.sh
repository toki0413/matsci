#!/usr/bin/env bash
# run38 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 相对 run37 的修复:
#   - 修 evolution/engine.py get_relevant_skills 的 tuple 排序崩溃
#     (score 相同时回退比较 SkillTemplate 对象 → TypeError),
#     该崩溃此前每轮刷 "error in _build_plan_prompt: evolution skill/patch fetch failed",
#     直接掐断了 RSI 产物作为 "Learned skills" 回流到 plan 阶段的通路.
# 其余配置与 run37 保持一致, 以隔离此次修复的效果.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run38
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600