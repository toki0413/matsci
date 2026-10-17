#!/usr/bin/env bash
# run36 launcher — 长程探索 (long-horizon) 验证轮.
#
# 与 run35 相比, 平台侧补了两处 long-horizon 缺口 (均为平台通用能力, 与命题无关):
#   1) step cap: CognitiveLoop 的 state.iteration 是**步**(每步一个 action),
#      一个 hypothesize→…→learn 循环约 5 步, 所以 -i 30 只够 ~6 轮就撞顶.
#      现长程模式下按挂钟预算反推宽松步数上限 (预算/10s), 并同步抬 goal.max_iterations.
#   2) wall-clock 硬停: 之前 wall_clock 只用于"让启发式早停让位", 从不作为终止器.
#      现 observe 每步查 wall_clock_expired, 耗尽即落 report — 挂钟成为真正的收口.
#   → 循环在"目标达成(F2/F17) 或 挂钟耗尽"前自主推进, RSI 产物有时间回流.
#
# 本轮: 挂钟 1800s, -i 30 (仅作下限). 预期明显超过 run34/35 的 5 轮上限.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run36
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 1800