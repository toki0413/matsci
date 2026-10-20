#!/usr/bin/env bash
# run42 launcher — 长程自主探索 (long-horizon), 书生自主推进.
# 相对 run41 的唯一改动:
#   hint block 从 hypothesize prompt 列表的末位 (最低优先级, _trim_to_budget
#   从尾部往回裁剪 → 首先被截断/删除) 提到 body 之后 (高优先级位).
#   背景: run41 日志已确认两级换名归约升级机制**确实在 fire**
#     - "renamed-reduction 3× consecutive: trigger counterexample hunt"
#     - "renamed-reduction 5× consecutive: escalate — block dominant family + force redirect"
#   但 _speculator_hint 里的"[强制重定向]"提示写进 hint_block 后位于 prompt 尾部,
#   预算裁剪时被优先裁掉 → 纠偏信息根本没到 LLM, 所以行为不变、继续换名归约.
#   run42 验证: hint 升优先级后, 重定向提示能否真正改变下一轮 hypothesize.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py

cd /workspace/research_outputs/shusheng_rsi_run42
exec python3 -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 30 --wall-clock-budget 3600