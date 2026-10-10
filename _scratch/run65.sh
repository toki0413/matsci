#!/usr/bin/env bash
# run65 — v28 修复复验 (实验组 A: 协作开 + 盲重建).
#
# 目的: run64 证明 v27 的 id 修复生效 (iter 14/19 不再因 id 跳过), 但
#   `summary_len=0` 仍在 —— 真因是 subagent.dispatch 把 chat() 流的**末条控制事件**
#   ({"_auto_continue": True}, 无 "messages") 当成 final_state, 而非 v27 以为的
#   "末条消息为空". v28 改为只认带非空 messages 的 state.
# 本轮用**同一 objective / 同旋钮 / 同 3600s / 同 run57 继承点**, 唯一差异: v28.
# 预期: collab_blind_reconstruct 不再出 summary_len=0, 而是真产出/进入解析分支.
#
# 与 run62/run64 严格对齐, 便于三段 A/B.
set -u
export HUGINN_PROVIDER=internlm
export HUGINN_MODEL=intern-s2-preview
export INTERNLM_API_KEY=sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba
export HUGINN_COGNITIVE_LLM_DECIDER=0
export HUGINN_CODELAB_TIMEOUT_S=900
export HUGINN_EXEC_ROUTE_DEBUG=1
export HUGINN_PERSISTENT_GOAL_MODE=1
export PYTHONPATH=/workspace/agent
export HUGINN_CODELAB_SCAFFOLD=/workspace/examples/codelab_scaffolds/network_rigidity.py
# --- 实验组 A: 协作开 + 盲重建 ---
export HUGINN_ENABLE_AGENT_COLLAB=1
export HUGINN_BLIND_RECONSTRUCTION=1

cd /workspace/research_outputs/shusheng_rsi_run65
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1