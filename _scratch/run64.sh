#!/usr/bin/env bash
# run64 — v27 接线修复复验 (实验组 A: 协作开 + 盲重建).
#
# 目的: run62 (实验组 A) 点亮了两条真问题 trace:
#   - collab_blind_reconstruct skip: no current_hyp_id_for_plan  (cognitive_loop 置位被 None 清空)
#   - collab_blind_reconstruct skip: dispatch returned nothing success=True summary_len=0
#     (subagent._extract_output 只取末条空消息)
# 两处已修 (v27). 本轮用**同一 objective / 同旋钮 / 同 3600s / 同 run57 继承点**,
# 唯一差异: 代码含 v27 修复. 复验两条 skip 是否转成真产出 (holds/mismatch).
#
# 与 run62 严格对齐, 便于 A/B.
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

cd /workspace/research_outputs/shusheng_rsi_run64
exec /workspace/.venv/bin/python -m huginn.cli.main autoloop "$(cat objective.txt)" \
  -i 60 --wall-clock-budget 3600 > run.log 2>&1