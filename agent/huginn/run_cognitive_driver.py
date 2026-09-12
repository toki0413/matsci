"""跑真实 agent 自主闭环 (Huginn run_cognitive).

用法(在 agent 根目录, 已装 deps 的 venv):
    export HUGINN_PROVIDER=deepseek HUGINN_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=...
    PYTHONPATH=. python -m huginn.run_cognitive_driver "<objective>" [max_iterations]

把闭环结果(AutoloopResult 关键字段 + 各 phase)落盘到 workspace 下 run_result.json,
方便后续/外部读取, 不依赖交互终端.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from huginn.autoloop.engine import AutoloopEngine
from huginn.memory.manager import MemoryManager


def _summarize_result(r) -> dict:
    phases = getattr(r, "phases", []) or []
    return {
        "success": bool(getattr(r, "success", None)),
        "objective": getattr(r, "objective", ""),
        "run_id": getattr(r, "run_id", ""),
        "goal_achieved": getattr(r, "goal_achieved", None),
        "n_phases": len(phases),
        "phase_names": [getattr(p, "name", "") for p in phases],
        "tests_passed": getattr(r, "tests_passed", None),
        "constraints_satisfied": getattr(r, "constraints_satisfied", None),
    }


async def _main(objective: str, max_iterations: int) -> None:
    import os
    from pathlib import Path as P
    # 强机长期运行: 优先用持久目录(环境注入), 避免写 /tmp 被清
    ws = P(os.environ.get("WORKSPACE_DIR", P("/tmp/huginn_autoloop_ws")))
    ws.mkdir(parents=True, exist_ok=True)
    eng = AutoloopEngine(workspace=str(ws), memory_manager=MemoryManager())
    result = await eng.run_cognitive(objective, max_iterations=max_iterations)
    summary = _summarize_result(result)
    out = ws / "run_result.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== RUN_RESULT_JSON_START ===")
    print(json.dumps(summary, ensure_ascii=False))
    print("=== RUN_RESULT_JSON_END ===", flush=True)


if __name__ == "__main__":
    objective = sys.argv[1] if len(sys.argv) > 1 else (
        "A rod under axial load F=500 N, cross-section A=0.01 m^2, yield stress 250 MPa. "
        "Compute the applied tensile stress sigma=F/A, compare to yield, and give a PASS/FAIL safety verdict."
    )
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    asyncio.run(_main(objective, max_iter))