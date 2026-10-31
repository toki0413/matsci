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

from huginn.autoloop.engine import AutoloopEngine
from huginn.env_access import env_bool
from huginn.memory.manager import MemoryManager
from huginn.task_contract import CONTRACT_FILENAME, ContractError, TaskContract, load_contract
from huginn.security.execution import unsafe_env_enabled
from huginn.trace import TraceSampler, default_census_path, trace_enabled


def ensure_contract(workspace) -> TaskContract | None:
    """起跑前校验工作区契约.

    有 ``contract.json`` → 加载并校验 (坏契约抛 ``ContractError``, 不猜);
    没有且未显式设 ``HUGINN_ALLOW_UNCONTRACTED=1`` → 拒绝启动.
    """
    from pathlib import Path as P

    path = P(workspace) / CONTRACT_FILENAME
    if path.is_file():
        return load_contract(path)
    if env_bool("HUGINN_ALLOW_UNCONTRACTED", False):
        return None
    raise ContractError(
        f"拒绝启动: 工作区缺少 {path}; 交付物验收无从判定. "
        "若确需无契约运行, 显式设置 HUGINN_ALLOW_UNCONTRACTED=1."
    )


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
        # 无 Landlock 的降级执行必须可审计, 不能看起来和容器隔离一样.
        "unsafe": unsafe_env_enabled(),
    }


async def _main(objective: str, max_iterations: int) -> None:
    import os
    from pathlib import Path as P
    # 强机长期运行: 优先用持久目录(环境注入), 避免写 /tmp 被清
    ws = P(os.environ.get("WORKSPACE_DIR", P("/tmp/huginn_autoloop_ws")))
    ws.mkdir(parents=True, exist_ok=True)
    try:
        task_contract = ensure_contract(ws)
    except ContractError as exc:
        print(f"[huginn] {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if task_contract is not None:
        print(
            f"[huginn] contract ok sha256={task_contract.sha256} "
            f"files={len(task_contract.required_files)}",
            flush=True,
        )
    eng = AutoloopEngine(workspace=str(ws), memory_manager=MemoryManager())
    sampler = TraceSampler() if trace_enabled() else None
    if sampler is not None:
        sampler.start()
    try:
        result = await eng.run_cognitive(objective, max_iterations=max_iterations)
    finally:
        if sampler is not None:
            sampler.stop()
    summary = _summarize_result(result)
    out = ws / "run_result.json"
    if sampler is not None:
        census = default_census_path()
        trace_path = sampler.write(ws, census)
        summary["trace_file"] = str(trace_path)
        report = sampler.snapshot(census)
        print(
            f"[huginn] trace ok modules={len(report['counts'])} "
            f"calls={report['total_calls']} file={trace_path}",
            flush=True,
        )
        if census is not None:
            zero = report["census"]["keep_zero"]
            print(
                f"[huginn] census KEEP={report['census']['keep_total']} "
                f"但零执行={len(zero)}: {', '.join(zero[:12])}",
                flush=True,
            )
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
