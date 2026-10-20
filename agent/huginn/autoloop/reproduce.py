"""复现重跑闭环 (reproduce): provenance 快照 → 同参重跑 → 数值 diff → 报告.

与 ``autoloop/replay_audit`` 的分工: replay_audit 审计**轨迹质量** (判空转);
本模块验证**结果可复现** —— 「假设—实验—验证—迭代」闭环的最后一环:
同输入 (同参数/同 seed) 重跑工具, 与已记录结果做逐字段数值比对, 报一致/漂移.

用法::

    from huginn.autoloop.reproduce import reproduce_snapshot

    report = reproduce_snapshot(
        {"tool_name": "numerical_tool", "input_params": {"x": 1.0},
         "output": {"y": 2.0}},
        reference_output={"y": 2.0},
    )
    assert report["status"] == "reproduced"

设计约束:
- 纯 stdlib (json/os/asyncio); 默认重跑器惰性 import ToolRegistry ——
  测试/沙箱可用注入 runner 的方式 weightless 跑全部判据.
- 诚实边界 ①: 只覆盖**确定性工具**; LLM 参与的步骤 (名字命中
  ``NON_DETERMINISTIC_TOOL_MARKERS``) 显式 status=not_reproducible, 可见跳过,
  不混入 diff (谎称"复现了"比"没复现"更糟).
- 诚实边界 ②: 快照只存了 output_hash (无逐字段值) 时退化为 hash 比对,
  报告 mode=hash 明示口径; hash 用 provenance.capture 同一判据重算, 不另造口径.
- 诚实边界 ③: 报告给**逐字段 diff** (字段路径/记录值/重跑值/相对误差),
  不给"有漂移"一句空话.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

__all__ = [
    "reproduce_snapshot",
    "reproduce_record",
    "numeric_diff",
    "registry_runner",
    "NON_DETERMINISTIC_TOOL_MARKERS",
]

# LLM 参与生成/判断的步骤: 同输入不保证同输出, 重跑无意义 → 显式跳过 (可见决策).
# 名单是子串匹配 (对 tool name / autoloop mode 名), 新增须同步 tests/test_reproduce.py
# 的名单棘轮 (只增不减).
NON_DETERMINISTIC_TOOL_MARKERS: tuple[str, ...] = (
    "llm",
    "deep_think",
    "self_observe",
    "hypothesis",
    "literature",
    "deli_research",
    "web_search",
    "agentic_search",
    "paper",
    "thesis",
    "orchestrate",
    "subagent",
    "review_committee",
    "nuwa",
    "report",
    "brainstorm",
    "plan",
    "execute",
)

# 逐字段 diff 的最大条数: 防一个全错的巨 diff 撑爆报告/上下文, 截断时显式标注.
_MAX_DIFFS = 50

# 数值判据默认: 相对容差 + 绝对容差 (双方都近零时相对误差无定义).
DEFAULT_TOLERANCE = 1e-9
DEFAULT_ATOL = 1e-12


def is_non_deterministic(tool_name: str) -> bool:
    """LLM 参与的步骤判据: 名字命中显式名单 (子串匹配)."""
    name = (tool_name or "").lower()
    return any(marker in name for marker in NON_DETERMINISTIC_TOOL_MARKERS)


def numeric_diff(
    recorded: Any,
    rerun: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    atol: float = DEFAULT_ATOL,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """递归逐字段 diff. 返回 [] 即一致 (在容差内).

    判据:
    - 数值: |a-b| <= atol 或 相对误差 <= tolerance 视为一致;
    - bool/str/None: 精确相等;
    - dict: 按键并集递归 (缺键报 missing);
    - list/tuple: 长度不等先报 length_mismatch, 再按下标递归;
    - 类型不同: 报 type_mismatch (bool 不算 int).
    """
    diffs: list[dict[str, Any]] = []

    def _add(field: str, **kw: Any) -> None:
        if len(diffs) < _MAX_DIFFS:
            diffs.append({"field": field, **kw})

    def _walk(a: Any, b: Any, path: str) -> None:
        # bool 是 int 子类, 先单独判, 不让 True == 1 混进数值判据
        a_bool, b_bool = isinstance(a, bool), isinstance(b, bool)
        if a_bool or b_bool:
            if not (a_bool and b_bool) or a != b:
                _add(path, recorded=a, rerun=b, reason="bool_mismatch")
            return
        if isinstance(a, int | float) and isinstance(b, int | float):
            delta = abs(a - b)
            if delta <= atol or delta <= tolerance * max(abs(a), abs(b)):
                return
            rel = delta / max(abs(a), abs(b)) if max(abs(a), abs(b)) > 0 else None
            _add(path, recorded=a, rerun=b, rel_err=rel)
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=str):
                if key not in a:
                    _add(f"{path}.{key}", recorded=None, rerun=b[key],
                         reason="missing_in_recorded")
                elif key not in b:
                    _add(f"{path}.{key}", recorded=a[key], rerun=None,
                         reason="missing_in_rerun")
                else:
                    _walk(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, list | tuple) and isinstance(b, list | tuple):
            if len(a) != len(b):
                _add(path, recorded_len=len(a), rerun_len=len(b),
                     reason="length_mismatch")
            for i in range(min(len(a), len(b))):
                _walk(a[i], b[i], f"{path}[{i}]")
            return
        if a != b or type(a) is not type(b):
            _add(path, recorded=a, rerun=b, reason="value_mismatch")

    _walk(recorded, rerun, prefix or "result")
    return diffs


def registry_runner(tool_name: str, input_params: dict[str, Any]) -> Any:
    """默认重跑器: ToolRegistry 解析同名工具 → 同参重调 → 返回 result.data.

    - 工具未注册 / 参数不合法 / 重跑失败: 抛 RuntimeError (由 reproduce_snapshot
      收敛为 status=error, 绝不让异常穿透).
    - 只能在**同步上下文**用 (内部 asyncio.run); 异步宿主请注入自己的 runner.
    """
    import asyncio

    from huginn.core_types import ToolContext
    from huginn.tools.registry import ToolRegistry

    tool = ToolRegistry.get(tool_name)
    if tool is None:
        raise RuntimeError(f"tool not registered: {tool_name!r}")
    try:
        args = tool.input_schema(**input_params)
    except Exception as exc:
        raise RuntimeError(f"input_params 不合法: {exc}") from exc
    context = ToolContext(
        session_id=f"reproduce:{tool_name}", workspace=os.getcwd()
    )
    result = asyncio.run(tool.call(args, context))
    if not result.success:
        raise RuntimeError(f"rerun failed: {result.error}")
    return result.data


def _rerun_output_hash(
    tool_name: str, input_params: dict[str, Any], output: Any
) -> str:
    """hash 口径单一来源: 用 provenance.capture 的同一判据重算 (不另造 hash)."""
    from huginn.provenance import capture

    snap = capture(tool_name, input_params, output=output)
    if not snap.output_hash:
        raise RuntimeError(
            "provenance flag disabled — output hash 不可用 (hash 口径比对无法进行)"
        )
    return snap.output_hash


def reproduce_snapshot(
    snapshot: dict[str, Any],
    *,
    reference_output: Any = None,
    runner: Callable[[str, dict[str, Any]], Any] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    """对一条 provenance 快照重跑并比对.

    Args:
        snapshot: ProvenanceSnapshot.to_dict() 形态的 dict
            (tool_name / input_params / 可选 output / output_hash / random_seed).
        reference_output: 逐字段比对的参考输出; 缺省时依次回退
            snapshot["output"] (快照带了完整输出) → output_hash (hash 口径).
        runner: 重跑器 ``(tool_name, input_params) -> output``; 缺省用
            registry_runner (需工具已注册, 同步上下文).
        tolerance / atol: 数值容差 (相对 / 绝对).

    Returns:
        {"tool_name", "status", "mode", "n_diffs", "diffs", "tolerance", ...}
        status ∈ reproduced | drifted | not_reproducible | error.
    """
    tool_name = snapshot.get("tool_name") or snapshot.get("tool") or ""
    out: dict[str, Any] = {
        "tool_name": tool_name,
        "status": "error",
        "mode": None,
        "n_diffs": 0,
        "diffs": [],
        "tolerance": tolerance,
        "random_seed": snapshot.get("random_seed"),
    }
    if not tool_name:
        out["error"] = "snapshot 缺 tool_name, 无从重跑"
        return out

    if is_non_deterministic(tool_name):
        out["status"] = "not_reproducible"
        out["reason"] = (
            "LLM 参与步骤 (命中 NON_DETERMINISTIC_TOOL_MARKERS), 同输入不保证同输出, "
            "显式跳过 diff —— 诚实报告'不可复现', 不谎称一致"
        )
        return out

    input_params = dict(snapshot.get("input_params") or {})
    try:
        rerun_output = (runner or registry_runner)(tool_name, input_params)
    except Exception as exc:  # 重跑失败收敛为报告, 不穿透
        out["error"] = f"rerun failed: {type(exc).__name__}: {exc}"
        return out

    # 参考输出口径: 显式 reference > 快照内嵌 output > output_hash
    if reference_output is None:
        reference_output = snapshot.get("output")
    if reference_output is not None:
        out["mode"] = "field_diff"
        diffs = numeric_diff(
            reference_output, rerun_output, tolerance=tolerance, atol=atol
        )
        out["diffs"] = diffs
        out["n_diffs"] = len(diffs)
        out["diffs_truncated"] = len(diffs) >= _MAX_DIFFS
        out["status"] = "reproduced" if not diffs else "drifted"
        return out

    recorded_hash = snapshot.get("output_hash") or ""
    if recorded_hash:
        out["mode"] = "hash"
        try:
            new_hash = _rerun_output_hash(tool_name, input_params, rerun_output)
        except Exception as exc:
            out["error"] = str(exc)
            return out
        if new_hash == recorded_hash:
            out["status"] = "reproduced"
        else:
            out["status"] = "drifted"
            out["diffs"] = [
                {
                    "field": "output_hash",
                    "recorded": recorded_hash,
                    "rerun": new_hash,
                    "reason": "hash_mismatch (快照未存逐字段值, 只能给 hash 级比对)",
                }
            ]
            out["n_diffs"] = 1
        return out

    out["error"] = (
        "快照既无逐字段输出 (output) 也无 output_hash, 没有可比对的参考结果"
    )
    return out


def reproduce_record(
    record: dict[str, Any],
    *,
    runner: Callable[[str, dict[str, Any]], Any] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """对一条 run 级 provenance record (含 tool_chain) 逐快照重跑并汇总.

    单条失败不拖垮整条链; 汇总按 status 计数, 总状态取最差
    (reproduced < drifted < not_reproducible/error 视为需关注).
    """
    chain = list(record.get("tool_chain") or [])
    results = [
        reproduce_snapshot(s, runner=runner, tolerance=tolerance) for s in chain
    ]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "run_id": record.get("run_id", ""),
        "n_snapshots": len(chain),
        "results": results,
        "counts": counts,
        "all_reproduced": bool(results) and all(
            r["status"] == "reproduced" for r in results
        ),
    }
