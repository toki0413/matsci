"""self_observe: agent 自读 trajectories + reflection sidecar, 返回压缩 pattern summary.

Gödel Agent self-perception: agent 在 S7 状态读自己的失败模式.
不返回原始全文 (避免污染主上下文), 只返回 pattern summary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# ponytail: 写死两个路径. trajectories 在 workspace(.huginn/trajectories),
# reflections 在 ~/.huginn/reflections. 升级: 从 config/session_state 注入路径.
TRAJECTORIES_DIR = Path(".huginn") / "trajectories"
REFLECTIONS_DIR = Path.home() / ".huginn" / "reflections"


def _iter_tool_calls(path: Path) -> list[dict[str, Any]]:
    """从 trajectory 文件抽 tool_calls 列表.

    兼容两种格式:
      - .json (telemetry.save_trajectory 的真实格式): 顶层 dict 带 tool_calls 数组
      - .jsonl (一行一条记录的备选格式): 逐行解析

    单文件损坏返回空列表, 不影响整体.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []

    # 优先按 .json 整体解析 (telemetry 实际格式)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            calls = payload.get("tool_calls")
            if isinstance(calls, list):
                return calls
        if isinstance(payload, list):
            return payload
    except json.JSONDecodeError:
        pass

    # 兜底: 按行解析 jsonl (event/tool_name/error 风格)
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def self_observe(window: int = 5) -> dict[str, Any]:
    """读最近 N 个 trajectory + reflection sidecar, 返回压缩 pattern summary.

    返回:
        {
            "recent_turns": int,
            "tool_failures": [{"tool": str, "count": int, "reasons": list[str]}],
            "reflection_flags": list[str],
            "summary": str,
        }
    """
    summary: dict[str, Any] = {
        "recent_turns": 0,
        "tool_failures": [],
        "reflection_flags": [],
        "summary": "",
    }

    # 1. trajectory 文件: 取最近 window 个, 按失败 tool 聚合
    if TRAJECTORIES_DIR.exists():
        # ponytail: glob *.json + *.jsonl, 按 mtime 排序. 升级: 按 timestamp 字段排序更准.
        traj_files = sorted(
            [*TRAJECTORIES_DIR.glob("*.json"), *TRAJECTORIES_DIR.glob("*.jsonl")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        recent_files = traj_files[:window]

        tool_errors: Counter[str] = Counter()
        error_reasons: dict[str, list[str]] = {}
        total_turns = 0

        for f in recent_files:
            for rec in _iter_tool_calls(f):
                total_turns += 1
                # 失败判定: telemetry 写 success=False 或 error 非空;
                # 备选 jsonl 写 event=tool_error.
                failed = (
                    rec.get("success") is False
                    or rec.get("error")
                    or rec.get("event") == "tool_error"
                )
                if not failed:
                    continue
                tool = rec.get("tool") or rec.get("tool_name") or "unknown"
                tool_errors[tool] += 1
                reason = str(rec.get("error") or rec.get("reason") or "")[:100]
                if reason and reason not in error_reasons.setdefault(tool, []):
                    error_reasons[tool].append(reason)

        summary["recent_turns"] = total_turns
        # ponytail: 简单 Counter 频次, 抓不到语义模式. 升级: LLM summary + 聚类.
        summary["tool_failures"] = [
            {"tool": t, "count": c, "reasons": error_reasons.get(t, [])[:3]}
            for t, c in tool_errors.most_common(5)
        ]

    # 2. reflection sidecar: ~/.huginn/reflections/*.jsonl, 每个文件一个 session
    if REFLECTIONS_DIR.exists():
        # ponytail: glob *.jsonl 按 mtime 排序. 升级: 按 session_id 过滤当前会话.
        refl_files = sorted(
            REFLECTIONS_DIR.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        recent_files = refl_files[:window]
        flags: list[str] = []
        for f in recent_files:
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            # 单文件内也限制最近 window 行, 避免老 session 淹没近期信号
            recent_lines = lines[-window:] if len(lines) > window else lines
            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 提取实质标记: 物理错误/警告 + mode 切换 + message
                msg = rec.get("message") or ""
                if rec.get("has_physics_errors"):
                    flags.append(f"[physics_error] {msg}"[:100])
                elif rec.get("has_physics_warnings"):
                    flags.append(f"[physics_warning] {msg}"[:100])
                elif rec.get("should_switch_mode"):
                    flags.append(f"[switch->{rec.get('suggested_mode', '?')}] {msg}"[:100])
                elif msg:
                    flags.append(str(msg)[:100])
        summary["reflection_flags"] = flags[:5]

    # 3. 拼一行 summary 字符串
    parts = [f"recent {summary['recent_turns']} turns"]
    if summary["tool_failures"]:
        top = summary["tool_failures"][0]
        reason = top["reasons"][0] if top["reasons"] else "unknown"
        parts.append(
            f"tool {top['tool']} failed {top['count']} times (reason: {reason})"
        )
    if summary["reflection_flags"]:
        parts.append(f"reflection flagged: {summary['reflection_flags'][0]}")
    summary["summary"] = "; ".join(parts)

    return summary


class SelfObserveToolInput(BaseModel):
    window: int = Field(
        default=5,
        ge=1,
        le=50,
        description="读最近 N 个 trajectory / reflection 文件",
    )


class SelfObserveTool(HuginnTool):
    """self_observe: agent 在 S7_SELF_MODIFY 调用, 读自己最近失败模式."""

    name = "self_observe"
    category = "meta"
    description = (
        "Read your own recent trajectories and reflection sidecar. "
        "Returns a compressed pattern summary of recent tool failures and "
        "reflection flags. Use in S7_SELF_MODIFY state to identify failure patterns."
    )
    read_only = True
    input_schema = SelfObserveToolInput

    def is_read_only(self, args: SelfObserveToolInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = SelfObserveToolInput(**args)
        try:
            data = self_observe(window=input_data.window)
            return ToolResult(data=data, success=True)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"self_observe failed: {e}")


if __name__ == "__main__" and "--self-check" in __import__("sys").argv:
    # self-check: 造 3 个含 circuit_open 失败的 trajectory (.json 真实格式, 带 tool_calls 数组),
    # 验证 self_observe 能抽出该失败模式. 跑完清理测试文件, 不留痕.
    import json as _json
    import sys as _sys

    _traj_dir = TRAJECTORIES_DIR
    _traj_dir.mkdir(parents=True, exist_ok=True)
    # 清掉上次崩溃残留的 selfcheck 文件, 避免污染计数
    for _stale in _traj_dir.glob("_selfcheck_*.json"):
        _stale.unlink(missing_ok=True)
    _created: list[Path] = []
    try:
        for i in range(3):
            # telemetry.save_trajectory 的真实格式: 顶层 dict 带 tool_calls 数组
            payload = {
                "version": "1.0",
                "timestamp": f"2026-01-0{i+1}T00:00:00",
                "tool_calls": [
                    {"step": 1, "tool": "file_read_tool", "success": False,
                     "error": "circuit_open error: connection refused"},
                    {"step": 2, "tool": "file_read_tool", "success": True, "result": "ok"},
                ],
            }
            f = _traj_dir / f"_selfcheck_{i}.json"
            f.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _created.append(f)

        result = self_observe(window=5)

        # circuit_open 必须出现在 summary 或某条 tool_failure 的 reasons 里
        all_reasons = sum([tf["reasons"] for tf in result["tool_failures"]], [])
        assert (
            "circuit_open" in result["summary"]
            or any("circuit_open" in r for r in all_reasons)
        ), f"circuit_open not found: {result}"
        # file_read_tool 必须被识别为失败工具
        assert any(tf["tool"] == "file_read_tool" for tf in result["tool_failures"]), \
            f"file_read_tool not in failures: {result['tool_failures']}"
        print("self_observe self-check PASS")
        print(f"  summary: {result['summary']}")
    finally:
        for f in _created:
            f.unlink(missing_ok=True)
    _sys.exit(0)
