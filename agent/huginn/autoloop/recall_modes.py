"""RecallModes — VISTA 借鉴的两个"模型可调"mode 协作对象.

把此前只存在于引擎内部、模型从不会调用的能力, 变成计划提示词里可选的 mode
(VISTA 的核心主张: 由模型自己决定何时回看/trace, 而非静态塞进上下文):

- ``trace_inspect``  — 把 engine.search_execution_traces 暴露成 read-only mode:
  模型可按关键词/tool/intent 取回自己的过程级执行 trace (含 exit_class).
- ``frame_inspect``  — 配合 FrameStore 的无损观测记忆: 回看某帧 / 裁剪区域 /
  read_pixels 取精确像素值.

两个 flag 都默认关. 关闭时: 计划提示词不列这两个 mode (模型不会产出), 且即便
被误产出, 执行入口也返回"disabled"说明而不改控制流 —— 向后兼容, 零回归.

与视觉方法族同款协作对象结构: 未定义的私有属性读写转发到 self.engine, 只读引擎
字段 (不复制状态).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 关键词: 只用于 description 的自由文本解析兜底 (模型偶尔不严格按 JSON 输出)
_KV_RE = re.compile(r"(?:^|[\s,;])([A-Za-z_]+)\s*[:=]\s*(\[[^\]]*\]|\"[^\"]*\"|'[^']*'|[^\s,;]+)")


def _flag_on(name: str) -> bool:
    try:
        from huginn.feature_flags import FeatureFlags

        return FeatureFlags.shared().is_enabled(name)
    except Exception:  # 防御: flag 层异常 → 当作关 (功能默认 off)
        logger.debug("feature flag read failed for %s", name, exc_info=True)
        return False


def _as_float_list(value: Any) -> list[float] | None:
    """把 [..] / 'a,b,c' / 'a b c' 解析成一维 float 列表; 失败 None."""
    if value is None:
        return None
    if isinstance(value, list | tuple):
        try:
            return [float(v) for v in value]
        except Exception:
            return None
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
        try:
            return [float(p) for p in parts]
        except Exception:
            return None
    return None


class RecallModes:
    """trace_inspect / frame_inspect 方法族协作对象."""

    def __init__(self, engine: Any) -> None:
        object.__setattr__(self, "engine", engine)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self.engine, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "engine":
            object.__setattr__(self, name, value)
            return
        if self.engine is self:
            object.__setattr__(self, name, value)
            return
        setattr(self.engine, name, value)

    # ── 帧捕获 (execute 尾部调用, flag 关时零成本直返) ────────────────
    def capture_if_enabled(self, tool: str, result: Any) -> dict[str, Any] | None:
        """把本轮 execute 产出的视觉帧原样落盘 (仅 visual_frame_memory 开时)."""
        if not _flag_on("visual_frame_memory"):
            return None
        try:
            b64: Any = None
            if isinstance(result, dict):
                b64 = result.get("_visual_base64") or result.get("visual_base64")
            if not b64:
                b64 = getattr(self.engine, "_visual_base64", "") or getattr(
                    self.engine, "_last_visual_base64", ""
                )
            if not b64:
                return None
            store = self._get_frame_store()
            if store is None:
                return None
            return store.capture(
                b64, turn=getattr(self.engine, "_iteration", None), tool=tool
            )
        except Exception:  # 防御: 捕获 best-effort, 绝不带挂 execute
            logger.debug("frame capture failed", exc_info=True)
            return None

    # ── description → 结构化参数 ────────────────────────────────────
    @staticmethod
    def _parse_params(description: str) -> dict[str, Any]:
        """从 DESCRIPTION 解析参数: 优先整段 JSON, 退化成内嵌 JSON, 再退化成 key=value."""
        text = (description or "").strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                logger.debug("recall mode json parse failed", exc_info=True)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                logger.debug("embedded json parse failed", exc_info=True)
        out: dict[str, Any] = {}
        for key, raw in _KV_RE.findall(text):
            val: Any = raw.strip().strip("\"'")
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                val = inner if inner else None
            out[key] = val
        return out

    @staticmethod
    def _truthy(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    # ── mode 1: 过程级 trace 检索 ───────────────────────────────────
    async def _execute_trace_inspect(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """read-only: 按关键词/tool/intent 检索自己的过程级执行 trace."""
        if not _flag_on("trace_inspect"):
            return {
                "mode": "trace_inspect",
                "success": False,
                "error": (
                    "trace_inspect disabled; enable with "
                    "HUGINN_FEATURE_TRACE_INSPECT=true"
                ),
            }
        p = self._parse_params(description)
        try:
            limit = int(p.get("limit", 8) or 8)
        except Exception:
            limit = 8
        hits = self.engine.search_execution_traces(
            query=str(p.get("query", "") or ""),
            tool=str(p.get("tool", "") or ""),
            intent=str(p.get("intent", "") or ""),
            exclude_self=self._truthy(p.get("exclude_self"), True),
            limit=max(1, min(limit, 50)),
        )
        return {
            "mode": "trace_inspect",
            "success": True,
            "count": len(hits),
            "query": {
                "query": str(p.get("query", "") or ""),
                "tool": str(p.get("tool", "") or ""),
                "intent": str(p.get("intent", "") or ""),
            },
            "traces": hits,
        }

    # ── mode 2: 无损帧回看 / 区域裁剪 / 像素读取 ─────────────────────
    async def _execute_frame_inspect(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """回看历史帧: view (整帧) / region (裁区域) / pixels (取精确 RGB)."""
        if not _flag_on("visual_frame_memory"):
            return {
                "mode": "frame_inspect",
                "success": False,
                "error": (
                    "frame_inspect disabled; enable with "
                    "HUGINN_FEATURE_VISUAL_FRAME_MEMORY=true"
                ),
            }
        store = self._get_frame_store()
        if store is None:
            return {"mode": "frame_inspect", "success": False, "error": "frame store unavailable"}

        p = self._parse_params(description)
        action = str(p.get("action", "view") or "view").strip().lower()
        frame_id = p.get("frame_id")
        try:
            frame_id = int(frame_id) if frame_id is not None else None
        except Exception:
            frame_id = None
        if frame_id is None:
            frame_id = store.last_frame_id()
        if frame_id is None:
            return {"mode": "frame_inspect", "success": False, "error": "no frames captured yet"}

        if action in ("region", "zoom", "crop"):
            box = _as_float_list(p.get("box")) or [0.0, 0.0, 1.0, 1.0]
            res = store.region(frame_id, box)
            kind = "region"
        elif action in ("pixels", "read_pixels", "sample"):
            points = p.get("points")
            if isinstance(points, str):
                # "x1,y1;x2,y2" 之类
                points = [
                    _as_float_list(seg)
                    for seg in re.split(r"[;|]", points)
                    if seg.strip()
                ]
            if not isinstance(points, list):
                points = []
            points = [q for q in (_as_float_list(q) for q in points) if q]
            res = store.read_pixels(
                frame_id, points, normalized=self._truthy(p.get("normalized"), False)
            )
            kind = "pixels"
        else:  # view
            raw = store.get_bytes(frame_id)
            if raw is None:
                res = {"error": f"frame {frame_id} not found"}
            else:
                res = {
                    "frame_id": frame_id,
                    "record": store.get_record(frame_id),
                    "image_b64": base64.b64encode(raw).decode(),
                }
            kind = "view"

        _err = res.get("error") if isinstance(res, dict) else "invalid result"
        out: dict[str, Any] = {
            "mode": "frame_inspect",
            "success": not _err,
            "action": kind,
            "frame_id": frame_id,
            "result": res,
        }
        if _err:
            out["error"] = _err
        # 把帧喂给多模态路径 (与 visual_inspect 同款: 结果带 _visual_base64),
        # 同时写回引擎的"最近帧"缓存, 让下一轮 visual_inspect 能接着用.
        b64 = res.get("image_b64") if isinstance(res, dict) else None
        if b64:
            out["_visual_base64"] = b64
            try:
                self.engine._last_visual_base64 = b64
            except Exception:  # 防御: 写回失败不影响返回
                logger.debug("frame b64 writeback failed", exc_info=True)
            meta = store.get_record(frame_id) or {}
            out["_visual_hint"] = (
                f"Re-viewed frame #{frame_id} "
                f"(tool={meta.get('tool', '?')}, turn={meta.get('turn', '?')}, "
                f"size={meta.get('w')}x{meta.get('h')}). "
                "This is the original pixels, not a text projection."
            )
        return out