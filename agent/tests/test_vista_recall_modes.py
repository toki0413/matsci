"""VISTA 借鉴落地回归网: trace_inspect / frame_inspect 两个"模型可调"mode.

覆盖:
  * FrameStore 无损帧存储 — 捕获/回看/裁剪/read_pixels.
  * 两个 mode 的 flag 门禁: 关时返回 disabled 且不改控制流 (向后兼容).
  * flag 开时: trace_inspect 读过程级台账; frame_inspect 回看历史帧 + 取像素.
  * 计划提示词 MODE 枚举门禁: 关时与历史逐字一致, 开时才追加新 mode.

全部 hermetic: 引擎 fixture 用与 test_autoloop_engine 同款 stub, 不触网/不跑子进程.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.frame_store import FrameStore
from huginn.feature_flags import FeatureFlags

_HISTORICAL_ENUM = "coder|workflow|explore|skill|visual_inspect"


def _run(coro):
    """跑一个协程到完成 (asyncio.get_event_loop 在 Py3.14 已无主线程隐式 loop)."""
    return asyncio.run(coro)


@pytest.fixture
def ff():
    """FeatureFlags 单例; 用完把本测试涉及的 flag 复原, 避免跨测试泄漏."""
    flags = FeatureFlags.shared()
    names = ("trace_inspect", "visual_frame_memory")
    for n in names:
        flags.reset(n)
    yield flags
    for n in names:
        flags.reset(n)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """与 test_autoloop_engine 同款轻量 stub 引擎 (workspace=tmp_path)."""
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    return AutoloopEngine(workspace=tmp_path)


def _png_b64(w: int = 3, h: int = 2, base=(10, 20, 30), dot=(200, 100, 50)) -> str:
    PIL = pytest.importorskip("PIL.Image")
    im = PIL.new("RGB", (w, h), base)
    im.putpixel((1, 0), dot)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── FrameStore ──────────────────────────────────────────────────────────
class TestFrameStore:
    def test_capture_and_readback(self, tmp_path: Path):
        store = FrameStore(tmp_path)
        assert store.frames() == []
        rec = store.capture(_png_b64(), turn=1, tool="code_lab")
        assert rec and rec["frame_id"] == 1
        assert store.frames()[0]["tool"] == "code_lab"
        assert store.get_bytes(1) is not None
        assert store.capture("") is None

    def test_read_pixels_exact(self, tmp_path: Path):
        store = FrameStore(tmp_path)
        store.capture(_png_b64(), turn=1)
        px = store.read_pixels(1, [[1, 0], [99, 99]])
        assert px["size"] == [3, 2]
        assert px["pixels"][0]["rgb"] == [200, 100, 50]
        assert px["pixels"][1]["in_bounds"] is False

    def test_region_crop(self, tmp_path: Path):
        store = FrameStore(tmp_path)
        store.capture(_png_b64(), turn=1)
        rg = store.region(1, [0.0, 0.0, 1.0, 1.0])
        assert rg.get("crop_size") == [3, 2], rg

    def test_missing_frame_is_graceful(self, tmp_path: Path):
        store = FrameStore(tmp_path)
        assert "error" in store.region(9, [0, 0, 1, 1])
        assert "error" in store.read_pixels(9, [[0, 0]])


# ── flag 门禁 (向后兼容) ─────────────────────────────────────────────────
class TestFlagGating:
    def test_modes_disabled_by_default(self, engine, ff):
        off_trace = _run(
            engine._execute_trace_inspect('{"query":"x"}', {})
        )
        off_frame = _run(
            engine._execute_frame_inspect('{"action":"view"}', {})
        )
        assert off_trace["success"] is False and "disabled" in off_trace["error"]
        assert off_frame["success"] is False and "disabled" in off_frame["error"]

    def test_mode_enum_historical_when_off(self, engine, ff):
        assert engine._plan_checker._plan_mode_enum() == _HISTORICAL_ENUM
        # 说明行为空串 → 提示词与历史逐字一致
        assert engine._plan_checker._plan_extra_mode_lines() == ""

    def test_mode_enum_extends_when_on(self, engine, ff):
        ff.enable("trace_inspect")
        ff.enable("visual_frame_memory")
        enum = engine._plan_checker._plan_mode_enum()
        assert enum.startswith(_HISTORICAL_ENUM)
        assert "trace_inspect" in enum and "frame_inspect" in enum
        lines = engine._plan_checker._plan_extra_mode_lines()
        assert "trace_inspect" in lines and "frame_inspect" in lines


# ── trace_inspect ───────────────────────────────────────────────────────
class TestTraceInspect:
    def test_recalls_ledger(self, engine, ff):
        ff.enable("trace_inspect")
        engine._execution_ledger.append(
            {
                "idx": 1,
                "tool": "code_lab",
                "intent": "h1",
                "ts": 1.0,
                "exit_class": "timeout",
                "result": "run timed out after 300s",
            }
        )
        out = _run(
            engine._execute_trace_inspect('{"query":"timed out"}', {})
        )
        assert out["success"] is True and out["count"] == 1
        assert out["traces"][0]["tool"] == "code_lab"

    def test_no_match_returns_empty(self, engine, ff):
        ff.enable("trace_inspect")
        engine._execution_ledger.append(
            {"idx": 1, "tool": "coder", "intent": "h1", "ts": 1.0, "result": "ok"}
        )
        out = _run(
            engine._execute_trace_inspect('{"query":"nonexistent-token"}', {})
        )
        assert out["success"] is True and out["count"] == 0


# ── frame_inspect ───────────────────────────────────────────────────────
class TestFrameInspect:
    def test_view_region_pixels(self, engine, ff):
        ff.enable("visual_frame_memory")
        store = engine._get_frame_store()
        store.capture(_png_b64(), turn=1, tool="code_lab")

        px = _run(
            engine._execute_frame_inspect(
                '{"action":"pixels","frame_id":1,"points":[[1,0]]}', {}
            )
        )
        assert px["success"] is True
        assert px["result"]["pixels"][0]["rgb"] == [200, 100, 50]

        rg = _run(
            engine._execute_frame_inspect(
                '{"action":"region","frame_id":1,"box":[0,0,1,1]}', {}
            )
        )
        assert rg["success"] is True and rg["result"]["crop_size"] == [3, 2]

        vw = _run(
            engine._execute_frame_inspect('{"action":"view","frame_id":1}', {})
        )
        assert vw["success"] is True
        assert vw.get("_visual_base64")
        assert engine._last_visual_base64  # 写回最近帧缓存

    def test_defaults_to_last_frame(self, engine, ff):
        ff.enable("visual_frame_memory")
        engine._get_frame_store().capture(_png_b64(), turn=1)
        out = _run(
            engine._execute_frame_inspect('{"action":"view"}', {})
        )
        assert out["frame_id"] == 1 and out["success"] is True

    def test_no_frames_yet(self, engine, ff):
        ff.enable("visual_frame_memory")
        out = _run(
            engine._execute_frame_inspect('{"action":"view"}', {})
        )
        assert out["success"] is False and "no frames" in out["error"]


# ── 捕获钩子: flag 关时零成本, 开时落盘 ────────────────────────────────────
class TestCaptureHook:
    def test_off_is_noop(self, engine, ff):
        assert engine._recall_modes.capture_if_enabled("code_lab", {}) is None
        assert not (engine.workspace / ".huginn" / "frames" / "index.json").exists()

    def test_on_captures_visual_b64(self, engine, ff):
        ff.enable("visual_frame_memory")
        rec = engine._recall_modes.capture_if_enabled(
            "code_lab", {"_visual_base64": _png_b64(), "result": 1}
        )
        assert rec and rec["frame_id"] == 1
        assert engine._get_frame_store().last_frame_id() == 1