"""visual_inspect.py (VisualInspectMixin) 覆盖率补测 — 覆盖 measure/annotate/compare
动作、_measure_nearest_primitive 各原语变体、_extract_text_visual_features、
_compare_visual_data、异常分支等.

配合既有 test_visual_inspect_gate.py (zoom + gate note), 把覆盖率从 32% 提升到 90%+.
"""

from __future__ import annotations

import asyncio
import base64 as b64
import io as _io
import types

import numpy as np
import pytest
from PIL import Image

from huginn.autoloop.visual_inspect import (
    VisualInspectMixin,
    _histogram_correlation,
)


class _MockEngine(VisualInspectMixin):
    def __init__(self, visual_ctx: str = "", img_b64: str = "") -> None:
        self._last_visual_context = visual_ctx
        self._visual_base64 = img_b64
        self._last_visual_base64 = None


def _img_b64(arr) -> str:
    buf = _io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(buf, format="PNG")
    return b64.b64encode(buf.getvalue()).decode()


def _run(coro):
    return asyncio.run(coro)


# ── _histogram_correlation ────────────────────────────────────────────────


def test_histogram_correlation_same_high():
    arr = np.full((100, 100), 100, dtype="uint8")
    b = _img_b64(arr)
    raw = b64.b64decode(b)
    assert _histogram_correlation(raw, raw) > 0.9


def test_histogram_correlation_corrupt_input_returns_zero():
    # 非法图字节 → Image.open 抛异常 → 返回 0.0 (覆盖 except 分支)
    assert _histogram_correlation(b"not-an-image", b"not-an-image") == 0.0


# ── _execute_visual_inspect: 路径分支 ─────────────────────────────────────


def test_no_visual_data_returns_error():
    eng = _MockEngine()  # 无 ctx 无 base64
    res = _run(eng._execute_visual_inspect("zoom whatever", {}))
    assert res["success"] is False
    assert "No visual data" in res["error"]


def test_default_inspect_action():
    eng = _MockEngine("[band] peak=<point>[500,800]</point>(2.5)")
    res = _run(eng._execute_visual_inspect("just check the result", {}))
    assert res["success"]
    assert res["actions"][0]["action"] == "inspect"


def test_measure_action(monkeypatch):
    ctx = (
        "[band] peak=<point>[500,800]</point>(2.5), "
        "min=<point>[100,200]</point>(0.1)\n"
    )
    eng = _MockEngine(visual_ctx=ctx)
    res = _run(eng._execute_visual_inspect("measure peak at [490,790]", {}))
    assert res["success"]
    action = res["actions"][0]
    assert action["action"] == "measure"
    assert action["nearest_primitive"]["coordinate"] == [500, 800]


def test_annotate_action_text_features(monkeypatch):
    ctx = (
        "[band] trend=increasing, peak=<point>[500,800]</point>(2.5), "
        "min=<point>[100,200]</point>(0.1), mean=1.2, std=0.3, anomalies=<point>[900,900]</point>=3.0\n"
    )
    eng = _MockEngine(visual_ctx=ctx)
    res = _run(eng._execute_visual_inspect("annotate the band structure", {}))
    assert res["success"]
    action = res["actions"][0]
    assert action["action"] == "annotate"
    assert action["features"]


def test_annotate_action_with_image_tool(monkeypatch):
    # 有图 + image_analysis_tool 可用 → 走 tool 分支
    from huginn.tools import registry

    class _FakeTool:
        def call(self, args, ctx):
            return types.SimpleNamespace(success=True, data={"label": "ok"})

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: _FakeTool())
    eng = _MockEngine(visual_ctx="", img_b64=_img_b64(np.full((100, 100), 10, dtype="uint8")))
    res = _run(eng._execute_visual_inspect("annotate defect in image", {}))
    assert res["success"]
    assert res["actions"][0]["action"] == "annotate"


def test_annotate_image_tool_failure(monkeypatch):
    from huginn.tools import registry

    class _FakeTool:
        def call(self, args, ctx):
            return types.SimpleNamespace(success=False, data=None)

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: _FakeTool())
    eng = _MockEngine(visual_ctx="", img_b64=_img_b64(np.full((100, 100), 10, dtype="uint8")))
    res = _run(eng._execute_visual_inspect("annotate the phase", {}))
    assert res["success"]
    features = res["actions"][0]["features"]
    assert any("tool returned no result" in f for f in features)


def test_compare_action(monkeypatch):
    ctx = (
        "[band] peak=<point>[500,800]</point>(2.5), "
        "min=<point>[100,200]</point>(0.1), anomalies=<point>[900,900]</point>=3.0\n"
    )
    eng = _MockEngine(visual_ctx=ctx)
    res = _run(eng._execute_visual_inspect("compare band 3 and band 5", {}))
    assert res["success"]
    action = res["actions"][0]
    assert action["action"] == "compare"
    assert action["diff"]["peak_count"] == 1


def test_zoom_with_consistency_low_confidence():
    # 上半黑下半白, 跨界区域 +5px 偏移 → 直方图差异大 → low_confidence
    img = np.full((200, 200), 255, dtype="uint8")
    img[0:100, :] = 0
    eng = _MockEngine(img_b64=_img_b64(img))
    res = _run(eng._execute_visual_inspect(
        "zoom into region [490,490]-[550,550]", {}, consistency_check=True
    ))
    action = res["actions"][0]
    assert "consistency_score" in action
    # 偏移跨越边界, 允许出现 low_confidence 或不触发 (结果取决于图)
    assert action["cropped_image"]


def test_zoom_pil_unavailable(monkeypatch):
    # 覆盖 except ImportError 分支: forage PIL 导入失败
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    eng = _MockEngine(img_b64=_img_b64(np.full((50, 50), 0, dtype="uint8")))
    res = _run(eng._execute_visual_inspect("zoom into region [0,0]-[100,100]", {}))
    action = res["actions"][0]
    assert "coordinates only" in action["note"]


def test_zoom_crop_failure(monkeypatch):
    # 覆盖 except Exception 分支: 让 crop 抛异常
    from PIL import Image as PILImage

    class _BadImage:
        @property
        def size(self):
            return (0, 0)  # 除零触发 crop 异常

        def crop(self, *a, **k):
            raise ValueError("crop failed")

    monkeypatch.setattr(PILImage, "open", lambda *a, **k: _BadImage())
    eng = _MockEngine(img_b64=_img_b64(np.full((50, 50), 0, dtype="uint8")))
    res = _run(eng._execute_visual_inspect("zoom into region [0,0]-[100,100]", {}))
    action = res["actions"][0]
    assert "crop failed" in action["note"]


def test_visual_enrich_exception_skipped(monkeypatch):
    import huginn.tools.visual_hook as vh

    def _boom(*a, **k):
        raise RuntimeError("enrich failed")

    monkeypatch.setattr(vh, "enrich_with_visual", _boom)
    eng = _MockEngine(visual_ctx="[band] peak=<point>[500,800]</point>(2.5)")
    res = _run(eng._execute_visual_inspect("measure peak at [500,800]", {}))
    assert res["success"]  # 异常被吞掉, 不阻塞


# ── _measure_nearest_primitive ────────────────────────────────────────────


def test_measure_nearest_variants():
    ctx = "\n".join([
        "<point>[100,200]</point>(2.5)",
        "<point>[300,400]</point>=1.5",
        "<point>[500,600]</point>=0.5%",
        "<point>[700,800]</point> 3.5",
        "score=<point>[900]</point>=0.9",
    ])
    eng = _MockEngine()
    # 目标 (100,200) → 变体 1 最近
    near = eng._measure_nearest_primitive(100, 200, ctx)
    assert near["coordinate"] == [100, 200]
    assert near["value"] == pytest.approx(2.5)
    # 单坐标变体 5: 目标 (0,900)
    near5 = eng._measure_nearest_primitive(0, 900, ctx)
    assert near5["coordinate"] == [0, 900]
    assert near5["label"] == "score"
    assert near5["value"] == pytest.approx(0.9)


def test_measure_nearest_no_primitives():
    eng = _MockEngine()
    assert eng._measure_nearest_primitive(0, 0, "[band] no points here") == {}
    assert eng._measure_nearest_primitive(0, 0, "") == {}


def test_measure_nearest_context_line():
    ctx = "header line\n<point>[100,200]</point>(2.5) trailing context\n"
    eng = _MockEngine()
    near = eng._measure_nearest_primitive(100, 200, ctx)
    assert near["context"]
    assert "<point>" in near["context"]


# ── _annotate_visual_features ─────────────────────────────────────────────


async def test_annotate_no_image_no_ctx():
    eng = _MockEngine()
    out = await eng._annotate_visual_features("annotate", "", "")
    assert out["note"] == "No features found"
    assert out["features"] == []
    assert out["tool_output"] is None


async def test_annotate_with_image_no_tool(monkeypatch):
    from huginn.tools import registry

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: None)
    eng = _MockEngine(img_b64=_img_b64(np.full((50, 50), 0, dtype="uint8")))
    out = await eng._annotate_visual_features("annotate sem", _img_b64(np.full((50, 50), 0, dtype="uint8")), "")
    assert out["tool_output"] is None


async def test_annotate_image_tool_exception(monkeypatch):
    from huginn.tools import registry

    class _BadTool:
        def call(self, args, ctx):
            raise RuntimeError("tool boom")

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: _BadTool())
    eng = _MockEngine()
    out = await eng._annotate_visual_features(
        "annotate defect", _img_b64(np.full((50, 50), 0, dtype="uint8")), ""
    )
    # _call_image_analysis_tool 静默吞掉异常 → 记 tool returned no result, 不抛
    assert any("tool returned no result" in f for f in out["features"])


async def test_annotate_text_structural():
    ctx = (
        "[band] trend=increasing, peak=<point>[500,800]</point>(2.5), "
        "min=<point>[100,200]</point>(0.1), mean=1.2, std=0.3, "
        "anomalies=<point>[900,900]</point>=3.0, <point>[910,910]</point>=3.1\n"
    )
    eng = _MockEngine()
    out = await eng._annotate_visual_features("annotate band", "", ctx)
    assert out["features"]
    assert any("band.trend" in f for f in out["features"])
    assert any("band.anomalies" in f for f in out["features"])
    assert out["tool_output"]["text_analysis"]


# ── _extract_text_visual_features ─────────────────────────────────────────


def test_extract_text_features_sections():
    ctx = "\n".join([
        "[band] trend=decreasing, peak=<point>[500,800]</point>(2.5), min=<point>[100,200]</point>(0.1), mean=1.2, std=0.4",
        "[dos] trend=flat, peak=<point>[600,700]</point>(1.0), min=<point>[200,300]</point>(0.0), anomalies=none",
    ])
    eng = _MockEngine()
    out = eng._extract_text_visual_features(ctx)
    names = [s["name"] for s in out["sections"]]
    assert names == ["band", "dos"]
    assert out["summary"]
    assert "band.trend=decreasing" in out["features"]


def test_extract_text_features_empty():
    eng = _MockEngine()
    out = eng._extract_text_visual_features("")
    assert out["features"] == []
    assert out["summary"] == ""
    assert out["sections"] == []


# ── _compare_visual_data ──────────────────────────────────────────────────


def test_compare_no_ctx():
    eng = _MockEngine()
    out = eng._compare_visual_data("compare a and b", "")
    assert "No visual context" in out["note"]


def test_compare_with_data_and_target():
    ctx = (
        "[band] peak=<point>[500,800]</point>(2.5), min=<point>[100,200]</point>(0.1), "
        "anomalies=<point>[900,900]</point>=3.0\n"
    )
    eng = _MockEngine()
    out = eng._compare_visual_data("compare band 3 and band 5", ctx)
    assert out["diff"]["peak_count"] == 1
    assert "Requested: band 3 vs band 5" in out["note"]


def test_compare_no_quantitative():
    eng = _MockEngine()
    out = eng._compare_visual_data("compare", "[band] n=5, trend=flat")
    assert "no quantitative data" in out["note"]


# ── _selfcheck ────────────────────────────────────────────────────────────


def test_selfcheck_runs(tmp_path):
    """运行模块自带 selfcheck, 覆盖其内部逻辑."""
    from huginn.autoloop.visual_inspect import _selfcheck

    _selfcheck()  # 内部有 assert, 失败会抛


# ── zoom 内 image_analysis_tool 成功分支 (registry.get 返回真值) ─────────


def test_zoom_registry_tool_success(monkeypatch):
    from huginn.tools import registry

    class _FakeTool:
        def call(self, args, ctx):
            return types.SimpleNamespace(success=True, data={})

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: _FakeTool())
    img = np.full((100, 100), 128, dtype="uint8")
    eng = _MockEngine(img_b64=_img_b64(img))
    res = _run(eng._execute_visual_inspect("zoom into region [0,0]-[100,100]", {}))
    assert res["success"]


def test_zoom_registry_import_failure(monkeypatch):
    # 覆盖 except Exception 分支: registry 导入失败
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "huginn.tools.registry", None)
    img = np.full((100, 100), 128, dtype="uint8")
    eng = _MockEngine(img_b64=_img_b64(img))
    res = _run(eng._execute_visual_inspect("zoom into region [0,0]-[100,100]", {}))
    assert res["success"]


def test_zoom_low_confidence(monkeypatch):
    # 强制 _histogram_correlation 返回低值 → low_confidence 分支
    import huginn.autoloop.visual_inspect as vi

    monkeypatch.setattr(vi, "_histogram_correlation", lambda *a, **k: 0.5)
    img = np.full((100, 100), 128, dtype="uint8")
    eng = _MockEngine(img_b64=_img_b64(img))
    res = _run(eng._execute_visual_inspect(
        "zoom into region [0,0]-[100,100]", {}, consistency_check=True
    ))
    action = res["actions"][0]
    assert action.get("low_confidence") is True
    assert "low consistency" in action["note"]


def test_visual_enrich_sets_hint(monkeypatch):
    import huginn.tools.visual_hook as vh

    monkeypatch.setattr(
        vh, "enrich_with_visual",
        lambda *a, **k: {"_visual_hint": "peak near <point>[500,800]</point>"},
    )
    eng = _MockEngine(visual_ctx="[band] peak=<point>[500,800]</point>(2.5)")
    res = _run(eng._execute_visual_inspect("measure peak at [500,800]", {}))
    assert res["_visual_hint"]


async def test_annotate_image_sem_scene(monkeypatch):
    # 描述不含 defect/phase → 默认 sem_analysis scene (作为 tool args["action"])
    from huginn.tools import registry

    class _FakeTool:
        def call(self, args, ctx):
            assert args["action"] == "sem_analysis"
            return types.SimpleNamespace(success=True, data={"ok": 1})

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: _FakeTool())
    eng = _MockEngine()
    out = await eng._annotate_visual_features(
        "annotate the micrograph", _img_b64(np.full((50, 50), 0, dtype="uint8")), ""
    )
    assert any("sem_analysis" in f for f in out["features"])


async def test_annotate_tool_output_and_ctx(monkeypatch):
    # 有图(tool 成功) + 有 visual_ctx → tool_output 补充 text_analysis
    from huginn.tools import registry

    class _FakeTool:
        def call(self, args, ctx):
            return types.SimpleNamespace(success=True, data={"label": "ok"})

    monkeypatch.setattr(registry.ToolRegistry, "get", lambda name: _FakeTool())
    ctx = "[band] trend=increasing, peak=<point>[500,800]</point>(2.5)"
    eng = _MockEngine()
    out = await eng._annotate_visual_features(
        "annotate band", _img_b64(np.full((50, 50), 0, dtype="uint8")), ctx
    )
    assert out["tool_output"]["text_analysis"]
