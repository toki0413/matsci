"""visualize_gate 渲染门禁纯函数单测."""

from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: E402  (用于数组类型标注)
from PIL import Image  # noqa: E402

from huginn.tools.visualize_gate import (
    derive_figure_gap_type,
    render_gate_decision,
    run_figure_gate,
    should_retry_render,
)

# --- derive_figure_gap_type -------------------------------------------------

def test_gap_none() -> None:
    assert derive_figure_gap_type([], []) == "none"


def test_gap_duplicate_top_priority() -> None:
    assert (
        derive_figure_gap_type(["blank"], ["duplicate_figure"]) == "duplicate"
    )


def test_gap_data_mismatch() -> None:
    assert (
        derive_figure_gap_type([], ["numeric_drift:band_gap_eV"]) == "data_mismatch"
    )


def test_gap_geometry() -> None:
    assert derive_figure_gap_type(["clipped"], []) == "render_geometry"
    assert derive_figure_gap_type(["blank"], []) == "render_geometry"


def test_gap_quality() -> None:
    assert derive_figure_gap_type(["cluttered"], []) == "render_quality"


def test_gap_data_beats_quality() -> None:
    assert (
        derive_figure_gap_type(["cluttered"], ["numeric_drift:x"]) == "data_mismatch"
    )


# --- should_retry_render ----------------------------------------------------

def test_retry_pass_never() -> None:
    assert should_retry_render("pass", "render_geometry", 0, 2) is False


def test_retry_rerenderable_gap() -> None:
    assert should_retry_render("fix_needed", "render_geometry", 0, 2) is True
    assert should_retry_render("fail", "data_mismatch", 0, 2) is True


def test_retry_duplicate_never() -> None:
    assert should_retry_render("fix_needed", "duplicate", 0, 2) is False


def test_retry_exhausted() -> None:
    assert should_retry_render("fix_needed", "render_geometry", 2, 2) is False


# --- render_gate_decision ---------------------------------------------------

def test_decision_pass() -> None:
    d = render_gate_decision(
        {"verdict": "pass", "flags": []},
        {"verdict": "pass", "flags": []},
        0,
    )
    assert d["action"] == "pass"
    assert d["directive"] == ""


def test_decision_retry() -> None:
    d = render_gate_decision(
        {"verdict": "fix_needed", "flags": ["cluttered"]},
        {"verdict": "pass", "flags": []},
        0,
        max_attempts=2,
    )
    assert d["action"] == "retry"
    assert d["gap_type"] == "render_quality"
    assert d["directive"]


def test_decision_degrade_duplicate() -> None:
    d = render_gate_decision(
        {"verdict": "pass", "flags": []},
        {"verdict": "fail", "flags": ["duplicate_figure"]},
        0,
    )
    assert d["action"] == "degrade"
    assert d["verdict"] == "fail"


def test_decision_degrade_fail_geometry_exhausted() -> None:
    # blank 是硬性 fail, 且已到尝试上限 → degrade 兜底
    d = render_gate_decision(
        {"verdict": "fail", "flags": ["blank"]},
        {"verdict": "pass", "flags": []},
        2,
        max_attempts=2,
    )
    assert d["action"] == "degrade"
    assert d["verdict"] == "fail"


def test_decision_finalize_quality_exhausted() -> None:
    # fix_needed 质量类, 但超上限 → finalize (接受并留痕)
    d = render_gate_decision(
        {"verdict": "fix_needed", "flags": ["cluttered"]},
        {"verdict": "pass", "flags": []},
        2,
        max_attempts=2,
    )
    assert d["action"] == "finalize"
    assert d["verdict"] == "fix_needed"


def test_decision_error() -> None:
    d = render_gate_decision(
        {"verdict": "error", "error": "file not found", "flags": []},
        None,
        0,
    )
    assert d["action"] == "error"
    assert "error" in d["directive"]


# --- run_figure_gate (批次 C 后半: 完整精修闭环) ---------------------------

def _write_png(path: Path, arr) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype="uint8")).save(path, format="PNG")


def _figure_like(width: int = 200, height: int = 150) -> np.ndarray:
    import numpy as np

    arr = np.full((height, width), 255, dtype="uint8")
    arr[20:height - 20, 20:width - 20] = 0  # 居中深色块, 四角留白
    return arr


def _blank(width: int = 200, height: int = 150) -> np.ndarray:
    import numpy as np

    return np.full((height, width), 255, dtype="uint8")


def test_run_gate_pass_no_rerender(tmp_path: Path) -> None:
    p = tmp_path / "ok.png"
    _write_png(p, _figure_like())
    calls: list[str] = []

    def render_fn(directive: str) -> str:
        calls.append(directive)
        return str(p)

    d = run_figure_gate(p, expected=None, index=None, render_fn=render_fn)
    assert d["action"] == "pass"
    assert calls == []  # pass 不触发重渲染
    assert d["image_path"] == str(p)


def test_run_gate_retry_rerenders(tmp_path: Path) -> None:
    # 首张为 blank (fail), render_fn 产出正常图 → 重渲染后 pass
    p = tmp_path / "bad.png"
    _write_png(p, _blank())
    good = tmp_path / "good.png"
    _write_png(good, _figure_like())

    def render_fn(directive: str) -> str:
        return str(good)

    d = run_figure_gate(p, expected=None, index=None, render_fn=render_fn)
    assert d["action"] == "pass"
    assert d["image_path"] == str(good)
    assert d["attempt"] == 1


def test_run_gate_rerender_fails_then_degrade(tmp_path: Path) -> None:
    # 首张 blank, render_fn 每次都产出 still-blank → 重渲染后仍 fail → degrade
    p = tmp_path / "bad.png"
    _write_png(p, _blank())
    worse = tmp_path / "worse.png"
    _write_png(worse, _blank())

    def render_fn(directive: str) -> str:
        return str(worse)

    d = run_figure_gate(
        p, expected=None, index=None, render_fn=render_fn, max_attempts=2
    )
    assert d["action"] == "degrade"
    assert d["verdict"] == "fail"
    assert d["attempt"] == 2


def test_run_gate_render_fn_none_no_loop(tmp_path: Path) -> None:
    # blank 且无 render_fn → 不进入重渲染, 直接降级为 fail (degrade)
    p = tmp_path / "bad.png"
    _write_png(p, _blank())
    d = run_figure_gate(p, expected=None, index=None, render_fn=None)
    assert d["action"] == "degrade"
    assert d["verdict"] == "fail"
    assert d["image_path"] == str(p)


def test_run_gate_rerender_same_path_breaks(tmp_path: Path) -> None:
    # render_fn 返回相同路径 → 判定为未产生新图, break 用当前 decision
    p = tmp_path / "bad.png"
    _write_png(p, _blank())

    def render_fn(directive: str) -> str:
        return str(p)  # 同路径 → 停止

    d = run_figure_gate(
        p, expected=None, index=None, render_fn=render_fn, max_attempts=2
    )
    assert d["action"] == "degrade"
    assert d["verdict"] == "fail"


def test_run_gate_error_short_circuit(tmp_path: Path) -> None:
    # 文件不存在 → QA error → 短路, 不调用 render_fn
    p = tmp_path / "missing.png"
    calls: list[str] = []

    def render_fn(directive: str) -> str:
        calls.append(directive)
        return str(p)

    d = run_figure_gate(p, expected=None, index=None, render_fn=render_fn)
    assert d["action"] == "error"
    assert calls == []
    assert d["image_path"] == str(p)
