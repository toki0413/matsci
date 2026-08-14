"""visualize_check 一致性校验纯函数单测.

用注入的 fake extractor / fake index 做确定性验证, 不依赖真实编码器/图片.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.tools.visualize_check import (
    VERDICT_ERROR,
    VERDICT_FAIL,
    VERDICT_FIX,
    VERDICT_PASS,
    check_figure_duplicate,
    check_figure_vs_expected,
    consistency_verdict,
    extract_figure_numeric,
)


def _fake_extractor(values: dict) -> object:
    def _ex(image_path: str) -> dict:
        return dict(values)

    return _ex


class _FakeIndex:
    """最小 ImageIndex 替身: 支持 search 结果 / 抛错 / 空结果."""

    def __init__(self, results: list[dict] | None = None, raise_: bool = False, empty: bool = False) -> None:
        self.results = results
        self.raise_ = raise_
        self.empty = empty

    def search(self, query, top_k: int = 3) -> list[dict]:
        if self.raise_:
            raise RuntimeError("index down")
        if self.empty:
            return []
        return (self.results or [])[:top_k]


def test_extract_figure_numeric_collects_scalars_and_lists() -> None:
    ex = _fake_extractor(
        {
            "band_gap_eV": 3.2,
            "n_peaks": 3,
            "peak_intensities": [0.3, 0.7, 1.0],
            "axis_calibration": "ocr",  # 字符串忽略
        }
    )
    out = extract_figure_numeric("x.png", extractor=ex)
    assert out["band_gap_eV"] == 3.2
    assert out["n_peaks"] == 3.0
    assert out["peak_intensities"] == [0.3, 0.7, 1.0]
    assert "axis_calibration" not in out


def test_extract_error_returns_empty(tmp_path: Path) -> None:
    ex = _fake_extractor({"error": "numpy not available"})
    assert extract_figure_numeric("x.png", extractor=ex) == {}


def test_extract_exception_returns_empty() -> None:
    def _boom(image_path: str) -> dict:
        raise RuntimeError("boom")

    assert extract_figure_numeric("x.png", extractor=_boom) == {}


def test_check_vs_expected_no_drift() -> None:
    ex = _fake_extractor({"band_gap_eV": 3.2, "peak_intensities": [0.3, 0.7, 1.0]})
    r = check_figure_vs_expected("x.png", {"band_gap_eV": 3.2}, extractor=ex)
    assert r["verdict_ok"] is True
    assert r["flags"] == []


def test_check_vs_expected_drift() -> None:
    ex = _fake_extractor({"band_gap_eV": 4.0})
    r = check_figure_vs_expected("x.png", {"band_gap_eV": 3.2}, extractor=ex)
    assert r["verdict_ok"] is False
    assert len(r["flags"]) == 1
    f = r["flags"][0]
    assert f["key"] == "band_gap_eV"
    assert f["deviation_pct"] == round((4.0 - 3.2) / 3.2 * 100, 1)  # 25.0


def test_check_vs_expected_missing_key_skipped() -> None:
    ex = _fake_extractor({"band_gap_eV": 3.2})
    r = check_figure_vs_expected("x.png", {"n_peaks": 5}, extractor=ex)
    assert r["verdict_ok"] is True  # 图上没提取到 n_peaks → 跳过, 不判 drift


def test_check_duplicate_with_similar_path() -> None:
    idx = _FakeIndex(
        [
            {"path": "a.png", "similarity": 0.95},
            {"path": "b.png", "similarity": 0.5},
        ]
    )
    r = check_figure_duplicate("self.png", idx)
    assert r["duplicate"] is True
    assert r["duplicate_paths"] == ["a.png"]


def test_check_duplicate_excludes_self() -> None:
    idx = _FakeIndex([{"path": "self.png", "similarity": 1.0}])
    r = check_figure_duplicate("self.png", idx)
    assert r["duplicate"] is False
    assert r["matches"] == []


def test_check_duplicate_no_index() -> None:
    r = check_figure_duplicate("x.png", None)
    assert r["duplicate"] is False
    assert r["note"] == "no index"


def test_consistency_pass(tmp_path: Path) -> None:
    ex = _fake_extractor({"band_gap_eV": 3.2})
    r = consistency_verdict("x.png", {"band_gap_eV": 3.2}, index=None, extractor=ex)
    assert r["verdict"] == VERDICT_PASS
    assert r["flags"] == []


def test_consistency_fix_on_drift(tmp_path: Path) -> None:
    ex = _fake_extractor({"band_gap_eV": 4.0})
    r = consistency_verdict("x.png", {"band_gap_eV": 3.2}, index=None, extractor=ex)
    assert r["verdict"] == VERDICT_FIX
    assert "numeric_drift:band_gap_eV" in r["flags"]


def test_consistency_fail_on_duplicate(tmp_path: Path) -> None:
    ex = _fake_extractor({})
    idx = _FakeIndex([{"path": "old.png", "similarity": 0.98}])
    r = consistency_verdict(
        "new.png", {"band_gap_eV": 3.2}, index=idx, extractor=ex
    )
    assert r["verdict"] == VERDICT_FAIL
    assert "duplicate_figure" in r["flags"]


def test_consistency_error_when_no_basis(tmp_path: Path) -> None:
    r = consistency_verdict("x.png", {}, index=None)
    assert r["verdict"] == VERDICT_ERROR
    assert "no expected values" in r["error"]


# ── 全分支扩展 (原 test_visualize_check_ext.py) ───────────────────────────

def test_extract_numeric_scalars_and_lists():
    def ex(p):
        return {"peak": 12.5, "intensity": 3, "list": [1.0, 2.5], "ok": True}
    out = extract_figure_numeric("x.png", extractor=ex)
    assert out == {"peak": 12.5, "intensity": 3.0, "list": [1.0, 2.5]}


def test_extract_skips_non_numeric_and_bool():
    def ex(p):
        return {"peak": 1.0, "label": "text", "flag": True, "mixed": [1, "a"]}
    out = extract_figure_numeric("x.png", extractor=ex)
    assert out == {"peak": 1.0}  # mixed 含字符串 → 整个键丢弃


def test_extract_returns_empty_on_error_key():
    def ex(p):
        return {"error": "unparseable"}
    assert extract_figure_numeric("x.png", extractor=ex) == {}


def test_extract_returns_empty_on_non_dict():
    def ex(p):
        return ["not", "a", "dict"]
    assert extract_figure_numeric("x.png", extractor=ex) == {}


def test_extract_returns_empty_on_exception():
    def ex(p):
        raise RuntimeError("extract crash")

    assert extract_figure_numeric("x.png", extractor=ex) == {}


def _exact_extractor(data):
    return lambda p: data


def test_vs_expected_scalar_match():
    ex = _exact_extractor({"energy": 10.0})
    res = check_figure_vs_expected("x.png", {"energy": 10.0}, extractor=ex)
    assert res["verdict_ok"] is True
    assert res["flags"] == []


def test_vs_expected_scalar_drift():
    ex = _exact_extractor({"energy": 12.0})
    res = check_figure_vs_expected(
        "x.png", {"energy": 10.0}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is False
    assert res["flags"][0]["key"] == "energy"
    assert res["flags"][0]["deviation_pct"] == pytest.approx(20.0, abs=0.1)


def test_vs_expected_scalar_within_tolerance():
    ex = _exact_extractor({"energy": 10.5})
    res = check_figure_vs_expected(
        "x.png", {"energy": 10.0}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True


def test_vs_expected_list_match():
    ex = _exact_extractor({"peaks": [1.0, 2.0]})
    res = check_figure_vs_expected(
        "x.png", {"peaks": [1.0, 2.0]}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True


def test_vs_expected_list_drift():
    ex = _exact_extractor({"peaks": [1.0, 3.0]})
    res = check_figure_vs_expected(
        "x.png", {"peaks": [1.0, 2.0]}, tolerance_pct=10.0, extractor=ex
    )
    assert len(res["flags"]) == 1
    assert res["flags"][0]["key"] == "peaks"


def test_vs_expected_list_length_mismatch_skipped():
    ex = _exact_extractor({"peaks": [1.0, 2.0, 3.0]})
    res = check_figure_vs_expected(
        "x.png", {"peaks": [1.0, 2.0]}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True


def test_vs_expected_near_zero_expected_skipped():
    ex = _exact_extractor({"gap": 0.5})
    res = check_figure_vs_expected(
        "x.png", {"gap": 0.0}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True  # abs(exp) < 1e-12 → continue


def test_vs_expected_act_list_scalar_expected():
    ex = _exact_extractor({"peak": [10.0, 20.0]})
    res = check_figure_vs_expected(
        "x.png", {"peak": 10.0}, tolerance_pct=10.0, extractor=ex
    )
    # exp 标量, act 列表 → 取第一项比对
    assert res["verdict_ok"] is True


def test_duplicate_search_raises():
    res = check_figure_duplicate("x.png", _FakeIndex(raise_=True))
    assert res["duplicate"] is False
    assert res["note"] == "search failed"


def test_duplicate_no_results():
    res = check_figure_duplicate("x.png", _FakeIndex(empty=True))
    assert res["duplicate"] is False
    assert res["note"] == "no results"


def test_duplicate_skips_entry_without_sim():
    res = check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "y.png"}]),
        threshold=0.92,
    )
    assert res["duplicate"] is False
    assert res["matches"] == []


def test_duplicate_above_threshold():
    res = check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "y.png", "similarity": 0.95}]),
        threshold=0.92,
    )
    assert res["duplicate"] is True
    assert res["duplicate_paths"] == ["y.png"]


def test_duplicate_below_threshold():
    res = check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "y.png", "similarity": 0.80}]),
        threshold=0.92,
    )
    assert res["duplicate"] is False
    assert res["matches"][0]["similarity"] == 0.80


def test_verdict_error_no_expected_no_index():
    res = consistency_verdict("x.png", {})
    assert res["verdict"] == VERDICT_ERROR
    assert "error" in res


def test_verdict_fix_on_drift():
    ex = _exact_extractor({"energy": 12.0})
    res = consistency_verdict(
        "x.png", {"energy": 10.0}, extractor=ex, index=_FakeIndex(empty=True)
    )
    assert res["verdict"] == VERDICT_FIX
    assert "numeric_drift:energy" in res["flags"]


def test_verdict_fail_on_duplicate():
    ex = _exact_extractor({"energy": 10.0})
    res = consistency_verdict(
        "x.png",
        {"energy": 10.0},
        extractor=ex,
        index=_FakeIndex(results=[{"path": "old.png", "similarity": 0.99}]),
        dup_threshold=0.92,
    )
    assert res["verdict"] == VERDICT_FAIL
    assert "duplicate_figure" in res["flags"]
