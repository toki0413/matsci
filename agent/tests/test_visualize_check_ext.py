"""visualize_check.py 全分支测试 — 图↔数据一致性校验.

纯函数 + 可注入 extractor/index, 不依赖真实编码器/chromadb.
用 fake extractor 模拟图上反提取, 覆盖标量/列表/误差/重复图/整体 verdict.
"""

from __future__ import annotations

import pytest

from huginn.tools import visualize_check as vc

# ── extract_figure_numeric ───────────────────────────────────────────────

def test_extract_numeric_scalars_and_lists():
    def ex(p):
        return {"peak": 12.5, "intensity": 3, "list": [1.0, 2.5], "ok": True}
    out = vc.extract_figure_numeric("x.png", extractor=ex)
    assert out == {"peak": 12.5, "intensity": 3.0, "list": [1.0, 2.5]}


def test_extract_skips_non_numeric_and_bool():
    def ex(p):
        return {"peak": 1.0, "label": "text", "flag": True, "mixed": [1, "a"]}
    out = vc.extract_figure_numeric("x.png", extractor=ex)
    assert out == {"peak": 1.0}  # mixed 含字符串 → 整个键丢弃


def test_extract_returns_empty_on_error_key():
    def ex(p):
        return {"error": "unparseable"}
    assert vc.extract_figure_numeric("x.png", extractor=ex) == {}


def test_extract_returns_empty_on_non_dict():
    def ex(p):
        return ["not", "a", "dict"]
    assert vc.extract_figure_numeric("x.png", extractor=ex) == {}


def test_extract_returns_empty_on_exception():
    def ex(p):
        raise RuntimeError("extract crash")

    assert vc.extract_figure_numeric("x.png", extractor=ex) == {}


# ── check_figure_vs_expected ─────────────────────────────────────────────

def _exact_extractor(data):
    return lambda p: data


def test_vs_expected_scalar_match():
    ex = _exact_extractor({"energy": 10.0})
    res = vc.check_figure_vs_expected("x.png", {"energy": 10.0}, extractor=ex)
    assert res["verdict_ok"] is True
    assert res["flags"] == []


def test_vs_expected_scalar_drift():
    ex = _exact_extractor({"energy": 12.0})
    res = vc.check_figure_vs_expected(
        "x.png", {"energy": 10.0}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is False
    assert res["flags"][0]["key"] == "energy"
    assert res["flags"][0]["deviation_pct"] == pytest.approx(20.0, abs=0.1)


def test_vs_expected_scalar_within_tolerance():
    ex = _exact_extractor({"energy": 10.5})
    res = vc.check_figure_vs_expected(
        "x.png", {"energy": 10.0}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True


def test_vs_expected_missing_key_skipped():
    ex = _exact_extractor({"other": 1.0})
    res = vc.check_figure_vs_expected("x.png", {"energy": 10.0}, extractor=ex)
    assert res["verdict_ok"] is True
    assert res["flags"] == []


def test_vs_expected_list_match():
    ex = _exact_extractor({"peaks": [1.0, 2.0]})
    res = vc.check_figure_vs_expected(
        "x.png", {"peaks": [1.0, 2.0]}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True


def test_vs_expected_list_drift():
    ex = _exact_extractor({"peaks": [1.0, 3.0]})
    res = vc.check_figure_vs_expected(
        "x.png", {"peaks": [1.0, 2.0]}, tolerance_pct=10.0, extractor=ex
    )
    assert len(res["flags"]) == 1
    assert res["flags"][0]["key"] == "peaks"


def test_vs_expected_list_length_mismatch_skipped():
    ex = _exact_extractor({"peaks": [1.0, 2.0, 3.0]})
    res = vc.check_figure_vs_expected(
        "x.png", {"peaks": [1.0, 2.0]}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True


def test_vs_expected_near_zero_expected_skipped():
    ex = _exact_extractor({"gap": 0.5})
    res = vc.check_figure_vs_expected(
        "x.png", {"gap": 0.0}, tolerance_pct=10.0, extractor=ex
    )
    assert res["verdict_ok"] is True  # abs(exp) < 1e-12 → continue


def test_vs_expected_act_list_scalar_expected():
    ex = _exact_extractor({"peak": [10.0, 20.0]})
    res = vc.check_figure_vs_expected(
        "x.png", {"peak": 10.0}, tolerance_pct=10.0, extractor=ex
    )
    # exp 标量, act 列表 → 取第一项比对
    assert res["verdict_ok"] is True


# ── check_figure_duplicate ───────────────────────────────────────────────

def test_duplicate_no_index():
    res = vc.check_figure_duplicate("x.png", None)
    assert res["duplicate"] is False
    assert res["note"] == "no index"


class _FakeIndex:
    def __init__(self, results=None, raise_=False, empty=False):
        self.results, self.raise_, self.empty = results, raise_, empty

    def search(self, query=None, top_k=None):
        if self.raise_:
            raise RuntimeError("index down")
        if self.empty:
            return []
        return self.results


def test_duplicate_search_raises():
    res = vc.check_figure_duplicate("x.png", _FakeIndex(raise_=True))
    assert res["duplicate"] is False
    assert res["note"] == "search failed"


def test_duplicate_no_results():
    res = vc.check_figure_duplicate("x.png", _FakeIndex(empty=True))
    assert res["duplicate"] is False
    assert res["note"] == "no results"


def test_duplicate_excludes_self():
    res = vc.check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "x.png", "similarity": 0.99}]),
        threshold=0.92,
    )
    assert res["duplicate"] is False
    assert res["matches"] == []


def test_duplicate_skips_entry_without_sim():
    res = vc.check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "y.png"}]),
        threshold=0.92,
    )
    assert res["duplicate"] is False
    assert res["matches"] == []


def test_duplicate_above_threshold():
    res = vc.check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "y.png", "similarity": 0.95}]),
        threshold=0.92,
    )
    assert res["duplicate"] is True
    assert res["duplicate_paths"] == ["y.png"]


def test_duplicate_below_threshold():
    res = vc.check_figure_duplicate(
        "x.png",
        _FakeIndex(results=[{"path": "y.png", "similarity": 0.80}]),
        threshold=0.92,
    )
    assert res["duplicate"] is False
    assert res["matches"][0]["similarity"] == 0.80


# ── consistency_verdict ──────────────────────────────────────────────────

def test_verdict_error_no_expected_no_index():
    res = vc.consistency_verdict("x.png", {})
    assert res["verdict"] == vc.VERDICT_ERROR
    assert "error" in res


def test_verdict_pass():
    ex = _exact_extractor({"energy": 10.0})
    res = vc.consistency_verdict(
        "x.png", {"energy": 10.0}, extractor=ex, index=_FakeIndex(empty=True)
    )
    assert res["verdict"] == vc.VERDICT_PASS


def test_verdict_fix_on_drift():
    ex = _exact_extractor({"energy": 12.0})
    res = vc.consistency_verdict(
        "x.png", {"energy": 10.0}, extractor=ex, index=_FakeIndex(empty=True)
    )
    assert res["verdict"] == vc.VERDICT_FIX
    assert "numeric_drift:energy" in res["flags"]


def test_verdict_fail_on_duplicate():
    ex = _exact_extractor({"energy": 10.0})
    res = vc.consistency_verdict(
        "x.png",
        {"energy": 10.0},
        extractor=ex,
        index=_FakeIndex(results=[{"path": "old.png", "similarity": 0.99}]),
        dup_threshold=0.92,
    )
    assert res["verdict"] == vc.VERDICT_FAIL
    assert "duplicate_figure" in res["flags"]
