"""锁定 L1b 文献检索过滤增强: OA 优先 + 引用数下限 + 跨源归一.

覆盖纯函数 (不打真实 API):
- ``_coerce_int``: 跨源引用数字段归一 (int/float/str/dict/None)
- ``_is_oa``: 跨源 OA 信号归一 (is_oa/open_access/oa_url/download_url)
- ``_apply_filters``: 引用数下限 + OA-only 过滤, 未知引用数不误杀
- ``_rerank``: 同等相关度下 OA 论文排前
"""

from __future__ import annotations

from typing import Any

from huginn.tools.literature.search_sources import (
    _apply_filters,
    _coerce_int,
    _is_oa,
    _rerank,
)
from huginn.tools.literature.tool import (
    LiteratureTool,
    _annotate_value_consistency,
)

# ── _coerce_int ─────────────────────────────────────────────────────────────

class TestCoerceInt:
    def test_none_and_empty(self):
        assert _coerce_int(None) is None
        assert _coerce_int("") is None
        assert _coerce_int("  ") is None

    def test_numeric_types(self):
        assert _coerce_int(7) == 7
        assert _coerce_int(7.9) == 7
        assert _coerce_int("123") == 123
        assert _coerce_int("45.6") == 45

    def test_bool_not_counted(self):
        assert _coerce_int(True) is None

    def test_dict_extracts_score(self):
        assert _coerce_int({"@score": "88"}) == 88
        assert _coerce_int({"value": 10}) == 10

    def test_garbage_returns_none(self):
        assert _coerce_int("n/a") is None
        assert _coerce_int([]) is None


# ── _is_oa ──────────────────────────────────────────────────────────────────

class TestIsOa:
    def test_is_oa_explicit(self):
        assert _is_oa({"is_oa": True}) is True
        # is_oa 显式 False 时, 即使有 download_url 也尊重源 (OpenAlex 语义)
        assert _is_oa({"is_oa": False, "download_url": "https://x/pdf"}) is False

    def test_open_access_bool(self):
        assert _is_oa({"open_access": True}) is True
        assert _is_oa({"open_access": False}) is False

    def test_oa_url_or_download_url(self):
        assert _is_oa({"oa_url": "https://oa.example/fulltext"}) is True
        assert _is_oa({"download_url": "https://core.example/full.pdf"}) is True

    def test_no_signal_false(self):
        assert _is_oa({"title": "x"}) is False
        assert _is_oa({}) is False


# ── _apply_filters ──────────────────────────────────────────────────────────

def _paper(**kw: Any) -> dict[str, Any]:
    base = {"title": "t", "citations": None}
    base.update(kw)
    return base


class TestApplyFilters:
    def test_no_filters_is_noop(self):
        ps = [_paper(citations=10), _paper(citations=None)]
        assert _apply_filters(ps) == ps

    def test_min_citations_keeps_unknown(self):
        # 未知引用数 (None) 必须保留, 防止误杀 arXiv/CORE
        ps = [
            _paper(citations=None),
            _paper(citations=50),
            _paper(citations=3),
        ]
        out = _apply_filters(ps, min_citations=10)
        assert {p["citations"] for p in out} == {None, 50}

    def test_min_citations_string_normalized(self):
        ps = [_paper(citations="200"), _paper(citations="5")]
        out = _apply_filters(ps, min_citations=50)
        assert [p["citations"] for p in out] == ["200"]

    def test_oa_only_keeps_only_oa(self):
        ps = [
            _paper(is_oa=True),
            _paper(download_url="https://x/pdf"),
            _paper(open_access=False),
            _paper(citations=999),  # 无 OA 信号
        ]
        out = _apply_filters(ps, oa_only=True)
        assert len(out) == 2
        assert all(("is_oa" in p) or ("download_url" in p) for p in out)

    def test_combined_and_not_mutating(self):
        ps = [_paper(is_oa=True, citations=8), _paper(is_oa=True, citations=90)]
        out = _apply_filters(ps, min_citations=50, oa_only=True)
        assert len(out) == 1
        assert out[0]["citations"] == 90
        # 原列表不被就地修改
        assert len(ps) == 2


# ── _rerank OA 优先 ─────────────────────────────────────────────────────────

class TestRerankOaPriority:
    def test_oa_ranks_first_at_same_relevance(self):
        # 同 relevance 下, OA 论文应排在非 OA 之前
        papers = [
            {"title": "band gap of Li2O", "citations": 5, "abstract": "band gap study", "source": "a"},
            {"title": "band gap of Li2O", "citations": 100,
             "abstract": "band gap study", "source": "b", "is_oa": True},
        ]
        ranked = _rerank("band gap Li2O", papers)
        assert ranked[0]["source"] == "b"  # OA 优先

    def test_relevance_still_master(self):
        # 相关度仍是主序: 高相关非 OA 排在低相关 OA 前
        hi = {"title": "band gap of Li2O with calculations",
              "abstract": "band gap band gap band gap", "citations": 5, "source": "a"}
        lo = {"title": "unrelated chemistry", "abstract": "", "citations": 999,
              "source": "b", "is_oa": True}
        ranked = _rerank("band gap Li2O", [hi, lo])
        assert ranked[0]["source"] == "a"


# ── L2a: 跨源数值一致性标注 ──────────────────────────────────────────────────

class TestAnnotateValueConsistency:
    def test_single_source_not_verifiable(self):
        r = [{"value": 1.23, "unit": "eV"}]
        out = _annotate_value_consistency(r)
        assert out["overall"]["verdict"] == "insufficient_data"
        assert out["by_unit"][0]["values"][0]["label"] == "single_source"

    def test_clustered_values_marked_consistent(self):
        r = [{"value": 2.0, "unit": "eV"},
             {"value": 2.05, "unit": "eV"},
             {"value": 2.1, "unit": "eV"}]
        out = _annotate_value_consistency(r)
        assert out["overall"]["verdict"] == "consensus"
        # 21.05 -> median 2.05, 2.0 偏差 ~0.024, 2.1 偏差 ~0.024, 都在 5% 内
        for u in out["by_unit"]:
            for v in u["values"]:
                assert v["label"] == "consistent"

    def test_outlier_marked_conflicting(self):
        r = [{"value": 2.0, "unit": "eV"},
             {"value": 2.05, "unit": "eV"},
             {"value": 5.0, "unit": "eV"}]
        out = _annotate_value_consistency(r)
        labels = [v["label"] for u in out["by_unit"] for v in u["values"]]
        assert "conflicting" in labels
        assert out["overall"]["verdict"] == "mixed"

    def test_groups_by_unit_separately(self):
        r = [{"value": 1.0, "unit": "eV"},
             {"value": 10.0, "unit": "J/mol"}]
        out = _annotate_value_consistency(r)
        assert out["overall"]["n_units"] == 2
        # 每个 unit 单源 -> single_source
        assert all(u["n_sources"] == 1 for u in out["by_unit"])

    def test_empty_input(self):
        out = _annotate_value_consistency([])
        assert out["by_unit"] == []
        assert out["overall"]["verdict"] == "insufficient_data"
        assert out["overall"]["n_sources"] == 0


# ── L2b: 引文图 snowballing 节点添加辅助 ──────────────────────────────────────

class TestAddNode:
    @staticmethod
    def _base():
        return (
            [{"paper_id": "seed"}],  # nodes
            {"seed"},                # visited
            [],                      # next_layer
            [],                      # next_dois
        )

    def test_adds_new_node_and_tracks_next_layer(self):
        nodes, visited, next_layer, next_dois = self._base()
        res = LiteratureTool._add_node(
            nodes, visited, next_layer, next_dois,
            child_id="A", title="Paper A", year=2020, doi="10.x/a",
            depth=1, max_nodes=10,
        )
        assert res is True
        assert "A" in visited
        assert next_layer == ["A"]
        assert next_dois == ["10.x/a"]
        assert nodes[-1]["depth"] == 1

    def test_duplicate_not_readded(self):
        nodes, visited, next_layer, next_dois = self._base()
        res = LiteratureTool._add_node(
            nodes, visited, next_layer, next_dois,
            child_id="seed", title="x", year=1, doi=None,
            depth=1, max_nodes=10,
        )
        assert res is True
        assert len(nodes) == 1  # 未重复加
        assert next_layer == []

    def test_cap_returns_false(self):
        nodes, visited, next_layer, next_dois = self._base()
        res = LiteratureTool._add_node(
            nodes, visited, next_layer, next_dois,
            child_id="B", title="x", year=1, doi=None,
            depth=1, max_nodes=1,  # 已满 1 个
        )
        assert res is False
        assert "B" not in visited
