"""PromptSecurity 全分支测试 — 不可信内容标记包装."""

from __future__ import annotations

from huginn.security.prompt_security import (
    _CLOSE,
    _OPEN,
    _PREAMBLE,
    untrusted_context_message,
    wrap_rag_chunks,
)


# ── untrusted_context_message ────────────────────────────────────────────

def test_empty_content_returns_as_is():
    assert untrusted_context_message("rag", "") == ""


def test_wraps_with_open_close_tags():
    out = untrusted_context_message("scrape", "hello")
    assert _OPEN.format(label="scrape") in out
    assert _CLOSE.format(label="scrape") in out
    assert "hello" in out


def test_includes_preamble():
    out = untrusted_context_message("rag", "x")
    assert _PREAMBLE in out


def test_source_included_in_open_tag():
    out = untrusted_context_message("rag", "x", source="s1")
    assert " source=s1" in out


def test_no_source_omits_marker():
    out = untrusted_context_message("rag", "x")
    assert " source=" not in out


def test_content_preserved():
    out = untrusted_context_message("rag", "secret text")
    assert "secret text" in out


# ── wrap_rag_chunks ──────────────────────────────────────────────────────

def test_wrap_document_and_preserve_raw():
    r = {"document": "secret text", "metadata": {"source": "s1"}}
    results = [r]
    out = wrap_rag_chunks(results)
    assert out is results  # in-place
    assert r["_raw_document"] == "secret text"
    assert "source=s1" in r["document"]
    assert "secret text" in r["document"]


def test_no_source_in_metadata():
    r = {"document": "txt"}
    wrap_rag_chunks([r])
    assert r["_raw_document"] == "txt"
    assert " source=" not in r["document"]


def test_non_string_document_untouched():
    r = {"document": 42}
    wrap_rag_chunks([r])
    assert r["document"] == 42
    assert "_raw_document" not in r


def test_empty_document_untouched():
    r = {"document": ""}
    wrap_rag_chunks([r])
    assert r["document"] == ""
    assert "_raw_document" not in r


def test_missing_document_untouched():
    r = {"metadata": {}}
    wrap_rag_chunks([r])
    assert "_raw_document" not in r


def test_empty_results_returns_empty():
    assert wrap_rag_chunks([]) == []