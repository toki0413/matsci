"""Tests for the lightweight symbol index."""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.coder.symbol_index import SymbolIndex


@pytest.fixture
def sample_workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        "def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return foo()\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "from a import foo\n\ndef qux():\n    return foo() + 1\n",
        encoding="utf-8",
    )
    (tmp_path / ".pytest_cache").mkdir(parents=True)
    (tmp_path / ".pytest_cache" / "ignored.py").write_text(
        "def ignored():\n    pass\n", encoding="utf-8"
    )
    return tmp_path


class TestSymbolIndex:
    def test_indexes_python_symbols(self, sample_workspace: Path):
        index = SymbolIndex(sample_workspace)
        index.build(extensions={".py"})

        foo_syms = index.symbols("foo")
        assert len(foo_syms) == 1
        assert foo_syms[0].kind == "function"
        assert foo_syms[0].line == 1

        bar_syms = index.symbols("Bar")
        assert len(bar_syms) == 1
        assert bar_syms[0].kind == "class"

        baz_syms = index.symbols("baz")
        assert len(baz_syms) == 1
        assert baz_syms[0].kind == "method"

    def test_indexes_references(self, sample_workspace: Path):
        index = SymbolIndex(sample_workspace)
        index.build(extensions={".py"})

        foo_refs = index.references("foo")
        # foo is referenced in a.py (return foo()) and b.py (return foo() + 1)
        assert len(foo_refs) >= 2
        files = {Path(ref.file) for ref in foo_refs}
        assert (sample_workspace / "a.py") in files
        assert (sample_workspace / "b.py") in files

    def test_files_using_symbol(self, sample_workspace: Path):
        index = SymbolIndex(sample_workspace)
        index.build(extensions={".py"})

        files = index.files_using("foo")
        assert sample_workspace / "a.py" in files
        assert sample_workspace / "b.py" in files

    def test_skips_ignored_directories(self, sample_workspace: Path):
        index = SymbolIndex(sample_workspace)
        index.build(extensions={".py"})

        assert index.symbols("ignored") == []
