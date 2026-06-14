"""Tests for the Rust extension module, when it is compiled/installed."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

huginn_ext = pytest.importorskip("huginn_ext")


pytestmark = [
    pytest.mark.skipif(
        not hasattr(huginn_ext, "tail_lines"),
        reason="huginn_ext compiled without tail_lines support",
    ),
    pytest.mark.skipif(
        not hasattr(huginn_ext, "top_k"),
        reason="huginn_ext compiled without top_k support",
    ),
    pytest.mark.skipif(
        not hasattr(huginn_ext, "sandbox"),
        reason="huginn_ext compiled without sandbox support",
    ),
]


def test_tail_lines_basic() -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for i in range(20):
            f.write(f"line {i + 1}\n")
        path = f.name

    try:
        lines = huginn_ext.tail_lines(path, 5)
        assert len(lines) == 5
        assert lines[0] == "line 16"
        assert lines[-1] == "line 20"
    finally:
        os.unlink(path)


def test_top_k_basic() -> None:
    import numpy as np

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    result = huginn_ext.top_k(query, matrix, k=2)
    assert result["indices"][0] == 0
    assert len(result["indices"]) == 2


def test_run_sandboxed_echo() -> None:
    result = huginn_ext.sandbox.run_sandboxed(
        "echo",
        args=["hello", "sandbox"],
        timeout=5.0,
    )
    assert result["success"] is True
    assert "hello sandbox" in result["stdout"]


def test_run_sandboxed_rejects_shell_meta() -> None:
    with pytest.raises(ValueError):
        huginn_ext.sandbox.run_sandboxed(
            "echo",
            args=["foo; bar"],
            timeout=5.0,
        )
