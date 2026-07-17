"""ContextBuilder.build_meta_trace_text 自检.

Oxelra Meta-Trace 接入 prompt 的最小验证: 文件不存在/空/正常/缓存命中 4 case.
ponytail: 不引入框架, 跑 `python -m pytest` 或直接 `python test_meta_trace_context.py`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 让 `python test_meta_trace_context.py` 也能跑 (无 pytest 时)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huginn.context_builder import ContextBuilder


def _make_builder(tmp_dir: Path) -> ContextBuilder:
    # memory_manager / cache_builder 在 build_meta_trace_text 路径里不被用到,
    # 传 None 走 ponytail "最少 mock" 路线.
    return ContextBuilder(
        memory_manager=None,
        workspace=str(tmp_dir),
        cache_builder=None,
    )


def _write_trace(tmp_dir: Path, entries: list[dict]) -> None:
    trace_path = tmp_dir / ".huginn" / "meta_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_no_trace_file_returns_empty(tmp_path: Path) -> None:
    b = _make_builder(tmp_path)
    assert b.build_meta_trace_text() == ""
    assert b.meta_trace_available() is False


def test_empty_trace_returns_empty(tmp_path: Path) -> None:
    _write_trace(tmp_path, [])
    b = _make_builder(tmp_path)
    assert b.build_meta_trace_text() == ""
    assert b.meta_trace_available() is True


def test_three_entries_formatted(tmp_path: Path) -> None:
    _write_trace(tmp_path, [
        {"iteration": 1, "darwin_score": 0.3, "supported_ratio": 0.1,
         "attempted": "test band gap hypothesis", "found": "1.12 eV",
         "evidence": ["DFT calc"], "limitations": ["small basis set"],
         "artifacts": ["out.json"], "next_hint": "try bigger basis"},
        {"iteration": 2, "darwin_score": 0.5, "supported_ratio": 0.3,
         "attempted": "refine with HSE06", "found": "1.18 eV",
         "evidence": ["HSE06"], "next_hint": "converge k-mesh"},
        {"iteration": 3, "darwin_score": 0.7, "supported_ratio": 0.6,
         "attempted": "converge k-mesh", "found": "1.15 eV",
         "next_hint": "write report"},
    ])
    b = _make_builder(tmp_path)
    text = b.build_meta_trace_text()
    # 最新在前
    assert "[iter 3]" in text
    assert "[iter 2]" in text
    assert "[iter 1]" in text
    # 倒序: iter 3 应在 iter 1 之前
    assert text.find("[iter 3]") < text.find("[iter 1]")
    assert "### Research Trace" in text
    assert "### End Research Trace" in text
    assert "1.12 eV" in text  # found 字段被拼进去
    assert "darwin=0.7" in text


def test_last_n_cap(tmp_path: Path) -> None:
    # 写 10 条, 默认 last_n=5, 应该只看到后 5 条 (iter 6-10)
    _write_trace(tmp_path, [
        {"iteration": i, "darwin_score": 0.1 * i, "supported_ratio": 0.05 * i}
        for i in range(1, 11)
    ])
    b = _make_builder(tmp_path)
    text = b.build_meta_trace_text(last_n=5)
    assert "[iter 10]" in text
    assert "[iter 6]" in text
    assert "[iter 5]" not in text  # 被截断


def test_cache_hit_same_mtime(tmp_path: Path) -> None:
    _write_trace(tmp_path, [{"iteration": 1, "darwin_score": 0.5}])
    b = _make_builder(tmp_path)
    first = b.build_meta_trace_text()
    # 不动文件, 再调一次 — 应该命中缓存返回同一字符串
    second = b.build_meta_trace_text()
    assert first == second
    assert first is second  # 同一对象 (缓存命中)


def test_invalid_json_skipped(tmp_path: Path) -> None:
    trace_path = tmp_path / ".huginn" / "meta_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        f.write("not json\n")
        f.write(json.dumps({"iteration": 1, "darwin_score": 0.5}) + "\n")
        f.write("{ broken\n")
    b = _make_builder(tmp_path)
    text = b.build_meta_trace_text()
    # 只 iter 1 这条合法, 应该被读出来
    assert "[iter 1]" in text


if __name__ == "__main__":
    # 无 pytest 时也能跑 — 手动 walk 所有 test_* 函数
    import tempfile
    failed = 0
    for name in [n for n in globals() if n.startswith("test_")]:
        fn = globals()[name]
        try:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
            print(f"PASS {name}")
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
