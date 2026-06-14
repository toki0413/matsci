"""Tests for tool output compression."""

from __future__ import annotations

from huginn.tools.compress import ToolOutputCompressor, compress_tool_output


class TestToolOutputCompressor:
    def test_short_string_unchanged(self):
        assert ToolOutputCompressor().compress("hello") == "hello"

    def test_long_text_truncated(self):
        text = "line\n" * 200
        out = ToolOutputCompressor(max_text_lines=10).compress(text)
        assert "omitted" in out
        assert out.count("\n") < 200

    def test_numeric_array_summarized(self):
        data = {"forces": list(range(1000))}
        out = ToolOutputCompressor(array_head_tail=3).compress(data)
        summary = out["forces"]
        assert summary["count"] == 1000
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
        assert len(summary["head"]) == 3
        assert len(summary["tail"]) == 3

    def test_list_of_objects_compressed(self):
        data = [{"step": i, "x": i * 0.1} for i in range(20)]
        out = ToolOutputCompressor(array_head_tail=2).compress(data)
        assert out["_type"] == "compressed_list"
        assert out["count"] == 20
        assert len(out["head"]) == 2
        assert len(out["tail"]) == 2

    def test_keep_keys_preserved(self):
        data = {
            "energy": -123.456,
            "very_long_log": "x" * 100_000,
            "trajectory": list(range(500)),
        }
        out = ToolOutputCompressor(max_output_tokens=100).compress(data)
        assert out["energy"] == -123.456
        assert "very_long_log" in out
        assert out["very_long_log"] != "x" * 100_000

    def test_convenience_entry_point(self):
        out = compress_tool_output({"values": list(range(100))}, max_output_tokens=50)
        assert out["values"]["count"] == 100
