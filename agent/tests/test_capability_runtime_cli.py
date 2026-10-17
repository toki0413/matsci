"""CLI 单测: 只验证 --json 输出结构与 --check 语义 (不真跑 L1)."""

from __future__ import annotations

import importlib.util
import json
import pathlib

# 按文件路径加载: 避免触发 huginn.cli 包 __init__ (它 import rich, 轻量环境可能缺).
_CLI_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "huginn"
    / "cli"
    / "capability_runtime.py"
)
_spec = importlib.util.spec_from_file_location("capability_runtime_cli", _CLI_PATH)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def test_json_output_has_l0_and_summary(capsys):
    rc = cr.main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "l0" in payload and "summary" in payload
    assert payload["l1"] == []


def test_json_is_deterministic(capsys):
    cr.main(["--json"])
    first = capsys.readouterr().out
    cr.main(["--json"])
    second = capsys.readouterr().out
    assert first == second


def test_check_returns_zero_when_everything_runnable(monkeypatch, capsys):
    monkeypatch.setattr(cr, "probe_runtime", lambda *a, **k: [])
    assert cr.main(["--json", "--check"]) == 0


def test_check_returns_one_when_degraded(monkeypatch, capsys):
    from huginn.capabilities import runtime_probe as rp

    monkeypatch.setattr(
        cr,
        "probe_runtime",
        lambda *a, **k: [
            rp.ProbeResult(name="X", kind="tool", status=rp.STATUS_MISSING, missing_deps=["x"])
        ],
    )
    assert cr.main(["--json", "--check"]) == 1


def test_out_writes_file(tmp_path, capsys):
    out = tmp_path / "state.json"
    rc = cr.main(["--json", "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["l0"] is not None


def test_out_creates_missing_parent_dirs(tmp_path, capsys):
    out = tmp_path / "nested" / "deeper" / "state.json"
    rc = cr.main(["--json", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
