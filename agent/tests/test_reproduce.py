"""复现重跑闭环 (reproduce) 单元测试 — G-C 能力锚点.

覆盖 ``huginn/autoloop/reproduce.py`` (核心判据) 与 ``huginn/cli/reproduce.py``
(入口聚合/退出码):

- numeric_diff 逐字段判据: 相对/绝对容差、bool≠int、缺键、长度、类型、嵌套
  路径、_MAX_DIFFS 截断;
- reproduce_snapshot 四态: reproduced / drifted (带 rel_err 的可见 diff) /
  not_reproducible (LLM 步显式跳过, 不谎称一致) / error (重跑失败收敛不穿透);
- NON_DETERMINISTIC_TOOL_MARKERS 名单棘轮: live 名单与本处冻结副本逐项
  相等 —— 增删名单必须同时改两处 (可见决策);
- hash 口径单一来源: 快照无逐字段 output 时退化为 hash 比对, 判据复用
  provenance.capture (不另造口径);
- CLI: 快照/记录/JSONL 装载、退出码 (全 reproduced=0, 其余=1, 空文件=2 防
  假绿)、坏行显式报行号。

weightless 策略 (对齐 smoke_specs._load_reproduce_module): reproduce 按文件
路径加载, 绕开 ``huginn/autoloop/__init__`` 的重引擎 import; CLI 按路径加载并
把 ``huginn.autoloop.reproduce`` 临时指向同一实现, 用完恢复 sys.modules.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

_AGENT = pathlib.Path(__file__).resolve().parents[1]

# weightless: 按文件路径加载核心判据 (绕开 autoloop/__init__ 的 engine 全链).
_REPRODUCE_PATH = _AGENT / "huginn" / "autoloop" / "reproduce.py"
_spec = importlib.util.spec_from_file_location("huginn_autoloop_reproduce", _REPRODUCE_PATH)
rp = importlib.util.module_from_spec(_spec)
sys.modules["huginn_autoloop_reproduce"] = rp
_spec.loader.exec_module(rp)

# 名单棘轮的冻结副本: live 名单增删须同步改这里 (只增不减, 改动 = 可见决策).
_EXPECTED_MARKERS = (
    "llm",
    "deep_think",
    "self_observe",
    "hypothesis",
    "literature",
    "deli_research",
    "web_search",
    "agentic_search",
    "paper",
    "thesis",
    "orchestrate",
    "subagent",
    "review_committee",
    "nuwa",
    "report",
    "brainstorm",
    "plan",
    "execute",
)


@pytest.fixture
def cli_mod():
    """按路径加载 huginn/cli/reproduce.py (绕开 cli/__init__ 的 typer/rich).

    ``huginn.autoloop.reproduce`` 临时指向按路径加载的同一实现; 用完恢复
    sys.modules 与包属性, 不污染同 worker 其他测试的导入面.
    """
    import huginn as huginn_pkg

    saved_autoloop = sys.modules.get("huginn.autoloop")
    saved_reproduce = sys.modules.get("huginn.autoloop.reproduce")
    saved_attr = getattr(huginn_pkg, "autoloop", None)

    stub = types.ModuleType("huginn.autoloop")
    stub.__path__ = []  # 命名空间包标记: 阻断 import 系统走真实 __init__
    stub.reproduce = rp
    sys.modules["huginn.autoloop"] = stub
    sys.modules["huginn.autoloop.reproduce"] = rp
    huginn_pkg.autoloop = stub

    cli_spec = importlib.util.spec_from_file_location(
        "huginn_cli_reproduce_test", _AGENT / "huginn" / "cli" / "reproduce.py"
    )
    mod = importlib.util.module_from_spec(cli_spec)
    try:
        cli_spec.loader.exec_module(mod)
        yield mod
    finally:
        for key, old in (
            ("huginn.autoloop", saved_autoloop),
            ("huginn.autoloop.reproduce", saved_reproduce),
        ):
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
        if saved_attr is None:
            if getattr(huginn_pkg, "autoloop", None) is stub:
                del huginn_pkg.autoloop
        else:
            huginn_pkg.autoloop = saved_attr


# ── numeric_diff 逐字段判据 ───────────────────────────────────────────────


def test_numeric_diff_equal_is_empty():
    recorded = {"a": 1.0, "b": [1, 2.5], "c": "text", "d": True, "e": None}
    assert rp.numeric_diff(recorded, dict(recorded)) == []


def test_numeric_diff_tolerance_and_atol():
    assert rp.numeric_diff(1.0, 1.0 + 5e-10) == []  # 相对容差内
    assert rp.numeric_diff(0.0, 1e-13) == []  # 双零近旁: atol 兜底
    diffs = rp.numeric_diff(1.0, 1.1)
    assert len(diffs) == 1
    assert abs(diffs[0]["rel_err"] - 1.0 / 11.0) < 1e-12
    assert len(rp.numeric_diff(0.0, 1e-6)) == 1  # 超出 atol → 显式报漂移


def test_numeric_diff_bool_is_not_int():
    assert rp.numeric_diff(True, 1) == [
        {"field": "result", "recorded": True, "rerun": 1, "reason": "bool_mismatch"}
    ]
    assert rp.numeric_diff(1, True) != []
    assert rp.numeric_diff(True, True) == []


def test_numeric_diff_missing_keys_are_visible():
    diffs = rp.numeric_diff({"a": 1}, {"a": 1, "b": 2})
    assert diffs[0]["field"] == "result.b"
    assert diffs[0]["reason"] == "missing_in_recorded"
    reverse = rp.numeric_diff({"a": 1, "b": 2}, {"a": 1})
    assert reverse[0]["reason"] == "missing_in_rerun"


def test_numeric_diff_length_and_nested_paths():
    diffs = rp.numeric_diff([1, 2], [1, 2, 3])
    assert diffs[0]["reason"] == "length_mismatch"
    nested = rp.numeric_diff({"x": [1, {"y": 2.5}]}, {"x": [1, {"y": 2.6}]})
    assert nested[0]["field"] == "result.x[1].y"


def test_numeric_diff_type_mismatch():
    diffs = rp.numeric_diff("1", 1)
    assert diffs[0]["reason"] == "value_mismatch"
    # 数值面有意宽容: int 与 float 数值相等不报 (同一判据, 不另造类型噪声)
    assert rp.numeric_diff(1, 1.0) == []


def test_numeric_diff_caps_runaway_report():
    a = {f"k{i}": float(i) for i in range(60)}
    b = {f"k{i}": float(i) + 1.0 for i in range(60)}
    assert len(rp.numeric_diff(a, b)) == 50  # _MAX_DIFFS 截断防巨 diff


# ── reproduce_snapshot 四态 ───────────────────────────────────────────────


def test_snapshot_reproduced_field_diff():
    report = rp.reproduce_snapshot(
        {"tool_name": "xrd_tool", "input_params": {"x": 2.0}, "output": {"y": 4.0}},
        runner=lambda name, params: {"y": params["x"] ** 2},
    )
    assert report["status"] == "reproduced"
    assert report["mode"] == "field_diff"
    assert report["n_diffs"] == 0


def test_snapshot_drifted_reports_field_level_diff():
    report = rp.reproduce_snapshot(
        {"tool_name": "xrd_tool", "input_params": {"x": 2.0}, "output": {"y": 4.0}},
        runner=lambda name, params: {"y": 5.0},
    )
    assert report["status"] == "drifted"
    assert report["n_diffs"] == 1
    d = report["diffs"][0]
    assert d["field"] == "result.y"
    assert d["recorded"] == 4.0 and d["rerun"] == 5.0
    assert abs(d["rel_err"] - 0.2) < 1e-9


def test_snapshot_runner_failure_converges_to_error():
    def _boom(name, params):
        raise RuntimeError("tool exploded")

    report = rp.reproduce_snapshot(
        {"tool_name": "xrd_tool", "input_params": {}, "output": {"y": 1.0}},
        runner=_boom,
    )
    assert report["status"] == "error"
    assert "rerun failed" in report["error"]


def test_snapshot_without_tool_name_is_error():
    report = rp.reproduce_snapshot({"input_params": {}})
    assert report["status"] == "error"
    assert "tool_name" in report["error"]


def test_snapshot_without_reference_is_error():
    report = rp.reproduce_snapshot(
        {"tool_name": "xrd_tool", "input_params": {}},
        runner=lambda name, params: {"y": 1.0},
    )
    assert report["status"] == "error"
    assert "没有可比对的参考结果" in report["error"]


def test_llm_step_is_explicitly_not_reproducible():
    def _never_called(name, params):
        raise AssertionError("LLM 步骤不得触达 runner (须先于重跑短路)")

    report = rp.reproduce_snapshot(
        {"tool_name": "literature_search", "input_params": {"q": "x"},
         "output": {"citations": []}},
        runner=_never_called,
    )
    assert report["status"] == "not_reproducible"
    assert report["mode"] is None
    assert report["n_diffs"] == 0
    assert "LLM" in report["reason"]


def test_hash_mode_reuses_provenance_capture():
    """快照无逐字段 output 时退化为 hash 口径, 判据复用 provenance.capture."""
    try:
        from huginn.provenance import capture
    except Exception as exc:  # 裸环境降级: CI 全量环境必可导入
        pytest.skip(f"provenance 栈在本环境不可导入: {exc}")
    snapshot = capture("t_hashmode", {"x": 1.0}, output={"a": 1.0}).to_dict()
    snapshot.pop("output", None)  # 强制 hash 口径 (有 output 则 field_diff 优先)
    assert snapshot.get("output_hash")

    ok = rp.reproduce_snapshot(
        snapshot, runner=lambda name, params: {"a": 1.0}
    )
    assert ok["mode"] == "hash"
    assert ok["status"] == "reproduced"

    drifted = rp.reproduce_snapshot(
        snapshot, runner=lambda name, params: {"a": 2.0}
    )
    assert drifted["status"] == "drifted"
    assert drifted["diffs"][0]["reason"].startswith("hash_mismatch")


# ── 名单棘轮 (只增不减; 增删须同步本文件) ─────────────────────────────────


def test_non_deterministic_marker_ratchet():
    """live 名单必须与冻结副本逐项相等 —— 改名单而不改测试 = 红 (可见决策)."""
    assert rp.NON_DETERMINISTIC_TOOL_MARKERS == _EXPECTED_MARKERS


@pytest.mark.parametrize(
    "name,expected",
    [
        ("literature_search", True),
        ("Deep_Think", True),  # 大小写不敏感
        ("hypothesis_generator", True),
        ("nuwa_distill", True),
        ("xrd_tool", False),
        ("numerical_tool", False),
        ("reproduce_diff", False),  # 不得误伤自身: "repro" 不含任何 marker
        ("vasp_tool", False),
    ],
)
def test_is_non_deterministic(name, expected):
    assert rp.is_non_deterministic(name) is expected


# ── reproduce_record 汇总 ──────────────────────────────────────────────────


def test_record_mixed_statuses_aggregate():
    chain = [
        {"tool_name": "xrd_tool", "input_params": {"x": 1}, "output": {"p": 1.0}},
        {"tool_name": "literature_search", "input_params": {}},
        {"tool_name": "numerical_tool", "input_params": {"x": 2}, "output": {"p": 2.0}},
    ]
    report = rp.reproduce_record(
        {"run_id": "r1", "tool_chain": chain},
        runner=lambda name, params: {"p": float(params["x"])},
    )
    assert report["run_id"] == "r1"
    assert report["n_snapshots"] == 3
    assert report["counts"] == {"reproduced": 2, "not_reproducible": 1}
    assert report["all_reproduced"] is False


def test_record_all_good_and_empty_chain():
    ok = rp.reproduce_record(
        {"run_id": "r", "tool_chain": [
            {"tool_name": "xrd_tool", "input_params": {"x": 1}, "output": {"p": 1.0}},
        ]},
        runner=lambda name, params: {"p": 1.0},
    )
    assert ok["all_reproduced"] is True
    empty = rp.reproduce_record({"run_id": "r", "tool_chain": []})
    assert empty["n_snapshots"] == 0
    assert empty["all_reproduced"] is False  # 空链 ≠ 全部复现 (防假绿)


# ── CLI 入口 (按路径加载 + sys.modules 恢复) ──────────────────────────────


def test_cli_not_reproducible_exits_1(cli_mod, tmp_path, capsys):
    snap = tmp_path / "lit.json"
    snap.write_text(
        json.dumps(
            {"tool_name": "literature_search", "input_params": {"q": "x"},
             "output_hash": "deadbeef"}
        ),
        encoding="utf-8",
    )
    rc = cli_mod.main(["--snapshot", str(snap)])
    assert rc == 1
    assert "not_reproducible" in capsys.readouterr().out


def test_cli_empty_file_is_not_fake_green(cli_mod, tmp_path, capsys):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    assert cli_mod.main(["--record", str(p)]) == 2
    assert "没有可重跑" in capsys.readouterr().err


def test_cli_missing_file_exits_2(cli_mod, tmp_path):
    assert cli_mod.main(["--snapshot", str(tmp_path / "ghost.json")]) == 2


def test_cli_bad_jsonl_line_reports_line_number(cli_mod, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"tool_name": "xrd_tool"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="不是合法 JSON"):
        cli_mod.main(["--record", str(p)])


def test_cli_record_summary_counts_and_exit_1(cli_mod, tmp_path, capsys):
    rec = {
        "run_id": "run-42",
        "tool_chain": [
            {"tool_name": "literature_search", "input_params": {}},
            {"tool_name": "deep_think_tool", "input_params": {}},
        ],
    }
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    rc = cli_mod.main(["--record", str(p)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "run-42" in out
    assert "not_reproducible=2" in out


def test_cli_all_reproduced_exits_0(cli_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli_mod,
        "reproduce_snapshot",
        lambda item, tolerance=1e-9: {
            "tool_name": item.get("tool_name", "?"),
            "status": "reproduced",
            "mode": "field_diff",
            "n_diffs": 0,
            "diffs": [],
            "tolerance": tolerance,
        },
    )
    monkeypatch.setattr(
        cli_mod,
        "reproduce_record",
        lambda item, tolerance=1e-9: {
            "run_id": item.get("run_id", ""),
            "n_snapshots": 1,
            "results": [{
                "tool_name": "xrd_tool",
                "status": "reproduced",
                "mode": "field_diff",
                "n_diffs": 0,
                "diffs": [],
            }],
            "counts": {"reproduced": 1},
            "all_reproduced": True,
        },
    )
    snap = tmp_path / "snap.json"
    snap.write_text(
        json.dumps({"tool_name": "xrd_tool", "input_params": {}}), encoding="utf-8"
    )
    rec = tmp_path / "rec.json"
    rec.write_text(
        json.dumps({"run_id": "r", "tool_chain": [
            {"tool_name": "xrd_tool", "input_params": {}},
        ]}),
        encoding="utf-8",
    )
    assert cli_mod.main(["--snapshot", str(snap)]) == 0
    assert cli_mod.main(["--record", str(rec)]) == 0


def test_cli_json_flag_emits_parseable_reports(cli_mod, tmp_path, capsys):
    p = tmp_path / "list.json"
    p.write_text(
        json.dumps([{"tool_name": "literature_search", "input_params": {}}]),
        encoding="utf-8",
    )
    rc = cli_mod.main(["--snapshot", str(p), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload[0]["status"] == "not_reproducible"


def test_cli_is_record_shape_split(cli_mod):
    assert cli_mod._is_record({"tool_chain": [], "run_id": "r"}) is True
    assert cli_mod._is_record({"tool_name": "x", "tool_chain": []}) is False
    assert cli_mod._is_record({"tool_name": "x"}) is False
