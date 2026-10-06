"""replay_audit 控制面 trace 计数 + 判词可见性 + 执行计数回归.

背景: replay_audit 是"离线重放历史轨迹, 核对无进展轮是否**可观测**"的工具
(见 :mod:`huginn.autoloop.replay_audit`). A2 后线上终止出口只留挂钟/目标达成,
故"无进展是否可观测"的实测证据 = run.log 里落盘的 ``control_trace`` 计数.

两个已修/须锁的坑:
1. **计数**: 本轮新登记的 ``progress_invariant`` / ``llm_unavailable`` /
   ``branch_slice_skip`` / ``code_lab_slice_skip`` 必须能被 ``scan_runlog`` 抓进
   ``exits.control_traces`` —— 抓不到就等于观测面失明.
2. **判词可见性**: 原 ``_verdict`` 只在 ``soft>0`` 时才打印 control_trace 行;
   零进展但无软动作的轮 (如 run80 只发 ``branch_slice_skip``) 会被整段吞掉,
   证据只在 ``--json`` 里可见 = 判词失效. 本测试锁"有任何 trace 就要报".
"""
from __future__ import annotations

import json
from pathlib import Path

from huginn.autoloop import replay_audit as ra

# 四类本轮新登记机制的真实 run.log 行形态 (字段顺序与线上 emit 一致).
_TRACE_LINES = [
    "control_trace name=branch_slice_skip iteration=1 "
    "evidence=slice=layer2 (return layer1): budget<212s action=skip",
    "control_trace name=code_lab_slice_skip iteration=3 "
    "evidence=slice=repair#1: budget<max(min,180s) remaining=20s action=skip",
    "control_trace name=progress_invariant iteration=7 "
    "evidence=window=4 tail=hypothesize,plan,hypothesize,plan action=force_route",
    "control_trace name=llm_unavailable iteration=9 "
    "evidence=action=hypothesize reason=transient_empty action=retry_in_place",
    # 第二次 branch_slice_skip: 验证"按名累加"而非"只记存在"
    "control_trace name=branch_slice_skip iteration=1 "
    "evidence=slice=layer1_value: budget<212s action=skip",
]


def _make_run(tmp_path: Path) -> str:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("\n".join(_TRACE_LINES) + "\n", encoding="utf-8")
    return str(run_dir)


def test_scan_runlog_counts_new_mechanism_traces(tmp_path: Path) -> None:
    """scan_runlog 按名累加四类新 trace (bad 名抓不到 = 观测失明)."""
    ev = ra.scan_runlog(_make_run(tmp_path))
    assert ev["traces"]["branch_slice_skip"] == 2, ev["traces"]
    assert ev["traces"]["code_lab_slice_skip"] == 1, ev["traces"]
    assert ev["traces"]["progress_invariant"] == 1, ev["traces"]
    assert ev["traces"]["llm_unavailable"] == 1, ev["traces"]


def test_audit_surfaces_traces_in_exits(tmp_path: Path) -> None:
    """audit 把计数带进 exits.control_traces (JSON 消费面)."""
    a = ra.audit(_make_run(tmp_path))
    ct = a["exits"]["control_traces"]
    assert ct.get("branch_slice_skip") == 2, ct
    assert ct.get("code_lab_slice_skip") == 1, ct
    assert ct.get("progress_invariant") == 1, ct
    assert ct.get("llm_unavailable") == 1, ct


def test_verdict_reports_traces_without_soft_actions(tmp_path: Path) -> None:
    """零进展 / 无软动作的轮也要在判词里露出在盘 trace (原 soft>0 门会吞掉)."""
    a = ra.audit(_make_run(tmp_path))
    # 前提: 该合成轨迹没有任何软动作 —— 正是会触发原门控 bug 的形态.
    assert a["exits"]["soft"] == 0, a["exits"]
    lines = ra._verdict(a)
    joined = "\n".join(lines)
    assert "落盘 control_trace" in joined, joined
    assert "branch_slice_skip×2" in joined, joined
    assert "progress_invariant×1" in joined, joined


# ── 执行计数: 不依赖 HUGINN_EXEC_ROUTE_DEBUG ──────────────────────────
#
# 坑: run.log 的 ``[code-lab-run]`` 行只在调试开关打开时输出(engine_act
# ``_run_code_lab``) ⇒ 正常 run 里 ``scan_runlog`` 恒得 execs=0, replay_audit
# 误报"执行=0"(run85 实测: 有真实 execute 阶段却报 0). 回退源 = episodic 的
# ``execute`` 动作(总落盘, 带 exec_ok).

def _write_episodic(run_dir: Path, actions: list[tuple[str, bool]]) -> None:
    shard = run_dir / ".huginn" / "memory" / "episodic" / "loop_x"
    shard.mkdir(parents=True)
    lines = []
    for i, (action, ok) in enumerate(actions, start=1):
        entry = {"iter": i, "action": action, "exec_ok": ok, "surprise": 0.0}
        lines.append(json.dumps({"iter": i, "ts": float(i), "entry": entry}))
    (shard / "shard_0_99.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_episodic_entries(run_dir: Path, entries: list[dict]) -> None:
    shard = run_dir / ".huginn" / "memory" / "episodic" / "loop_x"
    shard.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"iter": i, "ts": float(i), "entry": e})
        for i, e in enumerate(entries, start=1)
    ]
    (shard / "shard_0_99.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_episodic_exec_count_fallback_when_no_debug_lines(tmp_path: Path) -> None:
    """无调试行时, 执行数从 episodic 的 execute 动作取 (不再误报 0)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines here\n", encoding="utf-8")
    _write_episodic(run_dir, [
        ("hypothesize", False), ("plan", False),
        ("execute", True), ("validate", True),
        ("execute", False),  # 第二次 execute 未产出证据
    ])
    a = ra.audit(str(run_dir))
    assert a["execs"] == 2, a["execs"]  # 两个 execute 动作都算一次执行
    assert "执行=2" in "\n".join(ra._verdict(a))


def test_episodic_fallback_does_not_double_count_with_debug_lines(tmp_path: Path) -> None:
    """有调试行时用调试行(更细, 含重试), episodic 仅回退 ⇒ 不叠加."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    debug = [
        "[code-lab-run] res_none=False success=True nobj=3 reason=''",
        "[code-lab-run] res_none=False success=True nobj=3 reason=''",
        "[code-lab-run] res_none=False success=True nobj=3 reason=''",
    ]
    (run_dir / "run.log").write_text("\n".join(debug) + "\n", encoding="utf-8")
    _write_episodic(run_dir, [("execute", True)])
    a = ra.audit(str(run_dir))
    assert a["execs"] == 3, a["execs"]  # 调试行优先, 不叠加 episodic 的 1


# ── surprise 信号源: 不读 run.log 的自由文本 ───────────────────────────
#
# 坑: run.log 的 ``surprise=`` 全是自由文本(计划描述前缀 ``[auto-routed:
# surprise=1.00]`` / LLM 复述), 正则抓它会把"计划里字面复现"误报成路由退化.
# run65 实测: run.log 字面全 1.0, 但 episodic 秩信号有 4 个不同值 = 未退化.

def test_surprise_reads_unified_episodic_signal_not_runlog_text(tmp_path: Path) -> None:
    """run.log 满是 ``surprise=1.00`` 自由文本, 但 episodic 信号有区分 ⇒ 不报退化."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text(
        "[exec-route] mode='explore' desc[:80]='[auto-routed: surprise=1.00] foo'\n"
        "[exec-route] mode='explore' desc[:80]='... surprise=1.0 ...'\n",
        encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "hypothesize", "surprise": 1.0},
        {"action": "execute", "surprise": 0.142857},
        {"action": "validate", "surprise": 0.16},
    ])
    a = ra.audit(str(run_dir))
    assert set(a["input_frozen"]["surprise_values"]) == {1.0, 0.142857, 0.16}
    assert a["input_frozen"]["surprise_saturated"] is False
    assert "路由信号死" not in "\n".join(ra._verdict(a))


def test_surprise_saturated_only_when_signal_truly_constant(tmp_path: Path) -> None:
    """统一信号真恒定(>=2 样本)才算饱和, 报"路由信号死"."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "hypothesize", "surprise": 0.5},
        {"action": "execute", "surprise": 0.5},
    ])
    a = ra.audit(str(run_dir))
    assert a["input_frozen"]["surprise_saturated"] is True
    assert "路由信号死" in "\n".join(ra._verdict(a))


# ── 结构通道: 全 0 = 未激活(非结构域正常), 非"编码器坏" ─────────────────

def _sd(nonzero: bool) -> list[float]:
    return [1.5, 90.0] + [0.0] * 14 if nonzero else [0.0] * 16


def test_structure_all_zero_reports_unexercised_not_dead(tmp_path: Path) -> None:
    """全 0 结构描述符 ⇒ 报"未激活"(非结构域正常), 不得称"编码器坏/通道无信息"."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "hypothesize", "structure_desc": _sd(False)},
        {"action": "execute", "structure_desc": _sd(False)},
    ])
    a = ra.audit(str(run_dir))
    assert a["structure_channel_unexercised"] is True
    joined = "\n".join(ra._verdict(a))
    assert "结构通道未激活" in joined
    # 旧判词"→ 该通道无信息"(断言编码器坏)必须消失; 只留"不是编码器坏"的澄清.
    assert "该通道无信息" not in joined


def test_structure_nonzero_channel_reports_nothing(tmp_path: Path) -> None:
    """有非零结构描述符 ⇒ 通道已激活, 不报未激活."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "execute", "structure_desc": _sd(True)},
    ])
    a = ra.audit(str(run_dir))
    assert a["structure_channel_unexercised"] is False
    assert "结构通道未激活" not in "\n".join(ra._verdict(a))


# ── 输入/输出计数: 权威源 = episodic, 不依赖 HUGINN_EXEC_ROUTE_DEBUG ─────
#
# 坑: run.log 的 ``[code-lab-author] prompt_len=`` / ``[exec-route] obj_len=`` /
# ``[code-lab-run] nobj=`` 三行都在调试开关下 ⇒ 关掉时 scan_runlog 得空, "输入冻结 /
# 执行输出恒同"判定**静默失明**(run80-84 实测: 三项全空, 判词整段不触发). 权威源改
# 取 episodic 的 prompt_len / obj_len / nobj 结构化字段(每轮必然落盘).

def test_prompt_len_frozen_read_from_episodic_without_debug_lines(tmp_path: Path) -> None:
    """无调试行时, prompt_len 恒定从 episodic 取 ⇒ 冻结判定不被静默吞掉."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines here\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "execute", "prompt_len": 5379},
        {"action": "execute", "prompt_len": 5379},
        {"action": "hypothesize", "prompt_len": None},  # 未调作者: 不计入
    ])
    a = ra.audit(str(run_dir))
    assert a["input_frozen"]["prompt_len_values"] == {5379: 2}, a["input_frozen"]
    assert a["input_frozen"]["prompt_frozen"] is True
    assert "输入冻结: prompt_len 恒定" in "\n".join(ra._verdict(a))


def test_prompt_len_single_sample_not_frozen(tmp_path: Path) -> None:
    """只调过一次作者(单样本)不算冻结 —— 否则短 run 一律假阳性."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [{"action": "execute", "prompt_len": 5379}])
    a = ra.audit(str(run_dir))
    assert a["input_frozen"]["prompt_frozen"] is False
    assert "输入冻结: prompt_len" not in "\n".join(ra._verdict(a))


def test_obj_len_frozen_falls_back_to_episodic(tmp_path: Path) -> None:
    """无 prompt_len 采样时, obj_len 恒定(>=2 样本)从 episodic 取 ⇒ 报冻结."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "hypothesize", "obj_len": 509},
        {"action": "execute", "obj_len": 509},
    ])
    a = ra.audit(str(run_dir))
    assert a["input_frozen"]["goal_frozen"] is True
    assert "输入冻结: obj_len 恒定" in "\n".join(ra._verdict(a))


def test_nobj_constant_from_episodic_reports_zero_new_info(tmp_path: Path) -> None:
    """nobj 恒定从 episodic 取(不依赖调试行) ⇒ 报"执行层零新信息"."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("no debug lines\n", encoding="utf-8")
    _write_episodic_entries(run_dir, [
        {"action": "execute", "exec_ok": True, "nobj": 3},
        {"action": "execute", "exec_ok": True, "nobj": 3},
    ])
    a = ra.audit(str(run_dir))
    assert a["execs"] == 2
    assert a["nobj_distribution"] == {3: 2}, a["nobj_distribution"]
    assert "执行输出恒同" in "\n".join(ra._verdict(a))


def test_obj_len_regex_ignores_freetext_in_desc(tmp_path: Path) -> None:
    """[exec-route] 的 desc[:80] 是自由文本: 正则必须锚定真实字段位, 不被字面假命中."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text(
        "[exec-route] mode='explore' desc[:80]='plan literally quotes obj_len=999 x' "
        "obj_len=509 is_exp_desc=True is_exp_obj=True is_det_desc=False\n",
        encoding="utf-8")
    ev = ra.scan_runlog(str(run_dir))
    assert dict(ev["obj_lens"]) == {509: 1}, ev["obj_lens"]


# ── 反例搜索: 机制 WARNING 可观测 (旧 INFO 被静默吞掉) ───────────────────

def test_counterexample_hunt_emits_warning() -> None:
    """_trigger_counterexample_hunt 必须发 WARNING (run.log 只捕获 WARNING+)."""
    import logging

    from huginn.autoloop.cognitive_loop import CognitiveRunner

    class _Stub:
        _current_hyp_id_for_plan = None
        _force_imaginate = False
        _speculator_hint = ""
        memory = None

    stub = _Stub()
    records: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    _log = logging.getLogger("huginn.autoloop.cognitive_loop")
    _cap = _Cap(level=logging.INFO)
    _log.addHandler(_cap)
    _old = _log.level
    _log.setLevel(logging.INFO)
    try:
        CognitiveRunner._trigger_counterexample_hunt(stub)
    finally:
        _log.removeHandler(_cap)
        _log.setLevel(_old)

    assert stub._force_imaginate is True
    assert any(
        r.levelno == logging.WARNING and "counterexample hunt triggered" in r.getMessage()
        for r in records
    ), [(r.levelname, r.getMessage()) for r in records]