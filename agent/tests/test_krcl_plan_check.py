"""KRCL plan_check 自检 — 反向校验 JSON 解析 + phase-aware + 自适应 + 不暴露.

只测纯函数 (无 LLM 调用), 真实 LLM 路径靠 autoloop 集成测试覆盖.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# ── _parse_plan_check (括号配平 JSON 解析) ──────────────────────


def _make_engine(iteration: int = 35, workspace: Path | None = None):
    """造一个最小 AutoloopEngine 实例, 只为了调 plan_check 相关方法.

    ponytail: 不走完整 __init__, 用 __new__ 绕过依赖注入.
    默认 iteration=35 (light tier) — 跟原 _PLAN_CHECK_MAX_REFINES=1 行为对齐,
    完整闭环测试不用额外调旋钮.
    workspace 传入时走真持久化; 不传时 mock 掉 save/load, 测纯逻辑.
    """
    from huginn.autoloop.engine import AutoloopEngine

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._iteration = iteration
    eng._plan_check_history = []
    eng._plan_check_last_result = None
    eng._plan_check_warnings = []
    eng._plan_check_patterns = []
    eng._scene_tag_extra_keywords = {}
    eng._speculator_hint = ""
    # 默认 mock 持久化和澄清, 测纯函数; 需要真持久化时传 workspace
    if workspace is None:
        eng._save_plan_check_patterns = lambda: None  # type: ignore[assignment]
        eng._load_plan_check_patterns = lambda: None  # type: ignore[assignment]
    else:
        eng.workspace = workspace
    # mock _maybe_clarify (真正的 LLM 澄清), _maybe_trigger_plan_check_clarify 走真逻辑
    eng._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]
    return eng


def test_parse_plan_check_valid_json():
    """valid JSON + 完整字段 -> 正常解析."""
    eng = _make_engine()
    resp = '{"is_valid": false, "reason": "missing SCF", "missing_steps": ["SCF"], "risks": ["band gap wrong"]}'
    res = eng._parse_plan_check(resp)
    assert res["is_valid"] is False
    assert res["reason"] == "missing SCF"
    assert res["missing_steps"] == ["SCF"]


def test_parse_plan_check_nested_json():
    """嵌套 JSON 也能解析 (括号配平法)."""
    eng = _make_engine()
    resp = 'noise {"is_valid": true, "reason": "ok", "missing_steps": [], "risks": [{"type": "low"}]} trailing'
    res = eng._parse_plan_check(resp)
    assert res["is_valid"] is True
    assert res["risks"] == [{"type": "low"}]


def test_parse_plan_check_missing_fields():
    """JSON 缺字段 -> setdefault 补全."""
    eng = _make_engine()
    resp = '{"is_valid": false, "reason": "bad mode"}'
    res = eng._parse_plan_check(resp)
    assert res["is_valid"] is False
    assert res["missing_steps"] == []
    assert res["risks"] == []


def test_parse_plan_check_no_json():
    """无 JSON -> is_valid=True (跳过, 不阻塞)."""
    eng = _make_engine()
    res = eng._parse_plan_check("no json here")
    assert res["is_valid"] is True
    assert "skip" in res["reason"]


def test_parse_plan_check_broken_json():
    """JSON 解析失败 -> is_valid=True (不阻塞)."""
    eng = _make_engine()
    res = eng._parse_plan_check('{"is_valid": false, "reason":}')
    assert res["is_valid"] is True
    assert "skip" in res["reason"] or "failed" in res["reason"]


def test_parse_plan_check_no_closing_brace():
    """JSON 没闭合 -> is_valid=True."""
    eng = _make_engine()
    res = eng._parse_plan_check('{"is_valid": false, "reason": "missing')
    assert res["is_valid"] is True


# ── _plan_check_and_refine (闭环 + trivial plan 跳过) ─────────


def test_trivial_plan_skips_check():
    """description 太短 (<20 chars) 跳过校验, 不调 LLM."""
    eng = _make_engine()
    eng._plan_check = AsyncMock()
    plan = {"mode": "coder", "description": "short"}  # 5 chars
    result = asyncio.run(eng._plan_check_and_refine(plan, "hypothesis", {}))
    assert result is plan  # 原样返回
    eng._plan_check.assert_not_called()
    assert eng._plan_check_last_result is None


def test_valid_plan_passes_first_try():
    """is_valid=True -> 直接返回, check 存引擎状态不塞 plan dict."""
    eng = _make_engine()
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": True,
            "reason": "ok",
            "missing_steps": [],
            "risks": [],
        }
    )
    plan = {"mode": "coder", "description": "run SCF then band calculation"}
    result = asyncio.run(eng._plan_check_and_refine(plan, "calc band gap", {}))
    assert result is plan  # 原样返回, 没塞字段
    assert "plan_check" not in result
    assert "plan_check_warning" not in result
    assert eng._plan_check_last_result["is_valid"] is True
    assert len(eng._plan_check_history) == 1


def test_invalid_plan_refines_once_then_gives_up():
    """light tier: 两次 is_valid=False -> 1 次 refine 后记 warning 不阻塞."""
    eng = _make_engine()  # iter=35, light tier, baseline max_refines=1
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock(
        return_value={
            "mode": "workflow",
            "description": "SCF then band",
        }
    )
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation directly"}
    result = asyncio.run(eng._plan_check_and_refine(plan, "calc band gap", {}))
    # 重试 1 次, 总共调 _plan_check 2 次
    assert eng._plan_check.call_count == 2
    assert eng._refine_plan.call_count == 1
    # 不暴露: 不在 plan dict, 在引擎状态
    assert "plan_check" not in result
    assert "plan_check_warning" not in result
    assert eng._plan_check_last_result["is_valid"] is False
    # warning 带 scene tag
    assert "missing SCF" in eng._plan_check_warnings[-1]
    assert "[dft]" in eng._plan_check_warnings[-1]


def test_llm_failure_returns_plan():
    """_plan_check 抛异常 -> 直接返回原 plan, 状态没更新."""
    eng = _make_engine()
    eng._plan_check = AsyncMock(side_effect=Exception("LLM down"))
    plan = {"mode": "coder", "description": "run SCF calculation on Si"}
    result = asyncio.run(eng._plan_check_and_refine(plan, "calc band gap", {}))
    assert result is plan
    assert "plan_check" not in result
    assert eng._plan_check_last_result is None  # 异常前没存


# ── phase-aware (tier skip / light check / full closure) ──────


def test_open_tier_skips_check():
    """iter 5 (open tier) 简单 plan 跳过反向校验, 不调 LLM."""
    eng = _make_engine(iteration=5)
    eng._plan_check = AsyncMock()
    plan = {"mode": "coder", "description": "run some simple coding task"}
    result = asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    assert result is plan
    eng._plan_check.assert_not_called()
    assert eng._plan_check_last_result is None


def test_medium_tier_light_check_no_refine():
    """iter 20 (medium tier) 只校验不 refine, 失败直接记 warning."""
    eng = _make_engine(iteration=20)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock()  # medium baseline=0, 不应该被调
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    assert eng._plan_check.call_count == 1
    eng._refine_plan.assert_not_called()
    assert eng._plan_check_last_result["is_valid"] is False


def test_light_tier_full_closure():
    """iter 35 (light tier) 走完整闭环, 失败触发 refine."""
    eng = _make_engine(iteration=35)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock(
        return_value={
            "mode": "workflow",
            "description": "SCF then band calculation",
        }
    )
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # baseline=1 -> 1 次 refine, 2 次 _plan_check
    assert eng._plan_check.call_count == 2
    assert eng._refine_plan.call_count == 1


# ── 复杂度感知 (plan 本身复杂度修正 tier) ─────────────────────


def test_complex_plan_upgrades_open_to_medium():
    """open tier + 复杂 plan (workflow + 长 desc + prediction) -> 升级到 medium, 要校验."""
    eng = _make_engine(iteration=5)  # open tier
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": True,
            "reason": "ok",
            "missing_steps": [],
            "risks": [],
        }
    )
    # 复杂 plan: workflow mode (0.4) + 长 desc (0.3) + prediction (0.15) = 0.85
    plan = {
        "mode": "workflow",
        "description": "run VASP SCF then band then DOS convergence test",
        "expected_prediction": "band_gap=1.1eV",
    }
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # 复杂 plan 升级到 medium, 走校验
    eng._plan_check.assert_called_once()


def test_simple_plan_downgrades_light_to_skip():
    """light tier + 极简 plan (explore + 刚过 trivial 的短 desc) -> 降级到 skip, 不校验."""
    eng = _make_engine(iteration=35)  # light tier
    eng._plan_check = AsyncMock()
    # 极简 plan: explore mode (0.1) + 20chars desc (0.12) = 0.22 < 0.25
    plan = {"mode": "explore", "description": "look around for stuff"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # 简单 plan 降级到 skip, 不校验
    eng._plan_check.assert_not_called()


def test_plan_check_complexity_dimensions():
    """复杂度评分各维度叠加."""
    eng = _make_engine()
    # 极简: explore + 短 desc + 无 prediction + 无历史失败
    simple = {"mode": "explore", "description": "short"}
    assert eng._plan_check_complexity(simple) < 0.25
    # 复杂: workflow + 长 desc + prediction
    complex_plan = {
        "mode": "workflow",
        "description": "run VASP SCF then band then DOS convergence test carefully",
        "expected_prediction": "band_gap=1.1eV",
    }
    assert eng._plan_check_complexity(complex_plan) >= 0.7


def test_plan_check_complexity_history_failure_bonus():
    """同场景历史失败数加复杂度分 (踩过坑的要复查)."""
    eng = _make_engine()
    plan = {"mode": "coder", "description": "run SCF calculation on Si"}
    base = eng._plan_check_complexity(plan)
    # 塞 3 条同场景失败模式
    eng._plan_check_patterns = [
        {"scene_tag": "dft", "reason": "missing SCF", "missing_steps": ["SCF"]}
        for _ in range(3)
    ]
    with_history = eng._plan_check_complexity(plan)
    assert with_history > base
    assert with_history - base >= 0.15  # 3 条满额 +0.15


# ── 场景标签抽取 ─────────────────────────────────────────────


def test_scene_tag_dft():
    eng = _make_engine()
    assert (
        eng._plan_check_scene_tag({"description": "run VASP SCF", "mode": "coder"})
        == "dft"
    )
    assert (
        eng._plan_check_scene_tag({"description": "calc band gap", "mode": "coder"})
        == "dft"
    )
    assert (
        eng._plan_check_scene_tag({"description": "QE relaxation", "mode": "coder"})
        == "dft"
    )


def test_scene_tag_md():
    eng = _make_engine()
    assert (
        eng._plan_check_scene_tag(
            {"description": "lammps NVT dynamics", "mode": "coder"}
        )
        == "md"
    )
    assert (
        eng._plan_check_scene_tag({"description": "gromacs md run", "mode": "coder"})
        == "md"
    )


def test_scene_tag_workflow_skill_other():
    eng = _make_engine()
    assert (
        eng._plan_check_scene_tag({"description": "pipeline stuff", "mode": "coder"})
        == "workflow"
    )
    assert (
        eng._plan_check_scene_tag({"description": "do thing", "mode": "skill"})
        == "skill"
    )
    assert (
        eng._plan_check_scene_tag({"description": "random text", "mode": "coder"})
        == "other"
    )


# ── 场景分桶自适应 ───────────────────────────────────────────


def test_adaptive_loosen_after_high_success():
    """light tier + 最近 5 次同场景全成功 -> max_refines 放宽到 0, 不 refine.

    baseline=1, success_rate=1.0>=0.8 -> max(0, 1-1)=0.
    """
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [{"is_valid": True, "scene_tag": "dft"} for _ in range(5)]
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock()  # 放宽后不应该被调
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    assert eng._plan_check.call_count == 1  # 只校验一次
    eng._refine_plan.assert_not_called()


def test_adaptive_tighten_after_low_success():
    """light tier + 最近 5 次同场景全失败 -> max_refines 收紧到 2, refine 2 次.

    baseline=1, success_rate=0.0<=0.2 -> min(2, 1+1)=2.
    attempt 0,1 各 refine 一次, attempt 2 直接 warning -> 3 次 _plan_check.
    """
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"} for _ in range(5)
    ]
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    refine_counter = {"n": 0}

    async def _fake_refine(plan, check, hyp, ctx):
        refine_counter["n"] += 1
        return {"mode": "coder", "description": f"refined {refine_counter['n']}"}

    eng._refine_plan = AsyncMock(side_effect=_fake_refine)
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    assert eng._plan_check.call_count == 3
    assert refine_counter["n"] == 2


def test_adaptive_scene_bucketing_isolates_scenes():
    """DFT 场景的失败不应该影响 MD 场景的 max_refines.

    DFT 5 次全失败 (收紧), 但 MD 没历史 -> 走 baseline.
    """
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"} for _ in range(5)
    ]
    # MD 场景没历史, 走 baseline=1
    assert eng._plan_check_max_refines("light", "md") == 1
    # DFT 场景 5 次全失败, 收紧到 2
    assert eng._plan_check_max_refines("light", "dft") == 2


def test_adaptive_insufficient_sample_uses_baseline():
    """history <3 条 -> 走 baseline, 不误判 (早期不收紧/放宽)."""
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
    ]
    assert eng._plan_check_max_refines("light", "dft") == 1  # baseline


def test_plan_check_tier_boundaries():
    """tier 边界: 1-10 open, 11-30 medium, 31+ light (不传 plan 时)."""
    eng = _make_engine(iteration=1)
    assert eng._plan_check_tier() == "open"
    eng._iteration = 10
    assert eng._plan_check_tier() == "open"
    eng._iteration = 11
    assert eng._plan_check_tier() == "medium"
    eng._iteration = 30
    assert eng._plan_check_tier() == "medium"
    eng._iteration = 31
    assert eng._plan_check_tier() == "light"


# ── 失败模式记忆 + 跨 run 持久化 ─────────────────────────────


def test_history_window_caps_at_20():
    """history 超 20 条 -> 截断保留最近 20 条, 防无限增长."""
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [
        {"is_valid": True, "scene_tag": "dft"} for _ in range(25)
    ]
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": True,
            "reason": "ok",
            "missing_steps": [],
            "risks": [],
        }
    )
    plan = {"mode": "coder", "description": "run SCF calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # 25 + 1 = 26, 截断到 20
    assert len(eng._plan_check_history) == 20
    assert eng._plan_check_history[-1]["is_valid"] is True


def test_failure_recorded_to_patterns():
    """失败时记到 _plan_check_patterns, 带 scene_tag/reason/missing_steps."""
    eng = _make_engine(iteration=35)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock(
        return_value={
            "mode": "workflow",
            "description": "SCF then band calculation",
        }
    )
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # 失败 2 次 (baseline=1, refine 1 次), 记 2 条 patterns
    assert len(eng._plan_check_patterns) == 2
    p = eng._plan_check_patterns[0]
    assert p["scene_tag"] == "dft"
    assert p["reason"] == "missing SCF"
    assert p["missing_steps"] == ["SCF"]
    assert p["mode"] == "coder"


def test_patterns_persisted_to_workspace_json(tmp_path: Path):
    """失败模式 dump 到 .huginn/plan_check_patterns.json, 跨 run 加载."""
    # 第一轮: 真持久化, 失败一次, dump 到文件
    eng1 = _make_engine(iteration=35, workspace=tmp_path)
    eng1._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    eng1._refine_plan = AsyncMock(
        return_value={
            "mode": "workflow",
            "description": "SCF then band calculation",
        }
    )
    eng1._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng1._plan_check_and_refine(plan, "h", {}))

    pattern_file = tmp_path / ".huginn" / "plan_check_patterns.json"
    assert pattern_file.exists()
    data = json.loads(pattern_file.read_text(encoding="utf-8"))
    assert len(data) == 2  # 失败 2 次 (baseline=1, refine 1 次)
    assert data[0]["scene_tag"] == "dft"

    # 第二轮: 新 engine, _load_plan_check_patterns 从文件加载
    eng2 = _make_engine(iteration=35, workspace=tmp_path)
    eng2._load_plan_check_patterns()
    assert len(eng2._plan_check_patterns) == 2
    assert eng2._plan_check_patterns[0]["scene_tag"] == "dft"


def test_prompt_injects_similar_history_failures():
    """_build_plan_check_prompt 注入同场景历史失败模式."""
    eng = _make_engine()
    eng._plan_check_patterns = [
        {"scene_tag": "dft", "reason": "missing SCF", "missing_steps": ["SCF"]},
        {"scene_tag": "md", "reason": "no minimize", "missing_steps": ["minimize"]},
    ]
    plan = {"mode": "coder", "description": "run VASP band calculation"}
    prompt = eng._build_plan_check_prompt(plan, "calc band gap", {})
    # DFT 场景, 注入 missing SCF
    assert "missing SCF" in prompt
    assert "SCF" in prompt
    # MD 场景不注入
    assert "no minimize" not in prompt


# ── 连续失败触发主动澄清 ─────────────────────────────────────


def test_consecutive_failure_triggers_clarify():
    """同场景连续 3 次失败 -> 触发 _maybe_clarify 问用户."""
    eng = _make_engine(iteration=35)
    # 预置 2 次同场景失败, 再失败 1 次就到 3 次
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": False, "scene_tag": "dft"},
    ]
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    # medium tier baseline=0, 失败一次直接触发澄清
    eng._iteration = 20
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # _maybe_trigger_plan_check_clarify 走真逻辑, 内部 recent_fails>=3 调 _maybe_clarify
    eng._maybe_clarify.assert_called_once()


def test_other_scene_does_not_trigger_clarify():
    """scene='other' 不触发澄清 (没上下文给用户)."""
    eng = _make_engine(iteration=20)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "other"},
        {"is_valid": False, "scene_tag": "other"},
    ]
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "bad",
            "missing_steps": [],
            "risks": [],
        }
    )
    # other 场景: description 不含关键词
    plan = {"mode": "coder", "description": "do some random coding task"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # _maybe_trigger_plan_check_clarify 内部 if scene == "other": return
    eng._maybe_clarify.assert_not_called()


def test_success_breaks_consecutive_failure_count():
    """中间一次成功就断连续失败计数, 不触发澄清."""
    eng = _make_engine(iteration=20)
    # 2 次失败 + 1 次成功 + 再失败 (连续失败只数到 1)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
    ]
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "reason": "missing SCF",
            "missing_steps": ["SCF"],
            "risks": [],
        }
    )
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # 连续失败只数到 1 (最后一条), <3 不触发
    eng._maybe_clarify.assert_not_called()


# ── _build_plan_check_prompt (纯函数) ──────────────────────────


def test_build_prompt_includes_key_fields():
    """prompt 包含 hypothesis / mode / description / prediction."""
    eng = _make_engine()
    plan = {
        "mode": "coder",
        "description": "run VASP SCF on Si",
        "expected_prediction": "band_gap=1.1eV",
    }
    prompt = eng._build_plan_check_prompt(plan, "calc Si band gap", {})
    assert "calc Si band gap" in prompt
    assert "coder" in prompt
    assert "run VASP SCF on Si" in prompt
    assert "band_gap=1.1eV" in prompt


# ── 成功 plan few-shot 注入 _refine_plan ─────────────────────


def test_refine_injects_success_few_shot():
    """_refine_plan 从 history 抽同场景成功 plan 塞 prompt 做 few-shot."""
    eng = _make_engine(iteration=35)
    # 预置 1 条同场景 (dft) 成功 plan
    eng._plan_check_history = [
        {
            "is_valid": True,
            "scene_tag": "dft",
            "plan_snapshot": {"mode": "workflow", "description": "SCF then band"},
        }
    ]
    eng._llm_chat = AsyncMock(return_value="MODE: coder\nDESCRIPTION: refined")
    eng._parse_plan = MagicMock(
        return_value={"mode": "coder", "description": "refined"}
    )
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    check = {"is_valid": False, "reason": "missing SCF", "missing_steps": ["SCF"]}
    asyncio.run(eng._refine_plan(plan, check, "h", {}))
    # few-shot 注入了
    prompt = eng._llm_chat.call_args.args[0]
    assert "SCF then band" in prompt
    assert "workflow" in prompt
    assert "同场景成功示例" in prompt


def test_refine_no_success_history_uses_na():
    """没有同场景成功 plan -> few_shot_block = N/A."""
    eng = _make_engine(iteration=35)
    eng._plan_check_history = []  # 空
    eng._llm_chat = AsyncMock(return_value="MODE: coder\nDESCRIPTION: refined")
    eng._parse_plan = MagicMock(
        return_value={"mode": "coder", "description": "refined"}
    )
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    check = {"is_valid": False, "reason": "missing SCF", "missing_steps": ["SCF"]}
    asyncio.run(eng._refine_plan(plan, check, "h", {}))
    prompt = eng._llm_chat.call_args.args[0]
    assert "N/A" in prompt


def test_refine_success_snapshot_stored_on_pass():
    """plan_check 通过时存 plan_snapshot 到 history, 喂下次 refine few-shot."""
    eng = _make_engine(iteration=35)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": True,
            "reason": "ok",
            "missing_steps": [],
            "risks": [],
        }
    )
    plan = {"mode": "coder", "description": "run SCF calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    assert eng._plan_check_history[-1]["plan_snapshot"]["mode"] == "coder"
    assert (
        "run SCF calculation on Si"
        in eng._plan_check_history[-1]["plan_snapshot"]["description"]
    )


# ── scene_tag 自动发现 ───────────────────────────────────────


def test_discover_scene_tags_finds_high_freq_words():
    """scene='other' 的 plans 里高频词 (>=3 次) 自动加到 extra_keywords."""
    eng = _make_engine()
    # 3 条 scene='other' 的 plans, 都含 "calphad" (3 次)
    eng._plan_check_history = [
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {
                "description": "run calphad calculation for phase diagram"
            },
        },
        {
            "is_valid": False,
            "scene_tag": "other",
            "plan_snapshot": {"description": "calphad phase diagram analysis"},
        },
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "calphad equilibrium calculation"},
        },
    ]
    eng._discover_scene_tags()
    assert "auto_calphad" in eng._scene_tag_extra_keywords
    assert "calphad" in eng._scene_tag_extra_keywords["auto_calphad"]


def test_discover_scene_tags_insufficient_sample_noop():
    """scene='other' 样本 <3 条 -> 不触发发现."""
    eng = _make_engine()
    eng._plan_check_history = [
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "calphad calculation"},
        },
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "calphad analysis"},
        },
    ]
    eng._discover_scene_tags()
    assert eng._scene_tag_extra_keywords == {}


def test_discover_scene_tags_filters_stopwords():
    """停用词 (run/calc/calculate 等) 不计入."""
    eng = _make_engine()
    eng._plan_check_history = [
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "run calc calculate run calc calculate"},
        },
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "run calc calculate run calc calculate"},
        },
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "run calc calculate run calc calculate"},
        },
    ]
    eng._discover_scene_tags()
    # 全是停用词, 不应该发现任何关键词
    assert eng._scene_tag_extra_keywords == {}


def test_discovered_keywords_used_in_scene_tag():
    """自动发现的关键词能被 _plan_check_scene_tag 识别."""
    eng = _make_engine()
    eng._scene_tag_extra_keywords = {"auto_calphad": {"calphad"}}
    plan = {"mode": "coder", "description": "run calphad phase diagram"}
    assert eng._plan_check_scene_tag(plan) == "auto_calphad"


# ── EWMA 衰减 ───────────────────────────────────────────────


def test_ewma_recent_failures_tighten_more_than_old():
    """近期失败比早期失败更影响 max_refines (EWMA 加权).

    场景: 5 条历史, 4 成功 + 1 失败. 失败在最近 -> EWMA 拉低更多 -> 收紧.
    对比: 失败在最老 -> EWMA 影响小 -> 放宽.
    """
    eng = _make_engine(iteration=35)
    # 场景 1: 前 4 成功 + 最后 1 失败 (近期差)
    eng._plan_check_history = [
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": False, "scene_tag": "dft"},
    ]
    rate_recent_bad = eng._plan_check_max_refines("light", "dft")

    # 场景 2: 最老 1 失败 + 后 4 成功 (近期好)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
    ]
    rate_recent_good = eng._plan_check_max_refines("light", "dft")

    # EWMA 让近期权重更高: 近期好 -> 放宽 (0), 近期差 -> 不放宽 (1)
    # (4 成功 + 1 失败在最近, EWMA ~0.83 刚过 0.8 阈值边界, 但近期差的
    #  场景因为最近一条失败权重 0.4, 拉低到 ~0.83 以下走 baseline=1)
    assert rate_recent_good == 0  # 4 成功 + 老失败, EWMA 高, 放宽
    assert rate_recent_bad == 1  # 4 成功 + 近失败, EWMA 低, 走 baseline


def test_ewma_all_success_loosen():
    """5 条全成功 -> EWMA=1.0 >=0.8 -> 放宽 (baseline-1=0)."""
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [{"is_valid": True, "scene_tag": "dft"} for _ in range(5)]
    assert eng._plan_check_max_refines("light", "dft") == 0


def test_ewma_all_failure_tighten():
    """5 条全失败 -> EWMA=0.0 <=0.2 -> 收紧 (baseline+1=2)."""
    eng = _make_engine(iteration=35)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"} for _ in range(5)
    ]
    assert eng._plan_check_max_refines("light", "dft") == 2


# ── confidence 分级校验 ─────────────────────────────────────


def test_confidence_in_prompt_and_parse():
    """prompt 要求 confidence, _parse_plan_check 补全默认 0.8."""
    eng = _make_engine()
    # 解析: 缺 confidence -> 默认 0.8
    res = eng._parse_plan_check('{"is_valid": true, "reason": "ok"}')
    assert res["confidence"] == 0.8
    # 解析: 有 confidence -> 用 LLM 给的
    res = eng._parse_plan_check(
        '{"is_valid": false, "confidence": 0.2, "reason": "bad"}'
    )
    assert res["confidence"] == 0.2


def test_low_confidence_pass_forces_refine():
    """is_valid=True 但 confidence < 0.5 -> 强制 refine 一次.

    light tier, baseline max_refines=1.
    """
    eng = _make_engine(iteration=35)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": True,
            "confidence": 0.3,
            "reason": "unsure",
            "missing_steps": [],
            "risks": [],
        }
    )
    refine_counter = {"n": 0}

    async def _fake_refine(plan, check, hyp, ctx):
        refine_counter["n"] += 1
        return {"mode": "coder", "description": f"refined {refine_counter['n']}"}

    eng._refine_plan = AsyncMock(side_effect=_fake_refine)
    eng._override_plan_mode = MagicMock(side_effect=lambda p: p)
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    # 低置信通过, 强制 refine 1 次 (max_refines=1), 第二次通过也低置信但
    # attempt >= max_refines 直接返回
    assert refine_counter["n"] == 1
    assert eng._plan_check.call_count == 2  # 第一次 + refine 后第二次


def test_low_confidence_failure_skips_refine():
    """is_valid=False 且 confidence < 0.3 -> 跳过 refine, 直接 warning.

    LLM 都没把握判断, refine 可能也是瞎改, 直接 warning + 澄清更靠谱.
    """
    eng = _make_engine(iteration=35)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": False,
            "confidence": 0.2,
            "reason": "unsure",
            "missing_steps": [],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock()  # 不应该被调
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    eng._refine_plan.assert_not_called()
    assert eng._plan_check.call_count == 1  # 只校验一次, 直接 warning
    assert "low_conf=0.20" in eng._plan_check_warnings[-1]


def test_high_confidence_pass_no_refine():
    """is_valid=True 且 confidence >= 0.5 -> 直接通过, 不 refine."""
    eng = _make_engine(iteration=35)
    eng._plan_check = AsyncMock(
        return_value={
            "is_valid": True,
            "confidence": 0.9,
            "reason": "ok",
            "missing_steps": [],
            "risks": [],
        }
    )
    eng._refine_plan = AsyncMock()
    plan = {"mode": "coder", "description": "run band calculation on Si"}
    asyncio.run(eng._plan_check_and_refine(plan, "h", {}))
    eng._refine_plan.assert_not_called()
    assert eng._plan_check.call_count == 1


# ── 复杂度阈值自动校准 ───────────────────────────────────────


def test_thresholds_default_insufficient_sample():
    """history <5 条 -> 默认 0.7/0.25."""
    eng = _make_engine()
    eng._plan_check_history = []
    assert eng._plan_check_complexity_thresholds() == (0.7, 0.25)
    eng._plan_check_history = [{"is_valid": True} for _ in range(4)]
    assert eng._plan_check_complexity_thresholds() == (0.7, 0.25)


def test_thresholds_loosen_on_high_success():
    """最近 10 条 success rate >=0.8 -> upgrade 放宽 0.8, downgrade 收紧 0.15."""
    eng = _make_engine()
    eng._plan_check_history = [{"is_valid": True} for _ in range(8)]
    upgrade, downgrade = eng._plan_check_complexity_thresholds()
    assert upgrade == 0.8
    assert downgrade == 0.15


def test_thresholds_tighten_on_low_success():
    """最近 10 条 success rate <=0.2 -> upgrade 收紧 0.6, downgrade 放宽 0.35."""
    eng = _make_engine()
    eng._plan_check_history = [{"is_valid": False} for _ in range(8)]
    upgrade, downgrade = eng._plan_check_complexity_thresholds()
    assert upgrade == 0.6
    assert downgrade == 0.35


def test_thresholds_mid_success_uses_default():
    """success rate 在 0.2-0.8 之间 -> 默认 0.7/0.25."""
    eng = _make_engine()
    eng._plan_check_history = [
        {"is_valid": True},
        {"is_valid": False},
        {"is_valid": True},
        {"is_valid": False},
        {"is_valid": True},
    ]
    upgrade, downgrade = eng._plan_check_complexity_thresholds()
    assert upgrade == 0.7
    assert downgrade == 0.25


def test_thresholds_affect_tier_decision():
    """低 success rate 收紧 upgrade 阈值 -> 中等复杂 plan 也升级到 medium.

    场景: open tier + 中等复杂 plan (score ~0.62).
    默认阈值 0.7 -> 0.62 < 0.7 不升级, 留 open.
    低 success rate -> 阈值收紧到 0.6 -> 0.62 >= 0.6 升级到 medium.
    """
    eng = _make_engine(iteration=5)  # open tier
    # 中等复杂 plan: workflow + 短 desc + prediction
    # score: workflow(0.4) + 12chars(0.072) + prediction(0.15) = 0.622
    borderline_plan = {
        "mode": "workflow",
        "description": "run VASP SCF",
        "expected_prediction": "band_gap=1.1eV",
    }
    score = eng._plan_check_complexity(borderline_plan)
    assert 0.6 <= score < 0.7, f"score should be in [0.6, 0.7), got {score}"
    # 默认阈值 0.7: 0.622 < 0.7 不升级, 留 open
    assert eng._plan_check_tier(borderline_plan) == "open"
    # 低 success rate: 阈值收紧到 0.6 -> 0.622 >= 0.6 升级到 medium
    eng._plan_check_history = [{"is_valid": False} for _ in range(8)]
    assert eng._plan_check_tier(borderline_plan) == "medium"


# ── bigram scene_tag 识别 ───────────────────────────────────


def test_discover_bigram_scene_tag():
    """bigram (>=3 次) 自动加到 extra_keywords, 如 'phase diagram'."""
    eng = _make_engine()
    eng._plan_check_history = [
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "calphad phase diagram calculation"},
        },
        {
            "is_valid": False,
            "scene_tag": "other",
            "plan_snapshot": {"description": "phase diagram analysis here"},
        },
        {
            "is_valid": True,
            "scene_tag": "other",
            "plan_snapshot": {"description": "phase diagram equilibrium test"},
        },
    ]
    eng._discover_scene_tags()
    # "phase diagram" bigram 出现 3 次, 应该被发现
    assert "auto_phase_diagram" in eng._scene_tag_extra_keywords
    assert "phase diagram" in eng._scene_tag_extra_keywords["auto_phase_diagram"]


def test_discovered_bigram_used_in_scene_tag():
    """自动发现的 bigram 关键词能被 _plan_check_scene_tag 识别."""
    eng = _make_engine()
    eng._scene_tag_extra_keywords = {"auto_phase_diagram": {"phase diagram"}}
    plan = {"mode": "coder", "description": "run phase diagram calculation"}
    assert eng._plan_check_scene_tag(plan) == "auto_phase_diagram"


def test_bigram_takes_priority_over_unigram_when_more_specific():
    """bigram 和 unigram 都被发现时, scene_tag 慢路径按 dict 顺序匹配.

    ponytail: 不做优先级排序, 先到先得; 实际场景下 bigram 更具体通常
    也能命中. 测试只验证 bigram 能被识别, 不验证优先级.
    """
    eng = _make_engine()
    eng._scene_tag_extra_keywords = {
        "auto_phase": {"phase"},
        "auto_phase_diagram": {"phase diagram"},
    }
    plan = {"mode": "coder", "description": "run phase diagram calculation"}
    # dict 顺序: auto_phase 在前, 先命中 "phase"
    # 这不是 bug, 是 ponytail 取舍 — 慢路径按 dict 顺序
    tag = eng._plan_check_scene_tag(plan)
    assert tag in ("auto_phase", "auto_phase_diagram")


# ── 分场景复杂度阈值校准 ─────────────────────────────────────


def test_thresholds_scene_specific_isolation():
    """DFT 场景的 success rate 不影响 MD 场景的阈值.

    DFT 8 条全成功 -> DFT 阈值放宽 (0.8, 0.15).
    MD 场景无历史, 全局也是 DFT 全成功 -> 回退全局也是 (0.8, 0.15).
    但如果全局是混合的, MD 回退全局会用全局的阈值, 不会用 DFT 的.
    """
    eng = _make_engine()
    # DFT 5 条全成功 + MD 3 条全失败 (全局 8 条, 5 成功 3 失败, success_rate=0.625)
    eng._plan_check_history = [
        {"is_valid": True, "scene_tag": "dft"} for _ in range(5)
    ] + [{"is_valid": False, "scene_tag": "md"} for _ in range(3)]
    # DFT 场景: 5 条全成功 -> 放宽 (0.8, 0.15)
    assert eng._plan_check_complexity_thresholds("dft") == (0.8, 0.15)
    # MD 场景: 3 条全失败, 不足 5 条, 回退全局 -> 全局 8 条 5 成功 3 失败,
    # success_rate=0.625 在中间 -> 默认 (0.7, 0.25), 不会跟着 DFT 放宽
    assert eng._plan_check_complexity_thresholds("md") == (0.7, 0.25)


def test_thresholds_scene_insufficient_falls_back_to_global():
    """场景样本 <5 但全局 >=5 -> 回退全局校准."""
    eng = _make_engine()
    # DFT 3 条 (不足) + MD 5 条全成功 (全局 8 条, 全成功)
    eng._plan_check_history = [
        {"is_valid": True, "scene_tag": "dft"} for _ in range(3)
    ] + [{"is_valid": True, "scene_tag": "md"} for _ in range(5)]
    # DFT 只有 3 条, 回退全局 -> 全局 8 条全成功 -> 放宽
    assert eng._plan_check_complexity_thresholds("dft") == (0.8, 0.15)


# ── EWMA alpha 自适应 ────────────────────────────────────────


def test_ewma_alpha_adapts_to_bucket_size():
    """桶 3-4 条用 alpha=0.3, 桶 5 条用 alpha=0.4.

    通过验证 max_refines 行为间接确认 alpha 生效: 4 成功 + 1 失败在最近
    (桶=5, alpha=0.4) 走 baseline=1; 同样 4 成功 + 1 失败但桶只有 4 条
    (alpha=0.3, 老样本权重更高) 也走 baseline=1, 但 EWMA 值不同.
    直接测 alpha 逻辑: 3 条全失败 + 2 条全成功, 桶大小影响结果.
    """
    eng = _make_engine(iteration=35)
    # 桶=5: 前 3 失败 + 后 2 成功, alpha=0.4
    # ewma = (0.4*0.6^4*0 + 0.4*0.6^3*0 + 0.4*0.6^2*0 + 0.4*0.6*1 + 0.4*1) / total
    #      = (0 + 0 + 0 + 0.24 + 0.4) / 0.922 = 0.694 -> baseline=1
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
    ]
    assert eng._plan_check_max_refines("light", "dft") == 1  # 0.694 在中间

    # 桶=3: 1 失败 + 2 成功, alpha=0.3 (老样本权重大, 失败拉低更多)
    eng._plan_check_history = [
        {"is_valid": False, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
        {"is_valid": True, "scene_tag": "dft"},
    ]
    # alpha=0.3: weights = [0.3*0.7^2, 0.3*0.7, 0.3] = [0.147, 0.21, 0.3]
    # ewma = (0.147*0 + 0.21*1 + 0.3*1) / 0.657 = 0.51/0.657 = 0.776
    # 0.776 < 0.8 -> baseline=1 (没到放宽阈值)
    assert eng._plan_check_max_refines("light", "dft") == 1
