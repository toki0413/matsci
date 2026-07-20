"""数学结构形式化校验 — 从 engine.py 抽出的函数模块.

5 个来源 (从 engine._run_math_validation 等抽出):
  1. run_math_validation(engine, execution_result) — 三子项独立校验
     (BourbakiTool 守恒律 / LeanTool 变分原理 / AutoDiffTool 自动微分)
  2. collect_math_evidence(engine, execution_result, math_validation) —
     5 个数学证据 key (conservation_law / dimensional_consistent /
     pde_classification / sobol_top_features / constraint_check)
  3. query_kb_reference(engine, equations, lagrangian) — 查 KB 拿
     first-principles 参考块
  4. build_reviewer_prompt(execution_result, results, kb_text) — 纯函数,
     构造 reviewer persona prompt

不在本模块 (engine._verify_via_gp 是死代码, 定义但无任何 caller, 抽出时
直接删, 不迁移).

self 依赖 (通过 engine 访问):
  - workspace / settings (ToolContext 构造)
  - _get_kb() (KB 查询)
  - _last_execution_result (GP 验证用, 但已删 _verify_via_gp, 不再需要)

接入点:
  - engine._validate: 4 处调用 (run_math_validation / collect_math_evidence /
    build_reviewer_prompt)

ponytail: 函数模块 > Mixin (见 S2 mixin 评估). engine: Any 第一参,
SimpleNamespace 即可测, 无需构造完整 AutoloopEngine.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. run_math_validation ──────────────────────────────────

async def run_math_validation(engine: Any, execution_result: Any) -> dict[str, Any]:
    """把执行结果里的数学结构抽出来, 用数学工具做形式化校验.

    三个独立子项, 互不影响:
      A. 守恒律 (BourbakiTool.check_conservation) — equations 非空时跑
      B. 变分原理 (LeanTool.constitutive/variational_principle) — lagrangian 非空时跑
      C. 自动微分 (AutoDiffTool.gradient) — function spec 齐全时跑

    工具懒加载, 任一缺失/报错只记 *_error, 不阻断其余子项与主 validate 流程.
    engine 没有自己的 tool_registry, 这里直接构造工具实例 (它们都是无状态轻量构造).
    """
    from huginn.types import ToolContext

    out: dict[str, Any] = {}
    if not isinstance(execution_result, dict):
        return out

    tool_ctx = ToolContext(
        session_id=f"mathval_{uuid.uuid4().hex[:8]}",
        workspace=str(engine.workspace),
        config=engine.settings,
    )

    equations = execution_result.get("equations") or ""
    lagrangian = execution_result.get("lagrangian") or ""
    coords = execution_result.get("coordinates") or []
    velocities = execution_result.get("velocities")
    domain = execution_result.get("conservation_domain") or "continuum_mechanics"
    if equations:
        try:
            from huginn.tools.bourbaki_tool import BourbakiTool

            tool = BourbakiTool()
            raw = await tool.call(
                {
                    "task": "check_conservation",
                    "domain": domain,
                    "equations": equations,
                },
                tool_ctx,
            )
            # BourbakiTool.call 可能返回 dict 或 BourbakiResult; 统一成 dict
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            out["conservation"] = {
                "verified": raw.get("verified"),
                "message": raw.get("message", ""),
                "fallback": raw.get("fallback", False),
                "method": "bourbaki",
            }
        except Exception as e:
            out["conservation_error"] = str(e)

    # A2: KB 交叉验证 — 把守恒律方程 + Lagrangian 关键词拿去查 KB, 命中的
    # first-principles 参考块作为 reference_principles 写回, 让下游 reviewer
    # 能对照已知结论. KB 不可用/空查询都不阻断, 只是不写该字段.
    kb_ref = query_kb_reference(engine, equations, lagrangian)
    if kb_ref:
        out["reference_principles"] = kb_ref

    if lagrangian and coords:
        try:
            from huginn.tools.lean_tool import LeanTool, LeanToolInput

            tool = LeanTool()
            args = LeanToolInput(
                action="constitutive",
                sub_action="variational_principle",
                lagrangian=lagrangian,
                coordinates=list(coords),
                velocities=velocities,
            )
            vr = await tool.call(args, tool_ctx)
            out["variational"] = {
                "ok": bool(vr.success),
                "data": vr.data,
                "error": vr.error,
                "method": "lean",
            }
        except Exception as e:
            out["variational_error"] = str(e)

    func_spec = execution_result.get("autodiff")
    if isinstance(func_spec, dict) and func_spec.get("function_type"):
        try:
            from huginn.tools.sci.autodiff_tool import (
                AutoDiffInput,
                AutoDiffTool,
            )

            tool = AutoDiffTool()
            args = AutoDiffInput(
                action="gradient",
                function_type=func_spec.get("function_type", "custom"),
                function_params=func_spec.get("function_params", {}),
                variables=func_spec.get("variables", {}),
                target_variable=func_spec.get("target_variable"),
            )
            vr = await tool.call(args, tool_ctx)
            out["autodiff"] = {
                "ok": bool(vr.success),
                "data": vr.data,
                "error": vr.error,
            }
        except Exception as e:
            out["autodiff_error"] = str(e)

    return out


# ── 2. collect_math_evidence ────────────────────────────────

async def collect_math_evidence(
    engine: Any, execution_result: Any, math_validation: dict
) -> dict[str, Any]:
    """从 execution_result + math_validation 抽 5 个数学证据 key,
    供 PhaseGate 的 MathEvidenceChecker 做 Dempster-Shafer 合成.

    证据来源:
      1. conservation_law — 从 math_validation["conservation"] 透传
      2. dimensional_consistent — execution_result 带 equation 时跑
         symbolic_math_tool action=dimensional_analysis
      3. pde_classification — execution_result 带 pde_coefficients +
         expected_pde_class 时跑 symbolic_math_tool action=pde_classify
      4. sobol_top_features — execution_result 带 sobol_data +
         hypothesis_features 时跑 symbolic_regression_tool action=sobol_indices
      5. constraint_check — execution_result 带 expression + constraints
         时跑 symbolic_regression_tool action=constraint_check

    每项 best-effort: 数据不全/工具报错就跳过, 不写 key (math_checker 忽略缺失).
    """
    evidence: dict[str, Any] = {}
    if not isinstance(execution_result, dict):
        return evidence

    # 1. conservation_law — 从已有 math_validation 透传
    cons = math_validation.get("conservation")
    if isinstance(cons, dict) and "verified" in cons:
        evidence["conservation_law"] = {
            "verified": bool(cons["verified"]),
            "current": cons.get("message", ""),
            "symmetry": cons.get("method", ""),
        }

    from huginn.types import ToolContext

    tool_ctx = ToolContext(
        session_id=f"mathevid_{uuid.uuid4().hex[:8]}",
        workspace=str(engine.workspace),
        config=engine.settings,
    )

    # 2. dimensional_consistent — 跑量纲分析, 所有 quantity 都能解析 → True
    equation = (
        execution_result.get("equation") or execution_result.get("equations") or ""
    )
    if equation:
        try:
            from huginn.tools.symbolic_math.tool import (
                SymbolicMathInput,
                SymbolicMathTool,
            )

            tool = SymbolicMathTool()
            args = SymbolicMathInput(
                action="dimensional_analysis",
                expression=str(equation),
                target="validate_expression",
            )
            vr = await tool.call(args, tool_ctx)
            if vr.success and vr.data:
                quantities = vr.data.get("quantities", [])
                has_error = any("error" in q for q in quantities)
                evidence["dimensional_consistent"] = (
                    len(quantities) > 0 and not has_error
                )
        except Exception:
            logger.warning(
                "error in collect_math_evidence: dimensional_analysis failed",
                exc_info=True,
            )

    # 3. pde_classification — 跑 pde_classify, 比对 expected vs actual
    pde_coeffs = execution_result.get("pde_coefficients")
    expected_class = execution_result.get("expected_pde_class")
    if pde_coeffs and expected_class:
        try:
            from huginn.tools.symbolic_math.tool import (
                SymbolicMathInput,
                SymbolicMathTool,
            )

            tool = SymbolicMathTool()
            args = SymbolicMathInput(
                action="pde_classify",
                expression=str(pde_coeffs),
            )
            vr = await tool.call(args, tool_ctx)
            if vr.success and vr.data:
                actual = vr.data.get("classification", "")
                evidence["pde_classification"] = {
                    "consistent": actual.lower() == str(expected_class).lower(),
                    "expected": str(expected_class),
                    "actual": actual,
                }
        except Exception:
            logger.warning(
                "error in collect_math_evidence: pde_classify failed",
                exc_info=True,
            )

    # 4. sobol_top_features — 跑 sobol_indices, top features (S_i>0.1) 必须
    # 被 hypothesis_features 覆盖
    sobol_data = execution_result.get("sobol_data")
    hypothesis_features = execution_result.get("hypothesis_features")
    if sobol_data and hypothesis_features:
        try:
            from huginn.tools.sci.symbolic_regression_tool import (
                SymbolicRegressionInput,
                SymbolicRegressionTool,
            )

            tool = SymbolicRegressionTool()
            target_col = (
                sobol_data.get("target", "y")
                if isinstance(sobol_data, dict)
                else "y"
            )
            args = SymbolicRegressionInput(
                action="sobol_indices",
                data_json=sobol_data,
                target_column=target_col,
                n_sobol_samples=512,
            )
            vr = await tool.call(args, tool_ctx)
            if vr.success and vr.data:
                first_order = vr.data.get("first_order", {})
                if first_order:
                    top = [f for f, s in first_order.items() if s > 0.1]
                    evidence["sobol_top_features"] = {
                        "hypothesis_covers_top": set(top).issubset(
                            set(hypothesis_features)
                        ),
                        "top_features": top,
                        "hypothesis_features": list(hypothesis_features),
                    }
        except Exception:
            logger.warning(
                "error in collect_math_evidence: sobol_indices failed",
                exc_info=True,
            )

    # 5. constraint_check — 跑 constraint_check, 所有先验通过 → all_passed
    expr = execution_result.get("expression")
    constraints = execution_result.get("constraints")
    if expr and constraints:
        try:
            from huginn.tools.sci.symbolic_regression_tool import (
                SymbolicRegressionInput,
                SymbolicRegressionTool,
            )

            tool = SymbolicRegressionTool()
            args = SymbolicRegressionInput(
                action="constraint_check",
                probe_expression=str(expr),
                constraints=constraints,
            )
            vr = await tool.call(args, tool_ctx)
            if vr.success and vr.data:
                evidence["constraint_check"] = {
                    "all_passed": vr.data.get("all_passed", False),
                    "violations": vr.data.get("violations", []),
                }
        except Exception:
            logger.warning(
                "error in collect_math_evidence: constraint_check failed",
                exc_info=True,
            )

    return evidence


# ── 3. query_kb_reference ───────────────────────────────────

def query_kb_reference(engine: Any, equations: str, lagrangian: str) -> list[dict]:
    """查 KB 拿 first-principles 参考块. 把 equations + lagrangian 拼成
    query 串, 命中返回 [{text, source}], 失败/空都返回 []."""
    query = " ".join(filter(None, [equations, lagrangian])).strip()
    if not query:
        return []
    kb = engine._get_kb()
    if kb is None:
        return []
    try:
        if kb.count() == 0:
            return []
        chunks = kb.query(f"conservation law variational {query}", top_k=2)
        return [
            {"text": (c.get("text") or "")[:300], "source": c.get("source", "")}
            for c in chunks
            if c.get("text")
        ]
    except Exception:
        return []


# ── 4. build_reviewer_prompt (纯函数) ───────────────────────

def build_reviewer_prompt(
    execution_result: Any,
    results: dict[str, Any],
    kb_text: str = "",
) -> str:
    """构造让 reviewer persona 点评执行结果的 prompt."""
    try:
        exec_blob = json.dumps(execution_result, ensure_ascii=False, default=str)[
            :1500
        ]
    except Exception:
        exec_blob = str(execution_result)[:1500]
    try:
        res_blob = json.dumps(results, ensure_ascii=False, default=str)[:1500]
    except Exception:
        res_blob = str(results)[:1500]
    kb_section = f"\n{kb_text}\n" if kb_text else ""
    return (
        "Below is the execution result and validation summary from an "
        "autonomous materials-science research loop iteration.\n\n"
        f"Execution result:\n{exec_blob}\n\n"
        f"Validation summary:\n{res_blob}\n"
        f"{kb_section}"
        "As a critical peer reviewer, point out:\n"
        "1. Any methodological weakness or missing convergence check.\n"
        "2. Whether the result is reproducible and benchmarked.\n"
        "3. Whether the result aligns with the domain knowledge context above "
        "(if any), or contradicts known first-principles.\n"
        "4. Concrete next-step improvements.\n"
        "Be concise and direct."
    )


# ── self-check (assert-based, 无框架无 fixture) ─────────────

def _selfcheck() -> None:
    """20 项 assert 验证 4 个函数的核心行为.

    每项失败立即 AssertionError, 全过打印 "all self-checks passed".
    不依赖 KB / 工具 — 用 SimpleNamespace + monkey patch.
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # ── build_reviewer_prompt (纯函数, 5 项) ──
    # 1) 基本: exec + results 都进 prompt
    p = build_reviewer_prompt({"a": 1}, {"b": 2})
    assert "Execution result" in p
    assert "Validation summary" in p
    assert '"a": 1' in p
    assert '"b": 2' in p

    # 2) kb_text 非空 → 注入 Domain Knowledge section
    p = build_reviewer_prompt({}, {}, kb_text="SOME_KB_TEXT")
    assert "SOME_KB_TEXT" in p

    # 3) kb_text 空 → 不注入 (不出现空行)
    p = build_reviewer_prompt({}, {}, kb_text="")
    # 空 kb_section 不影响 reviewer 指令
    assert "As a critical peer reviewer" in p

    # 4) execution_result 不可 JSON 序列化 → 退到 str()
    class _NotJsonable:
        pass
    p = build_reviewer_prompt(_NotJsonable(), {})
    assert "_NotJsonable" in p

    # 5) results 不可 JSON 序列化 → 退到 str()
    p = build_reviewer_prompt({}, _NotJsonable())
    assert "_NotJsonable" in p

    # ── query_kb_reference (5 项) ──
    # 6) 无 query → [] (不需要 _get_kb)
    eng = SimpleNamespace()
    assert query_kb_reference(eng, "", "") == []

    # 7) kb=None → []
    eng = SimpleNamespace(_get_kb=lambda: None)
    assert query_kb_reference(eng, "", "L = T - V") == []
    assert query_kb_reference(eng, "x = y", "") == []

    # 8) kb.count() == 0 → []
    fake_kb = MagicMock()
    fake_kb.count.return_value = 0
    eng = SimpleNamespace(_get_kb=lambda: fake_kb)
    assert query_kb_reference(eng, "x = y", "") == []

    # 9) kb.query 返回有 text 的 chunks → 截断 + 带上 source
    fake_kb = MagicMock()
    fake_kb.count.return_value = 1
    fake_kb.query.return_value = [
        {"text": "mass conservation", "source": "wiki"},
        {"text": "", "source": "ignored"},  # 空 text 跳过
    ]
    eng = SimpleNamespace(_get_kb=lambda: fake_kb)
    out = query_kb_reference(eng, "x = y", "")
    assert len(out) == 1
    assert out[0]["text"] == "mass conservation"
    assert out[0]["source"] == "wiki"

    # 10) kb.query 抛异常 → []
    fake_kb = MagicMock()
    fake_kb.count.return_value = 1
    fake_kb.query.side_effect = RuntimeError("boom")
    eng = SimpleNamespace(_get_kb=lambda: fake_kb)
    assert query_kb_reference(eng, "x = y", "") == []

    # 11) text 截断 300 字符
    long_text = "x" * 500
    fake_kb = MagicMock()
    fake_kb.count.return_value = 1
    fake_kb.query.return_value = [{"text": long_text, "source": "src"}]
    eng = SimpleNamespace(_get_kb=lambda: fake_kb)
    out = query_kb_reference(eng, "x = y", "")
    assert len(out[0]["text"]) == 300

    # ── run_math_validation (7 项) ──
    # 12) 非 dict execution_result → {}
    eng = SimpleNamespace(workspace=MagicMock(), settings=None, _get_kb=lambda: None)
    out = asyncio.run(run_math_validation(eng, "not a dict"))
    assert out == {}

    # 13) 空 dict → {}
    out = asyncio.run(run_math_validation(eng, {}))
    assert out == {}

    # 14) equations 非空但 BourbakiTool 抛错 → 记 conservation_error
    import sys
    import huginn.tools.bourbaki_tool as _bt
    orig = _bt.BourbakiTool
    def _boom():
        raise RuntimeError("init failed")
    _bt.BourbakiTool = _boom
    try:
        out = asyncio.run(run_math_validation(eng, {"equations": "a = a"}))
        assert "conservation" not in out
        assert "conservation_error" in out
    finally:
        _bt.BourbakiTool = orig

    # 15) KB 有内容 → reference_principles 写入
    fake_kb = MagicMock()
    fake_kb.count.return_value = 1
    fake_kb.query.return_value = [{"text": "ref principle", "source": "kb"}]
    eng = SimpleNamespace(workspace=MagicMock(), settings=None, _get_kb=lambda: fake_kb)
    out = asyncio.run(run_math_validation(eng, {"equations": "mass conservation"}))
    assert "reference_principles" in out
    assert out["reference_principles"][0]["text"] == "ref principle"

    # 16) KB 空 → 无 reference_principles
    fake_kb = MagicMock()
    fake_kb.count.return_value = 0
    eng = SimpleNamespace(workspace=MagicMock(), settings=None, _get_kb=lambda: fake_kb)
    out = asyncio.run(run_math_validation(eng, {"equations": "x = y"}))
    assert "reference_principles" not in out

    # 17) lagrangian 没 coords → 跳过 variational
    eng = SimpleNamespace(workspace=MagicMock(), settings=None, _get_kb=lambda: None)
    out = asyncio.run(run_math_validation(eng, {"lagrangian": "L = T - V", "coordinates": []}))
    assert "variational" not in out

    # 18) autodiff 缺 function_type → 跳过
    eng = SimpleNamespace(workspace=MagicMock(), settings=None, _get_kb=lambda: None)
    out = asyncio.run(run_math_validation(eng, {"autodiff": {"function_params": {}}}))
    assert "autodiff" not in out

    # ── collect_math_evidence (3 项) ──
    # 19) conservation_law 透传
    eng = SimpleNamespace(workspace=MagicMock(), settings=None)
    math_val = {"conservation": {"verified": True, "message": "ok", "method": "bourbaki"}}
    out = asyncio.run(collect_math_evidence(eng, {}, math_val))
    assert out["conservation_law"]["verified"] is True
    assert out["conservation_law"]["symmetry"] == "bourbaki"

    # 20) 非 dict execution_result → {}
    eng = SimpleNamespace(workspace=MagicMock(), settings=None)
    out = asyncio.run(collect_math_evidence(eng, "not a dict", {}))
    assert out == {}

    # 21) 无 equation/pde/sobol/expr → 只有 conservation_law (如果 math_val 有)
    eng = SimpleNamespace(workspace=MagicMock(), settings=None)
    out = asyncio.run(collect_math_evidence(eng, {}, {}))
    assert out == {}  # math_val 也是空

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
