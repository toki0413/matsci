"""MathValidationMixin - math_validation 方法族, 从 engine.py 下沉.

P2 slim-down: 3 个 math validation 方法从 engine.py 迁入, 定义为 mixin class.
engine 通过多继承接入, 方法内通过 self 访问 engine 状态字段
(_last_execution_result / workspace / settings) 和 engine 方法
(_query_kb_reference).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class MathValidationMixin:
    """math_validation 方法族. 通过 self 访问 engine 状态."""

    async def _run_math_validation(self, execution_result: Any) -> dict[str, Any]:
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
            workspace=str(self.workspace),
            config=self.settings,
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
        kb_ref = self._query_kb_reference(equations, lagrangian)
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

    def _verify_via_gp(self, hyp_id: str, validation: dict) -> dict:
        """循环B: 用 GP 数值验证做独立路径. 与符号演绎 (循环A) 基底正交.

        升级: fit + leave-one-out 风格 predict, 检查后验均值与实验值
        的偏差是否在 ±2σ 内. 若有测试集 (X_test, y_test) 则用之, 否则
        在训练集上做 LOO-style 检查 (GP 在 fit 数据上 predict 时, sigma
        较小但仍有信号).

        ponytail: ±2σ 是 ~95% 置信区间. 拒绝域 = 5%. 升级路径:
        对假设的 testable_prediction 做数值区间解析, 用 KL(GP_posterior
        || hypothesis_interval) 代替 ±2σ 检查.
        """
        exec_res = getattr(self, "_last_execution_result", None)
        X = y = X_test = y_test = None
        for cand in (validation, exec_res if isinstance(exec_res, dict) else {}):
            X = cand.get("X") or cand.get("x_data") or cand.get("samples")
            y = cand.get("y") or cand.get("y_data") or cand.get("targets")
            X_test = cand.get("X_test") or cand.get("x_test")
            y_test = cand.get("y_test") or cand.get("y_true")
            if X is not None and y is not None:
                break
            X = y = None
        if X is None or y is None:
            return {"agrees": False, "reason": "no numeric data for GP fit"}

        try:
            import numpy as np

            from huginn.tools.sci.gp_tool import GPTool

            tool = GPTool()

            # 若有独立测试集, predict 在 X_test 上, 与 y_test 比对
            # 否则 fit 后 predict 在 X 上做自洽检查 (弱信号, sigma 小)
            pred_X = X_test if X_test is not None else X
            pred_y_ref = y_test if y_test is not None else y

            predict_res = tool.call(
                {
                    "action": "predict",
                    "X": X,
                    "y": y,
                    "X_new": pred_X,
                }
            )
            if not getattr(predict_res, "success", False):
                return {
                    "agrees": False,
                    "reason": "GP predict failed",
                    "error": getattr(predict_res, "error", ""),
                }

            data = getattr(predict_res, "data", None) or {}
            mu = np.asarray(data.get("mean", []), dtype=float)
            sigma = np.asarray(data.get("std", []), dtype=float)
            y_ref = np.asarray(pred_y_ref, dtype=float)

            # 后验一致检验: |y - mu| <= 2σ (95% CI)
            # sigma=0 时退化为 |y - mu| < eps (GP 完全过拟合)
            n = min(len(mu), len(y_ref))
            if n == 0:
                return {
                    "agrees": True,
                    "gp_fit": data,
                    "reason": "GP fit ok, no comparable points",
                }
            mu, sigma, y_ref = mu[:n], sigma[:n], y_ref[:n]
            eps = 1e-8
            deviation = np.abs(y_ref - mu)
            tolerance = np.maximum(2.0 * sigma, eps)
            agrees = bool(np.all(deviation <= tolerance))
            max_dev = float(np.max(deviation))
            max_tol = float(np.max(tolerance))
            return {
                "agrees": agrees,
                "gp_fit": data,
                "max_deviation": max_dev,
                "max_tolerance": max_tol,
                "n_points": n,
                "reason": (
                    f"posterior ±2σ check: max_dev={max_dev:.4g} "
                    f"vs tol={max_tol:.4g} over {n} points"
                ),
            }
        except Exception as e:
            return {"agrees": False, "reason": f"GP verify error: {e}"}

    async def _collect_math_evidence(
        self, execution_result: Any, math_validation: dict
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
            workspace=str(self.workspace),
            config=self.settings,
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
                    "error in _collect_math_evidence: dimensional_analysis failed",
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
                    "error in _collect_math_evidence: pde_classify failed",
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
                    "error in _collect_math_evidence: sobol_indices failed",
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
                    "error in _collect_math_evidence: constraint_check failed",
                    exc_info=True,
                )

        return evidence


