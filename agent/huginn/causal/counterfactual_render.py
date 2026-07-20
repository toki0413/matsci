"""CounterfactualRender — Phase 3: L3 反事实渲染 (abduction + prediction).

Pearl L3 反事实 P(Y_x | X=x', Y=y'):
  给定观测 (X=x', Y=y'), 问"如果当时 X=x 而不是 x', Y 会是多少?"

三步 (Pearl 2009):
  1. Abduction: 从观测反推 noise U 的后验 P(U | X=x', Y=y')
  2. Action: do(X=x), 把 X 的方程替换为常数 x
  3. Prediction: 用 abducted U 算 Y 在 do(X=x) 下的分布

实现: Monte Carlo abduction — 采样 noise, filter 匹配 evidence 的样本
作为后验. 不要求方程可逆 (方程是闭包形式, Monte Carlo 通用).

阶梯映射 (Pearl Causal Hierarchy):
  L1 观察  P(Y|X)           — vision_describe (感知层)
  L2 干预  P(Y|do(X))       — predict_intervention (Phase 1)
  L2+ 拟合 P(Y|do(X), data) — visual_causal_chain (Phase 2)
  L3 反事实 P(Y_x|X',Y')    — 本模块 (Phase 3)

设计原则 (ponytail):
  - Monte Carlo abduction (rejection sampling), 不要求方程可逆
  - evidence 只放 feature 节点观测, 条件节点走 base_conditions
  - 没匹配样本返 error, 不静默用先验 (反事实对噪声敏感, 静默会误导)
  - SCM.confirmed=False 显式警告 (跟 predict_intervention 一致)
  - 不动 predict_intervention._simulate_once (它内部重采 noise), 本地写 _simulate_with_noise

升级路径:
  - 方程可逆时用解析 abduction (快 + 准): 先识别 multiplicative/additive/log 形式
  - 多节点 evidence 用 importance sampling 替代 rejection sampling
  - Bayesian posterior over noise (MCMC)
  - 反事实渲染成图像 (visualize cf prediction as plot)
"""
from __future__ import annotations

import logging
import statistics
from typing import Any

from pydantic import BaseModel, Field

from huginn.causal.visual_scm import VisualSCM, get_template, list_templates
from huginn.causal.visual_causal_chain import (
    fit_scm_from_observations, Observation,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ── 核心算法: Monte Carlo abduction + 反事实预测 ────────────

def _simulate_with_noise(
    scm: VisualSCM,
    base_conditions: dict[str, float],
    intervention: dict[str, float],
    noise_sample: dict[str, float],
) -> dict[str, float]:
    """用给定 noise 样本模拟 SCM.

    跟 predict_intervention._simulate_once 不同: 不重新采 noise, 用传入的
    noise_sample. 这是 abduction 的关键 — 同一组 noise 要复用到 factual
    和 counterfactual 两次模拟.
    """
    values: dict[str, float] = {}
    order = scm.topological_order()
    for node in order:
        if node in intervention:
            values[node] = float(intervention[node])
        elif node in base_conditions:
            values[node] = float(base_conditions[node])
        else:
            noise = noise_sample.get(node, 0.0)
            try:
                val = scm.equations[node](values, noise)
            except Exception as exc:
                logger.debug("SCM equation %s failed: %s", node, exc)
                val = 0.0
            var = scm.nodes.get(node)
            if var and var.range:
                lo, hi = var.range
                val = max(lo, min(hi, val))
            values[node] = float(val)
    return values


def _abduct_noise(
    scm: VisualSCM,
    evidence: dict[str, float],
    base_conditions: dict[str, float],
    n_samples: int,
    tolerance: float,
    max_attempts: int | None = None,
) -> list[dict[str, float]]:
    """Monte Carlo abduction: 采 noise, filter 匹配 evidence 的样本作为后验.

    ponytail: rejection sampling. 简单但匹配率低时慢.
    升级路径: importance sampling / MCMC / 解析 abduction.
    """
    if max_attempts is None:
        max_attempts = n_samples * 100

    abducted: list[dict[str, float]] = []
    attempts = 0
    while len(abducted) < n_samples and attempts < max_attempts:
        attempts += 1
        noise_sample = {n: scm.noise[n]() for n in scm.noise}
        sim = _simulate_with_noise(scm, base_conditions, {}, noise_sample)
        match = True
        for k, v in evidence.items():
            if k not in sim:
                match = False
                break
            ref = max(abs(v), 1e-12)
            if abs(sim[k] - v) / ref > tolerance:
                match = False
                break
        if match:
            abducted.append(noise_sample)
    return abducted


def _stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0,
                "p95": 0.0, "min": 0.0, "max": 0.0}
    s_sorted = sorted(samples)
    n = len(s_sorted)
    return {
        "mean": statistics.mean(samples),
        "std": statistics.stdev(samples) if n > 1 else 0.0,
        "p5": s_sorted[int(0.05 * n)],
        "p50": s_sorted[int(0.50 * n)],
        "p95": s_sorted[int(0.95 * n)],
        "min": s_sorted[0],
        "max": s_sorted[-1],
    }


def counterfactual_render(
    scm: VisualSCM,
    evidence: dict[str, float],
    intervention: dict[str, float],
    targets: list[str],
    base_conditions: dict[str, float] | None = None,
    n_samples: int = 200,
    tolerance: float = 0.15,
    max_attempts_multiplier: int = 100,
) -> dict[str, Any]:
    """L3 反事实渲染: P(targets | do(intervention), evidence).

    Args:
        scm: 结构因果模型
        evidence: feature 节点的观测值 {particle_size: 200}.
                  条件节点 (T/t/p) 不要放这里, 走 base_conditions.
        intervention: 反事实 do(X=v) {T: 1800}
        targets: 要预测的节点名
        base_conditions: evidence 时的实际条件 {T: 1500, t: 2}
        n_samples: abduction 目标样本数 (匹配 evidence 的 noise 样本数)
        tolerance: evidence 匹配容差 (相对误差, 0.15 = 15%)
        max_attempts_multiplier: rejection sampling 最大尝试次数 = n_samples * 此值

    Returns:
        {
          "targets": {name: {mean, std, p5, p50, p95, min, max}},
          "factual": {name: {...}},  # 用 abducted noise 做 factual 模拟 (无干预)
          "lift": {name: {mean_lift, relative_lift, significant}},
          "evidence": {...},
          "intervention": {...},
          "base_conditions": {...},
          "n_abducted": int,
          "scm_confirmed": bool,
          "warning": str | None,
        }
    """
    base_conditions = base_conditions or {}

    # 校验
    missing_t = [t for t in targets if t not in scm.nodes]
    if missing_t:
        return {"error": f"targets 不在 SCM: {missing_t}",
                "scm_nodes": list(scm.nodes.keys())}
    bad_interv = [k for k in intervention if k not in scm.nodes]
    if bad_interv:
        return {"error": f"intervention 节点不在 SCM: {bad_interv}"}
    bad_ev = [k for k in evidence if k not in scm.nodes]
    if bad_ev:
        return {"error": f"evidence 节点不在 SCM: {bad_ev}"}
    overlap = set(evidence) & set(intervention)
    if overlap:
        return {"error": f"evidence 和 intervention 重叠: {overlap}. "
                          "evidence 放观测 feature, intervention 改条件."}

    # 1. Abduction
    abducted = _abduct_noise(
        scm, evidence, base_conditions, n_samples, tolerance,
        max_attempts=n_samples * max_attempts_multiplier,
    )
    if not abducted:
        return {
            "error": "abduction 失败: 无 noise 样本匹配 evidence "
                     "(可能 evidence 不可达, 或 tolerance 太小)",
            "evidence": evidence,
            "base_conditions": base_conditions,
            "n_attempts": n_samples * max_attempts_multiplier,
        }

    # 2. Action + Prediction: do(intervention) 用 abducted noise
    cf_samples: dict[str, list[float]] = {t: [] for t in targets}
    factual_samples: dict[str, list[float]] = {t: [] for t in targets}
    for noise in abducted:
        sim_cf = _simulate_with_noise(scm, base_conditions, intervention, noise)
        for t in targets:
            cf_samples[t].append(sim_cf.get(t, 0.0))
        sim_fact = _simulate_with_noise(scm, base_conditions, {}, noise)
        for t in targets:
            factual_samples[t].append(sim_fact.get(t, 0.0))

    # 3. 统计 + lift
    result: dict[str, Any] = {
        "targets": {t: _stats(cf_samples[t]) for t in targets},
        "factual": {t: _stats(factual_samples[t]) for t in targets},
        "evidence": evidence,
        "intervention": intervention,
        "base_conditions": base_conditions,
        "scm_name": scm.name,
        "scm_confirmed": scm.confirmed,
        "scm_source": scm.source,
        "scm_notes": scm.notes,
        "n_abducted": len(abducted),
        "tolerance": tolerance,
        "warning": None,
    }

    lift: dict[str, Any] = {}
    for t in targets:
        m_cf = result["targets"][t]["mean"]
        m_fact = result["factual"][t]["mean"]
        # 阈值 1e-30: diffusion D 可小到 1e-20, 不能用 1e-12 (会把 D 当 0)
        rel = (m_cf - m_fact) / m_fact if abs(m_fact) > 1e-30 else 0.0
        lift[t] = {
            "mean_lift": m_cf - m_fact,
            "relative_lift": rel,
            "significant": abs(rel) > 0.05,
        }
    result["lift"] = lift

    if not scm.confirmed:
        source_str = {
            "template": "KB 模板",
            "llm_draft": "LLM 草稿",
            "fitted": "数据拟合",
            "visual_chain_fit": "视觉因果链拟合",
        }.get(scm.source, scm.source)
        result["warning"] = (
            f"SCM '{scm.name}' 未确认 (source={source_str}). "
            "反事实预测仅供参考, 建议用户审核 SCM 结构后再用于决策."
        )

    return result


# ── HuginnTool 包装 ──────────────────────────────────────────

class CounterfactualRenderInput(BaseModel):
    scm_name: str = Field(
        ...,
        description="SCM 模板名 (sintering/ostwald_ripening/diffusion/phase_transition)",
    )
    evidence: dict[str, float] = Field(
        ...,
        description="观测的 feature 节点值 {particle_size: 200}. 条件节点走 base_conditions.",
    )
    intervention: dict[str, float] = Field(
        ...,
        description="反事实 do(X=v) {T: 1800}",
    )
    targets: list[str] = Field(
        ...,
        description="要预测的节点名列表",
    )
    base_conditions: dict[str, float] = Field(
        default_factory=dict,
        description="evidence 时的实际条件 {T: 1500, t: 2}",
    )
    n_samples: int = Field(200, description="abduction 目标样本数")
    tolerance: float = Field(0.15, description="evidence 匹配容差 (相对误差)")


class CounterfactualRenderTool(HuginnTool):
    """L3 反事实渲染: 给定观测, 问'如果当时 X 不同, Y 会是多少'."""

    name = "counterfactual_render"
    category = "causal"
    description = (
        "Counterfactual reasoning (Pearl L3): given observed evidence, "
        "predict what would have happened under a different intervention. "
        "Uses Monte Carlo abduction to infer noise posterior from evidence, "
        "then predicts targets under do(intervention). "
        "Use this to answer 'what if we had used X instead of X_observed' "
        "questions grounded in physics priors (Arrhenius/Ostwald/Fick/Avrami)."
    )
    input_schema = CounterfactualRenderInput
    read_only = True

    def is_read_only(self, args: CounterfactualRenderInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        input_data = (
            args if isinstance(args, CounterfactualRenderInput)
            else CounterfactualRenderInput(**args)
        )
        if input_data.scm_name not in list_templates():
            return ValidationResult(
                result=False,
                message=f"未知 SCM 模板: {input_data.scm_name}. 可用: {list_templates()}",
            )
        if not input_data.evidence:
            return ValidationResult(result=False, message="evidence 不能为空")
        if not input_data.intervention:
            return ValidationResult(result=False, message="intervention 不能为空")
        if not input_data.targets:
            return ValidationResult(result=False, message="targets 不能为空")
        if input_data.n_samples < 10:
            return ValidationResult(result=False, message="n_samples 至少 10")
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = (
            args if isinstance(args, CounterfactualRenderInput)
            else CounterfactualRenderInput(**args)
        )
        try:
            scm = get_template(input_data.scm_name)
            if scm is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"SCM 模板 '{input_data.scm_name}' 不存在",
                )
            result = counterfactual_render(
                scm=scm,
                evidence=input_data.evidence,
                intervention=input_data.intervention,
                targets=input_data.targets,
                base_conditions=input_data.base_conditions,
                n_samples=input_data.n_samples,
                tolerance=input_data.tolerance,
            )
            success = "error" not in result
            return ToolResult(
                data=result, success=success,
                error=None if success else result.get("error"),
            )
        except Exception as exc:
            logger.warning("counterfactual_render failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── self-check ───────────────────────────────────────────────

def _selfcheck() -> None:
    """12 项 assert 验证 counterfactual_render 核心行为."""
    import math as _math
    import random as _rng
    _rng.seed(42)

    from huginn.causal.visual_scm import (
        _template_diffusion, _template_sintering, _template_ostwald_ripening,
    )
    from huginn.causal.predict_intervention import _simulate_once

    # 1. diffusion 反事实: factual T=800, do(T=1000) → D 升几个数量级
    scm_d = _template_diffusion()
    base_sim = _simulate_once(scm_d, intervention={}, base_conditions={"T": 800})
    D_obs = base_sim["D"]
    r = counterfactual_render(
        scm_d,
        evidence={"D": D_obs},
        intervention={"T": 1000},
        targets=["D"],
        base_conditions={"T": 800},
        n_samples=100,
        tolerance=0.20,
    )
    assert "error" not in r, f"反事实失败: {r.get('error')}"
    assert r["n_abducted"] > 0
    factual_mean = r["factual"]["D"]["mean"]
    cf_mean = r["targets"]["D"]["mean"]
    # factual 应 ≈ D_obs (abducted noise 让模拟匹配 evidence)
    assert abs(factual_mean - D_obs) / D_obs < 0.30, (
        f"factual 应接近 evidence: factual={factual_mean}, evidence={D_obs}"
    )
    # counterfactual 应远大于 factual (T 升 200K, D 升几个数量级)
    assert cf_mean > factual_mean * 10, (
        f"counterfactual 应远大于 factual: cf={cf_mean}, factual={factual_mean}"
    )

    # 2. lift 字段正确
    assert "D" in r["lift"]
    assert r["lift"]["D"]["relative_lift"] > 1.0
    assert r["lift"]["D"]["significant"] is True

    # 3. 逆向反事实: factual T=1000, do(T=800) → D 降
    base_sim2 = _simulate_once(scm_d, intervention={}, base_conditions={"T": 1000})
    D_obs2 = base_sim2["D"]
    r2 = counterfactual_render(
        scm_d,
        evidence={"D": D_obs2},
        intervention={"T": 800},
        targets=["D"],
        base_conditions={"T": 1000},
        n_samples=100,
        tolerance=0.20,
    )
    assert "error" not in r2
    assert r2["targets"]["D"]["mean"] < r2["factual"]["D"]["mean"]

    # 4. evidence 不可达 → abduction 返空, counterfactual_render 返 error
    # T=800 时 D ≈ 1.6e-14, evidence D=1e-5 物理不可能匹配
    abducted_bad = _abduct_noise(
        scm_d, evidence={"D": 1e-5}, base_conditions={"T": 800},
        n_samples=20, tolerance=0.10, max_attempts=200,
    )
    assert len(abducted_bad) == 0, "不可达 evidence 不应有匹配样本"
    r3 = counterfactual_render(
        scm_d,
        evidence={"D": 1e-5},
        intervention={"T": 1000},
        targets=["D"],
        base_conditions={"T": 800},
        n_samples=20,
        tolerance=0.10,
        max_attempts_multiplier=10,  # 加速失败
    )
    assert "error" in r3, f"不可达 evidence 应返 error: {r3}"
    assert "abduction" in r3["error"]

    # 5. evidence 和 intervention 重叠 → 报错
    r4 = counterfactual_render(
        scm_d,
        evidence={"T": 800},     # T 是条件, 不该放 evidence
        intervention={"T": 1000},
        targets=["D"],
        base_conditions={"T": 800},
    )
    assert "error" in r4

    # 6. SCM confirmed=False → 警告
    scm_unconf = _template_diffusion()
    scm_unconf.confirmed = False
    scm_unconf.source = "llm_draft"
    base_sim6 = _simulate_once(scm_unconf, intervention={}, base_conditions={"T": 800})
    r6 = counterfactual_render(
        scm_unconf,
        evidence={"D": base_sim6["D"]},
        intervention={"T": 1000},
        targets=["D"],
        base_conditions={"T": 800},
        n_samples=50,
        tolerance=0.20,
    )
    assert r6["warning"] is not None
    assert "未确认" in r6["warning"]

    # 7. 端到端 Phase 2+3: fit_scm_from_observations → counterfactual_render
    # 用 sintering 真实参数 Ea=100, A0=1e8, r0=30 生成数据 (Phase 2 测试已验证)
    R_const = 8.314e-3
    obs: list[Observation] = []
    for T in [1200, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        for t in [1.0, 4.0]:
            r_true = (30.0**3 + 1e8 * _math.exp(-100.0 / (R_const * T)) * t) ** (1.0/3.0)
            obs.append(Observation(
                conditions={"T": float(T), "t": t},
                features={"particle_size": r_true},
            ))
    fitted_scm, _ = fit_scm_from_observations(obs, "sintering")
    assert fitted_scm.confirmed is False
    assert fitted_scm.source == "visual_chain_fit"

    # 8. 用 fitted_scm 做反事实: T=1500, t=2 → do(T=1800)
    base_sim7 = _simulate_once(fitted_scm, intervention={}, base_conditions={"T": 1500, "t": 2})
    ps_obs = base_sim7["particle_size"]
    r7 = counterfactual_render(
        fitted_scm,
        evidence={"particle_size": ps_obs},
        intervention={"T": 1800},
        targets=["particle_size"],
        base_conditions={"T": 1500, "t": 2},
        n_samples=100,
        tolerance=0.20,
    )
    assert "error" not in r7, f"端到端失败: {r7.get('error')}"
    assert r7["n_abducted"] > 0
    # factual 应匹配 evidence
    assert abs(r7["factual"]["particle_size"]["mean"] - ps_obs) / ps_obs < 0.30
    # counterfactual (T=1800) 应 > factual (T=1500)
    assert r7["targets"]["particle_size"]["mean"] > r7["factual"]["particle_size"]["mean"], (
        f"counterfactual 应大于 factual: cf={r7['targets']['particle_size']['mean']}, "
        f"fact={r7['factual']['particle_size']['mean']}"
    )

    # 9. fitted_scm 反事实有 visual_chain_fit 警告
    assert r7["warning"] is not None
    assert "视觉因果链拟合" in r7["warning"]

    # 10. Tool validate_input 拒绝空 evidence/intervention/targets + 未知 SCM
    import asyncio
    tool = CounterfactualRenderTool()
    loop = asyncio.new_event_loop()
    try:
        r = loop.run_until_complete(tool.validate_input(
            {"scm_name": "diffusion", "evidence": {},
             "intervention": {"T": 1000}, "targets": ["D"]}
        ))
        assert r.result is False
        r = loop.run_until_complete(tool.validate_input(
            {"scm_name": "diffusion", "evidence": {"D": 1},
             "intervention": {}, "targets": ["D"]}
        ))
        assert r.result is False
        r = loop.run_until_complete(tool.validate_input(
            {"scm_name": "diffusion", "evidence": {"D": 1},
             "intervention": {"T": 1000}, "targets": []}
        ))
        assert r.result is False
        r = loop.run_until_complete(tool.validate_input(
            {"scm_name": "nonexistent", "evidence": {"D": 1},
             "intervention": {"T": 1000}, "targets": ["D"]}
        ))
        assert r.result is False
        r = loop.run_until_complete(tool.validate_input(
            {"scm_name": "diffusion", "evidence": {"D": 1},
             "intervention": {"T": 1000}, "targets": ["D"]}
        ))
        assert r.result is True
    finally:
        loop.close()

    # 11. Tool call 成功
    tool2 = CounterfactualRenderTool()
    loop2 = asyncio.new_event_loop()
    try:
        from huginn.types import ToolContext
        ctx = ToolContext(session_id="selfcheck", workspace=".")
        # 用 diffusion 模板, evidence 用模型生成保证可达
        base_sim_t = _simulate_once(_template_diffusion(), intervention={}, base_conditions={"T": 800})
        r = loop2.run_until_complete(tool2.call(
            {"scm_name": "diffusion",
             "evidence": {"D": base_sim_t["D"]},
             "intervention": {"T": 1000},
             "targets": ["D"],
             "base_conditions": {"T": 800},
             "n_samples": 50,
             "tolerance": 0.30},
            ctx,
        ))
        assert r.success, f"Tool call 失败: {r.error}"
        assert "targets" in r.data
        assert r.data["scm_name"] == "diffusion"
    finally:
        loop2.close()

    # 12. _simulate_with_noise 一致性: 同 noise + 同 base 应得同结果
    scm_t = _template_sintering()
    noise_sample = {n: scm_t.noise[n]() for n in scm_t.noise}
    sim_a = _simulate_with_noise(scm_t, {"T": 1500, "t": 2}, {}, noise_sample)
    sim_b = _simulate_with_noise(scm_t, {"T": 1500, "t": 2}, {}, noise_sample)
    assert sim_a == sim_b, "相同 noise 应得相同结果"
    # 干预后条件节点取干预值
    sim_c = _simulate_with_noise(scm_t, {"T": 1500, "t": 2}, {"T": 1800}, noise_sample)
    assert sim_c["T"] == 1800.0

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
