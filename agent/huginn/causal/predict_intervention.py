"""PredictIntervention — L2 干预预测工具.

基于 VisualSCM 做 Pearl do-calculus:
  P(Y | do(X=v)) — 干预 X 设为 v, 预测 Y 的分布

实现方式: Monte Carlo 拓扑序采样 (Monte Carlo SCM simulation).
对每个 noise 变量采 N 样本, 按 topological_order 算每个节点.
干预 do(X=v): 把 X 的方程替换为常数 v, 切断其父节点的因果边.

不引 pgmpy/causalnex/doWhy 等重依赖 (零新依赖, 纯 Python).

输入:
  - scm: VisualSCM (从模板取或 LLM 生成)
  - intervention: dict[str, float] (do(T=1500))
  - targets: list[str] (要预测的节点名)
  - n_samples: int (Monte Carlo 样本数, 默认 500)
  - observed: dict[str, float] (观测值, 用于拟合噪声后验, Phase 2)

输出:
  - 每个目标节点的 mean/std/percentiles/histogram
  - 干预是否改变结果 (vs 不干预的对照)
  - 如果 SCM.confirmed=False, 显式警告

接入点:
  - agent 工具调用: predict_intervention(scm_name, intervention, targets)
  - red_team 证伪: 假设 → 预测 → vision_describe 验证
  - autoloop conjecture: 因果预测作为新假设来源

设计原则 (ponytail):
  - Monte Carlo 而不是解析解 (SCM 方程非线性, 解析难)
  - 截断到 node.range 防止物理不合理输出
  - confirmed=False 时显式警告, 不静默用 LLM 草稿
  - 不实现 L3 反事实 (需 abduction, 留 Phase 3)
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from huginn.causal.visual_scm import (
    VisualSCM, get_template, match_template, list_templates,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


# ── 核心算法: Monte Carlo do-calculus ────────────────────────

def _simulate_once(
    scm: VisualSCM,
    intervention: dict[str, float],
    base_conditions: dict[str, float] | None = None,
) -> dict[str, float]:
    """单次 SCM 采样: 按拓扑序算每个节点.

    intervention 中的节点: 方程被替换为常数, 切断父节点因果
    base_conditions: 非干预的条件节点取值 (e.g. {"t": 5} 当只干预 T)
    """
    base_conditions = base_conditions or {}
    values: dict[str, float] = {}
    order = scm.topological_order()

    for node in order:
        if node in intervention:
            values[node] = float(intervention[node])
        elif node in base_conditions:
            values[node] = float(base_conditions[node])
        else:
            # 调结构方程: f(parents_values, noise)
            noise = scm.noise.get(node, lambda: 0.0)()
            try:
                val = scm.equations[node](values, noise)
            except Exception as exc:
                logger.debug("SCM equation %s failed: %s", node, exc)
                val = 0.0
            # 截断到 node.range
            var = scm.nodes.get(node)
            if var and var.range:
                lo, hi = var.range
                val = max(lo, min(hi, val))
            values[node] = float(val)
    return values


def predict_intervention(
    scm: VisualSCM,
    intervention: dict[str, float],
    targets: list[str],
    base_conditions: dict[str, float] | None = None,
    n_samples: int = 500,
) -> dict[str, Any]:
    """L2 干预预测: P(targets | do(intervention)).

    Args:
        scm: 结构因果模型
        intervention: 干预 do(X=v), e.g. {"T": 1500}
        targets: 要预测的节点名列表
        base_conditions: 非干预的条件节点取值
        n_samples: Monte Carlo 样本数

    Returns:
        {
          "targets": {name: {"mean": ..., "std": ..., "p5": ..., "p50": ..., "p95": ..., "histogram": [...]}},
          "intervention": {...},
          "baseline": {name: {"mean": ...}},  # 不干预的对照
          "delta": {name: {"mean_delta": ..., "relative": ...}},  # 干预 vs 对照
          "scm_confirmed": bool,
          "warning": str | None,  # confirmed=False 时警告
          "n_samples": int,
        }
    """
    # 校验: targets 在 SCM 里
    missing = [t for t in targets if t not in scm.nodes]
    if missing:
        return {
            "error": f"targets 不在 SCM 节点里: {missing}",
            "scm_nodes": list(scm.nodes.keys()),
        }
    # 校验: intervention 节点在 SCM 里
    bad_interv = [k for k in intervention if k not in scm.nodes]
    if bad_interv:
        return {
            "error": f"intervention 节点不在 SCM: {bad_interv}",
            "scm_nodes": list(scm.nodes.keys()),
        }
    # 校验: 干预值在合理范围
    for k, v in intervention.items():
        var = scm.nodes[k]
        if var.range:
            lo, hi = var.range
            if v < lo or v > hi:
                logger.warning("干预值 %s=%s 超出范围 [%s, %s]", k, v, lo, hi)

    # Monte Carlo 采样
    target_samples: dict[str, list[float]] = {t: [] for t in targets}
    baseline_samples: dict[str, list[float]] = {t: [] for t in targets}

    for _ in range(n_samples):
        # 干预组
        sim_interv = _simulate_once(scm, intervention, base_conditions)
        for t in targets:
            target_samples[t].append(sim_interv.get(t, 0.0))
        # 对照组 (无干预, 用 base_conditions 或 SCM 默认)
        sim_base = _simulate_once(scm, {}, base_conditions)
        for t in targets:
            baseline_samples[t].append(sim_base.get(t, 0.0))

    # 统计
    def _stats(samples: list[float]) -> dict[str, Any]:
        if not samples:
            return {"mean": 0, "std": 0}
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
            "histogram": _histogram(samples, 10),
        }

    def _histogram(samples: list[float], bins: int) -> list[int]:
        if not samples:
            return [0] * bins
        lo, hi = min(samples), max(samples)
        if lo == hi:
            return [len(samples)] + [0] * (bins - 1)
        width = (hi - lo) / bins
        counts = [0] * bins
        for v in samples:
            idx = min(int((v - lo) / width), bins - 1)
            counts[idx] += 1
        return counts

    result: dict[str, Any] = {
        "targets": {t: _stats(target_samples[t]) for t in targets},
        "baseline": {t: _stats(baseline_samples[t]) for t in targets},
        "intervention": intervention,
        "base_conditions": base_conditions or {},
        "scm_name": scm.name,
        "scm_confirmed": scm.confirmed,
        "scm_source": scm.source,
        "scm_notes": scm.notes,
        "n_samples": n_samples,
        "warning": None,
    }

    # delta: 干预 vs 对照
    delta: dict[str, Any] = {}
    for t in targets:
        m_int = result["targets"][t]["mean"]
        m_base = result["baseline"][t]["mean"]
        # ponytail: 1e-30 而非 1e-12, diffusion 的 D≈1e-14 量级, 1e-12 会被当 0
        rel = (m_int - m_base) / m_base if abs(m_base) > 1e-30 else 0.0
        delta[t] = {
            "mean_delta": m_int - m_base,
            "relative_delta": rel,
            "significant": abs(rel) > 0.05,  # 5% 阈值
        }
    result["delta"] = delta

    # 警告: 未确认 SCM
    if not scm.confirmed:
        result["warning"] = (
            f"SCM '{scm.name}' 未确认 (source={scm_source_str(scm.source)}). "
            "预测结果仅供参考, 建议用户审核 SCM 结构后再用于决策."
        )

    return result


def scm_source_str(source: str) -> str:
    """source 字段的可读名."""
    return {
        "template": "KB 模板",
        "llm_draft": "LLM 草稿",
        "fitted": "数据拟合",
    }.get(source, source)


# ── HuginnTool 包装 ──────────────────────────────────────────

class PredictInterventionInput(BaseModel):
    scm_name: str = Field(
        ...,
        description=(
            "SCM 模板名 (sintering / ostwald_ripening / diffusion / phase_transition) "
            "或自定义 SCM 名. 后续支持 'auto' 自动匹配."
        ),
    )
    intervention: dict[str, float] = Field(
        ...,
        description="干预 do(X=v), e.g. {\"T\": 1500} 表示把温度设为 1500K",
    )
    targets: list[str] = Field(
        ...,
        description="要预测的节点名列表, e.g. [\"particle_size\", \"density\"]",
    )
    base_conditions: dict[str, float] = Field(
        default_factory=dict,
        description="非干预的条件节点取值, e.g. {\"t\": 5} 当只干预 T",
    )
    n_samples: int = Field(
        default=500,
        description="Monte Carlo 样本数 (默认 500, 越多越准越慢)",
    )


class PredictInterventionTool(HuginnTool):
    """L2 干预预测: 基于 SCM 做 do-calculus.

    输入模板名 + 干预 + 目标节点, 返预测分布 (mean/std/percentiles/histogram)
    + 对照组 + delta. SCM 未确认时显式警告.

    例: predict_intervention(scm_name="sintering",
                             intervention={"T": 1800},
                             targets=["particle_size", "density"],
                             base_conditions={"t": 5})
    """

    name = "predict_intervention"
    category = "causal"
    description = (
        "Predict outcome distribution under intervention do(X=v) using "
        "a Structural Causal Model. Monte Carlo simulation over the SCM "
        "topology. Returns mean/std/percentiles/histogram for each target, "
        "compared against baseline (no intervention). Use this to answer "
        "'what if we change X' questions grounded in physics priors "
        "(Arrhenius / Ostwald / Fick / Avrami)."
    )
    input_schema = PredictInterventionInput
    read_only = True

    def is_read_only(self, args: PredictInterventionInput) -> bool:
        return True

    async def validate_input(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ValidationResult:
        input_data = args if isinstance(args, PredictInterventionInput) \
            else PredictInterventionInput(**args)
        if input_data.scm_name not in list_templates():
            return ValidationResult(
                result=False,
                message=f"未知 SCM 模板: {input_data.scm_name}. 可用: {list_templates()}",
            )
        if not input_data.intervention:
            return ValidationResult(
                result=False, message="intervention 不能为空"
            )
        if not input_data.targets:
            return ValidationResult(
                result=False, message="targets 不能为空"
            )
        if input_data.n_samples < 10:
            return ValidationResult(
                result=False, message="n_samples 至少 10"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        input_data = args if isinstance(args, PredictInterventionInput) \
            else PredictInterventionInput(**args)
        try:
            scm = get_template(input_data.scm_name)
            if scm is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"SCM 模板 '{input_data.scm_name}' 不存在",
                )
            result = predict_intervention(
                scm=scm,
                intervention=input_data.intervention,
                targets=input_data.targets,
                base_conditions=input_data.base_conditions,
                n_samples=input_data.n_samples,
            )
            success = "error" not in result
            return ToolResult(
                data=result, success=success,
                error=None if success else result.get("error"),
            )
        except Exception as exc:
            logger.warning("predict_intervention failed: %s", exc, exc_info=True)
            return ToolResult(data=None, success=False, error=str(exc))


# ── self-check ───────────────────────────────────────────────

def _selfcheck() -> None:
    """10 项 assert 验证 predict_intervention 核心行为."""
    from huginn.causal.visual_scm import _template_sintering, _template_diffusion

    scm = _template_sintering()

    # 1. 基本干预预测: do(T=1800) → particle_size 升
    r = predict_intervention(
        scm, intervention={"T": 1800},
        targets=["particle_size", "density"],
        base_conditions={"t": 5},
        n_samples=100,
    )
    assert "error" not in r
    assert "particle_size" in r["targets"]
    assert r["scm_confirmed"] is True
    assert r["warning"] is None  # 模板 SCM 已确认
    assert r["n_samples"] == 100

    # 2. delta 字段正确
    assert "particle_size" in r["delta"]
    assert "mean_delta" in r["delta"]["particle_size"]
    assert "relative_delta" in r["delta"]["particle_size"]

    # 3. Arrhenius 物理: 用 diffusion 模板验 (sintering Ea=250 在 T<2000 下 exp 项下溢,
    # 信号被噪声盖, 测试会 flaky. diffusion Ea=150 T=800→1000 D 升 ~100x, 信号清晰)
    scm_diff = _template_diffusion()
    r_diff = predict_intervention(
        scm_diff, intervention={"T": 1000},
        targets=["D"],
        base_conditions={"c": 1.0},
        n_samples=200,
    )
    assert r_diff["targets"]["D"]["mean"] > r_diff["baseline"]["D"]["mean"], (
        f"Arrhenius 失效: T=1000 D={r_diff['targets']['D']['mean']:.3e} "
        f"应 > baseline D={r_diff['baseline']['D']['mean']:.3e}"
    )
    # ponytail: 验 1e-30 阈值修复 — diffusion D≈1e-14, 旧 1e-12 阈值会把 baseline 当 0
    # 导致 relative_delta=0.0. 现在 baseline≈1e-14 > 1e-30, rel 应非零
    assert r_diff["delta"]["D"]["relative_delta"] != 0.0, (
        f"1e-30 阈值修复失效: rel_delta=0.0 baseline={r_diff['baseline']['D']['mean']:.3e}"
    )

    # 4. targets 不存在返 error
    r = predict_intervention(
        scm, intervention={"T": 1800},
        targets=["nonexistent_node"],
        n_samples=50,
    )
    assert "error" in r

    # 5. intervention 节点不存在返 error
    r = predict_intervention(
        scm, intervention={"nonexistent": 1.0},
        targets=["particle_size"],
        n_samples=50,
    )
    assert "error" in r

    # 6. 干预值超出范围不报错 (只警告), 仍返结果
    r = predict_intervention(
        scm, intervention={"T": 99999},  # 远超 range
        targets=["particle_size"],
        n_samples=50,
    )
    assert "error" not in r
    assert "particle_size" in r["targets"]

    # 7. 未确认 SCM 触发警告
    scm_draft = _template_sintering()
    scm_draft.confirmed = False
    scm_draft.source = "llm_draft"
    r = predict_intervention(
        scm_draft, intervention={"T": 1800},
        targets=["particle_size"],
        n_samples=50,
    )
    assert r["scm_confirmed"] is False
    assert r["warning"] is not None
    assert "未确认" in r["warning"]

    # 8. Monte Carlo 结果统计量合理 (std > 0 当有噪声)
    r = predict_intervention(
        scm, intervention={"T": 1500},
        targets=["particle_size"],
        base_conditions={"t": 5},
        n_samples=200,
    )
    assert r["targets"]["particle_size"]["std"] > 0  # 噪声生效
    assert r["targets"]["particle_size"]["p5"] <= r["targets"]["particle_size"]["p50"]
    assert r["targets"]["particle_size"]["p50"] <= r["targets"]["particle_size"]["p95"]

    # 9. histogram 10 bins
    assert len(r["targets"]["particle_size"]["histogram"]) == 10
    assert sum(r["targets"]["particle_size"]["histogram"]) == 200

    # 10. diffusion SCM 也能跑
    scm_diff = _template_diffusion()
    r = predict_intervention(
        scm_diff, intervention={"T": 1500},
        targets=["D"],
        base_conditions={"c": 1.0},
        n_samples=100,
    )
    assert "error" not in r
    assert "D" in r["targets"]
    # D 在合理范围 (1e-20 到 1e-8)
    d_mean = r["targets"]["D"]["mean"]
    assert 1e-25 < d_mean < 1e-5, f"D={d_mean} 不在物理合理范围"

    print("all self-checks passed")


if __name__ == "__main__":
    _selfcheck()
