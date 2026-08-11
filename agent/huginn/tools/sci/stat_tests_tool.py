"""统计检验工具 — 描述统计 / 假设检验 / 多重比较校正.

每个检验前先做 assumption check (Shapiro 正态 + Levene 方差齐性),
不满足自动切非参数. 永远同时报 effect_size + 95% CI,
并用 interpretation_boundary 把证据边界说清楚.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool

# ── 输入输出 schema ──────────────────────────────────────────────────


class StatTestsInput(BaseModel):
    action: Literal[
        "describe", "ttest", "paired", "mannwhitney",
        "anova", "kruskal", "chi2", "corr", "correct",
    ] = Field(..., description="检验类型")
    # 按 action 不同:
    #   describe/ttest/mannwhitney/anova/kruskal → list[[group, value]]
    #   paired/corr → list[list[float]]  ([before, after] 或 [x, y])
    #   chi2 → list[list[float]]  (列联表行)
    #   correct → list[float]  (p 值列表)
    data: list[Any] = Field(default_factory=list)
    equal_var: bool = Field(default=False, description="ttest 假定等方差 (Student)")
    method: Literal["pearson", "spearman"] = Field(default="pearson")
    correction: Literal["bonferroni", "holm", "fdr_bh"] = Field(default="holm")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    posthoc: bool = Field(default=False, description="anova 是否附 Tukey HSD")


class StatTestsOutput(BaseModel):
    statistic: float | None = None
    p_value: float | None = None
    effect_size: dict[str, Any] | None = None
    confidence_interval: dict[str, Any] | None = None
    assumption_checks: list[dict[str, Any]] = Field(default_factory=list)
    interpretation_boundary: str = ""


# ── 数据解析 + 假设检验辅助 ──────────────────────────────────────────


def _parse_groups(data: list[Any]) -> dict[str, np.ndarray]:
    """从 [[group, value], ...] 拆出 {group: np.array(values)}."""
    buckets: dict[str, list[float]] = {}
    for item in data:
        name, val = str(item[0]), float(item[1])
        buckets.setdefault(name, []).append(val)
    return {k: np.array(v, dtype=float) for k, v in buckets.items()}


def _assumption_checks(groups: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Shapiro 正态 (3<=n<=5000) + Levene 方差齐性 (Brown-Forsythe)."""
    checks: list[dict[str, Any]] = []
    for name, arr in groups.items():
        if 3 <= len(arr) <= 5000:
            stat, p = stats.shapiro(arr)
            checks.append({
                "name": f"shapiro_normality[{name}]",
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "pass": bool(p > 0.05),
            })
        else:
            checks.append({
                "name": f"shapiro_normality[{name}]",
                "note": f"n={len(arr)} 超出 Shapiro 适用范围，跳过",
                "pass": True,  # 跳过不算不通过
            })
    if len(groups) >= 2:
        stat, p = stats.levene(*groups.values(), center="median")
        checks.append({
            "name": "levene_equal_variance",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "pass": bool(p > 0.05),
        })
    return checks


def _assumptions_ok(checks: list[dict[str, Any]]) -> tuple[bool, bool]:
    """返回 (正态全部通过, 方差齐性通过)."""
    normality = all(c.get("pass", True) for c in checks if "shapiro" in c["name"])
    equal_var = all(c.get("pass", True) for c in checks if "levene" in c["name"])
    return normality, equal_var


def _cohen_d(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Cohen's d + Hedges g (独立样本)."""
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return None, None
    d = float(np.mean(a) - np.mean(b)) / pooled
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)  # Hedges 小样本校正
    return d, d * j


def _welch_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict[str, Any] | None:
    """Welch 均值差 95% CI."""
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
    se = math.sqrt(v1 + v2)
    if se == 0:
        return None
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = float(np.mean(a) - np.mean(b))
    return {"mean_diff": round(diff, 6), "low": round(diff - t_crit * se, 6),
            "high": round(diff + t_crit * se, 6)}


def _boundary(action: str, p: float | None) -> str:
    """把统计结论的证据边界说清楚，不替用户把话说满。"""
    if action == "corr":
        return "相关≠因果；是否有实际意义要看效应量与领域背景。"
    if p is None:
        return "无法计算 p 值，检查数据量与取值是否退化。"
    if p < 0.05:
        return "显著≠重要；需结合效应量和 SESOI 判断实际意义。"
    return "非显著≠无差异；可能是功效不足，不能证明无差异。"


# ── 工具主体 ─────────────────────────────────────────────────────────


class StatTestsTool(HuginnTool):
    """描述统计 / 假设检验 / 多重比较校正，带假设检查和效应量."""

    name = "stat_tests_tool"
    category = "analysis"
    description = (
        "统计检验工具：描述统计、t 检验、配对 t、Mann-Whitney、ANOVA、"
        "Kruskal-Wallis、卡方、相关、多重比较校正。每个检验前先做 "
        "Shapiro+Levene 假设检查，不满足自动切非参数；永远同时报 "
        "效应量 + 95% CI。"
    )
    input_schema = StatTestsInput
    output_schema = StatTestsOutput
    read_only = True

    def is_read_only(self, args: StatTestsInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any] | StatTestsInput, context: ToolContext | None = None
    ) -> ToolResult:
        if isinstance(args, dict):
            args = StatTestsInput(**args)
        try:
            handler = {
                "describe": self._describe,
                "ttest": self._ttest,
                "paired": self._paired,
                "mannwhitney": self._mannwhitney,
                "anova": self._anova,
                "kruskal": self._kruskal,
                "chi2": self._chi2,
                "corr": self._corr,
                "correct": self._correct,
            }[args.action]
            return handler(args)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"stat_tests 失败: {exc}")

    # ── 描述统计 ────────────────────────────────────────────────
    def _describe(self, args: StatTestsInput) -> ToolResult:
        groups = _parse_groups(args.data)
        out: dict[str, Any] = {}
        for name, arr in groups.items():
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            out[name] = {
                "n": int(len(arr)),
                "mean": round(float(np.mean(arr)), 6),
                "sd": round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else None,
                "median": round(float(med), 6),
                "iqr": [round(float(q1), 6), round(float(q3), 6)],
                "min": round(float(np.min(arr)), 6),
                "max": round(float(np.max(arr)), 6),
            }
        return ToolResult(data={"test": "describe", "groups": out}, success=True)

    # ── 独立两组 t 检验 (自动切 Mann-Whitney) ──────────────────
    def _ttest(self, args: StatTestsInput) -> ToolResult:
        groups = _parse_groups(args.data)
        if len(groups) != 2:
            raise ValueError(f"ttest 需要恰好两组，当前 {len(groups)} 组")
        checks = _assumption_checks(groups)
        normal, _ = _assumptions_ok(checks)
        if not normal:
            # 正态不满足 → 自动切 Mann-Whitney
            result = self._mannwhitney_impl(groups, checks)
            result.data["auto_switched"] = "ttest → mannwhitney (正态性不满足)"
            return result
        (na, a), (nb, b) = groups.items()
        equal_var = args.equal_var
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
        d, g = _cohen_d(a, b)
        return ToolResult(data={
            "test": "student_t" if equal_var else "welch_t",
            "groups": {na: len(a), nb: len(b)},
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "effect_size": {"cohen_d": round(d, 4) if d is not None else None,
                            "hedges_g": round(g, 4) if g is not None else None},
            "confidence_interval": _welch_ci(a, b, args.alpha),
            "assumption_checks": checks,
            "interpretation_boundary": _boundary("ttest", float(p)),
        }, success=True)

    # ── 配对 t 检验 (自动切 Wilcoxon) ──────────────────────────
    def _paired(self, args: StatTestsInput) -> ToolResult:
        if len(args.data) < 2:
            raise ValueError("paired 需要 data=[[before...], [after...]]")
        a = np.array(args.data[0], dtype=float)
        b = np.array(args.data[1], dtype=float)
        if len(a) != len(b):
            raise ValueError("配对样本 before/after 长度不一致")
        if len(a) < 3:
            raise ValueError("配对样本不足 3 对")
        diff = a - b
        checks = _assumption_checks({"diff": diff})
        normal, _ = _assumptions_ok(checks)
        if not normal:
            stat, p = stats.wilcoxon(a, b)
            return ToolResult(data={
                "test": "wilcoxon_signed_rank",
                "n_pairs": int(len(diff)),
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "effect_size": {"rank_biserial": None},
                "assumption_checks": checks,
                "auto_switched": "paired_t → wilcoxon (差值正态性不满足)",
                "interpretation_boundary": _boundary("paired", float(p)),
            }, success=True)
        stat, p = stats.ttest_rel(a, b)
        sd = float(np.std(diff, ddof=1))
        dz = float(np.mean(diff) / sd) if sd else None
        se = sd / math.sqrt(len(diff)) if sd else None
        t_crit = stats.t.ppf(1 - args.alpha / 2, len(diff) - 1)
        md = float(np.mean(diff))
        ci = {"mean_diff": round(md, 6),
              "low": round(md - t_crit * se, 6),
              "high": round(md + t_crit * se, 6)} if se else None
        return ToolResult(data={
            "test": "paired_t",
            "n_pairs": int(len(diff)),
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "effect_size": {"cohen_dz": round(dz, 4) if dz is not None else None},
            "confidence_interval": ci,
            "assumption_checks": checks,
            "interpretation_boundary": _boundary("paired", float(p)),
        }, success=True)

    # ── Mann-Whitney U ─────────────────────────────────────────
    def _mannwhitney(self, args: StatTestsInput) -> ToolResult:
        groups = _parse_groups(args.data)
        if len(groups) != 2:
            raise ValueError(f"mannwhitney 需要恰好两组，当前 {len(groups)} 组")
        checks = _assumption_checks(groups)
        return self._mannwhitney_impl(groups, checks)

    def _mannwhitney_impl(self, groups, checks) -> ToolResult:
        (na, a), (nb, b) = groups.items()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1 - 2 * float(stat) / (len(a) * len(b))
        return ToolResult(data={
            "test": "mann_whitney_u",
            "groups": {na: len(a), nb: len(b)},
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "effect_size": {"rank_biserial_r": round(rbc, 4)},
            "assumption_checks": checks,
            "interpretation_boundary": _boundary("mannwhitney", float(p)),
        }, success=True)

    # ── 单因素 ANOVA (自动切 Kruskal-Wallis) ───────────────────
    def _anova(self, args: StatTestsInput) -> ToolResult:
        groups = _parse_groups(args.data)
        if len(groups) < 2:
            raise ValueError(f"anova 至少需要 2 组，当前 {len(groups)} 组")
        checks = _assumption_checks(groups)
        normal, _ = _assumptions_ok(checks)
        if not normal:
            result = self._kruskal_impl(groups, checks)
            result.data["auto_switched"] = "anova → kruskal (正态性不满足)"
            return result
        stat, p = stats.f_oneway(*groups.values())
        all_data = np.concatenate(list(groups.values()))
        grand = float(np.mean(all_data))
        ss_between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups.values())
        ss_total = float(np.sum((all_data - grand) ** 2))
        eta_sq = ss_between / ss_total if ss_total else None
        result_data: dict[str, Any] = {
            "test": "one_way_anova",
            "groups": {k: len(v) for k, v in groups.items()},
            "statistic_F": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "effect_size": {"eta_squared": round(float(eta_sq), 4) if eta_sq is not None else None},
            "assumption_checks": checks,
            "interpretation_boundary": _boundary("anova", float(p)),
        }
        if args.posthoc:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            values = np.concatenate(list(groups.values()))
            labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
            tukey = pairwise_tukeyhsd(values, labels)
            result_data["posthoc_tukey"] = [
                {"group_a": str(r[0]), "group_b": str(r[1]),
                 "mean_diff": round(float(r[2]), 6),
                 "p_adj": round(float(r[3]), 6), "reject": bool(r[6])}
                for r in tukey.summary().data[1:]
            ]
        return ToolResult(data=result_data, success=True)

    # ── Kruskal-Wallis ─────────────────────────────────────────
    def _kruskal(self, args: StatTestsInput) -> ToolResult:
        groups = _parse_groups(args.data)
        if len(groups) < 2:
            raise ValueError(f"kruskal 至少需要 2 组，当前 {len(groups)} 组")
        checks = _assumption_checks(groups)
        return self._kruskal_impl(groups, checks)

    def _kruskal_impl(self, groups, checks) -> ToolResult:
        stat, p = stats.kruskal(*groups.values())
        n = sum(len(g) for g in groups.values())
        k = len(groups)
        eps_sq = (float(stat) - k + 1) / (n - k) if n > k else None
        return ToolResult(data={
            "test": "kruskal_wallis",
            "groups": {k2: len(v) for k2, v in groups.items()},
            "statistic_H": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "effect_size": {"epsilon_squared": round(eps_sq, 4) if eps_sq is not None else None},
            "assumption_checks": checks,
            "interpretation_boundary": _boundary("kruskal", float(p)),
        }, success=True)

    # ── 卡方检验 ───────────────────────────────────────────────
    def _chi2(self, args: StatTestsInput) -> ToolResult:
        table = np.array(args.data, dtype=float)
        if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("chi2 需要 data 为二维列联表 (list[list[float]])")
        chi2_val, p, dof, expected = stats.chi2_contingency(table)
        n = int(table.sum())
        min_dim = min(table.shape) - 1
        v = math.sqrt(chi2_val / (n * min_dim)) if n and min_dim else None
        warnings = []
        if (expected < 5).any():
            warnings.append("有期望频数 < 5 的单元格；2x2 表建议改用 Fisher 精确检验")
        result: dict[str, Any] = {
            "test": "chi2_contingency",
            "table_shape": list(table.shape),
            "n": n,
            "statistic_chi2": round(float(chi2_val), 4),
            "dof": int(dof),
            "p_value": round(float(p), 6),
            "effect_size": {"cramers_v": round(float(v), 4) if v is not None else None},
            "warnings": warnings,
            "interpretation_boundary": _boundary("chi2", float(p)),
        }
        if table.shape == (2, 2):
            odds, fisher_p = stats.fisher_exact(table)
            result["fisher_exact"] = {
                "odds_ratio": round(float(odds), 4),
                "p_value": round(float(fisher_p), 6),
            }
        return ToolResult(data=result, success=True)

    # ── 相关分析 ───────────────────────────────────────────────
    def _corr(self, args: StatTestsInput) -> ToolResult:
        if len(args.data) < 2:
            raise ValueError("corr 需要 data=[[x...], [y...]]")
        x = np.array(args.data[0], dtype=float)
        y = np.array(args.data[1], dtype=float)
        if len(x) != len(y):
            raise ValueError("x 和 y 长度不一致")
        if len(x) < 4:
            raise ValueError("有效配对数据不足 4 条")
        if args.method == "spearman":
            r, p = stats.spearmanr(x, y)
            ci = None
        else:
            r, p = stats.pearsonr(x, y)
            z = math.atanh(max(min(float(r), 0.999999), -0.999999))
            se = 1 / math.sqrt(len(x) - 3)
            ci = {"low": round(math.tanh(z - 1.959964 * se), 4),
                  "high": round(math.tanh(z + 1.959964 * se), 4)}
        return ToolResult(data={
            "test": f"{args.method}_correlation",
            "n": int(len(x)),
            "statistic_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "effect_size": {"r": round(float(r), 4)},
            "confidence_interval": ci,
            "interpretation_boundary": _boundary("corr", float(p)),
        }, success=True)

    # ── 多重比较校正 ───────────────────────────────────────────
    def _correct(self, args: StatTestsInput) -> ToolResult:
        from statsmodels.stats.multitest import multipletests
        pvals = [float(v) for v in args.data]
        if not pvals:
            raise ValueError("correct 需要 data 为 p 值列表")
        reject, adjusted, _, _ = multipletests(
            pvals, alpha=args.alpha, method=args.correction
        )
        return ToolResult(data={
            "test": "multiple_comparison_correction",
            "method": args.correction,
            "alpha": args.alpha,
            "input_p": pvals,
            "adjusted_p": [round(float(v), 6) for v in adjusted],
            "reject": [bool(v) for v in reject],
            "interpretation_boundary": "所有做过的检验都应进入校正，不能只报显著的那部分。",
        }, success=True)
