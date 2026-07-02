"""UQ 全链路传播 — 沿研究阶段串联误差棒.

M3 (W4): autoloop 的 EXECUTION→VALIDATION→REPORTING 三阶段每步都可能带
不确定度, 单步的 GUM 传播 (uq_tool.propagate) 只管一个表达式. 这里把多个
stage 串成 pipeline, 每步要么直接给 (value, sigma) 测量值, 要么给一个依赖
上游 stage 的 sympy 表达式, pipeline 按拓扑序逐级传播.

两种传播方式:
- linear:  GUM 一阶, u_c² = Σ (∂f/∂x_i)² u_i²  (含相关项)
- monte_carlo:  从上游 Gaussian 采样, 过表达式, 取输出 std

典型用法::

    pipe = UQPipeline()
    pipe.add_stage(UQStage(name="encut", value=520, sigma=5))
    pipe.add_stage(UQStage(name="kpoints", value=0.03, sigma=0.005))
    pipe.add_stage(UQStage(
        name="bandgap",
        expression="0.5 * encut / 1000 + 10 * kpoints",
        dependencies=["encut", "kpoints"],
        method="monte_carlo",
    ))
    results = pipe.run()
    print(results["bandgap"])  # UQResult(value=.., sigma=.., contribution=..)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


# ── 数据结构 ────────────────────────────────────────────────────────────────


@dataclass
class UQStage:
    """pipeline 里的一个传播节点.

    value/sigma 直接给测量值时, expression 留空.
    用 expression 时, dependencies 列出引用的 upstream stage 名,
    表达式里直接用 stage 名当变量名 (sympy 兼容).
    """

    name: str
    value: float | None = None
    sigma: float = 0.0
    expression: str | None = None
    dependencies: list[str] = field(default_factory=list)
    method: Literal["linear", "monte_carlo"] = "linear"
    n_samples: int = 1000
    # 相关系数, key 用 "a_b" 跟 uq_tool 一致; linear 才用
    correlations: dict[str, float] | None = None


@dataclass
class UQResult:
    """一个 stage 传播后的结果."""

    name: str
    value: float
    sigma: float
    method: str
    # 每个上游对 sigma² 的贡献百分比, 方便定位误差主源
    contribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "sigma": self.sigma,
            "method": self.method,
            "contribution": dict(self.contribution),
        }


# ── 异常 ────────────────────────────────────────────────────────────────────


class UQPipelineError(Exception):
    """pipeline 拓扑/依赖/求值错误."""


# sympify 内部走 eval, 拦掉危险模式做 defense-in-depth
_DANGEROUS_PATTERNS = (
    "__", "import", "exec", "eval", "open(", "os.", "sys.",
    "subprocess", "globals", "locals", "getattr", "setattr",
)


# ── pipeline ────────────────────────────────────────────────────────────────


class UQPipeline:
    """多阶段不确定度传播.

    add_stage 注册节点, run() 按拓扑序逐级算 (value, sigma).
    环依赖 / 缺依赖 / 空 stage 名 都在 run() 时抛 UQPipelineError.
    """

    def __init__(self) -> None:
        self._stages: dict[str, UQStage] = {}
        self._order: list[str] = []

    def add_stage(self, stage: UQStage) -> None:
        if not stage.name:
            raise UQPipelineError("stage 名不能为空")
        if stage.name in self._stages:
            raise UQPipelineError(f"stage '{stage.name}' 已存在")
        if stage.expression is not None:
            self._validate_expression(stage.expression)
        self._stages[stage.name] = stage

    @staticmethod
    def _validate_expression(expr: str) -> None:
        """拦掉可能触发代码执行的模式, defense-in-depth."""
        low = expr.lower()
        for pat in _DANGEROUS_PATTERNS:
            if pat.lower() in low:
                raise UQPipelineError(
                    f"表达式包含禁用序列 '{pat}'"
                )

    def stages(self) -> list[UQStage]:
        return [self._stages[n] for n in self._order]

    def _topo_sort(self) -> list[str]:
        """Kahn 算法排拓扑序. 有环抛 UQPipelineError."""
        # 只检查 dependencies 里引用的 stage 是否都注册了
        for s in self._stages.values():
            for dep in s.dependencies:
                if dep not in self._stages:
                    raise UQPipelineError(
                        f"stage '{s.name}' 依赖未注册的 stage '{dep}'"
                    )

        in_deg = {n: 0 for n in self._stages}
        adj: dict[str, list[str]] = {n: [] for n in self._stages}
        for s in self._stages.values():
            for dep in s.dependencies:
                adj[dep].append(s.name)
                in_deg[s.name] += 1

        queue = [n for n, d in in_deg.items() if d == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in adj[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        if len(order) != len(self._stages):
            remaining = [n for n, d in in_deg.items() if d > 0]
            raise UQPipelineError(f"检测到环依赖, 涉及: {remaining}")
        return order

    def run(self, seed: int | None = None) -> dict[str, UQResult]:
        """按拓扑序逐级传播, 返回每个 stage 的 UQResult."""
        self._order = self._topo_sort()
        results: dict[str, UQResult] = {}
        for name in self._order:
            stage = self._stages[name]
            results[name] = self._eval_stage(stage, results, seed)
        return results

    def _eval_stage(
        self,
        stage: UQStage,
        results: dict[str, UQResult],
        seed: int | None,
    ) -> UQResult:
        # 直接测量值: 不用表达式, 直接包成 UQResult
        if stage.expression is None:
            if stage.value is None:
                raise UQPipelineError(
                    f"stage '{stage.name}' 既没 expression 也没 value"
                )
            return UQResult(
                name=stage.name,
                value=float(stage.value),
                sigma=float(stage.sigma),
                method="direct",
            )

        # 表达式 stage: 必须有依赖
        if not stage.dependencies:
            raise UQPipelineError(
                f"stage '{stage.name}' 有 expression 但没 dependencies"
            )

        if stage.method == "monte_carlo":
            return self._propagate_mc(stage, results, seed)
        return self._propagate_linear(stage, results)

    def _propagate_linear(
        self, stage: UQStage, results: dict[str, UQResult]
    ) -> UQResult:
        """GUM 一阶: u_c² = Σ (∂f/∂x_i)² u_i² + 相关交叉项."""
        try:
            import sympy as sp
        except ImportError as exc:
            raise UQPipelineError(
                "linear 传播需要 sympy. pip install sympy"
            ) from exc

        deps = stage.dependencies
        symbols = {d: sp.Symbol(d) for d in deps}
        expr = sp.sympify(stage.expression, locals=symbols)

        # 标称值求值
        nominal = {d: results[d].value for d in deps}
        f = sp.lambdify([symbols[d] for d in deps], expr, modules="numpy")
        value = float(f(*[nominal[d] for d in deps]))

        # 偏导 × 上游 sigma
        sens: dict[str, float] = {}
        for d in deps:
            deriv = sp.diff(expr, symbols[d])
            df = sp.lambdify([symbols[d] for d in deps], deriv, modules="numpy")
            sens[d] = float(df(*[nominal[d] for d in deps]))

        uc_sq = 0.0
        contribution: dict[str, float] = {}
        for d in deps:
            term_sq = (sens[d] * results[d].sigma) ** 2
            uc_sq += term_sq

        # 相关交叉项
        corr = stage.correlations or {}
        for i, di in enumerate(deps):
            for dj in deps[i + 1:]:
                r = self._lookup_corr(corr, di, dj)
                if r == 0.0:
                    continue
                cross = 2.0 * sens[di] * sens[dj] * results[di].sigma * results[dj].sigma * r
                uc_sq += cross

        uc_sq = max(uc_sq, 0.0)
        sigma = float(np.sqrt(uc_sq))

        for d in deps:
            term_sq = (sens[d] * results[d].sigma) ** 2
            contribution[d] = (term_sq / uc_sq * 100.0) if uc_sq > 0 else 0.0

        return UQResult(
            name=stage.name,
            value=value,
            sigma=sigma,
            method="linear",
            contribution=contribution,
        )

    def _propagate_mc(
        self, stage: UQStage, results: dict[str, UQResult], seed: int | None
    ) -> UQResult:
        """蒙特卡洛: 从上游 Gaussian 采样, 过表达式, 取输出 mean/std."""
        try:
            import sympy as sp
        except ImportError as exc:
            raise UQPipelineError(
                "monte_carlo 传播需要 sympy. pip install sympy"
            ) from exc

        deps = stage.dependencies
        symbols = {d: sp.Symbol(d) for d in deps}
        expr = sp.sympify(stage.expression, locals=symbols)
        f = sp.lambdify([symbols[d] for d in deps], expr, modules="numpy")

        rng = np.random.default_rng(
            None if seed is None
            else (seed + sum(ord(c) for c in stage.name)) % (2**32)
        )
        n = max(stage.n_samples, 10)
        # 每个依赖从其 (value, sigma) Gaussian 采样
        samples: dict[str, np.ndarray] = {}
        for d in deps:
            r = results[d]
            samples[d] = rng.normal(r.value, r.sigma, size=n)

        # 广播求值
        out = f(*[samples[d] for d in deps])
        out = np.asarray(out, dtype=float).ravel()

        value = float(np.mean(out))
        sigma = float(np.std(out, ddof=1)) if len(out) > 1 else 0.0

        # 贡献度: 用方差分解 (每个输入的扰动单独传播, 看输出方差变化占比)
        contribution: dict[str, float] = {}
        total_var = sigma**2 if sigma > 0 else 1e-30
        for d in deps:
            # 冻结其它输入在均值, 只抖动 d, 量输出方差
            frozen = {od: np.full(n, results[od].value) for od in deps}
            frozen[d] = samples[d]
            out_d = f(*[frozen[od] for od in deps])
            out_d = np.asarray(out_d, dtype=float).ravel()
            var_d = float(np.var(out_d, ddof=1)) if len(out_d) > 1 else 0.0
            contribution[d] = (var_d / total_var * 100.0) if total_var > 0 else 0.0

        return UQResult(
            name=stage.name,
            value=value,
            sigma=sigma,
            method="monte_carlo",
            contribution=contribution,
        )

    @staticmethod
    def _lookup_corr(
        corr: dict[str, float], a: str, b: str
    ) -> float:
        """查相关系数, 支持 'a_b' / 'a-b' / 'b_a' 多种 key 形式."""
        for key in (f"{a}_{b}", f"{a}-{b}", f"{b}_{a}", f"{b}-{a}"):
            if key in corr:
                return float(corr[key])
        return 0.0
