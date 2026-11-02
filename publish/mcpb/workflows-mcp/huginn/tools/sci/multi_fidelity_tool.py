"""multi_fidelity_tool — 多保真融合: 跨保真度数据源拟合代理模型 + 主动学习选点.

M2 (W3): 把不同保真度 (DFT / ML 势 / 经验公式 / 实验值) 的数据源注册进来,
用自回归 co-Kriging (Kennedy-O'Hagan 2000) 拟合多保真代理模型, 传播不确定度,
按成本加权选下一个评估点 (cost-aware EI).

模型: y_high(x) = rho * y_low(x) + delta(x) + noise
  - GP_low  拟合低保真数据
  - GP_delta 拟合 (y_high - rho * y_low_pred) 残差
  - rho 用最小二乘估计

actions:
- register_source: 注册数据源 (保真等级 + 每点成本 + 已有数据)
- fit_surrogate:  拟合多保真 GP
- propagate:      预测 + 不确定度传播
- select_next:    cost-aware 主动学习选点
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.gp_tool import NumPyGP
from huginn.tools.neural_proxy import NeuralPDEProxy
from huginn.tools.profile import ToolProfile

# ── data structures ──────────────────────────────────────────────────────────


@dataclass
class FidelitySource:
    """一个保真度数据源."""

    name: str
    level: int  # 0=最低保真, 越高越准
    cost: float  # 每次评估的代价 (秒或相对值)
    X: np.ndarray  # (n, d)
    y: np.ndarray  # (n,)


@dataclass
class MultiFidelitySurrogate:
    """自回归多保真 GP: y_high = rho * y_low + delta."""

    sources: list[FidelitySource]
    rho: float
    low_gp: NumPyGP
    delta_gp: NumPyGP | None  # 只有两级以上才有
    fitted: bool = False

    def predict(
        self, X: np.ndarray, level: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """在指定保真级预测. level=None 用最高级."""
        if not self.fitted:
            raise RuntimeError("surrogate 未拟合")
        if level is None:
            level = max(s.level for s in self.sources)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 从低到高逐级预测
        mu, var = self.low_gp.predict(X)
        var = var**2  # predict 返回 sigma, 转回 var 做传播
        if self.delta_gp is not None and level >= 1:
            d_mu, d_sigma = self.delta_gp.predict(X)
            mu = self.rho * mu + d_mu
            var = self.rho**2 * var + d_sigma**2
        return mu, np.sqrt(np.maximum(var, 0.0))


# ── pydantic input ───────────────────────────────────────────────────────────


class MultiFidelityInput(BaseModel):
    action: Literal[
        "register_source",
        "fit_surrogate",
        "propagate",
        "select_next",
        "bayesian_calibrate",
        "nested_doe",
        "variance_reduction",
        "neural_proxy",
    ] = Field(
        description=(
            "register_source: 注册保真数据源; "
            "fit_surrogate: 拟合多保真 GP; "
            "propagate: 预测 + 不确定度传播; "
            "select_next: cost-aware 主动学习选点; "
            "bayesian_calibrate: Kennedy-O'Hagan 2000 Bayesian model calibration (MCMC); "
            "nested_doe: 嵌套空间填充 DOE (LF ⊃ HF); "
            "variance_reduction: 控制变量方差缩减; "
            "neural_proxy: Transolver 神经 PDE 快速预估 + FEM 校验"
        )
    )
    name: str | None = Field(default=None, description="register_source: 数据源名称")
    level: int | None = Field(
        default=None, description="register_source: 保真等级 (0=最低)"
    )
    cost: float | None = Field(
        default=None, description="register_source: 每次评估代价"
    )
    X: list[list[float]] | None = Field(
        default=None, description="register_source: 输入点"
    )
    y: list[float] | None = Field(default=None, description="register_source: 输出值")
    X_new: list[list[float]] | None = Field(
        default=None, description="propagate: 预测点"
    )
    candidates: list[list[float]] | None = Field(
        default=None, description="select_next: 候选点集"
    )
    n_select: int = Field(default=1, description="select_next: 选几个点")
    maximize: bool = Field(default=True, description="select_next: 最大化还是最小化")
    length_scale: float = Field(default=1.0, description="fit_surrogate: GP 核长度")
    sigma_f: float = Field(default=1.0, description="fit_surrogate: GP 信号方差")
    sigma_n: float = Field(default=1e-4, description="fit_surrogate: GP 噪声")
    # bayesian_calibrate 专用
    X_hf: list[list[float]] | None = Field(
        default=None, description="bayesian_calibrate: 高保真输入点"
    )
    y_hf: list[float] | None = Field(
        default=None, description="bayesian_calibrate: 高保真观测值"
    )
    X_lf: list[list[float]] | None = Field(
        default=None, description="bayesian_calibrate: 低保真输入点"
    )
    y_lf: list[float] | None = Field(
        default=None, description="bayesian_calibrate: 低保真观测值"
    )
    theta_prior_low: list[float] | None = Field(
        default=None, description="bayesian_calibrate: 校准参数 θ 先验下界"
    )
    theta_prior_high: list[float] | None = Field(
        default=None, description="bayesian_calibrate: 校准参数 θ 先验上界"
    )
    theta_init: list[float] | None = Field(
        default=None, description="bayesian_calibrate: θ 初始值"
    )
    n_mcmc_samples: int = Field(
        default=1000,
        ge=100,
        le=20000,
        description="bayesian_calibrate: MCMC 采样数 (含 burn-in)",
    )
    n_burnin: int = Field(
        default=200,
        ge=0,
        le=5000,
        description="bayesian_calibrate: burn-in 样本数",
    )
    proposal_std: float = Field(
        default=0.1,
        gt=0.0,
        description="bayesian_calibrate: 随机游走 proposal 标准差",
    )
    # nested_doe 专用
    n_hf: int = Field(default=5, ge=1, description="nested_doe: 高保真点数")
    n_lf: int = Field(default=20, ge=1, description="nested_doe: 低保真点数 (>= n_hf)")
    dim: int = Field(default=2, ge=1, description="nested_doe: 输入维数")
    bounds_low: list[float] | None = Field(
        default=None, description="nested_doe: 每维下界"
    )
    bounds_high: list[float] | None = Field(
        default=None, description="nested_doe: 每维上界"
    )
    seed: int | None = Field(default=None, description="nested_doe: 随机种子")
    # variance_reduction 专用
    y_hf_samples: list[float] | None = Field(
        default=None, description="variance_reduction: 高保真样本"
    )
    y_lf_samples: list[float] | None = Field(
        default=None, description="variance_reduction: 对应低保真样本 (同长度)"
    )
    beta: float | None = Field(
        default=None,
        description="variance_reduction: 控制系数 β, None=自动最优",
    )
    # neural_proxy 专用
    mesh_data: dict | None = Field(
        default=None,
        description="neural_proxy: mesh 数据 {'nodes': [[x,y],...], 'elements': ...}",
    )
    boundary_conditions: dict | None = Field(
        default=None,
        description="neural_proxy: 边界条件 {'type': 'dirichlet', 'values': [...]}",
    )
    model_path: str | None = Field(
        default=None,
        description="neural_proxy: 训练好的 Transolver 权重路径",
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _mcmc_log_likelihood(
    th: np.ndarray,
    theta_mid: np.ndarray,
    rho: float,
    low_pred_hf: np.ndarray,
    delta_pred: np.ndarray,
    y_hf: np.ndarray,
    sigma_n: float,
) -> float:
    """theta 影响: y_lf 尺度因子 (1 + sum(th - theta_mid))."""
    scale = 1.0 + float(np.sum(th - theta_mid))
    mu = rho * low_pred_hf * scale + delta_pred
    resid = y_hf - mu
    return -0.5 * float(np.sum(resid**2)) / (sigma_n**2)


def _mcmc_log_prior(
    th: np.ndarray, theta_low: np.ndarray, theta_high: np.ndarray
) -> float:
    """Uniform 先验: 在 bounds 内 = 0 (log=0), 外 = -inf."""
    if np.any(th < theta_low) or np.any(th > theta_high):
        return -np.inf
    return 0.0


def _run_metropolis(
    theta: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
    theta_mid: np.ndarray,
    rho: float,
    low_pred_hf: np.ndarray,
    delta_pred: np.ndarray,
    y_hf: np.ndarray,
    sigma_n: float,
    d_theta: int,
    n_total: int,
    proposal_std: float,
    seed: int | None,
) -> tuple[list[np.ndarray], int]:
    """Metropolis-Hastings 随机游走, 返回 (samples, n_accept)."""

    def log_posterior(th: np.ndarray) -> float:
        lp = _mcmc_log_prior(th, theta_low, theta_high)
        if lp == -np.inf:
            return -np.inf
        return lp + _mcmc_log_likelihood(
            th, theta_mid, rho, low_pred_hf, delta_pred, y_hf, sigma_n
        )

    # Metropolis-Hastings 随机游走
    rng = np.random.default_rng(seed if seed is not None else None)
    current_theta = theta.copy()
    current_lp = log_posterior(current_theta)
    samples: list[np.ndarray] = []
    n_accept = 0

    for _ in range(n_total):
        proposal = current_theta + rng.normal(0, proposal_std, size=d_theta)
        proposal_lp = log_posterior(proposal)
        # 接受率 = min(1, exp(proposal - current))
        if proposal_lp > -np.inf:
            log_accept = proposal_lp - current_lp
            if np.log(rng.uniform()) < log_accept:
                current_theta = proposal
                current_lp = proposal_lp
                n_accept += 1
        samples.append(current_theta.copy())
    return samples, n_accept


def _latin_hypercube(
    n_lf: int,
    dim: int,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Latin hypercube sampling for LF."""
    # 每维分 n_lf 个等概率区间, 每区间抽一个均匀点, 然后逐维打乱
    X_lf = np.zeros((n_lf, dim))
    for j in range(dim):
        # 等概率分箱
        edges = np.linspace(bounds_low[j], bounds_high[j], n_lf + 1)
        # 每个箱内均匀采样
        pts = np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n_lf)])
        # 打乱顺序
        rng.shuffle(pts)
        X_lf[:, j] = pts
    return X_lf


def _min_pairwise_distance(X: np.ndarray) -> float:
    """最小点间距, 作为空间填充度指标."""
    if len(X) < 2:
        return 0.0
    from scipy.spatial.distance import pdist

    return float(np.min(pdist(X)))


def _control_variate_stats(
    y_hf: np.ndarray, y_lf: np.ndarray, beta_arg: float | None
) -> dict[str, Any]:
    """控制变量方差缩减的统计量与结果字典."""
    var_hf = float(np.var(y_hf, ddof=1))
    var_lf = float(np.var(y_lf, ddof=1))
    cov_hf_lf = float(np.cov(y_hf, y_lf, ddof=1)[0, 1])

    # 最优 beta
    beta_opt = 0.0 if var_lf < 1e-12 else cov_hf_lf / var_lf

    beta = beta_arg if beta_arg is not None else beta_opt

    # 控制变量估计: y_cv = y_hf - beta * (y_lf - mean(y_lf))
    # 注意: 如果 y_lf 是已知解析模型, E[y_lf] 是确定的; 如果是采样,
    # 用样本均值近似. 这里用样本均值.
    y_cv = y_hf - beta * (y_lf - np.mean(y_lf))
    var_cv = float(np.var(y_cv, ddof=1))

    reduction_ratio = 1.0 - (var_cv / var_hf) if var_hf > 1e-12 else 0.0

    return {
        "estimate": float(np.mean(y_cv)),
        "estimate_hf_only": float(np.mean(y_hf)),
        "variance_original": var_hf,
        "variance_reduced": var_cv,
        "reduction_ratio": round(reduction_ratio, 6),
        "beta": round(beta, 6),
        "beta_optimal": round(beta_opt, 6),
        "cov_hf_lf": round(cov_hf_lf, 6),
        "n_samples": len(y_hf),
        "method": "control_variate",
    }


def _calibration_payload(
    post_burnin: np.ndarray,
    n_total: int,
    n_burnin: int,
    acceptance_rate: float,
    rho: float,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    """组装 bayesian_calibrate 的结果字典 (后验统计 + 元信息)."""
    # 后验统计
    post_mean = post_burnin.mean(axis=0)
    post_std = post_burnin.std(axis=0)
    return {
        "posterior_mean": post_mean.tolist(),
        "posterior_std": post_std.tolist(),
        "posterior_samples": post_burnin.tolist(),
        "n_post_burnin": len(post_burnin),
        "n_mcmc_total": n_total,
        "n_burnin": n_burnin,
        "acceptance_rate": round(acceptance_rate, 4),
        "rho": rho,
        "theta_prior_low": theta_low.tolist(),
        "theta_prior_high": theta_high.tolist(),
        "method": "kennedy_ohagan_2000_metropolis_hastings",
    }


def _calibrate_posterior(
    inp: MultiFidelityInput,
    theta: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
    y_hf: np.ndarray,
    d_theta: int,
    calib: tuple,
) -> dict:
    """跑 MH 采样, 返回后验结果字典 (burn-in 后样本 + 统计)."""
    rho, low_pred_hf, delta_pred, sigma_n, theta_mid = calib
    n_total = inp.n_mcmc_samples
    samples, n_accept = _run_metropolis(
        theta,
        theta_low,
        theta_high,
        theta_mid,
        rho,
        low_pred_hf,
        delta_pred,
        y_hf,
        sigma_n,
        d_theta,
        n_total,
        inp.proposal_std,
        inp.seed,
    )
    # burn-in 后的样本
    post_burnin = np.array(samples[inp.n_burnin :])
    acceptance_rate = n_accept / n_total
    return _calibration_payload(
        post_burnin, n_total, inp.n_burnin, acceptance_rate, rho, theta_low, theta_high
    )


def _fem_validation(surrogate, sol_field: np.ndarray, mesh_data: Any) -> dict:
    """用已拟合的 surrogate 在 mesh 节点上预测, 算跟神经解的残差."""
    if surrogate is None or not surrogate.fitted:
        return {"note": "surrogate 未拟合, 跳过 FEM 校验 (先 fit_surrogate)"}
    try:
        nodes = mesh_data.get("nodes", mesh_data)
        X = np.array(nodes, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        fem_mu, fem_sigma = surrogate.predict(X)
        # 残差: 神经解 vs FEM 均值 (长度对齐才比)
        n = min(len(sol_field), len(fem_mu))
        if n > 0:
            resid = sol_field[:n] - fem_mu[:n]
            return {
                "mu": fem_mu[:n].tolist(),
                "sigma": fem_sigma[:n].tolist(),
                "residual_mean": float(np.mean(resid)),
                "residual_std": float(np.std(resid)),
                "residual_max": float(np.max(np.abs(resid))),
            }
        return {"note": "长度对不齐, 跳过残差"}
    except Exception as exc:
        return {"error": str(exc)}


def _neural_degraded(surrogate, reason: str, backend: str | None = None) -> ToolResult:
    """neural 不可用时的降级返回, 让上层走纯 FEM."""
    data: dict[str, Any] = {
        "neural_estimate": None,
        "degraded": True,
        "reason": reason,
    }
    if backend is not None:
        data["backend"] = backend
    data["fem_available"] = surrogate is not None and surrogate.fitted
    return ToolResult(data=data, success=True)


# ── tool ─────────────────────────────────────────────────────────────────────


class MultiFidelityTool(HuginnTool):
    """多保真融合: 注册数据源, 拟合自回归 GP, 传播不确定度, cost-aware 选点."""

    name = "multi_fidelity_tool"
    category = "sci"
    description = (
        "多保真融合: 注册不同保真度的数据源, 用自回归 co-Kriging 拟合代理模型, "
        "传播不确定度, 按 cost-aware EI 选下一个评估点."
    )
    input_schema = MultiFidelityInput
    read_only = True
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset(
            {ResearchPhase.PLANNING, ResearchPhase.VALIDATION, ResearchPhase.OPEN}
        ),
    )

    def __init__(self) -> None:
        self._sources: list[FidelitySource] = []
        self._surrogate: MultiFidelitySurrogate | None = None
        # 神经 PDE 代理: 延迟构造, 避免跟 _neural_proxy 方法重名被遮蔽
        self._proxy: NeuralPDEProxy | None = None

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = MultiFidelityInput(**args)
        try:
            if input_data.action == "register_source":
                return self._register_source(input_data)
            if input_data.action == "fit_surrogate":
                return self._fit_surrogate(input_data)
            if input_data.action == "propagate":
                return self._propagate(input_data)
            if input_data.action == "select_next":
                return self._select_next(input_data)
            if input_data.action == "bayesian_calibrate":
                return self._bayesian_calibrate(input_data)
            if input_data.action == "nested_doe":
                return self._nested_doe(input_data)
            if input_data.action == "variance_reduction":
                return self._variance_reduction(input_data)
            if input_data.action == "neural_proxy":
                return self._neural_proxy(input_data)
            return ToolResult(
                data=None, success=False, error=f"未知 action: {input_data.action}"
            )
        except Exception as exc:
            return ToolResult(
                data=None, success=False, error=f"MultiFidelityTool failed: {exc}"
            )

    # ── actions ──────────────────────────────────────────────────────

    def _register_source(self, inp: MultiFidelityInput) -> ToolResult:
        if not inp.name:
            return ToolResult(
                data=None, success=False, error="register_source 需要 name"
            )
        if inp.level is None:
            return ToolResult(
                data=None, success=False, error="register_source 需要 level"
            )
        if inp.X is None or inp.y is None:
            return ToolResult(
                data=None, success=False, error="register_source 需要 X 和 y"
            )
        if len(inp.X) != len(inp.y):
            return ToolResult(data=None, success=False, error="X 和 y 长度不一致")
        if len(inp.X) == 0:
            return ToolResult(data=None, success=False, error="数据点不能为空")

        # 同名覆盖
        self._sources = [s for s in self._sources if s.name != inp.name]
        src = FidelitySource(
            name=inp.name,
            level=inp.level,
            cost=inp.cost if inp.cost is not None else 1.0,
            X=np.array(inp.X, dtype=float),
            y=np.array(inp.y, dtype=float),
        )
        self._sources.append(src)
        # 新数据进来, 之前的 surrogate 作废
        self._surrogate = None
        return ToolResult(
            data={
                "name": src.name,
                "level": src.level,
                "cost": src.cost,
                "n_points": len(src.X),
                "dim": src.X.shape[1] if src.X.ndim > 1 else 1,
            },
            success=True,
        )

    def _fit_surrogate(self, inp: MultiFidelityInput) -> ToolResult:
        if len(self._sources) < 1:
            return ToolResult(data=None, success=False, error="未注册任何数据源")
        # 按保真级排序
        sources = sorted(self._sources, key=lambda s: s.level)
        low = sources[0]

        low_gp = NumPyGP(
            length_scale=inp.length_scale,
            sigma_f=inp.sigma_f,
            sigma_n=inp.sigma_n,
        )
        low_gp.fit(low.X, low.y)

        rho = 1.0
        delta_gp = None
        if len(sources) >= 2:
            high = sources[-1]
            # 低保真在高保真点的预测
            low_pred, _ = low_gp.predict(high.X)
            # rho 最小二乘: rho = (low_pred @ y_high) / (low_pred @ low_pred)
            denom = float(low_pred @ low_pred)
            rho = float(low_pred @ high.y) / denom if denom > 1e-12 else 1.0
            # 残差
            delta = high.y - rho * low_pred
            delta_gp = NumPyGP(
                length_scale=inp.length_scale,
                sigma_f=inp.sigma_f,
                sigma_n=inp.sigma_n,
            )
            delta_gp.fit(high.X, delta)

        self._surrogate = MultiFidelitySurrogate(
            sources=sources,
            rho=rho,
            low_gp=low_gp,
            delta_gp=delta_gp,
            fitted=True,
        )
        return ToolResult(
            data={
                "fitted": True,
                "n_sources": len(sources),
                "levels": [s.level for s in sources],
                "rho": rho,
                "has_delta": delta_gp is not None,
            },
            success=True,
        )

    def _propagate(self, inp: MultiFidelityInput) -> ToolResult:
        if self._surrogate is None or not self._surrogate.fitted:
            return ToolResult(
                data=None, success=False, error="未拟合 surrogate, 先调 fit_surrogate"
            )
        if not inp.X_new:
            return ToolResult(data=None, success=False, error="propagate 需要 X_new")
        X = np.array(inp.X_new, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mu, sigma = self._surrogate.predict(X)
        return ToolResult(
            data={
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "n_points": len(mu),
            },
            success=True,
        )

    def _select_next(self, inp: MultiFidelityInput) -> ToolResult:
        if self._surrogate is None or not self._surrogate.fitted:
            return ToolResult(
                data=None, success=False, error="未拟合 surrogate, 先调 fit_surrogate"
            )
        if not inp.candidates:
            return ToolResult(
                data=None, success=False, error="select_next 需要 candidates"
            )
        candidates = np.array(inp.candidates, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)

        mu, sigma = self._surrogate.predict(candidates)
        # 当前最优
        best = float(np.max(mu)) if inp.maximize else float(np.min(mu))

        # cost-aware EI: EI / cost
        results: list[dict[str, Any]] = []
        for i in range(len(candidates)):
            ei = self._expected_improvement(mu[i], sigma[i], best, inp.maximize)
            for src in self._surrogate.sources:
                cost = max(src.cost, 1e-8)
                acq = ei / cost
                results.append(
                    {
                        "candidate_idx": i,
                        "fidelity": src.name,
                        "level": src.level,
                        "cost": src.cost,
                        "ei": float(ei),
                        "acquisition": float(acq),
                        "mu": float(mu[i]),
                        "sigma": float(sigma[i]),
                    }
                )
        # 按 acquisition 降序排, 选 top-n
        results.sort(key=lambda r: r["acquisition"], reverse=True)
        selected = results[: inp.n_select]
        return ToolResult(
            data={
                "selected": selected,
                "n_candidates": len(candidates),
                "best_so_far": best,
            },
            success=True,
        )

    @staticmethod
    def _expected_improvement(
        mu: float, sigma: float, best: float, maximize: bool
    ) -> float:
        """标准 EI. sigma=0 时返回 0."""
        if sigma < 1e-10:
            return 0.0
        from scipy.stats import norm

        improvement = mu - best if maximize else best - mu
        z = improvement / sigma
        return float(improvement * norm.cdf(z) + sigma * norm.pdf(z))

    # ── paper-level deepening (E3) ──────────────────────────────────

    def _calibration_inputs(self, inp: MultiFidelityInput):
        """校验 calibration 输入并准备数组/theta, 失败时返回失败的 ToolResult."""
        if not inp.X_hf or not inp.y_hf or not inp.X_lf or not inp.y_lf:
            return ToolResult(
                data=None,
                success=False,
                error="bayesian_calibrate 需要 X_hf/y_hf/X_lf/y_lf",
            )
        if not inp.theta_prior_low or not inp.theta_prior_high:
            return ToolResult(
                data=None,
                success=False,
                error="bayesian_calibrate 需要 theta_prior_low/theta_prior_high",
            )

        X_hf = np.array(inp.X_hf, dtype=float)
        y_hf = np.array(inp.y_hf, dtype=float)
        X_lf = np.array(inp.X_lf, dtype=float)
        y_lf = np.array(inp.y_lf, dtype=float)
        theta_low = np.array(inp.theta_prior_low, dtype=float)
        theta_high = np.array(inp.theta_prior_high, dtype=float)
        d_theta = len(theta_low)

        # theta_init: 默认用先验中点
        theta = (
            np.array(inp.theta_init, dtype=float)
            if inp.theta_init
            else 0.5 * (theta_low + theta_high)
        )
        return X_hf, y_hf, X_lf, y_lf, theta_low, theta_high, theta, d_theta

    def _calibration_surrogate(
        self, inp: MultiFidelityInput, X_hf, y_hf, X_lf, y_lf, theta_low, theta_high
    ):
        """拟合低保真 GP + 残差 GP, 返回 (rho, low_pred_hf, delta_pred, sigma_n, theta_mid)."""
        # 简化版: theta 通过尺度因子影响 y_lf 预测, 而非完整 Kennedy-O'Hagan 的 theta-dependent 仿真器

        # 拟合低保真 GP
        low_gp = NumPyGP(
            length_scale=inp.length_scale,
            sigma_f=inp.sigma_f,
            sigma_n=inp.sigma_n,
        )
        low_gp.fit(X_lf, y_lf)
        low_pred_hf, _ = low_gp.predict(X_hf)

        # rho 最小二乘
        denom = float(low_pred_hf @ low_pred_hf)
        rho = float(low_pred_hf @ y_hf) / denom if denom > 1e-12 else 1.0

        # 残差 delta = y_hf - rho * low_pred_hf
        delta = y_hf - rho * low_pred_hf
        delta_gp = NumPyGP(
            length_scale=inp.length_scale,
            sigma_f=inp.sigma_f,
            sigma_n=inp.sigma_n,
        )
        delta_gp.fit(X_hf, delta)

        # theta 的影响: 假设 theta 影响 y_lf 的尺度, 即 y_lf(x, theta) = y_lf(x) * (1 + sum(theta))
        # 这样 MCMC 能在一个有意义的参数空间里探索. log-likelihood:
        #   log p(data | theta) = -0.5 * sum((y_hf - mu(theta))^2 / sigma_n^2)
        #   mu(theta) = rho * y_lf_pred * (1 + sum(theta - theta_mid)) + delta_pred
        theta_mid = 0.5 * (theta_low + theta_high)
        delta_pred, _ = delta_gp.predict(X_hf)
        sigma_n = max(inp.sigma_n, 1e-6)
        return rho, low_pred_hf, delta_pred, sigma_n, theta_mid

    def _bayesian_calibrate(self, inp: MultiFidelityInput) -> ToolResult:
        """Kennedy-O'Hagan 2000 Bayesian model calibration.

        模型: y_hf(x) = rho * y_lf(x, theta) + delta(x) + eps
        - delta(x) ~ GP(0, K_delta), 用残差 GP 拟合
        - theta ~ Uniform(theta_low, theta_high) 先验
        - rho: 固定 (从数据估计), 不参与 MCMC
        - eps ~ N(0, sigma_n^2)

        MCMC: Metropolis-Hastings 随机游走采样 theta 后验.
        log p(theta | data) ∝ log p(data | theta) + log p(theta)
        似然 = N(y_hf | rho * y_lf(X_hf, theta) + delta_pred(X_hf), sigma_n^2 I)

        返回: posterior samples (post-burnin), posterior mean/std, acceptance rate.
        """
        prepared = MultiFidelityTool._calibration_inputs(self, inp)
        if isinstance(prepared, ToolResult):
            return prepared
        X_hf, y_hf, X_lf, y_lf, theta_low, theta_high, theta, d_theta = prepared

        calib = MultiFidelityTool._calibration_surrogate(
            self, inp, X_hf, y_hf, X_lf, y_lf, theta_low, theta_high
        )

        return ToolResult(
            data=_calibrate_posterior(
                inp, theta, theta_low, theta_high, y_hf, d_theta, calib
            ),
            success=True,
        )

    def _nested_doe(self, inp: MultiFidelityInput) -> ToolResult:
        """嵌套空间填充 DOE (Qian 2009).

        生成 n_lf 个低保真点, 其中前 n_hf 个同时作为高保真点 (nested design).
        用 Latin hypercube + 前缀复用保证 HF ⊂ LF.

        Returns:
            X_hf: 高保真点 (前 n_hf 个 LF 点)
            X_lf: 全部低保真点 (含 HF 点作为前缀)
        """
        if inp.n_hf > inp.n_lf:
            return ToolResult(
                data=None,
                success=False,
                error=f"n_hf ({inp.n_hf}) 不能大于 n_lf ({inp.n_lf})",
            )

        rng = np.random.default_rng(inp.seed)
        bounds_low = np.array(inp.bounds_low or [0.0] * inp.dim, dtype=float)
        bounds_high = np.array(inp.bounds_high or [1.0] * inp.dim, dtype=float)
        if len(bounds_low) != inp.dim or len(bounds_high) != inp.dim:
            return ToolResult(
                data=None, success=False, error=f"bounds 长度应等于 dim={inp.dim}"
            )

        # Latin hypercube sampling for LF
        # 每维分 n_lf 个等概率区间, 每区间抽一个均匀点, 然后逐维打乱
        X_lf = _latin_hypercube(inp.n_lf, inp.dim, bounds_low, bounds_high, rng)

        # 嵌套: HF 取 LF 的前 n_hf 个点 (Qian 2009 nested design)
        X_hf = X_lf[: inp.n_hf].copy()

        # 计算 min-distance 作为空间填充度指标
        return ToolResult(
            data={
                "X_hf": X_hf.tolist(),
                "X_lf": X_lf.tolist(),
                "n_hf": inp.n_hf,
                "n_lf": inp.n_lf,
                "dim": inp.dim,
                "nested": True,
                "hf_min_distance": round(_min_pairwise_distance(X_hf), 6),
                "lf_min_distance": round(_min_pairwise_distance(X_lf), 6),
                "method": "nested_latin_hypercube_qian_2009",
            },
            success=True,
        )

    def _variance_reduction(self, inp: MultiFidelityInput) -> ToolResult:
        """控制变量方差缩减 (control variates).

        利用低保真模型 y_lf 作为控制变量, 缩减高保真估计 y_hf 的方差:
            y_cv = y_hf - beta * (y_lf - E[y_lf])
            Var(y_cv) = Var(y_hf) - 2*beta*Cov(y_hf, y_lf) + beta^2 * Var(y_lf)

        最优 beta* = Cov(y_hf, y_lf) / Var(y_lf), 此时方差缩减最大.

        Returns:
            estimate: 缩减后的估计值 (mean of y_cv)
            variance_original: Var(y_hf)
            variance_reduced: Var(y_cv)
            reduction_ratio: 1 - Var(y_cv)/Var(y_hf)
            beta: 使用的 beta 值
        """
        if not inp.y_hf_samples or not inp.y_lf_samples:
            return ToolResult(
                data=None,
                success=False,
                error="variance_reduction 需要 y_hf_samples 和 y_lf_samples",
            )
        y_hf = np.array(inp.y_hf_samples, dtype=float)
        y_lf = np.array(inp.y_lf_samples, dtype=float)
        if len(y_hf) != len(y_lf):
            return ToolResult(
                data=None,
                success=False,
                error=f"y_hf ({len(y_hf)}) 和 y_lf ({len(y_lf)}) 长度必须相同",
            )
        if len(y_hf) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="至少需要 2 个样本才能算方差",
            )

        return ToolResult(
            data=_control_variate_stats(y_hf, y_lf, inp.beta),
            success=True,
        )

    def _ensure_proxy(self, model_path: str | None) -> NeuralPDEProxy:
        # 延迟构造 proxy, 第一次用才建
        if self._proxy is None:
            self._proxy = NeuralPDEProxy()
        if model_path:
            self._proxy.load_model(model_path)
        return self._proxy

    def _neural_proxy(self, inp: MultiFidelityInput) -> ToolResult:
        """Transolver 神经 PDE 快速预估 + FEM 校验.

        流程:
          1. 有 neural proxy (torch+transolver+权重) -> 先跑神经快速预估
          2. 再用已拟合的多保真 surrogate 做 FEM 级校验 (残差 = |neural - fem|)
          3. neural 不可用 -> 直接返回降级提示, 让上层走纯 FEM
        """
        if not inp.mesh_data or not inp.boundary_conditions:
            return ToolResult(
                data=None,
                success=False,
                error="neural_proxy 需要 mesh_data 和 boundary_conditions",
            )

        proxy = MultiFidelityTool._ensure_proxy(self, inp.model_path)

        if not proxy.available():
            # 降级: 没装 torch/transolver, 直接提示用 FEM
            return _neural_degraded(self._surrogate, proxy.status())

        # 神经快速预估
        sol = proxy.predict(inp.mesh_data, inp.boundary_conditions)
        if not sol.available:
            return _neural_degraded(self._surrogate, sol.reason, sol.backend)

        result: dict[str, Any] = {
            "neural_estimate": {
                "field": sol.field.tolist(),
                "backend": sol.backend,
                "meta": sol.meta,
            },
            "degraded": False,
        }

        # FEM 校验
        result["fem_validation"] = _fem_validation(
            self._surrogate, sol.field, inp.mesh_data
        )

        return ToolResult(data=result, success=True)


__all__ = [
    "MultiFidelityTool",
    "MultiFidelityInput",
    "FidelitySource",
    "MultiFidelitySurrogate",
]
