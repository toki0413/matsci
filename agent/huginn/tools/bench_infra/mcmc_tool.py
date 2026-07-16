"""MCMC sampler tool — ABC rejection + Metropolis-Hastings, 无 sbi 依赖.

治 ζ_mcmc: agent 反复写采样器, 5-10 calls 浪费在 MH 样板.
统一工具: 给 prior + simulator + observation, 返回 posterior samples.
numpy 实现, 降级于 sbi 但数值稳定.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# ── 内置 benchmark simulator (同 matrix_tool) ────────────────────
def _get_simulator(task: str):
    if task == "linear_gaussian":
        def prior(n, rng): return rng.standard_normal((n, 4))
        def sim(theta, rng): return theta + 0.1 * rng.standard_normal(theta.shape)
        return prior, sim, 4, 4
    if task == "two_moons":
        def prior(n, rng): return rng.uniform(-1, 1, (n, 2))
        def sim(theta, rng):
            a = theta[:, 0] * np.pi
            r = np.abs(theta[:, 1])
            x = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
            return x + 0.1 * rng.standard_normal(x.shape)
        return prior, sim, 2, 2
    if task == "slcp":
        def prior(n, rng): return rng.uniform(-3, 3, (n, 4))
        def sim(theta, rng):
            n = theta.shape[0]
            means = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]], dtype=float)
            idx = rng.integers(0, 4, size=n)
            x = means[idx] + 0.3 * rng.standard_normal((n, 2))
            return np.tile(x, (1, 4)) + 0.05 * rng.standard_normal((n, 8))
        return prior, sim, 4, 8
    if task == "gaussian_mixture":
        def prior(n, rng): return rng.uniform(-3, 3, (n, 2))
        def sim(theta, rng):
            n = theta.shape[0]
            mask = rng.random(n) < 0.5
            mu = np.where(mask[:, None], theta, -theta)
            return mu + rng.standard_normal((n, 2))
        return prior, sim, 2, 2
    raise ValueError(f"Unknown task: {task}. Valid: linear_gaussian, two_moons, slcp, gaussian_mixture")


class MCMCInput(BaseModel):
    task: str = Field(
        default="linear_gaussian",
        description="SBI benchmark task: linear_gaussian | two_moons | slcp | gaussian_mixture",
    )
    observation: str = Field(
        description="JSON-encoded 1D array — observed data x_obs to condition on"
    )
    num_samples: int = Field(default=1000, gt=0, description="Number of posterior samples to draw")
    method: Literal["abc", "mh"] = Field(
        default="abc",
        description="abc = rejection ABC (faster, coarser); mh = Metropolis-Hastings (slower, finer)",
    )
    epsilon: float = Field(
        default=0.5, gt=0,
        description="ABC: accept if |sim(θ) - x_obs| < epsilon. Smaller = more accurate but slower.",
    )
    proposal_std: float = Field(
        default=0.5, gt=0,
        description="MH: Gaussian proposal std. Tune for ~25% acceptance rate.",
    )
    burn_in: int = Field(default=200, ge=0, description="MH: burn-in samples to discard")
    output_path: str = Field(default="outputs/posterior_samples.json")
    seed: int = Field(default=42)


class MCMCSamplerTool(HuginnTool):
    """Sample posterior via ABC rejection or Metropolis-Hastings."""

    name = "mcmc_sampler_tool"
    category = "analysis"
    description = (
        "Sample posterior p(θ | x_obs) given a benchmark simulator and observation. "
        "Two modes: 'abc' (rejection ABC, fast) or 'mh' (Metropolis-Hastings, finer). "
        "No sbi dependency — pure numpy. Returns samples + diagnostics."
    )
    destructive = False
    input_schema = MCMCInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = MCMCInput(**args)
        rng = np.random.default_rng(input_data.seed)

        try:
            x_obs = np.array(json.loads(input_data.observation), dtype=float).ravel()
        except (json.JSONDecodeError, ValueError) as e:
            return ToolResult(data=None, success=False, error=f"Invalid observation JSON: {e}")

        try:
            prior_fn, sim_fn, p_dim, x_dim = _get_simulator(input_data.task)
        except ValueError as e:
            return ToolResult(data=None, success=False, error=str(e))

        if x_obs.shape[0] != x_dim:
            return ToolResult(
                data=None, success=False,
                error=f"observation dim {x_obs.shape[0]} != task {input_data.task} expects {x_dim}",
            )

        if input_data.method == "abc":
            samples, diagnostics = self._abc(
                prior_fn, sim_fn, x_obs, input_data, rng
            )
        else:
            samples, diagnostics = self._mh(
                prior_fn, sim_fn, x_obs, input_data, rng
            )

        result = {
            "task": input_data.task,
            "method": input_data.method,
            "num_samples": int(len(samples)),
            "samples": samples.tolist(),
            "param_dim": p_dim,
            "diagnostics": diagnostics,
        }

        out_path = Path(input_data.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return ToolResult(
            data={
                "num_samples": len(samples),
                "mean": np.mean(samples, axis=0).tolist(),
                "std": np.std(samples, axis=0).tolist(),
                "diagnostics": diagnostics,
                "output_path": str(out_path),
            },
            success=True,
            side_effects=[str(out_path)],
        )

    def _abc(self, prior_fn, sim_fn, x_obs, cfg: MCMCInput, rng):
        """Rejection ABC: sample θ~prior, sim, accept if |sim(θ)-x_obs| < epsilon."""
        accepted = []
        n_tried = 0
        max_tries = cfg.num_samples * 1000  # 上限, 避免死循环
        while len(accepted) < cfg.num_samples and n_tried < max_tries:
            batch = max(100, cfg.num_samples - len(accepted))
            theta = prior_fn(batch, rng)
            x = sim_fn(theta, rng)
            # 距离: 每个样本的 |x - x_obs|
            dist = np.linalg.norm(x - x_obs[None, :], axis=1)
            mask = dist < cfg.epsilon
            for t in theta[mask]:
                if len(accepted) < cfg.num_samples:
                    accepted.append(t.tolist())
            n_tried += batch

        accepted = np.array(accepted[:cfg.num_samples]) if accepted else np.zeros((0, prior_fn(1, rng).shape[1]))
        diagnostics = {
            "acceptance_rate": round(len(accepted) / max(n_tried, 1), 4),
            "n_tried": n_tried,
            "epsilon": cfg.epsilon,
        }
        return accepted, diagnostics

    def _mh(self, prior_fn, sim_fn, x_obs, cfg: MCMCInput, rng):
        """Metropolis-Hastings: propose θ', accept based on |sim(θ')-x_obs|."""
        # 初始 θ 从 prior
        theta = prior_fn(1, rng)[0]
        x = sim_fn(theta[None, :], rng)[0]
        cur_dist = np.linalg.norm(x - x_obs)

        samples = []
        n_accept = 0
        total = cfg.burn_in + cfg.num_samples
        for i in range(total):
            # Gaussian proposal
            theta_prop = theta + cfg.proposal_std * rng.standard_normal(theta.shape)
            x_prop = sim_fn(theta_prop[None, :], rng)[0]
            prop_dist = np.linalg.norm(x_prop - x_obs)
            # accept if closer (ABC-MCMC: 接受概率 = min(1, exp(-(prop_dist - cur_dist) / T))
            # ponytail: T=epsilon^2, 简化但不离谱
            T = max(cfg.epsilon ** 2, 1e-6)
            log_ratio = -(prop_dist - cur_dist) / T
            if np.log(rng.random()) < log_ratio:
                theta = theta_prop
                cur_dist = prop_dist
                if i >= cfg.burn_in:
                    n_accept += 1
            if i >= cfg.burn_in:
                samples.append(theta.tolist())

        samples = np.array(samples) if samples else np.zeros((0, len(theta)))
        diagnostics = {
            "acceptance_rate": round(n_accept / max(cfg.num_samples, 1), 4),
            "burn_in": cfg.burn_in,
            "proposal_std": cfg.proposal_std,
            "final_distance": float(cur_dist),
        }
        return samples, diagnostics


if __name__ == "__main__":
    import asyncio

    async def _test():
        tool = MCMCSamplerTool()
        # 用 linear_gaussian, observation = [1, 0, 0, 0] (真值 θ ≈ [1, 0, 0, 0])
        obs = json.dumps([1.0, 0.0, 0.0, 0.0])

        r1 = await tool.call({
            "task": "linear_gaussian",
            "observation": obs,
            "num_samples": 100,
            "method": "abc",
            "epsilon": 0.3,
            "output_path": "_test_abc.json",
        })
        print(f"ABC: n={r1.data['num_samples']}, mean={r1.data['mean']}, acc={r1.data['diagnostics']['acceptance_rate']}")
        # posterior 均值应接近 [1, 0, 0, 0]
        mean = np.array(r1.data["mean"])
        assert abs(mean[0] - 1.0) < 0.5, f"ABC mean[0]={mean[0]}, expected ~1.0"

        r2 = await tool.call({
            "task": "linear_gaussian",
            "observation": obs,
            "num_samples": 100,
            "method": "mh",
            "epsilon": 0.2,
            "proposal_std": 0.3,
            "burn_in": 50,
            "output_path": "_test_mh.json",
        })
        print(f"MH: n={r2.data['num_samples']}, mean={r2.data['mean']}, acc={r2.data['diagnostics']['acceptance_rate']}")
        mean = np.array(r2.data["mean"])
        assert abs(mean[0] - 1.0) < 0.8, f"MH mean[0]={mean[0]}, expected ~1.0"

        Path("_test_abc.json").unlink(missing_ok=True)
        Path("_test_mh.json").unlink(missing_ok=True)
        print("[mcmc_tool] self-check OK")

    asyncio.run(_test())
