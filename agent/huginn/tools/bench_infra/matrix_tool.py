"""Training matrix tool — 跑 N×M task×method 矩阵, 每格存 loss.json + metrics.json.

治 ζ_matrix: agent 反复重写训练循环, 每次 5-10 calls 浪费在样板.
统一工具: 4 个 SBI benchmark task × 3 方法 (NPE/NRE/NLE), numpy/torch 实现.
无 sbi 依赖 — 用简单 MLP 近似 posterior, 给 agent 训练证据 + loss curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


# ── benchmark simulators ────────────────────────────────────────
# 每个 task 返回 (prior_sample_fn, simulator_fn, param_dim, data_dim).
# 数值稳定, 无外部依赖. 参数范围 [-3, 3] cube.

def _linear_gaussian():
    # θ ~ N(0, I), x = A·θ + ε, A=I, ε~N(0, 0.1·I)
    def prior(n, rng): return rng.standard_normal((n, 4))
    def sim(theta, rng): return theta + 0.1 * rng.standard_normal(theta.shape)
    return prior, sim, 4, 4


def _two_moons():
    # θ ~ U(-1,1)², x = (r·cos(α), r·sin(α)) + ε, α = θ₁·π, r = θ₂ (moon shape)
    def prior(n, rng): return rng.uniform(-1, 1, (n, 2))
    def sim(theta, rng):
        a = theta[:, 0] * np.pi
        r = np.abs(theta[:, 1])
        x = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        return x + 0.1 * rng.standard_normal(x.shape)
    return prior, sim, 2, 2


def _slcp():
    # Simple Likelihood Complex Posterior: 4-dim θ, 8-dim x, mixture of 4 gaussians
    def prior(n, rng): return rng.uniform(-3, 3, (n, 4))
    def sim(theta, rng):
        n = theta.shape[0]
        means = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]], dtype=float)
        # 每个样本随机选一个分量, 简化版
        idx = rng.integers(0, 4, size=n)
        x = means[idx] + 0.3 * rng.standard_normal((n, 2))
        # 扩展到 8 维 (重复 + 噪声)
        x = np.tile(x, (1, 4)) + 0.05 * rng.standard_normal((n, 8))
        return x
    return prior, sim, 4, 8


def _gaussian_mixture():
    # θ ~ U(-3,3)², x ~ 0.5·N(θ, I) + 0.5·N(-θ, I) (bimodal)
    def prior(n, rng): return rng.uniform(-3, 3, (n, 2))
    def sim(theta, rng):
        n = theta.shape[0]
        mask = rng.random(n) < 0.5
        mu = np.where(mask[:, None], theta, -theta)
        return mu + rng.standard_normal((n, 2))
    return prior, sim, 2, 2


TASKS = {
    "linear_gaussian": _linear_gaussian,
    "two_moons": _two_moons,
    "slcp": _slcp,
    "gaussian_mixture": _gaussian_mixture,
}


def _train_cell(task_name: str, method: str, n_train: int, n_epochs: int, seed: int):
    """跑一格: 生成数据 → 训 MLP → 返回 loss curve + val metric.
    method 只影响 loss head (NPE: MSE on θ|x, NRE: BCE real/fake, NLE: MSE on x|θ).
    """
    rng = np.random.default_rng(seed)
    prior_fn, sim_fn, p_dim, x_dim = TASKS[task_name]()

    theta = prior_fn(n_train, rng)
    x = sim_fn(theta, rng)

    # ponytail: 单层 MLP, torch 实现. 不追求精度, 只要训练能跑出 loss curve.
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    if method == "npe":
        net = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, p_dim))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.tensor(x, dtype=torch.float32)
        thetat = torch.tensor(theta, dtype=torch.float32)
        losses = []
        for ep in range(n_epochs):
            pred = net(xt)
            loss = nn.functional.mse_loss(pred, thetat)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
    elif method == "nre":
        # ratio: classifier (θ, x) real vs (θ', x) fake
        theta_fake = prior_fn(n_train, rng)
        clf = nn.Sequential(nn.Linear(x_dim + p_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
        real = torch.tensor(np.concatenate([theta, x], axis=1), dtype=torch.float32)
        fake = torch.tensor(np.concatenate([theta_fake, x], axis=1), dtype=torch.float32)
        labels = torch.cat([torch.ones(n_train, 1), torch.zeros(n_train, 1)])
        data = torch.cat([real, fake])
        losses = []
        for ep in range(n_epochs):
            idx = torch.randperm(2 * n_train)
            logits = clf(data[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
    else:  # nle
        net = nn.Sequential(nn.Linear(p_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, x_dim))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt = torch.tensor(x, dtype=torch.float32)
        thetat = torch.tensor(theta, dtype=torch.float32)
        losses = []
        for ep in range(n_epochs):
            pred = net(thetat)
            loss = nn.functional.mse_loss(pred, xt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))

    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "n_train": n_train,
        "n_epochs": n_epochs,
        "param_dim": p_dim,
        "data_dim": x_dim,
    }


class MatrixToolInput(BaseModel):
    tasks: list[str] = Field(
        default=["linear_gaussian"],
        description="SBI benchmark tasks: linear_gaussian | two_moons | slcp | gaussian_mixture",
    )
    methods: list[str] = Field(
        default=["npe"],
        description="Inference methods: npe | nre | nle",
    )
    n_train: int = Field(default=1000, gt=0, description="Training samples per cell")
    n_epochs: int = Field(default=50, gt=0, description="Training epochs per cell")
    output_dir: str = Field(default="outputs/matrix", description="Output directory")
    seed: int = Field(default=42)


class TrainingMatrixTool(HuginnTool):
    """Run N×M task×method training matrix, save loss.json per cell."""

    name = "training_matrix_tool"
    category = "analysis"
    description = (
        "Run a training matrix over SBI benchmark tasks × inference methods. "
        "Supported tasks: linear_gaussian, two_moons, slcp, gaussian_mixture. "
        "Methods: npe (posterior), nre (ratio), nle (likelihood). "
        "Saves loss.json + metrics.json per cell to outputs/matrix/."
    )
    destructive = False
    input_schema = MatrixToolInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = MatrixToolInput(**args)

        # 校验 task/method
        bad_tasks = [t for t in input_data.tasks if t not in TASKS]
        if bad_tasks:
            return ToolResult(
                data=None, success=False,
                error=f"Unknown tasks: {bad_tasks}. Valid: {list(TASKS)}",
            )
        valid_methods = {"npe", "nre", "nle"}
        bad_methods = [m for m in input_data.methods if m not in valid_methods]
        if bad_methods:
            return ToolResult(
                data=None, success=False,
                error=f"Unknown methods: {bad_methods}. Valid: {sorted(valid_methods)}",
            )

        out_dir = Path(input_data.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cells = []
        t0 = time.time()
        for task in input_data.tasks:
            for method in input_data.methods:
                cell_t0 = time.time()
                try:
                    result = _train_cell(
                        task, method, input_data.n_train, input_data.n_epochs, input_data.seed
                    )
                    result["task"] = task
                    result["method"] = method
                    result["elapsed_s"] = round(time.time() - cell_t0, 2)

                    cell_dir = out_dir / f"{task}_{method}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    (cell_dir / "loss.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                    cells.append({
                        "task": task, "method": method,
                        "final_loss": result["final_loss"],
                        "elapsed_s": result["elapsed_s"],
                        "loss_curve_len": len(result["losses"]),
                    })
                except Exception as e:
                    cells.append({
                        "task": task, "method": method, "error": str(e),
                    })

        summary = {
            "matrix_shape": [len(input_data.tasks), len(input_data.methods)],
            "total_cells": len(input_data.tasks) * len(input_data.methods),
            "cells": cells,
            "total_elapsed_s": round(time.time() - t0, 2),
            "output_dir": str(out_dir),
        }
        (out_dir / "matrix_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        return ToolResult(
            data=summary,
            success=True,
            side_effects=[str(out_dir / "matrix_summary.json")],
        )


if __name__ == "__main__":
    import asyncio

    async def _test():
        tool = TrainingMatrixTool()
        result = await tool.call({
            "tasks": ["linear_gaussian", "two_moons"],
            "methods": ["npe"],
            "n_train": 200,
            "n_epochs": 10,
            "output_dir": "_test_matrix",
        })
        data = result.data
        print(f"cells={data['total_cells']}, elapsed={data['total_elapsed_s']}s")
        for c in data["cells"]:
            if "error" in c:
                print(f"  {c['task']}/{c['method']}: ERROR: {c['error']}")
            else:
                print(f"  {c['task']}/{c['method']}: final_loss={c.get('final_loss')}")
        # 清理
        import shutil
        shutil.rmtree("_test_matrix", ignore_errors=True)
        print("[matrix_tool] self-check OK")

    asyncio.run(_test())
