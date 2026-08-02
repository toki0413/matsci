"""AlignmentFunction — GP 学习 source_space -> target_space 的映射.

神经科学依据: 联合皮层对齐不同模态的 latent space. 盲人触觉空间和视觉空间的
对齐本质上是学一个映射函数. 这里用 Gaussian Process — 给出预测 + 不确定性,
不确定性是发现信号的核心 (surprise = 偏差 / 不确定性).

GP 选 Matern(ν=2.5) 核: 比 RBF 更容许函数局部不光滑, 适合物理量映射.
每个目标维度一个 GP (不用 MultiOutputGP, 简单直接, 维度间独立假设够用).

surprise 公式: ||actual - predicted|| / (||uncertainty|| + eps)
- 高 surprise = 实际偏离预测 且 不确定性低 → 发现信号
- 低 surprise = 要么预测准, 要么本来就不确定

接入点: HypothesisManifold.set_alignment_function / _alignment_proposal /
check_surprise; rcb_runner 自动 fit.

ponytail: 每个目标维度独立 GP, 不建模维度间相关性.
ceiling: 维度间相关性丢失. 升级路径: MultiOutputGP 或 coregionalized GP.
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from huginn.metacog.alignment_dataset import AlignmentDataset

_EPS = 1e-8


class AlignmentFunction:
    """GP 对齐函数: source_space -> target_space, 带不确定性.

    source_space / target_space 只需鸭子类型: 有 .name (str) 和
    .encode(obj) -> np.ndarray. 不强依赖 LatentSpace ABC, 通用.
    """

    def __init__(
        self,
        source_space,
        target_space,
        min_samples: int = 10,
    ):
        self.source_space = source_space
        self.target_space = target_space
        self.min_samples = min_samples
        # 每个目标维度一个 GP — 简单, 维度间独立
        self._gps: list[GaussianProcessRegressor] = []
        self._fitted = False

    @property
    def ready(self) -> bool:
        """数据量足够且已 fit 才 ready."""
        return self._fitted and len(self._gps) > 0

    def fit(self, dataset: AlignmentDataset) -> None:
        """用 GP 拟合 source_vec -> target_vec.

        数据不足 (< min_samples) 直接返回, 不 fit — 让 ready 保持 False,
        调用方按 ready 退化.
        """
        X, y = dataset.get_pairs(self.source_space.name, self.target_space.name)
        if len(X) < self.min_samples:
            return
        # 每个目标维度各 fit 一个 GP
        self._gps = []
        for d in range(y.shape[1]):
            kernel = ConstantKernel(1.0) * Matern(nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
            )
            gp.fit(X, y[:, d])
            self._gps.append(gp)
        self._fitted = True

    def predict(self, source_obj) -> tuple[np.ndarray, np.ndarray]:
        """预测 target descriptor + 不确定性 (每维 std).

        source_obj 可以是任意 source_space.encode 能处理的对象.
        未 ready 抛 RuntimeError — 调用方应先查 ready.
        """
        if not self.ready:
            raise RuntimeError("AlignmentFunction not ready")
        source_vec = self.source_space.encode(source_obj).reshape(1, -1)
        means: list[float] = []
        stds: list[float] = []
        for gp in self._gps:
            mean, std = gp.predict(source_vec, return_std=True)
            means.append(float(mean[0]))
            stds.append(float(std[0]))
        return np.array(means), np.array(stds)

    def surprise(self, source_obj, actual_target_obj) -> float:
        """surprise = ||actual - predicted|| / (||uncertainty|| + eps).

        高 surprise = 偏差大且不确定性低 → 发现信号.
        eps 防止不确定性为 0 时除零 (GP 在训练点上 std 可能极小).
        """
        predicted, uncertainty = self.predict(source_obj)
        actual = self.target_space.encode(actual_target_obj)
        residual = np.linalg.norm(actual - predicted)
        unc = np.linalg.norm(uncertainty) + _EPS
        return float(residual / unc)
