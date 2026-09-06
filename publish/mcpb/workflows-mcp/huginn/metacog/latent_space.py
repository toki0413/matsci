"""LatentSpace — latent space 抽象基类 (对齐层 Step 1).

对齐层的基础抽象. 任何 latent space 子类只需实现 encode(), distance 自动派生为
L2, dim/name 由子类指定. 用在 StructureDescriptor / HapticDescriptor (Step 2),
后续 AlignmentFunction (Step 4) 拿它做 source/target 空间抽象.

通用: 不绑定材料科学. 序列空间 / 技能信念空间都能子类化, 只要能 encode 成向量.

Bourbaki 视角: LatentSpace 是个 set + metric 的代数结构, encode 是到 R^n 的
嵌入, distance 是 R^n 上拉回来的度量.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LatentSpace(ABC):
    """latent space 抽象基类. 子类只需实现 encode()."""

    @abstractmethod
    def encode(self, obj) -> np.ndarray:
        """把领域对象编码为 descriptor vector."""
        ...

    def distance(self, a, b) -> float:
        """两个对象的距离 (默认 L2 on encoded vectors)."""
        va, vb = self.encode(a), self.encode(b)
        return float(np.linalg.norm(va - vb))

    @property
    def dim(self) -> int:
        """descriptor 维度. 子类 override 返回固定值.

        不从 encode 自动推断是因为 encode 需要一个 sample obj, ABC 没法凭空造.
        子类知道自己的维度, 直接 return 即可.
        """
        raise NotImplementedError(f"{type(self).__name__} 需指定 dim")

    @property
    def name(self) -> str:
        """空间名称 (日志/序列化用). 子类可 override."""
        return self.__class__.__name__


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """assert demo: 子类实现 encode 后 distance/dim/name 可用, ABC 不可实例化."""

    class _Dummy(LatentSpace):
        @property
        def dim(self) -> int:
            return 3

        @property
        def name(self) -> str:
            return "dummy"

        def encode(self, obj) -> np.ndarray:
            return np.asarray(obj, dtype=float)

    d = _Dummy()
    assert d.dim == 3
    assert d.name == "dummy"
    # L2 距离: 3-4-5 直角三角形
    assert abs(d.distance([0, 0, 0], [3, 4, 0]) - 5.0) < 1e-9
    assert d.distance([1, 2, 3], [1, 2, 3]) < 1e-12

    # ABC 不能直接实例化
    try:
        LatentSpace()  # type: ignore[abstract]
        raise AssertionError("LatentSpace 应该是 abstract, 不能实例化")
    except TypeError:
        pass

    print("✓ latent_space self-check passed")
    print(f"  dummy dim={d.dim}, name={d.name}")
    print(f"  L2([0,0,0],[3,4,0]) = {d.distance([0,0,0],[3,4,0])}")


if __name__ == "__main__":
    _selfcheck()
