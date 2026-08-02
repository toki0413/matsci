"""AlignmentDataset — 任意两个 LatentSpace 之间的对齐数据存储.

神经科学依据: 联合皮层 (association cortex) 把不同模态的 latent space 对齐到
同一坐标系. 对齐函数本身就是发现 — 结构-性质映射、序列-结构映射都是两个
latent space 之间的映射. 这里只存数据, 映射学习见 alignment.py.

通用: 不绑定材料科学. (structure, haptic) 能存, (sequence, structure) 也能存.
数据来源: DFT 计算 / ML potential 预估 / 数据库查询 / 任意 (source, target) 对.

接入点: alignment.AlignmentFunction.fit 读 get_pairs; rcb_runner 加载/保存;
tools/adapter.py 力学结果后自动 add.

ponytail: JSON 序列化, 不上 SQLite. 数据量小 (百~千对), 跨任务复用靠文件.
ceiling: 全量重写 save, 没做增量 append. 升级路径: JSONL 追加写或换 SQLite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class AlignmentDataset:
    """存任意两个 LatentSpace 的对齐数据对.

    内部就是 list[dict], 每条记录 source_vec / target_vec / 空间名 / metadata.
    vec 存成 list (JSON 友好), 取出时 get_pairs 转回 np.ndarray.
    """

    def __init__(self):
        self._pairs: list[dict] = []

    def add(
        self,
        source_vec: np.ndarray,
        target_vec: np.ndarray,
        source_name: str,
        target_name: str,
        metadata: dict | None = None,
    ) -> None:
        """存一对 (source, target) 对齐数据.

        metadata 可选, 用于记录数据来源/置信度/计算参数等.
        """
        self._pairs.append(
            {
                "source_vec": np.asarray(source_vec, dtype=float).tolist(),
                "target_vec": np.asarray(target_vec, dtype=float).tolist(),
                "source_name": source_name,
                "target_name": target_name,
                "metadata": metadata or {},
            }
        )

    def get_pairs(
        self, source_name: str, target_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """取出某对空间的全部对齐数据, 返回 (X, y).

        无匹配返回 (empty(0,0), empty(0,0)) — 调用方按 len(X) 判空, 不抛异常.
        """
        X: list[list[float]] = []
        y: list[list[float]] = []
        for p in self._pairs:
            if p["source_name"] == source_name and p["target_name"] == target_name:
                X.append(p["source_vec"])
                y.append(p["target_vec"])
        if not X:
            return np.empty((0, 0)), np.empty((0, 0))
        return np.array(X), np.array(y)

    def count(
        self,
        source_name: str | None = None,
        target_name: str | None = None,
    ) -> int:
        """数据对数量. 不传参返回总数, 传了按空间名过滤."""
        if source_name is None and target_name is None:
            return len(self._pairs)
        return sum(
            1
            for p in self._pairs
            if (source_name is None or p["source_name"] == source_name)
            and (target_name is None or p["target_name"] == target_name)
        )

    def save(self, path: str | Path) -> None:
        """JSON 序列化到 path."""
        Path(path).write_text(json.dumps(self._pairs), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AlignmentDataset":
        """从 JSON 加载. 文件不存在/损坏会抛对应异常 — 由调用方决定降级."""
        ds = cls()
        text = Path(path).read_text(encoding="utf-8")
        ds._pairs = json.loads(text)
        return ds
