"""EncounterSpace — 情景索引 latent space (episodic replay Step 2).

把 episodic entry 编码为向量: 时间 + 空间 + 内部状态.
跟 AlignmentFunction 同一套 LatentSpace ABC, 可做 cue-based 检索/采样.

复用 StructureDescriptor 的 16 维空间维度, 加 4 维 (norm_iter, mode, phase, val_status).
"""
from __future__ import annotations

import numpy as np

from huginn.metacog.latent_space import LatentSpace

_MODE_ORD = {"": 0.0, "explore": 0.2, "exploit": 0.4, "verify": 0.6, "reflect": 0.8, "debug": 1.0}
_PHASE_ORD = {"": 0.0, "hypothesize": 0.25, "plan": 0.5, "execute": 0.75, "validate": 1.0}
_VAL_ORD = {"none": 0.0, "failed": 0.3, "passed": 1.0}


class EncounterSpace(LatentSpace):
    """情景索引: 时间 + 空间 + 内部状态 + surprise."""

    STRUCTURE_DIM = 16

    @property
    def dim(self) -> int:
        # 16 structure + norm_iter + mode + phase + val_status + surprise
        return self.STRUCTURE_DIM + 5

    @property
    def name(self) -> str:
        return "encounter"

    def encode(self, entry) -> np.ndarray:
        """entry: episodic entry dict (含 iter/mode/phase/val_status/structure_desc/surprise)."""
        # 结构维度 (16维)
        sd = entry.get("structure_desc") or []
        if isinstance(sd, list) and len(sd) >= self.STRUCTURE_DIM:
            struct_vec = np.array(sd[:self.STRUCTURE_DIM], dtype=float)
        else:
            struct_vec = np.zeros(self.STRUCTURE_DIM, dtype=float)

        # 时间维度 (1维): 归一化 iter
        # ponytail: 粗归一化 /1000, 不做动态 scale. ceiling: iter > 1000 时值 > 1, 但 L2 仍单调.
        norm_iter = float(entry.get("iter", 0)) / 1000.0

        # 内部状态 (3维)
        mode_v = _MODE_ORD.get(str(entry.get("mode", "")), 0.0)
        phase_v = _PHASE_ORD.get(str(entry.get("phase", "")), 0.0)
        val_v = _VAL_ORD.get(str(entry.get("val_status", "")), 0.0)

        # 桥 J: surprise 维度 — 高 surprise 情境是发现信号, replay 时能按 surprise
        # 回溯"上次高 surprise 时做了什么". ponytail: 直接 clip 到 [0,1], 不做 z-score.
        # ceiling: surprise 原始范围未校准, clip 会损失极端值区分度. 升级: running stats 归一化.
        surprise_v = float(entry.get("surprise", 0.0))
        surprise_v = max(0.0, min(1.0, surprise_v))

        return np.concatenate([struct_vec, [norm_iter, mode_v, phase_v, val_v, surprise_v]])
