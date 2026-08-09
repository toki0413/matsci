"""EpisodicIndexStore — 情景索引存储 (episodic replay Step 3).

用 EpisodicShardReader 读全部 episodic entry, 用 EncounterSpace 编码成向量,
建内存索引. 支持 cue_retrieve (top-k 最近邻) 和 sample (softmax 采样).

通用: 不绑定材料科学. EncounterSpace 可替换成任何 LatentSpace 子类.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from huginn.memory.episodic_shard import EpisodicShardReader
from huginn.metacog.encounter_space import EncounterSpace


class EpisodicIndexStore:
    def __init__(self, workspace: Path, task_id: str, space: EncounterSpace | None = None):
        self.workspace = Path(workspace)
        self.task_id = task_id
        self.space = space or EncounterSpace()
        self._entries: list[dict] = []
        self._vectors: np.ndarray | None = None

    def build(self):
        """读全部 episodic shard, 编码成向量索引."""
        reader = EpisodicShardReader(self.workspace, self.task_id)
        all_records = reader.iter_range(0, 10**9)  # 全部
        self._entries = [r.get("entry", r) for r in all_records]
        if self._entries:
            self._vectors = np.array([self.space.encode(e) for e in self._entries])
        else:
            self._vectors = np.empty((0, self.space.dim))
        return len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def ready(self) -> bool:
        return self._vectors is not None and len(self._vectors) > 0

    def cue_retrieve(self, cue_vec: np.ndarray, top_k: int = 5) -> list[tuple[dict, float]]:
        """沿时空索引检索最近邻. 返回 [(entry, distance), ...] 按距离升序."""
        if not self.ready:
            return []
        cue = np.asarray(cue_vec, dtype=float)
        diffs = self._vectors - cue
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argsort(dists)[:top_k]
        return [(self._entries[i], float(dists[i])) for i in idx]

    def sample(self, cue_vec: np.ndarray, temperature: float = 1.0) -> dict | None:
        """softmax(-d/tau) 加权采样, 不是硬 top-k."""
        if not self.ready:
            return None
        cue = np.asarray(cue_vec, dtype=float)
        diffs = self._vectors - cue
        dists = np.linalg.norm(diffs, axis=1)
        # softmax(-d/tau)
        logits = -dists / max(temperature, 1e-8)
        logits -= logits.max()  # 数值稳定
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        return self._entries[idx]
