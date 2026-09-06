"""EpisodicReplay — cue-based 情景重放 (episodic replay Step 4).

沿时空索引采样到最相关情境, 重放其轨迹.
成功情境 → "复用方法"; 失败情境 → "避免同样失败".

复用 EpisodicIndexStore 的 cue_retrieve (top-k 最近邻) 和 sample (softmax 采样).
"""
from __future__ import annotations

from typing import Any

from huginn.metacog.encounter_space import EncounterSpace
from huginn.metacog.episodic_index import EpisodicIndexStore


class EpisodicReplay:
    """沿时空索引采样回溯, 重放最相关情境的轨迹."""

    def __init__(self, index_store: EpisodicIndexStore, space: EncounterSpace | None = None):
        self.store = index_store
        # 没传 space 就复用 store 的 (保证和建索引时用同一个), 再不行才新建
        self.space = space or getattr(index_store, "space", None) or EncounterSpace()

    @staticmethod
    def _unwrap(record: dict) -> dict:
        """record (iter/ts/entry) → flat entry. 已经 flat 的原样返回.

        store 现在存的就是 flat entry (build 时 r.get('entry', r) 拆过),
        但留着这层防御: 万一 store 改回存 record, 这里不会炸.
        """
        inner = record.get("entry")
        if isinstance(inner, dict):
            flat = dict(inner)
            flat.setdefault("iter", record.get("iter", -1))
            flat.setdefault("ts", record.get("ts", 0.0))
            return flat
        return record

    @staticmethod
    def _build_replay(entry: dict, distance: float | None) -> dict[str, Any]:
        val = entry.get("val_status", "none")
        advice_raw = (entry.get("advice") or "")[:150]
        if val == "passed":
            replay_type = "replay_success"
            advice = f"复用方法: {advice_raw}"
        elif val == "failed":
            replay_type = "replay_failure"
            advice = f"避免同样失败: {advice_raw}"
        else:
            replay_type = "replay_neutral"
            advice = advice_raw
        out: dict[str, Any] = {
            "iter": entry.get("iter", -1),
            "action": (entry.get("action") or "")[:200],
            "advice": advice,
            "val_status": val,
            "replay_type": replay_type,
        }
        if distance is not None:
            out["distance"] = round(float(distance), 4)
        return out

    def replay(self, cue_entry: dict, top_k: int = 3) -> list[dict[str, Any]]:
        """给定当前 cue, 检索 top_k 最相关情境, 重放轨迹.

        cue_entry: 当前情境 dict (含 mode/phase/val_status/structure_desc 等)
        返回: [{"iter","action","advice","val_status","distance","replay_type"}, ...]
        """
        if not self.store.ready:
            return []
        cue_vec = self.space.encode(cue_entry)
        results = self.store.cue_retrieve(cue_vec, top_k=top_k)
        return [self._build_replay(self._unwrap(r), d) for r, d in results]

    def sample_and_replay(self, cue_entry: dict, temperature: float = 1.0) -> dict[str, Any] | None:
        """softmax 采样一个情境并重放 (不是硬 top-k). 没数据返 None."""
        if not self.store.ready:
            return None
        cue_vec = self.space.encode(cue_entry)
        record = self.store.sample(cue_vec, temperature=temperature)
        if record is None:
            return None
        return self._build_replay(self._unwrap(record), distance=None)


if __name__ == "__main__":
    # self-check: 写 20 条 entry, 建索引, replay top-3 + sample_and_replay
    import tempfile
    from pathlib import Path

    from huginn.memory.episodic_shard import EpisodicShardWriter

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        ws.mkdir()
        writer = EpisodicShardWriter(ws, task_id="self_check")
        for i in range(20):
            writer.append(i, {
                "iter": i, "mode": "explore", "phase": "execute",
                "val_status": "passed" if i % 3 == 0 else "failed",
                "structure_desc": [float(i)] * 16,
                "action": f"action_{i}", "advice": f"advice_{i}",
            })
        writer.flush_and_archive()

        store = EpisodicIndexStore(ws, "self_check")
        n = store.build()
        assert n == 20, f"build 应返回 20, got {n}"

        replay = EpisodicReplay(store)
        cue = {
            "iter": 5, "mode": "explore", "phase": "execute",
            "val_status": "passed", "structure_desc": [5.0] * 16,
        }
        results = replay.replay(cue, top_k=3)
        assert len(results) == 3, f"top-3 应返 3 条, got {len(results)}"
        for r in results:
            assert "distance" in r, "replay() 结果应有 distance"
            assert r["replay_type"] in ("replay_success", "replay_failure", "replay_neutral")
            assert r["iter"] != -1, "iter 不应是默认 -1"
        print(f"replay top-3: iters={[r['iter'] for r in results]}, "
              f"types={[r['replay_type'] for r in results]}")

        s = replay.sample_and_replay(cue, temperature=1.0)
        assert s is not None, "sample_and_replay 不应返 None (有 20 条)"
        assert "distance" not in s, "sample_and_replay 不带 distance"
        assert s["replay_type"] in ("replay_success", "replay_failure", "replay_neutral")
        print(f"sample_and_replay: iter={s['iter']}, type={s['replay_type']}")

        # 空索引 → replay 返 [], sample 返 None
        empty_store = EpisodicIndexStore(ws, "no_such_task")
        empty_store.build()
        empty_replay = EpisodicReplay(empty_store)
        assert empty_replay.replay(cue) == []
        assert empty_replay.sample_and_replay(cue) is None
        print("empty store: replay→[], sample→None OK")

        print("ALL OK")
