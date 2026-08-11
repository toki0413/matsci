"""Episodic memory 分片存储 — 按 task_id + iter_range 切 jsonl + gzip 归档.

设计:
- 每个 task 一组 shard, 路径: <workspace>/.huginn/memory/episodic/<task_id>/shard_<start>_<end>.jsonl
- shard_interval 控制 shard 大小 (默认 100 iter), env HUGINN_EPISODIC_SHARD_INTERVAL 可调
- 写满一个 shard (iter 跨边界) → 自动 gzip 归档到 memory/archive/<task_id>/shard_<start>_<end>.jsonl.gz
- flush_and_archive 显式归档当前未满 shard (task 结束时调一次)
- 跨 shard 查询: reader 同时扫 .jsonl (active) + .jsonl.gz (archived)

升级路径: semantic memory 用 KG 压缩 — 复用 metacog/unified_complex.py 的 KG 路径,
当前只在每个 entry 留 metadata.kg_summary 字段, 不实际接 KG.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import os
import shutil
import time
from pathlib import Path

from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 100


def _episodic_dir(workspace: Path, task_id: str) -> Path:
    return Path(workspace).resolve() / HUGINN_DIR_NAME / "memory" / "episodic" / task_id


def _archive_dir(workspace: Path, task_id: str) -> Path:
    return Path(workspace).resolve() / HUGINN_DIR_NAME / "memory" / "archive" / task_id


def _parse_shard_name(name: str) -> tuple[int, int] | None:
    """从 shard 文件名解析 (start, end). 失败返 None.

    支持 shard_<s>_<e>.jsonl 和 shard_<s>_<e>.jsonl.gz 两种.
    """
    stem = name
    if stem.endswith(".jsonl.gz"):
        stem = stem[: -len(".jsonl.gz")]
    elif stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    else:
        return None
    if not stem.startswith("shard_"):
        return None
    stem = stem[len("shard_"):]
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        logger.debug("best-effort op failed", exc_info=True)
        return None


class EpisodicShardWriter:
    """按 iter 范围分片写 episodic memory, 老分片 gzip 压缩归档.

    不线程安全 — 单 writer 单文件, 多线程需外部锁.
    """

    def __init__(
        self,
        workspace: Path,
        task_id: str,
        shard_interval: int | None = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.task_id = task_id
        if shard_interval is None:
            shard_interval = int(
                os.environ.get("HUGINN_EPISODIC_SHARD_INTERVAL", _DEFAULT_INTERVAL)
            )
        if shard_interval <= 0:
            raise ValueError(f"shard_interval must be > 0, got {shard_interval}")
        self.shard_interval = shard_interval

        self._episodic_dir = _episodic_dir(self.workspace, task_id)
        self._archive_dir = _archive_dir(self.workspace, task_id)
        self._episodic_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        # 当前 shard 状态
        self._cur_start: int | None = None
        self._cur_end: int | None = None
        self._fh = None  # 当前打开的 file handle

    def _shard_bounds(self, iter_n: int) -> tuple[int, int]:
        """iter_n 所属 shard 的 [start, end] (end inclusive)."""
        idx = iter_n // self.shard_interval
        start = idx * self.shard_interval
        end = start + self.shard_interval - 1
        return start, end

    def _open_shard(self, start: int, end: int) -> None:
        path = self._episodic_dir / f"shard_{start}_{end}.jsonl"
        # append 模式 + line buffering, 防覆盖 + 防中途崩溃丢数据
        self._fh = path.open("a", encoding="utf-8", buffering=1)
        self._cur_start = start
        self._cur_end = end
        logger.debug("episodic shard open: %s", path)

    def _close_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _archive_current(self) -> Path | None:
        """gzip 压缩当前 shard, 移到 archive/. 没有则返 None."""
        if self._cur_start is None or self._cur_end is None:
            return None
        src = self._episodic_dir / f"shard_{self._cur_start}_{self._cur_end}.jsonl"
        self._close_shard()
        if not src.exists() or src.stat().st_size == 0:
            # 空文件不归档, 直接删
            with contextlib.suppress(FileNotFoundError):
                src.unlink()
            self._cur_start = None
            self._cur_end = None
            return None
        dst = self._archive_dir / f"shard_{self._cur_start}_{self._cur_end}.jsonl.gz"
        with src.open("rb") as fin, gzip.open(dst, "wb") as gout:
            shutil.copyfileobj(fin, gout)
        src.unlink()
        archived_path = dst
        self._cur_start = None
        self._cur_end = None
        logger.debug("episodic shard archived: %s", archived_path)
        return archived_path

    def append(self, iter_n: int, entry: dict) -> None:
        """写一条 entry 到当前 shard, iter 跨边界时切新 shard + 归档旧 shard."""
        start, end = self._shard_bounds(iter_n)
        if self._cur_start is None:
            self._open_shard(start, end)
        elif start != self._cur_start:
            # 跨边界: 归档旧 shard, 开新 shard
            self._archive_current()
            self._open_shard(start, end)
        # entry 附 iter + ts, 方便后续查询排序
        record = {
            "iter": iter_n,
            "ts": time.time(),
            "entry": entry,
        }
        assert self._fh is not None  # mypy hint
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def flush_and_archive(self) -> Path:
        """归档当前未满 shard. task 结束时调用一次.

        返回归档文件路径; 当前没有可归档 shard 时返回 archive 目录.
        """
        archived = self._archive_current()
        if archived is not None:
            return archived
        # 没东西归档, 返 archive dir 占位 (调用方判空文件失败时再处理)
        return self._archive_dir


class EpisodicShardReader:
    """跨分片读 episodic memory, 同时支持 active (.jsonl) + archived (.jsonl.gz)."""

    def __init__(self, workspace: Path, task_id: str):
        self.workspace = Path(workspace).resolve()
        self.task_id = task_id
        self._episodic_dir = _episodic_dir(self.workspace, task_id)
        self._archive_dir = _archive_dir(self.workspace, task_id)

    def _all_shards(self) -> list[Path]:
        """所有 shard 文件 (active + archived), 按 start iter 升序."""
        shards: list[Path] = []
        if self._episodic_dir.exists():
            shards.extend(self._episodic_dir.glob("shard_*.jsonl"))
        if self._archive_dir.exists():
            shards.extend(self._archive_dir.glob("shard_*.jsonl.gz"))

        def _start_key(p: Path) -> int:
            b = _parse_shard_name(p.name)
            return b[0] if b is not None else -1

        shards.sort(key=_start_key)
        return shards

    def _read_shard(self, path: Path) -> list[dict]:
        """读一个 shard 文件 (自动区分 .jsonl / .jsonl.gz)."""
        out: list[dict] = []
        if path.name.endswith(".gz"):
            opener = gzip.open(path, "rt", encoding="utf-8")  # noqa: SIM115
        else:
            opener = path.open("r", encoding="utf-8")
        try:
            with opener as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 损坏行跳过, 不让一行坏数据干废整个 reader
                        logger.debug("best-effort op failed", exc_info=True)
                        continue
        except OSError as e:
            logger.warning("episodic shard read fail %s: %s", path, e)
        return out

    def iter_range(self, start_iter: int, end_iter: int) -> list[dict]:
        """跨分片读 [start_iter, end_iter] 范围内的 entry (两端 inclusive)."""
        if start_iter > end_iter:
            return []
        out: list[dict] = []
        for shard_path in self._all_shards():
            bounds = _parse_shard_name(shard_path.name)
            if bounds is None:
                continue
            s_start, s_end = bounds
            # shard 完全在范围外 → 跳过
            if s_end < start_iter or s_start > end_iter:
                continue
            for r in self._read_shard(shard_path):
                it = r.get("iter")
                if it is None:
                    continue
                if start_iter <= it <= end_iter:
                    out.append(r)
        out.sort(key=lambda r: r.get("iter", 0))
        return out

    def recent(self, n: int = 100) -> list[dict]:
        """读最近 n 条 (跨分片, 按 iter 降序取前 n, 再升序返回)."""
        all_records: list[dict] = []
        for shard_path in self._all_shards():
            all_records.extend(self._read_shard(shard_path))
        all_records.sort(key=lambda r: r.get("iter", 0), reverse=True)
        top = all_records[:n]
        top.reverse()  # 时间升序输出, 跟 iter_range 一致
        return top


if __name__ == "__main__":
    # self-check: 模拟 1000 iter 的 episodic memory,
    # 验证分片到 10 个文件 + flush_and_archive 后老分片 gzip 压缩
    import tempfile as _tf

    ws = Path(_tf.mkdtemp(prefix="huginn_episodic_")) / "ws"
    ws.mkdir()
    try:
        interval = 100
        os.environ["HUGINN_EPISODIC_SHARD_INTERVAL"] = str(interval)
        writer = EpisodicShardWriter(ws, task_id="self_check")
        # 1000 iter, 每 iter 一条 entry. 期望触发 9 次自动归档 (iter 100/200/.../900),
        # 之后 active shard = shard_900_999.jsonl, archive 里 9 个 .gz
        for i in range(1000):
            writer.append(i, {"event": "step", "score": i * 0.001})
        # flush_and_archive 把当前剩余 shard 也归档 → archive 10 个, active 0 个
        archived = writer.flush_and_archive()
        assert archived.exists(), f"archive path missing: {archived}"

        reader = EpisodicShardReader(ws, task_id="self_check")

        # 1. archive 里有 10 个 .jsonl.gz, active 里 0 个 .jsonl
        archived_files = sorted(reader._archive_dir.glob("shard_*.jsonl.gz"))
        assert len(archived_files) == 10, (
            f"expected 10 archived shards, got {len(archived_files)}: "
            f"{[p.name for p in archived_files]}"
        )
        active_files = list(reader._episodic_dir.glob("shard_*.jsonl"))
        assert len(active_files) == 0, (
            f"expected 0 active shards, got {len(active_files)}"
        )
        print(
            "1. 1000 iter → 10 archived shards OK: "
            f"{[p.name for p in archived_files]}"
        )

        # 2. iter_range 跨 shard 边界 (95..105) → 11 条
        rng = reader.iter_range(95, 105)
        assert len(rng) == 11, f"expected 11, got {len(rng)}"
        assert rng[0]["iter"] == 95
        assert rng[-1]["iter"] == 105
        print(f"2. iter_range(95,105) → {len(rng)} records OK")

        # 3. recent(5) 返回最近 5 条 (iter 995..999)
        rec = reader.recent(5)
        assert len(rec) == 5, f"expected 5, got {len(rec)}"
        assert [r["iter"] for r in rec] == [995, 996, 997, 998, 999], (
            f"iters wrong: {[r['iter'] for r in rec]}"
        )
        print(f"3. recent(5) → iters {[r['iter'] for r in rec]} OK")

        # 4. 完整范围读 1000 条
        all_recs = reader.iter_range(0, 999)
        assert len(all_recs) == 1000, f"expected 1000, got {len(all_recs)}"
        assert all_recs[0]["iter"] == 0
        assert all_recs[-1]["iter"] == 999
        print(f"4. iter_range(0,999) → {len(all_recs)} records OK")

        # 5. iter_range 反向 (start > end) → 空列表
        assert reader.iter_range(500, 100) == []
        print("5. iter_range(start>end) → [] OK")

        # 6. shard_interval 从 env 读
        os.environ["HUGINN_EPISODIC_SHARD_INTERVAL"] = "50"
        try:
            w2 = EpisodicShardWriter(ws, task_id="env_test")
            assert w2.shard_interval == 50, f"env override failed: {w2.shard_interval}"
        finally:
            del os.environ["HUGINN_EPISODIC_SHARD_INTERVAL"]
        print("6. HUGINN_EPISODIC_SHARD_INTERVAL env override OK")

        # 7. 默认 shard_interval = 100
        w3 = EpisodicShardWriter(ws, task_id="default_test")
        assert w3.shard_interval == 100
        print("7. default shard_interval = 100 OK")

        # 8. 边界 iter 精确归档 — shard_0_99 应在第 100 次 append 时归档
        os.environ["HUGINN_EPISODIC_SHARD_INTERVAL"] = "10"
        try:
            w4 = EpisodicShardWriter(ws, task_id="edge_test")
            for i in range(15):
                w4.append(i, {"i": i})
            # 0..9 写完, 第 10 次 append 触发归档 → archive 里 1 个 .gz
            # active 里 shard_10_19.jsonl (写了 10..14 共 5 条)
            r4 = EpisodicShardReader(ws, task_id="edge_test")
            arch4 = list(r4._archive_dir.glob("shard_*.jsonl.gz"))
            act4 = list(r4._episodic_dir.glob("shard_*.jsonl"))
            assert len(arch4) == 1, f"expected 1 archived, got {len(arch4)}"
            assert arch4[0].name == "shard_0_9.jsonl.gz"
            assert len(act4) == 1, f"expected 1 active, got {len(act4)}"
            assert act4[0].name == "shard_10_19.jsonl"
            rng4 = r4.iter_range(5, 12)
            assert len(rng4) == 8, f"expected 8, got {len(rng4)}"
            print(f"8. 边界归档 OK: {arch4[0].name} + {act4[0].name}")
        finally:
            del os.environ["HUGINN_EPISODIC_SHARD_INTERVAL"]

        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)
