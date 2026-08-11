"""Audit log subscriber — writes every event to ~/.huginn/events/audit.jsonl.

This is the provenance trail: append-only, one JSON object per line.
Inspired by Codex's rollout.jsonl — if something went wrong, you can
replay the audit log to reconstruct the full session timeline.

Tamper-evident: each record carries a SHA-256 hash of its content chained
to the previous record's hash. Any modification breaks the chain.
Design borrowed from OpenParallax's audit module.

Usage:
    from huginn.events.audit_log import install_audit_subscriber
    install_audit_subscriber()  # call once at startup

After that, every published event lands in audit.jsonl automatically.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.events.event_bus import AgentEvent, EventBus
from huginn.events.event_types import ALL
from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)

# 缓冲写入: 攒够 _FLUSH_BATCH 条或 _FLUSH_INTERVAL 秒后一次性落盘.
# 避免 Windows AV 对每次 open+write+close 都扫一遍.
_FLUSH_BATCH = 20
_FLUSH_INTERVAL = 2.0  # seconds

# 默认分片步长. RCB 跑 700 万 call 时单文件会撑爆磁盘 + grep/cat 都进不了,
# 每 N iter 切一个 jsonl, 老分片 gzip 归档. 用 HUGINN_TRACE_SHARD_INTERVAL env
# 覆盖, 默认 100. ponytail: 跨进程恢复时只扫当前分片末尾续 hash chain, 老分片
# 不重读. 升级路径: 整套替换成 Postgres (主从) 或 Cassandra (按 task_id 分桶,
# iter 作 cluster key), 文件分片只在单机 dev 场景留作 fallback.
_DEFAULT_SHARD_INTERVAL = 100


def _resolve_shard_interval() -> int:
    """从 HUGINN_TRACE_SHARD_INTERVAL 读分片步长, 异常回默认 100."""
    raw = os.environ.get("HUGINN_TRACE_SHARD_INTERVAL", str(_DEFAULT_SHARD_INTERVAL))
    try:
        v = int(raw)
        return v if v > 0 else _DEFAULT_SHARD_INTERVAL
    except (TypeError, ValueError):
        return _DEFAULT_SHARD_INTERVAL


def _gzip_file(src: Path, dst: Path) -> None:
    """gzip 压缩 src 到 dst, 完成后删 src. 原子性靠 tmp rename 兜底."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(src, "rb") as fin, gzip.open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.replace(tmp, dst)
    # 老分片已经被其他进程清掉就算了, 不阻塞
    with contextlib.suppress(OSError):
        src.unlink()


@dataclass
class _ShardState:
    """分片写入状态机, AuditLogger / meta_trace 共用.

    基本契约:
      - task_id 未设时, 所有写入走 default_path (单文件, 行为同 v1)
      - task_id 设置后, 按 iter_n // shard_interval 切分片, 文件名
        ``<base_dir>/<task_id>/<prefix>_<shard_start>.jsonl``
      - 跨 shard 边界时老分片 gzip 到
        ``<base_dir>/archive/<task_id>/<prefix>_<old_start>_<old_end>.jsonl.gz``

    ponytail: 不做并发锁, 调用方自己串行化 (AuditLogger 用 _lock,
    rcb_runner 主循环天然单线程). 升级路径: 状态外移到 sqlite/meta_db,
    让多进程能共享分片进度.
    """
    base_dir: Path
    default_path: Path | None = None
    task_id: str | None = None
    shard_interval: int = field(default_factory=_resolve_shard_interval)
    filename_prefix: str = "audit"
    current_shard_start: int = 0
    current_shard_path: Path | None = None

    def set_task_id(self, task_id: str) -> None:
        """RCB 任务启动时调, 后续写入按 task_id 分桶."""
        if self.task_id == task_id:
            return
        self.task_id = task_id
        self.current_shard_start = 0
        self.current_shard_path = None

    def _shard_path(self, start: int) -> Path:
        if not self.task_id:
            return self.default_path or (self.base_dir / f"{self.filename_prefix}.jsonl")
        d = self.base_dir / self.task_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.filename_prefix}_{start}.jsonl"

    def _archive_path(self, start: int, end: int) -> Path:
        d = self.base_dir / "archive" / self.task_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.filename_prefix}_{start}_{end}.jsonl.gz"

    def maybe_rotate(self, iter_n: int | None) -> Path:
        """根据 iter_n 决定目标 path, 跨边界时归档老分片."""
        if not self.task_id:
            return self.default_path or (self.base_dir / f"{self.filename_prefix}.jsonl")
        if iter_n is None or self.shard_interval <= 0:
            # task_id 设了但事件没带 iteration (e.g. tool.call), 留在当前分片
            if self.current_shard_path is None:
                self.current_shard_path = self._shard_path(self.current_shard_start)
            return self.current_shard_path
        shard_start = (iter_n // self.shard_interval) * self.shard_interval
        if self.current_shard_path is None:
            self.current_shard_start = shard_start
            self.current_shard_path = self._shard_path(shard_start)
            return self.current_shard_path
        if shard_start != self.current_shard_start:
            old_path = self.current_shard_path
            old_start = self.current_shard_start
            old_end = shard_start - 1
            if old_path.exists():
                _gzip_file(old_path, self._archive_path(old_start, old_end))
            self.current_shard_start = shard_start
            self.current_shard_path = self._shard_path(shard_start)
        return self.current_shard_path


def write_sharded_jsonl(state: _ShardState, record: dict, iter_n: int | None) -> None:
    """单条 record 写入分片 jsonl. AuditLogger flush 和 meta_trace 复用.

    失败只 log debug, 不抛 — 跟原 audit_log 一致, 别让 trace 写挂拖垮主循环.
    """
    try:
        target = state.maybe_rotate(iter_n)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("sharded jsonl write failed (iter=%s)", iter_n, exc_info=True)


def shard_iter_range(
    state: _ShardState,
    start_iter: int,
    end_iter: int,
) -> list[dict]:
    """跨分片扫描 [start_iter, end_iter] 范围的记录.

    老分片 (.gz) + 当前分片 (.jsonl) 都扫, 按文件 natural order 拼回.
    记录没带 iteration 字段的直接跳过 — 这类事件 (tool.call 等) 跟 iter
    无关, 不该出现在 iter_range 结果里.
    """
    records: list[dict] = []
    paths: list[Path] = []
    if state.task_id:
        arch_dir = state.base_dir / "archive" / state.task_id
        if arch_dir.exists():
            paths.extend(sorted(arch_dir.glob(f"{state.filename_prefix}_*_*.jsonl.gz")))
        if state.current_shard_path and state.current_shard_path.exists():
            paths.append(state.current_shard_path)
    else:
        single = state.default_path or (state.base_dir / f"{state.filename_prefix}.jsonl")
        if single.exists():
            paths.append(single)

    for p in paths:
        try:
            def opener(pp=p):
                return (gzip.open(pp, "rt", encoding="utf-8")
                            if pp.suffix == ".gz"
                            else open(pp, encoding="utf-8"))
            with opener() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("best-effort op failed", exc_info=True)
                        continue
                    it = (rec.get("data") or {}).get("iteration")
                    if it is None:
                        it = rec.get("iteration")
                    if it is None:
                        continue
                    try:
                        it = int(it)
                    except (TypeError, ValueError):
                        logger.debug("best-effort op failed", exc_info=True)
                        continue
                    if start_iter <= it <= end_iter:
                        records.append(rec)
        except Exception:
            logger.debug("shard scan failed for %s", p, exc_info=True)
            continue
    return records


_unsubscribe: Any = None  # stored so we can detach if needed


def _resolve_audit_path() -> Path:
    """Figure out where audit.jsonl lives.

    Tries the runtime home (HUGINN_CACHE_DIR or ~/.huginn), falls back
    to ~/.huginn directly. Never raises — worst case we write to the
    home directory.
    """
    base = get_runtime_home()
    events_dir = base / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir / "audit.jsonl"


# P0-2: Audit secret 脱敏. Portions derived from OpenWorker (MIT, Andrew Ng 2024).
# 插在 _compute_hash 之前调用, 保证 hash 是脱敏后的, 防 hash 泄露 secret.
_SECRET_KEYS = (
    "token", "secret", "password", "api_key", "apikey",
    "access_token", "bot_token", "app_token", "raw", "credential",
)
_BODY_KEYS = ("body", "content", "html", "text", "payload")


def _truncate(text: str, limit: int = 500) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summarize(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_summarize(v) for v in value[:10]]
    if isinstance(value, dict):
        return {str(k): _summarize(v) for k, v in list(value.items())[:20]}
    return _truncate(str(value))


def _sanitize_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Redact secret fields + summarize large values before hashing."""
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        lk = str(key).lower()
        if any(s in lk for s in _SECRET_KEYS):
            out[key] = "[redacted]"
        elif any(b == lk or lk.endswith("_" + b) for b in _BODY_KEYS):
            out[key] = "[redacted body]"
        else:
            out[key] = _summarize(value)
    return out


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """对 record.data.args 做脱敏. 返回新 dict, 不改原 record."""
    data = record.get("data")
    if not isinstance(data, dict):
        return record
    args = data.get("args")
    if not isinstance(args, dict):
        return record
    tool = str(data.get("tool") or data.get("event_type") or "")
    new_args = _sanitize_args(tool, args)
    # 不改原 data, 新建一份
    new_data = {**data, "args": new_args}
    return {**record, "data": new_data}


def _compute_hash(payload: dict[str, Any], prev_hash: str) -> str:
    """SHA-256 over canonical JSON of (payload_without_hash_fields, prev_hash).

    We strip _hash and _prev_hash from the payload before hashing so the
    hash only covers the event content, not the chain metadata itself.
    """
    content = {k: v for k, v in payload.items() if k not in ("_hash", "_prev_hash")}
    blob = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{blob}{prev_hash}".encode()).hexdigest()


class _BufferedAuditWriter:
    """缓冲写入器: 攒一批再 flush, 减少 open/close 次数.

    线程安全. flush 在后台 daemon 线程跑, 主线程只往 deque 里塞.
    进程退出时 daemon 线程自动终止, 最多丢 _FLUSH_BATCH 条未落盘事件.

    Hash chain: 每个 record 的 _hash = SHA-256(content + _prev_hash).
    链头 _prev_hash = "0" * 64 (genesis). 任何篡改都会断链.

    v26 Task 26.11: 加按 task_id + iter_range 分片. task_id 通过 set_task_id
    在 RCB 任务启动时设置; 之后 flush 按 event.data.iteration 路由到对应分片,
    老分片 gzip 归档到 archive/<task_id>/. ponytail: 跨进程恢复只读当前分片末尾
    续 hash, 老分片不重读 — 升级到 Postgres/Cassandra 时由 DB 维护全局链.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._base_dir = path.parent
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: deque[dict[str, Any]] = deque()
        self._last_flush = time.monotonic()
        self._stop = threading.Event()
        self._prev_hash = self._load_last_hash()
        # 分片状态. task_id 未设时退化为单文件 (兼容旧 audit.jsonl).
        self._shard = _ShardState(
            base_dir=self._base_dir,
            default_path=self._path,
            shard_interval=_resolve_shard_interval(),
            filename_prefix="audit",
        )
        self._thread = threading.Thread(
            target=self._flush_loop, name="audit-writer", daemon=True,
        )
        self._thread.start()

    @property
    def task_id(self) -> str | None:
        return self._shard.task_id

    @property
    def shard_interval(self) -> int:
        return self._shard.shard_interval

    @property
    def current_shard_start(self) -> int:
        return self._shard.current_shard_start

    @property
    def current_shard_path(self) -> Path | None:
        return self._shard.current_shard_path

    def set_task_id(self, task_id: str) -> None:
        """RCB 任务启动时调, 后续 flush 按 task_id 分桶写.

        会先把 buffer 里的事件 flush 到老 path 再切, 避免跨 task 串到同一分片.
        """
        with self._lock:
            self._flush_locked()
            self._shard.set_task_id(task_id)

    def _maybe_rotate_shard(self, iter_n: int | None) -> Path:
        """根据 iter_n 路由到当前应写入的分片 path. 调用方需持有 _lock."""
        return self._shard.maybe_rotate(iter_n)

    def iter_range(self, start_iter: int, end_iter: int) -> list[dict]:
        """跨分片读取 [start_iter, end_iter] 范围记录. RCB resume / 调试用."""
        with self._lock:
            # 先把 buffer 里的 flush 出去, 否则 iter_range 看不到
            self._flush_locked()
            return shard_iter_range(self._shard, start_iter, end_iter)

    def _load_last_hash(self) -> str:
        """Read the last _hash from an existing audit file to resume the chain.

        Returns "0"*64 if the file is missing, empty, or corrupted —
        a fresh chain is always valid.

        ponytail: 只扫默认 audit.jsonl. 分片场景下若 task_id 已知, 应当扫
        archive/<task_id>/*.gz + 当前分片末尾 — 但跨进程 resume RCB 本就
        罕见, 留给 Postgres/Cassandra 迁移时一并处理.
        """
        genesis = "0" * 64
        try:
            if not self._path.exists():
                return genesis
            last_hash = genesis
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        h = rec.get("_hash")
                        if h:
                            last_hash = h
                    except json.JSONDecodeError:
                        logger.debug("best-effort op failed", exc_info=True)
                        continue
            return last_hash
        except Exception:
            return genesis

    def append(self, event: AgentEvent) -> None:
        with self._lock:
            self._buffer.append(event.to_dict())
            should_flush = len(self._buffer) >= _FLUSH_BATCH
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Caller 必须持有 self._lock. 同批 events 可能跨 shard, 按 path 分桶."""
        if not self._buffer:
            return
        events = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()

        # Build hash-chained records + 按目标分片 path 分桶. hash chain 是
        # writer 全局序, 跨分片也连续 — 验证 chain 时按 (archive 旧→新, 当前)
        # 拼回即可.
        buckets: dict[Path, list[str]] = {}
        prev = self._prev_hash
        for ev in events:
            iter_n = (ev.get("data") or {}).get("iteration")
            try:
                iter_n = int(iter_n) if iter_n is not None else None
            except (TypeError, ValueError):
                logger.debug("best-effort op failed", exc_info=True)
                iter_n = None
            target = self._maybe_rotate_shard(iter_n)
            record = dict(ev)
            # P0-2: 脱敏在 hash 之前, 保证 hash 是脱敏后的
            record = _sanitize_record(record)
            record["_prev_hash"] = prev
            record["_hash"] = _compute_hash(record, prev)
            prev = record["_hash"]
            buckets.setdefault(target, []).append(
                json.dumps(record, ensure_ascii=False, default=str)
            )
        self._prev_hash = prev

        for path, lines in buckets.items():
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.writelines(line + "\n" for line in lines)
            except Exception:
                logger.debug(
                    "audit log flush failed (%d lines lost to %s)",
                    len(lines), path, exc_info=True,
                )

    def _flush_loop(self) -> None:
        while not self._stop.wait(timeout=_FLUSH_INTERVAL):
            # 定时 flush, 即使没攒够 batch 也落盘
            if time.monotonic() - self._last_flush >= _FLUSH_INTERVAL:
                self.flush()

    def stop(self) -> None:
        self._stop.set()
        self.flush()
        self._thread.join(timeout=1.0)


# 全局 writer 实例 (install 时创建, uninstall 时停止)
_writer: _BufferedAuditWriter | None = None


def _make_subscriber(path: Path):
    """Create a subscriber closure bound to a buffered writer."""
    global _writer
    _writer = _BufferedAuditWriter(path)

    def _on_event(event: AgentEvent) -> None:
        _writer.append(event)

    return _on_event


def install_audit_subscriber(
    bus: EventBus | None = None,
    path: Path | None = None,
) -> Any:
    """Attach the audit log subscriber to the event bus.

    Call this once at startup. Returns an unsubscribe function — calling
    it stops the audit log from receiving further events.

    Args:
        bus: EventBus to subscribe to. Defaults to the shared singleton.
        path: Override for the audit file path. Defaults to
              ~/.huginn/events/audit.jsonl (or $HUGINN_CACHE_DIR/events/).
    """
    global _unsubscribe
    bus = bus or EventBus.shared()
    audit_path = path or _resolve_audit_path()
    subscriber = _make_subscriber(audit_path)
    _unsubscribe = bus.subscribe(ALL, subscriber)
    logger.info("audit subscriber installed: %s", audit_path)
    return _unsubscribe


def uninstall_audit_subscriber() -> None:
    """Detach the audit subscriber and flush remaining buffer."""
    global _unsubscribe, _writer
    if _unsubscribe is not None:
        _unsubscribe()
        _unsubscribe = None
    if _writer is not None:
        _writer.stop()
        _writer = None


def set_audit_task_id(task_id: str) -> None:
    """RCB 任务启动时调, 让全局 audit writer 按 task_id 分片写.

    No-op if audit subscriber 未 install — 调用方不用判 None.
    """
    if _writer is not None:
        _writer.set_task_id(task_id)


def audit_iter_range(start_iter: int, end_iter: int) -> list[dict]:
    """跨分片读 audit log 的 [start_iter, end_iter] 记录. RCB resume 用.

    返回空 list if writer 未 install 或 task_id 未设.
    """
    if _writer is None:
        return []
    return _writer.iter_range(start_iter, end_iter)


# ── campaign/quality 业务订阅者 ─────────────────────────────────
# 之前 campaign.* / quality.check 事件只落 audit.jsonl, 无业务消费方 (emit→log→dead).
# 这里挂一个轻量订阅器: 把 iteration/hypothesis/retry/suspect/refine/quality.check
# 事件提到 logger.info + 可选 telemetry, 让用户/telemetry 能感知研究循环的实时进度.
# ponytail: 只 log, 不做状态更新. 升级: 接 progress 面板 / WebSocket push.
_CAMPAIGN_UNSUBSCRIBES: list[Any] = []


def install_campaign_subscriber(bus: EventBus | None = None) -> None:
    """订阅 campaign.* / quality.check 事件, 转 logger.info + telemetry.

    之前这些事件 emit 后只落 audit.jsonl, 业务侧无感知. 这里挂一个轻量订阅器,
    让 research loop 的关键节点 (iteration/hypothesis/retry/suspect/refine/quality)
    至少在 log 和 telemetry 里可见.
    ponytail: 只 log + Counter, 不做 UI push. 升级: WebSocket / progress 面板.
    """
    global _CAMPAIGN_UNSUBSCRIBES
    bus = bus or EventBus.shared()
    events = (
        "campaign.iteration", "campaign.hypothesis",
        "campaign.retry", "campaign.suspect",
        "campaign.refine", "quality.check",
    )

    def _on_campaign(event: AgentEvent) -> None:
        with contextlib.suppress(Exception):
            logger.info(
                "campaign event: %s iter=%s payload_keys=%s",
                event.type,
                event.data.get("iteration", "?"),
                list(event.data.keys()),
            )

    for evt_type in events:
        try:
            unsub = bus.subscribe(evt_type, _on_campaign)
            _CAMPAIGN_UNSUBSCRIBES.append(unsub)
        except Exception:
            logger.debug("campaign subscribe failed for %s", evt_type, exc_info=True)


def uninstall_campaign_subscriber() -> None:
    """Detach all campaign/quality subscribers."""
    global _CAMPAIGN_UNSUBSCRIBES
    for unsub in _CAMPAIGN_UNSUBSCRIBES:
        with contextlib.suppress(Exception):
            unsub()
    _CAMPAIGN_UNSUBSCRIBES = []


def verify_audit_chain(path: Path | None = None) -> bool:
    """Verify the integrity of the audit log hash chain.

    Returns True if every record's _hash matches the recomputed value
    and the chain is unbroken. Returns True for an empty/missing file.
    Logs the first broken record if verification fails.
    """
    audit_path = path or _resolve_audit_path()
    if not audit_path.exists():
        return True
    prev_hash = "0" * 64
    try:
        with open(audit_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("audit chain broken at line %d: invalid JSON", lineno)
                    return False
                stored_hash = rec.get("_hash", "")
                stored_prev = rec.get("_prev_hash", "")
                if stored_prev != prev_hash:
                    logger.warning(
                        "audit chain broken at line %d: prev_hash mismatch "
                        "(expected %s, got %s)",
                        lineno, prev_hash[:16], stored_prev[:16],
                    )
                    return False
                recomputed = _compute_hash(rec, prev_hash)
                if recomputed != stored_hash:
                    logger.warning(
                        "audit chain broken at line %d: hash mismatch "
                        "(expected %s, got %s)",
                        lineno, recomputed[:16], stored_hash[:16],
                    )
                    return False
                prev_hash = stored_hash
        return True
    except Exception:
        logger.debug("verify_audit_chain failed", exc_info=True)
        return False


# ── self-check ────────────────────────────────────────────────────
# 跑法: python -m huginn.events.audit_log
# 用 1000 条数据模拟 700 万 call trace (shard_interval=100), 验证:
#   1) 1000 条 → 10 个分片 (9 个 .gz 归档 + 1 个 .jsonl 当前)
#   2) 老分片被 gzip 压缩归档
#   3) 单文件 < 100MB (实际 100 条/文件远小于此, 但 assert 兜住磁盘爆掉)
#   4) shard_iter_range 跨分片读 50..150 → 101 条
#   5) _BufferedAuditWriter 跨分片 hash chain 连续
def _selfcheck() -> None:
    import tempfile

    from huginn.events.event_bus import AgentEvent

    print("Running audit_log sharding self-check...")

    # ── Test 1: _ShardState + write_sharded_jsonl 直接验证 ───────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.environ["HUGINN_TRACE_SHARD_INTERVAL"] = "100"
        try:
            state = _ShardState(
                base_dir=tmp_path,
                task_id="selfcheck_task",
                shard_interval=100,
                filename_prefix="audit",
            )
            # 1000 条 iteration 0..999, shard_interval=100 → 10 个分片
            for i in range(1000):
                rec = {"iteration": i, "data": {"iteration": i, "msg": "x" * 50}}
                write_sharded_jsonl(state, rec, i)

            arch_dir = tmp_path / "archive" / "selfcheck_task"
            gz_files = sorted(arch_dir.glob("audit_*_*.jsonl.gz"))
            assert len(gz_files) == 9, (
                f"expected 9 archived shards (0..899), got {len(gz_files)}: "
                f"{[p.name for p in gz_files]}"
            )
            # 当前分片应是 audit_900.jsonl
            cur_path = tmp_path / "selfcheck_task" / "audit_900.jsonl"
            assert cur_path.exists(), f"current shard missing: {cur_path}"
            assert state.current_shard_start == 900, (
                f"current_shard_start should be 900, got {state.current_shard_start}"
            )
            assert state.current_shard_path == cur_path

            # 单文件 < 100MB (实际几十 KB, 兜底磁盘爆掉)
            for p in gz_files + [cur_path]:
                size = p.stat().st_size
                assert size < 100 * 1024 * 1024, f"{p.name} too big: {size} bytes"

            # 归档分片可读 + 行数对 (每分片 100 条)
            with gzip.open(gz_files[0], "rt", encoding="utf-8") as f:
                lines = [_l for _l in f if _l.strip()]
            assert len(lines) == 100, (
                f"first shard should have 100 lines, got {len(lines)}"
            )

            # 跨分片读 iter 50..150 → 101 条 (跨 0_99 和 100_199 两个分片)
            records = shard_iter_range(state, 50, 150)
            iters = sorted(r["iteration"] for r in records)
            assert len(records) == 101, (
                f"expected 101 records (50..150 inclusive), got {len(records)}"
            )
            assert iters[0] == 50 and iters[-1] == 150
            print(
                f"  [OK] _ShardState: 9 .gz + 1 .jsonl, iter_range 50..150 "
                f"→ {len(records)} records"
            )
        finally:
            os.environ.pop("HUGINN_TRACE_SHARD_INTERVAL", None)

    # ── Test 2: _BufferedAuditWriter 跨分片 hash chain 连续 ──────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.environ["HUGINN_TRACE_SHARD_INTERVAL"] = "100"
        try:
            audit_path = tmp_path / "audit.jsonl"
            writer = _BufferedAuditWriter(audit_path)
            writer.set_task_id("chain_task")
            # 250 条 campaign.iteration 事件 → 跨 0/100/200 三个分片
            for i in range(250):
                writer.append(AgentEvent(
                    type="campaign.iteration",
                    timestamp=float(i),
                    data={"iteration": i, "tool": "bash"},
                    thread_id="t1",
                    source="selfcheck",
                ))
            writer.flush()
            writer.stop()

            # 把所有分片 (archive + 当前) 拼回, 验证 hash chain
            arch_dir = tmp_path / "archive" / "chain_task"
            all_records: list[dict] = []
            for gz in sorted(arch_dir.glob("audit_*_*.jsonl.gz")):
                with gzip.open(gz, "rt", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_records.append(json.loads(line))
            # 当前分片 audit_200.jsonl
            cur_path = tmp_path / "chain_task" / "audit_200.jsonl"
            assert cur_path.exists(), f"current shard missing: {cur_path}"
            with open(cur_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))

            assert len(all_records) == 250, (
                f"expected 250 records across shards, got {len(all_records)}"
            )
            prev = "0" * 64
            for i, rec in enumerate(all_records):
                assert rec["_prev_hash"] == prev, (
                    f"chain broken at {i}: prev_hash mismatch "
                    f"(expected {prev[:16]}, got {rec['_prev_hash'][:16]})"
                )
                recomputed = _compute_hash(rec, prev)
                assert rec["_hash"] == recomputed, (
                    f"chain broken at {i}: hash mismatch"
                )
                prev = rec["_hash"]
            # 应当有 2 个 .gz (0_99, 100_199) + 1 个 .jsonl (200)
            assert len(list(arch_dir.glob("audit_*_*.jsonl.gz"))) == 2
            print(
                f"  [OK] _BufferedAuditWriter: hash chain across 3 shards "
                f"({len(all_records)} records) verified"
            )
        finally:
            os.environ.pop("HUGINN_TRACE_SHARD_INTERVAL", None)

    # ── Test 3: task_id 未设时退化为单文件 (兼容旧路径) ─────────
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audit_path = tmp_path / "audit.jsonl"
        writer = _BufferedAuditWriter(audit_path)
        # 不调 set_task_id, 所有事件应写到 audit_path 单文件
        for i in range(5):
            writer.append(AgentEvent(
                type="tool.call",
                timestamp=float(i),
                data={"tool": "bash", "iteration": i},
                thread_id="t1",
                source="selfcheck",
            ))
        writer.flush()
        writer.stop()
        assert audit_path.exists(), "default audit.jsonl should be written"
        lines = [_l for _l in audit_path.read_text(encoding="utf-8").splitlines() if _l.strip()]
        assert len(lines) == 5, f"expected 5 lines in default path, got {len(lines)}"
        # 没有 task_id 子目录 / archive
        assert not (tmp_path / "archive").exists(), "archive dir should not exist"
        print("  [OK] backward compat: task_id unset → single audit.jsonl")

    print("\naudit_log sharding self-check passed.")


if __name__ == "__main__":
    _selfcheck()
