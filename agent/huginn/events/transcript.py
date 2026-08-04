"""对话转录存储 —— 订阅 EventBus 把原始事件落 JSONL.

对标 Kimi Code 的 transcript 包. huginn 的 telemetry.py 记录的是性能轨迹
(span 树 + 耗时 + 内存), 不是对话内容; memory/episodic_shard.py 是压缩后
的语义片段, 不是原始转录. benchmark 评审 (PaperBench/RCB) 需要回放 agent
每一步说了什么、调了什么工具、工具返回什么 —— 这层缺了.

设计:
  - 订阅 EventBus 的 TOOL_CALL / TOOL_RESULT / TOOL_ERROR / TOOL_BLOCKED
    / STEP_RETRY 事件, 把 AgentEvent 原样追加到 JSONL 文件
  - 一个 thread_id 一个文件, 避免单文件过大
  - 只追加不修改, 方便 grep / jq 分析
  - 不做加密, 不做压缩 —— 加密由 encryption_at_rest 统一处理, 压缩交给
    文件系统 (ZFS/Btrfs) 或日志轮转

不做的:
  - 不做结构化查询 (用 jq / sqlite 外部处理)
  - 不做 redaction (PII 过滤由 publish 侧负责)
  - 不做跨 thread 聚合 (benchmark 评审按 thread 看就行)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.events.event_bus import AgentEvent, EventBus
from huginn.events.event_types import (
    STEP_RETRY,
    TOOL_BLOCKED,
    TOOL_CALL,
    TOOL_ERROR,
    TOOL_RESULT,
)

logger = logging.getLogger(__name__)

# 订阅的事件类型 —— 只记 agent 行为事件, 不记 session/pipeline 等
_TRANSCRIPT_EVENT_TYPES = frozenset({
    TOOL_CALL, TOOL_RESULT, TOOL_ERROR, TOOL_BLOCKED, STEP_RETRY,
})


class TranscriptStore:
    """把 agent 行为事件按 thread_id 追加到 JSONL 文件.

    Usage::

        store = TranscriptStore.shared()
        store.start()  # 订阅 EventBus

        # ... agent 跑完一个 turn ...

        entries = store.load_transcript(thread_id)
        for e in entries:
            print(e["type"], e["timestamp"], e["data"])
    """

    _singleton: TranscriptStore | None = None
    _singleton_lock = threading.Lock()

    @classmethod
    def shared(cls) -> TranscriptStore:
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = self._resolve_base_dir(base_dir)
        self._lock = threading.RLock()
        self._started = False
        # ponytail: 不维护内存索引, 查询走 load_transcript 读文件. 一个 thread
        # 一个文件, 几千条 JSONL 行 grep/jq 秒查. 升级路径: 换 sqlite FTS5.

    @staticmethod
    def _resolve_base_dir(explicit: str | Path | None) -> Path:
        if explicit is not None:
            p = Path(explicit).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return p
        env = os.environ.get("HUGINN_TRANSCRIPT_DIR")
        if env:
            p = Path(env).expanduser()
            p.mkdir(parents=True, exist_ok=True)
            return p
        # 跟 audit log 同目录, 集中管理
        default = Path.home() / ".huginn" / "transcripts"
        default.mkdir(parents=True, exist_ok=True)
        return default

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """订阅 EventBus. 幂等, 重复调用安全."""
        with self._lock:
            if self._started:
                return
            bus = EventBus.shared()
            for evt_type in _TRANSCRIPT_EVENT_TYPES:
                bus.subscribe(evt_type, self._on_event)
            self._started = True
            logger.info("TranscriptStore started, dir=%s", self._base_dir)

    def stop(self) -> None:
        """取消订阅. 主要给测试用."""
        with self._lock:
            if not self._started:
                return
            # EventBus 没有 unsubscribe, 只能标记停止, 不再写文件
            self._started = False

    # ------------------------------------------------------------------ write

    def _on_event(self, event: AgentEvent) -> None:
        """EventBus 回调. 异常不抛, 不影响主流程."""
        if not self._started:
            return
        try:
            self._append(event)
        except Exception:
            logger.debug("transcript append failed", exc_info=True)

    def _append(self, event: AgentEvent) -> None:
        thread_id = event.thread_id or "_no_thread"
        # 文件名: <thread_id>.jsonl, thread_id 里可能有 / 会跨目录, 替换掉
        safe_id = thread_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        path = self._base_dir / f"{safe_id}.jsonl"

        entry = {
            "timestamp": event.timestamp,
            "type": event.type,
            "thread_id": event.thread_id,
            "source": event.source,
            "data": event.data,
        }
        line = json.dumps(entry, default=str, ensure_ascii=False)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ------------------------------------------------------------------ read

    def transcript_path(self, thread_id: str) -> Path:
        """返回某个 thread 的转录文件路径. 不保证存在."""
        safe_id = thread_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._base_dir / f"{safe_id}.jsonl"

    def load_transcript(self, thread_id: str) -> list[dict[str, Any]]:
        """加载某个 thread 的完整转录. 文件不存在返回空列表."""
        path = self.transcript_path(thread_id)
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 损坏行跳过, 不带挂整个转录
                        logger.debug("skip corrupted line in %s", path)
        return entries

    def list_threads(self) -> list[str]:
        """列出所有有转录文件的 thread_id."""
        threads: list[str] = []
        with self._lock:
            for p in self._base_dir.glob("*.jsonl"):
                threads.append(p.stem)
        return sorted(threads)

    def summary(self, thread_id: str) -> dict[str, Any]:
        """返回某个 thread 的统计摘要, 方便 benchmark 评审."""
        entries = self.load_transcript(thread_id)
        type_counts: dict[str, int] = {}
        for e in entries:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "thread_id": thread_id,
            "total_events": len(entries),
            "type_counts": type_counts,
            "first_event": entries[0]["timestamp"] if entries else None,
            "last_event": entries[-1]["timestamp"] if entries else None,
        }


# -- self-check -----------------------------------------------------------


if __name__ == "__main__":
    import asyncio
    import tempfile
    import time

    async def main():
        with tempfile.TemporaryDirectory() as td:
            store = TranscriptStore(base_dir=td)
            store.start()

            bus = EventBus.shared()

            # 模拟 agent 事件序列
            await bus.publish(AgentEvent(
                type=TOOL_CALL, timestamp=time.time(),
                data={"tool": "vasp_tool", "args": {"system": "Si"}},
                thread_id="t1", source="test",
            ))
            await bus.publish(AgentEvent(
                type=TOOL_RESULT, timestamp=time.time(),
                data={"tool": "vasp_tool", "success": True, "bandgap": 1.1},
                thread_id="t1", source="test",
            ))
            await bus.publish(AgentEvent(
                type=STEP_RETRY, timestamp=time.time(),
                data={"attempt": 2, "max_attempts": 3, "error_type": "RateLimitError"},
                thread_id="t1", source="agent.streaming",
            ))
            await bus.publish(AgentEvent(
                type=TOOL_CALL, timestamp=time.time(),
                data={"tool": "lmp_tool", "args": {}},
                thread_id="t2", source="test",
            ))

            # 给回调一点时间执行 (publish 是 async, 回调可能 future)
            await asyncio.sleep(0.1)

            # 验证 t1 转录
            entries = store.load_transcript("t1")
            assert len(entries) == 3, f"expected 3, got {len(entries)}"
            assert entries[0]["type"] == TOOL_CALL
            assert entries[1]["type"] == TOOL_RESULT
            assert entries[2]["type"] == STEP_RETRY
            assert entries[0]["data"]["tool"] == "vasp_tool"
            assert entries[1]["data"]["bandgap"] == 1.1

            # 验证 t2 转录独立
            entries_t2 = store.load_transcript("t2")
            assert len(entries_t2) == 1
            assert entries_t2[0]["data"]["tool"] == "lmp_tool"

            # 验证 list_threads
            threads = store.list_threads()
            assert set(threads) == {"t1", "t2"}, f"got {threads}"

            # 验证 summary
            s = store.summary("t1")
            assert s["total_events"] == 3
            assert s["type_counts"][TOOL_CALL] == 1
            assert s["type_counts"][TOOL_RESULT] == 1
            assert s["type_counts"][STEP_RETRY] == 1

            # 验证不存在的 thread
            assert store.load_transcript("nonexistent") == []
            assert store.summary("nonexistent")["total_events"] == 0

            store.stop()

        print("TranscriptStore self-checks passed.")

    asyncio.run(main())
