"""侧边任务 — 从主对话的某条消息 fork 出并行支线, 用完即弃或转正.

解决的问题 (对照 Codex / Antigravity / AgentScope 的侧边任务形态):

  - **不能 fork**: 主任务跑到一半产生次要需求时, 只能打断主线或新开会话
    重新交代背景. 这里允许从任意一条消息切出一个独立 thread, 继承该点之前
    的全部 user/assistant 上下文, 主线程继续跑.
  - **不能转正**: 侧边聊着聊着变成正式任务时, 没有任务卡片 / 历史 / 产物.
    这里 ``promote()`` 把侧边任务原地转正 (thread_id 不变, 历史与产物自然
    延续), 生成 task_card 供任务面板展示.

设计边界:
  - **不新建对话系统**: 侧边任务就是一个普通 thread (id = ``side_<hex>``),
    复用现有 checkpointer / WS / REST 链路; 本模块只管元数据与 fork 种子.
  - **fork 种子只注入一次**: 第一次 ``agent.chat()`` 时注入 ConversationTree,
    之后由 checkpointer 接管, 避免每轮重复放大上下文.
  - **假的不要**: 父线程不存在 / 索引越界 / 无模型时如实返回失败, 不返回
    "已继承空上下文" 这类看似成功的结果.
  - **零残留**: 侧边任务超过 TTL (默认 1h) 无活动自动清理, 连同 thread 注册
    与 checkpointer 行一起删; 转正后的任务不参与 TTL.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.permissions import WRITE_EXEC_TOOLS
from huginn.utils.common import atomic_write_json, now_iso
from huginn.utils.runtime import get_runtime_home

STATUS_SIDE = "side"
STATUS_PROMOTED = "promoted"

_SEED_ROLES = frozenset({"user", "assistant"})
_DEFAULT_TTL_S = 3600.0


def _ttl_seconds() -> float:
    try:
        return float(os.environ.get("HUGINN_SIDE_TASK_TTL_S", _DEFAULT_TTL_S))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_S


def _default_store_path() -> Path:
    raw = os.environ.get("HUGINN_SIDE_TASKS_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return get_runtime_home() / "side_tasks.json"


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user" and str(msg.get("content", "")).strip():
            return str(msg["content"]).strip()
    return ""


@dataclass
class SideTask:
    """一条侧边任务. ``id`` 同时是 thread_id — 转正时原地升级, 历史不搬家."""

    id: str
    title: str
    parent_thread_id: str = ""
    fork_message_index: int | None = None
    seed_messages: list[dict[str, Any]] = field(default_factory=list)
    seed_consumed: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    status: str = STATUS_SIDE
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    promoted_at: str | None = None
    task_card: dict[str, Any] | None = None

    @property
    def thread_id(self) -> str:
        return self.id

    @property
    def is_promoted(self) -> bool:
        return self.status == STATUS_PROMOTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "title": self.title,
            "parent_thread_id": self.parent_thread_id,
            "fork_message_index": self.fork_message_index,
            "seed_count": len(self.seed_messages),
            "seed_consumed": self.seed_consumed,
            "message_count": len(self.messages),
            "artifacts": list(self.artifacts),
            "status": self.status,
            "is_promoted": self.is_promoted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "promoted_at": self.promoted_at,
            "task_card": dict(self.task_card) if self.task_card else None,
        }

    def detail_dict(self) -> dict[str, Any]:
        """单任务详情: 列表接口的字段 + fork 种子 + 侧边对话历史."""
        data = self.to_dict()
        data["seed_messages"] = [dict(m) for m in self.seed_messages]
        data["messages"] = [dict(m) for m in self.messages]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SideTask:
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            parent_thread_id=data.get("parent_thread_id", ""),
            fork_message_index=data.get("fork_message_index"),
            seed_messages=list(data.get("seed_messages") or []),
            seed_consumed=bool(data.get("seed_consumed", False)),
            messages=list(data.get("messages") or []),
            artifacts=list(data.get("artifacts") or []),
            status=data.get("status", STATUS_SIDE),
            created_at=data.get("created_at", now_iso()),
            updated_at=data.get("updated_at", now_iso()),
            promoted_at=data.get("promoted_at"),
            task_card=dict(data["task_card"]) if data.get("task_card") else None,
        )


class SideTaskStore:
    """进程内 + 磁盘持久化的侧边任务表. 线程安全, 原子写."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path).expanduser() if path else _default_store_path()
        self._lock = threading.RLock()
        self._tasks: dict[str, SideTask] = {}
        self._loaded = False

    # ── persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        try:
            import json

            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        for item in data.get("tasks", []):
            try:
                task = SideTask.from_dict(item)
            except (KeyError, TypeError):
                continue
            self._tasks[task.id] = task

    def _save(self) -> None:
        atomic_write_json(
            self._path,
            {"tasks": [t.to_dict() | {
                "seed_messages": t.seed_messages,
                "messages": t.messages,
            } for t in self._tasks.values()]},
            indent=2,
        )

    def _touch(self, task: SideTask) -> None:
        task.updated_at = now_iso()

    # ── CRUD ──────────────────────────────────────────────────────

    def create(
        self,
        *,
        parent_thread_id: str,
        messages: list[dict[str, Any]],
        from_message_index: int | None = None,
        title: str = "",
        inherit_context: bool = True,
    ) -> SideTask:
        """从父线程消息的某个切点 fork 出侧边任务.

        ``from_message_index`` 是在 **user/assistant 过滤后** 的列表下标
        (含该条). None = 继承全部. 越界抛 ValueError — 不静默截断.
        """
        if not str(parent_thread_id).strip():
            raise ValueError("parent_thread_id is required")

        usable = [
            {"role": m["role"], "content": str(m.get("content", ""))}
            for m in messages
            if m.get("role") in _SEED_ROLES and str(m.get("content", "")).strip()
        ]
        if not inherit_context:
            seed: list[dict[str, Any]] = []
        elif from_message_index is None:
            seed = usable
        else:
            try:
                idx = int(from_message_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("from_message_index must be an integer") from exc
            if idx < 0 or idx >= len(usable):
                raise ValueError(
                    f"from_message_index {idx} out of range "
                    f"(0..{len(usable) - 1})"
                )
            seed = usable[: idx + 1]

        resolved_title = (title or "").strip() or _first_user_text(seed)
        if not resolved_title:
            resolved_title = "Side task"

        task = SideTask(
            id=f"side_{uuid.uuid4().hex[:8]}",
            title=resolved_title[:80],
            parent_thread_id=str(parent_thread_id).strip(),
            fork_message_index=from_message_index,
            seed_messages=seed,
        )
        with self._lock:
            self._load()
            self._tasks[task.id] = task
            self._save()
        return task

    def get(self, task_id: str) -> SideTask | None:
        with self._lock:
            self._load()
            return self._tasks.get(task_id)

    def list_tasks(self, status: str | None = None) -> list[SideTask]:
        with self._lock:
            self._load()
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.updated_at, reverse=True)

    def discard(self, task_id: str) -> SideTask | None:
        with self._lock:
            self._load()
            task = self._tasks.pop(task_id, None)
            if task is not None:
                self._save()
            return task

    # ── fork seed ─────────────────────────────────────────────────

    def peek_seed(self, thread_id: str) -> list[dict[str, Any]]:
        """取该 side thread 尚未注入的 fork 种子; 空列表表示无需注入."""
        with self._lock:
            self._load()
            task = self._tasks.get(thread_id)
            if task is None or task.status != STATUS_SIDE or task.seed_consumed:
                return []
            return [dict(m) for m in task.seed_messages]

    def consume_seed(self, thread_id: str) -> bool:
        with self._lock:
            self._load()
            task = self._tasks.get(thread_id)
            if task is None or task.seed_consumed or not task.seed_messages:
                return False
            task.seed_consumed = True
            self._touch(task)
            self._save()
            return True

    # ── conversation turns & artifacts ────────────────────────────

    def append_turn(self, task_id: str, role: str, content: str) -> bool:
        if role not in _SEED_ROLES or not str(content).strip():
            return False
        with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task.messages.append(
                {"role": role, "content": str(content), "at": now_iso()}
            )
            self._touch(task)
            self._save()
            return True

    def add_artifacts(self, task_id: str, artifacts: list[dict[str, Any]]) -> int:
        """追加产物, 按 path 去重. 返回新增条数."""
        if not artifacts:
            return 0
        added = 0
        with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is None:
                return 0
            seen = {
                str(a.get("path", "")) for a in task.artifacts if a.get("path")
            }
            for art in artifacts:
                path = str(art.get("path", "")).strip()
                if not path or path in seen:
                    continue
                seen.add(path)
                task.artifacts.append({"path": path, "tool": art.get("tool", "")})
                added += 1
            if added:
                self._touch(task)
                self._save()
            return added

    # ── promote ───────────────────────────────────────────────────

    def promote(self, task_id: str, title: str = "") -> SideTask | None:
        """侧边任务转正: 状态翻到 promoted, 生成任务卡片.

        thread_id 不变 — 历史消息、checkpointer、产物天然延续到正式任务.
        """
        with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is None or task.status != STATUS_SIDE:
                return None
            task.status = STATUS_PROMOTED
            task.promoted_at = now_iso()
            task.title = (title or task.title).strip()[:80] or task.title
            history = task.seed_messages + [
                {"role": m["role"], "content": m["content"]}
                for m in task.messages
            ]
            task.task_card = {
                "task_id": task.thread_id,
                "title": task.title,
                "goal": _first_user_text(history),
                "status": "active",
                "source_thread_id": task.parent_thread_id,
                "fork_message_index": task.fork_message_index,
                "history_count": len(history),
                "artifacts": list(task.artifacts),
                "created_at": task.created_at,
                "promoted_at": task.promoted_at,
            }
            self._touch(task)
            self._save()
            return task

    # ── TTL sweep ─────────────────────────────────────────────────

    def sweep(self, ttl_s: float | None = None, now: datetime | None = None) -> list[SideTask]:
        """清理超时未活动的**侧边**任务; 已转正的任务永不过期."""
        ttl = _ttl_seconds() if ttl_s is None else max(0.0, float(ttl_s))
        now_dt = now or datetime.now().astimezone()
        removed: list[SideTask] = []
        with self._lock:
            self._load()
            for task_id, task in list(self._tasks.items()):
                if task.status != STATUS_SIDE:
                    continue
                updated = _parse_ts(task.updated_at)
                if updated is None:
                    continue
                if updated.tzinfo is None:
                    updated = updated.astimezone()
                if (now_dt - updated).total_seconds() > ttl:
                    removed.append(self._tasks.pop(task_id))
            if removed:
                self._save()
        return removed


# ── shared singleton ──────────────────────────────────────────────

_shared: SideTaskStore | None = None
_shared_lock = threading.Lock()


def get_shared_side_task_store() -> SideTaskStore:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SideTaskStore()
        return _shared


def set_shared_side_task_store(store: SideTaskStore | None) -> None:
    """注入测试 store, 或 None 重置."""
    global _shared
    with _shared_lock:
        _shared = store


# ── zero-residue helpers ──────────────────────────────────────────


def resolve_checkpointer_path() -> Path:
    """与 checkpointer.create_checkpointer 相同的路径解析 (只读, 不建库)."""
    raw = os.environ.get("HUGINN_CHECKPOINTER_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return get_runtime_home() / "checkpoints.sqlite"


def purge_thread_checkpoints(thread_id: str, db_path: str | Path | None = None) -> int:
    """删除某 thread 在 SQLite checkpointer 里的全部行 (checkpoints + writes).

    返回删除的行数; 库不存在 / 表不存在 / 非 SQLite 后端时返回 0. 失败不抛 —
    "用完即弃" 的清理是 best-effort, 不能因为磁盘问题阻断 UI.
    """
    path = Path(db_path).expanduser() if db_path else resolve_checkpointer_path()
    if not path.exists():
        return 0
    deleted = 0
    try:
        conn = sqlite3.connect(str(path), timeout=5)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table in ("writes", "checkpoints"):
                if table not in tables:
                    continue
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
                )
                deleted += max(0, cur.rowcount)
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        return deleted
    return deleted


def extract_artifacts(
    messages: list[Any],
    write_tools: frozenset[str] = WRITE_EXEC_TOOLS,
) -> list[dict[str, Any]]:
    """从 LangChain 消息序列里抽"写类工具"的目标路径作为产物.

    只认 AIMessage.tool_calls 里的显式 path/file_path/output 参数; 读类工具
    (file_read / grep / glob) 不产生产物. 去重按 path.
    """
    found: dict[str, str] = {}
    for msg in messages or []:
        calls = getattr(msg, "tool_calls", None) or []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", ""))
            if name not in write_tools:
                continue
            args = call.get("args") or {}
            if not isinstance(args, dict):
                continue
            for key in ("path", "file_path", "output", "output_path"):
                val = args.get(key)
                if isinstance(val, str) and val.strip():
                    found.setdefault(val.strip(), name)
    return [{"path": path, "tool": tool} for path, tool in found.items()]


__all__ = [
    "STATUS_PROMOTED",
    "STATUS_SIDE",
    "SideTask",
    "SideTaskStore",
    "extract_artifacts",
    "get_shared_side_task_store",
    "purge_thread_checkpoints",
    "resolve_checkpointer_path",
    "set_shared_side_task_store",
]
