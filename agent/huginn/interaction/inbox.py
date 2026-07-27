"""Inbox — 跨会话人类注意力队列.

Portions derived from OpenWorker (https://github.com/andrewyng/openworker)
MIT License, Copyright (c) 2024 Andrew Ng
Source: coworker/inbox.py:1-368

当一个 session 在跑 Unattended 时, 别的 agent 需要人类响应 (审批/问答/通知)
就进 Inbox. Inbox 是 record of record, 各 surface (desktop UI / IM / mobile)
只是 transport. 同一个 item 从哪回答都安全 — 状态机 pending → resolved 一次,
first-responder-wins, 后到的回答 no-op.

5 KIND:
- approval: 请求允许执行某 tool (含 always 选项)
- question: 问用户一个问题, 可带 quick-reply options
- notification: 单向通知 (不需响应)
- directory: 请求授予某文件夹访问权
- plan: 请求批准一个 plan

用法:
    from huginn.interaction.inbox import get_inbox_store, inbox_approval_fn
    store = get_inbox_store()           # 进程级单例, JSON 持久化
    item = store.add_approval("sess_1", title="Run bash?", body="rm -rf /tmp/x")
    # ... 别的协程/进程从任意 surface 调:
    store.resolve(item.id, "allow")     # "allow" / "deny" / "always"
    # agent loop 端:
    approval_fn = inbox_approval_fn(store, "sess_1")
    action, payload = await approval_fn(code, risk, reason)  # 挂起到 resolved
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

KIND_APPROVAL = "approval"
KIND_QUESTION = "question"
KIND_NOTIFICATION = "notification"
KIND_DIRECTORY = "directory"  # agent 请求授予某文件夹
KIND_PLAN = "plan"  # agent 提交 plan 等批准

STATE_PENDING = "pending"
STATE_RESOLVED = "resolved"

# pending item 出现在哪. INLINE = attended session 直接在 composer 答 (服务端
# parked, 重连重发, 不进跨会话列表). INBOX = session 设成 Unattended, 进跨会话
# 队列. 两种都是同一个 parked/awaitable/resolve-from-anywhere record.
VIS_INLINE = "inline"
VIS_INBOX = "inbox"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def args_preview(arguments: Optional[dict], *, limit: int = 240) -> str:
    """tool call 参数的单行摘要, 给 approval card 用 (显示 path/content 而非只 tool 名)."""
    parts: list[str] = []
    for k, v in (arguments or {}).items():
        s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
        s = " ".join(str(s).split())  # collapse whitespace/newlines
        if len(s) > 80:
            s = s[:79] + "…"
        parts.append(f"{k}: {s}")
    out = " · ".join(parts)
    return out[: limit - 1] + "…" if len(out) > limit else out


@dataclass
class InboxItem:
    id: str
    session_id: str
    kind: str
    title: str
    body: str = ""
    state: str = STATE_PENDING
    resolution: Optional[str] = (
        None  # approval: "allow"/"deny"/"always"; question: answer text
    )
    inbox: str = "default"  # named inbox / delivery binding
    created_at: str = field(default_factory=_now)
    resolved_at: Optional[str] = None
    visibility: str = VIS_INBOX  # inline (attended) vs inbox (unattended)
    # 这个 prompt 阻塞的 tool_call (durable resume: 持久化后 restart 可重建挂起,
    # 继续那个 turn). 让 item 按 (session_id, tool_call_id) 幂等.
    tool_call_id: Optional[str] = None
    # Question 元数据: 可选 quick-reply choices + free-text 逃生口 (致敬
    # Claude Code 的 AskUserQuestion 结构化-but-总可回答 shape).
    options: list[str] = field(default_factory=list)
    allow_text: bool = (
        True  # 有 options 时也接受 typed answer (the "Other" escape)
    )
    multi: bool = False  # 允许选多个 option
    # Kind-specific payload (directory: suggested path/writable; plan: plan text; …).
    data: dict[str, Any] = field(default_factory=dict)


class InboxStore:
    """Inbox 持久化 + 状态机. 线程安全, 跨协程共享.

    JSON 文件持久化让多进程/重启后仍能读回 pending items. _waiters 是
    asyncio.Event, 让 agent loop 挂起等响应, 任意 surface 调 resolve 唤醒.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: dict[str, InboxItem] = {}
        self._waiters: dict[str, asyncio.Event] = {}
        self._load()

    # -- persistence ------------------------------------------------------------
    def _load(self) -> None:
        if self.path and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for raw in data.get("items", []):
                    item = InboxItem(**raw)
                    self._items[item.id] = item
            except (json.JSONDecodeError, TypeError):
                # 损坏的 inbox.json 不阻塞启动, 空表起步
                pass

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"items": [asdict(i) for i in self._items.values()]}, indent=2),
            encoding="utf-8",
        )

    # -- adding -----------------------------------------------------------------
    def add(
        self,
        session_id: str,
        kind: str,
        title: str,
        *,
        body: str = "",
        inbox: str = "default",
        visibility: str = VIS_INBOX,
        data: Optional[dict[str, Any]] = None,
        options=None,
        allow_text: bool = True,
        multi: bool = False,
        tool_call_id: Optional[str] = None,
    ) -> InboxItem:
        # 按 (session_id, tool_call_id) 幂等: durable resume 重提同 prompt 时
        # 复用已存在 (可能已 resolved) 的 item, 不重提.
        if tool_call_id:
            existing = self.for_tool_call(session_id, tool_call_id)
            if existing is not None:
                return existing
        item = InboxItem(
            id=uuid.uuid4().hex,
            session_id=session_id,
            kind=kind,
            title=title,
            body=body,
            inbox=inbox,
            visibility=visibility,
            data=dict(data or {}),
            options=list(options or []),
            allow_text=bool(allow_text),
            multi=bool(multi),
            tool_call_id=tool_call_id,
        )
        with self._lock:
            self._items[item.id] = item
            self._save()
        return item

    def for_tool_call(self, session_id: str, tool_call_id: str) -> Optional[InboxItem]:
        for i in self._items.values():
            if i.session_id == session_id and i.tool_call_id == tool_call_id:
                return i
        return None

    def add_approval(
        self,
        session_id: str,
        title: str,
        *,
        body: str = "",
        inbox: str = "default",
        visibility: str = VIS_INBOX,
        data: Optional[dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
    ) -> InboxItem:
        # data 携带 automation-run context 给 standing scoped approvals:
        # {task_id, task_title, standing_target?} — UI 卡片 "Allow every time" 的门控
        return self.add(
            session_id,
            KIND_APPROVAL,
            title,
            body=body,
            inbox=inbox,
            visibility=visibility,
            data=data,
            tool_call_id=tool_call_id,
        )

    def add_question(
        self,
        session_id: str,
        title: str,
        *,
        body: str = "",
        inbox: str = "default",
        visibility: str = VIS_INBOX,
        options: Optional[list[str]] = None,
        allow_text: bool = True,
        multi: bool = False,
        tool_call_id: Optional[str] = None,
    ) -> InboxItem:
        return self.add(
            session_id,
            KIND_QUESTION,
            title,
            body=body,
            inbox=inbox,
            visibility=visibility,
            options=options,
            allow_text=allow_text,
            multi=multi,
            tool_call_id=tool_call_id,
        )

    def add_directory(
        self,
        session_id: str,
        title: str,
        *,
        body: str = "",
        inbox: str = "default",
        visibility: str = VIS_INBOX,
        data: Optional[dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
    ) -> InboxItem:
        return self.add(
            session_id,
            KIND_DIRECTORY,
            title,
            body=body,
            inbox=inbox,
            visibility=visibility,
            data=data,
            tool_call_id=tool_call_id,
        )

    def add_plan(
        self,
        session_id: str,
        title: str,
        *,
        body: str = "",
        inbox: str = "default",
        visibility: str = VIS_INBOX,
        data: Optional[dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
    ) -> InboxItem:
        return self.add(
            session_id,
            KIND_PLAN,
            title,
            body=body,
            inbox=inbox,
            visibility=visibility,
            data=data,
            tool_call_id=tool_call_id,
        )

    def add_notification(
        self, session_id: str, title: str, *, body: str = "",
        inbox: str = "default", visibility: str = VIS_INBOX,
    ) -> InboxItem:
        return self.add(
            session_id,
            KIND_NOTIFICATION,
            title,
            body=body,
            inbox=inbox,
            visibility=visibility,
        )

    # -- queries ----------------------------------------------------------------
    def get(self, item_id: str) -> Optional[InboxItem]:
        return self._items.get(item_id)

    def list(
        self,
        *,
        session_id: Optional[str] = None,
        state: Optional[str] = None,
        inbox: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> list[InboxItem]:
        out = list(self._items.values())
        if session_id is not None:
            out = [i for i in out if i.session_id == session_id]
        if state is not None:
            out = [i for i in out if i.state == state]
        if inbox is not None:
            out = [i for i in out if i.inbox == inbox]
        if visibility is not None:
            out = [i for i in out if i.visibility == visibility]
        return sorted(out, key=lambda i: i.created_at)

    def pending(self, session_id: Optional[str] = None) -> list[InboxItem]:
        return self.list(session_id=session_id, state=STATE_PENDING)

    # -- the state machine ------------------------------------------------------
    def resolve(self, item_id: str, resolution: str) -> bool:
        """Resolve an item exactly once. First responder wins; later attempts are
        no-ops (return False). Fires any awaiting agent (the suspended approver)."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.state == STATE_RESOLVED:
                return False
            item.state = STATE_RESOLVED
            item.resolution = resolution
            item.resolved_at = _now()
            self._save()
        waiter = self._waiters.get(item_id)
        if waiter is not None:
            waiter.set()
        return True

    def resolve_session(
        self, session_id: str, resolution: str = "session deleted",
    ) -> int:
        """Resolve every still-pending item of a session (session 被删时调 —
        孤立的 approval/question 永远答不上. 释放 waiter, 返回关掉了几个)."""
        closed = 0
        for item in self.pending(session_id):
            if self.resolve(item.id, resolution):
                closed += 1
        return closed

    async def wait(self, item_id: str) -> str:
        """Await an item's resolution; returns the resolution string. Approver
        用这个挂起 agent 直到人类从任意 surface 回答."""
        item = self._items.get(item_id)
        if item is not None and item.state == STATE_RESOLVED:
            return item.resolution or ""
        ev = self._waiters.setdefault(item_id, asyncio.Event())
        await ev.wait()
        resolved = self._items.get(item_id)
        return (resolved.resolution if resolved else "") or ""

    # -- resume reconciliation --------------------------------------------------
    def reconcile_on_resume(self, session_id: str) -> dict:
        """用户 resume attended 控制时, 把该 session 还 pending 的 inline 显示
        (统一一处答) + 离开期间已答的 recap. 单一真相源: 每个 item 已有一个
        authoritative resolution."""
        pending = self.pending(session_id)
        recap = [i for i in self.list(session_id=session_id, state=STATE_RESOLVED)]
        return {
            "pending": [asdict(i) for i in pending],
            "recap": [asdict(i) for i in recap],
        }


# ── approver 路由 (适配 huginn ApprovalFn) ──────────────────────
#
# OpenWorker 的 inbox_approver 返回 ApprovalOutcome enum; huginn 的 ApprovalFn
# 返回 (action: str, payload: str | None) tuple. 这里做映射:
#   resolution "allow"  → ("approve", None)
#   resolution "always" → ("approve_always", None)
#   resolution "deny"   → ("deny", reason)
#   其他/超时            → ("deny", "no resolution")
#
# ponytail: 不引入 ApprovalOutcome enum, 直接用字符串 tuple 跟现有 code_act_loop
# 兼容. 升级路径: 加 InboxApproval dataclass 携带更多上下文 (task_id/standing).
ApprovalFn = Callable[[str, str, str], Awaitable[tuple[str, Optional[str]]]]


def inbox_approval_fn(
    store: InboxStore,
    session_id: str,
    *,
    inbox: str = "default",
    tool_call_id: Optional[str] = None,
) -> ApprovalFn:
    """造一个 ApprovalFn, 把 permission request 路由到 Inbox 并挂起到 resolved.

    给 code_act_loop 的 approval_fn 用: agent 调 risky tool 前, 这个函数把
    请求变成 Inbox item, 任意 surface 答复后 resolve, agent 继续.
    """

    async def approve(code: str, risk: str, reason: str) -> tuple[str, Optional[str]]:
        title = f"Run code (risk={risk})?"
        body = reason or args_preview({"code": code})
        item = store.add_approval(
            session_id,
            title,
            body=body,
            inbox=inbox,
            tool_call_id=tool_call_id,
            data={"risk": risk, "code_preview": code[:500]},
        )
        resolution = await store.wait(item.id)
        if resolution == "always":
            return ("approve_always", None)
        if resolution == "allow":
            return ("approve", None)
        if resolution == "deny":
            return ("deny", resolution or "user denied")
        return ("deny", f"unexpected resolution: {resolution}")

    return approve


# ── 进程级单例 + 路径解析 ──────────────────────────────────────
_singleton: Optional[InboxStore] = None
_singleton_lock = threading.Lock()


def _default_inbox_path() -> Path:
    """默认 inbox.json 路径: HUGINN_CACHE_DIR 或 ~/.huginn 下."""
    try:
        from huginn.utils.runtime import get_runtime_home
        base = get_runtime_home()
    except Exception:
        base = Path.home() / ".huginn"
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    return base / "inbox.json"


def get_inbox_store(path: Optional[str | Path] = None) -> InboxStore:
    """拿进程级 InboxStore 单例. 第一次调时初始化 + load 持久化文件.

    多 worker 部署时每个 worker 各自一份 (JSON 文件共享但 _waiters 不跨进程).
    跨进程唤醒需要外部信号 (EventBus/Redis pub-sub), 这里不引入.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                resolved_path = Path(path) if path else _default_inbox_path()
                _singleton = InboxStore(resolved_path)
    return _singleton


def reset_inbox_store() -> None:
    """测试用: 清掉单例. 下次 get_inbox_store 重新初始化."""
    global _singleton
    with _singleton_lock:
        _singleton = None


if __name__ == "__main__":
    import shutil
    import tempfile as _tf
    import asyncio as _asyncio

    ws = Path(_tf.mkdtemp(prefix="huginn_inbox_test_"))
    try:
        store = InboxStore(ws / "inbox.json")

        # 1. add → get → pending
        item = store.add_approval("s1", title="Run bash?", body="rm -rf /tmp/x")
        assert item.kind == KIND_APPROVAL
        assert item.state == STATE_PENDING
        assert store.get(item.id) is item
        pending = store.pending("s1")
        assert len(pending) == 1
        assert pending[0].id == item.id
        print("1. add/get/pending OK")

        # 2. resolve: first-responder-wins, 第二次 no-op
        assert store.resolve(item.id, "allow") is True
        assert store.resolve(item.id, "deny") is False  # 已 resolved
        assert store.get(item.id).state == STATE_RESOLVED
        assert store.get(item.id).resolution == "allow"
        assert store.get(item.id).resolved_at is not None
        assert store.pending("s1") == []
        print("2. resolve first-responder-wins OK")

        # 3. idempotent by (session_id, tool_call_id)
        item2 = store.add_approval(
            "s2", title="Run calc?", tool_call_id="tc_100",
        )
        item2_dup = store.add_approval(
            "s2", title="Run calc? (dup)", tool_call_id="tc_100",
        )
        assert item2.id == item2_dup.id, "同 tool_call_id 应复用 item"
        assert item2_dup.title == "Run calc?", "应保留原 item 不覆盖"
        # 不同 tool_call_id 不互染
        item3 = store.add_approval(
            "s2", title="Other", tool_call_id="tc_200",
        )
        assert item3.id != item2.id
        print("3. idempotent by (session_id, tool_call_id) OK")

        # 4. 5 KIND 都能加 + list 过滤
        store.add_question("s3", "Q1?", options=["A", "B"])
        store.add_notification("s3", "N1")
        store.add_directory("s3", "Grant /tmp?")
        store.add_plan("s3", "Plan X")
        all_s3 = store.list(session_id="s3")
        kinds = {i.kind for i in all_s3}
        assert kinds == {KIND_QUESTION, KIND_NOTIFICATION, KIND_DIRECTORY, KIND_PLAN}
        # state 过滤
        assert len(store.list(session_id="s3", state=STATE_PENDING)) == 4
        print("4. 5 KIND + list filter OK")

        # 5. resolve_session 关掉所有 pending
        closed = store.resolve_session("s3", "session deleted")
        assert closed == 4
        assert store.pending("s3") == []
        print("5. resolve_session OK")

        # 6. wait 挂起到 resolve (异步验证)
        async def _wait_test():
            item = store.add_approval("s_wait", title="wait test")
            async def _resolver():
                await _asyncio.sleep(0.05)
                assert store.resolve(item.id, "always") is True
            _asyncio.ensure_future(_resolver())
            res = await store.wait(item.id)
            assert res == "always"
        _asyncio.run(_wait_test())
        print("6. wait → resolve 唤醒 OK")

        # 7. 已 resolved 的 item.wait 立即返回 (不挂起)
        async def _resolved_wait():
            item = store.add_approval("s_done", title="done")
            store.resolve(item.id, "deny")
            res = await store.wait(item.id)
            assert res == "deny"
        _asyncio.run(_resolved_wait())
        print("7. wait on resolved → immediate OK")

        # 8. JSON 持久化: 新 store 加载同一文件应看到 items
        store2 = InboxStore(ws / "inbox.json")
        all_items = store2.list()
        assert len(all_items) > 0, "新 store 应从 JSON load items"
        # 找到最初加的 s1 item
        s1_items = [i for i in all_items if i.session_id == "s1"]
        assert len(s1_items) == 1
        assert s1_items[0].state == STATE_RESOLVED
        assert s1_items[0].resolution == "allow"
        print("8. JSON persistence round-trip OK")

        # 9. inbox_approval_fn 适配 huginn ApprovalFn 签名
        async def _approver_test():
            fn = inbox_approval_fn(store, "s_fn", tool_call_id="tc_fn")
            item_holder = {}
            async def _resolve_after_add():
                # 等 item 进 store
                for _ in range(50):
                    p = store.pending("s_fn")
                    if p:
                        item_holder["item"] = p[0]
                        break
                    await _asyncio.sleep(0.01)
                assert item_holder.get("item"), "item should be added"
                store.resolve(item_holder["item"].id, "always")
            _asyncio.ensure_future(_resolve_after_add())
            action, payload = await fn("print('hi')", "medium", "test reason")
            assert action == "approve_always", f"expected approve_always, got {action}"
            assert payload is None
        _asyncio.run(_approver_test())
        print("9. inbox_approval_fn → (approve_always, None) OK")

        # 10. inbox_approval_fn deny 路径
        async def _deny_test():
            fn = inbox_approval_fn(store, "s_deny", tool_call_id="tc_deny")
            async def _resolve_deny():
                for _ in range(50):
                    p = store.pending("s_deny")
                    if p:
                        store.resolve(p[0].id, "deny")
                        break
                    await _asyncio.sleep(0.01)
            _asyncio.ensure_future(_resolve_deny())
            action, payload = await fn("os.system('rm -rf /')", "high", "destructive")
            assert action == "deny", f"expected deny, got {action}"
        _asyncio.run(_deny_test())
        print("10. inbox_approval_fn → deny OK")

        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(ws, ignore_errors=True)
