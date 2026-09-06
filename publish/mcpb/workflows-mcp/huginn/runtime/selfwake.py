"""SelfWake — 让长时间运行的 agent 挂起 + 按触发器再唤醒.

Portions derived from OpenWorker (https://github.com/andrewyng/openworker)
MIT License, Copyright (c) 2024 Andrew Ng
Source: coworker/selfwake.py:1-185

把 always-on agent 变成 suspend/resume (事件驱动, ~零 idle 成本): session 睡着,
runtime 在 wake 到期时再调它. 三种触发器:
- timer: sleep_for / sleep_until (定时唤醒)
- completion: wake_on(job_id) — 后台 job 完成时唤醒
- event: wake_on_event(event_key) — connector/webhook 事件触发

WakeStore 持久化 wake records + due/complete 逻辑; WakeScheduler tick 消费
due() / complete_job() / fire_event() 并 resume session.

用法:
    from huginn.runtime.selfwake import get_wake_store, selfwake_tools
    store = get_wake_store()
    tools = selfwake_tools(store, session_id="sess_1")  # 注入给 agent
    # scheduler tick:
    for w in store.due():
        await resume_session(w.session_id)
        store.mark_fired(w.id)
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

KIND_TIMER = "timer"
KIND_COMPLETION = "completion"
KIND_EVENT = "event"  # connector/webhook 事件触发 (Phase 3)

STATE_PENDING = "pending"
STATE_DUE = "due"
STATE_FIRED = "fired"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Wake:
    id: str
    session_id: str
    kind: str
    state: str = STATE_PENDING
    fire_at: str | None = None  # ISO, timer wakes 用
    job_id: str | None = None  # completion wakes 用
    event_key: str | None = None  # event wakes 用
    note: str = ""
    created_at: str = field(default_factory=lambda: _now().isoformat())


class WakeStore:
    """Wake records 持久化 + due/complete 逻辑. 线程安全.

    JSON 文件持久化让重启后仍能读回 pending wakes, run-once-catch-up 时触发.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._wakes: dict[str, Wake] = {}
        if self.path and self.path.is_file():
            try:
                for raw in json.loads(self.path.read_text(encoding="utf-8")).get(
                    "wakes", []
                ):
                    w = Wake(**raw)
                    self._wakes[w.id] = w
            except (json.JSONDecodeError, TypeError):
                # 损坏的 wakes.json 不阻塞启动
                pass

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"wakes": [asdict(w) for w in self._wakes.values()]}, indent=2),
            encoding="utf-8",
        )

    def add_timer(self, session_id: str, fire_at: datetime, *, note: str = "") -> Wake:
        w = Wake(
            uuid.uuid4().hex,
            session_id,
            KIND_TIMER,
            fire_at=fire_at.isoformat(),
            note=note,
        )
        with self._lock:
            self._wakes[w.id] = w
            self._save()
        return w

    def add_completion(self, session_id: str, job_id: str, *, note: str = "") -> Wake:
        w = Wake(
            uuid.uuid4().hex, session_id, KIND_COMPLETION, job_id=job_id, note=note
        )
        with self._lock:
            self._wakes[w.id] = w
            self._save()
        return w

    def add_event(self, session_id: str, event_key: str, *, note: str = "") -> Wake:
        w = Wake(
            uuid.uuid4().hex, session_id, KIND_EVENT, event_key=event_key, note=note
        )
        with self._lock:
            self._wakes[w.id] = w
            self._save()
        return w

    def due(self, now: datetime | None = None) -> list[Wake]:
        """Timer wakes 已到 fire_at, 或 completion/event wakes 标记为 due."""
        now = now or _now()
        out = []
        for w in self._wakes.values():
            if w.state not in (STATE_PENDING, STATE_DUE):
                continue
            if (
                w.kind == KIND_TIMER
                and w.fire_at
                and datetime.fromisoformat(w.fire_at) <= now
            ) or w.kind in (KIND_COMPLETION, KIND_EVENT) and w.state == STATE_DUE:
                out.append(w)
        return out

    def complete_job(self, job_id: str) -> list[Wake]:
        """Mark completion wakes for ``job_id`` as due (the job exited)."""
        return self._mark_due(
            lambda w: w.kind == KIND_COMPLETION and w.job_id == job_id
        )

    def fire_event(self, event_key: str) -> list[Wake]:
        """Mark on-event wakes for ``event_key`` as due (connector/webhook fired)."""
        return self._mark_due(
            lambda w: w.kind == KIND_EVENT and w.event_key == event_key
        )

    def _mark_due(self, pred) -> list[Wake]:
        fired = []
        with self._lock:
            for w in self._wakes.values():
                if w.state == STATE_PENDING and pred(w):
                    w.state = STATE_DUE
                    fired.append(w)
            if fired:
                self._save()
        return fired

    def mark_fired(self, wake_id: str) -> None:
        with self._lock:
            w = self._wakes.get(wake_id)
            if w is not None:
                w.state = STATE_FIRED
                self._save()

    def pending(self, session_id: str | None = None) -> list[Wake]:
        return [
            w
            for w in self._wakes.values()
            if w.state != STATE_FIRED
            and (session_id is None or w.session_id == session_id)
        ]

    def get(self, wake_id: str) -> Wake | None:
        return self._wakes.get(wake_id)


def selfwake_tools(store: WakeStore, session_id: str) -> list:
    """4 个工具, agent 调它们来 schedule 自己的 resumption.

    返回 list of plain Python functions (CodeAct 模式直接注入 namespace).
    tool_call 模式需自行包装成 Tool dataclass.
    ponytail: 不引入 Tool dataclass 包装, CodeAct 直接用. 升级路径: 注册到
    ToolRegistry 时加 input_schema + description.
    """

    def sleep_for(seconds: int, note: str = "") -> dict:
        """Suspend and wake this session after `seconds`. 用作 polling/waiting
        而不烧 context."""
        w = store.add_timer(
            session_id, _now() + timedelta(seconds=int(seconds)), note=note
        )
        return {"ok": True, "wake_id": w.id, "fire_at": w.fire_at}

    def sleep_until(when_iso: str, note: str = "") -> dict:
        """Suspend and wake this session at an ISO-8601 timestamp."""
        when = datetime.fromisoformat(when_iso)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        w = store.add_timer(session_id, when, note=note)
        return {"ok": True, "wake_id": w.id, "fire_at": w.fire_at}

    def wake_on(job_id: str, note: str = "") -> dict:
        """Suspend and wake this session when a backgrounded job (`job_id`) completes."""
        w = store.add_completion(session_id, job_id, note=note)
        return {"ok": True, "wake_id": w.id, "job_id": job_id}

    def wake_on_event(event_key: str, note: str = "") -> dict:
        """Suspend and wake this session when a named event (`event_key`) fires —
        e.g. a connector/webhook signal an Ops agent watches for."""
        w = store.add_event(session_id, event_key, note=note)
        return {"ok": True, "wake_id": w.id, "event_key": event_key}

    return [sleep_for, sleep_until, wake_on, wake_on_event]


# ── 进程级单例 + 路径解析 ──────────────────────────────────────
_singleton: WakeStore | None = None
_singleton_lock = threading.Lock()


def _default_wake_path() -> Path:
    """默认 wakes.json 路径: HUGINN_CACHE_DIR 或 ~/.huginn 下."""
    try:
        from huginn.utils.runtime import get_runtime_home
        base = get_runtime_home()
    except Exception:
        base = get_runtime_home()
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    return base / "wakes.json"


def get_wake_store(path: str | Path | None = None) -> WakeStore:
    """进程级 WakeStore 单例."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                resolved_path = Path(path) if path else _default_wake_path()
                _singleton = WakeStore(resolved_path)
    return _singleton


def reset_wake_store() -> None:
    """测试用: 清掉单例."""
    global _singleton
    with _singleton_lock:
        _singleton = None


if __name__ == "__main__":
    import shutil
    import tempfile as _tf

    ws = Path(_tf.mkdtemp(prefix="huginn_selfwake_test_"))
    try:
        store = WakeStore(ws / "wakes.json")

        # 1. add_timer → due 检测
        past = _now() - timedelta(seconds=10)
        w1 = store.add_timer("s1", past, note="past timer")
        assert w1.kind == KIND_TIMER
        assert w1.state == STATE_PENDING
        due = store.due()
        assert len(due) == 1
        assert due[0].id == w1.id
        print("1. add_timer + due (past) OK")

        # 2. future timer 不 due
        future = _now() + timedelta(hours=1)
        w2 = store.add_timer("s2", future)
        assert len(store.due()) == 1, "future timer 不应 due"
        print("2. future timer not due OK")

        # 3. mark_fired 后不再 due
        store.mark_fired(w1.id)
        assert store.get(w1.id).state == STATE_FIRED
        assert store.due() == []
        print("3. mark_fired OK")

        # 4. completion wake: add 时不 due, complete_job 后 due
        w3 = store.add_completion("s3", job_id="job_100")
        assert w3.state == STATE_PENDING
        assert store.due() == [], "completion wake 未 complete 前不应 due"
        fired = store.complete_job("job_100")
        assert len(fired) == 1
        assert fired[0].id == w3.id
        assert store.get(w3.id).state == STATE_DUE
        due = store.due()
        assert len(due) == 1 and due[0].id == w3.id
        print("4. completion wake: add → complete_job → due OK")

        # 5. event wake: fire_event 后 due
        w4 = store.add_event("s4", event_key="webhook_x")
        assert w4.state == STATE_PENDING
        fired = store.fire_event("webhook_x")
        assert len(fired) == 1
        assert store.get(w4.id).state == STATE_DUE
        print("5. event wake: add → fire_event → due OK")

        # 6. complete_job / fire_event 不影响其他 wake
        w5 = store.add_completion("s5", job_id="job_200")
        w6 = store.add_event("s6", event_key="webhook_y")
        # complete job_100 不应影响 w5
        store.complete_job("job_100")
        assert store.get(w5.id).state == STATE_PENDING
        # fire_event webhook_x 不应影响 w6
        store.fire_event("webhook_x")
        assert store.get(w6.id).state == STATE_PENDING
        print("6. cross wake isolation OK")

        # 7. pending 过滤
        pending_all = store.pending()
        assert w1 not in pending_all  # FIRED
        assert w2 in pending_all      # PENDING (future timer)
        assert w3 in pending_all      # DUE (still pending-or-due)
        pending_s2 = store.pending(session_id="s2")
        assert len(pending_s2) == 1 and pending_s2[0].id == w2.id
        print("7. pending filter OK")

        # 8. selfwake_tools 4 个函数
        tools = selfwake_tools(store, "s_tools")
        assert len(tools) == 4
        sleep_for, sleep_until, wake_on, wake_on_event = tools
        # sleep_for 加 timer wake
        result = sleep_for(60, note="test")
        assert result["ok"] is True
        assert "wake_id" in result
        w_timer = store.get(result["wake_id"])
        assert w_timer is not None
        assert w_timer.session_id == "s_tools"
        assert w_timer.kind == KIND_TIMER
        # wake_on 加 completion wake
        result = wake_on("job_test", note="test")
        assert result["ok"] is True
        w_comp = store.get(result["wake_id"])
        assert w_comp.kind == KIND_COMPLETION
        assert w_comp.job_id == "job_test"
        # wake_on_event 加 event wake
        result = wake_on_event("evt_test")
        w_evt = store.get(result["wake_id"])
        assert w_evt.kind == KIND_EVENT
        assert w_evt.event_key == "evt_test"
        # sleep_until 加 timer wake
        result = sleep_until("2026-12-31T23:59:59+00:00")
        w_until = store.get(result["wake_id"])
        assert w_until.kind == KIND_TIMER
        print("8. selfwake_tools 4 functions OK")

        # 9. sleep_until 接 naive datetime (无 tzinfo) → 默认 UTC
        result = sleep_until("2026-12-31T23:59:59")  # 无 +00:00
        w_naive = store.get(result["wake_id"])
        # 内部存的 ISO 应带 +00:00
        assert "+00:00" in w_naive.fire_at
        print("9. sleep_until naive datetime → UTC OK")

        # 10. JSON 持久化 round-trip
        store2 = WakeStore(ws / "wakes.json")
        # 至少看到 w1 (FIRED) 和 w2 (PENDING)
        all_wakes = list(store2._wakes.values())
        assert len(all_wakes) > 0
        w1_loaded = store2.get(w1.id)
        assert w1_loaded is not None
        assert w1_loaded.state == STATE_FIRED
        w2_loaded = store2.get(w2.id)
        assert w2_loaded is not None
        assert w2_loaded.kind == KIND_TIMER
        print("10. JSON persistence round-trip OK")

        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(ws, ignore_errors=True)
