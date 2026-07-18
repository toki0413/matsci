"""Prospective Memory — 第 5 类记忆: 记住"未来要做的事".

与 retrospective memory (记过去) 对称, prospective memory 存的是"我打算
在某个时间/事件/依赖/条件满足时做 X". 每轮 agent loop 开始时调 scan_and_fire,
把到点的 intention 注入 context, 让 agent 知道"该做之前计划的事了".

设计:
- 落 .huginn/prospective.jsonl (append-only, 与 stable_principles 同风格)
- 状态变更 (fired/cancelled) 也追加新行, 读时重放 — 不改历史行
- 4 类触发: time / event / dependency / condition
- condition 用 AST 白名单求值, 不调 eval, 不允许任意表达式
"""
from __future__ import annotations

import ast
import json
import logging
import operator
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProspectiveIntention:
    intention_id: str
    description: str
    trigger_type: str  # "time" / "event" / "dependency" / "condition"
    trigger_payload: dict
    priority: int  # 0-9, 9 最高
    created_at: float  # time.time()
    source_step: int
    status: str = "pending"  # pending / fired / expired / cancelled
    fired_at: float | None = None


def _new_intention_id() -> str:
    """生成唯一 intention_id. uuid4 前 12 位够了, 量大再上全 hex."""
    return f"pim_{uuid.uuid4().hex[:12]}"


# ── 4 类触发判断 ─────────────────────────────────────────────────────────
# ponytail: 4 个函数都是单条件判断, 没必要做成类. 升级路径: 复合条件 (AND/OR)
# 时再加一个 _check_composite, 现在用不上.

def _check_time_trigger(payload: dict, current_state: dict) -> bool:
    """time 触发: trigger_payload['when'] 到点.

    payload['when'] 用 ISO 8601 字符串 (如 '2026-07-19T10:00'). 当前时间
    >= when 即触发. 解析失败返回 False — 宁可漏触发也别误触发.
    """
    when_str = payload.get("when")
    if not when_str:
        return False
    try:
        when = datetime.fromisoformat(when_str)
    except (ValueError, TypeError):
        return False
    return datetime.now() >= when


def _check_event_trigger(payload: dict, current_state: dict) -> bool:
    """event 触发: current_state['events'] 含 payload['event']."""
    events = current_state.get("events", []) or []
    target = payload.get("event")
    if not target:
        return False
    return target in events


def _check_dependency_trigger(payload: dict, current_state: dict) -> bool:
    """dependency 触发: current_step > depends_on_step."""
    depends_on = payload.get("depends_on_step")
    if depends_on is None:
        return False
    current_step = current_state.get("current_step")
    if not isinstance(current_step, int) or not isinstance(depends_on, int):
        return False
    return current_step > depends_on


# AST 白名单: 只允许 `Name op Constant` 形式, op ∈ 6 个比较之一.
# 不调 eval, 不允许 BinOp/Call/Attribute/Subscript, 防注入.
_SAFE_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
}


def _check_condition_trigger(payload: dict, current_state: dict) -> bool:
    """condition 触发: 用 AST 安全求值 payload['expr'].

    只支持 `var op value` 形式 (var 是 current_state['variables'] 的 key,
    op ∈ ==/!=/</>/<=/>=, value 是数字或字符串). 复杂表达式返回 False
    (不触发). 变量不存在也返回 False.
    """
    expr = payload.get("expr")
    if not expr or not isinstance(expr, str):
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    body = tree.body
    if not isinstance(body, ast.Compare):
        return False
    # 只允许单比较 (a < b), 不支持链式 (a < b < c)
    if len(body.ops) != 1 or len(body.comparators) != 1:
        return False
    left = body.left
    right = body.comparators[0]
    if not isinstance(left, ast.Name) or not isinstance(right, ast.Constant):
        return False
    op_type = type(body.ops[0])
    if op_type not in _SAFE_CMP_OPS:
        return False
    variables = current_state.get("variables", {}) or {}
    if left.id not in variables:
        return False
    var_val = variables[left.id]
    cmp_val = right.value
    # 类型不匹配 (如 str vs int) 直接返回 False, 不报错
    try:
        return _SAFE_CMP_OPS[op_type](var_val, cmp_val)
    except TypeError:
        return False


_TRIGGER_CHECKERS = {
    "time": _check_time_trigger,
    "event": _check_event_trigger,
    "dependency": _check_dependency_trigger,
    "condition": _check_condition_trigger,
}


@dataclass
class ProspectiveMemory:
    """Prospective memory store — append-only JSONL, 状态变更也追加.

    文件格式 (每行一个 JSON):
        {"kind": "create", "intention_id": ..., ...full fields...}
        {"kind": "fire",   "intention_id": ..., "fired_at": ..., "ts": ...}
        {"kind": "cancel", "intention_id": ..., "ts": ...}
    读时按 intention_id 重放, 最后一条状态变更生效.
    """
    workspace: Path
    path: Path

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.path = self.workspace / ".huginn" / "prospective.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── 内部: 重放 jsonl 得到当前状态 ──
    def _replay(self) -> dict[str, ProspectiveIntention]:
        """重放 jsonl, 返回 {intention_id: ProspectiveIntention} 当前状态."""
        if not self.path.exists():
            return {}
        state: dict[str, ProspectiveIntention] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # 损坏行跳过, 不让一行坏数据把整个 store 干废
                    continue
                kind = rec.get("kind")
                iid = rec.get("intention_id")
                if not iid:
                    continue
                if kind == "create":
                    state[iid] = ProspectiveIntention(
                        intention_id=iid,
                        description=rec.get("description", ""),
                        trigger_type=rec.get("trigger_type", ""),
                        trigger_payload=rec.get("trigger_payload", {}),
                        priority=int(rec.get("priority", 0)),
                        created_at=float(rec.get("created_at", 0.0)),
                        source_step=int(rec.get("source_step", 0)),
                        status=rec.get("status", "pending"),
                        fired_at=rec.get("fired_at"),
                    )
                elif kind == "fire":
                    if iid in state:
                        state[iid].status = "fired"
                        state[iid].fired_at = float(rec.get("fired_at", time.time()))
                elif kind == "cancel":
                    if iid in state:
                        state[iid].status = "cancelled"
        return state

    def _append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    # ── 公开 API ──
    def store(self, intention: ProspectiveIntention) -> str:
        """追加一条 intention 到 jsonl, 返回 intention_id."""
        self._append({
            "kind": "create",
            "intention_id": intention.intention_id,
            "description": intention.description,
            "trigger_type": intention.trigger_type,
            "trigger_payload": intention.trigger_payload,
            "priority": intention.priority,
            "created_at": intention.created_at,
            "source_step": intention.source_step,
            "status": intention.status,
            "fired_at": intention.fired_at,
            "ts": time.time(),
        })
        return intention.intention_id

    def list_pending(self) -> list[ProspectiveIntention]:
        """列出所有 pending 状态的 intention, 按 priority 降序."""
        state = self._replay()
        pending = [it for it in state.values() if it.status == "pending"]
        pending.sort(key=lambda x: x.priority, reverse=True)
        return pending

    def list_all(self) -> list[ProspectiveIntention]:
        """列出所有 intention (含已 fired/cancelled/expired)."""
        return list(self._replay().values())

    def cancel(self, intention_id: str) -> bool:
        """取消 intention, 追加状态变更记录. 返回是否成功.

        intention 不存在或非 pending 时返回 False.
        """
        state = self._replay()
        it = state.get(intention_id)
        if it is None or it.status != "pending":
            return False
        self._append({
            "kind": "cancel",
            "intention_id": intention_id,
            "ts": time.time(),
        })
        return True

    def mark_fired(self, intention_id: str) -> bool:
        """标记 intention 为 fired, 追加状态变更记录. 返回是否成功.

        intention 不存在或非 pending 时返回 False.
        """
        state = self._replay()
        it = state.get(intention_id)
        if it is None or it.status != "pending":
            return False
        now = time.time()
        self._append({
            "kind": "fire",
            "intention_id": intention_id,
            "fired_at": now,
            "ts": now,
        })
        return True

    def scan_and_fire(self, current_state: dict) -> list[ProspectiveIntention]:
        """每轮开始时调, 扫描 pending intentions, 触发满足条件的, 返回 fired 列表.

        current_state 含: current_step (int), events (list[str]),
        variables (dict[str, float]). 按优先级处理, fired 的立即标记.
        """
        fired: list[ProspectiveIntention] = []
        # list_pending 已按 priority 降序, 高优先级先判断
        for it in self.list_pending():
            checker = _TRIGGER_CHECKERS.get(it.trigger_type)
            if checker is None:
                continue
            try:
                triggered = checker(it.trigger_payload, current_state)
            except Exception:
                logger.debug("trigger check failed for %s", it.intention_id, exc_info=True)
                continue
            if triggered and self.mark_fired(it.intention_id):
                # mark_fired 成功后更新内存里的对象, 让调用方拿到 fired 状态
                it.status = "fired"
                it.fired_at = time.time()
                fired.append(it)
        return fired

    def format_for_context(self, fired: list[ProspectiveIntention]) -> str:
        """格式化 fired intentions 为 context 注入文本.

        格式: "你之前计划了 X（创建于 step N），现在是执行 X 的时候"
        多条用换行分隔.
        """
        if not fired:
            return ""
        lines = []
        for it in fired:
            lines.append(
                f"你之前计划了 {it.description}（创建于 step {it.source_step}），"
                f"现在是执行 {it.description} 的时候"
            )
        return "\n".join(lines)


# ── 自检 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="prospective_test_"))
    try:
        pm = ProspectiveMemory(tmp)

        # 1. store + list_pending 往返一致
        it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="复现 Figure 3 的 bootstrap CI",
            trigger_type="time",
            trigger_payload={"when": "2099-01-01T00:00"},  # 远未来, 不会触发
            priority=5,
            created_at=time.time(),
            source_step=3,
        )
        iid = pm.store(it)
        assert iid == it.intention_id
        pending = pm.list_pending()
        assert len(pending) == 1, f"expected 1 pending, got {len(pending)}"
        assert pending[0].description == "复现 Figure 3 的 bootstrap CI"
        assert pending[0].priority == 5
        assert pending[0].source_step == 3
        assert len(pm.list_all()) == 1

        # 2. time 触发: 过期 time intention, scan_and_fire 触发
        past_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="check old task",
            trigger_type="time",
            trigger_payload={"when": "2020-01-01T00:00"},  # 已过期
            priority=7,
            created_at=time.time(),
            source_step=1,
        )
        pm.store(past_it)
        fired = pm.scan_and_fire({"current_step": 10, "events": [], "variables": {}})
        assert len(fired) == 1, f"expected 1 fired, got {len(fired)}"
        assert fired[0].intention_id == past_it.intention_id
        pending_ids = {p.intention_id for p in pm.list_pending()}
        assert past_it.intention_id not in pending_ids

        # 3. event 触发: current_state 含该 event
        evt_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="on data_ready",
            trigger_type="event",
            trigger_payload={"event": "data_ready"},
            priority=6,
            created_at=time.time(),
            source_step=2,
        )
        pm.store(evt_it)
        # 不含该 event 时不触发
        fired = pm.scan_and_fire({"current_step": 10, "events": [], "variables": {}})
        assert len(fired) == 0
        # 含该 event 时触发
        fired = pm.scan_and_fire({"current_step": 10, "events": ["data_ready"], "variables": {}})
        assert len(fired) == 1
        assert fired[0].intention_id == evt_it.intention_id

        # 4. dependency 触发
        dep_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="after step 5",
            trigger_type="dependency",
            trigger_payload={"depends_on_step": 5},
            priority=4,
            created_at=time.time(),
            source_step=2,
        )
        pm.store(dep_it)
        # current_step=4 不触发
        fired = pm.scan_and_fire({"current_step": 4, "events": [], "variables": {}})
        assert len(fired) == 0, "step 4 should not trigger dep_on 5"
        # current_step=6 触发
        fired = pm.scan_and_fire({"current_step": 6, "events": [], "variables": {}})
        assert len(fired) == 1, "step 6 should trigger dep_on 5"
        assert fired[0].intention_id == dep_it.intention_id

        # 5. condition 触发
        cond_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="when memory_recall > 0.7",
            trigger_type="condition",
            trigger_payload={"expr": "memory_recall > 0.7"},
            priority=8,
            created_at=time.time(),
            source_step=4,
        )
        pm.store(cond_it)
        # 0.5 不触发
        fired = pm.scan_and_fire({"current_step": 10, "events": [], "variables": {"memory_recall": 0.5}})
        assert len(fired) == 0, "0.5 should not trigger > 0.7"
        # 0.8 触发
        fired = pm.scan_and_fire({"current_step": 10, "events": [], "variables": {"memory_recall": 0.8}})
        assert len(fired) == 1, "0.8 should trigger > 0.7"
        assert fired[0].intention_id == cond_it.intention_id

        # 6. condition 安全: 注入表达式返回 False, 不报错
        evil_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="evil",
            trigger_type="condition",
            trigger_payload={"expr": "__import__('os').system('rm -rf /')"},
            priority=9,
            created_at=time.time(),
            source_step=0,
        )
        pm.store(evil_it)
        # 不应该触发也不应该报错
        fired = pm.scan_and_fire({"current_step": 10, "events": [], "variables": {}})
        assert all(f.intention_id != evil_it.intention_id for f in fired), "evil expr must not fire"
        # 直接调 _check_condition_trigger 验证返回 False
        assert _check_condition_trigger({"expr": "__import__('os').system('rm -rf /')"}, {}) is False
        # 取消掉, 避免后续干扰
        pm.cancel(evil_it.intention_id)

        # 7. cancel + mark_fired 状态变更正确
        c_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="to be cancelled",
            trigger_type="event",
            trigger_payload={"event": "never"},
            priority=3,
            created_at=time.time(),
            source_step=1,
        )
        pm.store(c_it)
        assert pm.cancel(c_it.intention_id) is True
        # 再 cancel 应失败 (已 cancelled)
        assert pm.cancel(c_it.intention_id) is False
        # 已 cancel 的不能 mark_fired
        assert pm.mark_fired(c_it.intention_id) is False
        # 不存在的 intention
        assert pm.cancel("pim_nonexistent") is False
        assert pm.mark_fired("pim_nonexistent") is False

        # mark_fired 成功路径
        f_it = ProspectiveIntention(
            intention_id=_new_intention_id(),
            description="to be fired",
            trigger_type="event",
            trigger_payload={"event": "never"},
            priority=2,
            created_at=time.time(),
            source_step=1,
        )
        pm.store(f_it)
        assert pm.mark_fired(f_it.intention_id) is True
        # 再 fire 失败 (已 fired)
        assert pm.mark_fired(f_it.intention_id) is False
        # list_pending 不含
        assert all(p.intention_id != f_it.intention_id for p in pm.list_pending())

        # 8. format_for_context 输出含 description 和 source_step
        ctx = pm.format_for_context([
            ProspectiveIntention(
                intention_id="test",
                description="复现 Figure 3",
                trigger_type="time",
                trigger_payload={},
                priority=5,
                created_at=time.time(),
                source_step=7,
            )
        ])
        assert "复现 Figure 3" in ctx, f"description missing in: {ctx}"
        assert "step 7" in ctx, f"source_step missing in: {ctx}"
        # 空列表返回空串
        assert pm.format_for_context([]) == ""

        # 重启场景: 新实例能读到旧数据 (持久化生效)
        pm2 = ProspectiveMemory(tmp)
        all_it = pm2.list_all()
        # 之前存了: it, past_it, evt_it, dep_it, cond_it, evil_it, c_it, f_it = 8
        assert len(all_it) == 8, f"expected 8 after reload, got {len(all_it)}"

        print("All self-checks passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
