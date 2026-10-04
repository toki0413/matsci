"""通信契约 —— ASD-STE100 纪律在**结构化通信层**的落地。

ASD-STE100 的三条内核 (一词一义 / 一句一意 / 显式无歧义) 不只适用于自然语言,
同样适用于模块、事件、工作流、agent 之间的结构化消息。本模块把这三条变成
**确定性检查**, 覆盖仓库里**已有的**四个通信面, 不新增平行信封:

  - event   : ``AgentEvent`` (统一事件总线, 模块 ↔ 模块)
  - workflow: ``WorkflowStageEvent`` / ``Event`` 里带 workflow_name/stage_name 的
  - message : ``AgentMessage`` (agent ↔ agent 的对话消息)
  - result  : ``SubagentResult`` (子 agent → 主 agent 的交接结果)

检查清单 (逐条对应 STE 内核):
  1. 信封完备 —— type / timestamp / source 必须在 (显式, 收端无需回问)。
  2. 类型词表 —— type 必须点分小写且登记在册 (一词一义: 不许别名/自造)。
  3. 载荷契约 —— 已知事件 (发射点 docstring 声明过字段) 的必备字段必须在。
  4. 字段别名漂移 —— data 里出现 ``tool`` 而非 ``tool_name`` 等别名即报告。
  5. 结果自洽 —— ``SubagentResult`` 不许 "success=True 却带 error" 这类自相矛盾。

语义与仓库 fail-open 惯例一致: **只报告, 不拦截**。``hard`` 表示"声明了却没做到"
(收端无法解析), ``advisory`` 表示"能解析但违反纪律"。调用方决定是否告警/累积。

不 import ``event_bus`` (避免环): 全部走 duck typing。
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from huginn.events.event_types import ALL_TYPES

__all__ = [
    "Violation",
    "ContractReport",
    "KNOWN_EVENT_TYPES",
    "PAYLOAD_CONTRACTS",
    "FIELD_ALIASES",
    "audit_agent_event",
    "audit_plugin_event",
    "audit_agent_message",
    "audit_subagent_result",
    "lint",
]

# ── 事件词表 ─────────────────────────────────────────────────────────
# 单一真相: 事件类型的权威声明面就是 ``events/event_types.py`` 的 ``ALL_TYPES``
# (静态面由 ``huginn/cli/contract_audit.py`` 的事件面审计保证双向一致)。
# 这里**不再自建**第二份词表 —— 一词一义同样适用于"清单只有一个"。
# 真实存在但未声明就发布的事件 (team.* / campaign.control_trace /
# event_bus.dropped) 会如实报 advisory ``type.unknown``, 这正是 contract_audit
# 记的"候选缺口", 是治理信号而非噪声。
KNOWN_EVENT_TYPES: frozenset[str] = frozenset(t for t in ALL_TYPES if t != "*")

# 已知事件的**载荷契约**: 键 = 事件 type, 值 = 必备 data 字段。
# 来源是各发射点在 docstring / 注释里明说的"data 对齐契约", 不是这里新发明。
# 缺必备字段 = hard (声明了却没给, 收端无法解析)。
PAYLOAD_CONTRACTS: dict[str, tuple[str, ...]] = {
    # events/event_types.py: "data 字段: attempt / max_attempts / error_type /
    # error_message / wait_ms / states_yielded"
    "agent.step.retrying": ("attempt", "max_attempts", "error_type"),
    # event_types.py: "data 对齐契约 §3.2: id / kind / status / narrative /
    # agent_judgment / options"
    "decision.point": ("id", "kind", "status", "narrative"),
    # 控制面 trace: engine_reflect/cognitive_loop 固定记录 name + iteration
    "campaign.control_trace": ("name", "iteration"),
}

# 一词一义: 别名字段 → 规范名。data 里出现别名即 advisory 漂移。
# 记录的是仓库里**真实存在**的漂移 (如 tool_adapter 用 "tool", typed 事件用 "tool_name")。
FIELD_ALIASES: dict[str, str] = {
    "tool": "tool_name",
    "msg": "message",
    "sid": "session_id",
    "src": "source",
    "dst": "target",
    "dest": "target",
    "err": "error",
    "ts": "timestamp",
    "ctx": "context",
    "cfg": "config",
    "obj": "objective",
    "desc": "description",
}

# 点分小写命名空间: namespace.segment[.segment...]
_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
# 标识符 (source)
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_AGENT_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclasses.dataclass(frozen=True)
class Violation:
    """一条契约违规。severity: ``hard`` (收端无法解析) / ``advisory`` (可解析但违规)。"""

    rule: str
    severity: str
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ContractReport:
    """一次检查的报告。``ok`` 只看 hard —— advisory 不判失败。"""

    surface: str
    violations: list[Violation] = dataclasses.field(default_factory=list)

    @property
    def hard_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "hard")

    @property
    def advisory_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "advisory")

    @property
    def ok(self) -> bool:
        return self.hard_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "ok": self.ok,
            "hard_count": self.hard_count,
            "advisory_count": self.advisory_count,
            "violations": [v.to_dict() for v in self.violations],
        }


def _violation(rule: str, severity: str, field: str, message: str) -> Violation:
    return Violation(rule=rule, severity=severity, field=field, message=message)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """从 dataclass 对象或 dict 里取字段。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    """把任意消息对象折成 dict, 供面别推断 (不改动原对象)。"""
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:  # — 原因: 含不可序列化字段时 asdict 会炸, 退化为逐字段取
            pass
    out: dict[str, Any] = {}
    for k in ("type", "timestamp", "source", "data", "role", "content", "metadata",
              "summary", "success", "error", "workflow_name", "stage_name"):
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out


# ── 面 1: 统一事件信封 ────────────────────────────────────────────────

def audit_agent_event(ev: Any) -> list[Violation]:
    """检查 ``AgentEvent`` 信封 + 类型词表 + 载荷契约 + 别名漂移。"""
    out: list[Violation] = []
    etype = str(_field(ev, "type", "") or "")
    ts = _field(ev, "timestamp", None)
    source = str(_field(ev, "source", "") or "")
    data = _field(ev, "data", {}) or {}

    # 1. 信封完备
    if not etype:
        out.append(_violation("envelope.type", "hard", "type", "事件缺少 type"))
    else:
        # 2. 类型词表 (一词一义)
        if not _TYPE_RE.match(etype):
            out.append(_violation("type.naming", "hard", "type",
                          f"type={etype!r} 非点分小写命名空间 (需 namespace.segment)"))
        elif etype not in KNOWN_EVENT_TYPES:
            out.append(_violation("type.unknown", "advisory", "type",
                          f"type={etype!r} 未登记进事件词表 (登记或改用既有名)"))
    if ts is None:
        out.append(_violation("envelope.timestamp", "hard", "timestamp", "事件缺少 timestamp"))
    elif not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        out.append(_violation("envelope.timestamp", "advisory", "timestamp", "timestamp 非正数"))
    if not source:
        out.append(_violation("envelope.source", "advisory", "source", "事件未标 source (出端身份缺失)"))
    elif not _ID_RE.match(source):
        out.append(_violation("source.naming", "advisory", "source",
                      f"source={source!r} 非小写下划线标识"))

    # 3/4. 载荷
    if not isinstance(data, dict):
        out.append(_violation("payload.type", "hard", "data", "data 必须是 JSON object"))
        return out
    for key in data:
        canonical = FIELD_ALIASES.get(str(key))
        if canonical:
            out.append(_violation("field.alias", "advisory", str(key),
                          f"字段 {key!r} 是别名, 规范名 {canonical!r} (一词一义)"))
    required = PAYLOAD_CONTRACTS.get(etype)
    if required:
        for need in required:
            if need not in data:
                out.append(_violation("payload.required", "hard", need,
                              f"{etype} 缺少必备字段 {need!r} (载荷契约)"))
    try:
        json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError) as exc:  # — 原因: 只关心能否序列化, 具体异常类型无关
        out.append(_violation("payload.json", "advisory", "data", f"data 不可 JSON 序列化: {exc}"))
    return out


# ── 面 2: 插件 / 工作流事件 ───────────────────────────────────────────

def audit_plugin_event(ev: Any) -> list[Violation]:
    """检查插件 ``Event`` (含 ``WorkflowStageEvent``)。

    插件 Event.type 是 EventType 枚举; workflow 阶段的 Event 必须点名
    workflow_name / stage_name (否则收端不知是哪条流程的哪一步)。
    """
    out: list[Violation] = []
    etype = _field(ev, "type", None)
    if etype is None:
        out.append(_violation("envelope.type", "hard", "type", "插件事件缺少 type"))
    wf = _field(ev, "workflow_name", None)
    stage = _field(ev, "stage_name", None)
    # 只有明确带 workflow 语义的事件才要求 (避免误伤普通 Event)
    if wf is not None or stage is not None:
        if not str(wf or "").strip():
            out.append(_violation("workflow.name", "hard", "workflow_name",
                          "workflow 事件未标 workflow_name"))
        if not str(stage or "").strip():
            out.append(_violation("workflow.stage", "advisory", "stage_name",
                          "workflow 事件未标 stage_name (收端无法定位阶段)"))
    return out


# ── 面 3: agent 间对话消息 ────────────────────────────────────────────

def audit_agent_message(m: Any) -> list[Violation]:
    """检查 ``AgentMessage`` (agent ↔ agent / 主 ↔ 子 的对话消息)。"""
    out: list[Violation] = []
    role = str(_field(m, "role", "") or "")
    content = _field(m, "content", None)
    metadata = _field(m, "metadata", {}) or {}

    if role not in _AGENT_ROLES:
        out.append(_violation("message.role", "hard", "role",
                      f"role={role!r} 不在 {{system,user,assistant,tool}}"))
    if content is None:
        out.append(_violation("message.content", "hard", "content", "消息缺少 content"))
    elif isinstance(content, str) and not content.strip():
        out.append(_violation("message.content", "advisory", "content", "消息 content 为空串"))
    if metadata:
        try:
            json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError) as exc:  # — 原因: 同上, 只判可序列化性
            out.append(_violation("message.metadata", "advisory", "metadata",
                          f"metadata 不可 JSON 序列化: {exc}"))
    return out


# ── 面 4: 子 agent → 主 agent 的交接结果 ──────────────────────────────

def audit_subagent_result(r: Any) -> list[Violation]:
    """检查 ``SubagentResult`` 自洽性。

    这正是静默空转那个 bug 的机械闸口: 成功却带 error = 矛盾; 成功却 summary
    为空 = 喂回主上下文的是空串 (收端拿不到任何信息)。
    """
    out: list[Violation] = []
    success = _field(r, "success", None)
    summary = str(_field(r, "summary", "") or "")
    full_output = str(_field(r, "full_output", "") or "")
    error = _field(r, "error", None)

    if success is True and error:
        out.append(_violation("result.contradiction", "hard", "error",
                      "success=True 却带 error (自相矛盾)"))
    if success is False and not error:
        out.append(_violation("result.unexplained", "advisory", "error",
                      "success=False 未给 error 原因"))
    if success is True and not summary.strip():
        out.append(_violation("result.empty", "advisory", "summary",
                      "成功结果 summary 为空 (喂回主上下文的是空串)"))
    if not summary.strip() and not full_output.strip():
        out.append(_violation("result.empty", "hard", "summary",
                      "summary 与 full_output 同时为空 (无任何可交接内容)"))
    return out


# ── 统一入口 ─────────────────────────────────────────────────────────

_SURFACES = {
    "event": audit_agent_event,
    "plugin_event": audit_plugin_event,
    "workflow": audit_plugin_event,
    "message": audit_agent_message,
    "result": audit_subagent_result,
}


def _infer_surface(d: dict[str, Any]) -> str:
    if "role" in d and "content" in d:
        return "message"
    if "summary" in d and "success" in d:
        return "result"
    if "workflow_name" in d or "stage_name" in d:
        return "workflow"
    t = d.get("type")
    # 插件 Event.type 是 EventType 枚举; AgentEvent.type 是点分字符串.
    if t is not None and not isinstance(t, str):
        return "plugin_event"
    # source / data 是 AgentEvent 的判别位 (插件 Event 无 source).
    if "type" in d and ("timestamp" in d or "source" in d or "data" in d):
        return "event"
    if "type" in d:
        return "plugin_event"
    return "unknown"


def lint(payload: Any, surface: str | None = None) -> ContractReport:
    """对一条通信消息做契约检查。

    ``payload`` 可以是 dataclass 对象或 dict; ``surface`` 可显式指定
    (event / workflow / message / result), 省略时按字段自动推断。
    """
    d = _to_dict(payload)
    surf = surface or _infer_surface(d)
    fn = _SURFACES.get(surf)
    if fn is None:
        return ContractReport(surface=surf, violations=[
            _violation("surface.unknown", "hard", "",
               f"无法识别的通信面 (keys={sorted(d)})"),
        ])
    return ContractReport(surface=surf, violations=fn(payload))
