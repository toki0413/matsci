"""Task-scoped 智能体黑板 —— 结构化共享 + 来源标记, 替代"只传一段裸文本".

解决的问题(智能体孤岛):
  - orchestrator / team 的 sub-agent 之间只靠 ``outputs[task_id]`` 传纯字符串,
    下游无法区分"实测值 / 转述 / 预测", 也无法按条目前缀检索.
  - 本黑板以 ``(task_id, kind, provenance)`` 结构化承载, 下游用 ``digest`` 拿到
    **带来源标签** 的上下文, 从而在共享的同时保住"预测不得冒充实测"的红线.

诚实边界(对齐 world_state 的 provenance 纪律):
  - ``provenance="measured"`` 必须给 ``meta["source"]`` (工具名/数据文件),
    否则拒绝写入 —— 非实测观测不得伪装成实测.
  - ``digest(only_measured=True)`` 只投影实测条目, 用于"结论只能建立在实测上"的场景.

纯标准库, 无 LLM/网络; 线程安全(并行 sub-agent 会并发 post).
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

# 来源档位: 与 security.recovery_planner.Provenance 同语义(小写字符串投影).
PROVENANCE_LEVELS = ("measured", "reported", "predicted")
KINDS = ("finding", "evidence", "artifact", "error", "plan")

# digest 单条 value 的截断上限, 防止一个超长输出挤爆下游上下文.
_ENTRY_CHAR_CAP = 1200


@dataclass(frozen=True)
class Artifact:
    """黑板上的一个结构化条目 (不可变, 便于审计)."""
    key: str
    value: str
    kind: str
    producer: str
    provenance: str
    created_at: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Blackboard:
    """task-scoped 共享面: 写入带 producer/provenance, 读取可按 key/kind 过滤."""

    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id
        self._items: dict[str, list[Artifact]] = {}
        self._lock = threading.Lock()

    # ── 写 ──────────────────────────────────────────────────────

    def post(self, key: str, value: Any, *, producer: str = "",
             kind: str = "finding", provenance: str = "reported",
             meta: dict[str, Any] | None = None) -> Artifact:
        """写入一条共享条目. 返回写入的不可变 Artifact."""
        k = (key or "").strip()
        if not k:
            raise ValueError("blackboard key 不能为空")
        if kind not in KINDS:
            raise ValueError(f"未知 kind '{kind}' (可选 {KINDS})")
        if provenance not in PROVENANCE_LEVELS:
            raise ValueError(f"未知 provenance '{provenance}' (可选 {PROVENANCE_LEVELS})")
        md = dict(meta or {})
        if provenance == "measured" and not md.get("source"):
            # 红线: 没有实测来源就不能标 measured —— 非实测不得冒充实测.
            raise ValueError("provenance='measured' 需要 meta['source'] (工具名/数据来源)")
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, default=str)
        artifact = Artifact(
            key=k, value=text, kind=kind, producer=producer or "unknown",
            provenance=provenance,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            meta=md,
        )
        with self._lock:
            self._items.setdefault(k, []).append(artifact)
        return artifact

    # ── 读 ──────────────────────────────────────────────────────

    def read(self, key: str, *, provenance: str | None = None) -> Artifact | None:
        """读 key 的最新条目; 可限定 provenance."""
        with self._lock:
            items = list(self._items.get(key, ()))
        for art in reversed(items):
            if provenance is None or art.provenance == provenance:
                return art
        return None

    def entries(self, *, kind: str | None = None,
                only_measured: bool = False) -> list[Artifact]:
        """全部条目(按写入时间), 可按 kind/provenance 过滤."""
        with self._lock:
            flat = [a for items in self._items.values() for a in items]
        if kind is not None:
            flat = [a for a in flat if a.kind == kind]
        if only_measured:
            flat = [a for a in flat if a.provenance == "measured"]
        return sorted(flat, key=lambda a: a.created_at)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._items)

    def digest(self, keys: list[str] | None = None, *,
               max_chars: int = 4000, only_measured: bool = False) -> str:
        """把条目投影成可直接拼进下游 prompt 的上下文块 (带来源标签).

        空黑板/指定 keys 全缺 → 空串, 调用方可按需回退, 不产生"看起来有上下文"
        的空壳.
        """
        pool = self.entries(only_measured=only_measured)
        if keys:
            wanted = set(keys)
            pool = [a for a in pool if a.key in wanted]
        if not pool:
            return ""
        lines = [f"## Shared blackboard (task={self.task_id or 'n/a'})"]
        used = len(lines[0])
        for art in pool:
            val = art.value
            if len(val) > _ENTRY_CHAR_CAP:
                val = val[:_ENTRY_CHAR_CAP] + "…"
            src = art.meta.get("source")
            suffix = f" source={src}" if src else ""
            block = (f"- [{art.kind}/{art.provenance}] {art.key} "
                     f"(producer={art.producer}{suffix})\n{val}")
            if used + len(block) > max_chars:
                lines.append("- … (blackboard digest truncated)")
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)

    # ── 审计/快照 ───────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task_id": self.task_id,
                "items": {k: [a.to_dict() for a in v] for k, v in self._items.items()},
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
