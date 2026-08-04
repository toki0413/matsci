"""工具调用去重 —— 同一 step 内防止重复调用相同工具+相同参数.

对标 Kimi Code 的 toolDedupe 模块. huginn 现有 _read_only_cache 只对
read_only=True 的工具生效, 对 bash_tool / code_tool / vasp_tool 这类
有副作用的非只读工具, 同一 step 内重复调用不会被拦.

场景: agent 在 rate limit 重试后可能重复调用同一工具; 或 LLM 在 ReAct
循环里卡住重复调用同一工具. 现有 loop_detector 是行为级 (检测连续 N 次
相同工具), 本模块是 step 级 (同一次 graph.stream() 内重复即拦).

设计:
  - 每个 step 开始时 reset, step 结束自动过期
  - key = (tool_name, args_hash), 命中即返回上一次的结果
  - 可配置 max_entries 防止参数组合爆炸 (默认 64 够一个 step 用)
  - 线程安全 (RLock), 单进程多线程足够

不做的:
  - 不做跨 step 去重 (跨 step 由 _read_only_cache + loop_detector 管)
  - 不做 TTL (step 生命周期短, 不需要)
  - 不做 redaction (args_hash 是哈希, 不存原始参数)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


def _args_hash(args: Any) -> str:
    """对工具参数算稳定哈希. dict 顺序无关 (sort_keys=True)."""
    try:
        s = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # 不可序列化的参数 (如 file handle), 退化到 repr
        s = repr(args)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class ToolDeduper:
    """同一 step 内的工具调用去重. 命中即返回上一次结果.

    Usage::

        deduper = ToolDeduper()
        # step 开始
        deduper.step_reset()

        # 第一次调用
        result = await run_tool("vasp_tool", {"system": "Si"})
        deduper.record("vasp_tool", {"system": "Si"}, result)

        # 同一 step 内重复调用 → 命中
        cached = deduper.lookup("vasp_tool", {"system": "Si"})
        if cached is not None:
            return cached  # 跳过真实执行
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._lock = threading.RLock()
        # ponytail: 64 够覆盖一个 step 的工具调用数. 升级路径: 按 tool 分桶.
        self._max = max_entries
        # key: (tool_name, args_hash) -> result
        self._entries: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0

    def step_reset(self) -> None:
        """step 开始时清空. 跨 step 不去重."""
        with self._lock:
            self._entries.clear()
            self._hit_count = 0
            self._miss_count = 0

    def lookup(
        self,
        tool_name: str,
        args: Any,
    ) -> dict[str, Any] | None:
        """查同 (tool, args) 是否已调过. 命中返回上次结果, 未命中返回 None."""
        key = (tool_name, _args_hash(args))
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._hit_count += 1
                # 返回深拷贝, 防止调用方修改污染缓存
                import copy
                return copy.deepcopy(self._entries[key])
            self._miss_count += 1
            return None

    def record(
        self,
        tool_name: str,
        args: Any,
        result: dict[str, Any],
    ) -> None:
        """记录一次工具调用结果. 后续同 (tool, args) lookup 会命中."""
        # 只缓存成功结果, 错误结果不缓存 (让 LLM 重试)
        if isinstance(result, dict) and result.get("error"):
            return
        key = (tool_name, _args_hash(args))
        with self._lock:
            self._entries[key] = result
            self._entries.move_to_end(key)
            if len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        """返回命中统计, 方便 telemetry."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": (
                    self._hit_count / (self._hit_count + self._miss_count)
                    if (self._hit_count + self._miss_count) > 0
                    else 0.0
                ),
            }


# -- self-check -----------------------------------------------------------


if __name__ == "__main__":
    import copy

    d = ToolDeduper(max_entries=3)

    # 1. 空查 miss
    assert d.lookup("vasp_tool", {"system": "Si"}) is None
    assert d.stats()["misses"] == 1

    # 2. record + lookup 命中
    d.record("vasp_tool", {"system": "Si"}, {"bandgap": 1.1})
    r = d.lookup("vasp_tool", {"system": "Si"})
    assert r == {"bandgap": 1.1}
    assert d.stats()["hits"] == 1

    # 3. 参数顺序无关 (sort_keys)
    d.record("tool", {"a": 1, "b": 2}, {"ok": True})
    assert d.lookup("tool", {"b": 2, "a": 1}) == {"ok": True}

    # 4. 不同参数不命中
    assert d.lookup("vasp_tool", {"system": "Ge"}) is None

    # 5. 错误结果不缓存
    d.record("fail_tool", {}, {"error": "boom"})
    assert d.lookup("fail_tool", {}) is None

    # 6. LRU evict (max=3)
    d.step_reset()
    d.record("t1", {}, {"v": 1})
    d.record("t2", {}, {"v": 2})
    d.record("t3", {}, {"v": 3})
    d.record("t4", {}, {"v": 4})  # t1 被 evict
    assert d.lookup("t1", {}) is None
    assert d.lookup("t4", {}) == {"v": 4}
    assert len(d._entries) == 3

    # 7. step_reset 清空
    d.step_reset()
    assert len(d._entries) == 0
    assert d.stats()["hits"] == 0

    # 8. 深拷贝: 调用方修改返回值不污染缓存
    d.record("copy_test", {}, {"nested": {"val": 1}})
    r1 = d.lookup("copy_test", {})
    r1["nested"]["val"] = 999
    r2 = d.lookup("copy_test", {})
    assert r2["nested"]["val"] == 1, "deep copy failed"

    print("ToolDeduper self-checks passed.")
