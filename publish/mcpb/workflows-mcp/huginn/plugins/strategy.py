"""策略注册表 —— "Everything is a Plugin" 的形态 B (轻量策略选择).

与形态 A (事件钩子, 见 plugins/event_bus.py) 区分:
  - 形态 A: 有生命周期、需要顺序 + 阻断, 走 on_llm_request 等事件钩子
  - 形态 B: 纯策略选择、性能敏感、无嵌套副作用, 走这里

设计:
  - 一个策略注册表 = 一个 dict[name -> Callable] + priority 仲裁
  - 注册不做任何异步分发, register() 是 O(1) 插入
  - select(name) 按注册顺序返回第一个匹配策略, 无匹配返回 fallback
  - 支持按 name 精确匹配, 也支持 prefix 通配 (如 "tool.X" 匹配 "tool.X.compile")
  - 线程安全: 内部 lock 保护读写
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class StrategyRegistry(Generic[T]):
    """轻量策略注册表。

    用法::

        reg = StrategyRegistry[Callable[[Any], Any]](fallback=default)
        reg.register("vasp", vasp_compress, priority=100)
        reg.register("tool.", wildcard_compress, priority=50)
        fn = reg.resolve("vasp.run")   # 先精确 "vasp.run", 再前缀 "vasp", 再 "tool."
        result = fn(data)
    """

    def __init__(self, fallback: T | None = None) -> None:
        self._entries: list[tuple[str, int, T]] = []  # (name, priority, strategy)
        self._fallback = fallback
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        strategy: T,
        priority: int = 0,
    ) -> None:
        """注册策略。name 支持精确名或前缀 (以 '.' 结尾的前缀匹配名)。

        priority 越大, 在 resolve 时越优先被尝试 (同前缀冲突时高者胜).
        """
        with self._lock:
            self._entries.append((name, priority, strategy))
            # 稳定排序: priority 降序, 同 priority 保持注册顺序
            self._entries.sort(key=lambda e: e[1], reverse=True)

    def unregister(self, name: str) -> int:
        """按 name 移除所有匹配策略。返回移除条数。"""
        removed = 0
        with self._lock:
            keep = []
            for entry in self._entries:
                if entry[0] == name:
                    removed += 1
                else:
                    keep.append(entry)
            self._entries = keep
        return removed

    def resolve(self, key: str) -> T | None:
        """按 key 找到最匹配的策略。

        匹配优先级 (从高到低):
          1. 精确名 (key == name)
          2. 最长前缀 (key.startswith(name) 且 name 以 '.' 结尾)
          3. fallback
        同层冲突由注册 priority 决定 (越大越前).
        """
        with self._lock:
            exact: T | None = None
            prefix_candidates: list[tuple[int, T]] = []
            for name, priority, strategy in self._entries:
                if name == key:
                    # 精确匹配: 取第一个 (priority 已降序)
                    exact = strategy
                    break
                if name.endswith(".") and key.startswith(name):
                    prefix_candidates.append((priority, strategy))
            if exact is not None:
                return exact
            if prefix_candidates:
                # priority 已降序, 取第一个即最高
                return prefix_candidates[0][1]
        return self._fallback

    def ordered(self) -> list[tuple[str, int, T]]:
        """返回按 priority 升序排列的 (name, priority, strategy) 快照。

        用于"按优先级依次执行全部策略"的场景 (如 prompt 段拼接),
        resolve() 只取单个策略, 这里取全部并稳定排序。
        """
        with self._lock:
            return sorted(self._entries, key=lambda e: e[1])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = ["StrategyRegistry"]
