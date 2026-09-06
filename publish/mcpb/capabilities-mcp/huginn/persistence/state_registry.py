"""字段级状态注册表 —— 跨 step / 跨进程恢复的轻量 KV 存储.

对标 Kimi Code 的 defineState / IAgentStateService, 但不照搬它的 30KB
cascadeEngine —— Python 不需要 TS 那套装饰器元数据.

设计:
  - 每个字段用 (namespace, key) 唯一标识, namespace 防止跨服务撞名
  - 字段注册时给 default_factory, 读取时若缺失则调用 factory 写回
  - 后端默认 SQLite, 路径跟 checkpointer 同目录避免散落
  - 进程内 LRU 缓存减少 DB 读, 写穿透 (write-through)
  - 线程安全 (RLock), 不做跨进程锁 —— 单进程多线程足够

不做的:
  - 不做 watch / reactive (Kimi 也没做)
  - 不做 schema migration (字段是 JSON, 加字段不需要迁移)
  - 不做跨机器复制 (单机部署场景)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huginn.utils.runtime import get_runtime_home


class StateRegistry:
    """字段级状态注册表, 进程内 LRU + SQLite 持久化.

    Usage::

        reg = StateRegistry.shared()
        reg.register("step_retry", "failed_attempts", lambda: 0)
        reg.set("step_retry", "failed_attempts", 3)
        val = reg.get("step_retry", "failed_attempts")  # 3
    """

    _singleton: StateRegistry | None = None
    _singleton_lock = threading.Lock()

    @classmethod
    def shared(cls) -> StateRegistry:
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = self._resolve_db_path(db_path)
        self._lock = threading.RLock()
        # LRU 缓存, 减少热点字段的 DB 读. ponytail: 1024 够覆盖所有字段, 不做配置.
        # 上限到了 LRU evict 最老的, evict 后下次 get 会从 DB 重新加载.
        self._cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._cache_max = 1024
        # 注册过的字段: (namespace, key) -> default_factory
        self._fields: dict[tuple[str, str], Callable[[], Any]] = {}
        self._init_db()

    # ------------------------------------------------------------------ DB

    @staticmethod
    def _resolve_db_path(explicit: str | Path | None) -> Path:
        if explicit is not None:
            p = Path(explicit).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        env_path = os.environ.get("HUGINN_STATE_REGISTRY_PATH")
        if env_path:
            p = Path(env_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        # 跟 checkpointer 同目录, 不散落
        default = get_runtime_home() / "state_registry.sqlite"
        default.parent.mkdir(parents=True, exist_ok=True)
        return default

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_kv (
                        namespace TEXT NOT NULL,
                        key       TEXT NOT NULL,
                        value     TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (namespace, key)
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ API

    def register(
        self,
        namespace: str,
        key: str,
        default_factory: Callable[[], Any],
    ) -> None:
        """注册一个字段. 读取时若 DB 没有则调用 factory 生成默认值并写回.

        重复注册同一个 (namespace, key) 会覆盖 factory —— 给测试用, 生产
        代码不要重复注册.
        """
        with self._lock:
            self._fields[(namespace, key)] = default_factory

    def get(self, namespace: str, key: str) -> Any:
        """读取字段值. 若未注册则 raise KeyError; 若 DB 没有则用 factory 初始化."""
        nk = (namespace, key)
        with self._lock:
            # 缓存命中
            if nk in self._cache:
                self._cache.move_to_end(nk)
                return self._cache[nk]

            # DB 读取
            value = self._db_get(nk)
            if value is None:
                # 用 factory 初始化
                factory = self._fields.get(nk)
                if factory is None:
                    raise KeyError(
                        f"StateRegistry field not registered: {namespace}.{key}"
                    )
                value = factory()
                self._db_set(nk, value)

            self._cache[nk] = value
            self._cache.move_to_end(nk)
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
            return value

    def set(self, namespace: str, key: str, value: Any) -> None:
        """写入字段值. write-through: 同时更新缓存和 DB."""
        nk = (namespace, key)
        with self._lock:
            self._db_set(nk, value)
            self._cache[nk] = value
            self._cache.move_to_end(nk)
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    def delete(self, namespace: str, key: str) -> bool:
        """删除字段. 返回是否真的删除了 (False = 本来就没有)."""
        nk = (namespace, key)
        with self._lock:
            existed = nk in self._cache or self._db_get(nk) is not None
            self._cache.pop(nk, None)
            self._db_delete(nk)
            return existed

    def reset(self, namespace: str, key: str) -> Any:
        """重置字段为默认值. 用于 turn 开始时清零计数器."""
        nk = (namespace, key)
        with self._lock:
            factory = self._fields.get(nk)
            if factory is None:
                raise KeyError(
                    f"StateRegistry field not registered: {namespace}.{key}"
                )
            value = factory()
            self._db_set(nk, value)
            self._cache[nk] = value
            self._cache.move_to_end(nk)
            return value

    # ------------------------------------------------------------------ DB ops

    def _db_get(self, nk: tuple[str, str]) -> Any | None:
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT value FROM state_kv WHERE namespace=? AND key=?",
                nk,
            ).fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def _db_set(self, nk: tuple[str, str], value: Any) -> None:
        import time
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """
                INSERT INTO state_kv (namespace, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (nk[0], nk[1], json.dumps(value, default=str), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def _db_delete(self, nk: tuple[str, str]) -> None:
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "DELETE FROM state_kv WHERE namespace=? AND key=?", nk
            )
            conn.commit()
        finally:
            conn.close()


# -- self-check -----------------------------------------------------------


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        reg = StateRegistry(db_path=Path(td) / "test.sqlite")

        # 1. register + get 初始化
        reg.register("test", "counter", lambda: 0)
        assert reg.get("test", "counter") == 0, "initial value"
        assert reg.get("test", "counter") == 0, "cached value"

        # 2. set + get
        reg.set("test", "counter", 5)
        assert reg.get("test", "counter") == 5

        # 3. 跨实例持久化 (模拟进程重启)
        reg2 = StateRegistry(db_path=Path(td) / "test.sqlite")
        reg2.register("test", "counter", lambda: 0)
        assert reg2.get("test", "counter") == 5, "persisted across instances"

        # 4. reset 回默认
        assert reg2.reset("test", "counter") == 0
        assert reg2.get("test", "counter") == 0

        # 5. delete
        reg2.set("test", "counter", 10)
        assert reg2.delete("test", "counter") is True
        # delete 后 get 会用 factory 重新初始化
        assert reg2.get("test", "counter") == 0

        # 6. 未注册字段 raise KeyError
        try:
            reg2.get("test", "unregistered")
            raise AssertionError("should have raised KeyError")
        except KeyError:
            pass

        # 7. LRU evict (cache 满了不崩)
        reg3 = StateRegistry(db_path=Path(td) / "lru.sqlite")
        reg3._cache_max = 3
        for i in range(10):
            reg3.register("lru", f"k{i}", lambda i=i: i)
            reg3.set("lru", f"k{i}", i)
        # 所有值都能从 DB 读出来, 缓存 evict 不丢数据
        assert reg3.get("lru", "k5") == 5
        assert reg3.get("lru", "k0") == 0

    print("StateRegistry self-checks passed.")
