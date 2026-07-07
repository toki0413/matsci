"""工具结果缓存 —— LRU 内存层 + SQLite 持久化层。

外部 API 查询（Materials Project / OQMD 这类）很慢，同一结构的查询
重复跑代价太高。这里做两层缓存：进程内 LRU 命中快，SQLite 落盘后
重启不丢。写操作（relax / scf）不要用这个，只给读操作挂。
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 超过这个大小的返回值不缓存，避免把 SQLite 撑爆或内存占用失控
_MAX_CACHEABLE_BYTES = 1 * 1024 * 1024

# 默认 TTL: 24h；外部 API 类查询建议给 7 天
DEFAULT_TTL = 24 * 3600
EXTERNAL_API_TTL = 7 * 24 * 3600

# prefetch 白名单: 只对幂等无副作用的轻量工具预热缓存.
# 重型仿真工具 (vasp/lammps/qe/...) 绝不在这里, 避免误触发烧算力.
PREFETCH_SAFE_TOOLS: set[str] = {
    "structure_tool",          # analyze/read 都是纯读
    "materials_database_tool",  # mp_summary/mp_structure 只查不写
    "symbolic_math_tool",       # 纯符号计算, 无副作用
}


def _stable_hash(obj: Any) -> str:
    """把任意可序列化对象转成稳定 hash key。"""
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _default_cache_dir() -> Path:
    """缓存目录默认放在用户 home 下的 .huginn/cache。"""
    override = os.environ.get("HUGINN_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".huginn" / "cache"


class ToolCache:
    """LRU + SQLite 持久化的工具结果缓存，线程安全。

    用法::

        cache = ToolCache()
        cache.set(("mp_structure", "mp-149"), {"records": [...]}, ttl=604800)
        hit = cache.get(("mp_structure", "mp-149"))

    key 可以是 tuple / str / 任意可序列化对象，内部统一 hash 成字符串。
    """

    _singleton_lock = threading.Lock()
    _singleton: ToolCache | None = None

    def __init__(
        self,
        db_path: Path | str | None = None,
        max_lru_size: int = 512,
    ) -> None:
        if db_path is None:
            db_path = _default_cache_dir() / "tool_cache.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_lru = max_lru_size
        # 进程内 LRU: key_str -> (value, expire_ts)
        self._lru: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._init_db()

    @classmethod
    def shared(cls) -> ToolCache:
        """进程级单例，避免每个工具各开一个 SQLite 连接。"""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ---- SQLite ----

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    expire_ts REAL NOT NULL,
                    created_ts REAL NOT NULL,
                    tool_name TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expire ON tool_cache(expire_ts)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False: 工具可能在不同线程调用，靠外层 RLock 保证安全
        conn = sqlite3.connect(str(self._db_path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _key_to_str(key: Any) -> str:
        if isinstance(key, str):
            return key
        return _stable_hash(key)

    # ---- 公开接口 ----

    def get(self, key: Any) -> dict[str, Any] | None:
        """命中返回 dict，未命中或过期返回 None。"""
        key_str = self._key_to_str(key)
        now = time.time()

        # L1: 内存 LRU
        with self._lock:
            entry = self._lru.get(key_str)
            if entry is not None:
                value, expire_ts = entry
                if now < expire_ts:
                    self._lru.move_to_end(key_str)
                    return value
                # 过期了，删掉
                self._lru.pop(key_str, None)

        # L2: SQLite
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value_json, expire_ts FROM tool_cache WHERE cache_key = ?",
                    (key_str,),
                ).fetchone()
            if row is None:
                return None
            value_json, expire_ts = row
            if now >= expire_ts:
                self._invalidate_one(key_str)
                return None
            value = json.loads(value_json)
        except Exception as exc:
            logger.debug("tool_cache get failed for %s: %s", key_str, exc)
            return None

        # 回填 L1
        with self._lock:
            self._lru[key_str] = (value, expire_ts)
            self._evict_lru()
        return value

    def set(self, key: Any, value: dict[str, Any], ttl: int = DEFAULT_TTL) -> None:
        """写入缓存。value 太大（>1MB）直接跳过。"""
        try:
            value_json = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.debug("tool_cache skip unserializable value: %s", exc)
            return

        if len(value_json.encode("utf-8")) > _MAX_CACHEABLE_BYTES:
            logger.debug("tool_cache skip oversized value (%d bytes)", len(value_json))
            return

        key_str = self._key_to_str(key)
        expire_ts = time.time() + max(ttl, 0)
        now = time.time()

        with self._lock:
            self._lru[key_str] = (value, expire_ts)
            self._evict_lru()

        tool_name = ""
        if isinstance(key, tuple) and key:
            tool_name = str(key[0])

        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO tool_cache "
                    "(cache_key, value_json, expire_ts, created_ts, tool_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key_str, value_json, expire_ts, now, tool_name),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("tool_cache set failed for %s: %s", key_str, exc)

    def invalidate(self, pattern: str) -> int:
        """按 glob 模式批量失效，返回删掉的条目数。

        pattern 匹配的是 tool_name（缓存 key 元组的第一个元素），
        比如 invalidate("materials_database_tool") 会清掉该工具的所有缓存。
        LRU 里的 key 是 hash 串没法直接匹配，索性整个清掉，下次 get
        会从 SQLite 重新加载。
        """
        deleted = 0
        regex = re.compile(pattern.replace("*", ".*"))

        # LRU 直接全清（hash key 没法按 tool_name 匹配，清掉最安全）
        with self._lock:
            deleted += len(self._lru)
            self._lru.clear()

        # SQLite 按 tool_name 列精确匹配
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT cache_key, tool_name FROM tool_cache"
                ).fetchall()
            to_delete = [
                k for k, tn in rows if (tn and regex.search(tn))
            ]
            if to_delete:
                with self._connect() as conn:
                    conn.executemany(
                        "DELETE FROM tool_cache WHERE cache_key = ?",
                        [(k,) for k in to_delete],
                    )
                    conn.commit()
                # deleted 已经算了 LRU 的数，这里只加 SQLite 独有的
                deleted = max(deleted, len(to_delete))
        except Exception as exc:
            logger.debug("tool_cache invalidate failed: %s", exc)

        return deleted

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock:
            self._lru.clear()
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM tool_cache")
                conn.commit()
        except Exception as exc:
            logger.debug("tool_cache clear failed: %s", exc)

    # ---- prefetch ----

    def prefetch(
        self,
        tool_name: str,
        common_inputs: list[dict],
        runner: Callable[[str, dict], dict | None] | None = None,
    ) -> int:
        """对一批常见输入预跑工具并缓存, 返回成功缓存数.

        只对 PREFETCH_SAFE_TOOLS 里的工具有效, 重型工具直接返回 0.
        runner 是实际调用工具的函数 (tool_name, input) -> result_dict,
        由 IntentSpeculator 提供. 不传 runner 就只统计已缓存的条目数,
        不实际预跑.

        幂等: 已缓存的条目跳过, 不会重复跑.
        """
        if tool_name not in PREFETCH_SAFE_TOOLS:
            logger.debug("prefetch skip unsafe tool: %s", tool_name)
            return 0

        count = 0
        for inp in common_inputs:
            key = (tool_name, _stable_hash(inp))
            # 已缓存就跳过, 避免重复跑
            if self.get(key) is not None:
                count += 1
                continue
            if runner is None:
                continue
            try:
                result = runner(tool_name, inp)
                if result is not None:
                    ttl = (
                        EXTERNAL_API_TTL
                        if tool_name == "materials_database_tool"
                        else DEFAULT_TTL
                    )
                    self.set(key, result, ttl=ttl)
                    count += 1
            except Exception as exc:
                logger.debug("prefetch runner failed for %s: %s", tool_name, exc)
        return count

    # ---- 内部 ----

    def _evict_lru(self) -> None:
        while len(self._lru) > self._max_lru:
            self._lru.popitem(last=False)

    def _invalidate_one(self, key_str: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM tool_cache WHERE cache_key = ?", (key_str,)
                )
                conn.commit()
        except Exception:
            logger.debug("connect failed", exc_info=True)


# ---- 装饰器 ----

def cacheable(
    ttl_seconds: int = DEFAULT_TTL,
    key_fn: Callable[..., Any] | None = None,
    tool_name: str | None = None,
) -> Callable:
    """给工具方法挂缓存。

    参数:
        ttl_seconds: 缓存有效期，默认 24h；外部 API 查询给 7 天。
        key_fn: 自定义 key 构造函数 (lambda *args, **kwargs -> tuple)。
                返回 None 表示这次不缓存。不传就用所有位置参数构造。
        tool_name: 工具名，会拼进 cache key，方便 invalidate 按工具清。

    用法::

        class MaterialsDatabaseTool:
            @cacheable(ttl_seconds=7*24*3600, tool_name="materials_database_tool")
            async def call(self, args, context):
                ...

    支持 sync 和 async 方法。返回值必须是 dict 或可 model_dump 的对象。
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        cache = ToolCache.shared()
        tname = tool_name or getattr(func, "__qualname__", "").split(".")[0]

        def _build_key(args: tuple, kwargs: dict) -> tuple | None:
            try:
                if key_fn is not None:
                    inner = key_fn(*args, **kwargs)
                else:
                    inner = _default_key(args, kwargs)
            except Exception:
                return None
            if inner is None:
                return None
            return (tname, _stable_hash(inner))

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            ckey = _build_key(args, kwargs)
            if ckey is not None:
                hit = cache.get(ckey)
                if hit is not None:
                    logger.debug("cacheable hit: %s", tname)
                    return _restore(hit, func)
            result = await func(*args, **kwargs)
            if ckey is not None:
                payload = _extract(result)
                if payload is not None:
                    cache.set(ckey, payload, ttl=ttl_seconds)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            ckey = _build_key(args, kwargs)
            if ckey is not None:
                hit = cache.get(ckey)
                if hit is not None:
                    logger.debug("cacheable hit: %s", tname)
                    return _restore(hit, func)
            result = func(*args, **kwargs)
            if ckey is not None:
                payload = _extract(result)
                if payload is not None:
                    cache.set(ckey, payload, ttl=ttl_seconds)
            return result

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _default_key(args: tuple, kwargs: dict) -> Any:
    """默认从位置参数里挑可序列化的部分构造 key。

    跳过 self / session / context 这类不可序列化或每次都变的对象，
    只保留 pydantic Model / dict / str / int 这类稳定的输入。
    """
    picked = []
    for arg in args[1:]:  # 跳过 self
        if hasattr(arg, "model_dump"):
            picked.append(arg.model_dump())
        elif isinstance(arg, (str, int, float, bool, dict, list, tuple)):
            picked.append(arg)
        else:
            # 遇到不可序列化的就停，后面的多半是 context 之类
            break
    for k, v in kwargs.items():
        if hasattr(v, "model_dump"):
            picked.append({k: v.model_dump()})
        elif isinstance(v, (str, int, float, bool, dict, list, tuple)):
            picked.append({k: v})
    return picked if picked else None


def _extract(result: Any) -> dict[str, Any] | None:
    """把工具返回值提取成可缓存的 dict。"""
    # ToolResult (dataclass)
    if hasattr(result, "data") and hasattr(result, "success"):
        return {
            "__type__": "ToolResult",
            "data": result.data,
            "success": getattr(result, "success", True),
            "error": getattr(result, "error", None),
        }
    # pydantic model
    if hasattr(result, "model_dump"):
        return {"__type__": "model", "data": result.model_dump()}
    if isinstance(result, dict):
        return {"__type__": "dict", "data": result}
    return None


def _restore(cached: dict[str, Any], func: Callable) -> Any:
    """从缓存 dict 还原成工具原始返回类型。"""
    kind = cached.get("__type__")
    if kind == "ToolResult":
        from huginn.types import ToolResult

        return ToolResult(
            data=cached.get("data"),
            success=cached.get("success", True),
            error=cached.get("error"),
        )
    if kind == "dict":
        return cached.get("data")
    # model 类型没法无损还原，直接返回 dict（调用方一般只读字段）
    return cached.get("data")
