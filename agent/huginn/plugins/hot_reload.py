"""文件监听热重载 —— .py 变更触发对应插件 reload。

AstrBot 用 watchfiles.awatch + PythonFilter 盯着插件目录, 改了 .py 就
重新加载对应插件。本模块补上 loader.py 里故意留的口子:
  "故意不做的事 (留给后续): 不做文件监听自动热重载"

设计:
  - watchfiles 是基于 Rust notify 的封装, 资源占用低
  - 只盯 .py (PythonFilter), __pycache__ 之类忽略
  - 防抖: 编辑器一次保存常触发多次写事件, 100ms 窗口内合并成一次 reload
  - 通过环境变量 HUGINN_PLUGIN_RELOAD=1 显式开启 (默认关, 对齐 AstrBot 的 ASTRBOT_RELOAD)

用法:
    watcher = HotReloadWatcher(loader, plugins_dir=".")
    await watcher.start()  # 起后台监听任务, 立即返回
    await watcher.stop()   # 停止监听并清理
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from watchfiles import PythonFilter, awatch

logger = logging.getLogger("huginn.hot_reload")

# 对齐 AstrBot 的 ASTRBOT_RELOAD: 默认不开, 显式设 1 才启用
RELOAD_ENV_VAR = "HUGINN_PLUGIN_RELOAD"


class HotReloadWatcher:
    """盯插件目录, .py 改动时 reload 对应插件。

    用 watchfiles (Rust notify 封装) 监听, PythonFilter 只过 .py,
    自动忽略 __pycache__。

    防抖: 同一插件的多次快速变更合并成一次 reload (默认 100ms 窗口),
    避免编辑器一次保存触发多文件写入导致的重复 reload。
    """

    def __init__(
        self,
        loader: Any,  # PluginLoader, 用 Any 避免硬依赖
        plugins_dir: str | Path = ".",
        *,
        debounce_ms: int = 100,
    ) -> None:
        self._loader = loader
        self._plugins_dir = Path(plugins_dir)
        self._debounce_ms = debounce_ms
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._pending: set[str] = set()  # 等待 reload 的插件名
        self._debounce_task: asyncio.Task | None = None

    async def start(self) -> None:
        """后台开始监听, 立即返回。重复调用不会起多个任务。"""
        if self._task is not None:
            return  # 已经在跑了
        self._stopped.clear()
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("Hot-reload watcher started on %s", self._plugins_dir)

    async def stop(self) -> None:
        """停止监听, 等待清理完成。"""
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None
        logger.info("Hot-reload watcher stopped")

    async def _watch_loop(self) -> None:
        """主监听循环, 跑到 stop() 被调为止。"""
        try:
            async for changes in awatch(
                str(self._plugins_dir),
                watch_filter=PythonFilter(),
                recursive=True,
                stop_event=self._stopped,
            ):
                await self._handle_changes(changes)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Hot-reload watcher error: %s", e)

    async def _handle_changes(self, changes: set[tuple[Any, str]]) -> None:
        """把变更文件映射到插件名, 丢进防抖队列。

        防抖策略: 先收下所有受影响插件, 等 debounce_ms, 期间有新变更就
        重置计时器, 计时器走完再统一 reload 一次。
        """
        affected: set[str] = set()
        for change_type, file_path in changes:
            plugin_name = self._find_plugin_for_file(file_path)
            if plugin_name:
                affected.add(plugin_name)
                logger.debug(
                    "File %s changed -> plugin '%s' (change type: %s)",
                    file_path, plugin_name, change_type,
                )

        if not affected:
            return

        # 累加到 pending, (重新)启动防抖计时器
        self._pending.update(affected)
        if self._debounce_task is not None:
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_and_reload())

    def _find_plugin_for_file(self, file_path: str) -> str | None:
        """把文件路径映射回插件名。

        约定: 每个插件是 plugins_dir 下的一个子目录, 插件名 = 目录名。
        取相对路径的第一级作为插件名, 跳过 __pycache__ / 隐藏目录。

        文件不在插件目录内时返回 None。
        """
        try:
            path = Path(file_path).resolve()
            plugins_root = self._plugins_dir.resolve()
            rel = path.relative_to(plugins_root)
            parts = rel.parts
            if len(parts) < 1:
                return None
            plugin_name = parts[0]
            # 跳过 __pycache__ 这类内部目录
            if plugin_name.startswith("__") or plugin_name.startswith("."):
                return None
            return plugin_name
        except ValueError:
            return None

    async def _debounce_and_reload(self) -> None:
        """等一个防抖窗口, 然后把 pending 里的插件逐个 reload。"""
        try:
            await asyncio.sleep(self._debounce_ms / 1000.0)
        except asyncio.CancelledError:
            # 被新变更重置了计时器, 直接退出, 不 reload
            return

        to_reload = list(self._pending)
        self._pending.clear()
        self._debounce_task = None

        for plugin_name in to_reload:
            await self._reload_plugin(plugin_name)

    async def _reload_plugin(self, plugin_name: str) -> None:
        """通过 loader 重载单个插件。reload 本身是同步的, 丢线程池跑。"""
        try:
            logger.info("Reloading plugin '%s'...", plugin_name)
            result = await asyncio.to_thread(
                self._loader.reload, plugin_name
            )
            if result:
                logger.info("Plugin '%s' reloaded successfully", plugin_name)
            else:
                logger.warning("Plugin '%s' reload returned None/False", plugin_name)
        except Exception as e:
            logger.exception("Failed to reload plugin '%s': %s", plugin_name, e)


def is_hot_reload_enabled() -> bool:
    """读环境变量判断是否开启热重载。"""
    return os.getenv(RELOAD_ENV_VAR, "0") == "1"


__all__ = ["HotReloadWatcher", "is_hot_reload_enabled"]
