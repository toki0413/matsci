"""PluginLoader —— 扫描插件目录, 加载 metadata + 模块, 支持热重载。

参考 AstrBot 的插件加载, 但简化:
  - 每个插件目录含 metadata.yaml + main.py
  - main.py 里定义一个 Star 子类 (有多个时取第一个非抽象的)
  - 用 importlib 动态加载, 不污染 sys.modules
  - 热重载 = 卸载 + 重新加载 (不做文件 watch, 调用方主动触发)

故意不做的事 (留给后续):
  - 不做文件监听自动热重载 (要 watchdog 依赖, 先留接口)
  - 不做插件依赖解析 (AstrBot 也没真做)
  - 不做隔离的子解释器 (Python 这块太重, 走权限模型代替)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from huginn.api.context import PluginContext
from huginn.api.star import Star
from huginn.plugins.metadata import HUGINN_API_VERSION, PluginMetadata
from huginn.plugins.permissions import PermissionChecker

logger = logging.getLogger("huginn.plugin_loader")


# 默认插件目录: 项目根的 .huginn/plugins (相对 cwd)
# 也可以走 ~/.huginn/plugins 全局, 这里先取项目本地
DEFAULT_PLUGINS_DIR = ".huginn/plugins"


@dataclass
class LoadedPlugin:
    """已加载插件的运行时记录。"""

    metadata: PluginMetadata
    instance: Star
    module: Any
    plugin_dir: Path


# context 工厂签名: 拿 metadata, 返回 PluginContext
# 让调用方决定怎么注入 llm_client / tool_registry / storage / event_bus
ContextFactory = Callable[[PluginMetadata], PluginContext]


def default_context_factory(meta: PluginMetadata) -> PluginContext:
    """默认 context 工厂: 只填 plugin_name, 其他留空。

    生产环境应该传一个能注入真实 llm_client / storage 的工厂。
    """
    return PluginContext(plugin_name=meta.name)


@dataclass
class PluginLoader:
    """插件加载器。

    依赖:
      - registry: StarHandlerRegistry (必传)
      - permission_checker: PermissionChecker (必传, 权限强制)
      - event_bus: EventBus (可选, 用于注入到 PluginContext)
      - context_factory: 生成 PluginContext 的工厂
    """

    registry: Any = None
    permission_checker: PermissionChecker | None = None
    event_bus: Any = None
    context_factory: ContextFactory = default_context_factory
    plugins_dir: str | Path = DEFAULT_PLUGINS_DIR

    _loaded: dict[str, LoadedPlugin] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.registry is None:
            from huginn.plugins.registry import StarHandlerRegistry
            self.registry = StarHandlerRegistry()
        if self.permission_checker is None:
            self.permission_checker = PermissionChecker()

    # ── 加载 ─────────────────────────────────────────────────────────

    def discover(self) -> list[Path]:
        """扫描 plugins_dir, 返回所有含 metadata.yaml 的子目录。"""
        root = Path(self.plugins_dir)
        if not root.is_dir():
            return []
        result = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "metadata.yaml").is_file():
                result.append(child)
        return result

    def load_all(self) -> list[str]:
        """加载 plugins_dir 下所有插件。返回成功加载的插件名列表。"""
        loaded: list[str] = []
        for plugin_dir in self.discover():
            try:
                meta = self.load_one(plugin_dir)
                if meta is not None:
                    loaded.append(meta.name)
            except Exception as e:
                logger.exception("failed to load plugin from %s: %s", plugin_dir, e)
        return loaded

    def load_one(self, plugin_dir: str | Path) -> PluginMetadata | None:
        """加载单个插件目录。失败抛异常, 由 load_all 接。"""
        plugin_dir = Path(plugin_dir)
        meta_path = plugin_dir / "metadata.yaml"
        if not meta_path.is_file():
            logger.warning("skip %s: no metadata.yaml", plugin_dir)
            return None

        meta = PluginMetadata.from_yaml(meta_path)
        if not meta.name:
            raise ValueError(f"plugin at {plugin_dir} has empty name in metadata.yaml")

        # 版本约束检查
        if not meta.check_version_compatibility(HUGINN_API_VERSION):
            raise RuntimeError(
                f"plugin {meta.name!r} requires huginn {meta.huginn_version_range}, "
                f"but current API is {HUGINN_API_VERSION}"
            )

        # 已经加载过同名: 先卸载 (热重载场景)
        if meta.name in self._loaded:
            self.unload(meta.name)

        # 加载 main.py
        main_path = plugin_dir / "main.py"
        if not main_path.is_file():
            raise FileNotFoundError(f"plugin {meta.name!r}: main.py not found")

        module = self._load_module(meta.name, main_path)
        star_cls = self._find_star_subclass(module)
        if star_cls is None:
            raise RuntimeError(
                f"plugin {meta.name!r}: no Star subclass found in main.py"
            )

        # 实例化 + 注入 context
        context = self.context_factory(meta)
        # 让 event_bus 可用给插件主动 dispatch
        if self.event_bus is not None and context.event_bus is None:
            context.event_bus = self.event_bus
        instance = star_cls(context=context)
        # 把 metadata 里的版本/作者回填到 Star 类属性 (方便 registry 排查)
        instance.name = meta.name
        instance.version = meta.version
        instance.author = meta.author

        # 注册权限
        self.permission_checker.register(meta)

        # 收集 handler 注册到 registry
        metas = instance.collect_handlers()
        self.registry.register_iterable(metas)

        # 生命周期 on_load
        try:
            import asyncio
            coro = instance.on_load()
            if asyncio.iscoroutine(coro):
                # 同步加载上下文里没有 loop, 用 asyncio.run 跑
                # (注意: 如果调用方已在 event loop 内, 这里会炸, 那种场景应该用 load_one_async)
                try:
                    asyncio.get_running_loop()
                    # 已在 loop 内, 不能 asyncio.run, 创建 task 但不等
                    logger.warning(
                        "on_load of %s returned coro but we're in a running loop; "
                        "schedule without await", meta.name
                    )
                    from huginn.utils.concurrency import track_task
                    track_task(coro, name=f"plugin-onload-{meta.name}")
                except RuntimeError:
                    asyncio.run(coro)
        except Exception as e:
            logger.exception("on_load of %s failed: %s", meta.name, e)

        self._loaded[meta.name] = LoadedPlugin(
            metadata=meta, instance=instance, module=module, plugin_dir=plugin_dir
        )
        logger.info("loaded plugin %s v%s (%d handlers)",
                    meta.name, meta.version, len(metas))
        return meta

    async def load_one_async(self, plugin_dir: str | Path) -> PluginMetadata | None:
        """异步版 load_one: 在 event loop 内调用, 能正确 await on_load。

        逻辑跟 load_one 一样, 只是 on_load 用 await。为避免重复代码,
        load_one 内部 on_load 已经处理了 loop 内场景, 这里给个干净入口。
        """
        plugin_dir = Path(plugin_dir)
        meta_path = plugin_dir / "metadata.yaml"
        if not meta_path.is_file():
            return None

        meta = PluginMetadata.from_yaml(meta_path)
        if not meta.check_version_compatibility(HUGINN_API_VERSION):
            raise RuntimeError(
                f"plugin {meta.name!r} requires huginn {meta.huginn_version_range}"
            )

        if meta.name in self._loaded:
            self.unload(meta.name)

        main_path = plugin_dir / "main.py"
        module = self._load_module(meta.name, main_path)
        star_cls = self._find_star_subclass(module)
        if star_cls is None:
            raise RuntimeError(f"plugin {meta.name!r}: no Star subclass in main.py")

        context = self.context_factory(meta)
        if self.event_bus is not None and context.event_bus is None:
            context.event_bus = self.event_bus
        instance = star_cls(context=context)
        instance.name = meta.name
        instance.version = meta.version

        self.permission_checker.register(meta)
        metas = instance.collect_handlers()
        self.registry.register_iterable(metas)

        await instance.on_load()

        self._loaded[meta.name] = LoadedPlugin(
            metadata=meta, instance=instance, module=module, plugin_dir=plugin_dir
        )
        logger.info("loaded plugin %s v%s (%d handlers)",
                    meta.name, meta.version, len(metas))
        return meta

    # ── 卸载 / 热重载 ───────────────────────────────────────────────

    def unload(self, plugin_name: str) -> bool:
        """卸载插件: 调 on_unload, 从 registry/permission 清理。"""
        loaded = self._loaded.pop(plugin_name, None)
        if loaded is None:
            return False
        try:
            import asyncio
            coro = loaded.instance.on_unload()
            if asyncio.iscoroutine(coro):
                try:
                    asyncio.get_running_loop()
                    asyncio.ensure_future(coro)
                except RuntimeError:
                    asyncio.run(coro)
        except Exception as e:
            logger.exception("on_unload of %s failed: %s", plugin_name, e)

        self.registry.unregister_plugin(plugin_name)
        self.permission_checker.unregister(plugin_name)

        # 从 sys.modules 清掉, 避免下次 reload 拿到旧模块
        mod_name = f"huginn_plugin_{plugin_name}"
        sys.modules.pop(mod_name, None)
        logger.info("unloaded plugin %s", plugin_name)
        return True

    def reload(self, plugin_name: str) -> PluginMetadata | None:
        """热重载: 卸载后从原路径重新加载。

        实现简化: 重新扫描目录, 不做文件 watch。调用方按需触发。
        """
        loaded = self._loaded.get(plugin_name)
        plugin_dir = loaded.plugin_dir if loaded is not None else None
        if plugin_dir is None:
            # 没记录路径, 从 plugins_dir 推
            plugin_dir = Path(self.plugins_dir) / plugin_name
        self.unload(plugin_name)
        return self.load_one(plugin_dir)

    # ── 查询 ─────────────────────────────────────────────────────────

    def list_loaded(self) -> list[str]:
        return list(self._loaded.keys())

    def get_loaded(self, plugin_name: str) -> LoadedPlugin | None:
        return self._loaded.get(plugin_name)

    # ── 内部 ─────────────────────────────────────────────────────────

    def _load_module(self, plugin_name: str, main_path: Path):
        """用 importlib 加载 main.py 为独立模块, 不污染 sys.modules 太多。"""
        mod_name = f"huginn_plugin_{plugin_name}"
        # 已存在就 reload (热重载场景)
        if mod_name in sys.modules:
            return importlib.reload(sys.modules[mod_name])
        spec = importlib.util.spec_from_file_location(mod_name, main_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module from {main_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    def _find_star_subclass(self, module: Any) -> type[Star] | None:
        """在模块里找 Star 的子类 (非 Star 本身, 非抽象)。"""
        candidates: list[type[Star]] = []
        for attr in vars(module).values():
            if not isinstance(attr, type):
                continue
            if attr is Star:
                continue
            if issubclass(attr, Star):
                candidates.append(attr)
        if not candidates:
            return None
        # 多个时取第一个, 但记日志提示
        if len(candidates) > 1:
            logger.warning(
                "multiple Star subclasses found, using %s", candidates[0].__name__
            )
        return candidates[0]


__all__ = ["PluginLoader", "LoadedPlugin", "ContextFactory", "DEFAULT_PLUGINS_DIR"]
