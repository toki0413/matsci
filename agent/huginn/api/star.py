"""Star —— 插件基类。借鉴 AstrBot 的 Star, 但去掉 Context 上帝对象。

插件作者继承 Star, 在方法上叠 @filter.* 装饰器, loader 会扫描
所有装饰过的成员函数, 生成 StarHandlerMetadata 注册到 registry。

跟 AstrBot 的差异:
  - 不通过 Context 注入所有组件, 改用聚焦的 PluginContext (4 字段)
  - priority 既是类属性也是装饰器参数, 装饰器优先
  - 不在 Star 里塞 logger / config / persona —— 那些走 PluginContext.extra 或独立模块
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.api.context import PluginContext
from huginn.api.event import Event


class Star:
    """所有插件的基类。

    子类约定:
      - 类属性 `name` 跟 metadata.yaml 的 name 对齐 (loader 会校验)
      - 类属性 `priority` 是该插件所有 handler 的默认优先级, 默认 0
      - 用 @filter.xxx 装饰的方法会被 loader 收集
      - 可选实现 async def on_load(self) / on_unload(self) 做生命周期管理
    """

    # 子类覆盖
    name: str = ""
    author: str = ""
    version: str = "0.0.0"
    description: str = ""
    priority: int = 0

    def __init__(self, context: PluginContext | None = None) -> None:
        self.context = context or PluginContext(plugin_name=self.name)
        # 每个插件实例一个 logger, 名字带插件名, 方便日志过滤
        self.logger = logging.getLogger(f"huginn.plugin.{self.name}")

    # ── 生命周期钩子 (子类按需覆写) ───────────────────────────────────

    async def on_load(self) -> None:
        """插件加载后调用。可做资源初始化、连接建立等。"""

    async def on_unload(self) -> None:
        """插件卸载前调用。可做资源清理。"""

    # ── 便捷访问 ─────────────────────────────────────────────────────

    @property
    def storage(self):
        """快捷访问 storage。未注入时抛错, 不静默 None。"""
        return self.context.require_storage()

    @property
    def llm(self):
        """快捷访问 LLM client。"""
        return self.context.require_llm()

    async def dispatch(self, event: Event) -> None:
        """主动发事件。一般插件用不到, 留给需要联动其他插件的场景。"""
        if self.context.event_bus is None:
            self.logger.warning("dispatch called but event_bus is not injected")
            return
        event.plugin_name = self.name
        await self.context.event_bus.dispatch(event)

    # ── 内省 ─────────────────────────────────────────────────────────

    def collect_handlers(self) -> list:
        """扫描实例上所有带 _huginn_filters 标记的方法, 转 metadata。

        loader 调这个方法完成注册。放在 Star 上而不是 registry 上,
        是因为只有 Star 实例知道自己的 priority / name。
        """
        from huginn.api.filter import markers_to_metadata

        metas = []
        for attr_name in dir(self):
            if attr_name.startswith("__"):
                continue
            # 跳过 Star 基类自己定义的 property / 便捷访问器, 否则在依赖
            # 没注入时会抛 RuntimeError 把整个收集过程打断.
            if attr_name in {"llm", "storage", "context", "logger", "dispatch"}:
                continue
            try:
                method = getattr(self, attr_name)
            except Exception:
                # property / descriptor 在依赖缺失时可能抛, 直接跳过
                continue
            # 只处理 bound method (实例方法). bound method 会把属性查找转发给
            # 底层 __func__, 所以 _huginn_filters 能直接拿到.
            if not hasattr(method, "__func__"):
                continue
            if not getattr(method, "_huginn_filters", None):
                continue
            # 存 bound method 而不是 __func__, 调用时 self 已绑, 不用单独传
            metas.extend(
                markers_to_metadata(method, plugin_name=self.name, priority=self.priority)
            )
        return metas


__all__ = ["Star"]
