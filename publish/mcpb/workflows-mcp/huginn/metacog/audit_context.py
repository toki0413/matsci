"""审计/召回上下文 — metacog 子模块共享的叶子工具.

从 ``huginn/metacog/__init__.py`` 下沉而来: ``block_registry`` /
``completion_auditor`` / ``equivalence_auditor`` 都需要 ``recall_audit_context``,
若从包 ``__init__`` 取, 就让这三个子模块反向依赖包初始化, 与 ``__init__`` 对
它们的正向 import 构成导入环 (且依赖"函数定义必须早于子模块 import"的脆弱顺序).

本模块不 import 任何 ``huginn.metacog.*`` —— 是叶子, 所以子模块引用它不产生环.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def recall_context(category: str, query: str = "", top_k: int = 3) -> list[dict]:
    """通用 context recall — 从 long-term memory 按类别捞外置化上下文 (G19).

    支持的 category: autoloop_summary / benchmark_run_summary / skill_invocation /
    knowledge_seed / stable_principles / hypothesis / failure / subgoal 等.

    Args:
        category: memory.longterm 的 category 字段
        query: 可选, 用于 FTS 过滤
        top_k: 最多返回多少条

    返回 list[dict], 每条至少含 {"content": str, "category": str, ...}
    失败返回空 list — 调用方不应因 memory 不可用而崩溃.
    """
    try:
        from huginn.memory.longterm import LongTermMemory
        # ponytail: 每次新建实例, 不缓存. 当前 metacog 没有持有 memory_manager 的入口,
        # 临时实例化开销可接受 (SQLite 即开即关). 升级路径: 让 engine 注入 memory_manager, 复用连接池.
        mem = LongTermMemory()
        # semantic=False: 确定性 + 不依赖 vector_store 配置, audit/recall 一致
        results = mem.retrieve(
            query=query, category=category, top_k=top_k, semantic=False
        )
        return results if results else []
    except Exception as e:
        logger.debug(f"recall_context({category}) failed: {e}")
        return []


def recall_audit_context(category: str, query: str = "", limit: int = 20) -> list[dict]:
    """向后兼容 wrapper — 老的 audit 入口, 委托给通用 recall_context.

    签名不变 (limit 参数保留), 内部把 limit 映射到 top_k.
    """
    return recall_context(category=category, query=query, top_k=limit)
