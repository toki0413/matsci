"""jieba 懒加载共享工具.

多个模块 (context_builder / trace_topology / visual_hippocampus / aggregator /
middlewares / knowledge.store) 依赖 jieba 做中文分词, 过去各自复制了一份
`_JIEBA` 缓存 + `_get_jieba()` 惰性加载. 这里收敛为唯一实现, 各模块只负责
导入使用, 避免重复代码和每次 import 的开销.

约定: 首次尝试导入 jieba 后缓存结果 (None=未尝试, False=不可用, 否则为
jieba 模块). 调用方拿到 None 时回退英文/ASCII 原逻辑即可.
"""
from __future__ import annotations

from typing import Any

# 缓存: None=未尝试, False=不可用, 否则为 jieba 模块.
_JIEBA: Any | None = None


def get_jieba() -> Any | None:
    """懒加载 jieba. 首次尝试后缓存结果, 避免每次分词都 import."""
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            _JIEBA = jieba
        except Exception:
            _JIEBA = False
    return _JIEBA if _JIEBA else None
