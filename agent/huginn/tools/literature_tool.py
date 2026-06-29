"""shim: 实现已移至 huginn.tools.literature 包.

原 2720 行单文件拆为 6 模块:
  _http            - HTTP 层 + opener 单例
  search_sources   - 7 路学术搜索 + 去重
  pdf_fetch        - PDF 抓取 + Sci-Hub + DOI→OA 辅助
  crawl_web        - 爬虫 + 订阅源认证 session
  tool             - LiteratureTool 主体 + LiteratureInput
  __init__         - re-export
"""
from huginn.tools.literature import LiteratureInput, LiteratureTool  # noqa: F401
from huginn.tools.literature.search_sources import _dedup, _norm_title, _sort_papers  # noqa: F401
from huginn.tools.literature.crawl_web import (  # noqa: F401
    _PROVIDERS,
    _apply_ezproxy,
    _detect_provider,
    _list_sessions,
)

__all__ = ["LiteratureInput", "LiteratureTool"]
