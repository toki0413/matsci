"""文献调研工具包 —— 7 action: search/summarize/benchmark_lookup/fetch_pdf/citations/ingest_to_rag/crawl_web.

HTTP 层在 _http, 7 路学术搜索在 search_sources, PDF 抓取在 pdf_fetch,
爬虫与订阅源认证在 crawl_web, LiteratureTool 主体在 tool.
"""
from .tool import LiteratureInput, LiteratureTool

__all__ = ["LiteratureInput", "LiteratureTool"]
