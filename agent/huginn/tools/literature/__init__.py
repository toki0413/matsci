"""文献调研工具包 —— LiteratureTool (10 action) + LiteraturePipelineTool (MinerU 流水线).

HTTP 层在 _http, 7 路学术搜索在 search_sources, PDF 抓取在 pdf_fetch,
爬虫与订阅源认证在 crawl_web, LiteratureTool 主体在 tool.
MinerU VLM 解析 + Schema 抽取 + 跨文献聚合在 pipeline_tool.
"""
from .tool import LiteratureInput, LiteratureTool
from .pipeline_tool import LiteraturePipelineTool, LiteraturePipelineInput

__all__ = [
    "LiteratureInput",
    "LiteratureTool",
    "LiteraturePipelineInput",
    "LiteraturePipelineTool",
]
