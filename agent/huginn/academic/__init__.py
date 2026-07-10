"""学术期刊规范数据库模块.

提供期刊投稿规范查询、稿件合规检查、参考文献格式化等学术写作辅助功能.

主要接口:
  - journal_db: JournalSpec 数据类 + JOURNAL_DATABASE (14种期刊) + 查询函数
  - standards_checker: StandardsChecker 检查器 + CheckResult 结果类
  - paper_tool: PaperTool (HuginnTool 子类, 供 Agent 调用)
"""

from huginn.academic.journal_db import (
    JOURNAL_DATABASE,
    JournalSpec,
    REFERENCE_FORMATS,
    get_journal,
    get_reference_format,
    list_journals,
    search_journals,
)
from huginn.academic.standards_checker import CheckResult, StandardsChecker
from huginn.academic.paper_tool import PaperTool, PaperToolInput
from huginn.academic.deli_research import (
    DeliAutoResearch,
    DeliAutoResearchTool,
    ResearchStage,
    ResearchState,
)

__all__ = [
    "JOURNAL_DATABASE",
    "JournalSpec",
    "REFERENCE_FORMATS",
    "get_journal",
    "get_reference_format",
    "list_journals",
    "search_journals",
    "CheckResult",
    "StandardsChecker",
    "PaperTool",
    "PaperToolInput",
    "DeliAutoResearch",
    "DeliAutoResearchTool",
    "ResearchStage",
    "ResearchState",
]
