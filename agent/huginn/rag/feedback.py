"""检索质量反馈循环 — 检索 -> 使用 -> 反馈 -> 调权.

思路: agent 先用 rag_tool 搜出一批文档, 之后调用别的工具时, 这次调用
成功与否反过来反映了"这批检索结果到底有没有用". 把这个信号攒下来,
下次检索时给历史表现好的文档加权、差的降权, 形成闭环.

两条 POST_TOOL_USE hook 配合:
- rag_track_hook: 抓 rag_tool 的搜索结果, 记到模块级 _last_rag_search
- rag_feedback_hook: 非 rag_tool 工具跑完后, 如果前面有 RAG 搜索,
  把这次的成功/失败反馈给 tracker

都不 block, 只观察.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 置信度边界: 成功 +0.05 上限 0.95; 失败 -0.1 下限 0.1; 默认 0.5
_DEFAULT_CONFIDENCE = 0.5
_SUCCESS_DELTA = 0.05
_FAILURE_DELTA = 0.1
_SUCCESS_CAP = 0.95
_FAILURE_FLOOR = 0.1


class RetrievalFeedbackTracker:
    """记录每次 RAG 搜索的结果和后续反馈, 维护每个文档的置信度.

    _results: tool_call_id -> {query, result_ids, outcome, tool_name}
    _doc_confidence: doc_id -> 置信度 (在默认值基础上累加)
    """

    def __init__(self) -> None:
        self._results: dict[str, dict[str, Any]] = {}
        self._doc_confidence: dict[str, float] = {}
        self._total_feedbacks = 0

    def track_search(
        self, query: str, result_ids: list[str], tool_call_id: str
    ) -> None:
        """记录一次 RAG 搜索. tool_call_id 由调用方生成, 用来串联后续反馈."""
        self._results[tool_call_id] = {
            "query": query,
            "result_ids": list(result_ids),
            "outcome": None,
            "tool_name": "",
        }

    def record_outcome(
        self, tool_call_id: str, success: bool, tool_name: str = ""
    ) -> None:
        """记录基于这次搜索结果的后续操作是否成功.

        同一次搜索只记第一次反馈, 避免重复计分.
        """
        entry = self._results.get(tool_call_id)
        if entry is None:
            return
        if entry["outcome"] is not None:
            # 已经反馈过了, 不重复计分
            return
        entry["outcome"] = success
        entry["tool_name"] = tool_name
        self._total_feedbacks += 1

        delta = _SUCCESS_DELTA if success else -_FAILURE_DELTA
        for doc_id in entry["result_ids"]:
            cur = self._doc_confidence.get(doc_id, _DEFAULT_CONFIDENCE)
            if success:
                cur = min(_SUCCESS_CAP, cur + delta)
            else:
                cur = max(_FAILURE_FLOOR, cur + delta)
            self._doc_confidence[doc_id] = cur

    def get_confidence(self, doc_id: str) -> float:
        """返回某个文档的置信度, 没记录过就是默认值."""
        return self._doc_confidence.get(doc_id, _DEFAULT_CONFIDENCE)

    def get_feedback_stats(self) -> dict[str, Any]:
        """汇总统计: 搜索次数 / 反馈次数 / 平均置信度 / 加权降权文档数."""
        if self._doc_confidence:
            avg = sum(self._doc_confidence.values()) / len(self._doc_confidence)
        else:
            avg = _DEFAULT_CONFIDENCE
        docs_boosted = sum(
            1 for v in self._doc_confidence.values() if v > _DEFAULT_CONFIDENCE
        )
        docs_penalized = sum(
            1 for v in self._doc_confidence.values() if v < _DEFAULT_CONFIDENCE
        )
        return {
            "total_searches": len(self._results),
            "total_feedbacks": self._total_feedbacks,
            "avg_confidence": round(avg, 4),
            "docs_boosted": docs_boosted,
            "docs_penalized": docs_penalized,
        }

    def adjust_search_results(self, results: list[dict]) -> list[dict]:
        """按置信度重排搜索结果, 高的排前, 低的降权.

        stable 排序: 置信度相同的保留原始相对顺序 (原始顺序本身是按距离
        从近到远排的, 别把它打乱).
        """
        # ponytail: 按 confidence 降序稳定排序, O(n log n), 文档量级 (top_k)
        # 一般个位数到几十, 完全够用. 升级路径: 距离和置信度加权融合.
        return sorted(
            results,
            key=lambda r: -self._confidence_of(r),
        )

    def _confidence_of(self, result: dict) -> float:
        """从一条搜索结果里取 doc_id, 查它的历史置信度.

        兼容 rag_tool ({id, metadata}) 和 KB ({chunk_id, metadata}) 两种格式;
        取不到 doc_id 就退回默认置信度, 不影响排序.
        """
        doc_id = self._extract_doc_id(result)
        if doc_id is None:
            return _DEFAULT_CONFIDENCE
        return self.get_confidence(doc_id)

    @staticmethod
    def _extract_doc_id(result: dict) -> str | None:
        """从搜索结果里抿出文档标识, 用于查置信度."""
        # 直接字段: rag_tool 用 id, KB 用 chunk_id/doc_id
        rid = result.get("id") or result.get("doc_id") or result.get("chunk_id")
        if rid:
            return str(rid)
        # 退到 metadata 里找
        meta = result.get("metadata")
        if isinstance(meta, dict):
            mid = meta.get("doc_id") or meta.get("id")
            if mid:
                return str(mid)
        return None


# ── 模块级单例 + 最近一次 RAG 搜索指针 ──────────────────────────────

_tracker: RetrievalFeedbackTracker | None = None
# 最近一次 rag_tool 搜索: {tool_call_id, query, result_ids}
# 非 rag_tool 工具跑完时拿它来归因反馈; 用完置 None, 一次搜索只喂一次反馈
_last_rag_search: dict[str, Any] | None = None


def get_feedback_tracker() -> RetrievalFeedbackTracker:
    """模块级单例, 第一次访问时建."""
    global _tracker
    if _tracker is None:
        _tracker = RetrievalFeedbackTracker()
    return _tracker


def reset_last_rag_search() -> None:
    """清掉最近搜索指针. 主要是给测试和重置场景用."""
    global _last_rag_search
    _last_rag_search = None


# ── Hook 实现 ────────────────────────────────────────────────────


def _extract_result_ids(result: Any) -> list[str]:
    """从 rag_tool 的返回里抿出文档 id 列表.

    rag_tool 返回结构: {success, data: {results: [{id, document, metadata}, ...]}}
    共享 KB 模式下 id 是 chunk_id (形如 docid_N), 我们取前缀 doc_id 部分.
    这里简单取整条 id, 置信度按 id 维度跟踪, 粒度是 chunk 级——够用.
    """
    if not isinstance(result, dict):
        return []
    data = result.get("data")
    if not isinstance(data, dict):
        return []
    items = data.get("results") or []
    ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = item.get("id") or item.get("doc_id") or item.get("chunk_id")
        if rid:
            ids.append(str(rid))
    return ids


def _tool_succeeded(ctx: HookContext) -> bool:
    """判断一次工具调用算不算成功.

    error 非空 -> 失败; 否则看 result 里有没有 error 字段.
    """
    if ctx.error is not None:
        return False
    result = ctx.result if isinstance(ctx.result, dict) else {}
    if result.get("error"):
        return False
    if result.get("success") is False:
        return False
    return True


async def rag_track_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: rag_tool 跑完后, 把搜索结果记到 _last_rag_search.

    不 block.
    """
    global _last_rag_search
    if ctx.tool_name != "rag_tool":
        return None

    args = ctx.args if isinstance(ctx.args, dict) else {}
    query = str(args.get("query", ""))
    result_ids = _extract_result_ids(ctx.result)

    tool_call_id = uuid.uuid4().hex
    tracker = get_feedback_tracker()
    tracker.track_search(query, result_ids, tool_call_id)
    _last_rag_search = {
        "tool_call_id": tool_call_id,
        "query": query,
        "result_ids": result_ids,
    }
    logger.debug(
        "rag_track_hook: tracked search '%s' (%d results)", query, len(result_ids)
    )
    return None


async def rag_feedback_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 非 rag_tool 工具跑完后, 给最近一次 RAG 搜索记反馈.

    如果前面有 RAG 搜索 (_last_rag_search 非空), 把这次的成功/失败归因到
    那次搜索的文档上. 记完就清掉指针, 一次搜索只喂一次反馈.
    不 block.
    """
    global _last_rag_search
    if ctx.tool_name == "rag_tool":
        return None

    last = _last_rag_search
    if not last:
        return None

    success = _tool_succeeded(ctx)
    tracker = get_feedback_tracker()
    tracker.record_outcome(last["tool_call_id"], success, ctx.tool_name)
    logger.debug(
        "rag_feedback_hook: %s %s -> feedback recorded for search '%s'",
        ctx.tool_name,
        "success" if success else "failure",
        last.get("query", ""),
    )
    # 一次搜索喂一次反馈, 用完清掉
    _last_rag_search = None
    return None
