"""召回门面 — 按意图把"召回"路由到唯一正确后端.

此前召回散成三扇互不相通的门:

  - ``memory_tool.RecallTool``         语义检索 long-term memory (需 memory_manager)
  - ``recall_context_tool``            按 category 召回外置化上下文 (metacog)
  - ``observation_pack.recall_observation``  按 ``@obs:<uuid>`` 句柄分页回原文

模型必须自己判断该敲哪扇门, 选错就得到空结果 (典型"召回孤岛"). 本模块把
"意图 → 后端"的判定收口成确定性函数, 由 ``recall`` 工具作为统一门面调用.

设计边界:
  - **不新增存储、不复制后端逻辑**: 每个 intent 只是对既有后端的薄委托;
  - **不猜**: 缺少 memory_manager / category / handle 时如实返回 available=False,
    绝不把"没查到"伪装成"查到了空";
  - trace_inspect / frame_inspect 仍是 autoloop 引擎侧 mode (``recall_modes``),
    它们需要 engine 句柄, 工具上下文拿不到, 故**不**从本门面伪造可达性.
"""
from __future__ import annotations

from typing import Any

# ── 意图枚举 (字符串常量, 便于工具 schema / SSE 直接消费) ──────────────
INTENT_MEMORY = "memory"
INTENT_CONTEXT = "context"
INTENT_OBSERVATION = "observation"
INTENTS: tuple[str, ...] = (INTENT_MEMORY, INTENT_CONTEXT, INTENT_OBSERVATION)

AUTO = "auto"


def detect_intent(
    *,
    handle: str | None = None,
    category: str | None = None,
    mode: str = AUTO,
) -> str:
    """确定性意图判定 (无 LLM / 无随机).

    显式 mode 优先; auto 时按"句柄 > 类别 > 语义检索"的强度排序:
    句柄是唯一可精确定位一条记录的形式, 类别次之, 自由文本最弱.
    """
    m = (mode or AUTO).strip().lower()
    if m != AUTO:
        if m not in INTENTS:
            raise ValueError(f"unknown recall mode: {mode!r}; expected one of {INTENTS}")
        return m
    if (handle or "").strip():
        return INTENT_OBSERVATION
    if (category or "").strip():
        return INTENT_CONTEXT
    return INTENT_MEMORY


def _unavailable(intent: str, error: str) -> dict[str, Any]:
    """统一失败形状 — 调用方只需看 available / error, 不必猜后端差异."""
    return {
        "intent": intent,
        "source": intent,
        "available": False,
        "count": 0,
        "results": [],
        "error": error,
    }


def route_recall(
    *,
    query: str = "",
    category: str | None = None,
    top_k: int = 5,
    tier: str | None = None,
    handle: str | None = None,
    page: int = 0,
    page_size: int = 4000,
    mode: str = AUTO,
    memory_manager: Any = None,
    session_id: str = "default",
) -> dict[str, Any]:
    """按意图路由一次召回, 返回统一结构.

    返回字段: intent / source / available / count / results / error (可选) / found (observation).
    """
    try:
        intent = detect_intent(handle=handle, category=category, mode=mode)
    except ValueError as exc:
        return _unavailable("unknown", str(exc))

    if intent == INTENT_OBSERVATION:
        if not (handle or "").strip():
            return _unavailable(intent, "observation recall requires a handle like '@obs:<uuid>'")
        try:
            from huginn.tools.observation_pack import ObservationArchive

            res = ObservationArchive(session_id).recall(
                handle, page=max(0, int(page)), page_size=max(1, int(page_size))
            )
        except Exception as exc:  # noqa: BLE001 — 召回失败如实上报, 不静默吞
            return _unavailable(intent, f"{type(exc).__name__}: {exc}")
        if not res.get("found"):
            out = _unavailable(intent, str(res.get("error") or "observation not found"))
            out["found"] = False
            return out
        return {
            "intent": intent,
            "source": "observation_pack",
            "available": True,
            "count": 1,
            "results": [res],
            "found": True,
            "error": None,
        }

    if intent == INTENT_CONTEXT:
        if not (category or "").strip():
            return _unavailable(intent, "context recall requires a category")
        try:
            # 叶子模块导入 (不反向依赖 metacog.__init__, 断开导入环);
            # raise_on_error=True 让"查询失败"不被伪装成"查到了空".
            from huginn.metacog.audit_context import recall_context as _recall_context

            results = _recall_context(
                category=category.strip(),
                query=query,
                top_k=_clamp(top_k),
                raise_on_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _unavailable(intent, f"{type(exc).__name__}: {exc}")
        results = list(results or [])
        return {
            "intent": intent,
            "source": "metacog.recall_context",
            "available": True,
            "count": len(results),
            "results": results,
            "error": None,
        }

    # memory: 需要运行时注入的 memory_manager; 缺失时诚实失败 (保持既有语义)
    if memory_manager is None:
        return _unavailable(
            intent,
            "memory recall requires an active memory_manager "
            "(use mode='context' for category recall, or handle='@obs:...' for archived output)",
        )
    try:
        results = memory_manager.recall(
            query=query, category=category, tier=tier, top_k=_clamp(top_k)
        )
    except Exception as exc:  # noqa: BLE001
        return _unavailable(intent, f"{type(exc).__name__}: {exc}")
    results = list(results or [])
    return {
        "intent": intent,
        "source": "memory_manager",
        "available": True,
        "count": len(results),
        "results": results,
        "error": None,
    }


def _clamp(top_k: Any) -> int:
    """把 top_k 收敛到 [1, 50] —— 三个后端的上限语义一致, 避免路由后各自再猜."""
    try:
        n = int(top_k)
    except (TypeError, ValueError):
        n = 5
    return max(1, min(n, 50))


__all__ = [
    "AUTO",
    "INTENTS",
    "INTENT_CONTEXT",
    "INTENT_MEMORY",
    "INTENT_OBSERVATION",
    "detect_intent",
    "route_recall",
]
