"""视觉渲染门禁 — 精修闭环的判定层 (批次 C 纯函数核心).

对应 WorldClaw/Qwen 的 verification gate: 生成图后判定是否值得带修正指令
重渲染, 还是降级兜底. 复用 audit.py 的语义 (verdict ∈ {pass, fix_needed, fail},
gap_type 驱动 retry 决策), 这里定义图域专属的 gap_type 与 retry 判定.

本模块为纯函数; 批次 C 后半提供 run_figure_gate 精修闭环, 由 visualize_tool
在生成图后调用 (render_fn 传入真正重渲染回调). 输出 action:
pass / retry / degrade / finalize / error. decision 里的 directive 为机读
修正指令, 供精修循环作为重渲染指令.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 可重渲染修复的 gap 类型 (re-render 能解决)
RERENDERABLE = frozenset({"render_geometry", "render_quality", "data_mismatch"})

# QA flag → gap 类型映射
_QA_GEOMETRY = frozenset({"blank", "clipped"})
_QA_QUALITY = frozenset({"cluttered", "extremely_elongated"})

# 严重度顺序 (用于 derive 时取最高优先级)
_SEVERITY = ["duplicate", "data_mismatch", "render_geometry", "render_quality"]


def derive_figure_gap_type(
    qa_flags: list[str] | None = None,
    consistency_flags: list[str] | None = None,
) -> str:
    """从 QA + 一致性 flags 推出图域 gap_type.

    优先级: duplicate > data_mismatch > render_geometry > render_quality > none.
    """
    flags = list(qa_flags or []) + list(consistency_flags or [])
    if not flags:
        return "none"
    if any(f == "duplicate_figure" for f in flags):
        return "duplicate"
    if any(str(f).startswith("numeric_drift") for f in flags):
        return "data_mismatch"
    if any(f in _QA_GEOMETRY for f in flags):
        return "render_geometry"
    if any(f in _QA_QUALITY for f in flags):
        return "render_quality"
    return "none"


def should_retry_render(
    verdict: str, gap_type: str, attempt: int, max_attempts: int
) -> bool:
    """重渲染是否值得. 仅 fix_needed/fail 且 gap 可重渲染修复且未超上限."""
    if verdict not in ("fix_needed", "fail"):
        return False
    if gap_type not in RERENDERABLE:
        return False
    return attempt < max_attempts


def _error_directive(qa_result: dict[str, Any] | None, consistency_result: dict[str, Any] | None) -> str:
    parts = []
    for r in (qa_result, consistency_result):
        if isinstance(r, dict) and r.get("error"):
            parts.append(str(r["error"]))
    return "visual gate error: " + (" | ".join(parts) if parts else "unknown")


def _directive(qa_result: dict[str, Any] | None, consistency_result: dict[str, Any] | None) -> str:
    """拼接机读修正指令 (重渲染时传给精修循环)."""
    parts: list[str] = []
    if isinstance(qa_result, dict):
        try:
            from huginn.tools.visualize_qa import qa_directive

            d = qa_directive(qa_result)
            if d:
                parts.append(d)
        except Exception:
            logger.debug("visualize_gate: qa_directive failed", exc_info=True)
            flags = (qa_result or {}).get("flags", [])
            parts.extend(f"qa {f}" for f in flags)
    cons = consistency_result or {}
    for f in cons.get("flags", []):
        parts.append(f"figure {f}")
    return " | ".join(parts) if parts else "figure needs refinement"


def render_gate_decision(
    qa_result: dict[str, Any] | None,
    consistency_result: dict[str, Any] | None,
    attempt: int,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """组合门禁判定 (批次 C 精修循环入口).

    返回 {action, verdict, gap_type, directive, flags, attempt, max_attempts}.
    action:
      - pass    全部通过
      - retry   带 directive 重渲染
      - degrade 无法重渲染修复 (duplicate / 硬性 fail) → 降级兜底图
      - finalize fix_needed 但已超尝试上限 → 接受并留痕
      - error   QA/一致性无有效依据
    """
    qa_flags = list((qa_result or {}).get("flags", []))
    cons_flags = list((consistency_result or {}).get("flags", []))
    flags = qa_flags + cons_flags

    # error 短路
    if (qa_result or {}).get("verdict") == "error" or (
        consistency_result or {}
    ).get("verdict") == "error":
        return {
            "action": "error",
            "verdict": "error",
            "gap_type": "none",
            "directive": _error_directive(qa_result, consistency_result),
            "flags": flags,
            "attempt": attempt,
            "max_attempts": max_attempts,
        }

    if not flags:
        return {
            "action": "pass",
            "verdict": "pass",
            "gap_type": "none",
            "directive": "",
            "flags": [],
            "attempt": attempt,
            "max_attempts": max_attempts,
        }

    gap_type = derive_figure_gap_type(qa_flags, cons_flags)
    # 硬性 fail: duplicate 或几何性 flag (blank/clipped)
    effective_verdict = (
        "fail"
        if any(f == "duplicate_figure" or f in _QA_GEOMETRY for f in flags)
        else "fix_needed"
    )
    directive = _directive(qa_result, consistency_result)

    if should_retry_render(effective_verdict, gap_type, attempt, max_attempts):
        action = "retry"
    elif gap_type == "duplicate" or effective_verdict == "fail":
        action = "degrade"
    else:
        action = "finalize"

    return {
        "action": action,
        "verdict": effective_verdict,
        "gap_type": gap_type,
        "directive": directive,
        "flags": flags,
        "attempt": attempt,
        "max_attempts": max_attempts,
    }


def run_figure_gate(
    image_path: str | Path,
    expected: dict[str, Any] | None = None,
    index: Any | None = None,
    attempt: int = 0,
    max_attempts: int = 2,
    extractor: Any | None = None,
    render_fn: Any | None = None,
    revertible: Any | None = None,
) -> dict[str, Any]:
    """完整精修闭环 (批次 C 后半): 生成图后自动跑 QA + 一致性 + 门禁.

    对应 WorldClaw render-based refinement 的「生成 → 诊断 → 重渲染 → 复检」
    循环. 纯函数, render_fn 可注入 (测试用 fake; 生产由 visualize_tool 传入
    真正重渲染回调).

    - image_path: 已生成的图路径.
    - expected: 源数据/figure_ir 数值快照, 供一致性校验 (可 None, 无则只跑 QA).
    - index: ImageIndex, 供重复图检测 (可 None).
    - render_fn: (directive: str) -> str | None. 返回新图路径或 None (无法重渲染).
      仅当 decision.action == "retry" 且未超上限时调用.
    - revertible: T-BCSE-14 时间可组合 — 可选 RevertibleContext. 传入时, 精修
      循环产生的**中间重渲染产物**登记文件逆, 使最终图确定后中间废弃图可逆清理.
    - 返回最终 decision + 附加 "image_path" (重渲染后指向最终图).

    ponytail: 重渲染失败 (render_fn 返回 None / 图未变) 直接 break, 不无限循环.
    所有校验异常都被各子函数内部吞掉, 本函数不抛异常.
    """
    from huginn.tools.visualize_check import consistency_verdict
    from huginn.tools.visualize_qa import qa_figure

    # T-BCSE-14 空间可组合: 门禁依赖 QA + 一致性, 链不完整时自动降级,
    # 不再跑无依据的门禁判定 (声明即保证, 见 huginn/security/visual_chain.py).
    try:
        from huginn.security.visual_chain import gate_available

        if not gate_available():
            return {
                "action": "degrade",
                "verdict": "fail",
                "gap_type": "none",
                "directive": "visual gate chain unavailable (QA/consistency backend missing)",
                "flags": [],
                "attempt": attempt,
                "max_attempts": max_attempts,
                "image_path": str(image_path),
            }
    except Exception:
        logger.debug("visualize_gate: space-composition check failed", exc_info=True)

    def _track_intermediate(path: str | Path) -> None:
        # 重渲染产生的中间产物登记逆: 整体回滚时删除, 避免残留废弃图.
        if revertible is not None:
            p = Path(path)
            revertible.track(lambda: p.unlink(missing_ok=True))

    def _check(path: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
        qa = qa_figure(path)
        cons = (
            consistency_verdict(path, expected, index=index, extractor=extractor)
            if (expected or index is not None)
            else None
        )
        return qa, cons

    qa, cons = _check(image_path)
    decision = render_gate_decision(qa, cons, attempt, max_attempts)
    current = Path(image_path)
    while decision["action"] == "retry" and render_fn is not None:
        try:
            new_path = render_fn(decision["directive"])
        except Exception:
            logger.debug("visualize_gate: render_fn failed", exc_info=True)
            break
        if not new_path or Path(new_path) == current:
            break  # 未产生新图 → 停止, 避免死循环
        _track_intermediate(current)  # 旧图将被新图取代, 登记为可逆中间产物
        current = Path(new_path)
        qa, cons = _check(current)
        decision = render_gate_decision(
            qa, cons, decision["attempt"] + 1, max_attempts
        )
    # 已到 retry 但无法再重渲染 (无 render_fn / 未产生新图): 转降级, 不滞留 retry
    if decision["action"] == "retry":
        decision = render_gate_decision(
            qa, cons, max_attempts, max_attempts
        )
    decision["image_path"] = str(current)
    return decision
