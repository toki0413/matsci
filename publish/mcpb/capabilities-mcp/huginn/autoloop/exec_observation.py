"""执行观测层 (Exec Observation) — 只读的"执行健康"与"自演"探针.

借鉴 AI 渗透/红队项目的**失败目录** (不是它们的架构): RapidPen 给每个任务节点挂
``exit_class``, 把"成功 / 失败 / 环境错误 / 超时"分开, 避免把**工具故障误判成科学
结论**; EnIGMA 命名了 ``soliloquizing`` —— 模型不执行命令却自编"环境观测".

这两样都只当**负面知识**用: 本模块只做纯观测 —— 给执行台账打一个粗粒度健康标签,
并在报告面提示"声称执行但查无成功回执"的疑似自演. 不改任何控制流、不设门禁、不
产生打勾义务, 与 ``claim_grounding`` / ``_citation_gap`` 同一"只标注"哲学.

口径说明 (避免误用):
  - ``exit_class`` 描述的是**执行本身是否健康**, 不是科学结论. 一个正常跑完、
    但实验自身判定 ``success=False``(负结果) 的运行仍是 ``ok`` —— 负结果由台账里
    的 ``success`` 字段承载, 不应被记成 ``tool_error``.
  - 分类是**启发式**(关键词/字段), 允许粗粒度; 因下游只作标注, 偶发错标不会改变
    任何决策. 宁粗勿假精确.

纯标准库, 零网络 / 零 LLM / 幂等 (同输入同输出), 可独立单测.
"""
from __future__ import annotations

import json
from typing import Any

# 执行健康标签 (粗粒度, 顺序即判定优先级)
EXIT_CLASSES = ("ok", "empty", "tool_error", "timeout", "env_error")

# 超时: 算力/预算问题, 不是假设被证伪.
_TIMEOUT_MARKERS = ("超时", "timeout", "timed out", "timeouterror")
# 环境/基建: 缺依赖/权限/隔离进程失败/内存封顶 —— 与命题无关, 不是科学结论.
_ENV_MARKERS = (
    "no module named",
    "modulenotfounderror",
    "importerror",
    "permission denied",
    "permissionerror",
    "command not found",
    "filenotfounderror",
    "隔离执行无结果",
    "隔离执行结果格式异常",
    "memory limit",
    "memlimit",
    "out of memory",
)
# 执行报错: 没有证据产出时的失败信号.
_ERROR_MARKERS = ("traceback", "exception", "error", "报错", "失败", "failed")

# 判定"是否真有结果产出"的字段 (命中任一非空即视为执行发生过).
_EVIDENCE_KEYS = (
    "objectives",
    "result",
    "summary",
    "value",
    "evidence",
    "data",
    "metrics",
)

# 报告正文里的"已执行/已观测"式陈述标记 (自演探针用).
_CLAIM_MARKERS_CJK = (
    "执行",
    "运行",
    "实测",
    "观测",
    "命令输出",
    "运行结果",
    "算出",
    "测得",
)
_CLAIM_MARKERS_EN = (
    "i ran",
    "we ran",
    "executed",
    "observed",
    "the output",
    "command output",
    "measured",
    "results show",
)


def _payload_text(output: Any) -> str:
    """把任意执行结果渲染成小写文本, 供关键词判定 (失败回退 str)."""
    if isinstance(output, dict):
        try:
            return json.dumps(output, ensure_ascii=False, default=str).lower()
        except Exception:  # 防御: 不可序列化 -> 退回 str
            return str(output).lower()
    return str(output).lower()


def _has_evidence(output: Any) -> bool:
    """执行结果里是否含有真实产出 (objectives/result/summary... 任一非空).

    仅对 dict 判定: 非 dict (裸字符串) 无法从"非空"断言真的发生了执行 —— 一句
    traceback 也是非空字符串, 认它当证据会把工具故障误标成 ok. 故裸字符串不作证据,
    交由下面的关键词判定 (timeout / env / error / empty).
    """
    if not isinstance(output, dict):
        return False
    for key in _EVIDENCE_KEYS:
        if output.get(key) not in (None, "", {}, [], ()):
            return True
    return False


def classify_exit_class(output: Any) -> str:
    """给一次执行结果打粗粒度健康标签 (纯观测, 见模块口径说明).

    优先级: timeout > env_error > ok(有产出) > tool_error > empty.
    "有产出"先于报错判定 —— 实验跑完但负结果仍是 ok, 不能记成工具故障.
    """
    text = _payload_text(output)
    if any(m in text for m in _TIMEOUT_MARKERS):
        return "timeout"
    has_evidence = _has_evidence(output)
    if not has_evidence and any(m in text for m in _ENV_MARKERS):
        return "env_error"
    if has_evidence:
        return "ok"
    if any(m in text for m in _ERROR_MARKERS):
        return "tool_error"
    return "empty"


def _ledger_exit_class(entry: Any) -> str:
    """取台账条目的 exit_class; 缺字段 (老台账/替身) 时按 result 现场判定.

    台账里的 ``result`` 是 JSON 文本 —— 先尝试还原成 dict, 否则会丢掉"有产出"
    这条信号 (一串 ``{"objectives":...}`` 会被当成普通字符串).
    """
    if not isinstance(entry, dict):
        return "empty"
    ec = entry.get("exit_class")
    if isinstance(ec, str) and ec in EXIT_CLASSES:
        return ec
    raw = entry.get("result", "")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # 非 JSON 文本 → 原样交关键词判定
            pass
    return classify_exit_class(raw)


def exit_class_counts(ledger: Any) -> dict[str, int]:
    """统计台账里各 exit_class 的条数 (只读, 供 run.log 触发率口径)."""
    counts = {c: 0 for c in EXIT_CLASSES}
    for e in ledger or []:
        counts[_ledger_exit_class(e)] += 1
    return counts


def detect_soliloquy(claimed_text: str, ledger: Any) -> dict[str, Any]:
    """检测"声称执行/观测, 但台账查无任何成功回执"的疑似自演 (EnIGMA soliloquizing).

    只标记不做判断: ``flagged`` 为真时上游也仅附一条诚实告示, 不阻断任何东西.
    ``receipts`` 只数 ``exit_class == "ok"`` 的条目 —— 超时/工具报错都不算"观测到了".
    """
    text = (claimed_text or "").lower()
    claimed = any(m in text for m in _CLAIM_MARKERS_CJK) or any(
        m in text for m in _CLAIM_MARKERS_EN
    )
    counts = exit_class_counts(ledger)
    receipts = counts["ok"]
    return {
        "claimed": bool(claimed),
        "receipts": receipts,
        "total": sum(counts.values()),
        "counts": counts,
        "flagged": bool(claimed and receipts == 0),
    }