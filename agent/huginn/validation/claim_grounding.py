"""结论证伪门禁 (Claim-Grounding Gate) — "任何数值必须落在真实工具结果里".

补上 Huginn 缺位的最后一环: 现有 `completion_evidence` / `assess_gate` 管的是
"检索到的数值能否溯到来源" 与 "缺失自由度是否显式豁免" —— 但都**不校验模型合成结论
(报告 prose) 里出现的统计量是否真的来自已执行的工具输出**。这正是 LLM 编造数字的
重灾区 (面对大段开放任务时, 模型会"脑内模拟"并输出看似合理的 AIC/CI/参数)。

本模块是与 `assess_gate` 同一哲学 (AICC validate_gate: pass 禁带未落地的主张) 的
一个薄扩展, 但**只锁结论、不动方法**:

  - 方法自由: 不限定模型"必须走哪几步、调哪个工具、用什么参数"。
  - 结论证伪: 在模型交付最终报告时, 把报告中每个候选数值与「工具执行轨迹」比对;
    在轨迹里 → 放行; 不在 → 记为 unsubstantiated, 门禁返回 needs_grounding,
    由上层决定「要求模型删除该主张」或「让模型补跑工具以溯源」。

纯标准库, 零网络 / 零 LLM / 幂等 (同输入同输出), 可独立单测。
"""
from __future__ import annotations

import math
import re
from typing import Any

# 候选数值: 带小数的统计量 + 大整数(数据点坐标/推荐采样点等, 通常 abs>100)
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
# 阈值: 纯小整数(0~100 的整数)通常是无意义计数/比例, 不作为"主张数值"
_MAX_SMALL_ABS = 100
_MATCH_TOL_2 = 0.03   # 2 位小数匹配容差
_MATCH_TOL_1 = 0.05   # 1 位小数匹配容差


def _canon(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_numbers(text: str) -> list[float]:
    """抽取出 text 中所有数值 (原样 float)."""
    return [c for m in _NUM.findall(text or "") if (c := _canon(m)) is not None]


def _is_claim_candidate(token: str, value: float) -> bool:
    """候选主张数值: 带小数 或 绝对值 > _MAX_SMALL_ABS 的整数/负数."""
    f = abs(value)
    if "." in token:
        return True
    return f > _MAX_SMALL_ABS


def _strip_num_markers(text: str) -> str:
    """剔除 markdown 的小节号/有序列表编号等『编号型数字』(如 2.2 / 3.4 / 1).

    只处理行首的编号标记, 不影响正文中的真实数值.
    """
    import re as _re
    lines = []
    for ln in (text or "").splitlines():
        s = _re.sub(r"^\s*#{1,6}\s*", "", ln)          # 去掉 ATX 标题井号
        s = _re.sub(r"^\s*(\d+(?:\.\d+)*)\s*(?:[.:、，]\s*|\s(?![0-9]))", "", s, count=1)
        lines.append(s)
    return "\n".join(lines)


def _strip_ordinal_markers(text: str) -> str:
    """剔除『轮次/周期编号』型数字 (第238轮 / cycle 242 / 轮次 12 / 第 N 轮).

    长程自主管线每轮报告常带轮次编号, 是编排索引而非统计主张;
    与 `_strip_num_markers` 的小节号剔除同哲学 —— 不锁真实数值主张.
    """
    import re as _re
    s = _re.sub(r"第\s*\d+\s*轮", "", text or "")
    s = _re.sub(r"cycle\s*\d+", "", s, flags=_re.IGNORECASE)
    s = _re.sub(r"轮次\s*\d+", "", s)
    # 实验分支名 author_c{cycle} / author_cNNN: 命名标识符, 不是统计主张.
    # (trace 只含结果值, 不含实验名 → 否则兜底报告里分支名必被判未落地.)
    s = _re.sub(r"\bauthor_c\d+\b", "", s)
    s = _re.sub(r"\bS\d+_scan\b", "", s)
    return s


def extract_numeric_claims(text: str) -> list[float]:
    """从报告 prose 中抽出"候选主张数值" (统计量/坐标等, 桥掉无意义小整数)."""
    text = _strip_num_markers(text)   # 去小节号等编号型数字
    text = _strip_ordinal_markers(text)   # 去轮次编号等编排索引
    out: list[float] = []
    for m in _NUM.findall(text or ""):
        c = _canon(m)
        if c is None:
            continue
        if _is_claim_candidate(m, c):
            out.append(c)
    return out


def build_evidence_index(trace: list[str]) -> tuple[set[float], set[float]]:
    """把工具返回轨迹里的全部数值做成 1/2 位小数索引 (用于近邻匹配)."""
    e1: set[float] = set()
    e2: set[float] = set()
    for block in trace:
        for c in collect_numbers(block):
            e2.add(round(c, 2))
            e1.add(round(c, 1))
    return e1, e2


def _num_near(claim: float, e1: set[float], e2: set[float]) -> bool:
    r2 = round(claim, 2)
    if any(abs(r2 - v) <= _MATCH_TOL_2 for v in e2):
        return True
    r1 = round(claim, 1)
    if any(abs(r1 - v) <= _MATCH_TOL_1 for v in e1):
        return True
    # 小整数 (0~100) 若撞上轨迹里的普通整数也算落地 (如候选采样点 100)
    if 0 <= abs(claim) <= _MAX_SMALL_ABS and claim == round(claim):
        return any(abs(claim - v) <= 0.5 for v in e2)
    return False


def _derived_from(pool: list[float], c: float, tol: float = 0.05) -> bool:
    """判定 c 能否由轨迹中两个真实值经 +/−/×/÷ 推出 (允许的『衍生分析』).

    仅当两个操作数都真实出现在轨迹里才成立 → 仍然是可证伪的 (来自真值, 非编造).
    采用相对容差: 对较大数值 (如比值 157) 允许比例误差, 兼顾整数值取整.
    """
    tol = max(tol, 0.02 * abs(c))
    n = len(pool)
    for ia in range(n):
        a = pool[ia]
        for ib in range(n):
            b = pool[ib]
            cands = [a + b, a - b, b - a, a * b]
            if abs(b) > 1e-9:
                cands.append(a / b)
            if abs(a) > 1e-9:
                cands.append(b / a)
            for d in cands:
                if abs(d - c) <= tol:
                    return True
    return False


def verify_claims(
    final_text: str,
    tool_trace: list[str],
    *,
    include_all_numbers: bool = False,
    allow_derived: bool = False,
) -> dict[str, Any]:
    """结论证伪门禁主函数.

    Args:
        final_text: 模型最终报告 (prose)。
        tool_trace: 工具执行轨迹的返回串列表 (每个元素是一次工具结果)。
        include_all_numbers: True 时连无意义小整数也算主张 (更严格, 用于调试)。
        allow_derived: True 时, 未直接出现在轨迹中的数值, 若能由轨迹中两个真实值
            经 +/−/×/÷ 推出, 也视为已落地 (解锁『衍生分析』深度, 仍可证伪)。

    Returns:
        {"matched", "derived", "unsubstantiated", "verdict", "note"}
        verdict ∈ {"pass", "needs_grounding"}。
    """
    claims = extract_numeric_claims(final_text)
    if include_all_numbers:
        claims = collect_numbers(final_text)
    e1, e2 = build_evidence_index(tool_trace or [])
    # 衍生分析用 4 位小数的真实数值池 (保留精度, 否则比值/乘积失真)
    pool = sorted({round(c, 4) for block in (tool_trace or []) for c in collect_numbers(block)})
    matched: list[float] = []
    derived: list[float] = []
    unsubstantiated: list[float] = []
    for c in claims:
        if _num_near(c, e1, e2):
            matched.append(c)
        elif allow_derived and _derived_from(pool, c):
            derived.append(c)
        else:
            unsubstantiated.append(c)
    ok = not unsubstantiated
    return {
        "matched": [round(c, 3) for c in sorted(matched)],
        "derived": [round(c, 3) for c in sorted(derived)],
        "unsubstantiated": [round(c, 3) for c in sorted(unsubstantiated)],
        "verdict": "pass" if ok else "needs_grounding",
        "note": (
            "结论中每个数值均可回溯到工具执行轨迹 (含通过真实轨迹值推出的衍生量); "
            f"匹配 {len(matched)}, 衍生 {len(derived)}。"
            if ok else
            "以下数值不在工具执行轨迹中, 也无法由轨迹真值 +/−/×/÷ 推出, 疑似未落地主张: "
            f"{[round(c, 3) for c in sorted(unsubstantiated)]}。请删除主张或补跑工具使数值溯源。"
        ),
    }


__all__ = [
    "build_evidence_index",
    "collect_numbers",
    "extract_numeric_claims",
    "verify_claims",
]