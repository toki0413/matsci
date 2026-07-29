"""Claim Strength 规则 + AI 腔词黑名单.

用于检查科研文本里的过度声明和 AI 模板腔, 把证据边界守住.
纯 stdlib, 不依赖任何外部库.
"""

from __future__ import annotations

import re

# ── Claim Strength 规则 ─────────────────────────────────────────────
# 每条规则描述一种"证据不够但话说满了"的典型模式.
# 写作时按证据强度降级, 不按领域套模板.

CLAIM_STRENGTH_RULES: list[str] = [
    "相关≠因果：相关性分析只能写'相关、提示、可能有关'，不能写'证明、导致、直接调控'。",
    "显著≠重要：没有统计检验、阈值或图表标注时，不写'显著'；可写'呈升高趋势、有所增加'。",
    "仿真≠验证：模型拟合、仿真结果不足以直接写'机制已揭示'；优先写'提示可能路径、为后续验证提供线索'。",
    "预测≠因果：预测准确不等于因果成立；指标提升不等于真实场景有效。",
    "共现≠互作：共表达/共现不等于分子间有互作；需功能验证或直接证据。",
    "变化≠增强：表达量/响应量变化不等于功能增强；量变不等于质变。",
    "样本内≠外推：样本内效果不等于外推成立；交叉验证不等于真实泛化。",
    "消融≠机制：消融实验显示贡献不等于揭示了机制；只是必要不是充分。",
    "强词需证据：'关键因子、核心机制、首次证明、系统揭示'等强词，只有证据链足够时才保留。",
    "单源≠定论：单一数据源/单一方法的结论不能直接写定论，需多源交叉印证。",
]

# ── AI 腔词黑名单 ───────────────────────────────────────────────────
# 这些词在科研正文里几乎不会自然出现, 命中基本可以判定是 AI 生成.

AI_BUZZWORD_BLACKLIST: list[str] = [
    "全面赋能", "深度融合", "闭环机制", "底座", "生态", "沉淀",
    "系统性提升", "智能化解决方案", "显著赋能", "阶段门", "赋能",
    "抓手", "闭环", "链路", "对标", "拉齐", "颗粒度", "打法",
    "心智", "势能", "复用",
]


def check_claim_strength(text: str) -> list[str]:
    """检查文本里是否有过度声明, 返回违规规则列表.

    匹配几种典型模式:
    - 相关性分析 + 强因果词 (证明/导致/直接调控)
    - 无检验语境下用'显著'
    - 仿真/模型 + '机制已揭示'
    - 强词: 首次证明/核心机制/系统揭示/关键因子
    """
    violations: list[str] = []

    # 相关性分析搭配强因果词
    if re.search(r"相关.{0,6}(证明|导致|直接调控|决定)", text):
        violations.append(CLAIM_STRENGTH_RULES[0])

    # 无统计语境的'显著'— 排除'显著差异/显著相关'这类有检验意味的用法
    # ponytail: 简单启发式, 只抓'显著提升/改善/增加'这类无 p 值支撑的
    if re.search(r"显著(提升|改善|增加|降低|提高|增强|优于)", text) and not re.search(
        r"(p\s*[<<=]|p值|显著差异|显著相关|统计学显著|statistically)", text
    ):
        violations.append(CLAIM_STRENGTH_RULES[1])

    # 仿真/模型 + 机制已揭示
    if re.search(r"(仿真|模拟|模型拟合).{0,10}(机制.{0,2}揭示|阐明|阐明)", text):
        violations.append(CLAIM_STRENGTH_RULES[2])

    # 共现 + 互作 (没提功能验证)
    if re.search(r"共(表达|现).{0,8}(互作|相互作用|调控)", text) and "功能验证" not in text:
        violations.append(CLAIM_STRENGTH_RULES[4])

    # 强词检查
    strong_words = ["首次证明", "核心机制", "系统揭示", "关键因子", "完全阐明"]
    for word in strong_words:
        if word in text:
            violations.append(CLAIM_STRENGTH_RULES[8])
            break

    return violations


def check_ai_buzzwords(text: str) -> list[str]:
    """检查文本里是否命中 AI 腔词黑名单, 返回命中的词列表."""
    return [w for w in AI_BUZZWORD_BLACKLIST if w in text]
