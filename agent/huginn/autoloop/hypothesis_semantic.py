"""P1#1: 假设维度 / 方法族 / 失败类型 关键词匹配 → LLM 语义判定 (优雅降级).

三处原为纯关键词 `in` 匹配, 非语义判定 (ROADMAP autoloop#1):
  1. 假设维度抽取 (composition/temperature/defect/structure/transport)
  2. 方法族归类 (ml-potential/symbolic-regression/.../dft-direct)
  3. 失败语义分类 (tool_error/param_error/data_noise/hypothesis_error)

本模块接入 LLM 语义判定, 并保证无 LLM / flag 关 / LLM 异常 / 输出非法标签时
降级回关键词匹配, 行为向后兼容 (现有测试零改动).

设计 (everything is a plugin):
- set_model_provider(fn): 引擎初始化时注入 `lambda: self.model`, 供延迟取用 —
  避免把 model 穿过所有 staticmethod 签名 (lazy 解析, 不 import 引擎).
- _llm_classify(prompt, allowed, fallback): 核心判定器 — LLM 返回的标签必须
  落在 allowed 集合内 (大小写/空白/标点宽容), 否则用 fallback.

不引入 embedding, 与 _metacog_classify_family 同范式 (ROADMAP P1#1 注意).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from huginn.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)

_FLAG = "hypothesis_llm_semantic"


def _enabled() -> bool:
    """LLM 语义判定开关 (默认关). 关闭时 classify_* 直接走 fallback, 零开销."""
    try:
        return FeatureFlags.shared().is_enabled(_FLAG)
    except Exception:
        return False


# ── lazy model provider (plugin 形态: 运行时注入, 不 import 引擎) ──────────
_provider: Callable[[], Any] | None = None


def set_model_provider(fn: Callable[[], Any] | None) -> None:
    """注册/清除 LLM model provider. 引擎初始化时注入 self.model."""
    global _provider
    _provider = fn


def _model() -> Any:
    if _provider is None:
        return None
    try:
        return _provider()
    except Exception:
        logger.debug("hypothesis_semantic model provider failed", exc_info=True)
        return None


def _llm_classify(
    prompt: str,
    allowed: tuple[str, ...] | list[str],
    fallback: Callable[[], str],
) -> str:
    """核心语义判定器: LLM 返回标签必须落在 allowed 内, 否则 fallback.

    关 flag / 无 model / 调用异常 / 标签非法 → 一律 fallback (优雅降级).
    """
    if not _enabled():
        return fallback()
    m = _model()
    if m is None:
        return fallback()
    try:
        resp = m.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        logger.debug("hypothesis_semantic: LLM invoke failed, fallback", exc_info=True)
        return fallback()
    label = (text or "").strip().strip('"\'`').strip().lower()
    if not label:
        return fallback()
    # 宽容匹配: 先看任一 allowed 成员作为子串 (容 "标签: xxx" / "xxx." 前缀),
    # 再按空白/逗号拆分精确比 (容 "xxx, 因为..." 后缀).
    for cand in allowed:
        if cand in label:
            return cand
    for token in label.replace(",", " ").split():
        token = token.strip().strip(".")
        if token in allowed:
            return token
    logger.debug("hypothesis_semantic: LLM label %r not in allowed %r", label, allowed)
    return fallback()


# ── 1. 维度抽取 ─────────────────────────────────────────────────────────────
# 原 hypothesis_loop._DIMENSION_KEYWORDS 迁入, 保持确定性 fallback.
_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "composition": ("ca/si", "ca_si", "al2o3", "掺杂", "doping", "alloy",
                    "composition", "ratio", "化学计量", "stoichiometry"),
    "temperature": ("温度", "temperature", "thermal", "退火", "annealing",
                    "t-dependent", "phase transition", "相变"),
    "defect": ("缺陷", "defect", "vacancy", "空位", "dislocation",
               "位错", "interface", "界面", "itz"),
    "structure": ("结构", "structure", "crystal", "晶体", "lattice",
                  "晶格", "symmetry", "对称", "phase", "相"),
    "transport": ("输运", "diffusion", "扩散", "conductivity",
                  "电导", "mobility", "迁移率", "percolation"),
}
_DIMENSION_LABELS = ("composition", "temperature", "defect", "structure", "transport")


def _fallback_dimension(statement: str) -> str:
    if not statement:
        return ""
    low = statement.lower()
    for dim, keywords in _DIMENSION_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                return dim
    return ""


def classify_dimension(statement: str) -> str:
    """从假设陈述判 dimension. 无 LLM / 关 flag 时回退关键词命中 (命中第一个)."""
    if not statement:
        return ""

    def fb() -> str:
        return _fallback_dimension(statement)

    prompt = (
        "一个材料/物理/化学研究假设, 判断它主要落在哪个核心维度, "
        "从以下标签选一个, 只输出标签本身, 不要解释:\n"
        "composition | temperature | defect | structure | transport\n\n"
        f"假设: {statement}\n\n标签:"
    )
    return _llm_classify(prompt, _DIMENSION_LABELS, fb)


# ── 2. 方法族归类 ───────────────────────────────────────────────────────────
# 原 hypothesis_loop._metacog_classify_family 规则迁入.
_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ml-potential", ("mlp", "ml potential", "machine learning potential", "neural potential")),
    ("symbolic-regression", ("symbolic", "symreg", "siprend", "解析式")),
    ("gaussian-process", ("gp ", "gaussian process", "gpr", "核函数")),
    ("calphad-thermo", ("calphad", "相图", "phase diagram", "thermodynamic")),
    ("phase-field", ("phase field", "相场")),
    ("bourbaki-structure", ("bourbaki", "格论", "lattice theory", "拓扑")),
    ("extreme-argument", ("反例", "counterexample", "extreme", "极值")),
    ("computational-check", ("benchmark", "计算验证", "computational check")),
    ("dft-direct", ("dft", "第一性原理", "ab initio", "vasp", "qe", "cp2k")),
)
_FAMILY_LABELS = tuple(f for f, _ in _FAMILY_RULES)


def _fallback_family(hypothesis: str) -> str:
    text = (hypothesis or "").lower()
    for family, keywords in _FAMILY_RULES:
        if any(kw in text for kw in keywords):
            return family
    return "dft-direct"  # 默认族


def classify_family(hypothesis: str) -> str:
    """把假设归到方法族 (用于 method_registry 收敛度监控). 分类不准不致命."""
    if not hypothesis:
        return _fallback_family("")

    def fb() -> str:
        return _fallback_family(hypothesis)

    prompt = (
        "一个科学假设使用了哪种计算方法族, 从以下标签选一个, 只输出标签本身, "
        "不要解释:\n"
        + " | ".join(_FAMILY_LABELS)
        + "\n\n"
        f"假设: {hypothesis}\n\n标签:"
    )
    return _llm_classify(prompt, _FAMILY_LABELS, fb)


# ── 3. 失败语义分类 ─────────────────────────────────────────────────────────
# 只用于 _classify_failure 关键词链的 ambiguous 默认分支 (见 hypothesis_loop).
_FAILURE_LABELS = ("tool_error", "param_error", "data_noise", "hypothesis_error")


def classify_failure(text: str) -> str:
    """LLM 判定失败语义; 无 LLM / 非法输出时回退 hypothesis_error (原默认)."""

    def fb() -> str:
        return "hypothesis_error"

    if not text:
        return fb()
    prompt = (
        "一段工具/实验失败信息, 判断失败类型, 从以下标签选一个, 只输出标签本身, "
        "不要解释:\n"
        "tool_error(工具崩溃/超时/连接失败, 非假设问题) | "
        "param_error(输入参数错) | "
        "data_noise(结果不确定/噪声大) | "
        "hypothesis_error(结果与假设相反/假设本身错)\n\n"
        f"失败信息: {text[:800]}\n\n标签:"
    )
    return _llm_classify(prompt, _FAILURE_LABELS, fb)