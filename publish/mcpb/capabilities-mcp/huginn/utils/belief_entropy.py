"""Belief Entropy — 上下文压缩时的自检信号.

每次压缩对话历史时, 衡量模型对任务状态还有多"迷糊". 高熵 = 信息丢太多,
模型不确定任务进展; 低熵 = 压缩合理, 关键信息保留了.

三个测量维度:
1. H_logprob — 摘要 token 的归一化 Shannon 熵 (实时, 零额外调用)
2. C_fact    — 压缩后事实覆盖率 (1 次额外 LLM 调用, 可选)
3. R_loss    — 压缩比损失 (纯计算)

加权组合: H_b = α·H_logprob + β·(1-C_fact) + γ·R_loss

H_b 反馈到压缩策略: 高熵 → 下轮保守 (keep 更多消息), 低熵 → 更激进.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "BeliefEntropyConfig",
    "CompressionResult",
    "BeliefEntropy",
    "get_belief_entropy",
]


@dataclass
class BeliefEntropyConfig:
    """Belief Entropy 配置. 权重和阈值都可以调."""

    # 三个维度的权重, 和为 1
    weight_logprob: float = 0.5
    weight_fact: float = 0.3
    weight_ratio: float = 0.2

    # logprob 采样: 取 top-k 个 token 算熵
    logprob_top_k: int = 5

    # 事实覆盖检查: 最多抽多少个关键实体
    max_facts_to_check: int = 8

    # 自适应阈值
    threshold_low: float = 0.3   # 低于此值 → 压缩激进
    threshold_high: float = 0.7  # 高于此值 → 压缩保守

    # 自适应调整幅度
    keep_last_n_delta: int = 2    # 高熵时 keep_last_n 加多少
    budget_delta_ratio: float = 0.2  # 高熵时 budget 加多少比例

    # 是否启用事实覆盖检查 (需要额外 LLM 调用)
    fact_check_enabled: bool = False  # 默认关, 省调用

    # 历史窗口: 记最近 N 次的熵, 看趋势
    history_window: int = 10


@dataclass
class CompressionResult:
    """一次上下文压缩的结果, 带 Belief Entropy 信号."""

    summary: str = ""
    original_token_count: int = 0
    compressed_token_count: int = 0
    kept_messages: int = 0

    # Belief Entropy 三维度
    h_logprob: float = 0.0
    c_fact: float = 1.0   # 默认 1.0 = 全覆盖 (没做事实检查时)
    r_loss: float = 0.0
    h_belief: float = 0.0  # 加权组合后的总熵

    # 自适应建议
    adaptive_keep_last_n: int | None = None
    adaptive_budget_ratio: float | None = None

    # 调试信息
    facts_checked: list[str] = field(default_factory=list)
    facts_retained: list[str] = field(default_factory=list)


class BeliefEntropy:
    """计算 Belief Entropy 并给出自适应压缩建议.

    用法:
        be = get_belief_entropy()
        result = be.measure(
            summary="压缩后的摘要",
            original_tokens=5000,
            compressed_tokens=800,
            logprobs=response.logprobs,  # 可选
            original_context="原始上下文",  # 可选, 事实检查用
            model=model,  # 可选, 事实检查用
        )
        if result.h_belief > 0.7:
            # 模型很困惑, 下轮保留更多消息
            keep_last_n = result.adaptive_keep_last_n
    """

    def __init__(self, config: BeliefEntropyConfig | None = None) -> None:
        self._config = config or BeliefEntropyConfig()
        self._lock = threading.Lock()
        # 滑动窗口: 记最近 N 次的 h_belief, 看趋势
        self._history: list[float] = []

    @property
    def config(self) -> BeliefEntropyConfig:
        return self._config

    def measure(
        self,
        summary: str,
        original_tokens: int,
        compressed_tokens: int,
        logprobs: list[list[dict]] | None = None,
        original_context: str | None = None,
        model: Any = None,
    ) -> CompressionResult:
        """计算 Belief Entropy.

        params:
            summary: 压缩后的摘要文本
            original_tokens: 压缩前 token 数
            compressed_tokens: 压缩后 token 数
            logprobs: LLM 返回的 logprobs (每个 token 的 top-k 概率), 可选
            original_context: 原始上下文, 事实检查用, 可选
            model: LLM 实例, 事实检查用, 可选
        """
        cfg = self._config
        result = CompressionResult(
            summary=summary,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
        )

        # 1. R_loss: 压缩比损失 (永远能算, 最便宜)
        result.r_loss = self._compute_ratio_loss(original_tokens, compressed_tokens)

        # 2. H_logprob: token 级熵 (有 logprobs 才算)
        if logprobs is not None:
            result.h_logprob = self._compute_logprob_entropy(logprobs)
        else:
            # 没 logprobs 时, 用 summary 的信息密度做粗估:
            # 信息密度低 = 可能是泛泛而谈 = 熵高
            result.h_logprob = self._estimate_entropy_from_summary(summary)

        # 3. C_fact: 事实覆盖率 (需要额外 LLM 调用, 默认关)
        if cfg.fact_check_enabled and original_context and model is not None:
            result.c_fact, result.facts_checked, result.facts_retained = (
                self._compute_fact_coverage(original_context, summary, model)
            )
        else:
            # 长任务自动开启: 最近 3 次 h_belief 都 > threshold_high → 触发一次 fact check.
            # ponytail: 只在持续高熵时触发, 避免每次都调 LLM. 升级: 按任务类型差异化触发.
            with self._lock:
                recent_high = (
                    len(self._history) >= 3
                    and all(h > cfg.threshold_high for h in self._history[-3:])
                )
            if recent_high and original_context and model is not None:
                logger.info(
                    "belief_entropy: h_belief 持续高 (%.3f, %.3f, %.3f), 自动触发 fact_check",
                    *self._history[-3:]
                )
                result.c_fact, result.facts_checked, result.facts_retained = (
                    self._compute_fact_coverage(original_context, summary, model)
                )
            else:
                result.c_fact = 1.0  # 没检查就当全覆盖

        # 加权组合
        alpha, beta, gamma = cfg.weight_logprob, cfg.weight_fact, cfg.weight_ratio
        result.h_belief = (
            alpha * result.h_logprob
            + beta * (1.0 - result.c_fact)
            + gamma * result.r_loss
        )
        # clamp [0, 1]
        result.h_belief = max(0.0, min(1.0, result.h_belief))

        # 自适应建议
        result.adaptive_keep_last_n, result.adaptive_budget_ratio = (
            self._adaptive_params(result.h_belief)
        )

        # 记历史
        with self._lock:
            self._history.append(result.h_belief)
            if len(self._history) > cfg.history_window:
                self._history.pop(0)

        logger.debug(
            "belief entropy: H_b=%.3f (logprob=%.3f, fact=%.3f, ratio=%.3f), "
            "keep_last_n=%s, budget_ratio=%s",
            result.h_belief, result.h_logprob, result.c_fact, result.r_loss,
            result.adaptive_keep_last_n, result.adaptive_budget_ratio,
        )

        return result

    def _compute_ratio_loss(self, original: int, compressed: int) -> float:
        """压缩比损失: 压缩越狠损失越大.

        R_loss = 1 - min(1, compressed/original)
        完全不压缩 → 0, 压缩到 1/10 → 0.9
        """
        if original <= 0:
            return 0.0
        ratio = compressed / original
        return max(0.0, 1.0 - min(1.0, ratio))

    def _compute_logprob_entropy(self, logprobs: list[list[dict]]) -> float:
        """从 LLM logprobs 算归一化 Shannon 熵.

        logprobs 格式: [[{token: "...", logprob: -0.5}, ...], ...]
        每个 position 是 top-k 个 token 的概率分布.

        H = -Σ p_i log(p_i) / log(k), 归一化到 [0, 1]
        """
        k = self._config.logprob_top_k
        if k <= 1:
            return 0.0

        max_entropy = math.log(k)
        if max_entropy <= 0:
            return 0.0

        total_entropy = 0.0
        count = 0

        for position in logprobs:
            if not position:
                continue
            # 取 top-k, 算概率
            top_k = position[:k]
            probs = []
            for item in top_k:
                lp = item.get("logprob", -100.0)
                if isinstance(lp, (int, float)):
                    probs.append(math.exp(lp))
            if not probs:
                continue
            # 归一化概率 (top-k 不一定覆盖全部概率质量)
            total_p = sum(probs)
            if total_p <= 0:
                continue
            probs = [p / total_p for p in probs]
            # Shannon 熵
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            total_entropy += entropy
            count += 1

        if count == 0:
            return 0.0
        avg_entropy = total_entropy / count
        return min(1.0, avg_entropy / max_entropy)

    def _estimate_entropy_from_summary(self, summary: str) -> float:
        """没 logprobs 时的粗估: 信息密度越低, 熵越高.

        启发式: 名词/数字/术语密度高 = 信息密度高 = 低熵
        泛泛而谈 ("the user discussed various topics") = 高熵
        """
        if not summary or len(summary) < 10:
            return 0.5  # 空摘要, 不好说

        # 数数字、术语、实体
        numbers = len(re.findall(r'\d+\.?\d*', summary))
        # 大写开头的词 (可能是实体/术语)
        entities = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', summary))
        # 技术术语 (常见材料科学)
        terms = len(re.findall(
            r'\b(?:DFT|VASP|band\s*gap|POSCAR|CIF|dos|BZ|k-point|'
            r'convergence|relaxation|energy|structure|property|'
            r'material|calculation|simulation|lattice|parameter)\b',
            summary, re.I,
        ))
        # 总词数
        words = len(summary.split())

        if words == 0:
            return 0.5

        # 信息密度 = (数字 + 实体 + 术语) / 词数
        density = (numbers + entities + terms * 2) / max(words, 1)
        # density 高 → 熵低; density 低 → 熵高
        # 用 sigmoid 映射: density=0 → ~0.7, density=0.1 → ~0.5, density=0.3 → ~0.2
        entropy = 1.0 / (1.0 + math.exp(10.0 * (density - 0.1)))
        return min(1.0, max(0.0, entropy))

    def _compute_fact_coverage(
        self, original_context: str, summary: str, model: Any
    ) -> tuple[float, list[str], list[str]]:
        """事实覆盖率检查: 从原文抽关键事实, 看摘要保留了几个.

        返回 (覆盖率, 检查的事实列表, 保留的事实列表).
        需要额外 LLM 调用, 默认不启用.
        """
        # 从原文抽取关键实体和数字
        facts = self._extract_key_facts(original_context)
        if not facts:
            return 1.0, [], []

        # 用模型检查摘要里有没有提到这些事实
        retained = []
        for fact in facts:
            # 简单字符串匹配先过一遍
            if fact.lower() in summary.lower():
                retained.append(fact)
                continue

        # 字符串匹配没命中的, 可选调 LLM 做语义匹配
        unmatched = [f for f in facts if f not in retained]
        if unmatched and model is not None:
            try:
                llm_retained = self._llm_fact_check(summary, unmatched, model)
                retained.extend(llm_retained)
            except Exception:
                logger.debug("LLM fact check failed, using string match only")

        coverage = len(retained) / len(facts) if facts else 1.0
        return coverage, facts, retained

    def _extract_key_facts(self, context: str) -> list[str]:
        """从上下文抽取关键事实 (数字、实体、参数).

        不是 NER, 就是正则 + 频率. 够用就行.
        """
        facts: list[str] = []
        max_facts = self._config.max_facts_to_check

        # 数值参数 (e.g. "ENCUT = 520", "k-points: 4x4x4")
        for m in re.finditer(r'(\w+)\s*[=:]\s*(\d+\.?\d*)', context):
            facts.append(f"{m.group(1)}={m.group(2)}")
            if len(facts) >= max_facts:
                break

        # 化学式 (e.g. "Fe2O3", "LiCoO2")
        for m in re.finditer(r'\b([A-Z][a-z]?\d?)+\b', context):
            fact = m.group(0)
            if len(fact) > 2 and fact not in facts:
                facts.append(fact)
                if len(facts) >= max_facts:
                    break

        # 文件名 (e.g. "POSCAR", "INCAR", "CONTCAR")
        for m in re.finditer(r'\b(POSCAR|INCAR|CONTCAR|KPOINTS|OUTCAR)\b', context):
            if m.group(0) not in facts:
                facts.append(m.group(0))
                if len(facts) >= max_facts:
                    break

        return facts[:max_facts]

    def _llm_fact_check(
        self, summary: str, facts: list[str], model: Any
    ) -> list[str]:
        """让 LLM 判断摘要里是否包含了某些事实 (语义匹配)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        facts_str = "\n".join(f"  - {f}" for f in facts)
        messages = [
            SystemMessage(content=(
                "You are a fact checker. Given a summary and a list of facts, "
                "determine which facts are explicitly or implicitly mentioned "
                "in the summary. Output ONLY the fact strings that ARE present, "
                "one per line. If none are present, output nothing."
            )),
            HumanMessage(content=(
                f"Summary:\n{summary}\n\nFacts to check:\n{facts_str}\n\n"
                f"Which facts are present in the summary?"
            )),
        ]
        try:
            import asyncio
            try:
                asyncio.get_running_loop()
                resp = model.invoke(messages)
            except RuntimeError:
                resp = asyncio.run(model.ainvoke(messages))
            text = str(resp.content).strip()
        except Exception:
            return []

        # 按行解析, 返回命中的 fact
        retained = []
        for line in text.split("\n"):
            line = line.strip("- ").strip()
            for fact in facts:
                if fact.lower() in line.lower() or line.lower() in fact.lower():
                    retained.append(fact)
                    break
        return retained

    def _adaptive_params(self, h_belief: float) -> tuple[int | None, float | None]:
        """根据 Belief Entropy 给出自适应压缩建议.

        返回 (建议的 keep_last_n 调整, 建议的 budget 比例调整).
        None 表示不调整.
        """
        cfg = self._config
        if h_belief < cfg.threshold_low:
            # 低熵: 模型很清楚, 可以更激进
            return -1, 1.0 - cfg.budget_delta_ratio
        elif h_belief >= cfg.threshold_high:
            # 高熵: 模型很迷糊, 保守一点
            return cfg.keep_last_n_delta, 1.0 + cfg.budget_delta_ratio
        else:
            # 中熵: 不动
            return None, None

    def get_trend(self) -> float:
        """返回最近 N 次的熵趋势. 正数 = 熵在上升 (越来越迷糊), 负数 = 下降."""
        with self._lock:
            if len(self._history) < 2:
                return 0.0
            recent = self._history[-min(5, len(self._history)):]
            older = self._history[:-len(recent)] if len(self._history) > len(recent) else [0.5]
            avg_recent = sum(recent) / len(recent)
            avg_older = sum(older) / len(older) if older else 0.5
            return avg_recent - avg_older

    def get_history(self) -> list[float]:
        """返回历史 h_belief 序列 (副本)."""
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        """清空历史. 新会话时调."""
        with self._lock:
            self._history.clear()


# ── 单例 ──────────────────────────────────────────────────────

_singleton: BeliefEntropy | None = None
_singleton_lock = threading.Lock()


def get_belief_entropy() -> BeliefEntropy:
    """模块级单例. 配置通过环境变量可调."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                import os
                cfg = BeliefEntropyConfig(
                    fact_check_enabled=os.environ.get(
                        "HUGINN_BELIEF_ENTROPY_FACT_CHECK", "0"
                    ) == "1",
                    threshold_low=float(os.environ.get(
                        "HUGINN_BELIEF_ENTROPY_LOW", "0.3"
                    )),
                    threshold_high=float(os.environ.get(
                        "HUGINN_BELIEF_ENTROPY_HIGH", "0.7"
                    )),
                )
                _singleton = BeliefEntropy(cfg)
    return _singleton
