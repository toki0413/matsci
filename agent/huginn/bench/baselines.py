"""竞品 baseline 得分对标 (对标社区 AI Scientist 评测).

数据来源: 各 benchmark 官方论文/leaderboard (2025-2026).
用于和我们的 Huginn agent 得分横向对比.

竞品得分表 (按 benchmark 分):
  - PaperBench (20篇, 8316 rubric, 12h/篇): Claude 3.5 Sonnet 21.0%, o1 26.6%, 人类 41.4%
  - MLE-bench (75题, Kaggle奖牌): o1+AIDE 16.9% 铜牌, 7 金牌
  - RCBench (多方向, 多trial): Open Science Desktop 22.8, Claude Code 21.5, Codex CLI 18.4
  - MMMU (11500题, 多选): Gemini Ultra 59%, GPT-4V 56%
  - GPQA Diamond (~429题, 多选): 博士 65%, GPT-4 39%
  - HLE (~2500题, 多选): Gemini 3 Flash 33.7%, GPT-5 25.3%
  - AI Scientist v2: ICLR workshop 6.33 分 (超人类均分 4.87)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineScore:
    """单个竞品在某个 benchmark 上的得分."""
    agent: str           # 竞品名称
    benchmark: str       # benchmark 名
    score: float         # 得分 (百分比或绝对分)
    score_type: str      # "percent" / "score" / "medal"
    notes: str = ""      # 备注


# 社区竞品 baseline (2025-2026 公开数据)
BASELINES: list[BaselineScore] = [
    # PaperBench (论文复现, 20篇 ICML)
    BaselineScore("Claude 3.5 Sonnet", "PaperBench", 21.0, "percent", "开源框架, 12h/篇"),
    BaselineScore("o1", "PaperBench", 26.6, "percent", "3篇子集"),
    BaselineScore("Human (best of 3)", "PaperBench", 41.4, "percent", "48h, 3篇子集"),

    # MLE-bench (Kaggle 竞赛复现, 75题)
    BaselineScore("o1 + AIDE", "MLE-bench", 16.9, "percent", "铜牌率, 7金牌"),

    # ResearchClawBench (AI4S 全流程)
    BaselineScore("Open Science Desktop", "RCBench", 22.8, "score", "榜首, 2026.07"),
    BaselineScore("Claude Code", "RCBench", 21.5, "score", ""),
    BaselineScore("Codex CLI", "RCBench", 18.4, "score", ""),

    # MMMU (多模态知识, 11500题)
    BaselineScore("Gemini Ultra", "MMMU", 59.0, "percent", ""),
    BaselineScore("GPT-4V", "MMMU", 56.0, "percent", ""),

    # GPQA Diamond (博士级, ~429题)
    BaselineScore("PhD experts", "GPQA", 65.0, "percent", ""),
    BaselineScore("GPT-4", "GPQA", 39.0, "percent", ""),

    # HLE (极难, ~2500题)
    BaselineScore("Gemini 3 Flash", "HLE", 33.7, "percent", "2025 数据"),
    BaselineScore("GPT-5", "HLE", 25.3, "percent", "2025 数据"),

    # AI Scientist v2 (端到端科研)
    BaselineScore("AI Scientist v2", "ICLR-workshop", 6.33, "score", "超人类均分 4.87"),
    BaselineScore("AutoResearchClaw", "ARC-Bench", 54.7, "percent", "实验阶段"),
]


# 我们的 agent 对标映射 (我们的 suite -> 社区 benchmark)
SUITE_TO_BENCHMARK = {
    "general": "MMMU",       # 知识推理 -> MMMU/GPQA
    "physics": "PaperBench", # 数值物理 -> PaperBench (简化版)
    "lineage": "ARC-Bench",  # 谱系能力 -> ARC-Bench
    "repro": "PaperBench",   # 论文复现 -> PaperBench
    "optim": "MLE-bench",    # 算法优化 -> MLE-bench
    "research": "RCBench",   # 研究场景 -> RCBench
}


def get_baselines_for_suite(suite: str) -> list[BaselineScore]:
    """按我们的 suite 名返回对应的社区竞品 baseline."""
    bench = SUITE_TO_BENCHMARK.get(suite, "")
    return [b for b in BASELINES if b.benchmark == bench]


def format_baseline_table(suite: str, our_score: float, our_score_type: str = "percent") -> str:
    """格式化 baseline 对比表."""
    baselines = get_baselines_for_suite(suite)
    lines = [f"\n=== {suite} suite vs 社区竞品 ({SUITE_TO_BENCHMARK.get(suite, '?')}) ==="]
    lines.append(f"  Huginn (our):  {our_score:.1f}%")
    for b in baselines:
        unit = "%" if b.score_type == "percent" else "分"
        lines.append(f"  {b.agent:<25} {b.score:.1f}{unit}  {b.notes}")
    return "\n".join(lines)
