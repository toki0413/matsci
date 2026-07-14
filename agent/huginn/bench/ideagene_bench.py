"""IdeaGene-Bench 材料版 — 科研谱系能力评测.

受上海交大 IG-Bench (arXiv:2607.08758) 启发, 把材料科学方法演化建模为
可审计的"想法基因"谱系, 评测 agent 的谱系推理能力.

核心概念:
- Idea Genome: 方法的最小可继承零件 (驱动机制, 不是方法名)
- GenomeDiff: 亲子鉴定 — 驱动机制有没有被继承
- 演化动力学: 突变/适应辐射/杂交/物种形成 (算谱系) vs 生态位竞争/隔离 (不算)

四能力轴 (IG-Exam):
- T1 基因抽象: 读懂单个方法的驱动机制
- T2 继承追踪: 跨方法对齐基因
- T3 演化推理: 判断动力学类型
- T4 谱系验证: 抓内鬼/找错步/补缺失链

PES (Parent-Embedded Score) 三维:
- Heredity (遗传): 新方法是否继承了前作驱动机制
- Variation (变异): 新方法是否有新关键词
- Selection (选择): 新方法是否解决前作已知缺陷

材料科学谱系: DFT 交换关联泛函演化
  LDA → GGA → DFT+U → HSE06 → GW → BSE
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from huginn.bench.task import BenchmarkTask


# ── 材料科学方法谱系数据 ────────────────────────────────────────
# 每个方法的"基因": 驱动机制 + 生态位 + 已知缺陷


@dataclass
class MethodGene:
    """方法的想法基因 — 可审计的最小可继承零件."""

    name: str
    year: int
    driving_mechanism: str  # 核心驱动机制 (可继承)
    mechanism_keywords: list[str]  # 关键词 (用于 PES 评分)
    ecology: str  # 生态位 (任务/benchmark, 非血缘)
    known_gap: str  # 已知缺陷 (选择压力)


# DFT 交换关联泛函演化谱系
DFT_LINEAGE: list[MethodGene] = [
    MethodGene(
        name="LDA", year=1965,
        driving_mechanism="局域密度近似: 用均匀电子气的交换关联能近似非均匀系统",
        mechanism_keywords=["uniform electron gas", "local density", "homogeneous"],
        ecology="凝聚态电子结构计算",
        known_gap="无法处理非均匀电子气, 自相互作用误差",
    ),
    MethodGene(
        name="GGA", year=1992,
        driving_mechanism="广义梯度近似: 在 LDA 基础上加入电子密度梯度修正",
        mechanism_keywords=["gradient", "density gradient", "semilocal"],
        ecology="凝聚态电子结构计算",
        known_gap="仍低估带隙, 强关联系统失效",
    ),
    MethodGene(
        name="DFT+U", year=1994,
        driving_mechanism="在 GGA 基础上对局域 d/f 电子加 Hubbard U 修正",
        mechanism_keywords=["Hubbard U", "localized", "on-site Coulomb", "d electron", "f electron"],
        ecology="强关联电子系统 (过渡金属氧化物, 稀土)",
        known_gap="U 值经验依赖, 非局域关联仍缺失",
    ),
    MethodGene(
        name="HSE06", year=2003,
        driving_mechanism="杂化泛函: 混合精确交换 (Hartree-Fock) 与 GGA 交换, 含屏蔽",
        mechanism_keywords=["exact exchange", "hybrid", "screening", "range-separated", "Hartree-Fock"],
        ecology="凝聚态电子结构计算 (带隙预测)",
        known_gap="计算成本高, 金属系统屏蔽参数敏感",
    ),
    MethodGene(
        name="GW", year=1965,
        driving_mechanism="多体微扰理论: 用 Green 函数 G 和 screened Coulomb W 计算自能修正",
        mechanism_keywords=["self-energy", "Green function", "screened Coulomb", "many-body perturbation", "quasiparticle"],
        ecology="凝聚态准粒子能量计算",
        known_gap="不包含激子效应, 计算成本极高",
    ),
    MethodGene(
        name="BSE", year=1965,
        driving_mechanism="Bethe-Salpeter 方程: 在 GW 基础上包含电子-空穴相互作用 (激子)",
        mechanism_keywords=["exciton", "electron-hole", "Bethe-Salpeter", "screened interaction", "optical"],
        ecology="光学性质计算 (吸收谱, 激子结合能)",
        known_gap="计算量极大, 仅限小系统",
    ),
]


# ── PES 评分函数 ────────────────────────────────────────────────


def compute_pes(
    generated_text: str,
    parent: MethodGene,
    grandparent: MethodGene | None = None,
) -> dict[str, float]:
    """计算 Parent-Embedded Score (PES) 三维分数.

    Heredity (遗传): 新方法是否包含 parent 的驱动机制关键词
    Variation (变异): 新方法是否有 parent 没有的新关键词
    Selection (选择): 新方法是否提到 parent 的 known_gap

    返回 {"heredity": 0-1, "variation": 0-1, "selection": 0-1, "pes": 加权}
    """
    text_lower = generated_text.lower()

    # Heredity: parent 关键词命中率
    parent_hits = sum(1 for kw in parent.mechanism_keywords if kw.lower() in text_lower)
    heredity = parent_hits / len(parent.mechanism_keywords) if parent.mechanism_keywords else 0.0

    # Variation: 新关键词 (不在 parent 里的)
    all_parent_kws = set(kw.lower() for kw in parent.mechanism_keywords)
    if grandparent:
        all_parent_kws |= set(kw.lower() for kw in grandparent.mechanism_keywords)
    # 简化: 检查文本里是否有新材料科学关键词
    new_kws = ["machine learning", "neural", "surrogate", "embedding", "graph neural",
               "transfer learning", "active learning", "bayesian", "gaussian"]
    variation_hits = sum(1 for kw in new_kws if kw in text_lower)
    variation = min(1.0, variation_hits / 3.0)  # 3 个新关键词满分

    # Selection: 是否提到 parent 的 known_gap
    gap_kws = [w.lower() for w in re.findall(r"[a-zA-Z]+", parent.known_gap) if len(w) > 3]
    gap_hits = sum(1 for kw in gap_kws if kw in text_lower)
    selection = min(1.0, gap_hits / 3.0) if gap_kws else 0.0

    # PES 加权: Heredity 0.5 + Variation 0.3 + Selection 0.2 (论文发现 Heredity 是关键)
    pes = 0.5 * heredity + 0.3 * variation + 0.2 * selection

    return {
        "heredity": round(heredity, 3),
        "variation": round(variation, 3),
        "selection": round(selection, 3),
        "pes": round(pes, 3),
    }


# ── T1 基因抽象任务 ─────────────────────────────────────────────


def _t1_extract_mechanism(task_id: str, gene: MethodGene) -> BenchmarkTask:
    """T1: 给 agent 方法描述, 让它提取驱动机制."""
    prompt = (
        f"材料科学计算方法 {gene.name} (提出于 {gene.year} 年) 的核心驱动机制是什么？\n"
        f"生态位: {gene.ecology}\n"
        f"请用 2-3 句话说明该方法区别于其他方法的核心创新点。\n"
        f"不要只是描述方法名称, 要说明驱动机制 — 即'这个方法为什么能工作'。"
    )

    def evaluator(output: str) -> tuple[bool, str, float]:
        text = output.lower()
        hits = sum(1 for kw in gene.mechanism_keywords if kw.lower() in text)
        ratio = hits / len(gene.mechanism_keywords) if gene.mechanism_keywords else 0
        passed = ratio >= 0.34  # 至少命中 1/3 关键词
        reason = f"命中 {hits}/{len(gene.mechanism_keywords)} 关键词"
        return passed, reason, round(ratio, 3)

    return BenchmarkTask(
        id=task_id,
        category="ideagene_t1",
        prompt=prompt,
        evaluator=evaluator,
        tags=["ideagene", "t1", "gene_abstraction"],
    )


# ── T4 谱系验证任务 ─────────────────────────────────────────────


def _t4_verify_lineage(task_id: str, parent: MethodGene, child: MethodGene,
                       is_lineage: bool, correct_dynamics: str) -> BenchmarkTask:
    """T4: 给 agent 两个方法, 让它判断是否是谱系关系."""
    prompt = (
        f"判断以下两个材料科学计算方法之间是否存在驱动机制继承关系:\n"
        f"方法 A: {parent.name} ({parent.year}) — {parent.driving_mechanism}\n"
        f"方法 B: {child.name} ({child.year}) — {child.driving_mechanism}\n\n"
        f"请回答:\n"
        f"1. B 是否继承了 A 的驱动机制？(是/否)\n"
        f"2. 如果是, 演化动力学类型是什么？(突变/适应辐射/杂交/物种形成)\n"
        f"3. 如果否, 关系是什么？(生态位竞争/隔离)\n"
        f"请简要说明理由。"
    )

    def evaluator(output: str) -> tuple[bool, str, float]:
        text = output.lower()
        # 检查是否正确判断了谱系关系
        if is_lineage:
            # 应该判断为"是"
            inherited = any(w in text for w in ["是", "继承", "yes", "inherited", "lineage"])
            # 检查动力学类型
            dynamics_correct = correct_dynamics.lower() in text
            score = (0.5 if inherited else 0.0) + (0.5 if dynamics_correct else 0.0)
            passed = inherited and dynamics_correct
            reason = f"谱系判断={'正确' if inherited else '错误'}, 动力学={'正确' if dynamics_correct else '错误'}"
        else:
            # 应该判断为"否"
            not_inherited = any(w in text for w in ["否", "不是", "no", "not inherited", "ecological", "隔离"])
            score = 1.0 if not_inherited else 0.0
            passed = not_inherited
            reason = f"非谱系判断={'正确' if not_inherited else '错误'}"
        return passed, reason, round(score, 3)

    return BenchmarkTask(
        id=task_id,
        category="ideagene_t4",
        prompt=prompt,
        evaluator=evaluator,
        tags=["ideagene", "t4", "lineage_verification"],
    )


# ── PES 生成任务 (IG-Arena 简化版) ──────────────────────────────


def _arena_generate(task_id: str, parent: MethodGene, grandparent: MethodGene) -> BenchmarkTask:
    """IG-Arena: 给 agent 谱系上下文, 让它生成下一个方法."""
    prompt = (
        f"材料科学计算方法的演化谱系如下:\n"
        f"  {grandparent.name} ({grandparent.year}): {grandparent.driving_mechanism}\n"
        f"  → {parent.name} ({parent.year}): {parent.driving_mechanism}\n"
        f"  已知缺陷: {parent.known_gap}\n\n"
        f"请提出下一个方法 (谱系后代), 要求:\n"
        f"1. 继承 {parent.name} 的核心驱动机制\n"
        f"2. 解决已知缺陷: {parent.known_gap}\n"
        f"3. 说明新方法相比 {parent.name} 的变异点\n"
        f"用 3-5 句话描述。"
    )

    def evaluator(output: str) -> tuple[bool, str, float]:
        pes = compute_pes(output, parent, grandparent)
        # PES >= 0.4 算通过 (论文发现最强模型 PES 约 83.6/100)
        passed = pes["pes"] >= 0.4
        reason = f"H={pes['heredity']} V={pes['variation']} S={pes['selection']} PES={pes['pes']}"
        return passed, reason, pes["pes"]

    return BenchmarkTask(
        id=task_id,
        category="ideagene_arena",
        prompt=prompt,
        evaluator=evaluator,
        tags=["ideagene", "arena", "pes"],
    )


# ── 任务集构建 ──────────────────────────────────────────────────


def build_ideagene_tasks() -> list[BenchmarkTask]:
    """构建 IdeaGene-Bench 材料版任务集."""
    tasks: list[BenchmarkTask] = []

    # T1 基因抽象: 每个方法一题
    for i, gene in enumerate(DFT_LINEAGE):
        tasks.append(_t1_extract_mechanism(f"ideagene_t1_{gene.name.lower()}", gene))

    # T4 谱系验证: 相邻方法 + 跨方法
    # LDA→GGA: 突变 (驱动机制继承+局部改动)
    tasks.append(_t4_verify_lineage("ideagene_t4_lda_gga", DFT_LINEAGE[0], DFT_LINEAGE[1],
                                    is_lineage=True, correct_dynamics="突变"))
    # GGA→DFT+U: 适应辐射 (驱动机制保留+新生态位)
    tasks.append(_t4_verify_lineage("ideagene_t4_gga_dftu", DFT_LINEAGE[1], DFT_LINEAGE[2],
                                    is_lineage=True, correct_dynamics="适应辐射"))
    # DFT+U→HSE06: 突变 (驱动机制局部改动)
    tasks.append(_t4_verify_lineage("ideagene_t4_dftu_hse06", DFT_LINEAGE[2], DFT_LINEAGE[3],
                                    is_lineage=True, correct_dynamics="突变"))
    # HSE06→GW: 物种形成 (驱动机制被全新机制取代)
    tasks.append(_t4_verify_lineage("ideagene_t4_hse06_gw", DFT_LINEAGE[3], DFT_LINEAGE[4],
                                    is_lineage=True, correct_dynamics="物种形成"))
    # GW→BSE: 突变 (在 GW 基础上加激子)
    tasks.append(_t4_verify_lineage("ideagene_t4_gw_bse", DFT_LINEAGE[4], DFT_LINEAGE[5],
                                    is_lineage=True, correct_dynamics="突变"))
    # LDA vs GW: 隔离 (不同驱动机制, 不同生态位)
    tasks.append(_t4_verify_lineage("ideagene_t4_lda_gw", DFT_LINEAGE[0], DFT_LINEAGE[4],
                                    is_lineage=False, correct_dynamics="隔离"))

    # IG-Arena: 生成任务
    tasks.append(_arena_generate("ideagene_arena_gga_to_next", DFT_LINEAGE[1], DFT_LINEAGE[0]))
    tasks.append(_arena_generate("ideagene_arena_hse06_to_next", DFT_LINEAGE[3], DFT_LINEAGE[2]))
    tasks.append(_arena_generate("ideagene_arena_gw_to_next", DFT_LINEAGE[4], DFT_LINEAGE[3]))

    return tasks


# ── 自检 ────────────────────────────────────────────────────────


def _selfcheck():
    """IdeaGene-Bench 自检 — 不需要真 LLM, 只测 evaluator 逻辑."""
    tasks = build_ideagene_tasks()

    # T1: 正确答案应通过
    t1 = tasks[0]  # LDA
    correct_answer = "LDA 的核心驱动机制是利用 uniform electron gas 的 local density 近似, 假设电子气是 homogeneous 的"
    result = t1.evaluate(correct_answer)
    assert result.passed, f"T1 LDA 正确答案应通过: {result.reason}"
    assert result.score is not None and result.score >= 0.34, f"T1 分数应 >= 0.34, got {result.score}"

    # T1: 错误答案不应通过
    wrong_answer = "LDA 是一种计算方法"
    result = t1.evaluate(wrong_answer)
    assert not result.passed, "T1 LDA 错误答案不应通过"

    # T4: 正确判断谱系
    t4_lda_gga = next(t for t in tasks if t.id == "ideagene_t4_lda_gga")
    correct_t4 = "是的, GGA 继承了 LDA 的驱动机制, 属于突变. GGA 保留了 LDA 的均匀电子气近似基础, 加入了密度梯度修正."
    result = t4_lda_gga.evaluate(correct_t4)
    assert result.passed, f"T4 LDA→GGA 正确判断应通过: {result.reason}"

    # T4: 错误判断
    wrong_t4 = "不是, 两者没有关系"
    result = t4_lda_gga.evaluate(wrong_t4)
    assert not result.passed, "T4 LDA→GGA 错误判断不应通过"

    # T4: 非谱系 (LDA vs GW)
    t4_lda_gw = next(t for t in tasks if t.id == "ideagene_t4_lda_gw")
    correct_isolation = "否, LDA 和 GW 没有驱动机制继承. LDA 用 local density 近似, GW 用 Green 函数和 screened Coulomb. 属于隔离."
    result = t4_lda_gw.evaluate(correct_isolation)
    assert result.passed, f"T4 LDA vs GW 隔离判断应通过: {result.reason}"

    # PES: 高遗传答案
    arena = next(t for t in tasks if t.id == "ideagene_arena_gga_to_next")
    high_heredity = "在 GGA 的 gradient 修正基础上, 引入 machine learning surrogate model 来修正 density gradient 的 semilocal 误差, 解决带隙低估问题"
    pes = compute_pes(high_heredity, DFT_LINEAGE[1], DFT_LINEAGE[0])
    assert pes["heredity"] > 0.3, f"高遗传答案 heredity 应 > 0.3, got {pes['heredity']}"
    assert pes["variation"] > 0.0, f"应有变异, got {pes['variation']}"

    # PES: 低遗传答案 (只是话题相关, 没有驱动机制继承)
    low_heredity = "我建议用神经网络来预测材料性质"
    pes_low = compute_pes(low_heredity, DFT_LINEAGE[1], DFT_LINEAGE[0])
    assert pes_low["heredity"] < pes["heredity"], "低遗传答案 heredity 应低于高遗传答案"

    print(f"PASS: ideagene_bench ({len(tasks)} tasks, T1+T4+Arena)")


if __name__ == "__main__":
    _selfcheck()
