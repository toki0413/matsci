"""反完成审计 agent — 强制 agent 返回前过一遍"是否过早收敛"审计.

depth_search.PrematureConvergenceDetector 只查最小努力下限 (iteration/
families/live_components 三项硬性指标). 本模块在此基础上加三层:

1. 等价性陷阱: 调 equivalence_auditor, 命中换名归约就阻断 (prompt 里"等价
   引理不算接近完成"的硬性化)
2. 不完整性自白: agent 必须显式列出"还没探索什么", 强制跳出"看起来完整"
   的认知偏差. 没自白 = 没思考过自己漏了什么
3. 对抗否决: red_team / adversarial agent 标记的 premature_convergence 直接
   阻断, 审计 agent 不覆盖

四层全过才算 is_complete. 任一层挂掉 block_reason 都要给出可读原因, 让
根 agent 知道该补什么.

不依赖 engine / LLM. equivalence_auditor 不传 model 时走规则, 传了 model
走 LLM 增强 (失败降级到规则).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from huginn.metacog.depth_search import (
    MinEffortFloor,
    PrematureConvergenceDetector,
)
from huginn.metacog.equivalence_auditor import EquivalenceAuditor, EquivalenceVerdict
from huginn.metacog import recall_audit_context

logger = logging.getLogger(__name__)


@dataclass
class CompletionChecklist:
    """反完成审计清单 — 四层全过才算完整."""

    effort_floor_passed: bool = False
    effort_deficits: list[str] = field(default_factory=list)
    equivalence_traps_remaining: list[str] = field(default_factory=list)
    unexplored_declaration: str = ""
    unexplored_count: int = 0
    adversarial_veto: bool = False
    adversarial_veto_reason: str = ""
    # 历史子目标探索记录, 供根 agent 对比当前探索与历史 (不参与 is_complete 判定)
    historical_context: list[dict] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """全部通过才算完整."""
        return (
            self.effort_floor_passed
            and not self.equivalence_traps_remaining
            and not self.adversarial_veto
            and self.unexplored_count >= 1  # 必须有至少 1 条自白
        )

    def block_reason(self) -> str:
        """is_complete=False 时返回阻断原因拼串; 完整时返回空串."""
        reasons: list[str] = []
        if not self.effort_floor_passed:
            reasons.append(f"努力下限未达标: {'; '.join(self.effort_deficits)}")
        if self.equivalence_traps_remaining:
            reasons.append(f"未排除等价性陷阱: {self.equivalence_traps_remaining}")
        if self.adversarial_veto:
            reasons.append(f"对抗 agent 否决: {self.adversarial_veto_reason}")
        if self.unexplored_count < 1:
            reasons.append("缺少显式不完整性自白")
        return " | ".join(reasons)


def parse_unexplored_declaration(text: str) -> int:
    """解析 agent 输出的 "UNEXPLORED:" 自白块, 数条目数.

    廉价解析, 不上 NLP:
    - 优先数 "- " 开头的 markdown 列表项
    - 没有 "- " 就数分号或换行分隔的非空条目
    - 空文本 / 只声明"已充分探索" → 0 (不满足自白要求)

    "已充分探索" 是 agent 的另一种诚实声明, 但本审计要的是"还差什么",
    不是"已经够了". 所以它不算条目.
    """
    if not text or not text.strip():
        return 0

    stripped = text.strip()

    # ponytail: 只认"已充分探索"这一个豁免短语. 别的同义词以后真出问题再加.
    if stripped == "已充分探索" or stripped.lower() in ("fully explored", "n/a", "none"):
        return 0

    # markdown 列表项优先
    bullet_lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.lstrip().startswith("- ") and line.strip()[2:].strip()
    ]
    if bullet_lines:
        return len(bullet_lines)

    # 没列表就数分号/换行分隔的非空条目
    # ponytail: 单次 split, 不去重, 上限 99 防恶意输入
    chunks = [c.strip() for c in stripped.replace("\n", ";").split(";") if c.strip()]
    return min(len(chunks), 99)


class CompletionAuditor:
    """反完成审计 agent — 综合四层检查是否过早收敛.

    用法:
        auditor = CompletionAuditor()
        checklist = auditor.audit(
            iteration=5, families_explored=3, live_components=2,
            total_iterations=9,
            candidate_finding="我们解决了稳定性",
            original_problem="预测材料稳定性",
            reduction_chain="通过形成能预测",
            unexplored_declaration="UNEXPLORED:\\n- 高压相图\\n- 缺陷容忍度",
        )
        if not checklist.is_complete:
            # 不许返回, 把 block_reason() 喂回 agent 继续探索
    """

    def __init__(
        self,
        convergence_detector: PrematureConvergenceDetector | None = None,
        equivalence_auditor: EquivalenceAuditor | None = None,
    ) -> None:
        # 懒加载, 也支持外部注入 (测试用)
        self._detector = convergence_detector or PrematureConvergenceDetector()
        self._auditor = equivalence_auditor or EquivalenceAuditor()

    def audit(
        self,
        iteration: int,
        families_explored: int,
        live_components: int,
        total_iterations: int,
        candidate_finding: str = "",
        original_problem: str = "",
        reduction_chain: str = "",
        unexplored_declaration: str = "",
        adversarial_veto: bool = False,
        adversarial_veto_reason: str = "",
        provenance_snapshot: list | None = None,
    ) -> CompletionChecklist:
        """反完成审计: 综合检查是否过早收敛, 返回 CompletionChecklist.

        provenance_snapshot: 可选的 ProvenanceEntry 列表, 塞进 historical_context
        供根 agent 对比当前探索与历史工具产出. 不传则不影响现有行为.
        """
        # 1. 努力下限 (PrematureConvergenceDetector)
        status = self._detector.check(
            iteration=iteration,
            families_explored=families_explored,
            live_components=live_components,
            total_iterations=total_iterations,
        )
        blocked, deficit_msg = self._detector.should_block_return(status)
        effort_floor_passed = not blocked
        effort_deficits = status.deficits() if blocked else []

        # 2. 等价性陷阱 (EquivalenceAuditor) — 只在 agent 真的提交了发现时才审
        traps: list[str] = []
        if candidate_finding.strip():
            verdict: EquivalenceVerdict = self._auditor.audit(
                candidate_finding=candidate_finding,
                original_problem=original_problem,
                reduction_chain=reduction_chain,
            )
            if verdict.is_equivalent_renaming:
                trap_desc = (
                    f"{verdict.trap_category} -> {verdict.reduction_target}"
                    if verdict.reduction_target
                    else verdict.trap_category
                )
                traps.append(trap_desc)

        # 3. 不完整性自白
        unexplored_count = parse_unexplored_declaration(unexplored_declaration)

        # recall 历史子目标探索, 给根 agent 提供对比上下文 (不影响 is_complete 判定)
        # ponytail: 只挂到 historical_context, 不参与四层检查. 升级路径: 命中过深时计入 deficit.
        historical = recall_audit_context(
            category="subgoal",
            query=original_problem or candidate_finding,
            limit=10,
        )

        # ponytail: provenance_snapshot 可选, 不破坏现有调用; 升级路径是 audit 自动拉 ProvenanceRegistry.recent_entries()
        if provenance_snapshot:
            if isinstance(historical, list):
                historical = list(historical)
                historical.extend(provenance_snapshot)
            elif isinstance(historical, dict):
                historical["provenance"] = provenance_snapshot
            else:
                historical = list(provenance_snapshot)

        return CompletionChecklist(
            effort_floor_passed=effort_floor_passed,
            effort_deficits=effort_deficits,
            equivalence_traps_remaining=traps,
            unexplored_declaration=unexplored_declaration,
            unexplored_count=unexplored_count,
            adversarial_veto=adversarial_veto,
            adversarial_veto_reason=adversarial_veto_reason,
            historical_context=historical,
        )


# ── 自检 ─────────────────────────────────────────────────────────

def _selfcheck() -> None:
    auditor = CompletionAuditor()

    # ── parse_unexplored_declaration ──
    assert parse_unexplored_declaration("") == 0, "空文本应返回 0"
    assert parse_unexplored_declaration("   ") == 0, "纯空白应返回 0"
    assert parse_unexplored_declaration("已充分探索") == 0, "'已充分探索'不算条目"
    assert parse_unexplored_declaration("n/a") == 0, "'n/a'不算条目"

    # markdown 列表
    md = "UNEXPLORED:\n- 高压相图\n- 缺陷容忍度\n- 动力学路径"
    assert parse_unexplored_declaration(md) == 3, f"应数出 3 条, got {parse_unexplored_declaration(md)}"

    # 分号分隔
    semi = "高压相图; 缺陷容忍度; 动力学路径"
    assert parse_unexplored_declaration(semi) == 3, "分号分隔应数 3 条"

    # 换行分隔
    nl = "高压相图\ndefect tolerance\n动力学路径"
    assert parse_unexplored_declaration(nl) == 3, "换行分隔应数 3 条"

    # ── CompletionChecklist.is_complete / block_reason ──
    # 全过的 checklist
    full = CompletionChecklist(
        effort_floor_passed=True,
        effort_deficits=[],
        equivalence_traps_remaining=[],
        unexplored_declaration="- 高压相图",
        unexplored_count=1,
        adversarial_veto=False,
    )
    assert full.is_complete, "四层全过应 is_complete"
    assert full.block_reason() == "", "完整时 block_reason 应为空"

    # 努力下限未过
    low_effort = CompletionChecklist(
        effort_floor_passed=False,
        effort_deficits=["iteration=1 < min_iterations=3"],
        unexplored_count=2,
    )
    assert not low_effort.is_complete
    assert "努力下限未达标" in low_effort.block_reason()

    # 有等价陷阱
    trap = CompletionChecklist(
        effort_floor_passed=True,
        equivalence_traps_remaining=["property_reduction -> formation_energy_prediction"],
        unexplored_count=1,
    )
    assert not trap.is_complete
    assert "等价性陷阱" in trap.block_reason()

    # 缺自白
    no_decl = CompletionChecklist(effort_floor_passed=True, unexplored_count=0)
    assert not no_decl.is_complete
    assert "不完整性自白" in no_decl.block_reason()

    # 对抗否决
    veto = CompletionChecklist(
        effort_floor_passed=True,
        unexplored_count=1,
        adversarial_veto=True,
        adversarial_veto_reason="premature_convergence: 只探索了 1 个方法族",
    )
    assert not veto.is_complete
    assert "对抗 agent 否决" in veto.block_reason()
    assert "premature_convergence" in veto.block_reason()

    # ── CompletionAuditor.audit 综合场景 ──
    # 场景 A: 全过 — 后期迭代, 多方法族, 有自白, 无陷阱, 无否决
    ok = auditor.audit(
        iteration=9, families_explored=4, live_components=2, total_iterations=9,
        candidate_finding="我们用主动学习找到了相场边界",
        original_problem="找相场边界",
        reduction_chain="",
        unexplored_declaration="UNEXPLORED:\n- 高压区\n- 多组分体系",
    )
    assert ok.is_complete, f"全过应 is_complete, block={ok.block_reason()}"
    assert ok.effort_floor_passed
    assert ok.equivalence_traps_remaining == []
    assert ok.unexplored_count == 2

    # 场景 B: 努力下限未过 — 早期迭代, 单方法族
    early = auditor.audit(
        iteration=0, families_explored=1, live_components=1, total_iterations=9,
        candidate_finding="我们解决了稳定性预测",
        original_problem="预测材料稳定性",
        unexplored_declaration="- 还有高压相图没看",
    )
    assert not early.is_complete, "早期未达下限应阻断"
    assert not early.effort_floor_passed
    assert early.effort_deficits, "未达标应有 deficit 列表"
    assert "努力下限未达标" in early.block_reason()

    # 场景 C: 等价陷阱命中 — 性质归约
    trapped = auditor.audit(
        iteration=9, families_explored=4, live_components=2, total_iterations=9,
        candidate_finding="我们解决了稳定性预测问题",
        original_problem="预测材料稳定性",
        reduction_chain="通过形成能预测间接得到稳定性",
        unexplored_declaration="- 高压相图",
    )
    assert not trapped.is_complete, "等价陷阱应阻断"
    assert len(trapped.equivalence_traps_remaining) == 1, "应有 1 条陷阱"
    assert "property_reduction" in trapped.equivalence_traps_remaining[0]
    assert "等价性陷阱" in trapped.block_reason()

    # 场景 D: 缺自白
    no_confession = auditor.audit(
        iteration=9, families_explored=4, live_components=2, total_iterations=9,
        candidate_finding="主动学习找相场边界",
        original_problem="找相场边界",
        unexplored_declaration="",
    )
    assert not no_confession.is_complete, "缺自白应阻断"
    assert no_confession.unexplored_count == 0
    assert "不完整性自白" in no_confession.block_reason()

    # 场景 E: 自白只有"已充分探索" → 仍算缺自白
    lazy_confession = auditor.audit(
        iteration=9, families_explored=4, live_components=2, total_iterations=9,
        candidate_finding="主动学习找相场边界",
        original_problem="找相场边界",
        unexplored_declaration="已充分探索",
    )
    assert not lazy_confession.is_complete, "'已充分探索'不算自白"
    assert lazy_confession.unexplored_count == 0

    # 场景 F: 对抗否决
    vetoed = auditor.audit(
        iteration=9, families_explored=4, live_components=2, total_iterations=9,
        candidate_finding="主动学习找相场边界",
        original_problem="找相场边界",
        unexplored_declaration="- 高压相图",
        adversarial_veto=True,
        adversarial_veto_reason="premature_convergence: 缺陷容忍度未覆盖",
    )
    assert not vetoed.is_complete, "对抗否决应阻断"
    assert vetoed.adversarial_veto
    assert "对抗 agent 否决" in vetoed.block_reason()
    assert "缺陷容忍度" in vetoed.block_reason()

    # 场景 G: 无 candidate_finding → 跳过等价审计, 不应报陷阱
    no_candidate = auditor.audit(
        iteration=9, families_explored=4, live_components=2, total_iterations=9,
        candidate_finding="",
        original_problem="",
        unexplored_declaration="- 高压相图",
    )
    assert no_candidate.equivalence_traps_remaining == [], "无候选发现不应触发等价审计"
    assert no_candidate.is_complete, "其他都过时应 is_complete"

    # 场景 H: 多重失败 — block_reason 应拼多条
    multi = auditor.audit(
        iteration=0, families_explored=1, live_components=1, total_iterations=9,
        candidate_finding="我们解决了稳定性预测问题",
        original_problem="预测材料稳定性",
        reduction_chain="通过形成能预测间接得到稳定性",
        unexplored_declaration="",
        adversarial_veto=True,
        adversarial_veto_reason="premature",
    )
    assert not multi.is_complete
    reason = multi.block_reason()
    assert "努力下限" in reason
    assert "等价性陷阱" in reason
    assert "对抗 agent 否决" in reason
    assert "不完整性自白" in reason

    print("completion_auditor selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
