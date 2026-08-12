"""Agent Evolution Engine — self-improvement without LLM fine-tuning.

The core insight: Agent intelligence comes from
  1) System Prompt (what it knows, how it thinks)
  2) Tool Registry (what it can do)
  3) RAG Knowledge (what it can look up)
  4) Workflow Templates (how it plans tasks)
  5) Self-Healing Rules (how it recovers from errors)

We evolve these COMPONENTS, not the LLM weights.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .logger import ExecutionLogger

if TYPE_CHECKING:
    from huginn.skills.base import SkillDefinition

# Hard cap on learned rules. Beyond this the lowest-confidence ones get
# evicted to keep the rule set manageable and evaluation fast.
MAX_RULES = 100
_CONFIDENCE_FLOOR = 0.3


def _snake_case(name: str) -> str:
    """把任意名转成 snake_case 技能标识.

    自动提取的模板名形如 "Relax Workflow (VASP)", 而 SkillRegistry 用 name
    作 key 且 SkillTool 按 name 查找, 统一成 snake_case 才是合法标识.
    退化: 抽不到字母数字时返回 'evolved_skill'."""
    s = re.sub(r"[^a-zA-Z0-9]+", " ", name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s or "evolved_skill"


def _recompute_confidence(rule: EvolutionRule) -> float:
    """根据应用效果重算 confidence, 而非只看失败次数.

    之前 confidence = min(0.5 + count*0.1, 0.95) 只随失败模式出现次数涨,
    一条被 LLM 永远忽略的规则失败够多次也能到 0.95, 永不淘汰.

    新策略 (效果反馈):
      - 基线随 usage_count (应用次数) 涨, 而非失败次数 — 反映 "规则被实际用过"
      - usage_count > 0 且 success_count == 0: 规则被注入但从未真正帮上忙 → 降权
      - usage_count > 0 且 success_count > 0: 规则真的有效 → 加权
      - 结果夹在 [0.1, 0.98] 之间, 仍受 _CONFIDENCE_FLOOR 约束 (prune 时踢)

    v23 Round 8 校正: 注释之前说 "基线随失败次数涨 (保留原行为)", 但实现
    用的是 usage_count. 现修正注释匹配实现 — usage_count 是更合理的信号
    (反映规则被实际应用, 而非只是失败模式出现).
    """
    # EvolutionRule 是 dataclass, usage_count 是必填字段 (默认 0), 直接访问即可.
    # 之前用 getattr(rule, "usage_count", 0) 是过度防御, 会掩盖字段缺失的 bug.
    base = 0.5 + rule.usage_count * 0.05
    if rule.usage_count > 0:
        if rule.success_count == 0:
            # 被注入过但从未验证有效 — 降到基线以下, 让 prune 有机会淘汰
            base *= 0.5
        else:
            # 应用成功率加权 (success/usage, 上限 +0.3)
            success_rate = rule.success_count / max(rule.usage_count, 1)
            base += min(success_rate * 0.3, 0.3)
    return max(0.1, min(base, 0.98))


@dataclass
class EvolutionRule:
    """A learned rule for improving agent behavior."""

    rule_id: str
    rule_type: str  # "prompt_patch", "tool_strategy", "heuristic_fix", "skill_template"
    trigger: str  # Condition that activates this rule
    action: str  # What to do when triggered
    source: str  # How was this rule learned: "failure_analysis", "success_extraction", "user_feedback"
    # confidence: 创建时必须显式指定 (0.0 默认低于 _CONFIDENCE_FLOOR=0.3, 会被 prune 踢).
    # 各 source 的初始 confidence:
    #   - failure_analysis (heuristic_fix): min(0.5 + count*0.1, 0.95) — 失败模式越频繁越高
    #   - stable_principle: 0.9 — 长期蒸馏, 高置信
    #   - success_analysis (prompt_patch): 1.0 - success_rate — 工具越差 patch 越重要
    #   - reward_analysis (reward_patch): 1.0 - avg_reward — 同上
    # 后续由 _recompute_confidence 根据 usage_count/success_count 重算.
    confidence: float = 0.0  # 0-1, how reliable is this rule
    usage_count: int = 0  # 被实际应用 (apply_heuristic_fix) 的次数
    success_count: int = 0  # 应用后 mark_fix_success(succeeded=True) 的次数
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)


@dataclass
class SkillTemplate:
    """An extracted, reusable skill from successful executions."""

    skill_id: str
    name: str
    description: str
    trigger_keywords: list[str]
    workflow_steps: list[dict[str, Any]]
    required_tools: list[str]
    source_session: str
    extraction_confidence: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_skill_definition(self) -> SkillDefinition:
        """把 SkillTemplate 转成 SkillRegistry 可注册的 SkillDefinition.

        弥合两个技能池: evolution 自动提取的 SkillTemplate (evolved_skills.json)
        → 声明式 SkillRegistry. usage_count/success_count/extraction_confidence
        塞进 metadata['evolution'], 供运行时复用统计与"提升基元/淘汰"元规则取用.
        """
        from huginn.skills.base import SkillDefinition, SkillStep

        steps = [
            SkillStep(
                name=step.get("name") or f"step_{i}",
                tool=step["tool"],
                input_mapping={},
                output_key=f"step_{i}_out",
                on_failure="skip",
            )
            for i, step in enumerate(self.workflow_steps)
            if isinstance(step, dict) and step.get("tool")
        ]
        # workflow_steps 里只有 tool 名, 没有 input_mapping 值; 若一条 step 都
        # 抽不出来, 就用 required_tools 兜底保底可注册.
        if not steps:
            steps = [
                SkillStep(
                    name=f"step_{i}",
                    tool=tool,
                    input_mapping={},
                    output_key=f"step_{i}_out",
                    on_failure="skip",
                )
                for i, tool in enumerate(self.required_tools)
            ]
        return SkillDefinition(
            name=_snake_case(self.name),
            description=self.description,
            category="distilled",
            steps=steps,
            required_tools=list(self.required_tools),
            tags=list(self.trigger_keywords) + ["evolved"],
            metadata={
                "evolution": {
                    "skill_id": self.skill_id,
                    "source_session": self.source_session,
                    "extraction_confidence": self.extraction_confidence,
                    "usage_count": self.usage_count,
                    "success_count": self.success_count,
                    "created_at": self.created_at,
                }
            },
        )


class EvolutionEngine:
    """Orchestrates agent self-evolution across multiple dimensions.

    Evolves:
      - Prompt patches (what to add/change in system prompt)
      - Tool selection strategy (which tool to use when)
      - Heuristic error fixes (common error → automatic fix)
      - Skill templates (reusable workflow patterns)
      - Knowledge base updates (new facts to add to RAG)
    """

    def __init__(
        self,
        logger: ExecutionLogger | None = None,
        rules_path: str | None = None,
        skills_path: str | None = None,
    ):
        # 无参时走全局默认 logger (~/.huginn/logs/), 跟 agent 运行时同一份 rules
        if logger is None:
            logger = ExecutionLogger()
        self.logger = logger
        self.rules_path = (
            Path(rules_path)
            if rules_path
            else logger.persist_dir / "evolution_rules.json"
        )
        self.skills_path = (
            Path(skills_path)
            if skills_path
            else logger.persist_dir / "evolved_skills.json"
        )
        self.rules: list[EvolutionRule] = []
        self.skills: list[SkillTemplate] = []
        self._pending_fix_tool: str | None = None
        self._pending_fix_rule_id: str | None = None
        # File-level lock — _save_rules / _save_skills / _load_rules can be
        # called from background reflection threads, so we serialize disk access.
        self._lock = threading.Lock()
        self._load_rules()
        self._load_skills()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        with self._lock:
            if self.rules_path.exists():
                with self.rules_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.rules = [EvolutionRule(**r) for r in data]

    def _save_rules(self) -> None:
        with self._lock, self.rules_path.open("w", encoding="utf-8") as f:
            json.dump(
                [self._rule_to_dict(r) for r in self.rules],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _rule_to_dict(self, rule: EvolutionRule) -> dict[str, Any]:
        return {
            "rule_id": rule.rule_id,
            "rule_type": rule.rule_type,
            "trigger": rule.trigger,
            "action": rule.action,
            "source": rule.source,
            "confidence": rule.confidence,
            "usage_count": rule.usage_count,
            "success_count": rule.success_count,
            "created_at": rule.created_at,
            "tags": rule.tags,
        }

    def _load_skills(self) -> None:
        if self.skills_path.exists():
            with self.skills_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.skills = [SkillTemplate(**s) for s in data]

    def _save_skills(self) -> None:
        with self._lock, self.skills_path.open("w", encoding="utf-8") as f:
            json.dump(
                [self._skill_to_dict(s) for s in self.skills],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _skill_to_dict(self, skill: SkillTemplate) -> dict[str, Any]:
        return {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "trigger_keywords": skill.trigger_keywords,
            "workflow_steps": skill.workflow_steps,
            "required_tools": skill.required_tools,
            "source_session": skill.source_session,
            "extraction_confidence": skill.extraction_confidence,
            "usage_count": skill.usage_count,
            "success_count": skill.success_count,
            "created_at": skill.created_at,
        }

    def _prune_rules(self) -> None:
        """Evict stale rules: drop confidence < floor, then enforce MAX_RULES.

        修剪前先重算每条规则的 confidence, 让 usage_count 高但 success_count=0
        (被注入却从未真正有效) 的规则被降权淘汰, 而不是靠失败次数一直挂着.
        """
        for rule in self.rules:
            if rule.usage_count > 0:
                rule.confidence = _recompute_confidence(rule)
        self.rules = [r for r in self.rules if r.confidence >= _CONFIDENCE_FLOOR]
        if len(self.rules) > MAX_RULES:
            self.rules.sort(key=lambda r: r.confidence, reverse=True)
            del self.rules[MAX_RULES:]

    # ------------------------------------------------------------------
    # Core Evolution Cycles
    # ------------------------------------------------------------------

    def evolve_from_failures(self) -> list[EvolutionRule]:
        """Analyze recent failures and generate heuristic fix rules."""
        new_rules: list[EvolutionRule] = []
        patterns = self.logger.get_failure_patterns(min_count=2)

        for pat in patterns:
            # Check if we already have a rule for this
            existing = [r for r in self.rules if r.trigger == pat["pattern"]]
            if existing:
                continue

            tool = pat["tool"]
            error_snippet = pat["error"]

            # Generate heuristic fix based on error type
            fix_action = self._generate_heuristic_fix(tool, error_snippet)
            if fix_action:
                rule = EvolutionRule(
                    rule_id=f"heuristic_{tool}_{int(time.time() * 1000)}",
                    rule_type="heuristic_fix",
                    trigger=pat["pattern"],
                    action=fix_action,
                    source="failure_analysis",
                    confidence=min(0.5 + pat["count"] * 0.1, 0.95),
                    tags=["auto_generated", tool],
                )
                self.rules.append(rule)
                new_rules.append(rule)

        # 桥 I: stable_principles → evolution rules. 蒸馏出的原则作为高置信度
        # rule 注入规则库, 让 PMK/autoloop 消费 evolution_rules 时能感知到
        # 长期原则. ponytail: 原则文本同时当 trigger 和 action, 不做语义拆分.
        # ceiling: 纯文本匹配, 不捕捉原则的适用条件. 升级: LLM 抽 trigger/condition.
        try:
            from huginn.memory.longterm import load_stable_principles
            _principles = load_stable_principles()
        except Exception:
            _principles = []
        _existing_ids = {r.rule_id for r in self.rules}
        _existing_triggers = {r.trigger for r in self.rules}
        for i, p in enumerate(_principles):
            if not p or not p.strip():
                continue
            _rid = f"principle_{i}_{hash(p) & 0xFFFFFFFF:x}"
            if _rid in _existing_ids or p in _existing_triggers:
                continue
            rule = EvolutionRule(
                rule_id=_rid,
                rule_type="stable_principle",
                trigger=p,
                action=p,
                source="stable_principles",
                confidence=0.9,
                tags=["stable_principle", "auto_promoted"],
            )
            self.rules.append(rule)
            new_rules.append(rule)

        if new_rules:
            self._prune_rules()
            self._save_rules()
        return new_rules

    def evolve_from_successes(self) -> list[SkillTemplate]:
        """Extract reusable skill templates from successful executions."""
        new_skills: list[SkillTemplate] = []
        successes = [r for r in self.logger._tool_calls if r.success]

        # Group by calculation type and software
        from collections import defaultdict

        grouped = defaultdict(list)
        for r in successes:
            key = f"{r.calculation_type or 'unknown'}_{r.software or 'general'}"
            grouped[key].append(r)

        for key, records in grouped.items():
            if len(records) < 3:
                continue  # Need enough examples

            # Check if we already have a similar skill
            calc_type, software = key.rsplit("_", 1)
            existing = [
                s
                for s in self.skills
                if calc_type in s.trigger_keywords or software in s.trigger_keywords
            ]
            if existing:
                continue

            # Extract common workflow pattern
            tools_used = list({r.tool_name for r in records})
            skill = SkillTemplate(
                skill_id=f"skill_{key}_{int(time.time() * 1000)}",
                name=f"{calc_type.title()} Workflow ({software})",
                description=f"Auto-extracted workflow for {calc_type} using {software}",
                trigger_keywords=[calc_type, software],
                workflow_steps=[
                    {"tool": r.tool_name, "input_keys": list(r.tool_input.keys())}
                    for r in records[:5]
                ],
                required_tools=tools_used,
                source_session=records[0].session_id,
                extraction_confidence=min(0.4 + len(records) * 0.05, 0.9),
            )
            self.skills.append(skill)
            new_skills.append(skill)

        if new_skills:
            self._save_skills()
        return new_skills

    def sync_to_registry(self) -> list[SkillDefinition]:
        """把本地 SkillTemplate 池同步进 SkillRegistry, 弥合两个技能池.

        evolution 自动提取的模板 (self.skills) 默认只落在 evolved_skills.json,
        与声明式 SkillRegistry (presets) 是两套并行系统. 本方法把每个模板转成
        SkillDefinition 注册进 SkillRegistry, 使自动演化的能力进入主技能库 —
        能被 SkillTool 执行、被 /skills 列出、被技能树查询.

        已存在同名技能的跳过 (不覆盖 presets). 返回本次新注册的 SkillDefinition.
        """
        from huginn.skills.registry import SkillRegistry

        registered: list[SkillDefinition] = []
        for tpl in self.skills:
            try:
                skill = tpl.to_skill_definition()
            except Exception:
                continue
            if SkillRegistry.get(skill.name):
                continue
            try:
                registered.append(SkillRegistry.register(skill))
            except Exception:
                continue
        return registered

    def evolve_prompt_patches(self) -> list[EvolutionRule]:
        """Generate system prompt patches based on execution patterns."""
        new_rules: list[EvolutionRule] = []

        # Find tools with low success rate
        success_rates = self.logger.get_tool_success_rate()
        for tool, rate in success_rates.items():
            if rate < 0.7:
                # Suggest a prompt patch
                patch = self._generate_prompt_patch_for_tool(tool, rate)
                if patch:
                    rule = EvolutionRule(
                        rule_id=f"prompt_{tool}_{int(time.time() * 1000)}",
                        rule_type="prompt_patch",
                        trigger=f"tool_{tool}_low_success",
                        action=patch,
                        source="success_analysis",
                        confidence=1.0 - rate,
                        tags=["prompt", tool],
                    )
                    self.rules.append(rule)
                    new_rules.append(rule)

        if new_rules:
            self._prune_rules()
            self._save_rules()
        return new_rules

    def evolve_from_rewards(self) -> dict[str, Any]:
        """基于数值奖励 (R_phys) 做进化——高奖励提取技能, 低奖励生成提示补丁。

        和 evolve_from_successes/failures 互补: 那两个只看二值成败, 这里看连续
        奖励值, 能抓到 "成功但物理质量差" (success=True 但 R_phys 低) 的中间态,
        这是纯二值信号永远看不到的。
        """
        rewarded = [r for r in self.logger._tool_calls if r.reward is not None]
        if not rewarded:
            return {"high_reward_skills": [], "low_reward_patches": []}

        new_skills: list[SkillTemplate] = []
        new_rules: list[EvolutionRule] = []

        # 高奖励记录: 提取为可复用技能 (R_phys >= 0.7 视为高质量执行)
        high = [r for r in rewarded if r.reward >= 0.7 and r.success]
        from collections import defaultdict

        grouped = defaultdict(list)
        for r in high:
            key = f"{r.calculation_type or 'unknown'}_{r.software or 'general'}"
            grouped[key].append(r)
        for key, records in grouped.items():
            if len(records) < 2:
                continue
            calc_type, software = key.rsplit("_", 1)
            existing = [
                s
                for s in self.skills
                if calc_type in s.trigger_keywords or software in s.trigger_keywords
            ]
            if existing:
                continue
            # 按 reward 降序, 取 top 记录提取 workflow
            records.sort(key=lambda r: r.reward, reverse=True)
            tools_used = list({r.tool_name for r in records})
            avg_reward = sum(r.reward for r in records) / len(records)
            skill = SkillTemplate(
                skill_id=f"skill_reward_{key}_{int(time.time() * 1000)}",
                name=f"{calc_type.title()} High-Reward Workflow ({software})",
                description=f"Auto-extracted from R_phys>=0.7 executions, avg reward {avg_reward:.2f}",
                trigger_keywords=[calc_type, software],
                workflow_steps=[
                    {
                        "tool": r.tool_name,
                        "input_keys": list(r.tool_input.keys()),
                        "reward": r.reward,
                    }
                    for r in records[:5]
                ],
                required_tools=tools_used,
                source_session=records[0].session_id,
                extraction_confidence=min(0.5 + avg_reward * 0.4, 0.95),
            )
            self.skills.append(skill)
            new_skills.append(skill)

        # 低奖励记录: 生成提示补丁 (R_phys < 0.3 视为需要改进)
        low = [r for r in rewarded if r.reward < 0.3]
        tool_low_reward: dict[str, list[float]] = defaultdict(list)
        for r in low:
            tool_low_reward[r.tool_name].append(r.reward)
        for tool, rewards in tool_low_reward.items():
            avg_r = sum(rewards) / len(rewards)
            trigger = f"tool_{tool}_low_reward"
            existing = [r for r in self.rules if r.trigger == trigger]
            if existing:
                continue
            patch = self._generate_reward_patch_for_tool(tool, avg_r)
            if patch:
                rule = EvolutionRule(
                    rule_id=f"reward_patch_{tool}_{int(time.time() * 1000)}",
                    rule_type="prompt_patch",
                    trigger=trigger,
                    action=patch,
                    source="reward_analysis",
                    confidence=1.0 - avg_r,
                    tags=["reward", tool],
                )
                self.rules.append(rule)
                new_rules.append(rule)

        if new_skills:
            self._save_skills()
        if new_rules:
            self._prune_rules()
            self._save_rules()
        return {
            "high_reward_skills": [self._skill_to_dict(s) for s in new_skills],
            "low_reward_patches": [self._rule_to_dict(r) for r in new_rules],
        }

    def run_full_evolution_cycle(self) -> dict[str, Any]:
        """Run all evolution mechanisms and return a report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "failure_rules": [],
            "success_skills": [],
            "prompt_patches": [],
            "reward_evolution": {},
            "total_rules": len(self.rules),
            "total_skills": len(self.skills),
        }

        # Phase 1: Learn from failures
        failure_rules = self.evolve_from_failures()
        report["failure_rules"] = [self._rule_to_dict(r) for r in failure_rules]

        # Phase 2: Learn from successes
        success_skills = self.evolve_from_successes()
        report["success_skills"] = [self._skill_to_dict(s) for s in success_skills]

        # Phase 3: Prompt optimization
        prompt_patches = self.evolve_prompt_patches()
        report["prompt_patches"] = [self._rule_to_dict(r) for r in prompt_patches]

        # Phase 4: 基于 R_phys 数值奖励的进化 (阶段4 单轨回流)
        reward_result = self.evolve_from_rewards()
        report["reward_evolution"] = reward_result

        report["total_rules_after"] = len(self.rules)
        report["total_skills_after"] = len(self.skills)

        # Save report
        report_path = self.logger.persist_dir / "evolution_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Append to history for convergence tracking
        self._append_history(report)

        return report

    def _avg_confidence(self) -> float:
        """Return the average confidence of all learned rules."""
        if not self.rules:
            return 0.0
        return sum(r.confidence for r in self.rules) / len(self.rules)

    def _append_history(self, report: dict[str, Any]) -> None:
        """Append the current cycle metrics to a history file."""
        history_path = self.logger.persist_dir / "evolution_history.json"
        history: list[dict[str, Any]] = []
        if history_path.exists():
            try:
                with history_path.open("r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(
            {
                "timestamp": report.get("timestamp"),
                "total_rules": report.get("total_rules_after"),
                "total_skills": report.get("total_skills_after"),
                "avg_confidence": self._avg_confidence(),
                "new_failure_rules": len(report.get("failure_rules", [])),
                "new_success_skills": len(report.get("success_skills", [])),
                "new_prompt_patches": len(report.get("prompt_patches", [])),
                "new_reward_skills": len(
                    report.get("reward_evolution", {}).get("high_reward_skills", [])
                ),
                "new_reward_patches": len(
                    report.get("reward_evolution", {}).get("low_reward_patches", [])
                ),
            }
        )

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Runtime Application of Evolved Knowledge
    # ------------------------------------------------------------------

    def apply_heuristic_fix(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        error: str,
        min_confidence: float = 0.0,
    ) -> dict[str, Any] | None:
        """Check if we have a learned fix for this error and apply it.

        min_confidence: 低于此置信度的规则跳过. 写入门槛 0.3, 注入门槛应更高.
        """
        for rule in self.rules:
            if rule.rule_type != "heuristic_fix":
                continue
            if rule.confidence < min_confidence:
                continue
            if rule.trigger.startswith(f"{tool_name}|") and self._error_matches(
                rule.trigger.split("|", 1)[1], error
            ):
                rule.usage_count += 1
                self._pending_fix_tool = tool_name
                self._pending_fix_rule_id = rule.rule_id
                # 命中即落盘, 否则 usage_count 只在内存, 跨 session 丢.
                self._save_rules()
                fix = self._parse_fix_action(rule.action, tool_input)
                if fix:
                    fix["confidence"] = rule.confidence
                    fix["rule_id"] = rule.rule_id
                return fix
        return None

    def mark_fix_success(self, tool_name: str, succeeded: bool) -> None:
        """Track whether the last applied fix actually helped.

        Called from reflection after each tool result. If the tool that
        previously failed now succeeds, increment success_count on the
        rule that was applied. 同时重算 confidence, 让应用效果反馈进来
        (之前 confidence 只随失败次数涨, 被忽略的规则也能到 0.95).
        """
        if not succeeded:
            self._pending_fix_tool = None
            self._pending_fix_rule_id = None
            return
        rid = getattr(self, "_pending_fix_rule_id", None)
        if rid and tool_name == getattr(self, "_pending_fix_tool", ""):
            for rule in self.rules:
                if rule.rule_id == rid:
                    rule.success_count += 1
                    rule.confidence = _recompute_confidence(rule)
                    self._save_rules()
                    break
        self._pending_fix_tool = None
        self._pending_fix_rule_id = None

    def get_relevant_skills(self, query: str) -> list[SkillTemplate]:
        """Find skills relevant to a user query."""
        query_lower = query.lower()
        scored = []
        for skill in self.skills:
            score = sum(1 for kw in skill.trigger_keywords if kw.lower() in query_lower)
            if score > 0:
                scored.append((score, skill))
        scored.sort(reverse=True)
        return [s for _, s in scored[:5]]

    def get_prompt_patches(self) -> list[str]:
        """Get all active prompt patches sorted by confidence."""
        patches = [
            (r.confidence, r.action)
            for r in self.rules
            if r.rule_type == "prompt_patch"
        ]
        patches.sort(reverse=True)
        return [a for _, a in patches]

    # ------------------------------------------------------------------
    # Heuristic Fix Generators
    # ------------------------------------------------------------------

    def _generate_heuristic_fix(self, tool: str, error: str) -> str | None:
        """Generate a fix action string for a given error pattern."""
        error_lower = error.lower()

        # VASP-specific fixes
        if "vasp" in tool.lower() or "dft" in tool.lower():
            if (
                "electronic" in error_lower
                or "convergence" in error_lower
                or "scf" in error_lower
            ):
                return '{"ALGO": "Normal", "NELMIN": 6, "mixing": "improved"}'
            if "ionic" in error_lower or "relaxation" in error_lower:
                return '{"IBRION": 2, "POTIM": 0.1, "NSW": 200}'
            if "memory" in error_lower:
                return '{"NCORE": "increase", "KPAR": "increase"}'

        # Gaussian-specific fixes
        if "gaussian" in tool.lower():
            if "scf" in error_lower or "convergence" in error_lower:
                return '{"scf": "xqc", "integral": "ultrafine"}'
            if "basis" in error_lower:
                return '{"basis": "check_missing", "genecp": "add_if_needed"}'
            if "optimization" in error_lower:
                return '{"opt": "calcfc", "maxcycle": 200}'

        # LAMMPS-specific fixes
        if "lammps" in tool.lower() or "md" in tool.lower():
            if "lost atoms" in error_lower:
                return '{"timestep": "halve", "neighbor": "increase_skin"}'
            if "bond" in error_lower or "angle" in error_lower:
                return '{"fix_shake": "apply", "bond_style": "check"}'
            if "thermo" in error_lower or "temperature" in error_lower:
                return '{"fix_nvt": "check_damping", "timestep": "reduce"}'

        # 通用 fix — action 必须含 description, 否则 _try_evolved_fix 拿不到可重跑的描述
        if "not found" in error_lower or "no such file" in error_lower:
            return '{"description": "re-check file path and try alternative locations"}'
        if "permission denied" in error_lower or "access" in error_lower:
            return '{"description": "verify permissions and try alternative approach"}'
        if "timeout" in error_lower:
            return '{"description": "retry with smaller scope or increased timeout"}'

        # 通用工具 fallback: 至少给个带上下文的 description
        if tool.lower() in ("read_file", "edit_file", "bash_tool", "ls", "file_write_tool", "execute", "code_tool", "file_read_tool"):
            # ponytail: error 截 200 字符防 token 爆炸; 含双引号时 _parse_fix_action 的 fallback 兜底
            return f'{{"description": "retry {tool} with corrected input after checking error: {error[:200]}"}}'

        return None

    def _generate_prompt_patch_for_tool(
        self, tool: str, success_rate: float
    ) -> str | None:
        """Generate a prompt patch to improve tool usage."""
        patches = {
            "vasp_tool": f"When using VASP, always verify convergence settings. Current success rate: {success_rate:.1%}. Consider adding ALGO=Normal for problematic systems.",
            "gaussian_tool": f"When using Gaussian, verify basis set coverage for all elements. Current success rate: {success_rate:.1%}. Use SCF=XQC for convergence issues.",
            "lammps_tool": f"When using LAMMPS, start with smaller timesteps and gradually increase. Current success rate: {success_rate:.1%}. Check neighbor list settings.",
        }
        return patches.get(tool)

    def _generate_reward_patch_for_tool(
        self, tool: str, avg_reward: float
    ) -> str | None:
        """低奖励工具的提示补丁——引导 agent 校验物理合理性而非仅追求执行成功。"""
        patches = {
            "vasp_tool": f"VASP 平均 R_phys={avg_reward:.2f}, 物理校验不达标。下次执行前确认: 能量为负、力收敛 <0.01 eV/Å、带隙非负。",
            "gaussian_tool": f"Gaussian 平均 R_phys={avg_reward:.2f}, 物理校验不达标。下次确认: SCF 收敛、基组完整、几何优化收敛。",
            "lammps_tool": f"LAMMPS 平均 R_phys={avg_reward:.2f}, 物理校验不达标。下次确认: 能量守恒、温度稳定、无原子丢失。",
        }
        return patches.get(tool)

    def _error_matches(self, pattern: str, error: str) -> bool:
        """Check if an error matches a learned pattern.

        pattern 已经是泛化后的 (含 <path>/<num> 占位符).
        把 error 也泛化后再比较, 否则 "File 'X' not found" 永远不匹配
        "File '<path>' not found".
        """
        from .logger import _generalize_error

        gen_error = _generalize_error(error)
        return pattern.lower() in gen_error.lower() or gen_error.lower() in pattern.lower()

    def _parse_fix_action(
        self, action: str, original_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse fix action string into a tool input dict.

        保证返回的 dict 一定含 description, 让 _try_evolved_fix 能拿到可重跑的描述.
        """
        try:
            parsed = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            # 不是 JSON, 当作 description 字符串
            return {"description": action}

        if isinstance(parsed, dict):
            if "description" not in parsed:
                parsed["description"] = str(parsed)
            return parsed

        return {"description": str(parsed)}
