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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .logger import ExecutionLogger


@dataclass
class EvolutionRule:
    """A learned rule for improving agent behavior."""

    rule_id: str
    rule_type: str  # "prompt_patch", "tool_strategy", "heuristic_fix", "skill_template"
    trigger: str  # Condition that activates this rule
    action: str  # What to do when triggered
    source: str  # How was this rule learned: "failure_analysis", "success_extraction", "user_feedback"
    confidence: float = 0.0  # 0-1, how reliable is this rule
    usage_count: int = 0
    success_count: int = 0
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
        logger: ExecutionLogger,
        rules_path: str | None = None,
        skills_path: str | None = None,
    ):
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
        self._load_rules()
        self._load_skills()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        if self.rules_path.exists():
            with self.rules_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.rules = [EvolutionRule(**r) for r in data]

    def _save_rules(self) -> None:
        with self.rules_path.open("w", encoding="utf-8") as f:
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
        with self.skills_path.open("w", encoding="utf-8") as f:
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

        if new_rules:
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
        self, tool_name: str, tool_input: dict[str, Any], error: str
    ) -> dict[str, Any] | None:
        """Check if we have a learned fix for this error and apply it."""
        for rule in self.rules:
            if rule.rule_type != "heuristic_fix":
                continue
            if rule.trigger.startswith(f"{tool_name}|") and self._error_matches(
                rule.trigger.split("|", 1)[1], error
            ):
                rule.usage_count += 1
                return self._parse_fix_action(rule.action, tool_input)
        return None

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

        # General fixes
        if "timeout" in error_lower:
            return '{"timeout": "increase", "resource": "check"}'
        if "permission" in error_lower or "access" in error_lower:
            return '{"permissions": "check", "path": "verify"}'
        if "file" in error_lower and (
            "not found" in error_lower or "missing" in error_lower
        ):
            return '{"files": "check_existence", "paths": "verify"}'

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
        """Check if an error matches a learned pattern."""
        # Simple substring matching, can be enhanced with embeddings
        return pattern.lower() in error.lower() or error.lower() in pattern.lower()

    def _parse_fix_action(
        self, action: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse a fix action string and merge with existing input."""
        try:
            fix = json.loads(action)
            merged = dict(tool_input)
            merged.update(fix)
            return merged
        except json.JSONDecodeError:
            # If not valid JSON, treat as text instruction
            return {**tool_input, "__evolution_fix": action}
