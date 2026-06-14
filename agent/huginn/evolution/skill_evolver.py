"""Skill Evolver — extracts reusable skills from successful executions.

A "skill" in the Agent context is a reusable workflow pattern:
  - Trigger: what user query activates this skill
  - Steps: ordered tool calls with parameter templates
  - Validation: how to verify the skill worked
  - Metadata: source, confidence, usage stats

Unlike LLM fine-tuning which changes model weights,
skill evolution changes the Agent's BEHAVIOR REPERTOIRE.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class Skill:
    """A reusable, evolvable skill for the Agent."""
    skill_id: str
    name: str
    description: str
    domain: str  # e.g., "quantum_chemistry", "molecular_dynamics"
    trigger_patterns: List[str]  # Keywords/phrases that activate this skill
    workflow: List[Dict[str, Any]]  # Ordered steps [{"tool": "...", "params": {...}}]
    prerequisites: List[str] = field(default_factory=list)  # Required inputs
    expected_outputs: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)
    # Evolution metadata
    source: str = "manual"  # "manual", "extracted", "mutated"
    parent_skill: Optional[str] = None
    extraction_session: Optional[str] = None
    confidence: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        return cls(**data)


class SkillLibrary:
    """Persistent library of evolved skills."""

    def __init__(self, library_path: Optional[str] = None):
        self.library_path = Path(library_path) if library_path else Path.home() / ".huginn" / "skills.json"
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills: Dict[str, Skill] = {}
        self._load()

    def _load(self) -> None:
        if self.library_path.exists():
            with self.library_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    skill = Skill.from_dict(item)
                    self.skills[skill.skill_id] = skill

    def save(self) -> None:
        with self.library_path.open("w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.skills.values()], f, ensure_ascii=False, indent=2)

    def add(self, skill: Skill) -> None:
        self.skills[skill.skill_id] = skill
        self.save()

    def get(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)

    def find_by_trigger(self, query: str, top_k: int = 5) -> List[Skill]:
        """Find skills whose trigger patterns match the query."""
        query_lower = query.lower()
        scored = []
        for skill in self.skills.values():
            score = sum(2 if pat in query_lower else 1
                       for pat in skill.trigger_patterns
                       if pat in query_lower or query_lower in pat)
            if score > 0:
                scored.append((score * skill.confidence, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def find_by_domain(self, domain: str) -> List[Skill]:
        return [s for s in self.skills.values() if s.domain == domain]

    def get_high_confidence_skills(self, min_confidence: float = 0.7) -> List[Skill]:
        return [s for s in self.skills.values() if s.confidence >= min_confidence]

    def update_stats(self, skill_id: str, success: bool) -> None:
        skill = self.skills.get(skill_id)
        if skill:
            skill.usage_count += 1
            skill.last_used = datetime.now().isoformat()
            if success:
                skill.success_count += 1
            else:
                skill.failure_count += 1
            # Adjust confidence based on observed success rate
            skill.confidence = 0.7 * skill.confidence + 0.3 * skill.success_rate
            self.save()

    def mutate_skill(self, skill_id: str, mutation_type: str, params: Dict[str, Any]) -> Optional[Skill]:
        """Create a variant of an existing skill (exploration)."""
        parent = self.skills.get(skill_id)
        if not parent:
            return None

        new_id = f"{skill_id}_mut_{int(time.time() * 1000)}"
        new_skill = Skill(
            skill_id=new_id,
            name=f"{parent.name} (variant)",
            description=parent.description,
            domain=parent.domain,
            trigger_patterns=list(parent.trigger_patterns),
            workflow=list(parent.workflow),
            prerequisites=list(parent.prerequisites),
            expected_outputs=list(parent.expected_outputs),
            validation_checks=list(parent.validation_checks),
            source="mutated",
            parent_skill=skill_id,
            confidence=parent.confidence * 0.8,  # Lower confidence for mutants
        )

        if mutation_type == "add_step":
            step = params.get("step")
            if step:
                idx = params.get("index", len(new_skill.workflow))
                new_skill.workflow.insert(idx, step)
        elif mutation_type == "remove_step":
            idx = params.get("index")
            if idx is not None and 0 <= idx < len(new_skill.workflow):
                new_skill.workflow.pop(idx)
        elif mutation_type == "modify_param":
            step_idx = params.get("step_index", 0)
            param = params.get("param")
            value = params.get("value")
            if step_idx < len(new_skill.workflow) and param:
                new_skill.workflow[step_idx]["params"][param] = value
        elif mutation_type == "add_trigger":
            trigger = params.get("trigger")
            if trigger and trigger not in new_skill.trigger_patterns:
                new_skill.trigger_patterns.append(trigger)

        self.add(new_skill)
        return new_skill


class SkillExtractor:
    """Extracts skills from execution logs."""

    def __init__(self, library: SkillLibrary):
        self.library = library

    def extract_from_session(self, session_logs: List[Dict[str, Any]], min_success_rate: float = 0.8) -> List[Skill]:
        """Extract skills from a completed session's tool call logs."""
        # Group consecutive successful calls into workflows
        workflows = self._segment_workflows(session_logs)
        extracted = []

        for wf in workflows:
            if len(wf) < 2:
                continue  # Need at least 2 steps to be a workflow

            # Check if all steps succeeded
            success_rate = sum(1 for s in wf if s.get("success")) / len(wf)
            if success_rate < min_success_rate:
                continue

            # Extract domain from tool names
            domains = self._infer_domain(wf)
            domain = domains[0] if domains else "general"

            # Extract trigger keywords from inputs
            triggers = self._extract_triggers(wf)

            # Build workflow steps
            steps = []
            for step in wf:
                steps.append({
                    "tool": step.get("tool_name", "unknown"),
                    "params": {k: v for k, v in step.get("tool_input", {}).items() if not k.startswith("_")},
                    "purpose": step.get("purpose", ""),
                })

            skill_id = f"skill_{domain}_{hashlib.md5(json.dumps(steps, sort_keys=True).encode()).hexdigest()[:8]}"

            # Avoid duplicates
            if skill_id in self.library.skills:
                continue

            skill = Skill(
                skill_id=skill_id,
                name=f"Auto-extracted {domain} workflow",
                description=f"Extracted from session with {len(wf)} steps, {success_rate:.0%} success",
                domain=domain,
                trigger_patterns=triggers,
                workflow=steps,
                prerequisites=self._extract_prerequisites(wf),
                expected_outputs=self._extract_outputs(wf),
                source="extracted",
                extraction_session=wf[0].get("session_id"),
                confidence=success_rate * 0.9,
                tags=["auto_extracted", domain],
            )
            self.library.add(skill)
            extracted.append(skill)

        return extracted

    def _segment_workflows(self, logs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Segment a session log into individual workflow instances."""
        workflows = []
        current = []
        for log in logs:
            if log.get("tool_name") == "session_start" and current:
                workflows.append(current)
                current = []
            current.append(log)
        if current:
            workflows.append(current)
        return workflows if workflows else [logs]

    def _infer_domain(self, workflow: List[Dict[str, Any]]) -> List[str]:
        """Infer scientific domain from tool names and inputs."""
        domains = set()
        for step in workflow:
            tool = step.get("tool_name", "").lower()
            if any(x in tool for x in ["vasp", "dft", "band", "dos"]):
                domains.add("dft")
            if any(x in tool for x in ["gaussian", "orca", "fukui", "nci", "igmh"]):
                domains.add("quantum_chemistry")
            if any(x in tool for x in ["lammps", "md", "gromacs", "amber"]):
                domains.add("molecular_dynamics")
            if any(x in tool for x in ["tddft", "excited", "spectrum"]):
                domains.add("excited_state")
            if any(x in tool for x in ["multiwfn", "charge", "resp"]):
                domains.add("wavefunction_analysis")
        return list(domains)

    def _extract_triggers(self, workflow: List[Dict[str, Any]]) -> List[str]:
        """Extract likely trigger keywords from workflow inputs."""
        triggers = set()
        for step in workflow:
            inp = step.get("tool_input", {})
            for val in inp.values():
                if isinstance(val, str):
                    # Extract meaningful keywords
                    words = val.lower().split()
                    triggers.update(w for w in words if len(w) > 3 and w not in {
                        "calculate", "compute", "using", "with", "from", "file",
                        "path", "input", "output", "method", "basis",
                    })
        return list(triggers)[:10]  # Limit to top 10

    def _extract_prerequisites(self, workflow: List[Dict[str, Any]]) -> List[str]:
        """Extract required inputs for this workflow."""
        prereqs = set()
        for step in workflow:
            for key in step.get("tool_input", {}).keys():
                if key not in {"__auto_fixes", "__diagnosis"}:
                    prereqs.add(key)
        return list(prereqs)

    def _extract_outputs(self, workflow: List[Dict[str, Any]]) -> List[str]:
        """Extract expected outputs from result data."""
        outputs = set()
        for step in workflow:
            result = step.get("result_data", {})
            if isinstance(result, dict):
                outputs.update(result.keys())
        return list(outputs)


class SkillRanker:
    """Ranks skills by utility and evolutionary fitness."""

    @staticmethod
    def fitness(skill: Skill) -> float:
        """Compute evolutionary fitness of a skill."""
        if skill.usage_count == 0:
            return skill.confidence * 0.3  # Untested skills get low but non-zero

        # Fitness = success_rate * confidence * log(usage_count)
        import math
        usage_bonus = math.log1p(skill.usage_count)
        return skill.success_rate * skill.confidence * usage_bonus

    @staticmethod
    def select_skills(skills: List[Skill], n: int = 10) -> List[Skill]:
        """Select top-N skills by fitness (tournament selection style)."""
        ranked = [(SkillRanker.fitness(s), s) for s in skills]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in ranked[:n]]
