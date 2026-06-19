"""Unit tests for huginn/evolution/engine.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from huginn.evolution.engine import EvolutionEngine, EvolutionRule, SkillTemplate


@dataclass
class _StubRecord:
    session_id: str
    tool_name: str
    tool_input: dict
    success: bool = False
    error_message: str | None = None
    software: str | None = None
    calculation_type: str | None = None


class _StubLogger:
    """Minimal logger stub required by EvolutionEngine."""

    def __init__(self, persist_dir: str, tool_calls: list | None = None):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._tool_calls = tool_calls or []

    def get_failure_patterns(self, min_count: int = 2):
        from collections import Counter

        failures = [r for r in self._tool_calls if not r.success]
        counts = Counter()
        for f in failures:
            key = f"{f.tool_name}|{f.error_message or ''}"
            counts[key] += 1
        return [
            {"pattern": p, "tool": p.split("|", 1)[0], "error": p.split("|", 1)[1], "count": c}
            for p, c in counts.most_common()
            if c >= min_count
        ]

    def get_tool_success_rate(self):
        from collections import defaultdict

        stats = defaultdict(lambda: {"success": 0, "total": 0})
        for r in self._tool_calls:
            stats[r.tool_name]["total"] += 1
            if r.success:
                stats[r.tool_name]["success"] += 1
        return {
            t: s["success"] / s["total"] if s["total"] else 0.0
            for t, s in stats.items()
        }


@pytest.fixture
def logger(tmp_path):
    return _StubLogger(str(tmp_path / "logs"))


class TestEvolutionEngineInit:
    def test_default_paths(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        assert engine.rules_path == logger.persist_dir / "evolution_rules.json"
        assert engine.skills_path == logger.persist_dir / "evolved_skills.json"

    def test_custom_paths(self, tmp_path, logger):
        rules = tmp_path / "rules.json"
        skills = tmp_path / "skills.json"
        engine = EvolutionEngine(logger, rules_path=str(rules), skills_path=str(skills))
        assert engine.rules_path == rules
        assert engine.skills_path == skills

    def test_loads_existing_rules_and_skills(self, tmp_path, logger):
        rules = tmp_path / "rules.json"
        rules.write_text(
            json.dumps(
                [
                    {
                        "rule_id": "r1",
                        "rule_type": "heuristic_fix",
                        "trigger": "vasp_tool|convergence",
                        "action": '{"ALGO": "Normal"}',
                        "source": "failure_analysis",
                    }
                ]
            ),
            encoding="utf-8",
        )
        skills = tmp_path / "skills.json"
        skills.write_text(
            json.dumps(
                [
                    {
                        "skill_id": "s1",
                        "name": "Test",
                        "description": "desc",
                        "trigger_keywords": ["vasp"],
                        "workflow_steps": [],
                        "required_tools": [],
                        "source_session": "sess",
                    }
                ]
            ),
            encoding="utf-8",
        )
        engine = EvolutionEngine(
            logger, rules_path=str(rules), skills_path=str(skills)
        )
        assert len(engine.rules) == 1
        assert len(engine.skills) == 1


class TestPersistence:
    def test_save_and_load_rules(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        engine.rules.append(
            EvolutionRule(
                rule_id="r1",
                rule_type="prompt_patch",
                trigger="t1",
                action="a1",
                source="user_feedback",
            )
        )
        engine._save_rules()

        engine2 = EvolutionEngine(logger)
        assert len(engine2.rules) == 1
        assert engine2.rules[0].rule_id == "r1"

    def test_save_and_load_skills(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        engine.skills.append(
            SkillTemplate(
                skill_id="s1",
                name="Test Skill",
                description="desc",
                trigger_keywords=["md"],
                workflow_steps=[{"tool": "t"}],
                required_tools=["t"],
                source_session="sess",
            )
        )
        engine._save_skills()

        engine2 = EvolutionEngine(logger)
        assert len(engine2.skills) == 1
        assert engine2.skills[0].skill_id == "s1"


class TestEvolveFromFailures:
    def test_creates_failure_rule(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord("s", "vasp_tool", {}, False, "electronic convergence failed"),
            _StubRecord("s", "vasp_tool", {}, False, "electronic convergence failed"),
        ]
        engine = EvolutionEngine(logger)
        new_rules = engine.evolve_from_failures()
        assert len(new_rules) == 1
        assert new_rules[0].rule_type == "heuristic_fix"
        assert "vasp_tool" in new_rules[0].tags
        assert engine.rules_path.exists()

    def test_skips_existing_trigger(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord("s", "vasp_tool", {}, False, "electronic convergence failed"),
            _StubRecord("s", "vasp_tool", {}, False, "electronic convergence failed"),
        ]
        engine = EvolutionEngine(logger)
        engine.evolve_from_failures()
        second_run = engine.evolve_from_failures()
        assert len(second_run) == 0


class TestEvolveFromSuccesses:
    def test_extracts_skill(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord(
                "s",
                "vasp_tool",
                {"action": "relax"},
                True,
                software="VASP",
                calculation_type="geometry_optimization",
            )
            for _ in range(3)
        ]
        engine = EvolutionEngine(logger)
        new_skills = engine.evolve_from_successes()
        assert len(new_skills) == 1
        assert "VASP" in new_skills[0].trigger_keywords
        assert engine.skills_path.exists()

    def test_insufficient_records(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord(
                "s",
                "vasp_tool",
                {},
                True,
                software="VASP",
                calculation_type="DFT",
            )
            for _ in range(2)
        ]
        engine = EvolutionEngine(logger)
        assert len(engine.evolve_from_successes()) == 0


class TestEvolvePromptPatches:
    def test_creates_patch_for_low_success_tool(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord("s", "vasp_tool", {}, False),
            _StubRecord("s", "vasp_tool", {}, True),
        ]
        engine = EvolutionEngine(logger)
        patches = engine.evolve_prompt_patches()
        assert len(patches) == 1
        assert patches[0].rule_type == "prompt_patch"
        assert "vasp_tool" in patches[0].tags

    def test_no_patch_for_high_success_tool(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord("s", "vasp_tool", {}, True),
            _StubRecord("s", "vasp_tool", {}, True),
        ]
        engine = EvolutionEngine(logger)
        assert len(engine.evolve_prompt_patches()) == 0


class TestFullCycle:
    def test_run_full_evolution_cycle(self, tmp_path, logger):
        logger._tool_calls = [
            _StubRecord("s", "vasp_tool", {}, False, "electronic convergence failed"),
            _StubRecord("s", "vasp_tool", {}, False, "electronic convergence failed"),
            *[
                _StubRecord(
                    "s",
                    "vasp_tool",
                    {"action": "relax"},
                    True,
                    software="VASP",
                    calculation_type="geometry_optimization",
                )
                for _ in range(3)
            ],
        ]
        engine = EvolutionEngine(logger)
        report = engine.run_full_evolution_cycle()
        assert "failure_rules" in report
        assert "success_skills" in report
        assert "prompt_patches" in report
        assert "total_rules_after" in report
        assert "total_skills_after" in report
        assert (logger.persist_dir / "evolution_report.json").exists()
        assert (logger.persist_dir / "evolution_history.json").exists()


class TestRuntimeApplication:
    def test_apply_heuristic_fix_match(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        engine.rules.append(
            EvolutionRule(
                rule_id="fix1",
                rule_type="heuristic_fix",
                trigger="vasp_tool|electronic convergence failed",
                action='{"ALGO": "Normal"}',
                source="failure_analysis",
            )
        )
        fixed = engine.apply_heuristic_fix(
            "vasp_tool", {"ENCUT": 400}, "electronic convergence failed"
        )
        assert fixed == {"ENCUT": 400, "ALGO": "Normal"}

    def test_apply_heuristic_fix_no_match(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        engine.rules.append(
            EvolutionRule(
                rule_id="fix1",
                rule_type="heuristic_fix",
                trigger="vasp_tool|electronic convergence failed",
                action='{"ALGO": "Normal"}',
                source="failure_analysis",
            )
        )
        assert (
            engine.apply_heuristic_fix("lammps_tool", {}, "lost atoms") is None
        )

    def test_get_relevant_skills(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        engine.skills.append(
            SkillTemplate(
                skill_id="s1",
                name="MD Skill",
                description="desc",
                trigger_keywords=["lammps", "md"],
                workflow_steps=[],
                required_tools=[],
                source_session="sess",
            )
        )
        skills = engine.get_relevant_skills("run a LAMMPS MD simulation")
        assert len(skills) == 1
        assert skills[0].skill_id == "s1"

    def test_get_prompt_patches(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        engine.rules.append(
            EvolutionRule(
                rule_id="p1",
                rule_type="prompt_patch",
                trigger="low_success",
                action="patch text",
                source="success_analysis",
                confidence=0.8,
            )
        )
        assert engine.get_prompt_patches() == ["patch text"]


class TestInternalGenerators:
    def test_generate_heuristic_fix_vasp(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        assert "ALGO" in engine._generate_heuristic_fix("vasp_tool", "electronic convergence")
        assert "IBRION" in engine._generate_heuristic_fix("vasp_tool", "ionic relaxation")
        assert "NCORE" in engine._generate_heuristic_fix("vasp_tool", "out of memory")

    def test_generate_heuristic_fix_gaussian(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        assert "scf" in engine._generate_heuristic_fix("gaussian_tool", "SCF convergence")
        assert "basis" in engine._generate_heuristic_fix("gaussian_tool", "Missing basis")
        assert "opt" in engine._generate_heuristic_fix("gaussian_tool", "optimization")

    def test_generate_heuristic_fix_lammps(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        assert "timestep" in engine._generate_heuristic_fix("lammps_tool", "lost atoms")
        assert "fix_shake" in engine._generate_heuristic_fix("lammps_tool", "bond error")
        assert "fix_nvt" in engine._generate_heuristic_fix("lammps_tool", "thermo temp")

    def test_generate_heuristic_fix_general(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        assert "timeout" in engine._generate_heuristic_fix("generic_tool", "timeout")
        assert "files" in engine._generate_heuristic_fix("generic_tool", "file not found")

    def test_generate_prompt_patch_for_tool(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        patch = engine._generate_prompt_patch_for_tool("vasp_tool", 0.5)
        assert patch is not None and "VASP" in patch
        assert engine._generate_prompt_patch_for_tool("unknown_tool", 0.5) is None

    def test_error_matches(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        assert engine._error_matches("convergence", "SCF convergence failed")
        assert engine._error_matches("lost atoms", "lost atoms")
        assert not engine._error_matches("timeout", "convergence failed")

    def test_parse_fix_action_json(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        merged = engine._parse_fix_action('{"ALGO": "Normal"}', {"ENCUT": 400})
        assert merged == {"ENCUT": 400, "ALGO": "Normal"}

    def test_parse_fix_action_non_json(self, tmp_path, logger):
        engine = EvolutionEngine(logger)
        merged = engine._parse_fix_action("increase memory", {"ENCUT": 400})
        assert merged["ENCUT"] == 400
        assert merged["__evolution_fix"] == "increase memory"
