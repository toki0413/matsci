"""MatWorldBench — materials science domain benchmark.

Inspired by Qwen-AgentWorld's MatWorldBench concept: a fixed set of
real materials science questions with known answers, scored against
numerical tolerance bands.  The point is to give the agent a repeatable
"did you get the physics right" check that doesn't need an LLM judge.

Tasks cover five categories: structure / thermo / electronic /
mechanical / catalysis.  Expected values are literature / handbook
numbers; tolerances are deliberately loose enough that any decent DFT
or empirical estimate lands inside, but a hallucinated value won't.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# 五个领域, 跟 BenchTask.category 对应
CATEGORIES: tuple[str, ...] = (
    "structure", "thermo", "electronic", "mechanical", "catalysis",
)


@dataclass
class BenchTask:
    """单道 benchmark 题目.

    expected_result: {property_key: value} —— 跟 agent_output 同构.
    tolerance: {property_key: abs_band} —— 只对数值 key 有意义; 没列在
        tolerance 里的 key 按"严格相等"判 (string / bool 之类).
    """

    id: str
    category: str
    prompt: str
    expected_result: dict[str, Any]
    tolerance: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResult:
    """单题评测结果. score 是 0~1, 代表 expected_result 里有多少比例的
    key 落在容差带内."""

    task_id: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)


def _is_close(value: float, expected: float, tol: float) -> bool:
    # abs band 比较, 不用相对容差——bench 题的量级跨度太大, 绝对带更直观
    return abs(value - expected) <= tol


class MatWorldBench:
    """材料科学 benchmark 集合.

    evaluate(task_id, agent_output) 拿 agent 算出来的 dict 跟
    expected_result 逐 key 比对, 数值 key 走 tolerance, 其余严格相等.
    run_all(evaluator_fn) 把一个 callable 喂给所有题, 返回汇总.
    """

    TASKS: list[BenchTask] = [
        BenchTask(
            id="si_bandgap",
            category="electronic",
            prompt="Calculate the indirect band gap of crystalline silicon "
                   "(diamond structure) in eV.",
            expected_result={"band_gap_eV": 1.12},
            tolerance={"band_gap_eV": 0.15},
            metadata={"material": "Si", "structure": "diamond"},
        ),
        BenchTask(
            id="tio2_phase_stability",
            category="thermo",
            prompt="Determine the energy difference between rutile and anatase "
                   "TiO2 in eV per formula unit (rutile minus anatase).",
            expected_result={"delta_E_eV_per_fu": -0.06},
            tolerance={"delta_E_eV_per_fu": 0.05},
            metadata={"material": "TiO2", "phases": ["rutile", "anatase"]},
        ),
        BenchTask(
            id="cu_conductivity",
            category="electronic",
            prompt="Calculate the electrical conductivity of bulk copper at "
                   "room temperature in S/m.",
            expected_result={"conductivity_S_per_m": 5.96e7},
            tolerance={"conductivity_S_per_m": 1.0e7},
            metadata={"material": "Cu", "structure": "fcc"},
        ),
        BenchTask(
            id="fe_bcc_lattice",
            category="structure",
            prompt="Determine the equilibrium lattice constant of BCC iron "
                   "in angstroms.",
            expected_result={"lattice_constant_A": 2.866},
            tolerance={"lattice_constant_A": 0.05},
            metadata={"material": "Fe", "structure": "bcc"},
        ),
        BenchTask(
            id="mos2_bandgap",
            category="electronic",
            prompt="Calculate the band gap of monolayer MoS2 in eV.",
            expected_result={"band_gap_eV": 1.80},
            tolerance={"band_gap_eV": 0.20},
            metadata={"material": "MoS2", "form": "monolayer 2H"},
        ),
        BenchTask(
            id="diamond_bulk_modulus",
            category="mechanical",
            prompt="Calculate the bulk modulus of diamond in GPa.",
            expected_result={"bulk_modulus_GPa": 443.0},
            tolerance={"bulk_modulus_GPa": 30.0},
            metadata={"material": "C", "structure": "diamond"},
        ),
        BenchTask(
            id="pt_oxygen_adsorption",
            category="catalysis",
            prompt="Calculate the adsorption energy of atomic O on the Pt(111) "
                   "fcc hollow site in eV.",
            expected_result={"adsorption_energy_eV": -3.90},
            tolerance={"adsorption_energy_eV": 0.30},
            metadata={"material": "Pt", "surface": "111", "site": "fcc"},
        ),
        BenchTask(
            id="zno_bandgap",
            category="electronic",
            prompt="Calculate the band gap of wurtzite ZnO in eV.",
            expected_result={"band_gap_eV": 3.37},
            tolerance={"band_gap_eV": 0.30},
            metadata={"material": "ZnO", "structure": "wurtzite"},
        ),
        BenchTask(
            id="al_fcc_lattice",
            category="structure",
            prompt="Determine the equilibrium lattice constant of FCC "
                   "aluminum in angstroms.",
            expected_result={"lattice_constant_A": 4.05},
            tolerance={"lattice_constant_A": 0.05},
            metadata={"material": "Al", "structure": "fcc"},
        ),
        BenchTask(
            id="ni_cohesive_energy",
            category="thermo",
            prompt="Calculate the cohesive energy of FCC nickel in eV/atom.",
            expected_result={"cohesive_energy_eV_per_atom": 4.44},
            tolerance={"cohesive_energy_eV_per_atom": 0.20},
            metadata={"material": "Ni", "structure": "fcc"},
        ),
    ]

    def __init__(self, tasks: list[BenchTask] | None = None) -> None:
        # 允许传自定义题集, 默认用内置 TASKS
        self.tasks = tasks if tasks is not None else list(self.TASKS)
        self._by_id: dict[str, BenchTask] = {t.id: t for t in self.tasks}

    def get_task(self, task_id: str) -> BenchTask | None:
        return self._by_id.get(task_id)

    def evaluate(self, task_id: str, agent_output: dict[str, Any]) -> BenchResult:
        """比对单题. agent_output 的 key 要跟 expected_result 对齐.

        缺 key 算 fail, 多余 key 忽略. 数值 key 走 tolerance 绝对带,
        非 numerIC key (string/bool) 严格相等.
        """
        task = self._by_id.get(task_id)
        if task is None:
            return BenchResult(
                task_id=task_id, passed=False, score=0.0,
                details={"error": f"unknown task_id: {task_id}"},
            )

        if not isinstance(agent_output, dict):
            return BenchResult(
                task_id=task_id, passed=False, score=0.0,
                details={"error": "agent_output must be a dict"},
            )

        key_results: dict[str, dict[str, Any]] = {}
        n_pass = 0
        for key, expected in task.expected_result.items():
            got = agent_output.get(key)
            if got is None:
                key_results[key] = {
                    "expected": expected, "got": None, "pass": False,
                    "reason": "missing key",
                }
                continue

            tol = task.tolerance.get(key)
            if tol is not None and isinstance(expected, (int, float)) \
                    and isinstance(got, (int, float)):
                ok = _is_close(float(got), float(expected), tol)
            else:
                # 非数值或没给容差 -> 严格相等
                ok = got == expected

            key_results[key] = {
                "expected": expected, "got": got, "pass": ok,
            }
            if ok:
                n_pass += 1

        total = len(task.expected_result)
        score = n_pass / total if total else 0.0
        return BenchResult(
            task_id=task_id, passed=(n_pass == total), score=round(score, 4),
            details={"keys": key_results, "category": task.category},
        )

    def run_all(
        self,
        evaluator_fn: Callable[[BenchTask], dict[str, Any]],
    ) -> dict[str, Any]:
        """跑全部题. evaluator_fn 接收 BenchTask, 返回 agent_output dict.

        返回汇总: passed / failed 计数 + 每题 BenchResult.
        """
        results: list[BenchResult] = []
        n_pass = 0
        for task in self.tasks:
            try:
                output = evaluator_fn(task) or {}
            except Exception as exc:  # ponytail: 一个题炸了别让整批挂
                output = {"__error__": str(exc)}
            res = self.evaluate(task.id, output)
            results.append(res)
            if res.passed:
                n_pass += 1

        return {
            "total": len(self.tasks),
            "passed": n_pass,
            "failed": len(self.tasks) - n_pass,
            "pass_rate": round(n_pass / len(self.tasks), 4) if self.tasks else 0.0,
            "results": results,
        }


__all__ = [
    "CATEGORIES",
    "BenchTask",
    "BenchResult",
    "MatWorldBench",
]
