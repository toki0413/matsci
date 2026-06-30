"""Huginn agent question testing script.

Exercises core tools and agent flow with materials science questions
without requiring an external LLM provider.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "agent"))

from huginn.config import HuginnConfig
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext


def header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def make_context() -> ToolContext:
    return ToolContext(session_id="test", workspace=str(Path(".").resolve()))


def test_structure_tool() -> None:
    """Q1: Analyze a silicon crystal structure (FCC diamond)."""
    header("Q1: Structure analysis - Silicon diamond cubic")
    tmp = Path(tempfile.mkdtemp())
    poscar = tmp / "POSCAR"
    poscar.write_text(
        "Si diamond\n"
        "5.43\n"
        "0.0 0.5 0.5\n"
        "0.5 0.0 0.5\n"
        "0.5 0.5 0.0\n"
        "Si\n"
        "8\n"
        "Direct\n"
        "0.000 0.000 0.000\n"
        "0.250 0.250 0.250\n"
        "0.500 0.000 0.500\n"
        "0.750 0.250 0.750\n"
        "0.000 0.500 0.000\n"
        "0.250 0.750 0.250\n"
        "0.500 0.500 0.500\n"
        "0.750 0.750 0.750\n"
    )

    tool = ToolRegistry.get("structure_tool")
    if tool is None:
        print("  SKIP: structure_tool not registered")
        return

    ctx = make_context()
    args = tool.input_schema(action="analyze", file_path=str(poscar))
    result = asyncio.run(tool.call(args, ctx))
    print(f"  success: {result.success}")
    if result.success and result.data:
        d = result.data
        print(f"  formula: {d.get('formula')}")
        print(f"  spacegroup: {d.get('spacegroup')}")
        print(f"  lattice: {d.get('lattice_params')}")
        print(f"  num_atoms: {d.get('num_atoms')}")
        print(f"  volume: {d.get('volume'):.2f}")
        print(f"  density: {d.get('density'):.4f}")
        if d.get("num_atoms") == 8:
            print("  CHECK PASS: 8 atoms in conventional diamond cell")
        else:
            print(f"  CHECK WARN: expected 8 atoms, got {d.get('num_atoms')}")
    else:
        print(f"  error: {result.error}")


def test_code_tool_bandgap() -> None:
    """Q2: Compute semiconductor band gap statistics."""
    header("Q2: Code tool - band gap statistics")
    tool = ToolRegistry.get("code_tool")
    if tool is None:
        print("  SKIP: code_tool not registered")
        return

    code = (
        "import numpy as np\n"
        "materials = ['Si', 'Ge', 'GaAs', 'InP', 'GaP', 'ZnSe', 'CdTe']\n"
        "band_gaps = np.array([1.12, 0.67, 1.42, 1.35, 2.26, 2.70, 1.50])\n"
        "result = {\n"
        "    'mean': float(np.mean(band_gaps)),\n"
        "    'std': float(np.std(band_gaps)),\n"
        "    'min': float(np.min(band_gaps)),\n"
        "    'max': float(np.max(band_gaps)),\n"
        "    'direct_gap_count': int(np.sum(band_gaps > 1.0)),\n"
        "}\n"
        "print('Band gap statistics computed')\n"
    )
    ctx = make_context()
    result = tool.call(
        {
            "action": "execute",
            "code": code,
            "result_variable": "result",
            "timeout": 30.0,
        },
        ctx,
    )
    print(f"  success: {result.success}")
    if result.success:
        print(f"  stdout: {result.data.get('stdout', '').strip()[:200]}")
        rv = result.data.get("result_variable")
        if rv:
            print(f"  result: {rv}")
    else:
        print(f"  error: {result.error}")


def test_symbolic_math_tool() -> None:
    """Q3: Symbolic math - differentiate strain energy."""
    header("Q3: Symbolic math - Hooke's law derivation")
    tool = ToolRegistry.get("symbolic_math_tool")
    if tool is None:
        print("  SKIP: symbolic_math_tool not registered")
        return

    ctx = make_context()
    args = tool.input_schema(
        action="differentiate",
        expression="0.5 * E * epsilon**2",
        variable="epsilon",
        symbols=["E", "epsilon"],
    )
    result = asyncio.run(tool.call(args, ctx))
    print(f"  success: {result.success}")
    if result.success and result.data:
        print(f"  result: {str(result.data)[:300]}")
    else:
        print(f"  error: {result.error}")


def test_uq_tool() -> None:
    """Q4: Uncertainty quantification - Monte Carlo propagation."""
    header("Q4: UQ tool - Monte Carlo uncertainty propagation")
    tool = ToolRegistry.get("uq_tool")
    if tool is None:
        print("  SKIP: uq_tool not registered")
        return

    ctx = make_context()
    result = asyncio.run(tool.call(
        {
            "action": "monte_carlo",
            "expression": "E * epsilon",
            "variables": [
                {"name": "E", "distribution": "normal", "mean": 210.0, "std": 10.0},
                {"name": "epsilon", "distribution": "normal", "mean": 0.001, "std": 0.0001},
            ],
            "n_samples": 5000,
        },
        ctx,
    ))
    print(f"  success: {result.success}")
    if result.success and result.data:
        d = result.data
        print(f"  mean: {d.get('mean')}")
        print(f"  std: {d.get('std')}")
    else:
        print(f"  error: {result.error}")


def test_agent_flow_with_fake_model() -> None:
    """Q5: End-to-end agent flow with a fake LLM."""
    header("Q5: Agent flow - fake LLM, real tools")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    from huginn.agent import HuginnAgent
    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.manager import MemoryManager

    class FakeModel(BaseChatModel):
        responses: list[AIMessage]
        _index: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            resp = self.responses[self._index]
            self._index += 1
            return ChatResult(generations=[ChatGeneration(message=resp)])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop, run_manager, **kwargs)

        @property
        def _llm_type(self) -> str:
            return "fake"

        def bind_tools(self, tools, **kwargs):
            return self

    @tool
    def lattice_energy(a: float) -> float:
        """Compute Madelung energy for NaCl lattice (simplified)."""
        return -1.74756 * 1.44 / a

    tmp = Path(tempfile.mkdtemp())
    model = FakeModel(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "name": "lattice_energy",
                "args": {"a": 5.64},
                "id": "tc1",
            }],
        ),
        AIMessage(
            content="For NaCl with lattice constant a=5.64 Angstrom, the simplified "
                    "Madelung energy is approximately -0.447 eV per ion pair."
        ),
    ])
    memory = MemoryManager(longterm=LongTermMemory(str(tmp / "mem.db")))
    agent = HuginnAgent(
        model=model,
        tools=[lattice_energy],
        memory_manager=memory,
        checkpointer_path=str(tmp / "ckpt.sqlite"),
    )
    try:
        result = agent.invoke("Compute the Madelung energy for NaCl", thread_id="q5")
        msgs = result.get("messages", [])
        last = msgs[-1] if msgs else None
        if last and hasattr(last, "content"):
            print(f"  answer: {last.content}")
        session_msgs = agent.memory.session.messages
        print(f"  session messages: {len(session_msgs)}")
        print(f"  tool calls: {len(agent.memory.session.tool_calls)}")
        print("  CHECK PASS: agent flow completed")
    except Exception as e:
        print(f"  FAIL: {e}")
    finally:
        agent.close()


def test_memory_system() -> None:
    """Q6: Memory - store and recall a materials fact."""
    header("Q6: Memory - store/recall materials fact")
    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.manager import MemoryManager

    tmp = Path(tempfile.mkdtemp())
    memory = MemoryManager(longterm=LongTermMemory(str(tmp / "mem.db")))

    mid = memory.remember(
        "Diamond cubic silicon has an indirect band gap of 1.12 eV at 300K.",
        category="material_fact",
        tags=["silicon", "band_gap", "semiconductor"],
        importance=0.9,
    )
    print(f"  stored memory id: {mid}")

    # Test 1: exact substring match (should work)
    r1 = memory.recall("1.12 eV", top_k=3)
    print(f"  recall '1.12 eV': {len(r1)} entries")

    # Test 2: natural language query (LIKE fails, FTS5 would work)
    r2 = memory.recall("silicon band gap", top_k=3)
    print(f"  recall 'silicon band gap': {len(r2)} entries")

    # Test 3: single keyword
    r3 = memory.recall("silicon", top_k=3)
    print(f"  recall 'silicon': {len(r3)} entries")
    for r in r3:
        print(f"    - {r['content'][:80]}")

    if r3 and "1.12 eV" in r3[0]["content"]:
        print("  CHECK PASS: single-keyword recall works")
    elif r1 and not r2:
        print("  CHECK FAIL: multi-word query fails (LIKE substring limitation)")
        print("  BUG: FTS5 table exists but retrieve() uses LIKE instead of MATCH")
    else:
        print("  CHECK WARN: unexpected recall behavior")


def test_skills_registry() -> None:
    """Q7: Skills - list available preset workflows."""
    header("Q7: Skills registry - available workflows")
    from huginn.skills.registry import SkillRegistry

    skills = SkillRegistry.list_skills()
    print(f"  registered skills: {len(skills)}")
    for s in skills:
        print(f"    - {s}")


def test_validate_tool() -> None:
    """Q8: Validate - physics sanity check on a DFT result."""
    header("Q8: Validate tool - DFT result validation")
    tool = ToolRegistry.get("validate_tool")
    if tool is None:
        print("  SKIP: validate_tool not registered")
        return

    ctx = make_context()
    # Validate a mock DFT result with a negative formation energy
    args = tool.input_schema(
        result_type="dft",
        result_data={
            "energy": -10.5,
            "formation_energy": -0.5,
            "band_gap": 1.12,
            "converged": True,
            "forces_max": 0.01,
        },
    )
    result = asyncio.run(tool.call(args, ctx))
    print(f"  success: {result.success}")
    if result.success and result.data:
        d = result.data
        print(f"  all_passed: {d.get('all_passed')}")
        print(f"  summary: {d.get('summary')}")
        for chk in d.get("checks", [])[:5]:
            status = "PASS" if chk.get("passed") else "FAIL"
            print(f"    [{status}] {chk.get('name')}: {chk.get('message', '')[:60]}")
    else:
        print(f"  error: {result.error}")


def main() -> None:
    print("Huginn Agent Question Testing")
    print(f"Python: {sys.version.split()[0]}")

    cfg = HuginnConfig.from_env()
    register_all_tools(cfg)
    print(f"Tools registered: {len(ToolRegistry.list_tools())}")

    test_structure_tool()
    test_code_tool_bandgap()
    test_symbolic_math_tool()
    test_uq_tool()
    test_agent_flow_with_fake_model()
    test_memory_system()
    test_skills_registry()
    test_validate_tool()

    print(f"\n{'=' * 60}\nTesting complete\n{'=' * 60}")


if __name__ == "__main__":
    main()
