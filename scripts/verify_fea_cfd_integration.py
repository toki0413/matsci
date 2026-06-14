"""
Verification script for FEA/CFD integration into Huginn.

Checks:
  1. Prompts contain FEA/CFD knowledge
  2. Workflow templates are registered
  3. Diagnose tool covers FEA/CFD software
  4. Router retriever routes FEA/CFD queries
  5. Skills files exist
  6. Knowledge graph includes FEA/CFD entities
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add agent to path
AGENT_ROOT = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

REPO_ROOT = Path(__file__).resolve().parent.parent
SOBKO_ROOT = Path("C:/Users/wanzh/Sobko_MCP_project")


def check_prompts() -> bool:
    """Check that prompts.py contains FEA/CFD knowledge."""
    print("\n[1] Checking prompts.py for FEA/CFD knowledge...")
    prompts_path = REPO_ROOT / "agent" / "huginn" / "prompts.py"
    with open(prompts_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_keywords = [
        "ABAQUS", "OpenFOAM", "finite element", "CFD",
        "crystal plasticity", "turbulence", "RANS", "LES",
        "y+", "mesh convergence", "Navier-Stokes",
        "conjugate heat transfer", "Multiphase", "cohesive zone",
    ]
    missing = [kw for kw in required_keywords if kw not in content]
    if missing:
        print(f"  [FAIL] Missing keywords: {missing}")
        return False
    print(f"  [PASS] All {len(required_keywords)} FEA/CFD keywords found in prompts")
    return True


def check_workflow_templates() -> bool:
    """Check that FEA/CFD workflow templates are registered."""
    print("\n[2] Checking FEA/CFD workflow templates...")

    # Import templates module (which auto-registers)
    try:
        from huginn.workflows import templates
        from huginn.workflows.templates import WORKFLOW_TEMPLATES as _TEMPLATES
    except Exception as e:
        print(f"  [WARN] Could not import templates: {e}")
        return False

    expected_fea = ["structural_analysis", "crystal_plasticity", "fracture_mechanics"]
    expected_cfd = ["turbulent_flow", "multiphase_flow", "conjugate_heat_transfer"]
    expected = expected_fea + expected_cfd

    registered = list(_TEMPLATES.keys())
    missing = [t for t in expected if t not in registered]

    if missing:
        print(f"  [FAIL] Missing templates: {missing}")
        return False

    print(f"  [PASS] All {len(expected)} FEA/CFD templates registered")
    print(f"     FEA: {expected_fea}")
    print(f"     CFD: {expected_cfd}")
    return True


def check_diagnose_tool() -> bool:
    """Check diagnose tool covers FEA/CFD software."""
    print("\n[3] Checking diagnose_tool for FEA/CFD coverage...")
    tool_path = REPO_ROOT / "agent" / "huginn" / "tools" / "diagnose_tool.py"
    with open(tool_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_software = ["abaqus", "ansys", "openfoam", "fluent", "comsol"]
    missing = [sw for sw in required_software if sw not in content.lower()]
    if missing:
        print(f"  [FAIL] Missing software checks: {missing}")
        return False

    # Check for FEA/CFD-specific advice
    fea_advice = ["mesh", "boundary condition", "element", "contact"]
    cfd_advice = ["y+", "courant", "turbulence", "boundary"]
    has_fea = any(kw in content.lower() for kw in fea_advice)
    has_cfd = any(kw in content.lower() for kw in cfd_advice)

    if not has_fea:
        print("  [FAIL] Missing FEA-specific advice")
        return False
    if not has_cfd:
        print("  [FAIL] Missing CFD-specific advice")
        return False

    print(f"  [PASS] Diagnose tool covers {len(required_software)} FEA/CFD software")
    print(f"  [PASS] FEA advice: {has_fea}, CFD advice: {has_cfd}")
    return True


def check_router_retriever() -> bool:
    """Check router retriever routes FEA/CFD queries."""
    print("\n[4] Checking router_retriever for FEA/CFD routing...")
    retriever_path = REPO_ROOT / "agent" / "huginn" / "rag" / "router_retriever.py"
    with open(retriever_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_sw = ["ABAQUS", "ANSYS", "OpenFOAM", "Fluent", "COMSOL", "DAMASK"]
    required_methods = ["FEA", "CPFEM", "RANS", "LES", "Multiphase", "HeatTransfer"]

    missing_sw = [sw for sw in required_sw if sw not in content]
    missing_method = [m for m in required_methods if m not in content]

    if missing_sw:
        print(f"  [FAIL] Missing software keywords: {missing_sw}")
        return False
    if missing_method:
        print(f"  [FAIL] Missing method keywords: {missing_method}")
        return False

    print(f"  [PASS] Router retriever covers {len(required_sw)} FEA/CFD software")
    print(f"  [PASS] Router retriever covers {len(required_methods)} FEA/CFD methods")
    return True


def check_skills() -> bool:
    """Check FEA/CFD skill files exist."""
    print("\n[5] Checking FEA/CFD skill files...")
    skills_dir = REPO_ROOT / "skills"
    expected = ["solid_mechanics.md", "cfd.md"]

    missing = []
    for skill in expected:
        path = skills_dir / skill
        if not path.exists():
            missing.append(skill)
        else:
            size = path.stat().st_size
            print(f"  [PASS] {skill} ({size} bytes)")

    if missing:
        print(f"  [FAIL] Missing skills: {missing}")
        return False
    return True


def check_knowledge_graph() -> bool:
    """Check knowledge graph includes FEA/CFD entities."""
    print("\n[6] Checking knowledge graph for FEA/CFD entities...")
    kg_path = SOBKO_ROOT / "advanced_optimization" / "knowledge_graph.json"
    if not kg_path.exists():
        print(f"  [WARN] Knowledge graph not found at {kg_path}")
        return False

    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    entities = kg.get("entities", [])
    relations = kg.get("relations", [])

    # Check for FEA/CFD software
    fea_software = ["ABAQUS", "ANSYS", "COMSOL", "DAMASK", "OpenFOAM", "Fluent"]
    found_sw = [e["name"] for e in entities if e["name"] in fea_software]

    # Check for FEA/CFD methods
    fea_methods = ["FEA", "CPFEM", "RANS", "LES", "Multiphase", "Euler-Euler"]
    found_methods = [e["name"] for e in entities if e["name"] in fea_methods]

    # Check for FEA/CFD concepts
    fea_concepts = ["Solid Mechanics", "Fluid Mechanics", "Turbulence", "Crystal Plasticity"]
    found_concepts = [e["name"] for e in entities if e["name"] in fea_concepts]

    # Check relations
    fea_relations = [r for r in relations if r["source"] in fea_software or r["target"] in fea_concepts]

    print(f"  [PASS] Entities: {len(entities)} total")
    print(f"  [PASS] Relations: {len(relations)} total")
    print(f"  [PASS] FEA/CFD software: {found_sw}")
    print(f"  [PASS] FEA/CFD methods: {found_methods}")
    print(f"  [PASS] FEA/CFD concepts: {found_concepts}")
    print(f"  [PASS] FEA/CFD relations: {len(fea_relations)}")

    if len(found_sw) < 4 or len(found_methods) < 3 or len(fea_relations) < 10:
        print("  [FAIL] Insufficient FEA/CFD coverage in knowledge graph")
        return False
    return True


def main():
    print("=" * 60)
    print("FEA/CFD Integration Verification")
    print("=" * 60)

    results = []
    results.append(("Prompts", check_prompts()))
    results.append(("Workflow Templates", check_workflow_templates()))
    results.append(("Diagnose Tool", check_diagnose_tool()))
    results.append(("Router Retriever", check_router_retriever()))
    results.append(("Skills", check_skills()))
    results.append(("Knowledge Graph", check_knowledge_graph()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n[SUCCESS] All FEA/CFD integration checks PASSED!")
        return 0
    else:
        print(f"\n[WARN] {total - passed} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
