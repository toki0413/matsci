#!/usr/bin/env python3
"""
Unified verification script for all Huginn + Sobko optimizations.

Checks that every component is properly wired and accessible.

Usage:
    python scripts/verify_all_optimizations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Sobko may be sibling to huginn-agent or inside it
SOBKO_CANDIDATES = [
    REPO_ROOT.parent / "Sobko_MCP_project",
    REPO_ROOT / "Sobko_MCP_project",
    Path.home() / "Sobko_MCP_project",
]
SOBKO_ROOT = next((p for p in SOBKO_CANDIDATES if p.exists()), SOBKO_CANDIDATES[0])
AGENT_PKG = REPO_ROOT / "agent" / "huginn"


def check_file(path: Path, label: str) -> bool:
    ok = path.exists()
    status = "OK" if ok else "MISSING"
    print(f"  [{status}] {label}: {path}")
    return ok


def main():
    print("=" * 70)
    print("Huginn + Sobko Optimization Verification")
    print("=" * 70)

    all_ok = True

    # ------------------------------------------------------------------
    # 1. Sobko Database (Cleaned)
    # ------------------------------------------------------------------
    print("\n[1/7] Sobko Cleaned Database")
    cleaned = SOBKO_ROOT / "cleaned"
    all_ok &= check_file(cleaned / "chunks.jsonl", "chunks.jsonl")
    all_ok &= check_file(cleaned / "source_registry.jsonl", "source_registry.jsonl")
    all_ok &= check_file(cleaned / "sections.jsonl", "sections.jsonl")
    all_ok &= check_file(cleaned / "images.jsonl", "images.jsonl")
    all_ok &= check_file(cleaned / "quality_report.json", "quality_report.json")

    # Check that normalized/ is synced with cleaned/
    normalized = SOBKO_ROOT / "normalized"
    norm_chunks = normalized / "chunks.jsonl"
    clean_chunks = cleaned / "chunks.jsonl"
    if norm_chunks.exists() and clean_chunks.exists():
        norm_size = norm_chunks.stat().st_size
        clean_size = clean_chunks.stat().st_size
        synced = abs(norm_size - clean_size) < 1024
        status = "SYNCED" if synced else "OUT OF SYNC"
        print(f"  [{status}] normalized/ vs cleaned/ chunks")
        all_ok &= synced

    # ------------------------------------------------------------------
    # 2. Advanced Optimization Assets
    # ------------------------------------------------------------------
    print("\n[2/7] Advanced Optimization Assets")
    adv = SOBKO_ROOT / "advanced_optimization"
    all_ok &= check_file(adv / "knowledge_graph.json", "knowledge_graph.json")
    all_ok &= check_file(adv / "benchmark.json", "benchmark.json")
    all_ok &= check_file(adv / "lora_training_full.jsonl", "lora_training_full.jsonl")
    all_ok &= check_file(adv / "faq_qa.jsonl", "faq_qa.jsonl")
    all_ok &= check_file(adv / "hierarchical_index.json", "hierarchical_index.json")
    all_ok &= check_file(adv / "workflow_templates.jsonl", "workflow_templates.jsonl")
    all_ok &= check_file(adv / "troubleshooting_by_software.json", "troubleshooting_by_software.json")
    all_ok &= check_file(adv / "evaluation_report.json", "evaluation_report.json")

    # ------------------------------------------------------------------
    # 3. Huginn Code Modifications
    # ------------------------------------------------------------------
    print("\n[3/7] Huginn Code Modifications")
    all_ok &= check_file(AGENT_PKG / "prompts.py", "prompts.py (enhanced)")
    all_ok &= check_file(AGENT_PKG / "skills" / "wavefunction_analysis.md", "wavefunction_analysis.md skill")
    all_ok &= check_file(AGENT_PKG / "tools" / "diagnose_tool.py", "diagnose_tool.py")
    all_ok &= check_file(AGENT_PKG / "rag" / "router_retriever.py", "router_retriever.py")
    all_ok &= check_file(AGENT_PKG / "workflows" / "templates_qc.py", "templates_qc.py")

    # Check that prompts.py contains Sobko knowledge
    prompts_path = AGENT_PKG / "prompts.py"
    if prompts_path.exists():
        content = prompts_path.read_text(encoding="utf-8")
        has_sobko = "Fukui" in content or "福井函数" in content or "IGMH" in content
        status = "OK" if has_sobko else "MISSING CONTENT"
        print(f"  [{status}] prompts.py contains Sobko knowledge injection")
        all_ok &= has_sobko

    # ------------------------------------------------------------------
    # 4. Python Import Tests
    # ------------------------------------------------------------------
    print("\n[4/7] Python Import Tests")
    sys.path.insert(0, str(REPO_ROOT / "agent"))

    try:
        from huginn.workflows.templates import list_templates, get_template
        templates = list_templates()
        qc_templates = ["wavefunction_analysis", "reactivity_prediction", "weak_interaction", "excited_state", "charge_analysis"]
        missing = [t for t in qc_templates if not get_template(t)]
        if missing:
            print(f"  [FAIL] Missing QC templates: {missing}")
            all_ok = False
        else:
            print(f"  [OK] All {len(qc_templates)} QC templates registered")
    except Exception as e:
        print(f"  [FAIL] Workflow templates import: {e}")
        all_ok = False

    try:
        from huginn.tools.diagnose_tool import DiagnoseTool
        print(f"  [OK] diagnose_tool importable")
    except Exception as e:
        print(f"  [FAIL] diagnose_tool import: {e}")
        all_ok = False

    try:
        from huginn.rag.router_retriever import HierarchicalRetriever
        print(f"  [OK] router_retriever importable")
    except Exception as e:
        print(f"  [FAIL] router_retriever import: {e}")
        all_ok = False

    try:
        from huginn.workflows.engine import RetryPolicy
        rp = RetryPolicy(auto_diagnose=True, apply_auto_fix=True)
        print(f"  [OK] Self-healing RetryPolicy configurable")
    except Exception as e:
        print(f"  [FAIL] Self-healing RetryPolicy: {e}")
        all_ok = False

    # ------------------------------------------------------------------
    # 5. Knowledge Graph Integrity
    # ------------------------------------------------------------------
    print("\n[5/7] Knowledge Graph Integrity")
    kg_path = adv / "knowledge_graph.json"
    if kg_path.exists():
        with kg_path.open("r", encoding="utf-8") as f:
            kg = json.load(f)
        n_entities = len(kg.get("entities", []))
        n_relations = len(kg.get("relations", []))
        print(f"  [OK] Entities: {n_entities}, Relations: {n_relations}")
        if n_entities < 10 or n_relations < 50:
            print(f"  [WARN] Knowledge graph seems small")
    else:
        all_ok = False

    # ------------------------------------------------------------------
    # 6. Benchmark Integrity
    # ------------------------------------------------------------------
    print("\n[6/7] Benchmark Integrity")
    bm_path = adv / "benchmark.json"
    if bm_path.exists():
        with bm_path.open("r", encoding="utf-8") as f:
            bm = json.load(f)
        print(f"  [OK] Benchmark questions: {len(bm)}")
        cats = {}
        diffs = {}
        for q in bm:
            cats[q.get("category", "unknown")] = cats.get(q.get("category"), 0) + 1
            diffs[q.get("difficulty", "unknown")] = diffs.get(q.get("difficulty"), 0) + 1
        print(f"       Categories: {cats}")
        print(f"       Difficulties: {diffs}")
    else:
        all_ok = False

    # ------------------------------------------------------------------
    # 7. Training Data Integrity
    # ------------------------------------------------------------------
    print("\n[7/7] Training Data Integrity")
    train_path = adv / "lora_training_full.jsonl"
    if train_path.exists():
        count = 0
        categories = {}
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count += 1
                rec = json.loads(line)
                cat = rec.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
        print(f"  [OK] Training examples: {count}")
        print(f"       Categories: {categories}")
        if count < 100:
            print(f"  [WARN] Training data seems small")
    else:
        all_ok = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_ok:
        print("ALL CHECKS PASSED — Optimization fully wired and ready to use")
    else:
        print("SOME CHECKS FAILED — Review output above for details")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
