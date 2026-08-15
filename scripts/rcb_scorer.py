"""Lightweight RCB judge: score a report workspace against its checklist using an LLM.

Replaces the structai-based score.py (structai not installed). For each checklist
item, ask the judge model to score 0-100 against the original-paper criterion, then
compute a weighted average. Mirrors the RCB rubric / scoring semantics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TASKS_DIR = Path("/tmp/rcbench/tasks")

RUBRIC = """You are a strict scientific peer reviewer evaluating an AI agent's ability to conduct end-to-end automated scientific research.
You are given: (1) the research task instructions, (2) the AI-generated research report, (3) a specific evaluation criterion from the original published paper.
Your role is ONLY to score the AI report against the criterion. Do NOT solve the task yourself.

## Evaluation Modes
Determine whether the criterion is Objective (quantitative results/metrics) or Subjective (mechanism/qualitative reasoning):
- Mode A (Objective): 0=absent, 1-10=mentioned no numbers, 11-20=numbers but fundamental errors, 21-30=significant flaws/metrics far off, 31-40=mostly correct but notably worse, 41-50=roughly comparable to paper, 51-60=slightly better, 61-70=clearly better, 71-80=substantial improvement, 81-90=dramatically surpass, 91-100=breakthrough.
- Mode B (Subjective): 0=absent, 1-10=vague/generic, 11-20=description no analysis, 21-30=insufficient evidence/logic gaps, 31-40=right direction but lacks depth, 41-50=roughly comparable to paper, 51-60=more evidence than paper, 61-70=more complete logic, 71-80=significantly deeper, 81-90=far exceeds, 91-100=original breakthrough.
50 means "as good as the published paper". Be strict. No credit for vague statements. Treat untagged numbers / fabricated numbers harshly.

Return a JSON object: {"reasoning": "<2-3 sentences>", "score": <0-100>}"""


def load_checklist(task_id: str) -> list:
    p = TASKS_DIR / task_id / "target_study" / "checklist.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_instructions(workspace: Path) -> str:
    p = workspace / "INSTRUCTIONS.md"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def read_report(workspace: Path) -> str:
    p = workspace / "report" / "report.md"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    rd = workspace / "report"
    if rd.exists():
        for md in sorted(rd.glob("*.md")):
            return md.read_text(encoding="utf-8", errors="replace")
    return ""


def judge(item: dict, report_text: str, instructions: str, model: str, api_key: str, base_url: str) -> dict:
    from openai import OpenAI
    content = item.get("content", "")
    keywords = ", ".join(item.get("keywords", [])) or "None"
    itype = item.get("type", "text")
    report_excerpt = report_text[:12000] if report_text else "No report text available."
    if itype == "image":
        user = (f"## Research Task Instructions\n{instructions}\n\n"
                f"## Evaluation Criterion (from the original paper)\n{content}\n\n"
                f"## Key Visual Aspects to Verify\n{keywords}\n\n"
                f"## AI-Generated Report (excerpt)\n{report_excerpt}\n\n"
                f"Compare the AI-generated figures/analysis against the target from the original paper. "
                f"Superficially similar plots with wrong scales, missing data, or incorrect trends score low.\n\n"
                f"Return JSON {{\"reasoning\": \"...\", \"score\": 0-100}}")
    else:
        user = (f"## Research Task Instructions\n{instructions}\n\n"
                f"## Evaluation Criterion (from the original paper)\n{content}\n\n"
                f"## Key Technical Aspects to Verify\n{keywords}\n\n"
                f"## AI-Generated Research Report\n{report_excerpt}\n\n"
                f"Rate how well this report addresses the criterion compared to the original paper. "
                f"Return JSON {{\"reasoning\": \"...\", \"score\": 0-100}}")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    text = resp.choices[0].message.content
    try:
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start:end + 1])
    except Exception:
        return {"score": 0, "reasoning": text[:200]}
    return {"score": max(0, min(100, int(data.get("score", 0) or 0))),
            "reasoning": data.get("reasoning", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat"))
    ap.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")))
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_API_BASE", "https://api.deepseek.com/v1"))
    args = ap.parse_args()

    ws = Path(args.workspace)
    meta = json.load(open(ws / "_meta.json", encoding="utf-8"))
    task_id = args.task_id or meta.get("task_id", "")
    checklist = load_checklist(task_id)
    instructions = read_instructions(ws)
    report_text = read_report(ws)

    if not report_text.strip():
        print(json.dumps({"error": "no report found", "task_id": task_id}, ensure_ascii=False))
        return 1

    results = []
    total_w = sum(float(c.get("weight", 1.0)) for c in checklist)
    for i, item in enumerate(checklist):
        r = judge(item, report_text, instructions, args.model, args.api_key, args.base_url)
        w = float(item.get("weight", 1.0))
        results.append({"criterion": i, "type": item.get("type", "text"),
                        "weight": w, **r})
        print(f"[item {i}] type={item.get('type')} weight={w} score={r['score']} :: {r['reasoning'][:120]}", flush=True)

    weighted = sum(r["score"] * r["weight"] for r in results) / total_w
    out = {"task_id": task_id, "workspace": str(ws), "report_exists": True,
           "items": results, "weighted_score": round(weighted, 1),
           "n_items": len(results)}
    print("\n=== SCORE SUMMARY ===", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    (ws / "_rcb_score.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())