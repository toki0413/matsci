#!/usr/bin/env python3
"""Huginn 自主深研产品演示 — 同一管线跑多个域 backend.

证明: 深研闭环是域无关产品能力, 换 domain 只换 backend, 管线零改动.
  含热木星(大气环流GCM) 与 系外行星(真实目录第一性原理解析) 两个 backend.

用法: python examples/ai4s_product_demo.py --domain hotjupiter --dry
      python examples/ai4s_product_demo.py --domain exoplanet          (真机成文+门禁)
      python examples/ai4s_product_demo.py --all --dry
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from ai4s_backends import BACKENDS, GOALS, OBJECTIVES  # noqa: E402
from huginn.research import run_research_program  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def _one(domain: str, client, model: str, base_url: str | None) -> None:
    out = run_research_program(
        goal=GOALS[domain],
        experiments=BACKENDS[domain](),
        objectives_config=OBJECTIVES[domain],
        client=client, model=model, base_url=base_url,
        out_md=OUT / f"ai4s_product_{domain}_report.md",
    )
    print(f"\n=== domain={domain} ===")
    print(f"[program] explored={out.explored} pruned={out.pruned} "
          f"pareto_front={len(out.pareto_front)} convergence={out.converred}")
    for b in out.pareto_front:
        print(f"  surv → {b['name']}")
    print(f"[gate] {out.verdict} {out.ungrounded} | report_source={out.report_source}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=list(BACKENDS) + ["all"], default="hotjupiter")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    client = None
    if not args.dry:
        key = os.environ.get("INTERNLM_API_KEY")
        if not key:
            print("error: INTERNLM_API_KEY not set (或 --dry)", file=sys.stderr); return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=args.base_url or
                        os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"))

    domains = list(BACKENDS) if args.domain == "all" else [args.domain]
    for d in domains:
        _one(d, client, args.model, args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())