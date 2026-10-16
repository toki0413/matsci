"""能力可运行性 CLI: L0 依赖盘点 (+ 可选 L1 端到端 smoke).

用法:
    python -m huginn.cli.capability_runtime --json        # 打印 JSON
    python -m huginn.cli.capability_runtime --smoke       # 附带跑 L1
    python -m huginn.cli.capability_runtime --check       # 有降级能力则 exit 1
"""

from __future__ import annotations

import argparse
import json
import sys

from huginn.capabilities.runtime_probe import (
    build_report,
    probe_runtime,
    summary_line,
)


def _payload(report) -> dict:
    return {
        "l0": [r.to_dict() for r in report.l0],
        "l1": [r.to_dict() for r in report.l1],
        "summary": summary_line(report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--smoke", action="store_true", help="附带跑 L1 端到端 smoke")
    parser.add_argument("--out", default="", help="写到文件 (默认 stdout)")
    parser.add_argument("--check", action="store_true", help="有降级能力则 exit 1")
    args = parser.parse_args(argv)

    l0 = probe_runtime()
    l1 = []
    if args.smoke:
        from huginn.capabilities.smoke_specs import run_smoke

        l1 = run_smoke()
    report = build_report(l0, l1)
    payload = _payload(report)
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json
        else summary_line(report)
    )
    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.check and report.degraded():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
