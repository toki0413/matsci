"""复现重跑 CLI: 读 provenance 快照/记录 → 同参重跑 → 逐字段 diff 报告.

用法:
    python -m huginn.cli.reproduce --snapshot .huginn/provenance/xxx.json
    python -m huginn.cli.reproduce --record .huginn/provenance/provenance.jsonl
    python -m huginn.cli.reproduce --snapshot snap.json --tolerance 1e-6 --json

退出码: 全部 reproduced → 0; 有 drifted / not_reproducible / error → 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huginn.autoloop.reproduce import (
    DEFAULT_TOLERANCE,
    reproduce_record,
    reproduce_snapshot,
)


def _load_snapshots(path: Path) -> list[dict]:
    """读快照/记录文件 → 待重跑的 snapshot dict 列表 (逐条, 不合并).

    - 单 JSON 且含 tool_chain → 视为 run 级 record (拆成链上快照逐条跑);
    - 否则视为单条 ProvenanceSnapshot;
    - .jsonl → 每行一个 record/snapshot.
    解析失败显式报错退出, 不静默跳过坏行.
    """
    text = path.read_text(encoding="utf-8")
    out: list[dict] = []
    if path.suffix == ".jsonl":
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{i + 1} 不是合法 JSON: {exc}") from exc
            out.append(item)
        return out
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return [data]


def _is_record(item: dict) -> bool:
    return "tool_chain" in item and "tool_name" not in item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshot", help="单条快照 JSON (ProvenanceSnapshot dict)")
    src.add_argument("--record", help="run 级记录 (JSON with tool_chain / JSONL)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = parser.parse_args(argv)

    path = Path(args.snapshot or args.record)
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 2
    items = _load_snapshots(path)
    if not items:
        # 空文件 → 显式失败, 不给"0 条全绿"的假绿
        print(f"{path} 里没有可重跑的快照/记录", file=sys.stderr)
        return 2

    reports = []
    for item in items:
        if _is_record(item):
            reports.append(reproduce_record(item, tolerance=args.tolerance))
        else:
            reports.append(reproduce_snapshot(item, tolerance=args.tolerance))

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    else:
        for rep in reports:
            if "results" in rep:  # record 汇总
                print(
                    f"[run {rep['run_id'] or '?'}] {rep['n_snapshots']} snapshots: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(rep["counts"].items()))
                )
                for r in rep["results"]:
                    _print_snapshot(r, indent="  ")
            else:
                _print_snapshot(rep)

    ok = all(
        (rep.get("all_reproduced") if "results" in rep else rep["status"] == "reproduced")
        for rep in reports
    )
    return 0 if ok else 1


def _print_snapshot(rep: dict, indent: str = "") -> None:
    line = f"{indent}[{rep['tool_name']}] {rep['status']}"
    if rep.get("mode"):
        line += f" (mode={rep['mode']})"
    if rep.get("n_diffs"):
        line += f" — {rep['n_diffs']} field diff(s)"
    print(line)
    for d in rep.get("diffs", [])[:10]:
        print(f"{indent}    {d}")
    if len(rep.get("diffs", [])) > 10:
        print(f"{indent}    ... (+{len(rep['diffs']) - 10} more)")
    if rep.get("error"):
        print(f"{indent}    error: {rep['error']}")


if __name__ == "__main__":
    sys.exit(main())
