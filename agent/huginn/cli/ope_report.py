"""离线 OPE 报告 CLI — 从准入轨迹生成 IPS/SNIPS/DR + 影子安全指标.

用法：
    python -m huginn.cli.ope_report --tenant default \
        --db ~/.huginn/admission_trajectories.db --out docs/ope-report.json

只做离线评估；不训练、不上线、不改模型权重。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huginn.normative.learning import (
    TrajectoryIndex,
    build_ope_report,
    sample_from_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线 OPE 报告（准入策略）")
    parser.add_argument("--tenant", default=None, help="租户 id（默认取 HUGINN_TENANT_ID）")
    parser.add_argument("--db", default=None, help="轨迹库路径（默认 runtime home）")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--out", default=None, help="输出 JSON 路径；缺省打印 stdout")
    return parser


def build_report(
    *,
    tenant: str | None = None,
    db_path: str | None = None,
    train_ratio: float = 0.7,
    limit: int = 100_000,
) -> dict:
    if tenant is None:
        from huginn.env_access import env_str

        tenant = env_str("HUGINN_TENANT_ID") or "default"
    index = TrajectoryIndex(db_path)
    rows = index.list_samples(str(tenant), limit=int(limit))
    samples = [
        sample
        for row in rows
        if (
            sample := sample_from_payload(
                row.get("payload") or {},
                tenant_id=str(tenant),
                sample_id=str(row.get("sample_id") or ""),
            )
        )
        is not None
    ]
    report = build_ope_report(samples, train_ratio=float(train_ratio))
    report["tenant_id"] = str(tenant)
    report["total_samples"] = len(samples)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        tenant=args.tenant,
        db_path=args.db,
        train_ratio=args.train_ratio,
        limit=args.limit,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(str(target))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
