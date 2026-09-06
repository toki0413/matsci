"""CLI: 统一资产分享总线 — 能力 / 工作流 / 人格 / 技能 / 证明单元.

用法:
    huginn share list                          # 全资产统一清单 (含每类计数)
    huginn share list --kind workflow          # 只看某类
    huginn share export -o bundle.json         # 全部导出一份单文件
    huginn share export --kind persona --kind lean --o p.json
    huginn share import bundle.json            # 还原并登记
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from huginn.share import ShareManager


@click.group("share")
def share() -> None:
    """统一资产分享总线: list / export / import."""


@share.command("list")
@click.option("--kind", default=None, help="只看某类 (capability/workflow/persona/skill/lean)")
@click.option("--json", "as_json", is_flag=True, help="输出机器可读 JSON")
def list_cmd(kind: str | None, as_json: bool) -> None:
    """罗列全部可分享资产 (含每类计数)."""
    kinds = (kind,) if kind else None
    counts = ShareManager.list(kinds)
    click.echo("Huginn 可分享资产库 (ShareManager)")
    click.echo("-" * 52)
    for k, v in counts.items():
        if k == "total":
            continue
        if isinstance(v, dict):
            click.echo(f"  {v['title']:<8} available={v['available']:<5}"
                       f"imported={v['imported_this_pool']}")
    click.echo("-" * 52)
    click.echo(f"  合计 {counts['total']} 项可分享")
    if as_json:
        click.echo(json.dumps(counts, ensure_ascii=False, indent=2))


@share.command("export")
@click.option("--kind", "kinds", multiple=True, help="只导出某些类 (可多次), 缺省全部")
@click.option("--out", "-o", type=click.Path(dir_okay=False), default=None, help="写入文件")
def export_cmd(kinds: tuple[str, ...], out: str | None) -> None:
    """导出单文件 bundle."""
    b = ShareManager.export_bundle(kinds=kinds or None)
    text = json.dumps(b, ensure_ascii=False, indent=2, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        click.echo(f"已导出 {b['count']} 项 → {out}")
    else:
        click.echo(text)


@share.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def import_cmd(file: str) -> None:
    """从一个导出 bundle 还原并登记."""
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    res = ShareManager.import_bundle(data)
    click.echo(f"导入完成: ok={res['ok']}")
    for it in res["imported"]:
        click.echo(f"  ✓ {it}")
    for it in res["skipped"]:
        click.echo(f"  - {it}")
    if res["errors"]:
        click.echo("错误:")
        for e in res["errors"]:
            click.echo(f"  ✗ {e}")