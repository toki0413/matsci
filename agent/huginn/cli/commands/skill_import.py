"""skill-import: 从 OpenClaw / Hermes 导入技能并注册到 SkillRegistry。"""

from __future__ import annotations

from pathlib import Path

import click
from rich.table import Table

from huginn.cli.context import CliContext
from huginn.plugins.skill_importer import SkillImporter
from huginn.skills.registry import SkillRegistry


@click.command(name="skill-import")
@click.argument(
    "path", required=False, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--platform",
    type=click.Choice(["auto", "openclaw", "hermes"]),
    default="auto",
    help="源平台格式，auto 时按 frontmatter 自动识别",
)
@click.option("--list", "list_only", is_flag=True, help="列出已注册的技能")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="把导入的技能转成 Huginn 格式写到该目录",
)
@click.pass_obj
def skill_import(
    ctx: CliContext,
    path: Path | None,
    platform: str,
    list_only: bool,
    output: Path | None,
) -> None:
    """从 OpenClaw / Hermes 目录导入技能，注册后即可被 agent 调用。"""
    if list_only:
        _list_skills(ctx)
        return

    if path is None:
        ctx.console.print(
            "[yellow]请提供技能目录路径，或用 --list 查看已有技能[/yellow]"
        )
        return

    importer = SkillImporter()
    skills = importer.import_directory(path, platform)

    if not skills:
        ctx.console.print(f"[yellow]未在 {path} 找到可导入的 SKILL.md[/yellow]")
        return

    for s in skills:
        SkillRegistry.register(s)

    table = Table(title=f"导入 {len(skills)} 个技能")
    table.add_column("名称", style="bold")
    table.add_column("来源")
    table.add_column("步骤数", justify="right")
    table.add_column("工具")
    for s in skills:
        table.add_row(
            s.name,
            s.metadata.get("platform", "?"),
            str(len(s.steps)),
            ", ".join(s.required_tools) or "-",
        )
    ctx.console.print(table)

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for s in skills:
            importer.export_to_huginn(s, output / f"{s.name}.md")
        ctx.console.print(f"[green]✓[/green] 已导出 Huginn 格式到 {output}")


def _list_skills(ctx: CliContext) -> None:
    # 触发内置 preset 注册，保证列表里有料
    from huginn.skills import presets  # noqa: F401

    defs = SkillRegistry.get_all_definitions()
    if not defs:
        ctx.console.print("[yellow]当前没有已注册的技能[/yellow]")
        return

    table = Table(title=f"已注册技能 ({len(defs)})")
    table.add_column("名称", style="bold")
    table.add_column("分类")
    table.add_column("标签")
    for s in defs:
        table.add_row(s.name, s.category, ", ".join(s.tags) or "-")
    ctx.console.print(table)
