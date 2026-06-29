"""CLI command registry."""

from __future__ import annotations

import click

from huginn.cli.api_key_setup import api_keys
from huginn.cli.commands import (
    autoloop,
    autoresearch,
    background,
    bench,
    chat,
    coder,
    configure,
    diagnose,
    encrypt_config,
    evolve,
    execute,
    explore,
    export,
    hpc,
    kg,
    memory_maintenance,
    model_list,
    persona,
    plot,
    refactor,
    remote,
    replay,
    scheduler,
    seed_knowledge,
    serve,
    sessions,
    swarm,
    telemetry,
    tools,
    unified,
    version,
    visualize,
    workflow,
)


def register_commands(cli: click.Group) -> None:
    """Register all domain commands on the main CLI group."""
    cli.add_command(chat.chat)
    cli.add_command(coder.coder)
    cli.add_command(refactor.refactor)
    cli.add_command(explore.explore)
    cli.add_command(autoloop.autoloop)
    cli.add_command(serve.serve)
    cli.add_command(tools.tools)
    cli.add_command(version.version)
    cli.add_command(configure.configure)
    cli.add_command(bench.bench)
    cli.add_command(evolve.evolve)
    cli.add_command(execute.execute)
    cli.add_command(workflow.workflow)
    cli.add_command(diagnose.diagnose)
    cli.add_command(model_list.model_list)
    cli.add_command(memory_maintenance.memory_maintenance)
    cli.add_command(telemetry.telemetry)
    cli.add_command(seed_knowledge.seed_knowledge)
    cli.add_command(encrypt_config.encrypt_config)
    cli.add_command(export.export_data)
    cli.add_command(kg.build_kg)
    # 回放 agent 决策轨迹
    cli.add_command(replay.replay)
    # API key 配置入口
    cli.add_command(api_keys)

    # Command groups with subcommands
    cli.add_command(hpc.hpc)
    cli.add_command(remote.remote)
    cli.add_command(scheduler.scheduler)
    cli.add_command(autoresearch.autoresearch)
    cli.add_command(plot.plot)
    cli.add_command(unified.unified)
    cli.add_command(persona.persona)
    cli.add_command(swarm.swarm)
    cli.add_command(swarm.team)
    cli.add_command(visualize.visualize)
    cli.add_command(kg.kg)
    cli.add_command(sessions.sessions)
    # 后台任务管理
    cli.add_command(background.bg)
