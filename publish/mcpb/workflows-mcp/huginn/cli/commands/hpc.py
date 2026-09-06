"""HPC cluster job submission commands."""

from __future__ import annotations

import json

import click

from huginn.cli.context import CliContext


@click.group(name="hpc")
@click.pass_obj
def hpc(ctx: CliContext) -> None:
    """HPC cluster job submission commands."""


@hpc.command("test")
@click.option("--host", required=True, help="HPC host")
@click.option("--username", "-u", required=True, help="SSH username")
@click.option("--scheduler", default="slurm", type=click.Choice(["slurm", "pbs"]))
@click.option("--key-path", help="SSH private key path")
@click.option("--port", default=22, type=int)
@click.pass_obj
def hpc_test(
    ctx: CliContext,
    host: str,
    username: str,
    scheduler: str,
    key_path: str | None,
    port: int,
) -> None:
    """Test SSH connection to an HPC cluster."""
    from huginn.hpc.client import HPCClient, HPCConfig

    cfg = HPCConfig(
        host=host,
        username=username,
        scheduler=scheduler,
        key_path=key_path,
        port=port,
    )
    try:
        with HPCClient(cfg) as client:
            stdout, stderr, rc = client._exec("hostname")
            if rc == 0:
                ctx.console.print(f"[green]✓[/green] Connected to {host}: {stdout}")
            else:
                ctx.console.print(f"[red]✗[/red] {stderr or 'Connection failed'}")
    except Exception as e:
        ctx.console.print(f"[red]✗[/red] {e}")


@hpc.command("submit")
@click.option("--host", required=True)
@click.option("--username", "-u", required=True)
@click.option("--command", required=True, help="Command to run on the cluster")
@click.option("--job-name", default="huginn_job")
@click.option("--walltime", default="01:00:00")
@click.option("--nodes", default=1, type=int)
@click.option("--ntasks-per-node", default=4, type=int)
@click.option("--queue", help="Queue/partition")
@click.option("--scheduler", default="slurm", type=click.Choice(["slurm", "pbs"]))
@click.option("--key-path")
@click.option("--remote-work-dir", default="~/huginn_jobs")
@click.pass_obj
def hpc_submit(
    ctx: CliContext,
    host: str,
    username: str,
    command: str,
    job_name: str,
    walltime: str,
    nodes: int,
    ntasks_per_node: int,
    queue: str | None,
    scheduler: str,
    key_path: str | None,
    remote_work_dir: str,
) -> None:
    """Submit a job to a remote HPC cluster."""
    from huginn.hpc.client import HPCClient, HPCConfig

    cfg = HPCConfig(
        host=host,
        username=username,
        scheduler=scheduler,
        key_path=key_path,
        remote_work_dir=remote_work_dir,
    )
    try:
        with HPCClient(cfg) as client:
            script = client.generate_job_script(
                command=command,
                job_name=job_name,
                walltime=walltime,
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                queue=queue,
            )
            job_id = client.submit_job(script, job_name=job_name)
            ctx.console.print(f"[green]✓[/green] Submitted {job_name}: {job_id}")
    except Exception as e:
        ctx.console.print(f"[red]✗[/red] {e}")


@hpc.command("status")
@click.option("--host", required=True)
@click.option("--username", "-u", required=True)
@click.option("--job-id", required=True)
@click.option("--scheduler", default="slurm", type=click.Choice(["slurm", "pbs"]))
@click.option("--key-path")
@click.pass_obj
def hpc_status(
    ctx: CliContext,
    host: str,
    username: str,
    job_id: str,
    scheduler: str,
    key_path: str | None,
) -> None:
    """Poll status of a remote HPC job."""
    from huginn.hpc.client import HPCClient, HPCConfig

    cfg = HPCConfig(
        host=host, username=username, scheduler=scheduler, key_path=key_path
    )
    try:
        with HPCClient(cfg) as client:
            status = client.poll_status(job_id)
            ctx.console.print(
                json.dumps(
                    {
                        "job_id": status.job_id,
                        "state": status.state,
                        "exit_code": status.exit_code,
                        "runtime": status.runtime,
                        "message": status.message,
                    },
                    indent=2,
                    default=str,
                )
            )
    except Exception as e:
        ctx.console.print(f"[red]✗[/red] {e}")
