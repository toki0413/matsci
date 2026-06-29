"""Sessions command — browse and manage conversation history.

直接从 LangGraph SqliteSaver 的 checkpoints.db 里查会话列表,
不依赖 agent 运行时, 离线也能用。
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from huginn.cli.context import CliContext

# UUID v1 epoch (1582-10-15) 到 Unix epoch (1970-01-01) 的偏移, 单位 100-ns
_UUID_EPOCH_OFFSET = 0x01B21DD213814000


def _get_db_path() -> Path:
    """找到 checkpoints.sqlite 的位置。

    优先用环境变量, 没设就用默认路径 ~/.huginn/checkpoints.sqlite。
    """
    env_path = os.environ.get("HUGINN_CHECKPOINTER_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".huginn" / "checkpoints.sqlite"


def _uuid_time_ticks(uuid_str: str) -> int | None:
    """从 UUID v1/v6 里解出时间戳(100-ns 间隔, 自 1582-10-15)。

    v4 是纯随机, 没有时间信息, 返回 None。
    其他版本或解析失败也返回 None。
    """
    try:
        u = uuid.UUID(str(uuid_str))
    except (ValueError, AttributeError, TypeError):
        return None

    if u.version == 1:
        # v1 直接有 .time 属性
        try:
            return u.time
        except (AttributeError, ValueError):
            return None

    if u.version == 6:
        # v6 把时间字段重排成自然序, 需要手动拼回来
        val = u.int
        time_high = (val >> 96) & 0xFFFFFFFF
        time_mid = (val >> 80) & 0xFFFF
        # 跳过 4 位 version
        time_low = (val >> 64) & 0xFFF
        return (time_high << 28) | (time_mid << 12) | time_low

    return None


def _uuid_to_datetime(uuid_str: str) -> datetime | None:
    """把 checkpoint_id(UUID) 转成 datetime, 失败返回 None。"""
    ticks = _uuid_time_ticks(uuid_str)
    if ticks is None:
        return None
    unix_ticks = ticks - _UUID_EPOCH_OFFSET
    if unix_ticks <= 0:
        return None
    # 100-ns 间隔转秒
    return datetime.fromtimestamp(unix_ticks / 1e7, tz=timezone.utc)


def _truncate(text: str, width: int) -> str:
    """截断字符串, 超长加省略号。"""
    text = text.strip().replace("\n", " ")
    if len(text) <= width:
        return text
    return text[:width] + "..."


def _extract_preview(metadata_blob: bytes | str | None) -> str:
    """从 metadata JSON 里挖一句预览文本。

    LangGraph 的 metadata 里有 writes 字段, 记录了产生这个 checkpoint 的写入,
    里面通常藏着用户输入或 agent 回复的文本。
    """
    if not metadata_blob:
        return ""
    try:
        if isinstance(metadata_blob, bytes):
            metadata_blob = metadata_blob.decode("utf-8", errors="replace")
        meta = json.loads(metadata_blob)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""

    if not isinstance(meta, dict):
        return ""

    writes = meta.get("writes") or {}
    if not isinstance(writes, dict):
        return ""

    # writes 形如 {"__start__": {"messages": [{"content": "..."}]}}
    # 或 {"agent": {"messages": [{"content": "..."}]}}
    for values in writes.values():
        if isinstance(values, dict):
            msgs = values.get("messages")
            if isinstance(msgs, list):
                for msg in msgs:
                    text = _msg_to_text(msg)
                    if text:
                        return _truncate(text, 40)
        elif isinstance(values, list):
            for v in values:
                text = _msg_to_text(v)
                if text:
                    return _truncate(text, 40)
    return ""


def _msg_to_text(msg: Any) -> str:
    """从各种消息格式里抠出纯文本。"""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        content = msg.get("content") or msg.get("text") or msg.get("input")
        if isinstance(content, str):
            return content
        # content 有时是 list[dict](多模态消息), 取第一段文本
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("content")
                    if isinstance(t, str) and t.strip():
                        return t
                elif isinstance(part, str) and part.strip():
                    return part
    return ""


def _check_tables(conn: sqlite3.Connection) -> list[str]:
    """看库里实际有哪些表。"""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return [r[0] for r in rows]


def _list_threads(db_path: Path) -> list[dict]:
    """从数据库查询所有 thread_id 及其摘要信息。

    返回 list[dict], 每项含:
      - thread_id: 会话标识
      - created:   最近一次活动时间(datetime | None)
      - messages:  checkpoint 数(近似消息数)
      - preview:   最近一条写入的预览文本
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        tables = _check_tables(conn)
        if "checkpoints" not in tables:
            return []

        # 只看根命名空间(checkpoint_ns=''), 子图的 checkpoint 不重复计数
        # checkpoint_id 是 UUID v6, 字符串排序就是时间序
        rows = conn.execute(
            """
            SELECT thread_id,
                   MAX(checkpoint_id) AS latest_cp,
                   COUNT(*)           AS cp_count
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY latest_cp DESC
            """
        ).fetchall()

        results: list[dict] = []
        for thread_id, latest_cp, cp_count in rows:
            created = _uuid_to_datetime(latest_cp) if latest_cp else None
            preview = ""
            if latest_cp:
                meta_row = conn.execute(
                    "SELECT metadata FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_id = ? AND checkpoint_ns = '' "
                    "LIMIT 1",
                    (thread_id, latest_cp),
                ).fetchone()
                if meta_row:
                    preview = _extract_preview(meta_row[0])
            results.append(
                {
                    "thread_id": thread_id,
                    "created": created,
                    "messages": cp_count,
                    "preview": preview,
                }
            )
        return results
    finally:
        conn.close()


def _format_dt(dt: datetime | None) -> str:
    """datetime 转可读字符串, None 显示 —。"""
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


@click.group(name="sessions")
@click.pass_obj
def sessions(ctx: CliContext) -> None:
    """Browse and manage conversation sessions."""


@sessions.command("list")
@click.pass_obj
def list_sessions(ctx: CliContext) -> None:
    """List all saved sessions."""
    db_path = _get_db_path()
    threads = _list_threads(db_path)
    if not threads:
        ctx.console.print("[yellow]No saved sessions found.[/yellow]")
        return

    table = Table(title="Saved Sessions", show_lines=False)
    table.add_column("Thread ID", style="cyan", no_wrap=False)
    table.add_column("Created", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Preview", no_wrap=False)

    for t in threads:
        table.add_row(
            t["thread_id"],
            _format_dt(t["created"]),
            str(t["messages"]),
            t["preview"],
        )
    ctx.console.print(table)
    ctx.console.print(f"\n[dim]{len(threads)} session(s) in {db_path}[/dim]")


@sessions.command("show")
@click.argument("thread_id")
@click.pass_obj
def show_session(ctx: CliContext, thread_id: str) -> None:
    """Show details of a specific session."""
    db_path = _get_db_path()
    if not db_path.exists():
        ctx.console.print("[yellow]No saved sessions found.[/yellow]")
        return

    threads = _list_threads(db_path)
    match = [t for t in threads if t["thread_id"] == thread_id]
    if not match:
        ctx.console.print(f"[red]Session '{thread_id}' not found.[/red]")
        return

    info = match[0]
    ctx.console.print(f"[bold cyan]Thread ID:[/bold cyan]     {thread_id}")
    ctx.console.print(f"[bold green]Created:[/bold green]       {_format_dt(info['created'])}")
    ctx.console.print(f"[bold]Checkpoints:[/bold]     {info['messages']}")

    # 拿最近一条 checkpoint 的完整信息
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT checkpoint_id, parent_checkpoint_id, metadata "
            "FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = '' "
            "ORDER BY checkpoint_id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    finally:
        conn.close()

    if row:
        cp_id, parent_id, meta_blob = row
        ctx.console.print(f"[bold]Latest checkpoint:[/bold] {cp_id}")
        if parent_id:
            ctx.console.print(f"[bold]Parent:[/bold]           {parent_id}")
        if meta_blob:
            try:
                if isinstance(meta_blob, bytes):
                    meta_blob = meta_blob.decode("utf-8", errors="replace")
                meta = json.loads(meta_blob)
                ctx.console.print(
                    f"[bold]Metadata:[/bold]          {json.dumps(meta, ensure_ascii=False, indent=2)}"
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                ctx.console.print("[dim](metadata 不可读)[/dim]")

    # 尝试用 LangGraph saver 反序列化, 拿到真实的消息条数
    msg_count = _try_count_messages(db_path, thread_id)
    if msg_count is not None:
        ctx.console.print(f"[bold]Messages:[/bold]          {msg_count}")


def _try_count_messages(db_path: Path, thread_id: str) -> int | None:
    """用 SqliteSaver 的 API 拿 channel_values 里的消息条数。

    反序列化失败就返回 None, 不影响 show 命令的其他输出。
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return None

    try:
        cm = SqliteSaver.from_conn_string(str(db_path))
        saver = cm.__enter__()
        try:
            config = {"configurable": {"thread_id": thread_id}}
            tup = saver.get_tuple(config)
            if tup is None or tup.checkpoint is None:
                return None
            channel_values = tup.checkpoint.get("channel_values") or {}
            messages = channel_values.get("messages")
            if isinstance(messages, list):
                return len(messages)
            # messages 可能是序列化的 (type, blob) 元组
            if isinstance(messages, tuple) and len(messages) == 2:
                try:
                    serde = getattr(saver, "serde", None) or getattr(
                        saver, "jsonplus_serde", None
                    )
                    if serde is not None:
                        decoded = serde.loads_typed(messages)
                        if isinstance(decoded, list):
                            return len(decoded)
                except Exception:
                    return None
            return None
        finally:
            cm.__exit__(None, None, None)
    except Exception:
        return None


@sessions.command("delete")
@click.argument("thread_id")
@click.pass_obj
def delete_session(ctx: CliContext, thread_id: str) -> None:
    """Delete a specific session."""
    db_path = _get_db_path()
    if not db_path.exists():
        ctx.console.print("[yellow]No saved sessions found.[/yellow]")
        return

    conn = sqlite3.connect(str(db_path))
    try:
        tables = _check_tables(conn)
        if "checkpoints" not in tables:
            ctx.console.print("[yellow]No saved sessions found.[/yellow]")
            return

        # 先看一下这个 thread 有没有数据, 没有就提前提示
        row = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if not row or row[0] == 0:
            ctx.console.print(f"[yellow]Session '{thread_id}' not found.[/yellow]")
            return

        # 跟 SqliteSaver 的清表逻辑一致, checkpoints + writes 两张表都删
        cur = conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        deleted_cp = cur.rowcount
        deleted_w = 0
        if "writes" in tables:
            cur = conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
            )
            deleted_w = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    ctx.console.print(
        f"[green]Deleted session '{thread_id}': "
        f"{deleted_cp} checkpoint(s), {deleted_w} write(s).[/green]"
    )
