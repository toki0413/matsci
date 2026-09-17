"""ObservationPack 机制测试: 稳定句柄 + 分页精确召回 + 工具层."""

from __future__ import annotations

import os
import tempfile

import pytest

from huginn.tools.observation_pack import (
    ObservationArchive,
    RecallObservationTool,
    handle_for_body,
)


@pytest.fixture()
def archive_dir() -> str:
    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    yield tmp
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    os.environ.pop("HUGINN_CACHE_DIR", None)


def test_archive_and_exact_paged_recall(archive_dir: str) -> None:
    archive = ObservationArchive("t1")
    body = "_".join(f"row{i:04d}" for i in range(5000))
    handle, path = archive.archive_body(body)
    assert path.is_file()
    # 分页拼回无丢字
    rebuilt = ""
    page = 0
    while True:
        r = archive.recall(handle, page=page, page_size=700)
        assert r["found"], r
        rebuilt += r["text"][:700] if r["more"] else r["text"]
        if not r["more"]:
            break
        page += 1
    assert rebuilt == body


def test_malformed_and_missing_handle(archive_dir: str) -> None:
    archive = ObservationArchive("t2")
    assert not archive.recall("not-a-handle")["found"]
    assert not archive.recall("@obs:../../etc/passwd")["found"]
    assert not archive.recall("@obs:00000000-0000-0000-0000-000000000000")["found"]


def test_sessions_are_isolated(archive_dir: str) -> None:
    a = ObservationArchive("sess_a")
    b = ObservationArchive("sess_b")
    handle, _ = a.archive_body("secret-a-body")
    assert not b.recall(handle)["found"]


async def test_tool_returns_toolresult(archive_dir: str) -> None:
    from huginn.core_types import ToolContext

    archive = ObservationArchive("t3")
    handle, _ = archive.archive_body("x" * 10_000)
    tool = RecallObservationTool()
    ctx = ToolContext(session_id="t3", workspace=archive_dir, config=None)
    from huginn.tools.observation_pack import _RecallInput

    ok = await tool._execute(_RecallInput(handle=handle, page=1, page_size=100), ctx)
    assert ok.success and ok.data["page"] == 1
    bad = await tool._execute(_RecallInput(handle="bad", page=0), ctx)
    assert not bad.success


def test_handle_for_body_under_threshold_returns_none(archive_dir: str) -> None:
    assert handle_for_body("small", threshold_tokens=5) is None
    h = handle_for_body("x" * 100_000, session_id="t4", threshold_tokens=5)
    assert h is not None and h.startswith("@obs:")
