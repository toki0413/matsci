"""Evidence-Preserving Reducer 机制测试: 只保留逐字原文行进收据 + 回源密封."""

from __future__ import annotations

import os
import tempfile

import pytest

from huginn.tools.evidence_reducer import (
    Receipt,
    reduce_log_to_receipt,
    verify_receipt,
)
from huginn.tools.observation_pack import ObservationArchive


@pytest.fixture()
def archive():
    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    yield ObservationArchive("evid")
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    os.environ.pop("HUGINN_CACHE_DIR", None)


def test_receipt_keeps_only_verbatim_lines(archive) -> None:
    log = "\n".join([f"INFO step {i}" for i in range(30)] + ["ERROR crash detected"])
    rec = reduce_log_to_receipt(archive, log)
    assert rec.sealed
    assert all(ln in log for ln in rec.retained_lines())
    assert rec.omitted >= 1
    assert archive.recall(rec.handle)["found"]


def test_verify_fails_on_forged_quote(archive) -> None:
    rec = reduce_log_to_receipt(archive, "line one\nline two\n")
    forged = Receipt(handle=rec.handle, quotes=["totally fabricated quote"])
    assert verify_receipt(forged, archive) is False


def test_empty_log_receipt_is_sealed_empty(archive) -> None:
    rec = reduce_log_to_receipt(archive, "")
    assert rec.sealed and rec.omitted == 0 and not rec.retained_lines()


def test_receipt_roundtrip_to_dict(archive) -> None:
    rec = reduce_log_to_receipt(archive, "a\nb\nerror e\nc\nd\n")
    d = rec.to_dict()
    assert isinstance(d["head"], list) and d["sealed"] is True
    assert d["handle"].startswith("@obs:")