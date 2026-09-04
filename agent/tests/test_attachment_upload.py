"""附件统一落地 + 多文件上传测试.

覆盖:
  (a) 多文件都落地且内容完整
  (b) 分块写入不整读 (验证 read(chunk_size) 被分块调用)
  (c) 单文件端点仍工作 (向后兼容)
  (d) 大小 / 元数据正确

不启动完整后端, 用 starlette TestClient 挂最小 app (只含 knowledge 路由),
monkeypatch 掉鉴权与 KB, 聚焦于附件落地与上传逻辑。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.utils import attachment as att


class _FakeKB:
    """记录式假 KB: add_document 记录入参, 便于断言."""

    def __init__(self) -> None:
        self.added: list[list] = []
        self.saved_raw: list[tuple] = []

    def add_document(self, filename: str, content: bytes) -> dict:
        self.added.append([filename, bytes(content)])
        return {"doc_id": f"id-{len(self.added)}", "filename": filename}

    def save_raw(self, doc_id, filename, content) -> None:
        self.saved_raw.append((doc_id, filename, bytes(content)))


class _FakeUpload:
    """最小 UploadFile 替身: 记录每次 read(n) 的入参与返回长度."""

    def __init__(self, data: bytes, filename: str = "f.bin") -> None:
        self._data = data
        self._pos = 0
        self.filename = filename
        self.read_calls: list[tuple] = []  # (n, len)

    async def read(self, n: int = -1) -> bytes:
        limit = len(self._data) if n < 0 else self._pos + n
        chunk = self._data[self._pos : limit]
        self._pos += len(chunk)
        self.read_calls.append((n, len(chunk)))
        return chunk


@pytest.fixture
def client(monkeypatch):
    """最小 app: 只挂 knowledge 路由, 屏蔽鉴权与真实 KB."""
    from huginn.routes import knowledge as kr
    from huginn.security.auth import require_api_key

    kb = _FakeKB()
    ctx = SimpleNamespace(kb=kb)

    # 屏蔽 SmartIngester(免得拉重依赖 / 跑 OCR), 走 add_document 落库路径
    monkeypatch.setattr(
        "huginn.knowledge.smart_ingest.build_smart_ingester", lambda kb: None
    )
    monkeypatch.setattr(kr, "get_context", lambda: ctx)

    app = FastAPI()
    app.include_router(kr.router)
    app.dependency_overrides[require_api_key] = lambda: None
    with TestClient(app) as c:
        yield c, kb


def _post_many(app: TestClient, payloads: list[tuple]) -> object:
    return app.post("/knowledge/upload_many", files=[("files", p) for p in payloads])


def test_multi_upload_lands_all_files(client):
    """(a) 多个文件都落地且内容完整."""
    c, kb = client
    payloads = [
        ("a.txt", b"hello world", "text/plain"),
        ("b.bin", bytes(range(256)), "application/octet-stream"),
        ("nested/../../c.bin", b"\x00\x01\x02", "application/octet-stream"),
    ]
    resp = _post_many(c, payloads)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["landed"] == 3
    assert body["ingested"] == 3

    metas = {m["filename"]: m for m in body["files"]}
    # 清理后文件名: 路径穿越段被剥掉, 只留 basename
    assert set(metas) == {"a.txt", "b.bin", "c.bin"}

    # 每个文件都落地, 且内容与上传字节完全一致
    by_name = {
        "a.txt": b"hello world",
        "b.bin": bytes(range(256)),
        "c.bin": b"\x00\x01\x02",
    }
    for name, expect in by_name.items():
        p = Path(metas[name]["path"])
        assert p.is_file(), f"{name} 未落地"
        assert p.read_bytes() == expect

    # KB 摄入也成功, 每个文件各入库一次
    assert len(kb.added) == 3
    assert {e[0] for e in kb.added} == {"a.txt", "b.bin", "c.bin"}


async def test_stream_to_disk_chunking(tmp_path):
    """(b) stream_to_disk 分块写入, 不整读进内存 (read 被按 chunk_size 分块调用)."""
    data = b"a" * 2500
    up = _FakeUpload(data, filename="chunk.bin")
    info = await att.stream_to_disk(up, tmp_path, chunk_size=1000)
    assert info.size == 2500
    # 只按 chunk_size 读 (1000), 绝无一次性 read() 整读; 最后一次返回空 b"" 表示 EOF
    assert {c[0] for c in up.read_calls} == {1000}
    assert up.read_calls[-1][1] == 0  # EOF 空块
    assert sum(l for _, l in up.read_calls) == 2500
    assert info.path.read_bytes() == data
    # 元数据完整
    assert info.filename == "chunk.bin"
    assert info.size == len(data)
    assert info.compressed is False


def test_single_upload_still_works(client):
    """(c) 旧单文件端点仍工作(向后兼容)."""
    c, kb = client
    resp = c.post(
        "/knowledge/upload",
        files={"file": ("legacy.txt", b"old endpoint", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["document"]["filename"] == "legacy.txt"
    assert kb.added == [["legacy.txt", b"old endpoint"]]
    assert kb.saved_raw  # 原始文件也被补存


def test_upload_many_metadata_sizes(client):
    """(d) 大小/元数据正确."""
    c, kb = client
    resp = _post_many(
        c,
        [("m1.txt", b"abc", "text/plain"), ("m2.txt", b"12345", "text/plain")],
    )
    body = resp.json()
    assert body["landed"] == 2
    sizes = {m["filename"]: m["size"] for m in body["files"]}
    assert sizes == {"m1.txt": 3, "m2.txt": 5}
    # 落地目录在统一附件根下的批次子目录, 且 size 与实际字节一致
    directory = Path(body["directory"])
    assert directory.is_dir()
    for m in body["files"]:
        assert Path(m["path"]).parent == directory
        assert m["compressed"] is False
        assert m["size"] == Path(m["path"]).stat().st_size


def test_upload_many_empty(client):
    """缺少 files 字段时 FastAPI 校验拦截(422), 不进入落地."""
    c, kb = client
    resp = c.post("/knowledge/upload_many")
    assert resp.status_code == 422


def test_attachment_utils_dir_uses_runtime_home(monkeypatch, tmp_path):
    """附件根目录默认挂在 runtime home / attachments 下."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(cache))
    d = att.get_attachments_dir()
    assert d == cache / att.ATTACHMENTS_DIR_NAME
    assert d.is_dir()


async def test_land_batch_with_compress(tmp_path):
    """(d) land_files 批量落地 + 可选 zip 压缩 + manifest 元数据."""
    up1 = _FakeUpload(b"hello world", filename="a.txt")
    up2 = _FakeUpload(bytes(range(64)), filename="b.bin")
    batch = await att.land_files(
        [up1, up2], dest=tmp_path, compress=True, metadata={"src": "t"}
    )
    assert batch.count == 2
    assert batch.total_size == len(b"hello world") + 64
    # 每文件落盘且被标记为已压缩
    assert all(p.is_file() for p in [f.path for f in batch.files])
    assert all(f.compressed for f in batch.files)
    # 统一 zip 归档存在, 内含所有文件 + manifest.json
    assert batch.archive is not None and batch.archive.is_file()
    with zipfile.ZipFile(batch.archive) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert batch.files[0].path.name in names
        assert batch.files[1].path.name in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["metadata"] == {"src": "t"}
        assert manifest["count"] == 2
