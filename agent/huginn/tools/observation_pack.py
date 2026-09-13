"""observation_pack —— Huginn 版 SoL-Pi「ObservationPack」机制（落地②号）。

动机: Huginn 已有 ``compress.py::offload_tool_output`` 把超长工具输出**无损存盘**并
只回预览, 但缺 SoL-Pi 的物种差异——**稳定句柄身份 + 按需精确分页召回**。现在
offload 后模型只拿到一个裸文件路径(``_artifact_path``), 无法在后续轮次里按句柄
取回原文、也无法分页。

本模块补上:
  1. ``ObservationArchive``: 把全量正文按会话归档成稳定句柄 ``@obs:<obs_id>``,
     支持按页精确召回(句柄不可跨会话猜读, 只读固定 session 目录)。
  2. ``RecallObservationTool``: 暴露给 LLM 的召回工具——给出句柄就能分页取回
     原文, 让"大输出先放预览、需要细节再召回"成为事实。
  3. ``handle_for_body``: adapter 超长输出缝的接线入口, 把 offload 的结果同时
     注册进档案并返回 ``_recall_handle``。

与现有 ``offload_tool_output``/``ToolOutputCompressor`` 的关系(不是替代):
  - Compressor  = 有损压缩(保关键值, 截断数组)  → 进首屏;
  - Offload     = 无损存盘, 留预览 → 已有;
  - ObservationPack = 在 offload 之上加 **稳定句柄 + 分页召回**, 让模型按需读原文。

设计约束(贴合项目):
  纯本地文件存储, 无 LLM/网络; 失败一律返回可读错误而非抛异常; 句柄带会话隔离,
  拒绝读取会话目录之外的文件(路径穿越防护)。
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool

# 句柄前缀/格式: "@obs:<uuid>"。只认 hex+连字符(32~36), 挡住路径分隔符/点号
# → 强制句柄 id 只落进档案目录内, 防路径穿越。
_HANDLE_RE = re.compile(r"^@obs:([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$")

# 默认每页字符数(近似 ~1k token)
_DEFAULT_PAGE_CHARS = 4000

# 模具阈值: 超过则该输出值得进档案(复用 compress 的 offload 阈值语义)
_DEFAULT_ARCHIVE_THRESHOLD_TOKENS = 20000


def _session_root() -> Path:
    """回话级归档根目录: <cache>/observation-pack/<session-id>/。

    用 HUGINN_CACHE_DIR 反映 session; 未设置时回落到 runtime home, 与
    compress._offload_dir() 保持一致。
    """
    base = os.environ.get("HUGINN_CACHE_DIR")
    root = Path(base) if base else (get_runtime_home() / "cache")
    return root / "observation-pack"


def _resolve_dir_for_session(session_id: str) -> Path:
    """按 session 解析目录; 空 session 落到 default。返回已 mkdir 的目录。"""
    safe = (session_id or "default").replace("/", "_").replace("\\", "_")
    return _ensure_dir(_session_root() / safe)


def _ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_runtime_home() -> Path:
    try:
        from huginn.utils.runtime import get_runtime_home as _g

        return _g()
    except Exception:  # noqa: BLE001 — 独立可用兜底
        return Path.home() / ".huginn"


class ObservationArchive:
    """把全量工具输出存档为稳定句柄, 支持按页精确召回。

    句柄 = "@obs:<uuid>", 顶层一个档案只属于一个 session 目录, 防止跨会话猜读。
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or "default"
        self.dir = _resolve_dir_for_session(self.session_id)

    def archive_body(self, body: str, *, obs_id: str | None = None) -> tuple[str, Path]:
        """把 body 写入档案, 返回 (handle, path)。body 为空仍注册(标记空档)。"""
        obs_id = obs_id or uuid.uuid4().hex
        path = self.dir / f"{obs_id}.txt"
        path.write_text(body or "", encoding="utf-8")
        return f"@obs:{obs_id}", path

    def recall(
        self, handle: str, *, page: int = 0, page_size: int = _DEFAULT_PAGE_CHARS
    ) -> dict[str, Any]:
        """按句柄+页码精准召回原文。

        返回 dict: {found, handle, page, page_size, total_pages, text, more}.
        路径穿越防护: 句柄里的 id 必须严格匹配 UUID 形态, 且强制落在 self.dir
        下, 不拼接任意路径。
        """
        m = _HANDLE_RE.match(handle.strip())
        if not m:
            return {"found": False, "error": f"malformed handle: {handle!r}"}
        obs_id = m.group(1)
        path = self.dir / f"{obs_id}.txt"
        if not path.is_file():
            return {"found": False, "error": f"observation not found: {handle}"}
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as e:
            return {"found": False, "error": f"read failed: {e}"}
        page_size = max(1, int(page_size))
        start = max(0, int(page)) * page_size
        chunk = body[start : start + page_size]
        total_pages = max(1, -(-len(body) // page_size))
        page_num = max(0, int(page))
        return {
            "found": True,
            "handle": handle,
            "page": page_num,
            "page_size": page_size,
            "total_pages": total_pages,
            "text": chunk,
            "more": page_num + 1 < total_pages,
        }

    def handle_for_body(self, body: str) -> tuple[str, Path]:
        """便捷入口: 归档正文, 返回 (handle, path)。"""
        return self.archive_body(body)


class _RecallInput(BaseModel):
    handle: str = Field(..., description="例如 @obs:<uuid>; 由工具超长输出句柄给出")
    page: int = Field(default=0, ge=0, description="从 0 起的页码")
    page_size: int = Field(default=_DEFAULT_PAGE_CHARS, ge=1, le=20000)


class RecallObservationTool(HuginnTool):
    """按句柄精确分页召回此前被 offload 的超长工具输出原文。

    配合 ObservationPack 机制: 大输出首次只回预览+句柄, 需要细节时用本工具
    按句柄/页码取回原文, 避免整段重放占用上下文。
    """

    name = "recall_observation"
    description = (
        "Retrieve the exact original text of a previously offloaded (very large) "
        "tool output by its stable handle (@obs:<uuid>), with paged recall. "
        "Use when the preview shown earlier was insufficient and you need the "
        "full content, or a specific part, of a large result."
    )
    category = "meta"
    read_only = True
    destructive = False
    active = True
    input_schema = _RecallInput

    def is_available(self) -> bool:
        return True

    async def _execute(self, args: _RecallInput, context: ToolContext) -> ToolResult:
        session_id = getattr(context, "session_id", None) or "default"
        archive = ObservationArchive(session_id)
        res = archive.recall(args.handle, page=args.page, page_size=args.page_size)
        if not res["found"]:
            return ToolResult(success=False, data=None, error=res.get("error", "not found"))
        return ToolResult(success=True, data=res)


def handle_for_body(
    body: str,
    *,
    session_id: str | None = None,
    threshold_tokens: int = _DEFAULT_ARCHIVE_THRESHOLD_TOKENS,
) -> str | None:
    """把超长正文注册进档案, 返回句柄; 未到阈值或失败返回 None。

    adapter 超长输出缝的接线函数: 注册为句柄, 模型后续可 recall。
    失败静默返回 None(不阻塞), 由调用方决定是否回退到纯预览。
    """
    if not body or not isinstance(body, str):
        return None
    try:
        from huginn.utils.tokens import count_tokens

        if count_tokens(body) <= threshold_tokens:
            return None
    except Exception:  # noqa: BLE001 — token 计数失败不阻塞注册
        pass
    try:
        handle, _path = ObservationArchive(session_id).archive_body(body)
        return handle
    except Exception:  # noqa: BLE001 — 注册失败不阻塞
        return None


def _selfcheck() -> None:
    import tempfile
    import asyncio

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    try:
        # 1. 归档 + 召回精确一致
        archive = ObservationArchive("selftest")
        body = "_".join(f"line{i}" for i in range(1000))
        handle, path = archive.archive_body(body)
        assert handle.startswith("@obs:"), handle
        assert path.is_file()
        rec = archive.recall(handle, page=0, page_size=4000)
        assert rec["found"] and rec["total_pages"] >= 1, rec
        assert rec["handle"] == handle
        # 2. 分页: 拼回分页内容应还原全部正文(无丢字)
        rebuild = ""
        p = 0
        while True:
            r = archive.recall(handle, page=p, page_size=211)
            assert r["found"], r
            rebuild += r["text"][:211] if r["more"] else r["text"]
            if not r["more"]:
                break
            p += 1
        assert rebuild == body, "paged recall must reconstruct the exact body"
        # 3. 坏句柄 / 不存在 → found=False, 不抛
        assert not archive.recall("not-a-handle")["found"]
        assert not archive.recall("@obs:00000000-0000-0000-0000-000000000000")["found"]
        # 4. 路径穿越: 非法 id 不进文件系统
        assert not archive.recall("@obs:../../etc/passwd")["found"]
        # 5. Tool 层: _execute 返回 ToolResult
        from huginn.core_types import ToolContext

        tool = RecallObservationTool()
        ctx = ToolContext(session_id="selftest", workspace=tmp, config=None)

        async def _run():
            r1 = await tool._execute(
                _RecallInput(handle=handle, page=0, page_size=4000), ctx
            )
            assert r1.success and r1.data["handle"] == handle, r1
            rule_bad = _RecallInput(handle="bad", page=0)
            r2 = await tool._execute(rule_bad, ctx)
            assert not r2.success, r2
            miss = _RecallInput(
                handle="@obs:00000000-0000-0000-0000-000000000000", page=0
            )
            r3 = await tool._execute(miss, ctx)
            assert not r3.success, r3

        asyncio.run(_run())
        print("OK observation_pack self-check passed (archive / paged recall / tool)")
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("HUGINN_CACHE_DIR", None)


if __name__ == "__main__":
    _selfcheck()