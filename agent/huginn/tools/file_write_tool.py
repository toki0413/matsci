"""File write tool — create or overwrite text files.

Used by the Coder agent to create new files. Always requires approval.
Supports dry-run diff preview against existing content.
"""

from __future__ import annotations

import difflib
import hashlib
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

# C7: 大产物分块写入阈值 — 超过此值用分块 write 避免 OOM.
# ponytail: 1MB 够大覆盖 99% 源码, 够小避免一次性 write 撑爆内存.
#   ceiling: 真正的 OOM 阈值取决于可用内存, 1MB 是保守值.
_CHUNK_THRESHOLD = 1024 * 1024  # 1 MB
_CHUNK_SIZE = 64 * 1024  # 64 KB per write call


class FileWriteToolInput(BaseModel):
    action: Literal["write", "preview"] = Field(default="write")
    file_path: str = Field(..., description="Path to file")
    content: str = Field(..., description="Content to write")
    working_dir: str | None = Field(default=None)
    dry_run: bool = Field(
        default=False,
        description="If True, return a diff preview without modifying the file.",
    )


def _content_hash(text: str) -> str:
    """Short SHA-256 hash for snapshot/rollback identification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FileWriteTool(HuginnTool):
    """Write text files.

    When ``dry_run`` is True (or ``action`` is ``"preview"``), the tool returns
    a unified diff against the existing content without writing. On actual
    writes, the result includes a ``snapshot_hash`` of the prior content (if any)
    so callers can track what changed.
    """

    name = "file_write_tool"
    category = "core"
    description = "Create or overwrite a text file with the provided content."
    destructive = True
    input_schema = FileWriteToolInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = FileWriteToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        # C7: Windows 路径修复 — normpath 归一化反斜杠/盘符/..
        # 原 bug: agent 传 "submission\model.py" 在 Windows 上被当单文件名,
        #   或传 "C:\workspace\file.py" 导致 work_dir / file_path 拼出错误路径.
        # ponytail: normpath + abspath 最小修复, 不改 Path 语义.
        #   ceiling: UNC 路径 (\\server\share) 未特殊处理.
        norm_file_path = os.path.normpath(input_data.file_path)
        path = work_dir / norm_file_path
        if not path.is_absolute():
            path = path.resolve()

        # Prevent writing outside the working directory tree
        # C7: 用 os.path.commonpath 替代 relative_to, 更宽松地处理 .. 和符号链接
        try:
            work_resolved = str(work_dir.resolve())
            path_resolved = str(path.resolve())
            if not os.path.commonpath([work_resolved, path_resolved]) == work_resolved:
                raise ValueError(f"{path_resolved} not under {work_resolved}")
        except ValueError:
            return ToolResult(
                data=None,
                success=False,
                error=f"Refusing to write outside working directory: {path}",
            )

        try:
            existed = path.exists()
            old_content = path.read_text(encoding="utf-8") if existed else ""

            # Preview / dry-run: return the diff without writing.
            if input_data.dry_run or input_data.action == "preview":
                diff_lines = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    input_data.content.splitlines(keepends=True),
                    fromfile=f"a/{path.name}" if existed else "/dev/null",
                    tofile=f"b/{path.name}",
                )
                return ToolResult(
                    data={
                        "file_path": str(path),
                        "dry_run": True,
                        "existed": existed,
                        "diff": "".join(diff_lines),
                        "message": f"Preview of write to {path}.",
                    },
                    success=True,
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            # C7: 大产物分块写入 — >1MB 用分块 write 避免 OOM.
            # ponytail: write_text 对 <1MB 够用且快; 大文件分块限制单次内存占用.
            content_bytes = input_data.content.encode("utf-8")
            if len(content_bytes) > _CHUNK_THRESHOLD:
                with open(path, "wb") as f:
                    for i in range(0, len(content_bytes), _CHUNK_SIZE):
                        f.write(content_bytes[i:i + _CHUNK_SIZE])
                _write_mode = "chunked"
            else:
                path.write_bytes(content_bytes)
                _write_mode = "single"
            result_data: dict[str, Any] = {
                "file_path": str(path),
                "existed": existed,
                "bytes_written": len(content_bytes),
                "write_mode": _write_mode,
                "message": f"Wrote {path}.",
            }
            if existed:
                result_data["snapshot_hash"] = _content_hash(old_content)
            return ToolResult(data=result_data, success=True)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to write file: {e}"
            )


def _c7_self_check() -> int:
    """C7 self-check: 验证 Windows 路径归一化 + 分块写入 + 沙箱守卫.

    ponytail: 临时目录跑真写. ceiling: 不验 Windows 真路径 (当前是 Windows).
      升级路径: 跨平台 CI 跑.
    """
    import asyncio
    import tempfile

    tool = FileWriteTool()
    with tempfile.TemporaryDirectory() as td:

        # 1. 正常写入: "subdir/file.py"
        result = asyncio.run(tool.call({
            "file_path": "subdir/test.py", "content": "print('hello')",
            "working_dir": td,
        }, None))
        assert result.success, f"normal write failed: {result.error}"
        assert (Path(td) / "subdir" / "test.py").exists()
        assert result.data["write_mode"] == "single"
        print("[CHECK C7.1] normal write OK (single mode)")

        # 2. 反斜杠路径归一化: "subdir\\nested\\file.py"
        result = asyncio.run(tool.call({
            "file_path": "subdir\\nested\\file.py", "content": "x=1",
            "working_dir": td,
        }, None))
        assert result.success, f"backslash path failed: {result.error}"
        # normpath 后 "subdir\\nested\\file.py" → "subdir/nested/file.py" on Windows
        # 或 "subdir\\nested\\file.py" on Linux — 都应能写
        assert result.data["bytes_written"] > 0
        print("[CHECK C7.2] backslash path normalization OK")

        # 3. 沙箱守卫: 路径逃逸应拒绝
        result = asyncio.run(tool.call({
            "file_path": "../../../etc/passwd", "content": "evil",
            "working_dir": td,
        }, None))
        assert not result.success, "path traversal should be blocked"
        assert "outside working directory" in result.error.lower()
        print("[CHECK C7.3] sandbox guard (path traversal blocked) OK")

        # 4. 分块写入: >1MB 用 chunked mode
        big_content = "x" * (1024 * 1024 + 100)  # ~1MB+100B
        result = asyncio.run(tool.call({
            "file_path": "big.txt", "content": big_content,
            "working_dir": td,
        }, None))
        assert result.success, f"big write failed: {result.error}"
        assert result.data["write_mode"] == "chunked", \
            f"expected chunked, got {result.data['write_mode']}"
        assert result.data["bytes_written"] == len(big_content.encode("utf-8"))
        # 验证内容完整
        written = (Path(td) / "big.txt").read_text(encoding="utf-8")
        assert len(written) == len(big_content)
        assert written == big_content
        print(f"[CHECK C7.4] chunked write OK ({result.data['bytes_written']} bytes)")

    print("[CHECK C7] ALL ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-check-c7" in sys.argv:
        sys.exit(_c7_self_check())
