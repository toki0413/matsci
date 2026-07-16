"""Jupyter notebook cell 编辑工具.

支持 cell 的 insert / replace / delete. 用 nbformat (Jupyter 标准)
读写 .ipynb, 不自动创建新 notebook (YAGNI — 新建直接 touch 或 jupyter create).

ponytail: 只依赖 nbformat (Jupyter 必装), 不做 cell 执行 (执行用 bash_tool
调 jupyter nbconvert --execute 即可, 不重复造轮子).
ceiling: 不支持 cell 执行 / 输出抽取, 只做结构编辑.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from huginn.tools.base import HuginnTool
from huginn.types import ToolResult


class NotebookEditInput(BaseModel):
    notebook_path: str = Field(description=".ipynb 文件绝对路径.")
    edit_mode: str = Field(
        description="insert | replace | delete",
        default="replace",
    )
    cell_index: int = Field(
        default=0, ge=0,
        description="目标 cell 的 0-based 索引. insert 时表示插入位置 (原 cell 及之后后移).",
    )
    cell_type: str = Field(
        default="code",
        description="cell 类型: code | markdown (insert/replace 时用).",
    )
    source: str = Field(
        default="",
        description="cell 内容 (insert/replace 时必填, delete 时忽略).",
    )

    @field_validator("edit_mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("insert", "replace", "delete"):
            raise ValueError(f"edit_mode must be insert/replace/delete, got {v}")
        return v

    @field_validator("cell_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("code", "markdown"):
            raise ValueError(f"cell_type must be code/markdown, got {v}")
        return v


class NotebookEditOutput(BaseModel):
    notebook_path: str
    total_cells: int
    action: str
    cell_index: int


class NotebookEditTool(HuginnTool[NotebookEditInput, NotebookEditOutput]):
    name = "notebook_edit_tool"
    category = "core"
    description = (
        "编辑 Jupyter notebook cell (insert/replace/delete). "
        "支持 .ipynb 文件. 不执行 cell, 不自动创建 notebook. "
        "执行 notebook 用 bash_tool 调 jupyter nbconvert --execute."
    )
    destructive = False  # 改文件但不删数据 (cell delete 可撤销)
    read_only = False
    input_schema = NotebookEditInput
    output_schema = NotebookEditOutput

    async def call(self, args: NotebookEditInput, context) -> ToolResult:
        path = Path(args.notebook_path)
        if not path.exists():
            return ToolResult(
                data=None, success=False,
                error=f"notebook not found: {args.notebook_path}",
            )
        if path.suffix.lower() != ".ipynb":
            return ToolResult(
                data=None, success=False,
                error=f"not a .ipynb file: {args.notebook_path}",
            )
        try:
            import nbformat
        except ImportError as e:
            return ToolResult(
                data=None, success=False,
                error=f"nbformat not installed: {e}",
            )
        try:
            nb = nbformat.read(str(path), as_version=4)
            cells = nb.get("cells", [])
            n_before = len(cells)
            mode = args.edit_mode
            idx = args.cell_index

            if mode == "delete":
                if idx >= n_before:
                    return ToolResult(
                        data=None, success=False,
                        error=f"cell_index {idx} out of range (total {n_before})",
                    )
                del cells[idx]
            else:
                if mode == "insert":
                    if idx > n_before:
                        return ToolResult(
                            data=None, success=False,
                            error=f"insert index {idx} > total {n_before}",
                        )
                    new_cell = nbformat.v4.new_code_cell(args.source) \
                        if args.cell_type == "code" \
                        else nbformat.v4.new_markdown_cell(args.source)
                    cells.insert(idx, new_cell)
                else:  # replace
                    if idx >= n_before:
                        return ToolResult(
                            data=None, success=False,
                            error=f"cell_index {idx} out of range (total {n_before})",
                        )
                    new_cell = nbformat.v4.new_code_cell(args.source) \
                        if args.cell_type == "code" \
                        else nbformat.v4.new_markdown_cell(args.source)
                    cells[idx] = new_cell

            nb["cells"] = cells
            nbformat.write(nb, str(path))
            out = NotebookEditOutput(
                notebook_path=str(path),
                total_cells=len(cells),
                action=mode,
                cell_index=idx,
            )
            return ToolResult(
                data=out.model_dump(),
                success=True,
                side_effects=[f"notebook {mode} cell {idx}: {path.name}"],
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))
