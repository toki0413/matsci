"""Generative Design 工具: 代码驱动的设计闭环 (LLM 生代码 → 渲染 → 反馈).

设计思路 (长期2):
- 把 DesignAtom 的"渲染"和 Sandbox 的"执行"串成闭环
- LLM 一次调用就能拿到: 生成的代码 + 执行结果 + 输出文件路径
- 失败时返回 stderr + 行号, LLM 据此调整 atom 参数重试
- 适合: 反复试配色 / 试图表样式 / 试布局 → 看渲染效果 → 微调

actions:
- render_and_run:  接收 atoms, 渲染代码 → 写文件 → 执行 → 返回结果
- render_only:     只渲染不执行, 返回代码让 LLM 审查
- run_from_code:   LLM 自带代码, 直接执行 (跟 code_tool 类似但走同一接口)

闭环示例:
  用户: "画个柱状图, 配色用蓝色"
  → LLM 调 render_and_run atoms=[dataviz.bar(data=..., color='#2563eb')]
  → 工具返回: 代码 + stdout + 输出文件 bar.png 路径
  → LLM 看到成功, 把 bar.png 路径告诉用户
  → 用户: "换成红色"
  → LLM 调 render_and_run atoms=[dataviz.bar(data=..., color='#ef4444')]
  → 新 bar.png 覆盖, 闭环完成
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.tools.design_atom_tool import (
    DesignAtomTool,
    _ATOM_REGISTRY,
    _RENDERERS,
)
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class GenerativeDesignInput(BaseModel):
    action: Literal["render_and_run", "render_only", "run_from_code"] = Field(...)

    # render_and_run / render_only 时必填
    atoms: list[dict[str, Any]] | None = Field(
        default=None,
        description="原子列表, 每项 {atom_name, params}",
    )
    output_format: Literal["html", "python", "auto"] = Field(
        default="auto", description="输出格式"
    )

    # run_from_code 时必填
    code: str | None = Field(default=None, description="直接执行的代码")
    language: Literal["python", "html"] = Field(
        default="python", description="代码语言"
    )

    # 通用可选
    work_dir: str | None = Field(
        default=None, description="工作目录, 默认临时目录"
    )
    filename: str | None = Field(
        default=None, description="输出文件名, 默认 design_output.<ext>"
    )
    timeout: int = Field(default=30, description="执行超时秒数")


class GenerativeDesignTool(HuginnTool):
    """Generative Design: 渲染 → 执行 → 反馈 一站式闭环."""

    name = "generative_design_tool"
    category = "design"
    description = (
        "Code-driven generative design loop: render DesignAtoms to code, "
        "execute it in sandbox, return stdout/stderr/output files. "
        "Actions: render_and_run (render atoms + execute), "
        "render_only (just render code for review), "
        "run_from_code (execute LLM-provided code)."
    )
    input_schema = GenerativeDesignInput

    # 不强依赖 sandbox, 没有就降级成只渲染不执行
    def __init__(self, sandbox: Any | None = None) -> None:
        super().__init__()
        self.sandbox = sandbox
        self._atom_tool = DesignAtomTool()

    def is_read_only(self, args: GenerativeDesignInput) -> bool:
        # render_only 是只读, render_and_run 会执行代码不算只读
        return args.action == "render_only"

    async def validate_input(
        self, args: GenerativeDesignInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action in ("render_and_run", "render_only"):
            if not args.atoms:
                return ValidationResult(
                    result=False,
                    message=f"{args.action} 需要 atoms 列表",
                )
            for a in args.atoms:
                if "atom_name" not in a:
                    return ValidationResult(
                        result=False,
                        message="atoms 每项必须有 atom_name",
                    )
                if a["atom_name"] not in _ATOM_REGISTRY:
                    return ValidationResult(
                        result=False,
                        message=f"未知原子: {a['atom_name']}",
                    )
        if args.action == "run_from_code" and not args.code:
            return ValidationResult(
                result=False, message="run_from_code 需要 code"
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GenerativeDesignInput(**args)

        try:
            if input_data.action == "render_only":
                return self._render_only(input_data)

            if input_data.action == "render_and_run":
                return await self._render_and_run(input_data)

            if input_data.action == "run_from_code":
                return await self._run_from_code(input_data)

            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {input_data.action}",
            )

        except Exception as e:
            logger.warning("generative_design_tool failed: %s", e, exc_info=True)
            return ToolResult(data=None, success=False, error=str(e))

    def _render_only(self, args: GenerativeDesignInput) -> ToolResult:
        """只渲染代码不执行."""
        fmt = args.output_format
        if fmt == "auto":
            fmt = self._detect_format(args.atoms or [])
        code = self._build_code(args.atoms or [], fmt)
        return ToolResult(
            data={
                "code": code,
                "format": fmt,
                "atom_count": len(args.atoms or []),
            },
            success=True,
        )

    async def _render_and_run(
        self, args: GenerativeDesignInput
    ) -> ToolResult:
        """渲染 + 执行 + 返回反馈."""
        fmt = args.output_format
        if fmt == "auto":
            fmt = self._detect_format(args.atoms or [])
        code = self._build_code(args.atoms or [], fmt)

        work_dir = Path(args.work_dir) if args.work_dir else Path.cwd() / "_design_runs"
        work_dir.mkdir(parents=True, exist_ok=True)

        ext = "py" if fmt == "python" else "html"
        filename = args.filename or f"design_output.{ext}"
        file_path = work_dir / filename
        file_path.write_text(code, encoding="utf-8")

        # html 不执行, 直接返回文件路径让用户在浏览器打开
        if fmt == "html":
            return ToolResult(
                data={
                    "code": code,
                    "format": "html",
                    "file_path": str(file_path),
                    "message": (
                        f"HTML 已生成: {file_path}. 用浏览器打开预览. "
                        "需要调整就改 atom 参数重新 render_and_run."
                    ),
                },
                success=True,
            )

        # python: 调 sandbox 执行
        if not self.sandbox:
            return ToolResult(
                data={
                    "code": code,
                    "format": "python",
                    "file_path": str(file_path),
                    "message": (
                        "无 sandbox 可用, 代码已写入文件但未执行. "
                        f"可手动运行: python {file_path}"
                    ),
                },
                success=True,
            )

        try:
            start = time.time()
            result = self.sandbox.run(
                code,
                work_dir=str(work_dir),
                timeout=args.timeout,
            )
            dt = time.time() - start
            # SandboxExecutor.run 返回 dict: {stdout, stderr, returncode, ...}
            stdout = ""
            stderr = ""
            returncode = 0
            output_files: list[str] = []
            if isinstance(result, dict):
                stdout = result.get("stdout", "") or ""
                stderr = result.get("stderr", "") or ""
                returncode = int(result.get("returncode", 0) or 0)
                output_files = result.get("output_files", []) or []
            elif hasattr(result, "stdout"):
                stdout = getattr(result, "stdout", "") or ""
                stderr = getattr(result, "stderr", "") or ""
                returncode = int(getattr(result, "returncode", 0) or 0)

            # 扫工作目录看新生成的文件 (png/json/csv 等)
            try:
                after_files = set(work_dir.iterdir())
                before = {file_path}  # 至少有自己
                new_files = [
                    str(p) for p in after_files - before
                    if p.is_file() and p.suffix in (".png", ".jpg", ".svg", ".json", ".csv", ".pdf")
                ]
                if new_files:
                    output_files = list(set(output_files + new_files))
            except Exception:
                pass

            return ToolResult(
                data={
                    "code": code,
                    "format": "python",
                    "file_path": str(file_path),
                    "stdout": stdout[:4000],  # 截断防 token 爆
                    "stderr": stderr[:4000],
                    "returncode": returncode,
                    "output_files": output_files,
                    "duration_ms": int(dt * 1000),
                    "success": returncode == 0,
                    "message": (
                        "执行成功, 输出文件: "
                        + ", ".join(output_files)
                        if returncode == 0 and output_files
                        else (
                            "执行成功" if returncode == 0
                            else f"执行失败 returncode={returncode}, stderr 见上"
                        )
                    ),
                },
                success=returncode == 0,
            )
        except Exception as e:
            return ToolResult(
                data={
                    "code": code,
                    "format": "python",
                    "file_path": str(file_path),
                    "error": str(e),
                    "message": f"sandbox 执行异常: {e}",
                },
                success=False,
                error=str(e),
            )

    async def _run_from_code(
        self, args: GenerativeDesignInput
    ) -> ToolResult:
        """直接执行 LLM 提供的代码, 跟 render_and_run 共用执行路径."""
        work_dir = Path(args.work_dir) if args.work_dir else Path.cwd() / "_design_runs"
        work_dir.mkdir(parents=True, exist_ok=True)
        ext = "py" if args.language == "python" else "html"
        filename = args.filename or f"design_output.{ext}"
        file_path = work_dir / filename
        file_path.write_text(args.code or "", encoding="utf-8")

        if args.language == "html" or not self.sandbox:
            return ToolResult(
                data={
                    "code": args.code,
                    "format": args.language,
                    "file_path": str(file_path),
                    "message": f"代码已写入 {file_path}",
                },
                success=True,
            )

        try:
            start = time.time()
            result = self.sandbox.run(
                args.code,
                work_dir=str(work_dir),
                timeout=args.timeout,
            )
            dt = time.time() - start
            stdout = ""
            stderr = ""
            returncode = 0
            if isinstance(result, dict):
                stdout = result.get("stdout", "") or ""
                stderr = result.get("stderr", "") or ""
                returncode = int(result.get("returncode", 0) or 0)
            elif hasattr(result, "stdout"):
                stdout = getattr(result, "stdout", "") or ""
                stderr = getattr(result, "stderr", "") or ""
                returncode = int(getattr(result, "returncode", 0) or 0)
            return ToolResult(
                data={
                    "code": args.code,
                    "file_path": str(file_path),
                    "stdout": stdout[:4000],
                    "stderr": stderr[:4000],
                    "returncode": returncode,
                    "duration_ms": int(dt * 1000),
                    "success": returncode == 0,
                },
                success=returncode == 0,
            )
        except Exception as e:
            return ToolResult(
                data={"code": args.code, "error": str(e)},
                success=False,
                error=str(e),
            )

    def _detect_format(self, atoms: list[dict[str, Any]]) -> str:
        for a in atoms:
            if a["atom_name"].startswith("dataviz."):
                return "python"
        return "html"

    def _build_code(self, atoms: list[dict[str, Any]], fmt: str) -> str:
        """复用 DesignAtomTool 的渲染逻辑拼代码."""
        snippets = []
        for a in atoms:
            name = a["atom_name"]
            params = a.get("params", {})
            defaults = {
                k: v["default"]
                for k, v in _ATOM_REGISTRY[name]["params"].items()
            }
            merged = {**defaults, **(params or {})}
            renderer = _RENDERERS.get(name)
            if renderer:
                snippets.append(renderer(merged))

        if fmt == "python":
            parts = [
                "# GenerativeDesignTool generated code",
                "# Arial 20pt+ bold 按用户规范",
                "",
            ]
            for a, s in zip(atoms, snippets):
                parts.append(f"# === {a['atom_name']} ===")
                parts.append(s)
                parts.append("")
            return "\n".join(parts)

        # html: 拼完整文档
        css_parts = []
        html_parts = []
        svg_parts = []
        for a, s in zip(atoms, snippets):
            cat = _ATOM_REGISTRY[a["atom_name"]]["category"]
            if cat == "style":
                css_parts.append(f"/* {a['atom_name']} */\n{s}")
            elif cat == "layout" and "<" in s:
                html_parts.append(f"<!-- {a['atom_name']} -->\n{s}")
            elif cat == "layout":
                css_parts.append(f"/* {a['atom_name']} */\n{s}")
            elif cat == "geometry":
                svg_parts.append(f"<!-- {a['atom_name']} -->\n{s}")
        css_block = "\n".join(css_parts) or "/* no css */"
        html_block = "\n".join(html_parts)
        svg_block = "\n".join(svg_parts)
        return (
            "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            "<title>Generative Design</title>\n<style>\n"
            f"{css_block}\n</style>\n</head>\n<body>\n"
            f"{html_block}\n{svg_block}\n</body>\n</html>\n"
        )
