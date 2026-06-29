"""Structure analysis tool — read, analyze, and transform crystal structures.

A read-only tool for structural analysis. Safe to auto-execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.tools.local_structure_db import LocalStructureDB
from huginn.tools.tool_cache import cacheable
from huginn.types import HandleType, ToolContext, ToolResult, ValidationResult
from huginn.validation.handle_validator import HandleValidator


def _file_mtime(path: str) -> float:
    """取文件 mtime，文件不存在返回 0。"""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


# batch_validate 并发验证的最大并发数, 文件 IO + pymatgen 解析, 给 5 够了
_BATCH_VALIDATE_CONCURRENCY = 5


class StructureToolInput(BaseModel):
    action: Literal["read", "analyze", "convert", "compare", "batch_validate"] = Field(...)
    file_path: str | None = Field(
        default=None,
        description="Path to structure file (POSCAR, CIF, XYZ, etc.)",
    )
    output_format: Literal["poscar", "cif", "xyz", "json"] | None = Field(default=None)
    reference_path: str | None = Field(default=None, description="For compare action")
    # batch_validate 专用: 一次验证多个结构文件路径
    files: list[str] | None = Field(
        default=None,
        description="For batch_validate: list of structure file paths to validate",
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> "StructureToolInput":
        """batch_validate 用 files 字段, 其它 action 用 file_path.

        防止 LLM 漏填, 提前在 schema 层就报错, 比等到 call() 里挂掉
        再返回错误友好得多.
        """
        if self.action == "batch_validate":
            if not self.files:
                raise ValueError(
                    "batch_validate requires 'files' (non-empty list of paths)"
                )
        else:
            if not self.file_path:
                raise ValueError(
                    f"action '{self.action}' requires 'file_path'"
                )
        return self


class StructureToolOutput(BaseModel):
    formula: str | None = None
    spacegroup: str | None = None
    lattice_params: dict[str, float] | None = None
    num_atoms: int | None = None
    volume: float | None = None
    density: float | None = None
    warnings: list[str] = []


class StructureTool(HuginnTool):
    """Tool for reading and analyzing crystal structures."""

    name = "structure_tool"
    category = "core"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.PLANNING, ResearchPhase.EXECUTION}),
    )
    description = (
        "Read, analyze, and convert crystal structure files (POSCAR, CIF, XYZ)"
    )
    input_schema = StructureToolInput
    output_schema = StructureToolOutput

    def is_read_only(self, args: StructureToolInput) -> bool:
        return True

    async def validate_input(
        self, args: StructureToolInput, context: ToolContext
    ) -> ValidationResult:
        """Pre-flight: verify structure file exists.

        file_path 也支持传 mp_id 或化学式（如 mp-149 / Si），命中本地
        结构库就算通过，不用非得是磁盘文件。

        batch_validate 不在这里逐个检查文件 —— 缺文件的条目会在 call()
        里返回 per-file error, 让 LLM 看到哪些挂了哪些过了, 比全拦下来
        有用. 这里只确认 files 非空 (model_validator 已经兜底).
        """
        if args.action == "batch_validate":
            # model_validator 已经保证 files 非空, 这里直接放行
            return ValidationResult(result=True)

        vr = HandleValidator.validate(HandleType.FILE_PATH, args.file_path, context)
        if not vr.result:
            # 文件不存在时退一步查本地结构库，命中就放行
            if LocalStructureDB.shared().get(args.file_path) is not None:
                return ValidationResult(result=True)
            return ValidationResult(
                result=False,
                message=f"Structure file not found: {args.file_path}",
                error_code=404,
            )
        if args.reference_path:
            vr2 = HandleValidator.validate(HandleType.FILE_PATH, args.reference_path, context)
            if not vr2.result:
                return ValidationResult(
                    result=False,
                    message=f"Reference file not found: {args.reference_path}",
                    error_code=404,
                )
        return ValidationResult(result=True)

    async def call(self, args: StructureToolInput, context: ToolContext) -> ToolResult:
        # batch_validate 走单独路径, 不查本地结构库 (本地库是按 mp_id
        # 查的, 跟 "验证文件是不是合法结构" 不是一回事)
        if args.action == "batch_validate":
            return await self._handle_batch_validate(args, context)
        # 先查本地结构库，命中直接返回，不用走文件解析
        local = LocalStructureDB.shared().get(args.file_path)
        if local is not None:
            return ToolResult(data=_local_to_output(local), success=True)
        return await self._call_cached(args, context)

    async def _handle_batch_validate(
        self, args: StructureToolInput, context: ToolContext
    ) -> ToolResult:
        """批量验证多个结构文件, 并发受 _BATCH_VALIDATE_CONCURRENCY 限制.

        每个文件单独走 _call_cached (action="read"), 复用 mtime 缓存,
        重复 batch_validate 同一组文件不重读. 单个文件失败 (不存在 /
        解析失败) 不影响其它, 错误塞进该条的 error 字段返回.

        Returns:
            ToolResult.data = {
                "action": "batch_validate",
                "total": N,
                "valid": M,
                "invalid": N - M,
                "results": [
                    {"file": "...", "valid": True, "info": {...}, "error": None},
                    {"file": "...", "valid": False, "info": None, "error": "..."},
                    ...
                ],
            }
        """
        import asyncio

        files = args.files or []
        if not files:
            return ToolResult(
                data=None,
                success=False,
                error="batch_validate requires non-empty 'files' list",
            )

        # 去重保序, 同一个文件验两遍没意义
        seen: set[str] = set()
        unique_files: list[str] = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)

        sem = asyncio.Semaphore(_BATCH_VALIDATE_CONCURRENCY)

        async def _validate_one(file_path: str) -> dict[str, Any]:
            """验单个文件: 走 _call_cached 复用 mtime 缓存."""
            single_args = StructureToolInput(action="read", file_path=file_path)
            async with sem:
                try:
                    result = await self._call_cached(single_args, context)
                except Exception as exc:
                    return {
                        "file": file_path,
                        "valid": False,
                        "info": None,
                        "error": str(exc),
                    }
            if not result.success:
                return {
                    "file": file_path,
                    "valid": False,
                    "info": None,
                    "error": result.error or "unknown error",
                }
            return {
                "file": file_path,
                "valid": True,
                "info": result.data,
                "error": None,
            }

        # 并发跑所有文件, 单个挂掉不影响其它
        per_file = await asyncio.gather(
            *[_validate_one(f) for f in unique_files]
        )

        valid_count = sum(1 for r in per_file if r["valid"])
        data = {
            "action": "batch_validate",
            "total": len(per_file),
            "valid": valid_count,
            "invalid": len(per_file) - valid_count,
            "results": per_file,
        }
        # 整体 success=True: 即使部分文件无效, batch 本身跑成功了,
        # 调用方应该看 per-file 的 valid 字段判断. success=False 留给
        # batch 本身挂掉的场景 (上面已经 early return 了).
        return ToolResult(data=data, success=True)

    @cacheable(
        ttl_seconds=24 * 3600,
        tool_name="structure_tool",
        # 把文件 mtime 拼进 key，文件改了缓存自动失效.
        # batch_validate 不走 _call_cached (在 call() 里就分流了),
        # 所以 file_path 在这里一定是非 None, 不用担心 _file_mtime 挂.
        key_fn=lambda self, args, ctx: {
            "action": args.action,
            "file_path": args.file_path,
            "mtime": _file_mtime(args.file_path),
            "output_format": args.output_format,
            "reference_path": args.reference_path,
        },
    )
    async def _call_cached(
        self, args: StructureToolInput, context: ToolContext
    ) -> ToolResult:
        path = Path(args.file_path)

        if not path.exists():
            return ToolResult(data=None, success=False, error=f"File not found: {path}")

        try:
            # Try to use pymatgen if available
            try:
                from pymatgen.core import Structure

                structure = Structure.from_file(str(path))

                # Use SpacegroupAnalyzer for robust symmetry detection.
                # The raw get_space_group_info() can misidentify spacegroups
                # when the cell is in a non-standard setting.
                spacegroup = None
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                    sga = SpacegroupAnalyzer(structure)
                    spacegroup = sga.get_space_group_symbol()
                except Exception:
                    # Fall back to direct method if analyzer fails
                    if hasattr(structure, "get_space_group_info"):
                        spacegroup = structure.get_space_group_info()[0]

                output = StructureToolOutput(
                    formula=structure.formula,
                    spacegroup=spacegroup,
                    lattice_params={
                        "a": structure.lattice.a,
                        "b": structure.lattice.b,
                        "c": structure.lattice.c,
                        "alpha": structure.lattice.alpha,
                        "beta": structure.lattice.beta,
                        "gamma": structure.lattice.gamma,
                    },
                    num_atoms=len(structure),
                    volume=structure.volume,
                    density=structure.density,
                )

                return ToolResult(data=output.model_dump(), success=True)

            except ImportError:
                # Fallback: basic file info
                content = path.read_text(encoding="utf-8", errors="ignore")
                lines = content.strip().split("\n")

                output = StructureToolOutput(
                    warnings=["pymatgen not installed — providing basic file info only"]
                )

                # Simple POSCAR detection
                if (
                    path.name.upper().startswith("POSCAR") or path.suffix == ".vasp"
                ) and len(lines) >= 6:
                    try:
                        num_atoms = sum(int(x) for x in lines[5].split())
                        output.num_atoms = num_atoms
                    except ValueError:
                        pass

                return ToolResult(data=output.model_dump(), success=True)

        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to parse structure: {e}"
            )


def _local_to_output(struct: dict) -> dict:
    """把本地结构库的 dict 转成 StructureToolOutput 格式。"""
    lattice = struct.get("lattice_params", {})
    positions = struct.get("atomic_positions", {})
    return StructureToolOutput(
        formula=struct.get("formula_pretty") or struct.get("formula"),
        spacegroup=struct.get("space_group"),
        lattice_params={
            "a": lattice.get("a"),
            "b": lattice.get("b"),
            "c": lattice.get("c"),
            "alpha": lattice.get("alpha"),
            "beta": lattice.get("beta"),
            "gamma": lattice.get("gamma"),
        },
        num_atoms=positions.get("num_sites"),
        volume=struct.get("volume"),
        density=struct.get("density"),
        warnings=[f"from local structure db ({struct.get('mp_id', 'unknown')})"],
    ).model_dump()
