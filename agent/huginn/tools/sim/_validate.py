"""共享的工作目录有效性校验 — gaussian / orca / vasp 的 validate_input 共用.

DRY: 三个仿真工具的 ``validate_input`` 逐字相同 (HandleValidator 校 FILE_PATH,
失败回 404). 复制版本让 "工作目录缺失" 的错误码/文案有了三个漂移点.
"""
from __future__ import annotations

from typing import Any

from huginn.core_types import HandleType, ToolContext, ValidationResult
from huginn.validation.handle_validator import HandleValidator


async def validate_working_dir(args: Any, context: ToolContext) -> ValidationResult:
    """校验 ``args.working_dir`` 存在; 不存在返回 404."""
    vr = HandleValidator.validate(HandleType.FILE_PATH, args.working_dir, context)
    if not vr.result:
        return ValidationResult(
            result=False,
            message=f"Working directory not found: {args.working_dir}",
            error_code=404,
        )
    return ValidationResult(result=True)
