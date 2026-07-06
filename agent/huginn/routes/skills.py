"""Skill listing and execution endpoints."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from huginn.server_core import get_agent_factory, get_memory_manager
from huginn.skills.base import DeclarativeSkillExecutor
from huginn.skills.registry import SkillRegistry
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext

logger = logging.getLogger(__name__)

router = APIRouter(tags=["skills"])


@router.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    """List all registered skills."""
    # Ensure presets are loaded and registered
    from huginn.skills import presets  # noqa: F401

    return [
        {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in skill.parameters
            ],
            "tags": skill.tags,
        }
        for skill in SkillRegistry.get_all_definitions()
    ]


@router.post("/skills/execute")
async def execute_skill(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a skill by name with the provided arguments."""
    from huginn.skills import presets  # noqa: F401

    skill_name = params.get("skill")
    skill_args = params.get("args", {})

    skill = SkillRegistry.get(skill_name)
    if not skill:
        return {"error": f"Skill '{skill_name}' not found"}

    executor = DeclarativeSkillExecutor(ToolRegistry)
    context = ToolContext(
        session_id="http",
        workspace=".",
        memory_manager=get_memory_manager(),
        agent_factory=get_agent_factory(),
    )
    result = await executor.execute(skill, skill_args, context.__dict__)
    # Keep the response JSON-serializable: drop non-primitive objects from the
    # skill's final context (e.g. MemoryManager, AgentFactory).
    if isinstance(result, dict) and "context" in result:
        safe_types = (str, int, float, bool, type(None), list, dict, tuple)
        result["context"] = {
            k: v
            for k, v in result["context"].items()
            if isinstance(v, safe_types)
        }
    return result


@router.post("/skills/import")
async def import_skill(
    file: UploadFile | None = File(None),
    path: str | None = Form(None),
    platform: str = Form("auto"),
) -> dict[str, Any]:
    """导入 OpenClaw / Hermes / Huginn 格式的技能文件或目录。

    两种用法二选一：
    - 上传单个 SKILL.md（multipart file 字段）
    - 给服务器本地路径 path（文件或目录均可）
    platform 默认 auto，按 frontmatter 自动识别来源。
    导入成功的技能会注册进 SkillRegistry，立刻能在 GET /skills 里看到。
    """
    from huginn.plugins.skill_importer import SkillImporter

    importer = SkillImporter()
    skills = []

    if file is not None:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="上传文件为空")
        # 落临时文件再交给导入器，和 document.py 一个套路
        tmp = Path(tempfile.gettempdir()) / f"skill_{uuid.uuid4().hex}.md"
        try:
            tmp.write_bytes(raw)
            skills = [importer.import_file(tmp, platform)]
        finally:
            tmp.unlink(missing_ok=True)
    elif path:
        p = Path(path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"路径不存在: {path}")
        if p.is_dir():
            skills = importer.import_directory(p, platform)
        else:
            skills = [importer.import_file(p, platform)]
    else:
        raise HTTPException(
            status_code=400, detail="需要上传文件或提供 path 参数"
        )

    for s in skills:
        SkillRegistry.register(s)

    logger.info("从 %s 导入 %d 个技能", path or file.filename, len(skills))
    return {
        "count": len(skills),
        "imported": [
            {
                "name": s.name,
                "description": s.description,
                "platform": s.metadata.get("platform", "unknown"),
                "steps": len(s.steps),
                "required_tools": s.required_tools,
                "tags": s.tags,
            }
            for s in skills
        ],
    }
