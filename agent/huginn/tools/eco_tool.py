"""生态管理工具 —— agent 通过对话安装/启用 skill、插件、MCP server。

把三类"跨运行时生态"的能力接入工具层, 让 agent 不必依赖 HTTP/CLI 也能装载能力:

- ``skill_install`` : 从本地路径 / https raw URL / ``github://owner/repo[/path]``
  拉取 SKILL.md, 落盘到 skills 目录 (默认 ``<workspace>/.huginn/skills``), 再用
  ``SkillImporter`` 解析并注册进 ``SkillRegistry``.
- ``plugin_enable`` : 用 ``PluginLoader.load_one`` 热载一个本地插件.
- ``plugin_disable``: 用 ``PluginLoader.unload`` 卸载一个已加载插件.
- ``mcp_connect``   : 用 ``MCPClientManager`` 注册并连接一个 MCP server
  (stdio / SSE), 再把发现的工具经 ``register_mcp_tools`` 注册进 ``ToolRegistry``.

安全边界: 本工具只搬运"skill/plugin 元数据与 MCP 连接配置", 不从网络内容构造任意
代码执行。
- skill_install 只写 SKILL.md 文本并解析其 frontmatter, 不执行其 steps.
- plugin_enable 加载的是本地 plugins 目录里的插件 (其 main.py 本质是插件代码, 属
  插件系统固有行为), agent 不能借此把任意远端代码变成插件.
- mcp_connect 的 stdio 命令受 ``mcp_client.validate_mcp_command`` 白名单约束
  (默认 python/node/npx/uvx 等), SSE 不拉起本地进程.

可注入点 (单测用 monkeypatch 替换以隔离真实网络 / MCP 握手):
``_skills_install_dir`` / ``_fetch_url_text`` / ``_get_plugin_loader`` /
``_resolve_mcp_manager`` / ``_build_mcp_config`` / ``_register_mcp_tools_and_refresh``。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)

# skill 落盘根目录可通过 HUGINN_SKILLS_DIR 覆盖, 默认 <workspace>/.huginn/skills.
# 与插件目录 (.huginn/plugins) 同层, 语义上是"生态里的技能库".
_SKILLS_DIR_ENV = "HUGINN_SKILLS_DIR"
_DEFAULT_SKILLS_DIR = Path(HUGINN_DIR_NAME) / "skills"

# 与 skill_importer / skill_loader 保持一致的 frontmatter 正则 (仅做目录命名用).
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_GITHUB_SCHEME = "github://"


class EcoToolInput(BaseModel):
    action: Literal[
        "skill_install", "plugin_enable", "plugin_disable", "mcp_connect"
    ] = Field(...)
    # -- skill_install --
    local_path: str | None = Field(
        default=None, description="本地 SKILL.md 文件, 或含 SKILL.md 的目录路径"
    )
    url: str | None = Field(
        default=None,
        description=(
            "skill 来源: https raw URL / github://owner/repo[/path]; "
            "或 mcp_connect(SSE) 时的 SSE 端点"
        ),
    )
    platform: str = Field(
        default="auto", description="skill 源格式: auto/openclaw/hermes"
    )
    # -- plugin / mcp --
    name: str | None = Field(
        default=None,
        description="插件目录名/路径, 或 MCP server 名 (mcp_connect 必填)",
    )
    # -- mcp_connect --
    transport: str = Field(default="stdio", description="stdio 或 sse")
    command: str | None = Field(
        default=None, description="stdio 命令 (受 validate_mcp_command 白名单约束)"
    )
    args: list[str] = Field(default_factory=list, description="stdio 启动参数")
    env: dict[str, str] | None = Field(
        default=None, description="stdio 子进程额外环境变量"
    )


class EcoToolOutput(BaseModel):
    success: bool
    action: str
    result: Any = None
    error: str | None = None


class EcoTool(HuginnTool[EcoToolInput, EcoToolOutput]):
    """安装/启用 skill、插件、MCP server 的生态管理工具."""

    name = "eco_tool"
    category = "meta"
    description = (
        "Manage the runtime ecosystem from conversation: skill_install (install a "
        "SKILL.md from a local path, https raw URL, or github://owner/repo[/path]), "
        "plugin_enable/plugin_disable (hot-load/unload a Star plugin by directory "
        "name or path), mcp_connect (register+connect an MCP server via stdio "
        "command or SSE url and register its tools). Requires local_path or url for "
        "skills, name for plugins/MCP."
    )
    destructive = False
    read_only = False  # 三类操作都会改变运行时生态状态, 非只读
    input_schema = EcoToolInput
    output_schema = EcoToolOutput

    async def call(self, args: EcoToolInput, context: ToolContext) -> ToolResult:
        # 兼容两种调用方: 直接传 Pydantic 模型, 或传 dict (对齐 ConfigDomainTool).
        if isinstance(args, dict):
            inp = EcoToolInput(**args)
        else:
            inp = args
        if inp.action == "skill_install":
            return self._skill_install(inp)
        if inp.action == "plugin_enable":
            return self._plugin_enable(inp)
        if inp.action == "plugin_disable":
            return self._plugin_disable(inp)
        if inp.action == "mcp_connect":
            return await self._mcp_connect(inp)
        return self._out_err(inp.action, f"未知 action: {inp.action}")

    # ── skill_install ──────────────────────────────────────────────

    def _skill_install(self, args: EcoToolInput) -> ToolResult:
        """拉取/复制 SKILL.md 落盘并注册进 SkillRegistry."""
        from huginn.plugins.skill_importer import SkillImporter
        from huginn.skills.registry import SkillRegistry

        install_dir = self._skills_install_dir()
        install_dir.mkdir(parents=True, exist_ok=True)
        importer = SkillImporter()

        if args.local_path:
            src = Path(args.local_path)
            if not src.exists():
                return self._out_err(args.action, f"本地路径不存在: {args.local_path}")
            if src.is_dir():
                self._copy_skill_dir(src, install_dir)
                targets: list[Path] = [install_dir / src.name]
            else:
                text = src.read_text(encoding="utf-8")
                targets = [self._write_skill_text(text, install_dir, src.stem)]
        elif args.url:
            try:
                text = self._fetch_url_text(args.url)
            except Exception as exc:
                logger.warning("skill 拉取失败: %s", exc)
                return self._out_err(args.action, f"拉取 skill 失败: {exc}")
            targets = [
                self._write_skill_text(text, install_dir, self._url_basename(args.url))
            ]
        else:
            return self._out_err(
                args.action, "skill_install 需要 local_path 或 url 参数之一"
            )

        skills = []
        for t in targets:
            if t.is_dir():
                skills.extend(importer.import_directory(t, args.platform))
            else:
                skills.append(importer.import_file(t, args.platform))

        for skill in skills:
            SkillRegistry.register(skill)

        return self._out_ok(
            args.action,
            {
                "count": len(skills),
                "location": str(install_dir),
                "installed": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "platform": s.metadata.get("platform", "unknown"),
                        "steps": len(s.steps),
                    }
                    for s in skills
                ],
            },
        )

    def _skills_install_dir(self) -> Path:
        """skill 落盘根目录: $HUGINN_SKILLS_DIR > <workspace>/.huginn/skills."""
        raw = os.environ.get(_SKILLS_DIR_ENV)
        if raw:
            return Path(raw)
        workspace = os.environ.get("HUGINN_WORKSPACE", ".")
        return Path(workspace) / _DEFAULT_SKILLS_DIR

    @staticmethod
    def _copy_skill_dir(src: Path, install_dir: Path) -> None:
        """把整个技能目录复制进 install_dir (保留脚本等副产物)."""
        target = install_dir / src.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)

    def _write_skill_text(self, text: str, install_dir: Path, fallback: str) -> Path:
        """按 frontmatter 里的 name (没有则用 fallback) 落盘 SKILL.md."""
        name = self._derive_skill_name(text, fallback)
        target = install_dir / name / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def _derive_skill_name(self, text: str, fallback: str) -> str:
        """优先取 frontmatter 的 name, 否则用 fallback 做 slug."""
        match = _FRONTMATTER_RE.match(text)
        if match:
            try:
                fm = yaml.safe_load(match.group(1)) or {}
                raw = fm.get("name")
                if isinstance(raw, str) and raw.strip():
                    return self._slug(raw)
            except Exception:
                logger.debug("skill 名解析失败, 用 fallback", exc_info=True)
        return self._slug(fallback)

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_\-.]", "_", value).strip("_")
        return cleaned or "skill"

    def _fetch_url_text(self, url: str) -> str:
        """从 https raw URL 或 github:// 取 SKILL.md 文本. 失败抛清晰错误."""
        import urllib.request

        http_url = self._to_http_url(url)
        try:
            with urllib.request.urlopen(http_url, timeout=30) as resp:
                data = resp.read()
        except Exception as exc:
            raise ValueError(
                f"网络拉取失败 {http_url!r}: {exc.__class__.__name__}: {exc}"
            ) from exc
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"SKILL.md 不是 utf-8 编码: {exc}") from exc

    @staticmethod
    def _to_http_url(url: str) -> str:
        """把 github://owner/repo[/path] 转成 raw.githubusercontent URL."""
        if url.startswith(_GITHUB_SCHEME):
            return EcoTool._github_to_raw(url)
        if url.startswith(("https://", "http://")):
            return url
        raise ValueError(
            f"不支持的 URL: {url!r} (支持 https(s) raw URL 或 github://owner/repo[/path])"
        )

    @staticmethod
    def _github_to_raw(url: str) -> str:
        rest = url[len(_GITHUB_SCHEME) :].rstrip("/")
        parts = rest.split("/")
        if len(parts) < 2:
            raise ValueError(f"github:// 至少需要 owner/repo: {url!r}")
        owner, repo = parts[0], parts[1]
        sub = "/".join(parts[2:])
        base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD"
        if sub:
            return f"{base}/{sub}/SKILL.md"
        return f"{base}/SKILL.md"

    @staticmethod
    def _url_basename(url: str) -> str:
        """从 URL 取一个可读的 fallback 目录名."""
        segment = url.rstrip("/").split("?")[0].split("/")[-1]
        if segment.lower() == "skill.md":
            segment = url.rstrip("/").split("/")[-2]
        return segment or "skill"

    # ── plugin_enable / plugin_disable ─────────────────────────────

    def _plugin_enable(self, args: EcoToolInput) -> ToolResult:
        loader = self._get_plugin_loader()
        if not args.name:
            return self._out_err(args.action, "plugin_enable 需要 name (目录名或路径)")
        plugin_dir = self._resolve_plugin_dir(args.name, loader)
        if plugin_dir is None or not (plugin_dir / "metadata.yaml").is_file():
            return self._out_err(
                args.action, f"插件不存在或缺少 metadata.yaml: {args.name}"
            )
        try:
            meta = loader.load_one(plugin_dir)
        except Exception as exc:
            logger.warning("插件 %s 加载失败: %s", args.name, exc)
            return self._out_err(args.action, f"插件加载失败 {args.name}: {exc}")
        if meta is None:
            return self._out_err(args.action, f"插件未返回 metadata: {args.name}")
        return self._out_ok(
            args.action,
            {
                "plugin": meta.name,
                "version": meta.version,
                "loaded": meta.name in loader.list_loaded(),
            },
        )

    def _plugin_disable(self, args: EcoToolInput) -> ToolResult:
        loader = self._get_plugin_loader()
        if not args.name:
            return self._out_err(args.action, "plugin_disable 需要 name")
        if not loader.unload(args.name):
            return self._out_err(args.action, f"插件未加载, 无需卸载: {args.name}")
        return self._out_ok(args.action, {"plugin": args.name})

    def _get_plugin_loader(self):
        """优先取 server context 里的共享 PluginLoader, 否则新建一个默认 loader."""
        from huginn.plugins.loader import PluginLoader
        from huginn.server_core import get_context

        ctx = get_context()
        loader = getattr(ctx, "plugin_loader", None)
        if loader is not None:
            return loader
        return PluginLoader()

    @staticmethod
    def _resolve_plugin_dir(name: str, loader) -> Path | None:
        """name 若是绝对/相对存在的目录直接用; 否则按默认插件目录解析插件名."""
        direct = Path(name)
        if direct.is_dir():
            return direct
        base = (
            Path(getattr(loader, "plugins_dir", "."))
            if loader is not None
            else Path(".")
        )
        candidate = base / name
        return candidate if candidate.is_dir() else None

    # ── mcp_connect ────────────────────────────────────────────────

    async def _mcp_connect(self, args: EcoToolInput) -> ToolResult:
        mgr = self._resolve_mcp_manager()
        if not args.name:
            return self._out_err(args.action, "mcp_connect 需要 name")
        try:
            config = self._build_mcp_config(
                name=args.name,
                transport=args.transport,
                command=args.command,
                args=args.args,
                env=args.env,
                url=args.url,
            )
            await mgr.connect(config, origin="eco_tool")
            registered = await self._register_mcp_tools_and_refresh(mgr, args.name)
        except Exception as exc:
            logger.warning("MCP 连接失败 %s: %s", args.name, exc)
            return self._out_err(args.action, f"MCP 连接失败: {exc}")
        return self._out_ok(
            args.action,
            {
                "server": args.name,
                "transport": args.transport,
                "tools": [
                    {"name": t.name, "description": t.description} for t in registered
                ],
            },
        )

    @staticmethod
    def _resolve_mcp_manager():
        """从 server context 取共享 MCPClientManager, 没有则新建并挂回去."""
        from huginn.server_core import get_context

        ctx = get_context()
        mgr = getattr(ctx, "mcp_manager", None)
        if mgr is not None:
            return mgr
        from huginn.mcp_client import MCPClientManager

        mgr = MCPClientManager()
        ctx.mcp_manager = mgr
        return mgr

    @staticmethod
    def _build_mcp_config(
        name: str,
        transport: str,
        command: str | None,
        args: list[str],
        env: dict[str, str] | None,
        url: str | None,
    ):
        """构造 MCPServerConfig; stdio 命令过白名单校验, SSE 需 url."""
        from huginn.mcp_client import MCPServerConfig, validate_mcp_command

        if transport == "sse":
            if not url:
                raise ValueError("SSE transport 需要 url 参数")
            return MCPServerConfig(
                name=name, command="", args=[], transport="sse", url=url
            )
        cmd = validate_mcp_command(command or "python")
        return MCPServerConfig(
            name=name,
            command=cmd,
            args=list(args or []),
            env=dict(env) if env else None,
        )

    @staticmethod
    async def _register_mcp_tools_and_refresh(mgr, server_name: str):
        """注册 MCP 工具/提示词, 并刷新已实例化 agent 的工具缓存.

        对齐 routes/mcp.py 的 reconnect 逻辑: 更新 ToolRegistry 后, 若 server context
        里已有 agent, 调 refresh_tools_from_registry 让它立刻看到新工具.
        """
        import contextlib

        from huginn.tools.mcp_adapter import register_mcp_prompts, register_mcp_tools

        registered = register_mcp_tools(mgr, server_name=server_name)
        with contextlib.suppress(Exception):
            await register_mcp_prompts(mgr, server_name=server_name)
        try:
            from huginn.server_core import get_context

            agent = getattr(get_context(), "agent", None)
            if agent is not None:
                with contextlib.suppress(Exception):
                    agent.refresh_tools_from_registry()
        except Exception:
            logger.debug("agent 工具刷新失败(非致命)", exc_info=True)
        return registered

    # ── 输出封装 ───────────────────────────────────────────────────

    @staticmethod
    def _out_ok(action: str, result: Any) -> ToolResult:
        return ToolResult(
            data=EcoToolOutput(success=True, action=action, result=result).model_dump(),
            success=True,
        )

    @staticmethod
    def _out_err(action: str, error: str) -> ToolResult:
        return ToolResult(
            data=EcoToolOutput(success=False, action=action, error=error).model_dump(),
            success=False,
            error=error,
        )
