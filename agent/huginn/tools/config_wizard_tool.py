"""配置向导工具 —— 帮用户选 provider / 校验配置 / 迁移 env / 接本地模型。

actions:
- list_providers:    列出 19 个 provider + 默认 base_url + 是否需要 key + 特点说明
- recommend_provider: 按自然语言需求 ("本地部署" / "便宜" / "最强推理") 推荐
- validate_config:   检查当前配置完整性 (缺 key / base_url 空 / model 名错)
- migrate_from_env:  扫环境变量 (OPENAI_API_KEY 等) 迁移到 config 文件
- setup_local_model: 接本地大模型 (ollama / vllm / llama.cpp / TGI), 生成配置 + 测连通 + 保存
- get_privacy:       查当前隐私级别 + 脱敏统计
- set_privacy:       设隐私级别 (off / redact / local_only)
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from huginn.config import HuginnConfig, ModelConfig
from huginn.models.registry import (
    _DOMESTIC_OPENAI_COMPATIBLE,
    _PROVIDER_DEFAULTS,
    _PROVIDER_KEY_ENV,
    create_langchain_model,
    resolve_provider_key,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolResult, ValidationResult

# Provider 特点说明, 给 LLM 做推荐时参考
_PROVIDER_NOTES: dict[str, str] = {
    "anthropic": "Claude 系列, 推理/代码强, 支持 extended thinking, 价格中高",
    "openai": "GPT 系列, 生态最全, function calling 稳定, 价格中高",
    "deepseek": "国产性价比之王, deepseek-chat 便宜, deepseek-reasoner 推理强",
    "google-genai": "Gemini 系列, 多模态强, 长上下文, 免费额度大方",
    "openrouter": "聚合平台, 一个 key 接所有模型, 适合多模型切换",
    "nvidia": "NVIDIA NIM, 企业级部署, Llama 等开源模型托管",
    "ollama": "本地方案首选, 一行命令起模型, 无需 key, 适合开发/隐私场景",
    "vllm": "本地高吞吐推理, 适合大模型生产部署, OpenAI 兼容 API",
    "local": "本地 OpenAI 兼容端点 (llama.cpp / TGI 等), 通用兜底",
    "siliconflow": "国产聚合平台, 模型多, 免费额度够测试",
    "moonshot": "Kimi, 长上下文 (200k+), 文档分析强",
    "zhipu": "智谱 GLM-4, 国产全能, glm-4-flash 免费",
    "baichuan": "百川, 中文场景强, 价格友好",
    "dashscope": "通义千问, 阿里云生态, qwen-max 推理强",
    "qianfan": "百度千帆, ERNIE 系列, 中文友好",
    "doubao": "字节豆包, 价格极低, 适合高频调用",
    "hunyuan": "腾讯混元, 多模态, 中文友好",
    "openai-compatible": "自定义 OpenAI 兼容端点, 灵活兜底",
    "default": "未配置, 退回环境变量",
}

# 本地模型方案 → 默认端口 / 走哪个 provider
_LOCAL_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "ollama": {
        "provider": "ollama",
        "default_port": 11434,
        "base_url_template": "http://{host}:{port}",
        "needs_key": False,
        "note": "Ollama 自带 /api/chat, langchain_ollama 直接对接",
    },
    "vllm": {
        "provider": "vllm",
        "default_port": 8000,
        "base_url_template": "http://{host}:{port}/v1",
        "needs_key": False,
        "note": "vLLM 起 OpenAI 兼容 server, 走 /v1/chat/completions",
    },
    "llama.cpp": {
        "provider": "local",
        "default_port": 8080,
        "base_url_template": "http://{host}:{port}/v1",
        "needs_key": False,
        "note": "llama.cpp server 模式, OpenAI 兼容, 走 /v1/chat/completions",
    },
    "text-generation-inference": {
        "provider": "local",
        "default_port": 8080,
        "base_url_template": "http://{host}:{port}/v1",
        "needs_key": False,
        "note": "HuggingFace TGI, OpenAI 兼容 /v1, 工具调用支持看模型",
    },
}


class ConfigWizardInput(BaseModel):
    # model_ 前缀字段和 Pydantic 默认 protected namespace 冲突, 关掉避免警告
    model_config = ConfigDict(protected_namespaces=())

    action: Literal[
        "list_providers",
        "recommend_provider",
        "validate_config",
        "migrate_from_env",
        "setup_local_model",
        "list_features",
        "toggle_feature",
        "get_privacy",
        "set_privacy",
    ] = Field(..., description="向导动作")

    # recommend_provider 用
    requirement: str | None = Field(
        default=None,
        description="用户需求描述, 如 '我要本地部署' / '我要便宜的' / '我要最强推理'",
    )

    # toggle_feature / list_features 用
    feature: str | None = Field(
        default=None,
        description="要开关的功能名, 仅 toggle_feature 用. 如 'speculator' / 'provenance'",
    )
    enabled: bool | None = Field(
        default=None,
        description="True 打开, False 关闭, 仅 toggle_feature 用",
    )

    # set_privacy 用
    level: str | None = Field(
        default=None,
        description="隐私级别, 仅 set_privacy 用. 可选 'off' / 'redact' / 'local_only'",
    )

    # setup_local_model 用
    model_type: Literal["ollama", "vllm", "llama.cpp", "text-generation-inference"] | None = Field(
        default=None, description="本地推理后端类型"
    )
    model_name: str | None = Field(default=None, description="模型名, 如 llama3:70b / Qwen2.5-72B-Instruct")
    port: int | None = Field(default=None, description="服务端口, 不传用默认值")
    host: str | None = Field(default="localhost", description="服务主机, 默认 localhost")
    alias: str | None = Field(default=None, description="模型别名, 不传用 model_name 推断")

    # 通用: 配置文件路径
    config_path: str | None = Field(
        default=None, description="配置文件路径, 不传用 HUGINN_CONFIG_FILE 或默认 huginn.toml"
    )


def _resolve_config_path(explicit: str | None = None) -> Path:
    """配置文件路径解析: 显式 > HUGINN_CONFIG_FILE > 工作目录/huginn.toml。"""
    if explicit:
        return Path(explicit)
    raw = os.environ.get("HUGINN_CONFIG_FILE")
    if raw:
        return Path(raw)
    return Path(os.environ.get("HUGINN_WORKSPACE", ".")) / "huginn.toml"


def _load_config(path: Path) -> HuginnConfig:
    """加载配置, 文件不存在就退回 from_env。"""
    if path.exists():
        try:
            return HuginnConfig.load(path)
        except Exception:
            return HuginnConfig.from_env()
    return HuginnConfig.from_env()


def _provider_full_info(provider: str) -> dict[str, Any]:
    """组装 provider 完整信息 (含特点说明)。"""
    domestic = _DOMESTIC_OPENAI_COMPATIBLE.get(provider, {})
    info = {
        "provider": provider,
        "default_base_url": domestic.get("base_url") if domestic else None,
        "default_model": _PROVIDER_DEFAULTS.get(provider),
        "needs_api_key": bool(_PROVIDER_KEY_ENV.get(provider)),
        "env_var": _PROVIDER_KEY_ENV.get(provider) or None,
        "note": _PROVIDER_NOTES.get(provider, ""),
    }
    # ollama / vllm / local 是无 key 本地方案, 单独标
    if provider == "ollama":
        info["default_base_url"] = info["default_base_url"] or "http://localhost:11434"
        info["needs_api_key"] = False
    if provider in ("vllm", "local"):
        info["needs_api_key"] = False
    return info


class ConfigWizardTool(HuginnTool):
    """配置向导: 选 provider / 校验 / 迁移 / 接本地模型。"""

    name = "config_wizard_tool"
    category = "meta"
    description = (
        "Configuration wizard for LLM providers. Actions: "
        "list_providers (list 19 supported providers with defaults), "
        "recommend_provider (recommend based on natural-language requirement), "
        "validate_config (check current config completeness), "
        "migrate_from_env (scan env vars and migrate API keys to config file), "
        "setup_local_model (configure ollama/vllm/llama.cpp/TGI local deployment), "
        "list_features / toggle_feature (runtime feature flags), "
        "get_privacy / set_privacy (privacy level: off / redact / local_only)."
    )
    input_schema = ConfigWizardInput
    read_only = False  # migrate/setup 会写配置文件

    async def validate_input(
        self, args: ConfigWizardInput, context: Any = None
    ) -> ValidationResult:
        if args.action == "recommend_provider" and not args.requirement:
            return ValidationResult(
                result=False, message="recommend_provider 需要 requirement 参数",
            )
        if args.action == "setup_local_model":
            if not args.model_type:
                return ValidationResult(
                    result=False, message="setup_local_model 需要 model_type",
                )
            if not args.model_name:
                return ValidationResult(
                    result=False, message="setup_local_model 需要 model_name",
                )
        if args.action == "toggle_feature":
            if not args.feature:
                return ValidationResult(
                    result=False, message="toggle_feature 需要 feature 参数",
                )
            if args.enabled is None:
                return ValidationResult(
                    result=False, message="toggle_feature 需要 enabled 参数",
                )
        if args.action == "set_privacy":
            if not args.level:
                return ValidationResult(
                    result=False,
                    message="set_privacy 需要 level 参数 (off / redact / local_only)",
                )
        return ValidationResult(result=True)

    async def call(self, args: dict[str, Any], context: Any = None) -> ToolResult:
        try:
            inp = ConfigWizardInput(**args)
            if inp.action == "list_providers":
                return self._list_providers()
            if inp.action == "recommend_provider":
                return self._recommend_provider(inp.requirement or "")
            if inp.action == "validate_config":
                return self._validate_config(inp.config_path)
            if inp.action == "migrate_from_env":
                return self._migrate_from_env(inp.config_path)
            if inp.action == "setup_local_model":
                return await self._setup_local_model(inp)
            if inp.action == "list_features":
                return self._list_features()
            if inp.action == "toggle_feature":
                return self._toggle_feature(inp.feature or "", bool(inp.enabled))
            if inp.action == "get_privacy":
                return self._get_privacy()
            if inp.action == "set_privacy":
                return self._set_privacy(inp.level or "")
            return ToolResult(
                data=None, success=False,
                error=f"未知 action: {inp.action}",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=str(e))

    # ── list_providers ──────────────────────────────────────────

    def _list_providers(self) -> ToolResult:
        order = [
            "anthropic", "openai", "deepseek", "google-genai", "openrouter",
            "nvidia", "ollama", "vllm", "local", "siliconflow",
            "moonshot", "zhipu", "baichuan", "dashscope", "qianfan",
            "doubao", "hunyuan", "openai-compatible", "default",
        ]
        providers = [_provider_full_info(p) for p in order]
        return ToolResult(
            data={
                "providers": providers,
                "count": len(providers),
                "message": (
                    "共 19 个 provider. 本地方案 (ollama/vllm/local) 无需 key, "
                    "云端方案需要对应环境变量的 API key. "
                    "用 recommend_provider 按需求推荐, setup_local_model 接本地模型."
                ),
            },
            success=True,
        )

    # ── recommend_provider ──────────────────────────────────────

    def _recommend_provider(self, requirement: str) -> ToolResult:
        """按关键词匹配需求, 返回推荐 + 配置示例。"""
        req = requirement.lower()
        # 关键词 → provider 候选, 按匹配度排序
        rules = [
            (["本地", "local", "离线", "offline", "自己机器", "无 key", "免费", "隐私"], "ollama"),
            (["vllm", "高吞吐", "生产部署", "gpu 集群"], "vllm"),
            (["llama.cpp", "gguf", "cpu", "单机"], "local"),
            (["便宜", "性价比", "低价", "省钱", "免费额度"], "deepseek"),
            (["最强推理", "推理强", "reasoning", "o1", "o3", "deepseek-reasoner"], "deepseek"),
            (["claude", "anthropic", "代码", "extended thinking"], "anthropic"),
            (["gpt", "openai", "生态全", "function calling"], "openai"),
            (["gemini", "google", "多模态", "长上下文", "免费"], "google-genai"),
            (["聚合", "一个 key", "多模型", "openrouter"], "openrouter"),
            (["国产", "中文", "国内"], "siliconflow"),
            (["kimi", "长文档", "200k"], "moonshot"),
            (["glm", "智谱"], "zhipu"),
            (["通义", "qwen", "阿里"], "dashscope"),
            (["ernie", "百度", "千帆"], "qianfan"),
            (["豆包", "字节", "极低价"], "doubao"),
            (["混元", "腾讯"], "hunyuan"),
            (["百川", "baichuan"], "baichuan"),
        ]

        hits: list[tuple[str, str]] = []
        for keywords, provider in rules:
            for kw in keywords:
                if kw in req:
                    hits.append((kw, provider))
                    break

        if not hits:
            # 没匹配上, 给个默认推荐
            recommended = "deepseek"
            reason = f"未识别到明确需求 '{requirement}', 默认推荐 deepseek (性价比高, 国内可用)"
        else:
            # 取第一个命中, 优先级按 rules 顺序
            recommended = hits[0][1]
            matched_kw = hits[0][0]
            reason = f"识别到关键词 '{matched_kw}', 推荐 {recommended}"

        info = _provider_full_info(recommended)
        # 拼一个配置示例, 让用户照着填
        example = {
            "alias": recommended.replace("-", "_"),
            "provider": recommended,
            "model": info["default_model"] or "<模型名>",
            "base_url": info["default_base_url"] or "<自定义>",
            "api_key": f"<env:{info['env_var']}>" if info["needs_api_key"] else None,
        }
        return ToolResult(
            data={
                "recommended_provider": recommended,
                "reason": reason,
                "provider_info": info,
                "config_example": example,
                "other_candidates": [p for _, p in hits[1:3]],
                "message": (
                    f"推荐 {recommended}. 配置示例: {example}. "
                    f"用 setup_local_model 接本地模型, 或直接在前端设置页添加."
                ),
            },
            success=True,
        )

    # ── validate_config ─────────────────────────────────────────

    def _validate_config(self, config_path: str | None) -> ToolResult:
        """检查当前配置: 缺 key / base_url 空 / model 名空 / 活跃 model 是否存在。"""
        path = _resolve_config_path(config_path)
        cfg = _load_config(path)
        issues: list[dict[str, Any]] = []

        if not cfg.models:
            issues.append({
                "severity": "warn",
                "field": "models",
                "message": "模型池为空, agent 无法启动. 用 setup_local_model 或前端添加模型.",
            })
        else:
            for m in cfg.models:
                if not m.alias:
                    issues.append({"severity": "error", "field": f"models[].alias", "message": "alias 为空"})
                if not m.model and m.provider not in ("default",):
                    issues.append({
                        "severity": "warn",
                        "field": f"models[{m.alias}].model",
                        "message": f"model 名为空, provider={m.provider}",
                    })
                # 本地 provider 不需要 key, 跳过
                if m.provider in ("ollama", "vllm", "local"):
                    continue
                resolved = resolve_provider_key(m.provider, m.api_key)  # type: ignore[arg-type]
                if not resolved:
                    env_var = _PROVIDER_KEY_ENV.get(m.provider, "?")
                    issues.append({
                        "severity": "error",
                        "field": f"models[{m.alias}].api_key",
                        "message": f"未配置 api_key (env:{env_var} 也没设)",
                    })
                # 检查 base_url 格式
                if m.base_url and not re.match(r"^https?://", m.base_url):
                    issues.append({
                        "severity": "warn",
                        "field": f"models[{m.alias}].base_url",
                        "message": f"base_url 不是 http(s):// 开头: {m.base_url}",
                    })

        # 检查活跃 model 是否存在
        lead = next((a for a in cfg.agents if a.id == "lead"), None)
        if lead and lead.model_alias:
            if not any(m.alias == lead.model_alias and m.enabled for m in cfg.models):
                issues.append({
                    "severity": "error",
                    "field": "agents[lead].model_alias",
                    "message": f"活跃 model '{lead.model_alias}' 不存在或未启用",
                })
        elif cfg.models:
            issues.append({
                "severity": "warn",
                "field": "agents[lead].model_alias",
                "message": "未设置活跃 model, 会用第一个 enabled 的 model",
            })

        ok = not any(i["severity"] == "error" for i in issues)
        return ToolResult(
            data={
                "valid": ok,
                "issues": issues,
                "issue_count": len(issues),
                "config_path": str(path),
                "config_exists": path.exists(),
                "model_count": len(cfg.models),
                "message": (
                    "配置有效, 可以正常使用" if ok
                    else f"发现 {sum(1 for i in issues if i['severity']=='error')} 个错误, 需修复"
                ),
            },
            success=True,
        )

    # ── migrate_from_env ────────────────────────────────────────

    def _migrate_from_env(self, config_path: str | None) -> ToolResult:
        """扫环境变量, 把找到的 API key 迁移到 config 文件的 models 列表。"""
        path = _resolve_config_path(config_path)
        cfg = _load_config(path)

        # 扫所有 provider 对应的环境变量
        found: list[dict[str, Any]] = []
        existing_aliases = {m.alias for m in cfg.models}
        existing_keys = {m.provider for m in cfg.models if m.api_key}

        for provider, env_var in _PROVIDER_KEY_ENV.items():
            if not env_var:
                continue
            val = os.environ.get(env_var)
            if not val:
                continue
            # 已有同 provider 的 key 就跳过, 不重复添加
            if provider in existing_keys:
                found.append({
                    "provider": provider,
                    "env_var": env_var,
                    "status": "skipped",
                    "reason": "config 里已有该 provider 的 key",
                })
                continue
            # 用 env: 引用, 不把明文 key 写进文件
            alias = provider.replace("-", "_")
            # alias 去重
            base_alias = alias
            n = 2
            while alias in existing_aliases:
                alias = f"{base_alias}_{n}"
                n += 1
            existing_aliases.add(alias)
            cfg.models.append(ModelConfig(
                alias=alias,
                provider=provider,  # type: ignore[arg-type]
                model=_PROVIDER_DEFAULTS.get(provider),
                api_key=f"env:{env_var}",
            ))
            found.append({
                "provider": provider,
                "env_var": env_var,
                "status": "migrated",
                "alias": alias,
            })

        if not any(f["status"] == "migrated" for f in found):
            return ToolResult(
                data={
                    "migrated_count": 0,
                    "found": found,
                    "message": "未在环境变量里找到新的 API key (或已全部迁移过)",
                },
                success=True,
            )

        try:
            cfg.save(path, format="toml")
        except Exception as e:
            return ToolResult(
                data=None, success=False,
                error=f"保存配置失败: {e}",
            )

        migrated = [f for f in found if f["status"] == "migrated"]
        return ToolResult(
            data={
                "migrated_count": len(migrated),
                "found": found,
                "config_path": str(path),
                "message": (
                    f"从环境变量迁移了 {len(migrated)} 个 API key 到 {path}. "
                    f"用 env: 引用, 明文 key 不会写进文件. "
                    f"重启 agent 或调 POST /config/active-model 切换活跃 model."
                ),
            },
            success=True,
        )

    # ── setup_local_model ───────────────────────────────────────

    async def _setup_local_model(self, inp: ConfigWizardInput) -> ToolResult:
        """接本地大模型: 生成 ModelConfig + 测连通 + 保存。"""
        model_type = inp.model_type
        if not model_type:
            return ToolResult(data=None, success=False, error="model_type 必填")
        preset = _LOCAL_MODEL_PRESETS.get(model_type)
        if preset is None:
            return ToolResult(
                data=None, success=False,
                error=f"不支持的 model_type: {model_type}. 支持: {list(_LOCAL_MODEL_PRESETS)}",
            )

        host = inp.host or "localhost"
        port = inp.port or preset["default_port"]
        base_url = preset["base_url_template"].format(host=host, port=port)
        provider = preset["provider"]
        model_name = inp.model_name or ""
        # alias 不传就用 model_name 推断 (去特殊字符)
        alias = inp.alias or re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:32]

        # 构造 ModelConfig
        config = ModelConfig(
            alias=alias,
            provider=provider,  # type: ignore[arg-type]
            model=model_name,
            api_key=None,  # 本地模型不需要 key
            base_url=base_url,
            temperature=0.7,
            enabled=True,
        )

        # 测连通性 (10s 超时, 跟路由层一致)
        test_result = await self._test_local_connectivity(config)

        # 不管测试成功失败都写入配置, 用户可以之后修
        path = _resolve_config_path(inp.config_path)
        cfg = _load_config(path)
        # alias 已存在就覆盖, 否则追加
        idx = next((i for i, m in enumerate(cfg.models) if m.alias == alias), None)
        if idx is not None:
            cfg.models[idx] = config
        else:
            cfg.models.append(config)

        # 如果没有活跃 model, 把这个设为活跃
        lead = next((a for a in cfg.agents if a.id == "lead"), None)
        set_active = False
        if lead is None or not lead.model_alias:
            from huginn.config import AgentProfileConfig
            if lead is None:
                cfg.agents.append(AgentProfileConfig(
                    id="lead", name="Lead", model_alias=alias
                ))
            else:
                lead.model_alias = alias
            set_active = True

        try:
            cfg.save(path, format="toml")
        except Exception as e:
            return ToolResult(
                data=None, success=False,
                error=f"保存配置失败: {e}",
            )

        # 给点后续建议
        recommendations: list[str] = []
        if not test_result["success"]:
            recommendations.append(
                f"连通性测试失败: {test_result['error']}. "
                f"确认 {model_type} 服务已启动 (检查 {base_url}), 模型已加载."
            )
        if model_type == "ollama":
            recommendations.append("用 `ollama list` 查看已安装的模型, `ollama pull <model>` 下载新模型.")
        if model_type == "vllm":
            recommendations.append("vLLM 启动示例: `python -m vllm.entrypoints.openai.api_server --model <path> --port 8000`")
        if model_type == "llama.cpp":
            recommendations.append("llama.cpp 启动示例: `./server -m model.gguf --port 8080`")
        if set_active:
            recommendations.append(f"已将 {alias} 设为活跃 model.")

        return ToolResult(
            data={
                "config": {
                    "alias": config.alias,
                    "provider": config.provider,
                    "model": config.model,
                    "base_url": config.base_url,
                    "api_key": None,
                },
                "test_result": test_result,
                "config_path": str(path),
                "set_as_active": set_active,
                "recommendations": recommendations,
                "message": (
                    f"已配置本地 {model_type} 模型 {model_name} (alias={alias}). "
                    f"连通性: {'✅ 通' if test_result['success'] else '❌ 未通, 见 recommendations'}. "
                    f"配置已写入 {path}."
                ),
            },
            success=True,
        )

    async def _test_local_connectivity(self, config: ModelConfig) -> dict[str, Any]:
        """跑一次轻量调用验证本地模型可达。"""
        import time as _time
        start = _time.perf_counter()
        try:
            client = create_langchain_model(
                provider=config.provider,  # type: ignore[arg-type]
                model_name=config.model,
                api_key="not-needed",
                base_url=config.base_url,
                temperature=0.0,
                max_tokens=16,
            )
            response = await asyncio.wait_for(
                asyncio.to_thread(client.invoke, "Hello"),
                timeout=10.0,
            )
            latency_ms = int((_time.perf_counter() - start) * 1000)
            text = ""
            if hasattr(response, "content"):
                text = str(response.content)
            return {
                "success": True,
                "latency_ms": latency_ms,
                "error": None,
                "model_response": text[:200],
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "latency_ms": int((_time.perf_counter() - start) * 1000),
                "error": f"请求超时 (10s), 确认服务已启动: {config.base_url}",
                "model_response": "",
            }
        except Exception as e:
            return {
                "success": False,
                "latency_ms": int((_time.perf_counter() - start) * 1000),
                "error": str(e),
                "model_response": "",
            }

    # ── list_features / toggle_feature ──────────────────────────

    def _list_features(self) -> ToolResult:
        """列出所有 feature flag 的当前状态."""
        from huginn.feature_flags import FeatureFlags

        flags = FeatureFlags.shared().list_flags()
        return ToolResult(
            data={
                "features": flags,
                "count": len(flags),
                "message": (
                    f"共 {len(flags)} 个 feature flag, 默认全开. "
                    f"用 toggle_feature 开关单个, 运行时改动不写盘."
                ),
            },
            success=True,
        )

    def _toggle_feature(self, feature: str, enabled: bool) -> ToolResult:
        """运行时开关某个 feature flag."""
        from huginn.feature_flags import FeatureFlags

        ff = FeatureFlags.shared()
        # 未知 feature 给个明确报错, 别让用户以为开关成功了
        known = {f["name"] for f in ff.list_flags()}
        if feature not in known:
            return ToolResult(
                data=None,
                success=False,
                error=f"未知 feature: {feature}. 可用: {sorted(known)}",
            )
        ff.toggle(feature, enabled)
        new_state = ff.is_enabled(feature)
        return ToolResult(
            data={
                "feature": feature,
                "enabled": new_state,
                "requested": enabled,
                "applied": new_state == enabled,
                "message": (
                    f"{feature} 已{'打开' if new_state else '关闭'}. "
                    f"运行时改动不写盘, 要持久化用 persist_to_config."
                ),
            },
            success=True,
        )

    # ── get_privacy / set_privacy ─────────────────────────────

    def _get_privacy(self) -> ToolResult:
        """返回当前隐私级别 + 已脱敏统计."""
        from huginn.privacy_guard import PrivacyGuard

        g = PrivacyGuard.shared()
        info = g.summary()
        return ToolResult(
            data={
                **info,
                "available_levels": dict(g.LEVELS),
                "message": (
                    f"当前隐私级别: {info['level']}. "
                    f"已脱敏 {info['redact_count']} 次. "
                    f"用 set_privacy 切换级别 (off / redact / local_only)."
                ),
            },
            success=True,
        )

    def _set_privacy(self, level: str) -> ToolResult:
        """切换隐私级别. 三个级别互斥."""
        from huginn.privacy_guard import PrivacyGuard

        g = PrivacyGuard.shared()
        if level not in g.LEVELS:
            return ToolResult(
                data=None,
                success=False,
                error=f"未知隐私级别: {level}. 可选: {list(g.LEVELS)}",
            )
        g.set_level(level)
        return ToolResult(
            data={
                "level": level,
                "level_description": g.LEVELS[level],
                "send_to_cloud": g.should_send_to_cloud(),
                "message": (
                    f"隐私级别已设为 '{level}'. "
                    f"{g.LEVELS[level]}. 运行时改动不写盘."
                ),
            },
            success=True,
        )
