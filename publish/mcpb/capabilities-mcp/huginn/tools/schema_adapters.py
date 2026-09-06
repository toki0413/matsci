"""多 provider 工具 schema 适配器。

借鉴 AstrBot 的 ToolSet (astrbot/core/agent/tool.py) —— 它在同一个工具集上提供
openai_schema()/anthropic_schema()/google_schema() 转换, 让 agent 切换底层 LLM
时不用重写工具定义。这里把同样的思路抽成纯函数, 输入是 registry 已经生成好的
OpenAI 格式 schema, 输出是对应 provider 需要的格式。
"""

from __future__ import annotations

import copy
from collections.abc import Callable

# provider 名称类型, 方便标注
ProviderType = str  # "openai" | "anthropic" | "google" | "mistral"


def to_openai_schema(schemas: list[dict]) -> list[dict]:
    """OpenAI function calling 格式 (当前默认)。

    registry 已经按 OpenAI 格式生成, 这里只做一层清洗 —— 把 destructive /
    read_only / metadata 这些内部字段剥掉, 只留 API 真正需要的部分。
    """
    result = []
    for s in schemas:
        result.append({
            "type": "function",
            "function": {
                "name": s["function"]["name"],
                "description": s["function"]["description"],
                "parameters": s["function"]["parameters"],
            },
        })
    return result


def to_anthropic_schema(schemas: list[dict]) -> list[dict]:
    """Anthropic tool use 格式。

    Anthropic 用更扁的结构: {name, description, input_schema}, 没有 type/function
    这层嵌套。
    """
    result = []
    for s in schemas:
        result.append({
            "name": s["function"]["name"],
            "description": s["function"]["description"],
            "input_schema": s["function"]["parameters"],
        })
    return result


def to_google_schema(schemas: list[dict]) -> list[dict]:
    """Google/Gemini function declaration 格式。

    Google 用 {name, description, parameters}, 但不支持完整 JSON Schema ——
    $defs/$ref 得拍平, anyOf 要换成 oneOf。详见 _simplify_for_google。
    """
    result = []
    for s in schemas:
        params = _simplify_for_google(s["function"]["parameters"])
        result.append({
            "name": s["function"]["name"],
            "description": s["function"]["description"],
            "parameters": params,
        })
    return result


def to_mistral_schema(schemas: list[dict]) -> list[dict]:
    """Mistral 工具格式, 和 OpenAI 完全一样, 直接复用。"""
    return to_openai_schema(schemas)


def _simplify_for_google(schema: dict) -> dict:
    """去掉 Google 不支持的 JSON Schema 特性 ($defs / $ref / anyOf)。

    这里只做最简单的处理: $defs 和 $ref 直接删掉 (Google 期望内联定义),
    anyOf 换成 oneOf (Google 偏好 oneOf)。嵌套的 properties / items 递归清理。
    """
    s = copy.deepcopy(schema)
    # Google 要求内联, 把 $defs 整个删掉
    s.pop("$defs", None)
    # 简单情况: 直接移除 $ref, 不做真正的引用解析
    s.pop("$ref", None)
    # anyOf → oneOf, Google 的 schema 解析器更认 oneOf
    if "anyOf" in s:
        s["oneOf"] = s.pop("anyOf")
    # 递归清理嵌套属性
    if "properties" in s:
        for key, val in s["properties"].items():
            if isinstance(val, dict):
                s["properties"][key] = _simplify_for_google(val)
    if "items" in s and isinstance(s["items"], dict):
        s["items"] = _simplify_for_google(s["items"])
    return s


# provider → 适配函数的注册表
SCHEMA_ADAPTERS: dict[ProviderType, Callable[..., list[dict]]] = {
    "openai": to_openai_schema,
    "anthropic": to_anthropic_schema,
    "google": to_google_schema,
    "gemini": to_google_schema,  # gemini 是 google 的别名
    "mistral": to_mistral_schema,
}


def adapt_schemas(schemas: list[dict], provider: ProviderType) -> list[dict]:
    """把工具 schema 转成指定 LLM provider 需要的格式。

    未知 provider 回落到 OpenAI 格式 —— OpenAI 的 function calling 是事实上的
    基线, 大多数 provider 都兼容。
    """
    adapter = SCHEMA_ADAPTERS.get(provider, to_openai_schema)
    return adapter(schemas)
