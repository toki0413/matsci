"""多 provider schema 适配器的单元测试。

覆盖:
  - to_openai_schema / to_anthropic_schema / to_google_schema / to_mistral_schema
    的结构正确性 + 内部字段剥离
  - adapt_schemas 的 provider 分发 + 未知 provider 回落
  - ToolRegistry.get_schemas_for_provider 按 provider 返回正确格式
  - active=False 的工具在 get_all_schemas / get_schemas_for_provider 里都被过滤

不依赖完整的工具注册流程, 用 mock schema + mock tool 隔离测试。
"""

from __future__ import annotations

import pytest

from huginn.tools.defaults import ToolMetadata
from huginn.tools.schema_adapters import (
    SCHEMA_ADAPTERS,
    _simplify_for_google,
    adapt_schemas,
    to_anthropic_schema,
    to_google_schema,
    to_mistral_schema,
    to_openai_schema,
)


# ── mock 数据: 模拟 registry.get_all_schemas() 的输出 ───────────────


def _make_mock_schemas() -> list[dict]:
    """造两份 OpenAI 格式的 schema, 带上 registry 会附加的内部字段。

    第一份带 $defs / $ref / 嵌套 anyOf, 用来验证 google 简化逻辑;
    第二份是干净的最简 schema, 用来验证基本透传。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_literature",
                "description": "Search materials science literature",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "search keywords"},
                        "year": {
                            # 嵌套 anyOf, google 要转成 oneOf
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["query"],
                    # Google 不支持 $defs / $ref
                    "$defs": {
                        "Filter": {
                            "type": "object",
                            "properties": {"field": {"type": "string"}},
                        }
                    },
                    "$ref": "#/$defs/Filter",
                },
            },
            "destructive": False,
            "read_only": True,
            "metadata": ToolMetadata(
                is_read_only=True, requires_confirmation=False
            ),
        },
        {
            "type": "function",
            "function": {
                "name": "run_vasp",
                "description": "Submit a VASP calculation job",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "encut": {"type": "number", "description": "cutoff energy"},
                    },
                    "required": ["encut"],
                },
            },
            "destructive": False,
            "read_only": False,
            "metadata": ToolMetadata(
                is_read_only=False, requires_confirmation=True
            ),
        },
    ]


# ════════════════════════════════════════════════════════════════════
# to_openai_schema
# ════════════════════════════════════════════════════════════════════


def test_openai_schema_structure():
    """每个元素都是 {type: function, function: {name, description, parameters}}."""
    schemas = _make_mock_schemas()
    result = to_openai_schema(schemas)

    assert len(result) == 2
    for item in result:
        assert item["type"] == "function"
        assert "function" in item
        func = item["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func


def test_openai_schema_strips_internal_fields():
    """destructive / read_only / metadata 这些内部字段要被剥掉."""
    schemas = _make_mock_schemas()
    result = to_openai_schema(schemas)

    for item in result:
        assert "destructive" not in item
        assert "read_only" not in item
        assert "metadata" not in item
        # function 子字典里也不应该混进这些字段
        assert "destructive" not in item["function"]
        assert "metadata" not in item["function"]


def test_openai_schema_preserves_name_and_params():
    """name 和 parameters 要原样透传, 不能丢."""
    schemas = _make_mock_schemas()
    result = to_openai_schema(schemas)

    names = [item["function"]["name"] for item in result]
    assert names == ["search_literature", "run_vasp"]
    # parameters 内容要和原始一致
    assert result[0]["function"]["parameters"]["type"] == "object"
    assert "query" in result[0]["function"]["parameters"]["properties"]


# ════════════════════════════════════════════════════════════════════
# to_anthropic_schema
# ════════════════════════════════════════════════════════════════════


def test_anthropic_schema_flat_structure():
    """Anthropic 用扁平结构 {name, description, input_schema}."""
    schemas = _make_mock_schemas()
    result = to_anthropic_schema(schemas)

    assert len(result) == 2
    for item in result:
        assert "name" in item
        assert "description" in item
        assert "input_schema" in item
        # 不能有 type/function 这层嵌套
        assert "type" not in item
        assert "function" not in item


def test_anthropic_schema_input_schema_is_parameters():
    """input_schema 的值应该等于原始 parameters."""
    schemas = _make_mock_schemas()
    result = to_anthropic_schema(schemas)

    assert result[0]["name"] == "search_literature"
    assert result[0]["input_schema"]["type"] == "object"
    assert "query" in result[0]["input_schema"]["properties"]
    assert result[1]["name"] == "run_vasp"
    assert "encut" in result[1]["input_schema"]["properties"]


def test_anthropic_schema_strips_internal_fields():
    """Anthropic 格式同样不能带 destructive / read_only / metadata."""
    schemas = _make_mock_schemas()
    result = to_anthropic_schema(schemas)

    for item in result:
        assert "destructive" not in item
        assert "read_only" not in item
        assert "metadata" not in item


# ════════════════════════════════════════════════════════════════════
# to_google_schema
# ════════════════════════════════════════════════════════════════════


def test_google_schema_structure():
    """Google 用 {name, description, parameters}."""
    schemas = _make_mock_schemas()
    result = to_google_schema(schemas)

    assert len(result) == 2
    for item in result:
        assert "name" in item
        assert "description" in item
        assert "parameters" in item
        assert "type" not in item
        assert "function" not in item


def test_google_schema_strips_defs_and_ref():
    """$defs 和 $ref 要被删掉."""
    schemas = _make_mock_schemas()
    result = to_google_schema(schemas)

    params = result[0]["parameters"]
    assert "$defs" not in params
    assert "$ref" not in params


def test_google_schema_converts_anyof_to_oneof():
    """嵌套的 anyOf 要递归转成 oneOf."""
    schemas = _make_mock_schemas()
    result = to_google_schema(schemas)

    year_prop = result[0]["parameters"]["properties"]["year"]
    assert "anyOf" not in year_prop
    assert "oneOf" in year_prop
    # 内容要保留
    assert len(year_prop["oneOf"]) == 2


def test_google_schema_preserves_plain_properties():
    """没有 anyOf/$defs 的普通属性要原样保留."""
    schemas = _make_mock_schemas()
    result = to_google_schema(schemas)

    query_prop = result[0]["parameters"]["properties"]["query"]
    assert query_prop["type"] == "string"
    # run_vasp 的 schema 没有 $defs/anyOf, 应该原样透传
    assert "encut" in result[1]["parameters"]["properties"]


def test_simplify_for_google_recursive_items():
    """items 里的 anyOf 也要被递归清理."""
    schema = {
        "type": "array",
        "items": {
            "anyOf": [{"type": "string"}, {"type": "number"}],
        },
    }
    result = _simplify_for_google(schema)
    assert "anyOf" not in result["items"]
    assert "oneOf" in result["items"]


# ════════════════════════════════════════════════════════════════════
# to_mistral_schema
# ════════════════════════════════════════════════════════════════════


def test_mistral_schema_same_as_openai():
    """Mistral 格式和 OpenAI 完全一致."""
    schemas = _make_mock_schemas()
    mistral_result = to_mistral_schema(schemas)
    openai_result = to_openai_schema(schemas)

    assert mistral_result == openai_result
    # 确认结构正确
    assert mistral_result[0]["type"] == "function"
    assert "function" in mistral_result[0]


# ════════════════════════════════════════════════════════════════════
# adapt_schemas
# ════════════════════════════════════════════════════════════════════


def test_adapt_schemas_dispatches_openai():
    """provider='openai' 走 OpenAI 适配."""
    schemas = _make_mock_schemas()
    result = adapt_schemas(schemas, "openai")
    assert result == to_openai_schema(schemas)
    assert result[0]["type"] == "function"


def test_adapt_schemas_dispatches_anthropic():
    """provider='anthropic' 走 Anthropic 适配."""
    schemas = _make_mock_schemas()
    result = adapt_schemas(schemas, "anthropic")
    assert "input_schema" in result[0]
    assert "function" not in result[0]


def test_adapt_schemas_dispatches_google():
    """provider='google' 走 Google 适配."""
    schemas = _make_mock_schemas()
    result = adapt_schemas(schemas, "google")
    assert "parameters" in result[0]
    assert "$defs" not in result[0]["parameters"]


def test_adapt_schemas_dispatches_gemini_alias():
    """provider='gemini' 是 google 的别名, 走 Google 适配."""
    schemas = _make_mock_schemas()
    result = adapt_schemas(schemas, "gemini")
    assert result == to_google_schema(schemas)


def test_adapt_schemas_dispatches_mistral():
    """provider='mistral' 走 Mistral 适配 (同 OpenAI)."""
    schemas = _make_mock_schemas()
    result = adapt_schemas(schemas, "mistral")
    assert result == to_openai_schema(schemas)


def test_adapt_schemas_unknown_provider_falls_back_to_openai():
    """未知 provider 回落到 OpenAI 格式."""
    schemas = _make_mock_schemas()
    result = adapt_schemas(schemas, "some_unknown_provider")
    assert result == to_openai_schema(schemas)
    assert result[0]["type"] == "function"


def test_adapt_schemas_empty_list():
    """空列表输入返回空列表."""
    assert adapt_schemas([], "openai") == []
    assert adapt_schemas([], "anthropic") == []


def test_schema_adapters_registry_keys():
    """SCHEMA_ADAPTERS 注册表包含全部已知 provider."""
    expected = {"openai", "anthropic", "google", "gemini", "mistral"}
    assert expected.issubset(SCHEMA_ADAPTERS.keys())


# ════════════════════════════════════════════════════════════════════
# ToolRegistry 集成测试
# ════════════════════════════════════════════════════════════════════


class _MockTool:
    """轻量 mock 工具, 只暴露 get_all_schemas 需要的属性。

    不继承 HuginnTool 是为了避开 ABC + Generic 的实例化开销, 测试只关心
    schema 序列化路径。
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        read_only: bool = False,
        destructive: bool = False,
        active: bool = True,
        params: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.read_only = read_only
        self.destructive = destructive
        self.active = active
        self._params = params or {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
        }

    @property
    def input_json_schema(self) -> dict | None:
        return self._params


@pytest.fixture
def mock_registry(monkeypatch):
    """用 mock 工具替换 ToolRegistry 的类级状态, 测试后自动恢复。

    放三个工具: 两个 active, 一个 active=False (验证过滤)。
    """
    from huginn.tools.registry import ToolRegistry

    tools = {
        "search_literature": _MockTool(
            "search_literature",
            "Search literature",
            read_only=True,
            active=True,
            params={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "year": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "$defs": {"Filter": {"type": "object"}},
                "$ref": "#/$defs/Filter",
                "required": ["query"],
            },
        ),
        "run_vasp": _MockTool(
            "run_vasp",
            "Run VASP",
            read_only=False,
            active=True,
        ),
        "disabled_tool": _MockTool(
            "disabled_tool",
            "Should be hidden from LLM",
            active=False,
        ),
    }
    monkeypatch.setattr(ToolRegistry, "_tools", tools)
    monkeypatch.setattr(ToolRegistry, "_schemas_cache", None)
    return ToolRegistry


def test_get_schemas_for_provider_openai(mock_registry):
    """get_schemas_for_provider('openai') 返回 OpenAI 格式, 且过滤 disabled."""
    result = mock_registry.get_schemas_for_provider("openai")

    names = [s["function"]["name"] for s in result]
    assert "search_literature" in names
    assert "run_vasp" in names
    assert "disabled_tool" not in names
    # OpenAI 格式校验
    for s in result:
        assert s["type"] == "function"
        assert "function" in s


def test_get_schemas_for_provider_anthropic(mock_registry):
    """get_schemas_for_provider('anthropic') 返回扁平 Anthropic 格式."""
    result = mock_registry.get_schemas_for_provider("anthropic")

    names = [s["name"] for s in result]
    assert "search_literature" in names
    assert "run_vasp" in names
    assert "disabled_tool" not in names
    for s in result:
        assert "input_schema" in s
        assert "function" not in s


def test_get_schemas_for_provider_google(mock_registry):
    """get_schemas_for_provider('google') 返回 Google 格式, $defs/anyOf 被清理."""
    result = mock_registry.get_schemas_for_provider("google")

    names = [s["name"] for s in result]
    assert "disabled_tool" not in names
    # search_literature 的 parameters 要被简化
    search = next(s for s in result if s["name"] == "search_literature")
    assert "$defs" not in search["parameters"]
    assert "$ref" not in search["parameters"]
    year_prop = search["parameters"]["properties"]["year"]
    assert "anyOf" not in year_prop
    assert "oneOf" in year_prop


def test_get_schemas_for_provider_default_is_openai(mock_registry):
    """不传 provider 时默认走 OpenAI."""
    result = mock_registry.get_schemas_for_provider()
    assert result[0]["type"] == "function"
    assert "function" in result[0]


def test_get_schemas_for_provider_mistral(mock_registry):
    """get_schemas_for_provider('mistral') 和 openai 一致."""
    mistral_result = mock_registry.get_schemas_for_provider("mistral")
    openai_result = mock_registry.get_schemas_for_provider("openai")
    assert mistral_result == openai_result


# ════════════════════════════════════════════════════════════════════
# active 工具过滤
# ════════════════════════════════════════════════════════════════════


def test_get_all_schemas_filters_inactive(mock_registry):
    """get_all_schemas() 不返回 active=False 的工具."""
    schemas = mock_registry.get_all_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "disabled_tool" not in names
    assert "search_literature" in names
    assert "run_vasp" in names


def test_get_all_schemas_still_has_metadata(mock_registry):
    """get_all_schemas() 仍然保留 destructive/read_only/metadata 内部字段."""
    schemas = mock_registry.get_all_schemas()
    for s in schemas:
        assert "destructive" in s
        assert "read_only" in s
        assert "metadata" in s
        assert isinstance(s["metadata"], ToolMetadata)


def test_get_schemas_for_provider_filters_inactive(mock_registry):
    """get_schemas_for_provider 在所有 provider 下都过滤 disabled_tool."""
    for provider in ("openai", "anthropic", "google", "mistral"):
        result = mock_registry.get_schemas_for_provider(provider)
        names = [s.get("name") or s.get("function", {}).get("name") for s in result]
        assert "disabled_tool" not in names, f"provider={provider} 泄漏了 disabled_tool"


def test_inactive_tool_still_callable_via_get(mock_registry):
    """active=False 的工具虽然对 LLM 不可见, 但 ToolRegistry.get() 仍能拿到."""
    tool = mock_registry.get("disabled_tool")
    assert tool is not None
    assert tool.active is False


def test_get_all_schemas_cache_excludes_inactive(mock_registry):
    """缓存生效后, active=False 的工具依然不会出现."""
    # 第一次调用填充缓存
    first = mock_registry.get_all_schemas()
    # 第二次走缓存
    second = mock_registry.get_all_schemas()
    assert first is second  # 同一个缓存对象
    names = [s["function"]["name"] for s in second]
    assert "disabled_tool" not in names
