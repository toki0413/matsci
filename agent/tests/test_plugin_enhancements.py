"""测试 hot_reload / autofix_hook / config_schema 三个插件增强模块。

mock 策略:
  - EventBus: 用 MagicMock, dispatch 用 AsyncMock
  - PluginLoader: 用 MagicMock
  - AutoFixLoop: 用 MagicMock, apply_fix 直接返回构造好的 dict
  - watchfiles.awatch: 用 monkeypatch 换成 async generator, 模拟文件变更
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.api.event import EventType
from huginn.plugins import hot_reload
from huginn.plugins.autofix_hook import AutoFixHandler, ToolErrorEvent, ToolRetryEvent
from huginn.plugins.config_schema import (
    SCHEMA_FILENAME,
    ConfigField,
    generate_defaults,
    load_schema,
    merge_defaults,
    validate_config,
)
from huginn.plugins.hot_reload import HotReloadWatcher, is_hot_reload_enabled


# ── 辅助: 构造一个会产出若干变更集、之后挂起的假 awatch ─────────────────


def _make_fake_awatch(change_sets):
    """造一个假的 awatch: 依次 yield 变更集, 然后 hang 住等取消。"""

    async def fake_awatch(*args, **kwargs):
        for changes in change_sets:
            yield changes
        # yield 完后挂起, 等 stop() 取消任务
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    return fake_awatch


def _make_idle_awatch():
    """造一个永不产出、只等取消的假 awatch (用于生命周期测试)。"""

    async def idle_awatch(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return
        yield  # 不可达, 仅为让函数成为 async generator

    return idle_awatch


# ════════════════════════════════════════════════════════════════════
# hot_reload 测试
# ════════════════════════════════════════════════════════════════════


def test_watcher_construction():
    # 基本构造, 字段都按预期落位
    loader = MagicMock()
    watcher = HotReloadWatcher(loader, plugins_dir="/tmp/plugins", debounce_ms=50)

    assert watcher._loader is loader
    assert watcher._plugins_dir == Path("/tmp/plugins")
    assert watcher._debounce_ms == 50
    assert watcher._task is None
    assert watcher._pending == set()


def test_find_plugin_for_file(tmp_path):
    # 文件 -> 插件名映射
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    watcher = HotReloadWatcher(MagicMock(), plugins_dir=plugins_dir)

    # 正常子目录: 取第一级目录名
    assert watcher._find_plugin_for_file(
        str(plugins_dir / "plugin_a" / "main.py")
    ) == "plugin_a"
    # 嵌套子目录也只取第一级
    assert watcher._find_plugin_for_file(
        str(plugins_dir / "plugin_a" / "sub" / "utils.py")
    ) == "plugin_a"

    # __pycache__ 排除
    assert watcher._find_plugin_for_file(
        str(plugins_dir / "__pycache__" / "main.cpython-311.pyc")
    ) is None
    # __ 开头的目录都排除
    assert watcher._find_plugin_for_file(
        str(plugins_dir / "__internal__" / "x.py")
    ) is None
    # 隐藏目录排除
    assert watcher._find_plugin_for_file(
        str(plugins_dir / ".hidden" / "main.py")
    ) is None

    # 目录外的文件 -> None
    assert watcher._find_plugin_for_file(str(tmp_path / "outside.py")) is None


def test_is_hot_reload_enabled(monkeypatch):
    # 默认关
    monkeypatch.delenv("HUGINN_PLUGIN_RELOAD", raising=False)
    assert is_hot_reload_enabled() is False

    # 设 1 开
    monkeypatch.setenv("HUGINN_PLUGIN_RELOAD", "1")
    assert is_hot_reload_enabled() is True

    # 其它值都视为关
    monkeypatch.setenv("HUGINN_PLUGIN_RELOAD", "0")
    assert is_hot_reload_enabled() is False
    monkeypatch.setenv("HUGINN_PLUGIN_RELOAD", "yes")
    assert is_hot_reload_enabled() is False


@pytest.mark.asyncio
async def test_start_stop_lifecycle(monkeypatch):
    # start/stop 生命周期, 用 idle awatch 避免真去监听磁盘
    monkeypatch.setattr(hot_reload, "awatch", _make_idle_awatch())

    loader = MagicMock()
    watcher = HotReloadWatcher(loader, plugins_dir=".")

    assert watcher._task is None
    await watcher.start()
    assert watcher._task is not None
    assert not watcher._task.done()

    await watcher.stop()
    assert watcher._task is None


@pytest.mark.asyncio
async def test_start_idempotent(monkeypatch):
    # 重复 start 不会起多个任务
    monkeypatch.setattr(hot_reload, "awatch", _make_idle_awatch())

    watcher = HotReloadWatcher(MagicMock(), plugins_dir=".")
    await watcher.start()
    first_task = watcher._task
    await watcher.start()
    assert watcher._task is first_task
    await watcher.stop()


@pytest.mark.asyncio
async def test_file_change_triggers_reload(tmp_path, monkeypatch):
    # 改了一个 .py -> 对应插件被 reload 一次
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "plugin_a").mkdir()
    (plugins_dir / "plugin_a" / "main.py").write_text("# init", encoding="utf-8")

    loader = MagicMock()
    loader.reload.return_value = True

    change_sets = [
        {(1, str(plugins_dir / "plugin_a" / "main.py"))},
    ]
    monkeypatch.setattr(hot_reload, "awatch", _make_fake_awatch(change_sets))

    watcher = HotReloadWatcher(loader, plugins_dir=plugins_dir, debounce_ms=10)
    await watcher.start()
    # 等 debounce + reload 完成 (10ms 防抖 + to_thread 开销)
    await asyncio.sleep(0.1)
    await watcher.stop()

    loader.reload.assert_called_once_with("plugin_a")


@pytest.mark.asyncio
async def test_debounce_coalesces_rapid_changes(tmp_path, monkeypatch):
    # 同一插件三次快速变更, 防抖窗口内合并成一次 reload
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "plugin_a").mkdir()

    loader = MagicMock()
    loader.reload.return_value = True

    change_sets = [
        {(1, str(plugins_dir / "plugin_a" / "main.py"))},
        {(1, str(plugins_dir / "plugin_a" / "utils.py"))},
        {(1, str(plugins_dir / "plugin_a" / "config.py"))},
    ]
    monkeypatch.setattr(hot_reload, "awatch", _make_fake_awatch(change_sets))

    watcher = HotReloadWatcher(loader, plugins_dir=plugins_dir, debounce_ms=50)
    await watcher.start()
    # 三次变更都在 50ms 防抖窗口内到达 -> 合并
    await asyncio.sleep(0.15)
    await watcher.stop()

    assert loader.reload.call_count == 1
    loader.reload.assert_called_with("plugin_a")


@pytest.mark.asyncio
async def test_multiple_plugins_reload_separately(tmp_path, monkeypatch):
    # 两个不同插件的变更 -> 各 reload 一次
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "plugin_a").mkdir()
    (plugins_dir / "plugin_b").mkdir()

    loader = MagicMock()
    loader.reload.return_value = True

    change_sets = [
        {(1, str(plugins_dir / "plugin_a" / "main.py"))},
        {(1, str(plugins_dir / "plugin_b" / "main.py"))},
    ]
    monkeypatch.setattr(hot_reload, "awatch", _make_fake_awatch(change_sets))

    watcher = HotReloadWatcher(loader, plugins_dir=plugins_dir, debounce_ms=30)
    await watcher.start()
    await asyncio.sleep(0.1)
    await watcher.stop()

    assert loader.reload.call_count == 2
    called = {call.args[0] for call in loader.reload.call_args_list}
    assert called == {"plugin_a", "plugin_b"}


@pytest.mark.asyncio
async def test_pycache_changes_ignored(tmp_path, monkeypatch):
    # __pycache__ 下的变更不触发 reload
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "plugin_a").mkdir()

    loader = MagicMock()
    loader.reload.return_value = True

    change_sets = [
        {(1, str(plugins_dir / "__pycache__" / "main.cpython-311.pyc"))},
    ]
    monkeypatch.setattr(hot_reload, "awatch", _make_fake_awatch(change_sets))

    watcher = HotReloadWatcher(loader, plugins_dir=plugins_dir, debounce_ms=10)
    await watcher.start()
    await asyncio.sleep(0.1)
    await watcher.stop()

    loader.reload.assert_not_called()


@pytest.mark.asyncio
async def test_reload_failure_is_swallowed(tmp_path, monkeypatch):
    # loader.reload 抛异常不能让 watcher 整个挂掉
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "plugin_a").mkdir()

    loader = MagicMock()
    loader.reload.side_effect = RuntimeError("boom")

    change_sets = [
        {(1, str(plugins_dir / "plugin_a" / "main.py"))},
    ]
    monkeypatch.setattr(hot_reload, "awatch", _make_fake_awatch(change_sets))

    watcher = HotReloadWatcher(loader, plugins_dir=plugins_dir, debounce_ms=10)
    await watcher.start()
    await asyncio.sleep(0.1)
    # 异常被吞, stop 仍能正常结束
    await watcher.stop()

    loader.reload.assert_called_once_with("plugin_a")


# ════════════════════════════════════════════════════════════════════
# autofix_hook 测试
# ════════════════════════════════════════════════════════════════════


def _make_event_bus():
    """造一个带 AsyncMock dispatch 的假 EventBus。"""
    bus = MagicMock()
    bus.registry = MagicMock()
    bus.dispatch = AsyncMock()
    return bus


def test_tool_error_event_construction():
    ev = ToolErrorEvent(
        tool_name="vasp_tool",
        error_message="ZBRENT fatal",
        current_params={"ALGO": "Fast"},
    )
    assert ev.tool_name == "vasp_tool"
    assert ev.error_message == "ZBRENT fatal"
    assert ev.current_params == {"ALGO": "Fast"}
    assert ev.output_files == []
    assert ev.type == EventType.ON_PLUGIN_ERROR


def test_tool_error_event_default_type():
    # 不传 type 也能构造, 默认归到 ON_PLUGIN_ERROR
    ev = ToolErrorEvent(tool_name="qe_tool", error_message="x")
    assert ev.type == EventType.ON_PLUGIN_ERROR


def test_tool_retry_event_construction():
    ev = ToolRetryEvent(
        tool_name="vasp_tool",
        fixed_params={"ALGO": "Normal"},
        matched_patterns=["robust SCF"],
        attempt=2,
    )
    assert ev.tool_name == "vasp_tool"
    assert ev.fixed_params == {"ALGO": "Normal"}
    assert ev.matched_patterns == ["robust SCF"]
    assert ev.attempt == 2
    assert ev.type == EventType.ON_PLUGIN_ERROR


@pytest.mark.asyncio
async def test_handle_tool_error_fix_found_dispatches_retry():
    # 命中修复规则 -> dispatch 一个 ToolRetryEvent, 内部标记被剥掉
    bus = _make_event_bus()
    autofix = MagicMock()
    autofix.apply_fix.return_value = {
        "ALGO": "Normal",
        "NELMIN": 6,
        "__auto_fix": "Switch to more robust SCF algorithm",
        "__auto_fix_patterns_matched": 2,
    }

    handler = AutoFixHandler(bus, autofix=autofix)
    event = ToolErrorEvent(
        tool_name="vasp_tool",
        error_message="ZBRENT: fatal error in bracketing",
        current_params={"ALGO": "Fast"},
    )
    await handler._handle_tool_error(event)

    autofix.apply_fix.assert_called_once_with(
        "vasp_tool", "ZBRENT: fatal error in bracketing", {"ALGO": "Fast"}
    )
    bus.dispatch.assert_awaited_once()

    dispatched = bus.dispatch.await_args.args[0]
    assert isinstance(dispatched, ToolRetryEvent)
    assert dispatched.tool_name == "vasp_tool"
    assert dispatched.fixed_params == {"ALGO": "Normal", "NELMIN": 6}
    # 内部标记不能泄漏到 retry 参数里
    assert "__auto_fix" not in dispatched.fixed_params
    assert "__auto_fix_patterns_matched" not in dispatched.fixed_params
    assert dispatched.attempt == 1
    assert len(dispatched.matched_patterns) == 1
    assert "robust SCF" in dispatched.matched_patterns[0]


@pytest.mark.asyncio
async def test_handle_tool_error_no_fix_no_dispatch():
    # apply_fix 返回 None -> 不 dispatch
    bus = _make_event_bus()
    autofix = MagicMock()
    autofix.apply_fix.return_value = None

    handler = AutoFixHandler(bus, autofix=autofix)
    event = ToolErrorEvent(
        tool_name="vasp_tool",
        error_message="some unknown error",
        current_params={},
    )
    await handler._handle_tool_error(event)

    autofix.apply_fix.assert_called_once()
    bus.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_handle_tool_error_skips_missing_fields():
    # 没 tool_name / error_message -> 直接跳过, 不调 autofix
    bus = _make_event_bus()
    autofix = MagicMock()

    handler = AutoFixHandler(bus, autofix=autofix)
    event = ToolErrorEvent(tool_name="", error_message="", current_params={})
    await handler._handle_tool_error(event)

    autofix.apply_fix.assert_not_called()
    bus.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_retry_cap_blocks_after_max():
    # 达到 MAX_RETRIES 后不再尝试
    bus = _make_event_bus()
    autofix = MagicMock()
    autofix.apply_fix.return_value = {
        "ALGO": "Normal",
        "__auto_fix": "d",
        "__auto_fix_patterns_matched": 1,
    }

    handler = AutoFixHandler(bus, autofix=autofix)
    handler._retry_counts["vasp_tool"] = AutoFixHandler.MAX_RETRIES

    event = ToolErrorEvent(
        tool_name="vasp_tool", error_message="ZBRENT", current_params={}
    )
    await handler._handle_tool_error(event)

    autofix.apply_fix.assert_not_called()
    bus.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_retry_count_increments():
    # 每次成功修复后计数 +1
    bus = _make_event_bus()
    autofix = MagicMock()
    autofix.apply_fix.return_value = {
        "ALGO": "Normal",
        "__auto_fix": "d",
        "__auto_fix_patterns_matched": 1,
    }

    handler = AutoFixHandler(bus, autofix=autofix)
    assert handler._retry_counts.get("vasp_tool", 0) == 0

    await handler._handle_tool_error(
        ToolErrorEvent(tool_name="vasp_tool", error_message="ZBRENT", current_params={})
    )
    assert handler._retry_counts["vasp_tool"] == 1

    await handler._handle_tool_error(
        ToolErrorEvent(tool_name="vasp_tool", error_message="ZBRENT", current_params={})
    )
    assert handler._retry_counts["vasp_tool"] == 2


@pytest.mark.asyncio
async def test_retry_attempt_number_in_event():
    # retry 事件里的 attempt 跟内部计数一致
    bus = _make_event_bus()
    autofix = MagicMock()
    autofix.apply_fix.return_value = {
        "ALGO": "Normal",
        "__auto_fix": "d",
        "__auto_fix_patterns_matched": 1,
    }

    handler = AutoFixHandler(bus, autofix=autofix)
    handler._retry_counts["vasp_tool"] = 1  # 已经试过一次

    await handler._handle_tool_error(
        ToolErrorEvent(tool_name="vasp_tool", error_message="ZBRENT", current_params={})
    )
    dispatched = bus.dispatch.await_args.args[0]
    assert dispatched.attempt == 2


def test_reset_retries_single():
    handler = AutoFixHandler(_make_event_bus())
    handler._retry_counts["vasp_tool"] = 2
    handler._retry_counts["qe_tool"] = 1

    handler.reset_retries("vasp_tool")
    assert "vasp_tool" not in handler._retry_counts
    assert handler._retry_counts.get("qe_tool") == 1


def test_reset_retries_all():
    handler = AutoFixHandler(_make_event_bus())
    handler._retry_counts["vasp_tool"] = 2
    handler._retry_counts["qe_tool"] = 1

    handler.reset_retries()
    assert handler._retry_counts == {}


def test_register_with_real_registry():
    # register 把 handler 挂到真实 registry 上
    from huginn.plugins.registry import StarHandlerRegistry

    registry = StarHandlerRegistry()
    bus = MagicMock()
    bus.registry = registry

    handler = AutoFixHandler(bus)
    handler.register()

    handlers = registry.get_handlers(EventType.ON_PLUGIN_ERROR)
    assert len(handlers) == 1
    meta = handlers[0]
    assert meta.priority == AutoFixHandler.PRIORITY
    assert meta.plugin_name == "__autofix__"
    assert meta.handler == handler._handle_tool_error
    assert meta.name == "autofix_on_tool_error"


def test_register_without_registry_logs_and_skips():
    # registry 为 None 时不能注册, 但不抛
    bus = MagicMock()
    bus.registry = None
    handler = AutoFixHandler(bus)
    handler.register()  # 不抛异常即可


@pytest.mark.asyncio
async def test_end_to_end_through_real_eventbus():
    # 真实 EventBus + 真实 registry + mock autofix, 走一遍 dispatch
    from huginn.plugins.event_bus import EventBus
    from huginn.plugins.registry import StarHandlerRegistry

    registry = StarHandlerRegistry()
    bus = EventBus(registry=registry, swallow_exceptions=True)

    autofix = MagicMock()
    autofix.apply_fix.return_value = {
        "ALGO": "Normal",
        "__auto_fix": "robust SCF",
        "__auto_fix_patterns_matched": 1,
    }
    handler = AutoFixHandler(bus, autofix=autofix)
    handler.register()

    event = ToolErrorEvent(
        tool_name="vasp_tool",
        error_message="ZBRENT: fatal error",
        current_params={"ALGO": "Fast"},
    )
    result = await bus.dispatch(event)

    # handler 执行了, 而且 dispatch 了 retry 事件 (retry 事件又触发同一 handler,
    # 但 retry 事件没 error_message, _handle_tool_error 会直接 return)
    assert result.executed >= 1
    autofix.apply_fix.assert_called_once()


# ════════════════════════════════════════════════════════════════════
# config_schema 测试
# ════════════════════════════════════════════════════════════════════


def test_load_schema(tmp_path):
    schema = {
        "encut": {
            "type": "number",
            "default": 520,
            "min": 200,
            "max": 2000,
            "description": "cutoff",
        },
        "xc": {
            "type": "string",
            "default": "PBE",
            "enum": ["PBE", "LDA"],
            "description": "xc functional",
        },
        "kpar": {"type": "integer", "default": 2, "required": True},
    }
    (tmp_path / SCHEMA_FILENAME).write_text(json.dumps(schema), encoding="utf-8")

    fields = load_schema(tmp_path)
    assert fields is not None
    assert set(fields.keys()) == {"encut", "xc", "kpar"}

    assert fields["encut"].type == "number"
    assert fields["encut"].default == 520
    assert fields["encut"].min == 200
    assert fields["encut"].max == 2000
    assert fields["encut"].description == "cutoff"

    assert fields["xc"].enum == ["PBE", "LDA"]
    assert fields["kpar"].required is True
    assert fields["kpar"].type == "integer"


def test_load_schema_missing_file(tmp_path):
    # 没有 schema 文件 -> None
    assert load_schema(tmp_path) is None


def test_load_schema_invalid_json(tmp_path):
    # 坏 JSON -> None, 不抛
    (tmp_path / SCHEMA_FILENAME).write_text("not json {", encoding="utf-8")
    assert load_schema(tmp_path) is None


def test_load_schema_defaults_type_field(tmp_path):
    # 没写 type 时默认 string
    (tmp_path / SCHEMA_FILENAME).write_text(
        json.dumps({"foo": {"default": "bar"}}), encoding="utf-8"
    )
    fields = load_schema(tmp_path)
    assert fields is not None
    assert fields["foo"].type == "string"
    assert fields["foo"].default == "bar"


def test_generate_defaults_with_explicit_defaults():
    fields = {
        "s": ConfigField(name="s", type="string", default="hi"),
        "n": ConfigField(name="n", type="number", default=3.14),
        "i": ConfigField(name="i", type="integer", default=5),
        "b": ConfigField(name="b", type="boolean", default=True),
        "a": ConfigField(name="a", type="array", default=[1, 2]),
        "o": ConfigField(name="o", type="object", default={"k": "v"}),
    }
    d = generate_defaults(fields)
    assert d == {
        "s": "hi",
        "n": 3.14,
        "i": 5,
        "b": True,
        "a": [1, 2],
        "o": {"k": "v"},
    }


def test_generate_defaults_zero_values_when_no_default():
    # 没显式 default 时, 按类型给零值
    fields = {
        "s": ConfigField(name="s", type="string"),
        "n": ConfigField(name="n", type="number"),
        "i": ConfigField(name="i", type="integer"),
        "b": ConfigField(name="b", type="boolean"),
        "a": ConfigField(name="a", type="array"),
        "o": ConfigField(name="o", type="object"),
    }
    d = generate_defaults(fields)
    assert d == {"s": "", "n": 0, "i": 0, "b": False, "a": [], "o": {}}


def test_validate_config_type_errors():
    fields = {
        "s": ConfigField(name="s", type="string"),
        "n": ConfigField(name="n", type="number"),
        "i": ConfigField(name="i", type="integer"),
        "b": ConfigField(name="b", type="boolean"),
        "a": ConfigField(name="a", type="array"),
    }
    errors = validate_config(
        {"s": 123, "n": "x", "i": 1.5, "b": "yes", "a": "no"}, fields
    )
    assert len(errors) == 5
    joined = " ".join(errors)
    assert "expected string" in joined
    assert "expected number" in joined
    assert "expected integer" in joined
    assert "expected boolean" in joined
    assert "expected array" in joined


def test_validate_config_enum_violation():
    fields = {"xc": ConfigField(name="xc", type="string", enum=["PBE", "LDA"])}
    errors = validate_config({"xc": "B3LYP"}, fields)
    assert len(errors) == 1
    assert "not in" in errors[0]


def test_validate_config_enum_ok():
    fields = {"xc": ConfigField(name="xc", type="string", enum=["PBE", "LDA"])}
    assert validate_config({"xc": "PBE"}, fields) == []


def test_validate_config_range_below_min():
    fields = {"encut": ConfigField(name="encut", type="number", min=200, max=2000)}
    errors = validate_config({"encut": 100}, fields)
    assert len(errors) == 1
    assert "below minimum" in errors[0]


def test_validate_config_range_above_max():
    fields = {"encut": ConfigField(name="encut", type="number", min=200, max=2000)}
    errors = validate_config({"encut": 5000}, fields)
    assert len(errors) == 1
    assert "above maximum" in errors[0]


def test_validate_config_required_missing():
    fields = {"kpar": ConfigField(name="kpar", type="integer", required=True)}
    errors = validate_config({}, fields)
    assert len(errors) == 1
    assert "Missing required" in errors[0]


def test_validate_config_required_present_ok():
    fields = {"kpar": ConfigField(name="kpar", type="integer", required=True)}
    assert validate_config({"kpar": 4}, fields) == []


def test_validate_config_unknown_field_allowed():
    # 未知字段放行 (向前兼容)
    fields = {"encut": ConfigField(name="encut", type="number", min=200, max=2000)}
    assert validate_config({"encut": 520, "future_field": "whatever"}, fields) == []


def test_validate_config_all_ok():
    fields = {
        "encut": ConfigField(name="encut", type="number", min=200, max=2000),
        "xc": ConfigField(name="xc", type="string", enum=["PBE", "LDA"]),
        "kpar": ConfigField(name="kpar", type="integer", required=True),
    }
    assert validate_config(
        {"encut": 520, "xc": "PBE", "kpar": 2}, fields
    ) == []


def test_merge_defaults_user_overrides():
    fields = {
        "encut": ConfigField(name="encut", type="number", default=520),
        "xc": ConfigField(name="xc", type="string", default="PBE"),
        "kpar": ConfigField(name="kpar", type="integer", default=2),
    }
    merged = merge_defaults({"encut": 600}, fields)
    # 用户值优先, 其余用默认
    assert merged == {"encut": 600, "xc": "PBE", "kpar": 2}


def test_merge_defaults_empty_user():
    fields = {
        "encut": ConfigField(name="encut", type="number", default=520),
    }
    merged = merge_defaults({}, fields)
    assert merged == {"encut": 520}


def test_round_trip_load_merge_validate(tmp_path):
    # 完整链路: load schema -> generate defaults -> merge user -> validate
    schema = {
        "encut": {
            "type": "number",
            "default": 520,
            "min": 200,
            "max": 2000,
            "description": "cutoff",
        },
        "xc_functional": {
            "type": "string",
            "default": "PBE",
            "enum": ["PBE", "LDA", "HSE06"],
            "description": "xc",
        },
        "kpar": {"type": "integer", "default": 2, "required": True},
    }
    (tmp_path / SCHEMA_FILENAME).write_text(json.dumps(schema), encoding="utf-8")

    fields = load_schema(tmp_path)
    assert fields is not None

    defaults = generate_defaults(fields)
    assert defaults == {"encut": 520, "xc_functional": "PBE", "kpar": 2}

    # 用户只改 encut, 其余走默认
    merged = merge_defaults({"encut": 700}, fields)
    assert merged == {"encut": 700, "xc_functional": "PBE", "kpar": 2}
    assert validate_config(merged, fields) == []

    # 非法值: encut 太小 + xc 不在 enum -> 至少两个错误
    bad = merge_defaults({"encut": 50, "xc_functional": "B3LYP"}, fields)
    errors = validate_config(bad, fields)
    assert len(errors) >= 2
    joined = " ".join(errors)
    assert "below minimum" in joined
    assert "not in" in joined


def test_round_trip_missing_required_fails(tmp_path):
    # required 字段缺省时, 即便 merge 补了默认值, 单独校验空配置也能抓到
    schema = {
        "kpar": {"type": "integer", "default": 2, "required": True},
    }
    (tmp_path / SCHEMA_FILENAME).write_text(json.dumps(schema), encoding="utf-8")

    fields = load_schema(tmp_path)
    assert fields is not None

    errors = validate_config({}, fields)
    assert any("Missing required" in e for e in errors)
