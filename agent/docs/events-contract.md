# 事件契约 (插件事件面)

自动生成: `python -m huginn.cli.config_audit --events --out docs/events-contract.md`.
**EventType** 是插件可监听的事件全集 (plugins/api/event.py); `dispatch` 是代码里引用该事件的位置, 提示它在哪里被发出 (静态扫描, 取前 3 处)。**UnifiedBus 发射接口**是语义化统一入口, 每次 publish 扇出到 HookManager / 内部 EventBus / 插件 EventBus / PetBus 各子系统。

### EventType 成员

| 事件 | 分组 | 派发点 |
|---|---|---|
| ON_HUGINN_LOADED | 生命周期 | lifespan.py:779 |
| ON_PLUGIN_LOADED | 生命周期 | plugins/loader.py:238, plugins/loader.py:291 |
| ON_PLUGIN_UNLOADED | 生命周期 | plugins/loader.py:331 |
| ON_PLUGIN_ERROR | 生命周期 | plugins/loader.py:225, plugins/loader.py:317 |
| ON_AGENT_BEGIN | Agent 流水线 | events/unified_bus.py:151 |
| ON_AGENT_DONE | Agent 流水线 | events/unified_bus.py:182 |
| ON_LLM_REQUEST | LLM 调用 | api/event.py:94, api/filter.py:190, events/unified_bus.py:266 |
| ON_LLM_RESPONSE | LLM 调用 | api/event.py:106, api/filter.py:203, events/unified_bus.py:276 |
| ON_TOOL_CALL | 工具调用三段式 | api/event.py:121, api/filter.py:224, events/unified_bus.py:240 +1 处 |
| ON_TOOL_EXECUTE | 工具调用三段式 | tools/adapter.py:915 |
| ON_TOOL_RESPOND | 工具调用三段式 | api/event.py:134, api/filter.py:242, events/unified_bus.py:246 +1 处 |
| ON_WORKFLOW_BEGIN | Workflow 钩子 (材料科研特色) | autoloop/cognitive_loop.py:179 |
| ON_WORKFLOW_STAGE_START | Workflow 钩子 (材料科研特色) | autoloop/cognitive_loop.py:772, autoloop/cognitive_loop.py:848 |
| ON_WORKFLOW_STAGE_DONE | Workflow 钩子 (材料科研特色) | api/filter.py:255, autoloop/cognitive_loop.py:803, autoloop/cognitive_loop.py:871 +1 处 |
| ON_WORKFLOW_DONE | Workflow 钩子 (材料科研特色) | autoloop/cognitive_loop.py:305 |
| ON_WORKFLOW_FAILED | Workflow 钩子 (材料科研特色) | autoloop/cognitive_loop.py:805, autoloop/cognitive_loop.py:877 |
| ON_MESSAGE_RECEIVED | 消息 | api/filter.py:129, api/filter.py:143, api/filter.py:157 +2 处 |
| ON_BEFORE_MESSAGE_SENT | 消息 | events/unified_bus.py:300 |
| ON_AFTER_MESSAGE_SENT | 消息 | events/unified_bus.py:309 |

### UnifiedBus 发射接口 (语义化)

- `publish_after_message_sent()`
- `publish_before_message_sent()`
- `publish_compact()`
- `publish_csm_transition()`
- `publish_llm_request()`
- `publish_llm_response()`
- `publish_message_received()`
- `publish_pet_mood()`
- `publish_session_end()`
- `publish_session_start()`
- `publish_step_retry()`
- `publish_stop()`
- `publish_tool_post()`
- `publish_tool_pre()`
