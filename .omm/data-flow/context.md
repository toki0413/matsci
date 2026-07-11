# 数据流 — 关键设计决策与约束

## 核心设计决策

### 1. 每轮全量上下文重建
`ContextBuilder.build_input_messages()` 在每次 LLM 调用前完整重建上下文，不做增量缓存。这意味着记忆、KG、KB 的最新状态总能在下一轮生效，代价是每轮的查询开销。

### 2. KB ↔ Memory 交叉引用
当 KB 检索到 chunk 后，取 top-2 chunk 的文本作为查询回查长期记忆（`memory.recall_for_prompt`），形成双向链接：如果 Agent 之前学过与该 KB 内容相关的知识，会在此处浮现。这是数据流中唯一的跨存储交叉引用。

### 3. 流状态增量处理
`_process_stream_state()` 通过 `_state_msg_offsets` 字典记录每个 thread_id 已处理的消息偏移量，只处理新增消息，避免重复写入记忆和对话树。

### 4. 自动提升策略
只有 `vasp_tool`、`lammps_tool`、`structure_tool` 的成功结果会被自动提升到长期记忆。其他工具结果仅保留在会话记忆中。这个白名单在 `MemoryManager.add_tool_call()` 中硬编码。

### 5. 工具结果压缩
`ToolAdapter` 在工具执行后检查输出 token 数，超过 `max_tool_output_tokens` 时触发摘要压缩（`compression_max_tokens` 控制压缩目标）。压缩后的结果才进入 LLM 上下文。

### 6. Provenance contextvars 追踪
工具调用的 provenance 通过 `contextvars.ContextVar` 传递，Autoloop Engine 在 `run()` 时设置 collector，`HuginnTool.call()` 自动写入快照。非 Autoloop 场景下 collector 为 None，零开销。

### 7. 上下文压缩
当消息 token 数超过模型上下文窗口时，`StreamingMixin` 触发压缩：
- `pre_compact` 钩子执行
- `compact_messages()` 或 `summarize_compact_messages()` 压缩历史
- L1 结构坐标始终保留（写入 `_build_compact_summary` 前缀）

## 约束

- **记忆查询容错** — 所有记忆/KG/KB 查询都包裹在 try-except 中，失败时返回空字符串而非中断流程
- **工具调用上限** — `max_tool_calls` 限制单轮总调用数，`max_tool_calls_per_tool` 限制单工具调用数
- **重试预算** — 同一工具同一参数最多重试 3 次，超过后停止并向用户解释根因
- **隐私保护** — `PrivacyGuard` 在消息进入 LLM 前扫描密钥，可选 redact 或 block
- **checkpointer 影响** — 当 checkpointer 活跃时，对话历史从 checkpoint 恢复而非手动注入（`include_history` 为 False）
