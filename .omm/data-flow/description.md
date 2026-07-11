# Huginn 数据流

## 主链路：用户输入 → 响应

数据从用户输入进入系统，经过上下文构建、LLM 推理、工具执行、记忆写入，最终流回用户。完整链路如下：

### 1. 用户输入接入
- **CLI 路径**：用户在终端输入 → `cli/main.py` → `HuginnAgent.invoke()` 或 `agent.chat()`
- **Server 路径**：用户通过 WebSocket 发送消息 → `routes/ws.py` → `ws_helpers._handle_user_input()` → `get_agent()` 获取 Agent 实例 → `agent.chat()`

### 2. 上下文构建（ContextBuilder.build_input_messages）
每轮 LLM 调用前，`ContextBuilder` 并行组装以下上下文片段：
- **记忆召回** — `MemoryManager.recall_for_prompt(query)` 从长期记忆中检索相关条目；同时从 `research_log` 注入已验证/进行中的猜想
- **知识图谱查询** — `ProjectKnowledgeGraph.query(query, depth, top_k)` 查询项目级实体关系
- **领域知识库检索** — `KnowledgeBase.query(query, top_k=5)` 通过 ChromaDB 向量检索种子知识；KB chunk 结果再回查记忆，形成 KB↔Memory 交叉引用
- **情绪追踪** — `EmotionTracker.update_from_message(message)` 更新 persona 情绪轨迹
- **对话历史** — `ConversationTree.active_path()` 重建对话分支历史
- **计划/会话连续性** — 注入当前计划步骤、L1 结构坐标、上一会话摘要
- **认知提示** — 根据 `CognitiveStateMachine` 状态注入 dual-mode 注意力提示
- **演化规则** — 从 `evolution_rules.json` 注入高置信度历史教训

所有片段合并为一条 SystemMessage 插入消息列表，由 `PromptCacheBuilder` 组装最终输入。

### 3. Agent 图执行
- `build_graph()` 创建 `deepagents.create_deep_agent` 或降级为 `create_react_agent`
- 图以 system_prompt + tools + checkpointer 配置，递归上限 250（chat）/ 500（research）

### 4. LLM 推理 + 工具执行循环
- LLM 返回 AIMessage（可能含 tool_calls）
- 每个 tool_call 经过 Hook 系统：
  - `pre_tool_use` 钩子可拦截/改参/阻断
  - 工具执行（`HuginnTool.call()` → 权限检查 → 输入验证 → 执行 → provenance 快照）
  - `post_tool_use` 钩子观察结果
- 工具结果作为 ToolMessage 回传给 LLM，进入下一轮推理

### 5. 流状态处理（_process_stream_state）
每个 graph state 处理时：
- AIMessage → `memory.add_message("assistant", content)` + `conversation_tree.add_message()`
- ToolMessage → `memory.add_message("tool", content)` + `session_state.add_tool_result()`
- 重要工具结果（vasp_tool/lammps_tool/structure_tool）自动提升到长期记忆
- token 用量统计 → telemetry + Prometheus

### 6. 响应回传
- 流式 chunk 通过 WebSocket 实时推送给用户
- 阶段转换标记 `[PHASE:XXX]` 被检测并触发 `PhaseManager.transition()`
- 会话结束时 `STOP` 钩子触发，记忆持久化

## 旁路数据流

### Autoloop 数据流
Autoloop Engine 不走标准 chat 循环，而是按七阶段顺序执行：
- Perceive（LLM + 搜索工具）→ Hypothesize（动态 persona 选择）→ Plan（WorkflowEngine）→ Execute（直接调 Workflow/Coder）→ Validate（reviewer persona + validation 模块）→ Learn（Memory 写入 + evolution 规则提取）→ Report（ReportTool）
- 全程 provenance 记录到 JSONL，可回放

### Deli 研究数据流
DeliAutoResearch 管线独立运行：
- 每阶段用专门 sub-agent（LLM 调用走 `huginn.llm.get_model()`）
- 阶段间设 integrity gate，不通过则打回重做
- `ResearchWorkflow` 包装为 async generator，yield 事件给 WebSocket 层
