# Persona-Memory-Knowledge 循环

## 循环概述

Huginn 的 persona、memory、knowledge 三个子系统构成一个闭环反馈系统。Persona 塑造 LLM 的行为风格和领域偏向；LLM 的输出通过工具执行写入 memory 和 KG/KB；下一轮上下文构建时，memory 召回、KB 检索、KG 查询的结果连同 persona 情绪状态一起注入，影响后续推理。

## 循环的四个阶段

### 1. Persona 注入
两条路径触发 persona 选择：

**Chat 模式**：`PersonaMatcher.match(query)` 通过语义相似度（ChromaDB embedding）或关键词重叠，从 `PersonaManager` 管理的 persona 列表中选出最匹配的角色。内置角色包括 default、dft_expert、md_expert、reviewer、tutor、research。选中 persona 的 `system_prompt` 通过 `ContextMixin._effective_system_prompt()` 注入。

**Autoloop 模式**：`_PHASE_PERSONAS` 表为七阶段各指定一个 persona：
- perceive → default
- hypothesize → 动态选择（`_pick_hypothesis_persona` 按研究类型选 dft_expert 或 md_expert）
- plan → default
- execute → 无（直接调 workflow，不走 LLM）
- validate → reviewer（批判性审视）
- learn → default
- report → tutor（教学风格输出）

`persona_loader.py` 支持从 Nuwa 风格的 SKILL.md markdown 文件加载自定义 persona。

### 2. Memory 写入
LLM 输出和工具执行结果写入记忆：
- **会话记忆**：`MemoryManager.add_message()` / `add_tool_call()` 记录每条消息和工具调用
- **长期记忆**：重要工具结果（vasp_tool/lammps_tool/structure_tool）自动提升；Autoloop `_learn()` 显式写入迭代结果，包含 hypothesis、validation、persona 名称、r_phys 分数、visual primitives、surprise 分数，并打上 `persona:{name}` 和 `r_phys:{score}` 标签
- **研究日志**：猜想以 verified/in_progress 状态记录，供后续轮次注入
- **KG 更新**：`ProjectKnowledgeGraph` 在 Autoloop 执行中积累项目实体与关系

### 3. KB/KG 查询
下一轮上下文构建时：
- **Memory 召回**：`ContextBuilder.build_memory_text(query)` → `MemoryManager.recall_for_prompt()` 检索相关长期记忆 + 研究日志猜想
- **KB 检索**：`ContextBuilder.build_kb_text(query)` → `KnowledgeBase.query()` 向量检索种子知识文档；top-2 chunk 回查 memory，形成 KB↔Memory 交叉引用
- **KG 查询**：`ContextBuilder.build_kg_text(query)` → `ProjectKnowledgeGraph.query()` 检索项目级实体关系

### 4. Persona 反馈
- **情绪反馈**：`EmotionTracker.update_from_message(message)` 更新 persona 的情绪状态（valence/arousal/trust/interest 等维度），`context_prompt()` 将当前情绪注入上下文，影响 LLM 的语气和态度。情绪轨迹按 workspace + persona 持久化。
- **Persona 历史反馈**：Autoloop `_learn()` 将 persona 名称和 r_phys 分数写入 memory tags。`_pick_hypothesis_persona` 可查询历史效果，选择表现更好的 persona。
- **演化规则反馈**：工具失败 → `EvolutionEngine` 提取规则 → `build_evolution_rules()` 注入高置信度教训 → 影响 LLM 后续工具选择策略。
- **阶段-persona 联动**：验证阶段自动切换到 reviewer persona 做批判性审视；报告阶段切换到 tutor persona 做教学化输出。阶段转换由 `PhaseGate` 和 LLM 的 `[PHASE:XXX]` 标记驱动。

## 循环的闭合点

整个循环在 Autoloop 的 validate→learn 转折点闭合：reviewer persona 驱动的验证结果（r_phys 分数）被 `_learn()` 写入 memory，带上了 persona 标签和物理校验分数。这个分数同时回流到 EvolutionEngine 作为 reward 信号，驱动基于奖励的进化。下一轮 `_hypothesize` 时，memory 召回带上历史 persona 效果，`_pick_hypothesis_persona` 据此优化选择，形成完整的 persona → memory → knowledge → persona 反馈环。
