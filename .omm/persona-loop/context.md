# Persona-Memory-Knowledge 循环 — 关键设计决策与约束

## 核心设计决策

### 1. 双路径 Persona 选择
Persona 选择有两条独立路径，取决于运行模式：
- **Chat 模式**：`PersonaMatcher` 按查询语义匹配 persona，用户无感知
- **Autoloop 模式**：`_PHASE_PERSONAS` 硬编码阶段-persona 映射，`_hypothesize` 中 `_pick_hypothesis_persona` 按研究内容动态选择（MD 信号 → md_expert，否则 → dft_expert）

这种设计让 chat 模式灵活响应用户意图，autoloop 模式则保证每个阶段有最适合的专家视角。

### 2. Persona 标签化记忆
Autoloop `_learn()` 将 persona 名称作为 memory tag 写入（`persona:{name}`），同时写入 r_phys 物理校验分数。这使得：
- `_pick_hypothesis_persona` 可查询历史 persona 效果
- Memory 检索可按 persona 过滤
- r_phys 分数直接作为 memory importance（`min(0.9, float(r_phys))`）

### 3. 情绪轨迹持久化
`EmotionTracker` 的情绪状态按 workspace + persona 持久化到 JSON 文件。情绪不是随机的，而是交互历史的计算摘要：praise 提升 valence，criticism 降低 trust，task_success 提升 arousal，silence 增加 loneliness。情绪随时间衰减。

### 4. KB ↔ Memory 交叉引用
ContextBuilder 在 KB 检索到 chunk 后，用 chunk 文本回查 memory（仅 top-2 chunk）。这创建了双向链接：如果 Agent 之前学过与 KB 内容相关的知识（如某次仿真结果），它会在 KB 检索结果下方以 `↳ Memory:` 形式浮现。这是 persona 循环中知识层与记忆层的唯一交叉点。

### 5. 阶段-Persona 联动
`PhaseGate` 在 validate→learn 转折点评估证据充分性。验证阶段自动使用 reviewer persona 做批判性审视，发现 high 级别问题则阻断进入 learn。报告阶段切换到 tutor persona 做教学化输出。这种联动是硬编码的，不由 LLM 决定。

### 6. Evolution 奖励回流
Autoloop `_learn()` 将 r_phys 物理校验分数喂给 `EvolutionEngine` 作为 reward 信号。EvolutionEngine 基于奖励信号学习规则（哪些工具组合在什么条件下有效），高置信度规则通过 `build_evolution_rules()` 注入上下文，影响 LLM 后续工具选择。这形成了 persona → 工具选择 → 验证 → reward → 规则 → persona 的完整闭环。

### 7. 研究日志作为 Memory 扩展
`ResearchLog` 独立于 `LongTermMemory`，专门跟踪猜想（conjecture）的生命周期：in_progress → verified/refuted。`ContextBuilder.build_memory_text()` 在召回长期记忆后，额外注入最近的 verified 和 in_progress 猜想。这让 persona 在形成新假设时能参考已验证的猜想链。

## 约束

- **PersonaMatcher 降级** — 当 ChromaDB embedding 模型不可用时，降级为关键词重叠评分，匹配精度下降但功能不中断
- **情绪维度范围** — 所有情绪维度归一化到 [-1, 1] 或 [0, 1]，valence/arousal 可为负，trust/affection/interest 在 [0, 1]
- **Memory importance 上限** — Autoloop 迭代写入的 importance 上限为 0.9（`min(0.9, float(r_phys))`），避免单次高分校验垄断检索
- **演化规则注入上限** — 只注入 confidence >= 0.5 的规则，最多 5 条，避免 prompt 膨胀
- **Persona 文件格式** — 支持三种来源：内置 Python 定义（`BUILT_IN_PERSONAS`）、JSON 文件（`persona_templates.json`）、Nuwa 风格 SKILL.md markdown
- **_PHASE_PERSONAS 完整性** — `assert set(_PHASE_PERSONAS.keys()) == set(AUTOLOOP_PHASES)` 确保每个阶段都有 persona 映射（None 也是合法值）
