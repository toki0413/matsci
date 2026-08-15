# Everything is a Plugin — 段插件化设计

> 状态: 设计稿(未实施)
> 范围: prompt 段插件化 + 两形态插件框架 + 其他子系统适配评估
> 关联: `prompt_builder.py`, `context.py`, `plugins/`, `core_types.py`, 记忆整理, compaction

---

## 1. 哲学

**"Everything is a plugin" 转译: 任何一段可独立启停、可替换、可由第三方扩展的能力, 都注册为插件。**

对 prompt 体系: prompt 段是注册在 `on_llm_request` 事件上的 handler, 按 priority 依次在 `system_prompt` 上 append。段之间不再有核心依赖, 第三方可以注册自己的段而不改一行核心代码。

对工具 / 记忆 / 压缩等系统: 同理, 把"策略分歧点"抽象成可插拔单元。

---

## 2. 两形态框架(核心心智模型)

**不是所有东西都应该套成事件总线。** 事件总线是重量级(异步、权限、yield 流式、顺序仲裁), 对其上每个调用都走事件流会引入不必要的开销。要区分两种形态:

| 形态 | 机制 | 适用场景 | 例子 |
|---|---|---|---|
| **形态 A: 事件钩子** | `@filter.on_xxx(priority=N)` 注册 async handler, 可改事件对象, 可 `event.stop()` 阻断 | 有生命周期、需要顺序 + 阻断、粒度到单次发生时 | prompt 段、工具调用拦截、on_tool_respond |
| **形态 B: 策略注册表** | `dict[str, Callable]` + priority, 选一个策略执行, 无副作用事件流 | 性能敏感、纯策略选择、无嵌套副作用 | compaction policy、tool result 压缩、记忆整理、沙箱选择 |

**选择判据:**
1. 有生命周期钩子(发生在某时刻、可被阻断) → 形态 A
2. 只是"这条数据怎么处理"的多种策略选择 → 形态 B
3. 性能敏感路径(如 O(n) 压缩) → 形态 B 优先, 避免事件分发开销

---

## 3. 形态 A: 事件钩子 — prompt 段插件化

### 3.1 接口契约

```python
class PromptSegmentPlugin(Star):
    """Prompt 段插件的基类。
    子类定义 handler, 用 @filter.on_llm_request(priority=N) 装饰。
    handler 签名: async def handler(self, event: LLMRequestEvent) -> None
    """
```

handler 只做 `event.system_prompt += "\n\n## SEGMENT_NAME\n{content}"`, 不问不改其他 handler 已追加内容。段间通过 priority 隐式排序, 不做显式依赖声明。

### 3.2 priority 顺序约定(数字越小越先执行)

| Priority | 段 | 内容 |
|---|---|---|
| 0–99 | 框架保留基础段 | persona, mode, phase, safety |
| 100–199 | 框架保留动态段 | metacog, tools, thinking |
| 200–299 | 插件段 | 合规声明, 领域安全规则 |
| 300+ | 保留 | 未来扩展 |

### 3.3 阻断语义

复用 `event.stop()`。高优先级 handler 可阻断后续低优先级 handler 追加。例如: safety 段判定请求违规时可 stop, 阻止模型看见其他段。

### 3.4 现有六段 → 插件映射

| 当前段 | 来源 | 成为插件 | Priority |
|---|---|---|---|
| PERSONA | `prompt_builder.py` | `PersonaPlugin`, 读 runtime persona | 10 |
| MODE | `prompt_builder.py` | `ModePlugin`, 读 mode 查 `_MODE_INSTRUCTIONS` | 20 |
| PHASE | `prompt_builder.py` | `PhasePlugin`, 读 phase 查 `PHASE_PROMPTS` + G51 notes | 30 |
| METACOG | `prompt_builder.py` | `MetacogPlugin`, S7 时注入 | 40 |
| TOOLS | `prompt_builder.py`(占位串) | `ToolsPlugin`, 真正按 mode/phase 过滤 | 50 |
| [PHASE] 转移 | `context.py` 尾部 | `PhaseTransitionPlugin` | 60 |
| External Thinking | `context.py` feature flag | `ThinkingPlugin`, 按 metacog/phase 判断 | 70 |
| SAFETY | `prompt_builder.py` | `SafetyPlugin`, 固定输出 | 90 |

### 3.5 External Thinking 迁移

现状: feature flag(默认关), 在 `context.py` 尾部条件拼接"先调 deep_think 再动手"指令。

插件化后:

```python
async def handler(self, event):
    if not FeatureFlags.shared().is_enabled("external_thinking"):
        return  # 默认关, 零开销
    # 或升级: S7 自修改态 / hypothesis|validate phase → 自动开
    segment = self._build_thinking_segment(event)
    if segment:
        event.system_prompt += "\n\n" + segment
```

好处: 不再散落在 context.py 做条件注入; 元认知状态可在 handler 里读; 第三方可替换内置注入策略。

### 3.6 迁移路径

```
Phase 0: 不动 — 硬编码, 零改动
Phase 1: 注册 — 建六个 PromptSegmentPlugin, 内容与现有六段相同, 注册但不切换,
         引入 _PLUGIN_PROMPT_ENABLED flag(默认关), 用 diff 测试保证内容一致
Phase 2: 切换 — flag 默认开, build_prompt() 改为 dispatch on_llm_request + 取
         event.system_prompt, 保留硬编码作 fallback
Phase 3: 插件化 — ThinkingPlugin 迁出 context.py, 公开基类文档
```

### 3.7 不改的事

- `HUGINN_SYSTEM_PROMPT` 仍是 persona 段核心内容
- `build_prompt()` 签名不变, 内部改 dispatch
- feature flag 机制不变, external_thinking 关闭行为不变
- 现有测试不修改(Phase 1 的 diff 测试是新增)

---

## 4. 形态 B: 策略注册表 — 其他子系统适配评估

### 4.1 Tool Result 压缩 — 高适配(优先做)

现状三层:
- 结构化分类 `ErrorKind`(`core_types.py`) — 基础, 非插件
- 输出压缩 `ToolOutputCompressor`(`compress.py`): `keep_keys`、数组摘要、LLM 中段摘要、磁盘卸载 — **策略分歧明显**
- 返回后钩子: 已有 `on_tool_respond`(可改 event.result)

插件化价值: 把"不同类型工具输出怎么处理"抽象成 `ToolResultCompressor` 插件, 按 tool_name 匹配。现在 `keep_keys` 是全局一组, 无法按工具差异化(DFT 保留 energy/band_gap, MD 数组摘要, 日志头尾截断)。正好补上现有代码标注的"按工具类型差异化 TTL/keep_keys"缺口。

### 4.2 Compaction — 中适配(策略注册表, 克制)

现状(`context.py`, `session.py`, `compress.py`):
- 已有策略分歧: drop-oldest vs LLM summarize, 选择已参数化
- 但保护规则散落: `tool_result_ttl`, `keep_keys`, `_THINKING_BLOCK_TYPES`, `root_content_markers`, `_PROTECTED_ROLES` 分布在各模块

插件化价值: 把"哪些消息受保护 / 哪些 key 保留 / 哪些块永不裁剪"抽象成 `CompactionPolicy` 插件。现在新增一种保护要改核心。

**警醒**: compaction 是性能敏感路径(`test_compact_messages_linear_performance` 要求 O(n)), 事件总线分发有开销。**不该**把每次压缩变成事件流, 应做成**可插拔的策略注册表**(Select a policy, not dispatch events)。

### 4.3 记忆整理(memory maintainer) — 高适配

策略分歧点极多: decay / prune / dedupe / cluster / compress, 各有阈值。第三方加"按重要性排序的淘汰策略"现在得改核心。适合策略注册表。

### 4.4 工具调度 / 错误恢复 / 沙箱选择 — 已有事件, 不必全量插件化

- 工具调度: 已有 `ToolCallEvent` 三段式钩子, 够用
- 错误恢复(autoheal): 已有 `ErrorKind` 二次分类, 适合策略注册表而非事件流
- 沙箱选择(Linux/MacOS/Windows): 策略分歧明显, 但跨进程边界, 插件化意义有限

---

## 5. 落地优先级

| 优先级 | 系统 | 形态 | 说明 |
|---|---|---|---|
| 1 | Prompt 段 | 事件钩子 | 已设计, 先做 |
| 2 | Tool Result 压缩 | 策略注册表 | 回报最高, 补 keep_keys 全局化缺口 |
| 3 | Compaction 策略 | 策略注册表 | 收敛散落保护规则 |
| 4 | 记忆整理 | 策略注册表 | 策略多, 收益相对低, 后置 |

---

## 6. Sheaf Gluing 视角(设计动机)

六段 prompt 不是孤立列表, 而是局部约束的 glue 问题。每段是"自己的触发条件域"上的局部解, 全局 system_prompt 要把 overlap 区域一致拼起来。段间优先级冲突就是 Čech H¹ 障碍。priority 仲裁 + `stop()` 就是消障碍的机制 — 这是把"硬编码 join"升级成"可仲裁的 gluing"的数学动机。