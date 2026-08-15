# Everything is a Plugin — 段插件化设计

> 状态: prompt 段 + tool result 压缩 + compaction 策略均已用形态 B 落地; 其余子系统评估中
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

| 形态 | 机制 | 适用场景 | 例子 |
|---|---|---|---|
| **形态 A: 事件钩子** | `@filter.on_xxx(priority=N)` 注册 async handler, 可改事件对象, 可 `event.stop()` 阻断 | 有生命周期、需要顺序 + 阻断、粒度到单次发生时 | 工具调用拦截、on_tool_respond |
| **形态 B: 策略注册表** | `dict[str, Callable]` + priority, 选一个策略执行, 无副作用事件流 | 性能敏感、纯策略选择、无嵌套副作用 | prompt 段、compaction policy、tool result 压缩、记忆整理、沙箱选择 |

**选择判据:**
1. 有生命周期钩子(发生在某时刻、可被阻断) → 形态 A
2. 只是"这条数据怎么处理"的多种策略选择 → 形态 B
3. 性能敏感路径(如 O(n) 压缩) → 形态 B 优先, 避免事件分发开销

> **决策记录 (prompt 段):** 原设计把 prompt 段归形态 A (事件钩子)。最终落地选定
> **形态 B (同步注册表)**。原因: `build_prompt()` 是主路径**同步**调用, 形态 A 的
> async handler 会把同步链改成异步分发, 引入协程/事件循环复杂度与风险; 而六段只是
> "不同段按顺序拼一段文本"的纯策略选择, 无阻断需求 (`stop()` 语义对 prompt 段无真实
> 价值)。故用 `StrategyRegistry` (plugins/strategy.py) + `PromptSegmentFn`
> (plugins/prompt_segments.py) 落地, register O(1)、assemble O(N) 纯拼接、零异步。

---

## 3. prompt 段插件化 (形态 B: 同步注册表, 已落地)

### 3.1 接口契约

```python
# plugins/prompt_segments.py
PromptSegmentFn = Callable[[str, str, str, str | None], str]
#                       (mode, phase, metacog_state, system_prompt) -> str  (空串 = 跳过)

register_prompt_segment(name, fn, priority=None)   # 注册/覆盖段
render_prompt_segment(name, mode, phase, metacog, sp)   # 单段渲染 (context.py 用)
assemble_prompt_segments(mode, phase, metacog, sp)      # 按 priority 升序拼全部
```

handler 只返回自己那段的文本, 不问不改其他段。段间通过 priority 隐式排序, 不做显式
依赖声明。同名注册覆盖内置段 (priority 高者 / 同 priority 后注册者生效)。

### 3.2 priority 顺序约定(数字越小越先拼)

| Priority | 段 | 内容 |
|---|---|---|
| 10 | persona | runtime persona / 内置最小 persona |
| 20 | mode | `_MODE_INSTRUCTIONS` |
| 30 | phase | `PHASE_PROMPTS` + G51 notes |
| 40 | metacog | S7 时注入 self-modify 指令 |
| 50 | tools | 占位串 (升级路径: 按 mode/phase 过滤) |
| 100 | thinking | external_thinking feature-flag 注入 (默认关) |
| 200 | safety | 固定输出, 最后 |

### 3.3 阻断语义(形态 B 不提供)

形态 B 无 `stop()`。prompt 段无真实阻断需求 — 若未来需要"某段违规则整个 system_prompt
不产出", 应升级回形态 A 或用前置校验。当前不引入该复杂度。

### 3.4 现有六段 → 插件映射

| 当前段 | 来源 | 成为插件 | Priority |
|---|---|---|---|
| PERSONA | `prompt_builder.py` | `_persona_plugin` | 10 |
| MODE | `prompt_builder.py` | `_mode_plugin` | 20 |
| PHASE | `prompt_builder.py` | `_phase_plugin` | 30 |
| METACOG | `prompt_builder.py` | `_metacog_plugin` | 40 |
| TOOLS | `prompt_builder.py`(占位串) | `_tools_plugin` | 50 |
| External Thinking | `context.py` feature flag | `_thinking_plugin` | 100 |
| SAFETY | `prompt_builder.py` | `_safety_plugin` | 200 |

注册发生在 `prompt_builder` 模块 import 时 (`_register_builtin_segments()`)。
`context.py` 主路径不再硬编码 external_thinking, 改为 `render_prompt_segment("thinking", ...)`
(详见 §3.5), 并在调用点先 import `prompt_builder` 确保段已注册。

### 3.5 External Thinking 迁移

现状: feature flag(默认关), 原在 `context.py` 尾部条件拼接"先调 deep_think 再动手"指令。

插件化后 (已落地):

```python
def _thinking_plugin(mode, phase, metacog_state, system_prompt):
    # 默认关, 零开销; flag 层异常 fail-open 返回空串
    if not FeatureFlags.shared().is_enabled("external_thinking"):
        return ""
    return "## External Thinking\nBefore you answer, modify code, or call other tools, ..."
```

`context.py` 主路径改为调用 `render_prompt_segment("thinking", ...)`, 行为与旧硬编码
一致 (flag 开 → 注入, 关/异常 → 不注入), 但逻辑单一来源、可被第三方替换。

### 3.6 迁移路径 (已完成)

```
Phase 0: 不动 — 硬编码, 零改动
Phase 1: 注册 — 建六个 PromptSegmentPlugin, 内容与现有六段相同, 用 diff 测试
         (tests/test_prompt_segments.py) 保证内容一致
Phase 2: 切换 — build_prompt() 改为 assemble_prompt_segments(), 保留硬编码作 fallback
Phase 3: 插件化 — external_thinking 从 context.py 迁为 _thinking_plugin
```

### 3.7 不改的事

- `HUGINN_SYSTEM_PROMPT` 仍是 persona 段核心内容
- `build_prompt()` 签名不变, 内部改走注册表
- feature flag 机制不变, external_thinking 关闭行为不变
- 既有测试不修改(Phase 1 的 diff 测试是新增)

---

## 4. 形态 B: 策略注册表 — 其他子系统适配评估

### 4.1 Tool Result 压缩 — 高适配(优先做)

现状三层:
- 结构化分类 `ErrorKind`(`core_types.py`) — 基础, 非插件
- 输出压缩 `ToolOutputCompressor`(`compress.py`): `keep_keys`、数组摘要、LLM 中段摘要、磁盘卸载 — **策略分歧明显**
- 返回后钩子: 已有 `on_tool_respond`(可改 event.result)

插件化价值: 把"不同类型工具输出怎么处理"抽象成 `ToolResultCompressor` 插件, 按 tool_name 匹配。现在 `keep_keys` 是全局一组, 无法按工具差异化(DFT 保留 energy/band_gap, MD 数组摘要, 日志头尾截断)。正好补上现有代码标注的"按工具类型差异化 TTL/keep_keys"缺口。

### 4.2 Compaction — 已落地 (策略注册表)

现状(`context.py`, `session.py`, `compress.py`):
- 已有策略分歧: drop-oldest vs LLM summarize, 选择已参数化
- 但保护规则散落: `tool_result_ttl`, `keep_keys`, `_THINKING_BLOCK_TYPES`, `root_content_markers`, `_PROTECTED_ROLES` 分布在各模块

插件化落地 (已做): 新建 `plugins/compaction_policy.py`, 声明式 `CompactionPolicy`
(protected_roles / never_trim_block_types / root_content_markers), 注册表聚合并集。
`compact_messages` 与 `summarize_compact_messages` 改从注册表取保护集, 不再硬编码
`_PROTECTED_ROLES` / `_THINKING_BLOCK_TYPES`。新增一种保护 → 注册一个策略, 不改核心。

**性能**: compaction 是 O(n) 路径, 用同步注册表 (形态 B) 而非事件总线。聚合是
O(P) (P = 策略数, 个位), 相对 O(n) 可忽略; 不做事件分发、无 async 模型切换。

**警醒**: 事件总线分发有开销, 不该把每次压缩变成事件流 — 用可插拔策略注册表
(Select a policy, not dispatch events)。

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
| 1 | Prompt 段 | 形态 B | ✅ 已落地 |
| 2 | Tool Result 压缩 | 策略注册表 | ✅ 已落地 |
| 3 | Compaction 策略 | 策略注册表 | ✅ 已落地 |
| 4 | 记忆整理 | 策略注册表 | 策略多, 收益相对低, 后置 |

---

## 6. Sheaf Gluing 视角(设计动机)

六段 prompt 不是孤立列表, 而是局部约束的 glue 问题。每段是"自己的触发条件域"上的局部解, 全局 system_prompt 要把 overlap 区域一致拼起来。段间优先级冲突就是 Čech H¹ 障碍。priority 仲裁 + `stop()` 就是消障碍的机制 — 这是把"硬编码 join"升级成"可仲裁的 gluing"的数学动机。