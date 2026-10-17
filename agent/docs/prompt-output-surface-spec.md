# Prompt / 输出面「声明 vs 生效」缺口 — 系统修复 Spec

> Status: ACTIVE (as-built, 详见 §14)
> Source: 对话审计 (输出规范梳理 + 「声明面↔生效面」一致性排查)
> Date: 2026-10-16
> Scope: 单 spec; 落地由 writing-plans 承接

## 1. 背景与问题

输出规范与 prompt 段存在一条**「声明面 ≠ 生效面」**的缺口族：段 / 事件 / 规范被
**注册或声明**了，但在实际会话路径上**从不生效**，且**无任何门禁**把这种"假接线"
变成红灯。

仓库在治理层已用「孤儿门 / 悬空门 / 幻影消费者门 / 触发无人接」这套词汇建模同类
缺陷（见 `huginn/cli/governance_audit.py` 模块 docstring 与 `docs/mece-audit.md`
的十七面），但**该判据尚未覆盖 prompt 段与钩子事件的"生效面"**。于是 prompt 面成了
治理层自身的熵。

### 已确认缺口（证据）

| 编号 | 缺口 | 证据（file:line） |
|---|---|---|
| G-1 | **主路径只渲染 `thinking` 段**；`writing` / `multimodal` / `safety` 及**全部插件段**在主路径不生效 | `huginn/agent/context.py:96-106`（仅 `render_prompt_segment("thinking", …)`）；`huginn/agent/context.py:191-204`（仅 `metacog == s7_self_modify` 才委托 `build_prompt`） |
| G-2 | 反防御写作规范（`writing_segment`）在正常会话缺席 | `huginn/agent/prompt_builder.py:202-217`；只在 `build_prompt` 组装的 s7 路径生效 |
| G-3 | 第三方插件段在主路径永不生效 | `huginn/plugins/asd_ste100/main.py:310` 注册段（priority 60）；主路径不消费；仅 s7 / autoloop 消费 |
| G-4 | 规范文本重复来源，且多数无同步门 | safety：`huginn/prompts.py:193-196` vs `huginn/agent/prompt_builder.py:167-176`；mode 规范：`huginn/agent/context.py:60-81` vs `prompt_builder._MODE_INSTRUCTIONS`（`huginn/cli/config_audit.py:734-740`） |
| G-5 | 钩子事件存在"声明但无触发 / 无订阅" | `SUBAGENT_STOP` 定义于 `huginn/hooks/__init__.py:37` 并列入 `ALL_EVENTS:50`，但全仓未见触发点与订阅者 |
| G-6 | 输出规范本身零门禁 | `tests/test_prompt_segments.py` 只测装配行为，不断言"每段是否在某生产路径渲染"，也无重复文本 / 必需段检查 |

### 正面参照（已有"同步门"的正例，本 spec 复用其范式）

`MATH_DEPTH_GUIDE`（`huginn/prompts.py:199-246`）与 `engine._MATH_DEPTH_PROMPT_BLOCK`
（`huginn/autoloop/engine.py:284-288`）的重复源，由
`tests/test_prompts_math_depth.py::TestEnginePromptConsistency` 守住 —— 证明"重复源 +
同步门"在本仓库可行。

## 2. 目标

1. **消除静默失效**：每个 prompt 段 / 钩子事件必须**要么生效、要么显式声明"不该在这里生效"**
   （可见决策）—— 二者其一，无第三种状态。
2. **单一来源**：同名规范文本不得两处实现；保留重复的必须加同步门。
3. **可观察 + 可门禁**：生效面入治理台账（棘轮）+ 新增 `tests/test_prompt_contract.py`。
4. **不改变既有行为**：本 spec **不强行**把 s7-only 段搬到主路径；先让失效**可见**，再按需统一。

## 3. 非目标 (YAGNI)

- 不引入模板 DSL，不新建第二套 prompt 装配器。
- 不对自由文本强加 JSON schema（会伤科研表达）。
- 不强制统一三路径的段集合（先可见决策，再评估是否统一）。
- 不重写 reasoning / 写作段的内容本身。

## 4. 架构与边界

三面判据（对齐治理层既有的"存在 / 触发 / 登记"三段）：

```
声明面 (registered / declared)
   ├─ register_prompt_segment(...) / ALL_EVENTS 事件常量
生效面 (rendered / triggered)
   ├─ 生产路径实际渲染点 / 实际触发点
可观察面 (台账 / 契约)
   └─ docs/governance-state.md + tests/test_prompt_contract.py
```

引入**段生效声明**作为单一事实来源：每个段名声明它应在哪些路径生效
`applies_to: {main, s7, autoloop}`；门禁断言"**声明 == 实际渲染**"。

## 5. 组件与改动

| 单元 | 类型 | 职责 |
|---|---|---|
| `huginn/plugins/prompt_segments.py` | 改 | 段元数据携带 `applies_to` 声明；提供 `segment_applicability()` 查询 |
| `huginn/agent/output_contract.py`（或注册 `output` 段） | ~~新增~~ | **未落地**：输出规范单一来源复用既有段 (`writing`/`safety`/`multimodal`)，不新建独立模块 (YAGNI，见 §14) |
| `huginn/agent/context.py` | 改 | 主路径按 `applies_to` 渲染应生效的段（不再只渲染 `thinking`） |
| `huginn/cli/governance_audit.py` | 改 | 新增 `PROMPT_SPECS` + `prompt_segment_floor`（weightless：只登记段名与数量） |
| `tests/test_prompt_contract.py` | 新增 | 生效面门禁：声明齐全 + 声明=渲染 + 必需段 floor + 重复文本检测 |
| `tests/test_hook_event_coverage.py` | ~~新增~~ | **未落地**：钩子事件面已由 `tests/test_contract_audit.py` 的 `build_hook_contract` 全量覆盖 (单一来源，避免重复门，见 §14) |
| 重复源统一 | 改 | safety / mode 规范收成单一来源，或加同步门（照 MATH_DEPTH 范式） |

## 6. 关键接口

```python
# huginn/plugins/prompt_segments.py

def segment_applicability() -> dict[str, frozenset[str]]:
    """name -> 生效路径集合, 子集 ∈ {"main","s7","autoloop"}.

    未声明的 registered 段 → 门禁红;
    声明集合与任一生产路径的实际渲染集合不符 → 门禁红。
    """
```

```python
# huginn/agent/output_contract.py —— 输出规范单一来源

def output_contract(mode: str) -> str:
    """按 mode 返回输出规范段 (基础规则 + mode 加性差异), 与 §7/writing/safety 同源。"""
```

## 7. 数据流与生效时序

1. **注册段时**声明 `applies_to`。
2. **各路径按声明渲染**段：main 路径不再只渲染 `thinking`。
3. **入账**：`governance_audit` 收段名 + 数量（floor 棘轮）。
4. **门禁**：`test_prompt_contract` 断言声明 ↔ 渲染一致；改路径不改声明 → 红。
5. **钩子**：`ALL_EVENTS` 每项在 `test_hook_event_coverage` 中登记触发点或豁免。

## 8. 错误处理与诚实边界

- 段渲染异常沿用现有隔离（跳过该段，组装永不崩）。
- **生效声明缺失 = 红**，不是静默跳过。
- 只陈述"该段在该路径**生效**"，**不**暗示"该规范一定被模型**遵守**" —— prompt 是
  引导，非保证（与 capability note 的诚实边界同调）。

## 9. 测试策略

- **声明齐全性**：每个 `registered_prompt_segments()` 项都有 `applies_to`。
- **声明=渲染**：对 main / s7 / autoloop 三路径分别断言"实际渲染集合 == 声明集合"。
- **必需段 floor**：关键段（`output` / `writing` / `safety` / `multimodal`）不得从声明中消失。
- **重复文本检测**：同义规范词条不得两处；白名单例外须登记（可见决策）。
- **钩子事件**：`ALL_EVENTS` 每项有触发点或豁免登记（`SUBAGENT_STOP` 二选一）。
- **weightless**：全部不 import 重栈（不拉 langchain/pydantic）。

## 10. 治理集成

- `governance_audit.py`：新增 `PROMPT_SPECS` 与 `prompt_segment_floor`（weightless：
  只登记清单与数量，不真渲染）；再生成 `docs/governance-state.md` 新增「prompt 段生效面」节。
- `tests/test_arch_gate_registry.py`：新增冻结棘轮 —— 段清单 / 数量不得静默缩水。

## 11. 验收标准 (AC)

- **AC-1** 每个 registered prompt 段都有 `applies_to` 声明；缺声明 → 红。
- **AC-2** main / s7 / autoloop 三路径的**实际渲染段集合**与**声明集合**逐路径相等；不等 → 红。
- **AC-3** `writing` / `output` / `safety` / `multimodal` 在主路径**按声明生效**（或显式声明
  不生效），不再是静默孤儿。
- **AC-4** 第三方插件段（如 `asd_ste100`）的生效路径被显式声明，并可被门禁核对。
- **AC-5** 重复规范源（safety / mode）要么收成单一来源，要么有同步门。
- **AC-6** `ALL_EVENTS` 每项有触发点或显式豁免登记；`SUBAGENT_STOP` 二者择一。
- **AC-7** `governance-state.md` 新增「prompt 段生效面」节，与 `render_markdown` 输出逐字相等。
- **AC-8** 段清单 / 数量下降 → 红（棘轮）。

## 12. 风险与权衡

- **行为变化风险**：让主路径渲染更多段会改变 prompt → 以"先声明、按声明生效、可回退"
  降低风险；关键段变更需基准对照。
- **prompt 膨胀**：段增多需 token 预算断言（与输出规范单一来源合并处理）。
- **误判**：某段可能**有意**只在 s7 生效；本 spec 用"显式声明"容纳，不强制统一。
- **钩子豁免滥用**：豁免须登记为可见决策，不得静默。

## 13. 后续（超出本 spec）

- 在声明稳定后，评估是否**统一三路径的段集合**（若仍觉分裂）。
- 把"生效面"审计扩展到 tools / feature-flag 面（与
  `docs/capabilities-and-registration-spec.md` 对齐）。

## 14. 落地结果 (as-built, 2026-10-16)

已落地 (对应 AC)：

- **AC-1/AC-2/AC-3** — `prompt_segments._APPLIES_TO` 段生效声明 + `segment_applicability()`
  查询；`render_prompt_segments_for_path(path, …)` 按声明过滤渲染；`context.py` 主路径改用
  它 (不再硬编码只渲染 `thinking`) → `writing` / `multimodal` / `thinking` 在主路径**按声明生效**。
  门禁 `tests/test_prompt_contract.py` 逐路径断言「声明 == 实际渲染集合」。
- **AC-4** — 第三方插件段 (`asd_ste100`) 在 `register_prompt_segment` 显式声明
  `applies_to=("main","s7","autoloop")`；门禁扫描全仓 `register_prompt_segment` 调用，
  非内置段缺 `applies_to` → 红。
- **AC-5** — safety 规范两份来源 (`prompts.py` §8 ↔ `prompt_builder.safety_segment`)
  由**同步门**守住共同语义锚点 (照 MATH_DEPTH 范式，非逐字相等)。
- **AC-6** — 钩子事件面沿用既有 `tests/test_contract_audit.py` 的 `build_hook_contract`
  (逐事件判 wired / trigger-only / dead)，**未**新建重复门 (单一来源)。
- **AC-7** — `docs/governance-state.md` 新增「prompt 段生效面」节 (由 `render_markdown`
  生成，逐字一致)。
- **AC-8** — 台账 `PROMPT_SPECS` + `BUDGET_SPECS["prompt_segment_floor"]` (下限 13; 初版 9,
  后随 §3/§4/§7 拆为独立段 `core_principles`/`behavioral_discipline`/`response_format` 上调)；
  `tests/test_arch_gate_registry.py` 的段数棘轮 + 清单↔源一致门兜底。

与初版设计的**偏离** (诚实记录，均为减配而非缩水)：

- **未新建 `huginn/agent/output_contract.py` / `output` 段**：本仓输出规范已分段承载
  (`writing` / `safety` / `multimodal`)，另立 §7 单一来源模块会与其重复，违反本 spec
  自身的"单一来源"目标。关键段 floor 覆盖 `writing`/`safety`/`multimodal`/`thinking`。
- **未新建 `tests/test_hook_event_coverage.py`**：钩子面「声明↔触发↔注册」已由
  `test_contract_audit` 的 `build_hook_contract` 全量覆盖，重复门只增维护面。
- **未强制统一三路径段集合**：按 §3 非目标，先让失效可见、按声明生效，统一留待 §13。

