# 模型面即产物（Model Surface as Artifact）

日期：2026-11-01
worktree: 否 —— 改的是主路径组装与表示层，单线程执行
前置：`docs/prompt-output-surface-spec.md`（2026-10-16，段级「声明=生效」已落地）

## A 问题与根因

症状：机制（技能/工具/插件/工作流/门/checkpoint）注册了，模型不知道它们存在、也不会主动调用。
不是缺机制，是缺「注册面 → 模型可见面」的表示与 owner。

根因（可证伪的三条代码事实）：

1. `assemble_prompt_segments` / `render_prompt_segments_for_path` 返回 `str`，
   `"\n\n".join(parts)` 处段名与来源丢失。
2. 主路径 `_effective_system_prompt` 把 9 个阶段用字符串 `+=` / f-string 串起来
   （`huginn/agent/context.py:270-285`），发送前只是一根字符串。
3. 插件可在事件里直接改字符串（`huginn/plugins/ponytail/main.py` 的
   `inject_rules`，行号为当时快照；该死钩子已在本批 §F 删除）。

推论：模型面没有对象，因此（a）无法断言「发送了什么」，门只能断言生产者的声明；
（b）每加一个机制 = 多一个字符串生产者，默认不可见；（c）同日 11 个组装根各自拼接，
无 owner。2026-10-16 的定律只延伸到「段」这一层，缺陷于是向上迁移到「注册物」这一层。

## B 表示层

`huginn/agent/model_surface.py`（**零 huginn 依赖，只 import dataclasses**）：

- `SurfaceSection(name, source, text)`；`source` 必填且非空（来源=注册表名/模块名）。
- `Surface(sections)`：`render()` 以 `"\n\n"` 连接非空段；`names()`、`sources()`、
  `char_count()`、`text_of(name)`。
- `compose(sections) -> Surface`：段名重复、name/source 为空 → `ValueError`。
  不静默去重：同名两段是接线错误，必须红。

invariant：同一组连续函数调用不变（纯函数，无全局状态），`render()` 结果与
逐段 `"\n\n".join` 等价。

## C 主路径迁移

`huginn/agent/context.py` 的 `_effective_system_prompt` 改为：
`_effective_surface() -> Surface`，由它登记每段 `(name, source)`，再 `render()`。

- 各阶段函数从「接收 base 并原地追加」改为「返回自己的 fragment」，拼接权收归 `Surface`。
- 迁移必须**字节不变**：迁移前基线 `default=10630/sha e60431842920cce7`、
  `research=10899/7a2ae63600207b84`、`extreme=11008/50fa17409ae2de3e`（stub context、
  `workspace=/tmp`、`system_prompt="base prompt"`）。
- CSM 命中时仍整体替换：登记为单段 `("csm","prompt_builder")`，不伪造子段。
- 迁移完成即**删除**旧字符串拼装路径，不留兼容分支。

failure modes：任何阶段抛异常都必须仍被该阶段原来的 `try/except` 隔离，
`Surface` 只做拼接，不引入新的异常源。

## D 披露面与门

`huginn/agent/mechanism_census.py`（stdlib；Catalog 在函数内懒 import）：

- `KIND_DISCLOSURE`：`kind -> "always" | "on_demand" | "hidden"<理由>`，覆盖
  `huginn.catalog.models.KINDS` 全部 7 类；缺声明 = 门红（二态法则的细化）。
- `census_text()` 生成 `## MECHANISMS` 段：小集合（skill/plugin/workflow）列名；
  大集合（tool）只给数量 + 「不在可见列表里就调 `tool_search`」；prompt/model/mcp 按声明。
- 披露做成 **prompt 段**（`register_prompt_segment("mechanisms", …, applies_to=("main","s7","autoloop"))`），
  而不是主路径私有的一次性注入：三条生产路径共用同一实现，autoloop / s7 自动获得披露。
  第 4 条生产路径 CodeAct 见 §F，直接消费同一 `census_text()`。
- `census_text()` **永不返回空串**：无条目或 Catalog 不可用时如实告知，
  否则「声明该段生效」与「实际渲染为空」对不上，披露会退化成静默。

门 `tests/test_model_surface_contract.py`（纯 stdlib 部分可离线跑）：

- Surface 语义：顺序、来源、空段跳过、重复名抛错。
- 结构不变量：`compose_main_surface` 内**不得**出现 `base +=` 字符串拼接（ast 扫描）；
  必须调用 `compose(`，`_effective_system_prompt` 必须调用 `render()`。
- 披露双向：`KIND_DISCLOSURE` 覆盖 KINDS；`hidden` 必须带理由；`always` 的 kind
  名称必须出现在 `census_text()` 里。
- 运行时（`importorskip("langchain.agents.middleware")`）：stub context 的真实
  Surface 必须含 `## MECHANISMS`，且每段 source 都在登记集合内。

## E 收编台账

其余组装根（autoloop / CodeAct / Coder / 子代理 / 澄清 / 中间件 / 流式 / 插件回调）
不在首批迁移中，逐个登记在门内的 `_LEGACY_SURFACE_ROOTS`（附理由），
按「构造 `SystemMessage(content=...)` 且**未**用 `compose()` 组装」的文件集合做双向精确核对：
新增字符串根 → 红；迁移或删除后未摘除 → 红。首批只迁 1 个根（main）；
第二批迁 CodeAct（见 §F），台账 15 → 14。

## F 第二批迁移与顽疾核查

CodeAct（第 4 条生产路径）并入同一表示层：

- `_build_surface(tools) -> Surface`，段为 `codeact / atomworld / cognitive_map / mechanisms`；
  `_build_system_prompt` 只调 `render()`，两个后缀函数不再自带 `\n\n`（分隔符归 Surface）。
- `census_text()` 作为 `mechanisms` 段接入 CodeAct，「注册面 → 模型可见面」覆盖
  main / s7 / autoloop / codeact 四条路径；新门 `test_codeact_path_discloses_mechanisms`。
- 字节不变：剔除新增 mechanisms 段后，sha256 与迁移前一致（后缀关 802/a4ca1cf93cbea693，
  后缀开 1852/b0d5d717b2ca6bf5）。
- 台账语义收紧为「构造 SystemMessage 且**未**用 `compose()` 组装」；15 → 14。
  `test_surface_backed_roots_render` 兜住「调用 compose 却不 render」的假迁移。

顽疾核查结论（同族缺陷是否还在别处）：

- 表示层已覆盖四条生产路径；剩余 14 个字符串根中，12 个是追加 hints 或子 LLM 窄任务
  （不承载能力披露），2 个是真正独立面（autoloop engine_act / coder loop）。
- **hook 层的同类缺口（已按行为中性处置）**：`api/filter.on_llm_request` 没有任何派发者构造
  `LLMRequestEvent`；`unified_bus.publish_llm_request` 发的是 base Event（只带
  `messages_count`），`plugins/ponytail.inject_rules` 读 `event.system_prompt` 会抛
  AttributeError（best-effort 吞掉）；即便读到，改的也只是一次性事件对象，无人回写。
  ⇒ 「声明了注入、实际从未生效」在插件钩子层成立过。
- 处置（用户决策 = 删除死钩子，行为中性）：删除 `inject_rules`、`LLMRequestEvent` import、
  `_RULES_*` 常量与 `inject_system_prompt` 配置承诺；`_MODE_MAP` 收敛为合法模式集合
  `_VALID_MODES`（`/ponytail` 校验仍用）。测试删 `TestLLMInjection`，handler 集合断言改为
  只含 `ON_MESSAGE_RECEIVED`。规则文本仍由 `SKILL.md` 经 skill loader 到达模型，prompt
  字节不变。通用 `on_llm_request` 装饰器 API 未接线但保留，记录在 deferred。

## failure modes

- 迁移后字节不一致 → 回退本次迁移，不改基线；不得靠改测试让步。
- `catalog` 不可用时 `census_text()` 仍返回非空说明段（不是空串），门可无条件断言
  `## MECHANISMS` 存在；若哪天退化成空串，门会红。
- 门的 stdlib 部分不得 import langchain/pydantic；运行时部分必须 `importorskip`。

## 非目标

- 不重写任何 prompt 文案（writing/safety/multimodal 等文本一字不动）。
- 不把全部组装根一次性统一；不新增第二套 prompt 装配器。
- 不保证模型遵守披露内容；只保证「系统知道自己告诉了模型什么」。
- 不改 autoloop 的插件段消费语义。
- 不接线 `on_llm_request` 钩子：唯一的死 handler 已删除（见 §F），装饰器 API 保留但无派发者。

## deferred

- 治理台账侧再收一层：把组装根清单（当前在门内）提升到 `governance_audit`
  （需与 `governance-state.md` 一起重生成，单独批次）。
- 其余 14 个字符串根逐批迁移到 `compose`；子 LLM 窄任务可长期留在台账（理由已写明）。
- `on_llm_request` 装饰器 API 仍无派发者（`EventType.ON_LLM_REQUEST` 仍被 publish，
  仅观测用途）；若将来要让插件参与 prompt 注入，须先让 bus 构造 `LLMRequestEvent`
  并把改动回写进组装面，否则还是声明面 ≠ 生效面。
- `CatalogEntry.disclosure` 字段（把披露等级从 kind 级细化到条目级）。

## 结果（as-built）

- 表示层：`huginn/agent/model_surface.py`（零 huginn 依赖）；主路径
  `compose_main_surface(ctx) -> Surface`，`_effective_system_prompt` 只调 `render()`。
- 字节不变：迁移后 default / research / extreme 三模式 prompt 的 sha256 与迁移前基线
  逐字相同（10630 / e60431842920cce7、10899 / 7a2ae63600207b84、
  11008 / 50fa17409ae2de3e）；剔除新增 `mechanisms` 段后哈希仍为 e60431842920cce7。
- 披露：`mechanisms` 段注册为内置段，`applies_to=("main","s7","autoloop")`；实测三条路径
  的 prompt 均含 `## MECHANISMS`，内容为 56 个 skill、15 个 workflow 逐条 + tool
  的 `tool_search` 入口；主路径 10630 → 11472 字符，census 热态 ~0.5ms
  （1.2s 级开销来自既有 `capability_note` 的 L0 探测，非本次引入）。
  CodeAct 另经 `_build_surface` 消费同一 `census_text()`：802 → 1663 字符。
- 台账：`_LEGACY_SURFACE_ROOTS` 14 个字符串根双向精确核对（CodeAct 已迁出）；
  `PROMPT_SPECS` 13 → 14 段、`prompt_segment_floor` 13 → 14。
- 门：`tests/test_model_surface_contract.py` 16 项；本轮相关集合 45 项、治理/文档门
  全绿（`env-contract.md`、`feature-flags-contract.md` 因行号位移已重生成）。
- hook 层：ponytail 的 `inject_rules` / `inject_system_prompt` 已删除（T7）；
  `docs/plugins-contract.md` 重生成后事件钩子表为空，Prompt 段表补回 `mechanisms`（priority 45）。
  `docs/pluginized-segments-design.md` 的「prompt 段是 on_llm_request handler」错误陈述已改为
  「形态 B 策略注册表 + `render_prompt_segments_for_path`」。
