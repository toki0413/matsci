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
3. 插件可在事件里直接改字符串（`huginn/plugins/ponytail/main.py:189-191`）。

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
**不在本次迁移**，逐个登记在门内的 `_LEGACY_SURFACE_ROOTS`（15 个文件，附理由），
按 `SystemMessage(content=...)` 出现的**文件集合**做双向精确核对：新增根 → 红；
迁移或删除后未摘除 → 红。本 spec 只迁 1 个根（main），以证明表示层可承载真实字节。
autoloop 虽仍以自己的字符串拼装，但**披露**已通过 `mechanisms` 段覆盖（见 §D）。

## failure modes

- 迁移后字节不一致 → 回退本次迁移，不改基线；不得靠改测试让步。
- `catalog` 段在无 Catalog 环境为空 → 门只在 `entries > 0` 时断言存在，避免假红。
- 门的 stdlib 部分不得 import langchain/pydantic；运行时部分必须 `importorskip`。

## 非目标

- 不重写任何 prompt 文案（writing/safety/multimodal 等文本一字不动）。
- 不把 11 个组装根一次性统一；不新增第二套 prompt 装配器。
- 不保证模型遵守披露内容；只保证「系统知道自己告诉了模型什么」。
- 不改 autoloop 的插件段消费语义。

## deferred

- 治理台账侧再收一层：把组装根清单（当前在门内）提升到 `governance_audit`
  （需与 `governance-state.md` 一起重生成，单独批次）。
- 其余 10 个组装根迁移到 `compose`，逐批把文件从 `_LEGACY_SURFACE_ROOTS` 摘除。
- `CatalogEntry.disclosure` 字段（把披露等级从 kind 级细化到条目级）。

## 结果（as-built）

- 表示层：`huginn/agent/model_surface.py`（零 huginn 依赖）；主路径
  `compose_main_surface(ctx) -> Surface`，`_effective_system_prompt` 只调 `render()`。
- 字节不变：迁移后 default / research / extreme 三模式 prompt 的 sha256 与迁移前基线
  逐字相同（10630 / e60431842920cce7、10899 / 7a2ae63600207b84、
  11008 / 50fa17409ae2de3e）；剔除新增 `mechanisms` 段后哈希仍为 e60431842920cce7。
- 披露：`mechanisms` 段注册为内置段，`applies_to=("main","s7","autoloop")`；实测三条路径
  的 prompt 均含 `## MECHANISMS`，内容为 56 个 skill、15 个 workflow 逐条 + 161 个 tool
  的 `tool_search` 入口；主路径 10630 → 11472 字符，census 热态 ~0.5ms
  （1.2s 级开销来自既有 `capability_note` 的 L0 探测，非本次引入）。
- 台账：组装根以门内 `_LEGACY_SURFACE_ROOTS`（15 个文件）双向精确核对；`PROMPT_SPECS`
  13 → 14 段、`prompt_segment_floor` 13 → 14，`docs/governance-state.md` 已重生成
  （差异仅 mechanisms 行与下限值）。
- 门：`tests/test_model_surface_contract.py` 15 项、治理与流程/卫生门共 116 项通过。
