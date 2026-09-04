# 工业 Skill 接入指南（对齐"工业智能体开发套件"范式）

> 状态：active
> 一句话：**不接西门子套件本体**，把「Skill Creator · Agent Framework · Workflow · ICX」这套
> 工业智能体范式映射到 Huginn 现有体系，用我们已有的 Skill/工具/MCP 体系封装 OT 工程
> Know-how，得到可复用的工业 Skill。示例与字段均与代码解析器对齐，防契约漂移。
> **字段级规范见**：[industrial-skill-metadata-spec.md](industrial-skill-metadata-spec.md)（外部接收时先读它定义字段）。

---

## 1. 为什么/何时用本指南

Huginn 面向物理/材料/仿真，本身已具备完整的"技能·工具·工作流"体系。西门子 Xcelerator
工业智能体开发套件（Skill Creator / Agent Framework / Workflow / ICX）本质是把**工程
Know-how 封装成 Agent 可调用 Skill + 编排成工作流 + 可追溯执行**。这套范式我们每一种
能力都已具备，无需绑定西门子托管生态；要做的是**用统一规范封装工业/OT 技能**，并明确
它们如何落进 Huginn。

适用场景：要在产线上把重复性工程知识（数据采集、校验、配置、报告）做成 agent 可
一键调用的能力；或对接具体工业后端（PLC/TIA、SCADA、MES、OPC-UA/MQTT）时，用
本规范的 Skill 作为能力封装层。

## 2. 范式对照

| 西门子工业智能体开发套件 | Huginn 对应 | 说明 |
|---|---|---|
| Skill Creator（把 Know-how 封装为 Skill） | `SkillDefinition` / `SkillRegistry` / `SkillImporter` + `SKILL.md` | 声明式技能或指令式技能 |
| Agent Framework（智能体编排） | `subagent_tool` / `orchestrator` / cognitive engine | 多步骤编排与协同 |
| Workflow（多步骤工作流） | `SkillStep`（含 `condition`/`loop_until`/`parallel_group`）+ `workflows/*` | 确定性、可并行、可校验 |
| ICX（AI 调度总台、数据/决策追踪） | `ToolScheduler` + `control_safety` 策略清单 + `deep_think` 推理刻录 | 信任边界、仲裁、可追溯 |
| OT 数据接入（PLC/边缘/数据采集） | 外部计算工具适配器 / MCP 桥（`eco_tool.mcp_connect`） | 不硬编码协议 |

## 3. 两条建模路径（关键取舍）

### 3a. 声明式：`SkillDefinition`（确定性 / 强校验）
适合"输入固定、手段固定、要可校验/可回滚"的工业动作（设备批量配置、标准校验、
参数下发）。用结构化字段：

- `parameters`：对外参数（类型/必填/默认）
- `steps`：一串工具调用，支持 `validation` 表达式、`retry`/`on_failure`、
  `condition`（if 门）、`loop_until`（带 `loop_max_iterations` 上限）、
  `parallel_group`（同组并行，如"N 个传感同时就绪才合成结论"）
- `required_tools` / `required_env_vars`：依赖声明（配合 `control_safety` 策略清单）
- `domain` / `stage` / `function` / `parent`：行业语义索引，支撑技能树查询

由 `DeclarativeSkillExecutor` 结构化执行。**这是对齐西门子"确定性、面向闭环执行"主张的本体。**

### 3b. 指令式：`SKILL.md`（自然语言 / 自由链路）
适合"正文即提示词、靠 LLM 理解与编排"的成熟 Know-how（诊断、分析、解读），例如仓库内
`instrument-ingest`、`workflow_skill_creator`。frontmatter 给模型可用信息，正文是
LLM 直接执行的步骤规范。

### 3c. 外部后端桥（OT/PLC/第三方）
任何 Skill 的步骤工具最终可落到：
- 本地已注册工具（进程内 `ToolRegistry`）
- 外部进程/远程作业（`security/compute_adapter.py`，复用安全与生命周期）
- MCP server 能力（`eco_tool.mcp_connect` → `mcp_adapter.register_mcp_tools`，动态写回注册表）
- 计算后端（Container / Sandbox 执行器，走 `command_filter` 白名单）

## 4. frontmatter 字段（唯一事实源，防漂移）

`SkillImporter` 认识的**原生**字段只有这些，工业 Skill 请只用它们，不要新增私有字段：

- `name`、`description`、`category`（缺省 `general`）
- `steps`（描述性字符串或结构化列表）、`allowed-tools`（=`required_tools`）
- `when_to_use`（`metadata.when_to_use`）
- `tags`、`paths`、`model`、`effort`

三个平台自动识别：`trigger_conditions`→OpenClaw；`trigger`→Hermes；`allowed-tools`→Huginn 原生。
行业语义（domain/stage/function/parent）在声明式 `SkillDefinition` 里表达，不塞进 SKILL.md frontmatter。

> 防漂移护栏：本指南的示例已落盘为真实 `SKILL.md`，并由
> `tests/test_industrial_skill_guide.py` 用 `SkillImporter` 实测解析——若字段或格式漂移，
> 测试即红。

## 5. 端到端生命周期

1. **开发**：写 `SKILL.md`（原生 frontmatter）或声明 `SkillDefinition`；外部动作配 `scripts/` 保真实现
2. **导入**：`eco_tool.skill_install`（本地路径 / URL / `github://`）或 `SkillImporter.import_file/import_directory`
3. **注册**：`SkillRegistry.register(...)`，随后可被 agent 按名调用
4. **执行**：声明式走 `DeclarativeSkillExecutor`；指令式由 LLM 按正文执行；步骤落本地工具/MCP/计算后端
5. **安全**：工具调用经 `command_filter` + `control_safety` 策略清单 + `ToolSpec` 契约握手
6. **迭代**：复用 `deep_think` 推理刻录与 `memory` 召回沉淀经验

## 6. 最小示例

真实落盘示例：`huginn/plugins/science-skills/skills/industrial-asset-state/SKILL.md`
（设备状态采集 → 结构化回传，示范"OT 数据采集型 Know-how"封装）。字段全部为原生、
无外部依赖、无脚本，可被 `eco_tool.skill_install` / `SkillImporter` 直接解析注册。

## 7. 诚实边界

- **能直接调工具的，就不要包一层 Skill**：Skill 的价值在"可复用 + 可校验 + 可在编排里被
  组合"，不能用它掩盖本可直接完成的一次性动作。
- **不接私有托管生态**：西门子套件本体（Xcelerator/SKILL Creator SaaS）不在本仓库做硬绑定；
  需要对接具体工业后端时，用第 3c 节的 MCP/桥接路径，协议只落在后端适配层。
- **非确定性内容明确标注**：强校验/需回滚的走声明式；走指令式的要写清正文步骤规范。