# 工业 Skill 元数据规范（Industrial Skill Metadata Spec）

> 状态：active
> 一句话：定义工业/OT 工程 Know-how 封装为 Huginn Skill 的**唯一元数据事实源**。
> 字段与类型严格对应 `SkillDefinition`（`huginn/skills/base.py`）与 `SkillImporter`
> （`huginn/plugins/skill_importer.py`），不引入私有字段；任何字段/格式漂移由
> `test_industrial_skill_guide.py` 自检拦截。
> 前置阅读：[industrial-skill-guide.md](industrial-skill-guide.md)（范式与承载路径）。

---

## 1. 范围

本规范覆盖两类元数据载体：

| 载体 | 位置 | 用途 |
|---|---|---|
| **SKILL.md frontmatter** | `huginn/plugins/<…>/skills/<name>/SKILL.md` | 指令式技能、跨平台可搬运（OpenClaw/Hermes/Huginn） |
| **SkillDefinition**（dataclass） | 进程内 `SkillRegistry` 注册 | 声明式技能、需强校验/可回滚/可组合执行 |

二者可互转：`SkillImporter.import_file(...)` 把 SKILL.md → SkillDefinition；`export_to_huginn(...)` 反向写出。

## 2. SKILL.md frontmatter 字段规范（原生字段集，逐字段）

YAML frontmatter 用 `---` 包裹，紧接在文件头。**能用且只能用下列字段**，字段名区分大小写。

| 字段 | 类型 | 必填 | 默认 | 校验/约束 | 映射到 SkillDefinition |
|---|---|---|---|---|---|
| `name` | string | 是 | 父目录名 | 唯一标识；缺省回退目录名 | `name` |
| `description` | string | 是 | — | 触发诊断；**值内不可含半角 `冒号+空格`（会 YAML ScannerError），用全角「：」或整体加引号** | `description` |
| `category` | string | 否 | `general` | 惯例枚举：`general`/`computation`/`analysis`/`diagnostics`/`reporting` | `category` |
| `steps` | `list[str]` 或 `list[dict]` | 否 | `[]` | 描述性字符串或结构化 step；字符串被转成 `manual` 占位 step（不结构化执行） | `steps` |
| `allowed-tools` | `list[str]` | 否 | `[]` | Huginn 原生写法 → `required_tools`；**必须是已注册工具**，否则不可结构化执行（回退指令式） | `required_tools` |
| `tools` | `list[str]` | 否 | `[]` | 等价 `allowed-tools`；OpenClaw 习惯写法 | `required_tools` |
| `when_to_use` | string | 否 | `""` | 触发场景，注入 `metadata.when_to_use` | `metadata.when_to_use` |
| `trigger_conditions` | `list[str]` | 否 | `[]` | **OpenClaw 独有**；多条件用 ` OR ` 拼接 | `metadata.when_to_use` |
| `trigger` | string/list | 否 | `""` | **Hermes(agentskills.io) 独有** | `metadata.when_to_use` |
| `tags` | `list[str]` | 否 | `[]` | 建议含 `industrial` + 领域标签 | `tags` |
| `paths` | `list[str]` | 否 | `[]` | 关联脚本/资源相对路径 | `metadata.paths` |
| `model` | string | 否 | `null` | 建议模型档位 | `metadata.model` |
| `effort` | string | 否 | `null` | 强度/难度建议 | `metadata.effort` |

**平台自动识别_：** `trigger_conditions`→OpenClaw；`trigger`→Hermes；`allowed-tools`→Huginn 原生；均无则按 `tools` 猜（有→OpenClaw，无→Hermes）。显式指定 `platform=auto` 时据此，也可 `import_file(path, platform="huginn")` 强制。

## 3. 结构化扩展字段（SkillDefinition）

仅声明式（Python）路径可用，用于"确定性、可校验、可回滚"的工业动作。不写进 SKILL.md frontmatter。

| 字段 | 类型 | 说明 |
|---|---|---|
| `parameters` | `list[SkillParameter{name,type,description,default,required}]` | 对外参数契约 |
| `steps` | `list[SkillStep{name,tool,input_mapping,output_key,validation,on_failure,retries,condition,loop_until,loop_max_iterations,parallel_group}]` | 步骤 + 控制流（if 门/循环/并行组） |
| `required_tools` / `required_env_vars` | `list[str]` | 运行依赖声明（配合 `control_safety` 策略清单） |
| `references` / `estimated_cost` / `tags` | list / dict / list | 引用、成本估算、标签 |
| `metadata` | dict | **第三方/厂商特有字段兜底**；按命名空间存放（如 `metadata["vendor:somenet"]`），不污染原生字段 |
| `domain` | string | 工业领域：`automation`/`plc`/`robotics`/`scada`/`qms`/`heating`…（自由，建议小写蛇形） |
| `stage` | string | 研究/工程阶段：`hypothesis`/`computation`/`analysis`/`reporting` |
| `function` | string | 功能：`search`/`design`/`predict`/`validate` |
| `parent` | string | 技能树父技能名（派生/复用） |

`parallel_group` 语义（对工业"多传感同时就绪才合成"尤其有用）：同组 step 用 `asyncio.gather` 并行、一起 commit，组内 step 必须互不依赖对方的 `output_key`。

## 4. 第三方格式 → Huginn 映射

| 源格式字段 | 目标 | 说明 |
|---|---|---|
| `tools` → `required_tools` | `allowed-tools`(导出) | — |
| `trigger_conditions`/`trigger` → `metadata.when_to_use` | `when_to_use`(导出) | 多条件 ` OR ` 拼接 |
| 描述性 `steps`（字符串） | `_MANUAL_TOOL("manual")` 占位 step | 仅作 prompt，不结构化执行；要执行需补 `tool` 映射 |

## 5. 校验规则与常见陷阱

1. **YAML 冒号陷阱**：`description` 值内不要用"半角冒号 + 空格"（如 `chain-of-thought: ...`），会被 `yaml.safe_load` 判为 map 而 `ScannerError`。改用全角「：」或给整段值加引号。
2. **工具必须已注册**：`allowed-tools`/`steps[].tool` 引用未注册工具时，Skill 无法结构化执行——运行时仅在 `ToolRegistry` 查到才走 `DeclarativeSkillExecutor`，否则回退指令式（LLM 读正文）。
3. **文件命名**：必须是 `SKILL.md`（`name` 缺省时用父目录名，故目录名要规范）。
4. **不新增私有字段**：若要表达行业特有元数据，一律进 `metadata` 命名空间，禁止在 frontmatter 或 SkillDefinition 顶层加字段——否则 `test_industrial_skill_guide.py` 的"仅原生字段"检查会红。
5. **category 沿用惯例枚举**，避免造出无解析者/无注册表识别的类别。

## 6. 端到端完整示例

**指令式（SKILL.md，frontmatter 仅原生字段）：**

```markdown
---
name: industrial-asset-state
description: 把产线/设备状态整合成结构化回传 JSON 供入库比对；触发场景包括采集设备状态、汇总 OT 传感读数、判断是否超阈值告警。
category: diagnostics
allowed-tools: [file_read_tool, numerical_tool]
when_to_use: 需要规整产线设备状态数据并做阈值判定时
tags: [industrial, ot-data-collection, asset-state]
---

# industrial-asset-state

采集 → 规整 → 阈值判定 → 告警的正文步骤规范……
```

**声明式（SkillDefinition）：**

```python
from huginn.skills.base import SkillDefinition, SkillParameter, SkillStep

SkillDefinition(
    name="industrial-asset-state",
    category="diagnostics",
    domain="automation",
    stage="analysis",
    function="validate",
    parameters=[SkillParameter(name="sources", type="list", description="传感数据源", required=True)],
    steps=[SkillStep(name="collect", tool="file_read_tool",
                     input_mapping={"path": "sources"}, output_key="raw",
                     validation="len(raw) > 0", on_failure="retry")],
    required_tools=["file_read_tool", "numerical_tool"],
    tags=["industrial"],
)
```

## 7. 契约与漂移护栏

- 本规范字段 = `SkillImporter` 原生字段集 + `SkillDefinition` 字段（见 `docs/INDEX.md` 登记）。
- 自检：`tests/test_industrial_skill_guide.py` 实测解析落盘示例、断言仅用原生字段、断言本规范/示例已登记，漂移即红。
- 新增工业 Skill 时：按 §2/§3 声明字段 → 落盘 SKILL.md（指令式）或注册 SkillDefinition（声明式）→ 用 `eco_tool.skill_install` 或 `SkillImporter` 导入 → 由自检守护字段一致。