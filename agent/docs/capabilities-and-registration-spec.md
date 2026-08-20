# 能力增强与外部接入制度（统一 Spec）

> 本文档将三块能力增强合并为一个统一路线：**写作引导、原生多模态引导、插件/MCP 注册制度**。
> 三者共享同一底座——`prompt_segments` 段插件 + `ToolRegistry`/各类注册表 + 统一 Catalog 控制面，
> 以便一次设计、协同落地，后续维护与功能增强有单一文档可依。
>
> 状态标注：✅ 已落地 · 🟡 待落地（一期/低风险） · ⏳ 二期/高成本

---

## 目录
- [A. 写作引导（anti-defensive-writing）](#a-写作引导)
- [B. 原生多模态引导（现状盘点 + 落地方案）](#b-原生多模态引导)
- [C. 插件 / MCP 注册制度优化](#c-插件--mcp-注册制度优化)
- [D. 协同点与统一底座](#d-协同点与统一底座)
- [E. 落地顺序与验收](#e-落地顺序与验收)

---

## A. 写作引导

### 状态：✅ 已实现并落地

已按 [`Kiterlin/anti-defensive-writing`](https://github.com/Kiterlin/anti-defensive-writing) 方法论落地，
覆盖主张先行、避免过度免责、保留必要 scope/method/safety 等原则。

| 落地项 | 载体 | 说明 |
|---|---|---|
| Prompt 引导段 | `huginn/agent/prompt_builder.py` `writing_segment()` | 注入"主导主张/正面 scope/精确措辞"六条原则；已挂进 `segments` 列表 + 插件注册 |
| 注册位 | `huginn/plugins/prompt_segments.py` `_PRIORITY["writing"]=60` | 夹在 tools 与 thinking 之间 |
| Skill 定义 | `plugins/science-skills/skills/anti-defensive-writing/SKILL.md` | 逐条对齐 repo 检测清单、改写流程、materials-science 示例 |
| 检测脚本 | 同目录 `scripts/review.py` | 正则启发式定位防御性措辞（免责/堆叠 hedge/not-X-but-Y 等），已修语法错 |

> 后续增强入口：把 `multimodal` 引导段也注册进 `prompt_segments`（见 B-1），
> 与 `writing` 段同一机制、同一优先级区。

---

## B. 原生多模态引导

### 状态：现状盘点（建筑材料就绪）+ 🟡 落地方案

**目标**：让"图"在 agent 里始终走对的路径——当前模型支持原生视觉时直接交给 LLM
（原生多模态），不支持时明确引导到视觉成员 / 视觉记忆 / 定量工具，而不是静默降级。

### B.1 现状盘点（已实现）

入口在 `agent/streaming.py` 的 `_route_vision`，`vision/router.py` 里 `route_vision`
根据 `models/registry.py` 的 `ModelCaps.vision` 决定三档：

| 路径 | 触发条件 | 行为 |
|---|---|---|
| `NATIVE_LLM / BOTH` | 当前模型 `caps.vision=True` | 原图以 image block 直接送 LLM + CV 预分析并行注入 `SystemMessage` |
| `CV_TOOLS` | 当前模型无 vision | CV hints + visual memory（编码器检索）+ `image_analysis_tool` 定量 |
| 跨 agent 委托 | 主 agent 无 vision 且有 VISION 角色成员 | 委托视觉成员描述，注入文本上下文 |

配套已实现：
- `vision/modality_router.py`：动态分辨率（等比降采样）+ 模态识别（SEM/TEM/EDS/PLOT…）+ action 建议。
- `vision/symbol_encoder.py`：`visual_to_symbols` 图表数据提取（文本 + 结构化双格式）。
- `perception/visual_encoder.py` + `image_index.py`：本地视觉编码（I-JEPA/CLIP）+ 图片索引做"视觉记忆"相似检索。
- `tools/image_analysis/`：SEM/TEM/EDS/particle 定量分析 action。
- `tools/visualize_tool.py` / `visualize_qa.py`：输出侧图表生成与 QA 质检。
- 请求体限额已抬高到 100MB（`HUGINN_MAX_BODY_SIZE_MB`），给多模态输入留了余量。

### B.2 核心缺口

- 默认模型 DeepSeek Flash 在 `models/registry.py` 标为 `vision=False`，`NATIVE_LLM / BOTH` 默认不触发，
  图像永远走 CV_TOOLS 兜底；"原生看图"只在用户切到带 vision 模型或配了 VISION 团队成员时才可达。
- 无 `multimodal` 引导段，agent 不知道"该直接看还是该调定量工具"的最优取舍。
- 能力标签模型级硬编码、无运行时探测；UI/用户无从得知当前模型能否看图。
- 本地视觉 LLM（decoder）未常驻，语义兜底只依赖跨 agent 委托，链路较重。

### B.3 落地方案

按"引导、路由、能力可见性"三条线，复用既有插件/中间件机制。

1. **🟡 Prompt 层：新增 `multimodal` 引导段**（与 writing 段同机制）
   复用 `prompt_segments` 注册表，加 `multimodal_segment()`，命中图像时高优。
   位置：`_PRIORITY["multimodal"]=55`（紧挨 tools/writing）；接入点：`prompt_builder.segments` + 同款注册。
2. **✅ 路由层：CV_TOOLS 分支显式决策提示**
   `build_cv_context` 末尾附能力自省段 "Native vision unavailable…Pick one explicit route: ① delegate to a vision-specialized agent, ② use visual-memory similar images, ③ call image_analysis_tool"，把静默降级改成可见决策点。该路径经 `_compute_strip_flag` 门控只对 `vision=False` 模型触发（vision=True 保留 image_url）。测试：`tests/test_vision_router.py::test_explicit_route_hint_for_text_model`。
3. **✅ 能力可见性：运行时 `vision` 探测端点 + UI 提示**
   已落地 `GET /models/caps`（`routes/config.py`，`require_admin_key` 保护，按当前活跃模型返回 `vision/tools/reasoning/streaming`，未知模型 fail-closed）；前端 `desktop/…/SettingsPanel.tsx` 活跃模型选择处显示"支持看图 / 文本模型"徽章（后端不可用静默跳过）。测试：`tests/test_server.py::TestModelCaps`。
4. **✅ 语义兜底：本地 decoder 常驻**
   下沉本地视觉解码到 `huginn/vision/local_decoder.py`（纯 stdlib urllib，无新依赖）：文本模型看图时优先调本地 vision LLM（Ollama 上的 qwen2.5-vl 系）`decode_image()` 转一段文本描述，失败/无模型再回落跨 agent 视觉委托。可用性探测带 TTL 常驻缓存（`HUGINN_LOCAL_VISION_MODEL/TTL/TIMEOUT` 可配，已登记 env 契约）；`agent/streaming.py` CV_TOOLS 分支接入并修掉解码成功分支的 `NameError` 竞态。零显存/依赖评估结论：解码完全委托给本机 Ollama，agent 侧无显存压力。（成本：一次本地 LLM 调用 vs 一次完整 agent 往返。）测试：`tests/test_local_decoder.py`（无模型/前缀匹配/解码成功失败/编码/缓存）。

---

## C. 插件 / MCP 注册制度优化

### 状态：🟡 统一 Control-plane 方案

**目标**：统一碎片化注册，用户**一条路**接入想要的外部 MCP / 插件 / skill，
项目后续维护与增强有单一控制面。遵循"复用既有注册表"——不重造执行层，只加薄控制面。

### C.1 现状与痛点

当前是 **5 套平行注册表 + 3 个 MCP 配置源 + 1 个运行时 API**：

| 注册面 | 载体 | 发现约定 |
|---|---|---|
| `ToolRegistry` | `tools/registry.py` | 代码 `register()` |
| `ModelRegistry` | `models/registry.py` | 代码 + 配置 |
| `StarHandlerRegistry` | `plugins/registry.py` | `loader.py` 扫 `metadata.yaml + main.py + Star 子类` |
| `skill_loader` | `plugins/skill_loader.py` | 递归扫 `SKILL.md` |
| `prompt_segments` | `plugins/prompt_segments.py` | 代码 `register_prompt_segment()` |

MCP 配置源：`config.mcp_servers` / 仓库根 `.mcp.json` / 内置 `mat-db`·`math-anything`·可选`tooluniverse` / 运行时 API（`routes/mcp.py`）。

痛点：
1. **用户接入门槛高**：三种插件作者模型 + 三个 MCP 配置源，无统一路径与说明书。
2. **安全策略不一致**：`routes/mcp.py` connect 走命令白名单 `{python,node,npx,uvx...}`，
   而 `.mcp.json`/`config.mcp_servers` 在 `lifespan.py` **不过任何白名单**——同一功能两条安全标准。
3. **来源不可追踪、无去重**：同一 server 两源重复注册；工具标签无 `origin`，热重载易残留。
4. **无统一启停 / 清单接口**：MCP 只能整 server 连断，无"临时禁用某工具"统一开关，前端需拼 5 个模块。

### C.2 统一注册制度（控制面）

**核心**：一套清单 `Catalog` + 单一发现器 `CatalogManager`，执行仍委托既有注册表。
控制面只管"有哪些、开关开没开、从哪来、归不归属"。

- 统一清单项 `CatalogEntry`（`huginn/catalog/models.py`）：
  `id / kind(mcp|plugin|skill|tool|prompt|model) / name / origin(builtin|.mcp.json|config|api|skill_dir) / enabled / version / registered_names`。
- 单一发现器 `discover()`：复用 `plugins/loader.discover()`、`science_skills_bridge.discover()`、
  `_load_mcp_json_servers()` 为各 source 采集函数，按优先级 `builtin → .mcp.json → config → dirs → runtime` 去重归一，**不重写扫描**。
- `CatalogManager`（挂 `server_core.get_context().catalog`）：
  `discover_all / apply(reconcile) / list / set_enabled / uninstall / snapshot / restore`。
  原则：只做"发现→注册→追踪→启停→快照"编排，不存执行逻辑。

### C.3 MCP 注册制度

- **单一事实来源 + 归一化**：`config.mcp_servers` 与 `.mcp.json` 归一成 `CatalogEntry(kind="mcp")` 汇入清单，
  `.mcp.json` 优先级最高；`lifespan.py` 内置分支改为 `servers/` 目录自动发现（origin=`builtin`）。
- **安全策略统一**：新增配置 `MCP_ALLOWED_COMMANDS`（默认取原 hardcode 集合）：
  - `stdio`：注册/连接前统一过 `_validate_command(command, transport)`，**所有入口**（config/.mcp.json/API/CLI）都走。
  - `sse/streamable-http`：不启动本地进程，只校验 URL。
  - `routes/mcp.py` 的 hardcode 白名单删除、改读配置——消除两条安全标准。
- **Secret 沿用既有**：`api_key`/`headers`/env 令牌走 `config.py` mask；
  清单序列化（`/mcp/status`、`catalog.list`）对敏感键也 mask，不新增明文落盘。
- **运行时 API 收敛**：新增 `GET /catalog`、`PATCH/DELETE /catalog/{id}`；保留既有 connect/disconnect/reconnect（含 `refresh_tools_from_registry`）。

### C.4 插件 / Skill 注册制度

- **统一清单 + 可选统一 manifest**：插件/skill 目录顶部允许 `manifest.yaml`
  （等价物：已有 `metadata.yaml` / `SKILL.md` frontmatter），声明
  `name / kind(plugin|skill|tool) / version / description / tools / prompts / scripts / entrypoint / paths(条件激活)`。
- 发现器按 `kind` 路由：`tool→ToolRegistry`、`plugin→loader+StarHandlerRegistry`、
  `skill→science_skills_bridge/script runner`、`prompt→prompt_segments`。
- 对已存在 `metadata.yaml`/`SKILL.md` 仅做缺省 kind 推断，**不强制迁移**（向后兼容）。
- 来源追踪：注册结果写回 `CatalogEntry.registered_names`，`ToolRegistry` 不新增字段。
- 禁用语义统一由 Catalog 提供（HuginnTool 走 `unregister` + 恢复挂回；skill 沿用 activate/deactivate；prompt 段沿用 priority 过滤）。
- **prompt 段**：`multimodal`/`writing` 都登记 `CatalogEntry(kind="prompt")`，可列举与单独禁用。

### C.5 用户统一接入路径（CLI + API 等价）

```
huginn catalog list
huginn mcp  add/rm   --name notes --command npx -y @mcp/notes
huginn skill add/rm  --dir ~/skills/my-skill
GET /catalog         # 查看/启停全部接入项
```

对外文档统一表述三步：① MCP 写一份配置 → `huginn mcp add` 一次接入；
② 插件/skill 放进 `plugins_dir`/`skills_dir` → `huginn skill add` 一步注册；③ `GET /catalog` 统一启停。

---

## D. 协同点与统一底座

三块能力共享同一底座，设计上互相复用、一次落地：

| 底座机制 | 写作（A） | 多模态（B） | 注册制度（C） |
|---|---|---|---|
| `prompt_segments` 段插件 | ✅ `writing` 段 | 🟡 `multimodal` 段 | `CatalogEntry(kind="prompt")` 追踪/禁用 |
| `ToolRegistry` | — | 图表/定量工具 | 统一注册 + snapshot/restore + origin 追踪 |
| Catalog 控制面 | 引导段可禁 | 工具来源追踪 | 全部接清单 |

> 落地顺序原则（结合记忆中的既有决策）：**不重复造注册表**（ToolRegistry/ModelRegistry/SkillLoader 已够用），
> 只补 `Catalog` 控制面；先做零行为变化的**安全收敛**，再做开放能力增强。

---

## E. 落地顺序与验收

### 落地顺序

1. **🟡 A 前提**：写作引导已落地，可用做 prompt 段范本。
2. **🟡 B-1/B-2**：新增 `multimodal` prompt 段 + CV_TOOLS 决策提示（纯文本/配置，风险最低，立即可做）。
3. **✅ C 安全收敛**：`MCP_ALLOWED_COMMANDS` 配置化 + 全入口统一 `_validate_command`（零行为变化，消除安全不一致）。
   - 补充：MCP 配置序列化脱敏落地 —— `mcp_client.mask_mcp_config()`（嵌套 env/headers 敏感键掩码），应用于 `list_servers()`；审计日志沿用既有 `events/audit_log._sanitize_args`。`config.to_dict` 不暴露 `mcp_servers`，`get_server_status` 仅回传计数，均无凭据泄漏窗。
   - 测试：`tests/test_mcp_command_validation.py`（白名单 3 条 + 脱敏 3 条）。
4. **✅ C Catalog 骨架**：`CatalogEntry` + `CatalogManager`，接入 ToolRegistry / MCPManager 只看清单。
   - 已落地：`huginn/catalog/{models,manager}.py` + 挂 `server_context.catalog`；`discover_all`（tool 经 ToolRegistry、model 经 ModelRegistry、skill 经 SkillRegistry、prompt 经 prompt_segments、mcp 经 get_server_status）、list/get、set_enabled、uninstall、snapshot/restore。
   - 边界（`ponytail:`）：set_enabled/uninstall 一期只改追踪标记，不触底层注册表；来源精确归一（builtin/.mcp.json/config）与 reconcile 归第 5 步。
   - 测试：`tests/test_catalog.py`（discover/list、启停状态跨 rediscover 保留、快照往返）。
5. **✅ C 来源归一 + `GET /catalog`**：builtin/.mcp.json/config 汇入清单，统一列表。
   - 已落地：`MCPClientManager` 增加 per-server 来源追踪（`_origins` + `register_server`/`connect` origin 参数 + `set_server_origin`），`list_servers()` 返回 origin；lifespan 启动时给每 server 标 origin（servers/=builtin，`.mcp.json`=.mcp.json，`load_from_config`=config）；catalog `_collect_mcps` 改用 `list_servers()` 单点归一；`GET /catalog`（`routes/catalog.py`，触发 discover_all 返回含 origin/enabled 的归一清单）。
   - 测试：`tests/test_catalog.py` 更新为 list_servers 视图（来源/默认 runtime/enabled 保持/快照/Oorigin 优先级，5 条）。
6. **✅ C 启停/卸载 API + CLI**：`PATCH/DELETE /catalog/{id}`、`huginn catalog list|enable|disable|delete`。
   - 已落地：`catalog/reconcile.py` 把启停标记落到注册表 —— mcp 下线断会话+工具摘除、上线 reconnect+重注册，tool 下线 stash 暂存+上线复原（prompt/skill/model 暂无禁用语义，返回 unsupported 不强行落地）；API 侧 `PATCH/DELETE /catalog/{id}`（`routes/catalog.py`）；CLI 侧 `huginn catalog` 子命令组（`cli/commands/catalog_cmd.py`，离线复用注册表发现，enable/disable/delete 委托 reconcile）。
   - 边界（`ponytail:`）：CLI 逐次独立进程，tool 的 stash 复原仅进程内有效，跨进程 re-enable 返回 unsupported；跨进程持久启停走 server 运行时 `PATCH /catalog`。
   - 测试：`tests/test_catalog.py` 增补 CLI `catalog list` 冒烟（6 条）。
7. **✅ C 统一 manifest.yaml**：新接入项声明 kind/tools/prompts，向后兼容既有约定。
   - 已落地：`plugins/manifest.py` 提供 `parse_dir/discover/write_manifest` —— 新增接入目录优先读 `manifest.yaml`（显式声明 `kind/name/version/description/tools/prompts/scripts/entrypoint/paths`），无则回退 `metadata.yaml`（kind=plugin）、`SKILL.md`（kind=skill），按目录内容推断缺省 kind，归一成同一 spec dict。只做读+归一，不碰执行层；kind→注册表的路由走 catalog。
   - catalog 增 `_collect_plugins(plugins_dir)`，`GET /catalog` 与 `huginn catalog list` 传入 `DEFAULT_PLUGINS_DIR`，使插件/skill/tool 目录也进入统一清单（origin=dirs）。
   - 测试：`tests/test_manifest.py`（3 来源归一 + kind 推断 + catalog 采集，5 条）。
8. **🟡 B-3 + ⏳**：`/models/caps` 端点 + 前端提示；本地 vision decoder 常驻（二期）。

### 验收清单

- `huginn catalog list` 并列展示 MCP/插件/skill/工具，含 `origin/enabled`。
- 同一 server 在 `.mcp.json` 与 `config.mcp_servers` 同时出现只注册一次（按优先级）。
- `MCP_ALLOWED_COMMANDS` 收紧后，config/API/.mcp.json 三入口对非法 command 均拒绝；SSE 不受限。
- `PATCH /catalog/{id} {"enabled": false}` 后工具从 agent 工具列表消失，`true` 恢复（MCP 触发重连）。
- `multimodal` 引导段命中图像时生效、未命中零开销；`writing` 段维持现状通过既有测试。
- Catalog `snapshot/restore` 全量往返一致；既有 5 类注册表既有测试全绿（兼容不改行为）。