# How to Read Huginn — 新人上手导览

> 给第一次接触本仓库的人。先读这篇，再到 [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) 补"为什么"。
> 目标：30 分钟内知道"代码都放哪、一条请求怎么走通、我要改的东西去哪找、文档哪些可全信哪些要警惕"。

---

## 0. 三条铁律（先记牢）

1. **契约文档可全信，模块注释要警惕。**
   `docs/` 下的 `env-contract.md` / `events-contract.md` / `tools-contract.md` / `routes-contract.md` 等由代码自动再生，**不会过期**；
   但**模块 docstring / 架构状态注释可能滞后于代码**（我们修过一批写着"未接入主循环"、其实已接入的模块）。
   → 判断"这个模块到底有没有被真正调用"，用搜索工具（Grep）查它的 import/调用点为准，别只读它自己写的注释。
2. **数学是审计层，不是决策必须。** 主循环里计算同调/sheaf/Hodge 等不变量，作用是 **advisory** 回灌 prompt，
   不强行阻断假设。别把"结构不变量"当成"必过门禁"。
3. **改动要防回归。** 工具、phase、route 改了要进 `tests/`；测试用 `app_client` fixture，不要在模块顶层裸建 `TestClient`。

---

## 1. 从哪开始读（3 个入口）

| 入口 | 内容 | 什么时候看 |
|---|---|---|
| [INDEX.md](INDEX.md) | 所有文档导航 + 状态（active/staging/report） | 先看状态标签，避免读 staging 当已实现 |
| [architecture.md](architecture.md) | 模块职责 + 数据流 + 规则 | 想知道"谁是谁、层之间怎么流" |
| [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) | 全局 how+why、设计原则、诚实边界 | 想理解"为什么这么设计" |

---

## 2. 目录地图一页纸

```
agent/huginn/
  agent/       单 agent 主循环：core/session/streaming/reflection
  agents/      多 agent：orchestrator、subagent、swarm、team、loop_detector
  autoloop/    自主探索闭环：CognitiveLoop、engine 7 phase、red_team、engine_observe
  tools/       大量工具 + ToolRegistry + 安全元数据（fail-closed 默认）
  skills/      声明式技能：base/registry/presets
  memory/      3-tier 记忆：session/longterm/manager
  knowledge/   KB：store/chunker/auto_ingest + seed/(53 个内置知识)
  kg/          知识图谱 + CLAIM：graph/entities/hypergraph/claim_audit
  metacog/     元认知：同调/sheaf/Hodge/范畴 functor/cognitive map/mental imagery
  causal/      SCM 生成、干预、反事实
  events/      事件总线 + 审计日志
  runtime/     任务生命周期、状态、trace
  security/    auth、策略、沙箱、脚本执行
  hpc/         远程执行
  cli/         ~40 命令，入口 main.py
  routes/      FastAPI 路由（全在 /v1）
```

---

## 3. 一条主线走通（Trace One Query）

```
用户 query
  → CLI/WS/HTTP 入口
    → HuginnAgent 主循环 (agent/core.py)
      → prompt_builder + memory 注入
        → LLM reasoning（streaming 捕获 reasoning_content → reasoning_trace）
          → tool 调用（本机 / MCP / HPC，走 ToolRegistry + whitelist）
            → ToolResult → session + longterm memory
              → 成功 → 蒸馏知识 confirmed → auto_ingest_to_kb
      → response（streaming / SSE / WS）
```

**自主探索（autoloop）的另一条主线：**
```
Objective
  → CognitiveLoop [observe → decide → execute → reflect]（4 钩子，签名稳定）
    → engine 7 phase [perceive → hypothesize → plan → execute → validate → learn → report]
      → 每轮跑 _metacog_topology_audit（同调/sheaf/Hodge，advisory）
        → red_team 对假设做批判（带 severity 的结构判据）
          → 失败方向进 FailedDirectionStore → 下轮避免
```

---

## 4. 我要改 X，去哪找？

| 我要做 | 去哪 |
|---|---|
| 加一个工具 | `huginn/tools/`，实现 `HuginnTool`，然后在 `huginn/tools/__init__.py` 登记 |
| 改 agent 行为 | `autoloop/engine.py` 的 7 个 phase 方法 |
| 改假设验真逻辑 | `autoloop/hypothesis_loop.py`（含拓扑审计入口） |
| 改"开新实验/分叉" | `agents/` 的 speculator/orchestrator |
| 改知识检索/缓存 | `knowledge/store.py`（KB LRU、BM25）、`perception/rag_bridge.py` |
| 改结论审计 | `kg/claim_audit.py`、`kg/hypergraph.py` |
| 改记忆衰减/去重 | `memory/decay.py`、`memory/` 的 maintainer |
| 新增路由 | `routes/`，挂 `/v1` |
| 改技能导入 | `plugins/skill_importer.py` + `cli/commands/skill_import.py` |

---

## 5. 文档真相层级（哪些能信、哪些要警惕）

1. **契约面（可信）**：`-contract.md` 契约文件由代码生成，永不漂移。
2. **架构现状（可信）**：[tech-spec.md](tech-spec.md) 是现状事实记录；[architecture.md](architecture.md) 是模块导览。
3. **模块头注释（要交叉验证）**：docstring 里的"架构状态"可能滞后于代码——**以调用点为真相**。
4. **spec（分状态）**：标 `active` 的是已落地；标 `staging`/`building` 的是设计稿，别当已实现（例：`reward_design.md`）。
5. **research 目录（高级）**：`experimental/` 与部分标"research 层"的 metacog 模块是 future hook，不是当前承诺。

---

## 6. 踩坑备忘录（前人经验）

- **看模块是否真被调，用 Grep 找调用点，别信 docstring。**（我们踩过：8 个模块 header 写"未接入"，实际主循环在用。）
- **中文整段在 embedding 里可能当单个 token**，做相似度匹配先分词（jieba）。
- **改动契约面要同步 `docs/` 再提交**，否则漂移。
- **别假设"有但死"的代码在工作**：真要复用它，先确认生产路径有调用（`grep <名字> -- <主循环文件>`）。
- **提交前先确认分支干净**，这个仓库历史被重写过，force 有坑。