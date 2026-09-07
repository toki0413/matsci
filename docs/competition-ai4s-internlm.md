# 书生大模型 + Huginn 科学智能体 — 参赛技术方案

> 赛道：方向1 AI4S 科学智能体
> 基座：上海人工智能实验室书生大模型（Intern-S2-Preview / Intern-S1-Pro / InternVL）
> 框架：Huginn（MIT，通用科研自动化智能体，`huginn-agent` v1.3.x）

---

## 1. 作品定位

**Huginn——一个会"解释分歧"而非"检索到就下结论"的科学智能体。**

面对"3 篇论文给出 3 个互相矛盾的数值"，Huginn 不做笼统地报「conflicting」，而是把它们
**按真实物理自由度分解归因**（方法族 / 泛函 / 温度 / 压力……）：组内收敛视为一致、组间差异
显式归因到缺失的自由度，并绑定每一条数值的来源证据（fid + sha256 快照）供独立复核；补后仍缺的
关键自由度落入显式豁免决策档（哈希链），交由门禁判放行。**不假装一致，也不捏造自由度。**

这恰好对应赛题要求的「假设—实验—验证—迭代」科学发现闭环。

## 2. 与赛题要求逐条映射

| 赛题「搜、读、算、做、写」全链条 | Huginn 对应能力 |
|---|---|
| **搜** 科学文献 / 多模态数据检索 | `literature_tool`/`agentic_search_tool`/`web_search_tool` + `completion_evidence`（缺度追问）
| **读** 图表 / 谱图 / 分子结构 / 实验曲线 | 内置 InternVL（多模态）+ ChromaDB 加密 RAG + MinerU 文献解析 + `vision`/`image_analysis`
| **算** 科学计算工具自主调用 | VASP/QE/CP2K/Gaussian/ORCA、LAMMPS/GROMACS/OpenMM、OpenFOAM/FEniCS/COMSOL/Abaqus/Elmer、DFT/MLFF/符号回归、量纲分析、Lean4 形式化验证
| **做** 实验 / 仿真方案设计 | `experiment_protocol_tool` + `autoloop`（自主探索）+ 技能预设（band_structure / relaxation / defect / surface / melt-quench / high-throughput…）
| **写** 报告 | `report_tool`（Markdown / LaTeX / HTML 一键生成）
| **验证 / 迭代** | `conjecture_engine` + `completion_evidence` 门禁 + `evolution/knowledge_distiller` 带证自证成长

## 3. 书生大模型接入

书生 ChatAPI 为 OpenAI 兼容端点，本方案为其新增**原生 `internlm` provider**：

| 项 | 值 |
|---|---|
| base_url | `https://chat.intern-ai.org.cn/api/v1` |
| 环境变量 | `INTERNLM_API_KEY` |
| 任务编排 / 主智能体 | `intern-s2-preview`（35B-A3B 科学多模态推理，256K 上下文，深度思考 + 工具调用）|
| 多模态解析 | `internvl-latest`（图表 / 谱图 / 分子结构）|
| 次模型档 | `intern-s1-pro` |

能力声明（`models/registry.py::MODEL_CAPABILITIES`）：`intern-s2-preview` 标记
`vision/tools/reasoning/streaming=True`，从而在 fail-closed 的能力路由中正确放行工具调用。

```bash
export INTERNLM_API_KEY=<你的书生 token>
huginn-agent chat --provider internlm --model intern-s2-preview
```

## 4. 技术栈与创新点（对应赛题热门前沿技术）

- **Harness**：自带通用科研编排框架（非 material 专用）——单网关、子智能体、swarm/team、自主科研闭环。
- **MCP 科学工具**：内置 `mat-db`（材料数据库 AFLOW/NOMAD/MP）、`math-anything`、`vision-pixel` 三个 MCP server。
- **多智能体协作**：`review_committee`（结构审查委员会）、`orchestrator`、`subagent` 模拟科研团队分工。
- **上下文工程 + 长记忆**：三级记忆（会话 / SQLite+FTS5 长期 / 自动promotion）+ 知识蒸馏 + 知识图谱。
- **语义 / 结构归因**：按物理自由度对文献值局部-整体分解（本项目核心竞争力）。
- **两正交控制轴**：极简模式 `ModelTier` × 思考强度 `ThinkingIntensity`，适配从本地弱模型到顶端大模型的任一层级。

## 5. 可复现验证

| 脚本 | 内容 | 依赖 |
|---|---|---|
| `agent/scripts/smoke_internlm.py` | Intern-S2-Preview 自主编排「搜文献→符号回归→归因结论」最小 AI4S 闭环 | 仅 `openai` + 网络 |
| `examples/demo_evidence_chain.py` | 缺度追问 + 对象级取证 + 门禁闭环（零网络 / 确定性） | 无 |
| `examples/co_ox_sensitivity.py` 等 | 科学计算 / 不确定性 / 实验设计独立复现 | numpy + matplotlib |

`smoke_internlm.py` 实测输出（Li2O 带隙）：

```
[1] search_literature  -> PBE 4.1 eV / exp 5.8 eV / HSE06 6.0 eV
[2] run_symbolic_regression -> E_g(P) = 5.7940 + 0.0255*P (GPa)
[3] 结论: PBE 泛函缺精确交换项系统低估带隙, HSE06 引入部分精确交换更接近实验,
     不同的理论方法对电子相关效应的处理差异导致数值分歧。
```

## 6. 安全与可落地

- 单网关 + 统一鉴权 / 审计 / 错误信封；破坏性工具默认进容器沙箱 + 命令白名单 + 超时输出上限。
- `requirements.lock`（uv pip compile）保证依赖可复现，CI 门禁拦截漂移。
- `huginn-agent share export/import` 把能力 / 工作流 / 人格 / 技能整体打包随拿随走。

## 7. 运行前提

正式跑训练/推理需在完整环境完成：

```bash
cd agent
pip install -e ".[all]"
export INTERNLM_API_KEY=<你的书生 token>
python -m huginn.server                     # API 网关
huginn-agent chat --provider internlm       # 接入书生
```