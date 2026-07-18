# 12 ｜ 对标 ResearchClaw：huginn 工作流差距诊断

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-07-17 |
| 诊断对象 | huginn-agent 的科研工作流（autoloop / ResearchPhase / RCB 3-step）对标 ResearchClaw、AutoResearchClaw、Claude Science |
| 证据来源 | 本地 `ResearchClawBench/`（evaluation/score.py、checklist.json、8 个 Material workspace 的 `_score.json`/`_agent_output.jsonl`/`_meta.json`）、`agent/huginn/` 源码、git 提交时间线、`audit_20260717/04/16/20/21` 已确认事实、kimi_search_v2 获取的公开架构资料（arxiv 2606.07591v3、github.com/ymx10086/ResearchClaw、github.com/aiming-lab/AutoResearchClaw、TechCrunch Claude Science 报道） |
| 方法 | 静态只读 + 轨迹逐事件回放 + 评分 JSON 逐项归因 + 竞品公开架构调研；未运行动态实验；推断处显式标注 |
| 与审计 21 的关系 | 审计 21 的竞品机制为「推断」；本报告用竞品一手公开资料替换推断，并补充官方 RCBench 跑分校准、两代 runner 的 git 时间线、Gen-2（3-step）运行的逐步失败回放 |

---

## ① 数据与方法

### 1.1 数据

**huginn 侧（本地证据）**：
- RCB 成绩：`ResearchClawBench/score_results.json` + 各 workspace `_score.json`。Material_000=10.5、Material_001=15.75、Material_002=17.28、Material_003 五次 2.25–12.75（最好 12.75 来自 `Material_003_20260714_234743`，最差 2.25 来自 `Material_003_20260716_190227`）。
- 运行轨迹：`_agent_output.jsonl`（RCB `evaluation/run_task.py:123-128` 逐行捕获 agent subprocess 的 stdout+stderr）。
- 自进化产物：`agent/huginn_autoloop_report_loop_*.md` 共 16 份。
- 运行器版本时间线：`git log -- agent/huginn/cli/rcb_runner.py`（文件 2026-07-16 20:44 首次入库，commit `1ac780b`；入库前以未跟踪文件存在于工作区——推断，依据是 07-16 15:23 运行的 `_meta.json` 已记录 `agent_cmd=...rcb_runner.py`）。

**竞品侧（公开资料）**：
- ResearchClaw（ymx10086/ResearchClaw，RCBench 官方内置 agent 之一）GitHub README。
- AutoResearchClaw（aiming-lab/AutoResearchClaw）GitHub README（23 阶段管线 + 完整 config 参考）。
- Claude Science（TechCrunch 2026-06-30 报道）。
- RCBench 官方论文 arxiv 2606.07591v3 Table 5（7 个 autonomous agent + 17 个 ResearchHarness LLM 的完整跑分）。
- ResearchHarness 工具面（同论文 §3.3、Table 4）。

### 1.2 方法

1. 从 `_score.json` 的 judge reasoning 逐条定位失分点；
2. 用 `_agent_output.jsonl` 逐事件回放失败时刻的 agent 决策；
3. 用 git 时间线确定每个被评分运行实际使用的 runner 代际；
4. 用竞品公开架构做逐环节机制对比；
5. 归因遵循「直接失分原因 vs 背景缺陷」纪律。

### 1.3 关键校准事实（官方榜单）

RCBench 论文 Table 5（judge = GPT-5.1；50 分 = 达到目标论文水平）：

| 系统 | Overall | Material |
|---|---|---|
| Claude Code (Claude-Opus-4.6) | **21.5** | **25.5** |
| Codex CLI (GPT-5.4) | 18.4 | 13.0 |
| OpenClaw (GPT-5.4) | 16.6 | 12.9 |
| **ResearchClaw (GPT-5.4)** | 16.3 | **19.3** |
| EvoScientist (GPT-5.4) | 15.5 | 13.5 |
| ARIS Codex | 13.6 | 12.4 |
| Nanobot (GPT-5.4) | 12.8 | 13.0 |
| DeepSeek-V4-Pro（ResearchHarness 裸模） | 17.1 | **24.6**（LLM 组 Material 第一） |
| **huginn（deepseek-chat，本地 judge）** | — | **13.7**（4 题均值；逐题最好均值 14.1） |

注意三点：(a) huginn 本地评分 judge 与 agent 同源（deepseek-chat，`audit_20260717/16_评测体系完整性.md` P1-3），与官方 GPT-5.1 judge 的分数只可作量级对比；(b) huginn Material 均分 13.7 与 Codex CLI/OpenClaw/EvoScientist/Nanobot（12.4–13.5）同档，但**同模型家族的 DeepSeek-V4-Pro 配轻量 ResearchHarness 拿到 24.6**——scaffold 差异值得约 11 分；(c) 官方错误分析（论文 §4.5）：7 个 agent 的失败集中在 **Experiment Design Mismatch / Evidence Mismatch / Scientific Core Missing**，而非 Execution Failure——前沿 agent 的失败是「跑偏」，huginn 的失败（见下）是更原始的「没跑完就写占位报告」，失败类别不同。

---

## ② 核心发现（按证据强度排序）

### F1 ｜ Material_000 权重 0.5 项 0 分的直接原因：per-tool 预算在数据加载试错中耗尽 → agent 用「Expected」占位表交差 【证据确凿】

证据链（每一环都有原文）：

1. **预算配置**：`agent/huginn/cli/rcb_runner.py:305-306` —— `max_tool_calls=150, max_tool_calls_per_tool=50`。
2. **执行通道收窄**：`rcb_runner.py:278` system prompt 自承 "code_tool: run Python. Sandbox BLOCKS open() and os — CANNOT write files via code_tool"；`:283-284` 自承 "code_tool security scanner may false-positive on eval() in torch/numpy"。于是规定动作是「file_write_tool 写脚本 + bash_tool 跑」。
3. **预算消耗实况**：`ResearchClawBench/workspaces/Material_000_20260716_152336/_agent_output.jsonl` 事件 5–38 全是 `.pt` 数据加载试错（PyTorch zip + 自定义 pickle 类 `RealisticCrystalData`，沙箱反复拦截）；`code/` 目录 15 个脚本中 10 个是数据加载变体（`inspect_data*.py`/`load_data_v1-v5.py`/`inspect_pkl.py`/`inspect_zip.py`/`extract_tensors.py`/`test_stub.py`）；事件 31 数据才加载成功。
4. **预算触顶**：事件 53 "The bash_tool budget is exhausted. Let me switch to code_tool for the rest."；事件 56 "The code_tool is also blocked."；事件 58 "Both tool budgets are exhausted. Let me write the report now using the EDA figures we have"。机制：`agent/huginn/agents/tool_budget.py:50-72` 的 `record()` 判定超限 → `agent/huginn/tools/adapter.py:721-723` 把 `{"error": f"工具调用预算耗尽: {budget_reason}", "_budget_exceeded": True}` 喂回 LLM。
5. **占位交付**：`report/report.md:143` 标题原文 "**Table 2: Expected Candidate Discovery Performance Metrics**"；`:152` "The model is expected to identify a significant fraction of the 43 hidden altermagnets"；`outputs/` 目录为空；轨迹末行（事件 84）："the full fine-tuning + candidate prediction pipeline (`main_pipeline.py`) is written and ready to execute but could not complete due to tool budget limits."
6. **评分兑现**：`_score.json` item 2（weight 0.5）score=0，judge 原文："The AI report provides only qualitative descriptions ... lacks any of the required numerical metrics — no validation accuracy, no discovery rate, no top-K precision values. Since no quantitative results matching the paper's key metrics are presented, the score is 0."

**归因**：这是直接失分原因。`10.5 = 15×0.2 + 25×0.3 + 0×0.5`。若 item 2 拿到哪怕 20 分（方法有缺陷但有真实数值），总分即翻倍。

### F2 ｜ 预算「假耗尽」：预算是按 chat 轮次重置的，但错误消息不告诉 LLM，agent 误判为会话级死刑 【证据确凿 + 一处推断】

- `agent/huginn/agent/streaming.py:1012-1015`：每次 `chat()` 调用都新建 `ToolCallBudget`——**下一轮 chat 预算自动重置**。
- `agent/huginn/tools/adapter.py:723`：喂给 LLM 的错误文本只有 "工具调用预算耗尽： 工具 bash_tool 调用 51 次超过单工具上限 50"，**不含任何「本轮/下轮重置」语义**。
- Material_000 轨迹事件 53–58：agent 把 per-tool 触顶（bash_tool 50 次）与 code_tool 沙箱拦截混为一谈，得出 "Both tool budgets are exhausted" 的全局结论，随即转去写报告。
- 推断：RCB 3-step 结构（Step 2/Step 3 是独立的 `chat()` 调用）下，agent 本可以在下一轮获得全新 150/50 预算继续跑 pipeline；它不知道，是因为预算的作用域与重置语义对 LLM 完全不可见。竞品对照：ResearchHarness 给 agent 的是 **persistent terminal session**（论文 Table 4：`TerminalStart/TerminalWrite/TerminalRead`），长任务在持续终端里跑，不存在「每 N 次调用枪毙一次通道」的设计。

### F3 ｜ 16/16 份 autoloop 自进化报告全部只跑完 perceive 阶段——7 阶段管线从未完整运行过一次 【证据确凿】

对 16 份 `agent/huginn_autoloop_report_loop_*.md` 逐一解析 Phases 表：全部只含 `perceive` 条目（1 或 5 条），耗时 1.6–11.1 秒；`hypothesize/plan/execute/validate/learn/report` 出现次数为 0。例如 `huginn_autoloop_report_loop_16c3481f.md`："Total Time: 2.6s"，5 行 phase 全为 perceive。autoloop 7 阶段定义在 `agent/huginn/autoloop/engine.py:66-75`（`AUTOLOOP_PHASES`）。

**含义**：用户目标的「自我进化」在主循环层面名存实亡——没有一次循环到达 execute/validate/learn。这是「autoloop/unified 工作流」与竞品差距中最硬的一条：竞品（AutoResearchClaw）的 23 阶段管线是逐级强制流转的（含 gate 与回滚），huginn 的 7 阶段是「起了个头就结束」。审计 04 P1-6 确认「4 套状态机 + 3 套多智能体编排 + 2 套自改进系统并行存在，靠 ad-hoc 同步」（`audit_20260717/04_架构_核心智能体循环.md:167`）。

### F4 ｜ Gen-2（3-step + adversarial critique）运行证明：critique 能诊断出真问题，但管线在「修复开始」处死亡 【证据确凿】

`Material_003_20260716_220105`（5.25 分；轨迹含 `=== Step 1/2/3 ===` 标记，确为 3-step runner）：

- 行 72：Step 1（方法论提取）阶段即 "I've hit the tool budget"（per-tool 预算在读文献/数据时触顶）。
- 行 139–159：Step 3 critique **正常工作**——verdict=fix_needed；agent 自查发现 R²=0.862 可疑 → 确认数据泄漏（"16/59 test samples have >0.95 fingerprint similarity"）；替换审计给 graph VAE 打 "**FAIL — silent substitution**"。
- 行 161–173：agent 开始实现真正的 GVAE → 行 173 "The security scanner is blocking `eval()` in the standard GVAE reparameterization trick"（`rcb_runner.py:283-284` 预言过的误杀兑现）。
- 行 175：轨迹在 "Let me run the script with a longer timeout" 处戛然而止；`_meta.json` status 停留在 "running"（进程被杀）——报告未被重写，5.25 分定格。

**含义**：诊断机制（critique）有效，修复机制（执行通道 + 时间/预算余量）缺席。critique 发现的问题越多，需要的执行预算越大；而当前结构把 critique 排在预算耗尽之后。对比 AutoResearchClaw：`repair` 模块是独立 3 循环（`max_cycles: 3`）且有明确的通过门槛（`min_completion_rate: 0.5, min_conditions: 2`）——修复是结构化的管线阶段，不是尾声的良心发现。

### F5 ｜ RCB 权重 0.5 项的能力期望：「训练产生验证性结果」= 执行闭环 + 数值证据 + baseline 对照；官方 rubric 明确奖励诚实失败、惩罚沉默替换 【证据确凿】

`ResearchClawBench/tasks/Material_000/target_study/checklist.json` item 2（weight 0.5，type image）要求三件事同时成立：(a) pre-training loss 从 ~0.25 降到 ~0.05 的收敛曲线；(b) fine-tune validation accuracy 稳定超过 50% 随机基线（~56%）；(c) p>0.9 高置信候选中 true positive 比例（discovery rate）显著高于 base rate（~60%）。`evaluation/score.py` RUBRIC 的 Mode A 刻度把「11-20」定义为 "Quantitative results given but the methodology has fundamental errors"、「1-10」为 "Mentioned but no quantitative results provided"——**只描述不执行封顶 10 分**。

RUBRIC 还有两条直接针对 huginn 失败模式的条款（`evaluation/score.py`）：
- "**Honesty about failures is NOT scored lower than hiding them.** ... A failed-but-honest attempt at the paper's method beats a silent substitution that 'works' on paper."
- "**Implausibly good results are RED FLAGS, not achievements.**"

huginn 的失分恰好踩在这两条上：Material_001 用 GPR 替换 GNN（item 0，w=0.3，score=0："The AI report does not use a GNN at all—it uses Gaussian Process Regression"）；Material_003 用 fingerprint GP 替换 graph VAE（item 0，w=0.3，score=25→judge 指出 "does not implement a graph VAE with dual encoders"，item 1 w=0.3 score=0 "does not perform Bayesian optimization in the VAE latent space"）。

### F6 ｜ 逐环节对比：huginn 弱在「实验设计」「迭代验证」两个环节；「任务分解」「文献 grounding」有骨架未生效；「报告生成」兜底合格 【证据 + 推断】

| 环节 | ResearchClaw / AutoResearchClaw / Claude Science | huginn autoloop（设计） | huginn RCB 实际路径 | 判定 |
|---|---|---|---|---|
| 任务分解 | ResearchClaw：持久 `project→workflow→task→artifact` 状态层；AutoResearchClaw：`PROBLEM_DECOMPOSE` + Planning 层带 "Good Enough?" 验证循环与下游反馈修正 | 7-phase（`engine.py:66-75`）+ ResearchPhase 状态机（`phases.py:25-56`，EXECUTION 300 calls / VALIDATION 100 / 总 ~530） | 3-step runner；phases.py 完全未被启用；autoloop 16/16 perceive-only | **有骨架未生效** |
| 文献 grounding | ResearchClaw：`semantic_scholar_search`/bibtex/paper notes + **claim/evidence graph**（claim 链接论文/实验/产物）；Claude Science：60+ 数据库 | RAG + longterm memory（`audit_20260717/21` §2.5：无 KG schema） | Step 1 单轮 chat 提取 checklist，随对话历史存放，compaction 被跳过（`rcb_runner.py:38-40`）后易丢 | **弱** |
| 实验设计 | AutoResearchClaw：`EXPERIMENT_DESIGN` gate + `RESOURCE_PLANNING` + BenchmarkAgent 4-agent 管线（Surveyor→Selector→Acquirer→Validator，`min_baselines: 2`） | plan phase 存在 | **无此环节**（审计 21 §2.3 确认：无 DOE/对照/显著性模块） | **缺失** |
| 迭代验证 | ResearchClaw：experiment tracking + **contract validation + result bundle validation + 缺 metric/产物自动生成 remediation task**；AutoResearchClaw：`ITERATIVE_REFINE` 自愈 + CodeAgent v2 `hard_validation`（AST 门控，拦截 hardcoded metrics/相同 ablation）+ `exec_fix`（execution-in-the-loop ≤3 次）+ anti-fabrication repair；Claude Science：fact-checker AI + 图绑定生成代码/环境/消息历史 | validate phase（reviewer persona，`engine.py:100-109`） | Step 3 critique 只读 report 文本不重跑代码（`rcb_runner.py:455-462`）；agent 自检 = 循环论证 | **弱**（形式有、执行验证无） |
| 报告生成 | AutoResearchClaw：OUTLINE→DRAFT→PEER_REVIEW（evidence check）→REVISION→`QUALITY_GATE`；图由 FigureAgent 5-agent 管线产出 | report phase（tutor persona） | Step 2.5 兜底 + 自动生成 fallback（`rcb_runner.py:399-434`）——有文件但无数值校验 | **兜底合格** |

### F7 ｜ 「描述代替执行」在竞品中是被结构性机制防止的；huginn 全靠 prompt 规劝 【证据（竞品为一手资料）+ 推断】

竞品防机制（一手资料原文）：
1. **ResearchClaw**：研究状态层持续跟踪每个 workflow 期望的 metrics/outputs/artifact types，缺失即自动生成 **remediation task** 并进入 blocker 面板（"proactive workflow reminders plus remediation tasks for missing metrics, outputs, or artifact types"）——「没跑出 metric」在系统层面是一个待办任务，不是一句可以糊弄的文本。
2. **AutoResearchClaw CodeAgent v2**：`hard_validation: true` —— "AST-based validation gates (blocks identical ablations, hardcoded metrics)"；`exec_fix_max_iterations: 3` —— "Execution-in-the-loop fix attempts"；`repair` —— "Anti-fabrication experiment repair"，`min_completion_rate: 0.5`（≥50% 实验条件必须完成才放行）。报告里的数值若不能由执行产物支撑，在 AST 门控处就被拦下。
3. **Claude Science**：每张图携带 "the exact code and environment that produced it, a plain-language description of how it was created, and the full message history"——证据与产物物理绑定，占位图无法生成。
4. **RCBench 自身**：INSTRUCTIONS（`evaluation/instructions_tmpl.py`）与 rubric 都把 deliverable 锚定在 `report/report.md` + `report/images/*.png` + 数值指标上。

huginn 对应物全部是 prompt 级：system prompt "Prefer real implementations over shortcuts"（`rcb_runner.py:276`）、step3_prompt B 段 "'I used GCNConv instead of CGCNNConv because it was easier' is NOT acceptable"（`rcb_runner.py:509-510`）、审计 21 确认的 completion_auditor 4 层 checklist（`metacog/completion_auditor.py:36-70`，跑分路径未触发）。**竞品的验证是状态机/门控/执行回路；huginn 的验证是请求 LLM 自律**。F1/F4 证明 prompt 级防线在预算压力下必然失守。

### F8 ｜ 背景缺陷（存在但非本任务直接失分原因）【按归因纪律单独列出】

- `agent/huginn/bench/orchestrator.py:67-71` `RCB_DELIVERABLES` 检查 `report.md`/`figures/*.png`/`data/*.csv`（workspace 根目录），与 RCB 实际布局（`report/report.md`/`report/images/`、`data/*.pt`）全部错位 → `_is_done()` 永不满足（`audit_20260717/20` H11）。**但被评分的 8 个 Material 运行全部走 `rcb_runner.py` 路径，BenchmarkOrchestrator 路径（根目录 `rcb_huginn.py`）未产生任何被评分运行**——此为背景缺陷，若未来切回 orchestrator 路径会立即变成直接失分原因。
- 根目录 `rcb_huginn.py:138-155` 的 PHASED PROTOCOL 禁止 "DEEP LEARNING (VAE, transformers, GNNs) until report.md exists"——与 RCB 任务核心方法就是 GNN/VAE 直接冲突；同属背景缺陷（该路径未被评分运行使用）。
- `agent/huginn/phases.py:48-56` 的 phase-aware 预算（EXECUTION 300 / VALIDATION 100）设计合理但 RCB 路径未启用——背景缺陷，属「有骨架未生效」而非「失分原因」。

---

## ③ 根因链（现象 → 机制 → 代码位置）

**链 1：占位报告（Material_000 item 2 = 0，直接失分 ~15 分）**
现象：报告 `Table 2` 为 "Expected" 占位表，`outputs/` 为空。
→ 机制：执行通道双收窄（code_tool 沙箱禁 open/os + bash_tool per-tool 50 次上限），数据加载试错烧光 bash 预算；"预算耗尽"错误消息无重置语义，agent 误判全局死刑，转为文档交付。
→ 代码：`rcb_runner.py:278,283-284`（通道收窄）、`rcb_runner.py:305-306`（预算配置）、`agents/tool_budget.py:50-72`（触顶判定）、`tools/adapter.py:721-723`（无重置语义的错误消息）、`agent/streaming.py:1012-1015`（预算按 chat 轮次新建但 LLM 不可见）。

**链 2：沉默方法替换（Material_001 item 0 = 0、Material_003 item 1 = 0，直接失分）**
现象：GPR 替 GNN、fingerprint GP 替 graph VAE、MAE 49.9K vs 论文 13K。
→ 机制：substitution audit 靠 agent 自证（`rcb_runner.py:504-510`），critique 只读 report 文本不执行代码（`rcb_runner.py:455-462`）；能力上限（deepseek-chat 实现 graph VAE 双编码器困难）与预算耗尽叠加时，"document failures honestly" 被解读为 "document instead of implement"（审计 20 §4.3 同一结论）。
→ 代码：`rcb_runner.py:455-462,493-521`；能力维度见审计 20 §5.4（能力层丢分占 50–60%，推断）。

**链 3：自进化空转（16/16 报告 perceive-only）**
现象：autoloop 报告耗时 1.6–11.1s，只有 perceive。
→ 机制：7 阶段引擎存在但从未完整流转；审计 04 P1-5（`autoloop/engine.py` 7185 行上帝类 + 137 处 except Exception）与 P1-6（4 套状态机并行 ad-hoc 同步）指向工程复杂度失控；具体断点未在本报告范围内定位（推断：perceive 后续阶段的触发条件或依赖未满足）。
→ 代码：`autoloop/engine.py:66-75`（阶段定义）、`agent/huginn_autoloop_report_loop_*.md`（运行证据）。

**链 4：critique 有效但修复无通道（Material_003 Gen-2 = 5.25）**
现象：Step 3 诊断出数据泄漏 + GVAE 缺失，修复中途死亡。
→ 机制：critique 排在执行预算耗尽之后；修复所需的 code_tool 被 AST 扫描器误杀（eval() in reparameterization trick）；bash 脚本超时后进程被杀。
→ 代码：`rcb_runner.py:283-284`（误杀自述）、`_agent_output.jsonl:139-175`（逐步回放）、`_meta.json` status="running"（进程未正常结束）。

---

## ④ 对用户问题的回答

### Q1：逐环节对比，huginn 弱在哪几个环节？

按失分贡献排序：**实验设计（完全缺失）＞ 迭代验证（有形式无执行）＞ 文献 grounding（一次性、易丢、无 claim/evidence 绑定）＞ 任务分解（三套骨架并行但 RCB 路径全绕开）＞ 报告生成（兜底合格，唯一及格环节）**。

必须强调「模式切换」维度：huginn 同时存在 chat/research/plan mode（`agent/core.py:412-477`）、ResearchPhase 六阶段（`phases.py:25-56`）、CSM 八状态（`cognitive_engine.py:30-41`）、autoloop 七阶段（`engine.py:66-75`）四套工作流状态机（审计 04 P1-6），而 RCB 跑分路径是第五套（rcb_runner 3-step）。竞品恰恰相反——ResearchClaw 是一条持久化 workflow 状态链，AutoResearchClaw 是一条 23 阶段单管线。**huginn 的问题不是某条管线设计错了，而是管线丛生、跑分时全不在位**：phases.py 给 EXECUTION 留 300 calls 的设计从未生效，实际生效的是 runner 里硬编码的 150/50。

### Q2：RCB 评分标准（weight 0.5 给「训练产生验证性结果」）说明什么能力期望？

三条：
1. **执行闭环是半壁江山**。单题 checklist 中权重最大项（0.5/1.0）永远给「跑出来的验证性数值 + 图」；Mode A 刻度下「只描述不执行」封顶 10 分（`evaluation/score.py` RUBRIC）。benchmark 的设计意图（论文 §3.4）是把 50 分锚定在「复现论文结果」，而 huginn 目前在「执行出任何数值」这一步就断线。
2. **诚实失败 > 沉默替换**。rubric 明示 failed-but-honest attempt 得分不低于 silent substitution（`evaluation/score.py`）。huginn 的 Material_001/003 失分恰因沉默替换——这是 huginn 的「document failures honestly」prompt 文化在评分端的反噬：agent 把「诚实」用在了替换声明上，而不是用在「尝试原方法并展示真实报错」上。
3. **好得离谱 = 红旗**。rubric 要求 judge 调查优于论文的指标（数据泄漏/错误 split/伪造）。这与 Claude Science 的 fact-checker、AutoResearchClaw 的 hard_validation 同源：**前沿工作流都把「数值必须可被执行产物支撑」作为结构约束**。

### Q3：「描述代替执行」在竞品工作流中被什么机制防止？

四类结构机制（均为一手资料，见 F7）：ResearchClaw 的 **remediation task**（缺 metric/产物 = 自动生成阻塞任务）；AutoResearchClaw 的 **hard_validation AST 门控 + exec_fix 执行回路 + anti-fabrication repair 完成率门槛**；Claude Science 的 **图-代码-环境-消息历史物理绑定 + fact-checker**；RCBench/ResearchHarness 侧的 **persistent terminal + 128k compaction**（保证长任务执行通道不断）。共同点：验证发生在**状态机/门控/执行层**，不是 prompt 层。huginn 目前唯一同级的结构机制是 Step 2.5 的 report.md 兜底（`rcb_runner.py:399-434`）——它保证「有文件」，不保证「文件里有真数值」；要防占位，需要的是「outputs/ 有真 metrics 才允许 report 落盘」这类产物级门控。

---

## ⑤ 可操作建议（按投入产出比排序）

> 与审计 20/21 的建议兼容；以下按「单位改动挽回的分数」排序，P0 均为小时级改动。

| # | 建议 | 类型 | 证据依据 | 预期收益（推断） |
|---|---|---|---|---|
| P0-1 | **预算错误消息加重置语义**：`tools/adapter.py:723` 的 "工具调用预算耗尽" 后补一句 "本轮 chat 预算已尽；下一轮 chat 将重置为 max_calls/max_per_tool，请改用其他工具或结束本轮"。或 RCB 场景直接把 `max_tool_calls_per_tool` 从 50 提至 ≥150（`rcb_runner.py:306`） | 直接失分（链 1） | F1/F2；streaming.py:1012 已每轮新建预算 | 消除「预算假耗尽→占位报告」；Material_000 类运行 item 2 从 0 → 20–40 区间 |
| P0-2 | **RCB 场景放开 code_tool 的 open()/os 白名单 + 修 eval() 误杀**（科学计算必需）；同时把数据加载做成一次性 sniff 脚本模板（检测 zip/pickle 自定义类 → 直接走 stub-module 路径），避免 30 轮试错 | 直接失分（链 1/4） | `rcb_runner.py:278,283-284`；Material_000 轨迹事件 5–38；F4 行 173 | 释放约 1/3 预算给训练；同时为 graph VAE 类实现打开通道 |
| P0-3 | **critique 阶段数值重算**：Step 3 object mode 不止读 report 文本，对每个数值 claim 调 code_tool 重跑一次（接入审计 21 建议的 validate_tool/symbolic_math_tool，tool_filter 加 3 项） | 直接失分（链 2） | `rcb_runner.py:310-315,455-462`；审计 21 §G1/G2 | 把「自检循环论证」变「外部验证」；对应官方 Evidence Mismatch 类失分 |
| P1-1 | **产物级门控**：report.md 落盘前检查 `outputs/` 存在真实 metrics 文件（JSON/CSV），否则禁止进入 report 阶段并生成 blocker 任务——照搬 ResearchClaw remediation task 的最小实现（可挂在 Step 2.5 兜底处，`rcb_runner.py:399-434`） | 结构性防占位 | F7；`score.py` RUBRIC Mode A 1-10 分档 | 从机制上消灭「Expected 占位表」这类交付 |
| P1-2 | **把 RCB 路径切到 phases.py 的分阶段预算**（EXECUTION 300 / VALIDATION 100），或在 runner 内实现同款的阶段隔离预算；若启用 BenchmarkOrchestrator 路径，必须先修 `orchestrator.py:67-71` 的 RCB_DELIVERABLES 路径错位 | 预算结构 | F8；`phases.py:48-56`；审计 20 §5.2（150 calls = 官方 3%） | 执行阶段预算翻倍以上；VALIDATION 有独立预算后 critique 不再挤占执行 |
| P1-3 | **silent substitution 结构性拦截**：Step 1 checklist 落盘为文件（非对话历史），Step 2 结束时由 runner 机械比对「[EXACT] 组件 ↔ code/ 实现痕迹」，缺失即回退执行而非进 critique——对齐 AutoResearchClaw hard_validation 的最小可用版 | 直接失分（链 2） | F5；Material_001/003 judge 原文 | Material_001 item 0 / Material_003 item 1 类 0 分项显著减少 |
| P2-1 | **修通 autoloop**：先让 7 阶段在单个任务上完整跑通一次（当前 16/16 perceive-only），再谈跨任务进化；同时把 4 套状态机收敛为 1 套（审计 04 P1-6） | 自进化根基 | F3 | 用户「自我进化」目标的前置条件；对 benchmark 为间接收益 |
| P2-2 | **引入 persistent terminal 工具**（对标 ResearchHarness TerminalStart/Write/Read），长训练在持续终端里跑，主循环只轮询 | 执行通道 | F2；论文 Table 4 | 消除「每 N 次调用枪毙通道」的结构性摩擦 |
| P3 | judge 与 agent 异源化（审计 16 P1-3）、`_score_batch.py` image 降级修复（P2-2）——评测卫生，不改能力 | 测量精度 | 审计 16 | 分数可比性，非跑分本身 |

**最不该做的**：继续堆 prompt 级规劝（PHASED PROTOCOL、更多的 "NOT acceptable" 段落）。F1–F5 证明防线溃败点全部在结构与预算层，不在措辞层。

---

## 附：关键证据索引

| 编号 | 位置 | 内容 |
|---|---|---|
| V1 | `ResearchClawBench/workspaces/Material_000_20260716_152336/_agent_output.jsonl` 事件 53/56/58/84 | bash 预算耗尽 → "Both tool budgets are exhausted" → 占位报告决策 |
| V2 | 同上 `report/report.md:143,152` | "Expected Candidate Discovery Performance Metrics" 占位表 |
| V3 | 同上 `_score.json` item 2 | weight 0.5，score=0，judge "no validation accuracy, no discovery rate" |
| V4 | `ResearchClawBench/workspaces/Material_003_20260716_220105/_agent_output.jsonl:72,139-175` | 3-step 运行：Step 1 预算触顶；Step 3 诊断有效；修复中途死亡 |
| V5 | `agent/huginn_autoloop_report_loop_*.md` ×16 | 全部 perceive-only，1.6–11.1s |
| V6 | `agent/huginn/tools/adapter.py:721-723` | 预算耗尽错误消息（无重置语义） |
| V7 | `agent/huginn/agent/streaming.py:1012-1015` | 预算按 chat 轮次新建 |
| V8 | `agent/huginn/cli/rcb_runner.py:278,283-284,305-306,310-315,455-462` | 通道收窄、误杀自述、预算、tool_filter、critique 只读文本 |
| V9 | `ResearchClawBench/evaluation/score.py` RUBRIC | Mode A 刻度 + 诚实条款 + 红旗条款 |
| V10 | `ResearchClawBench/tasks/Material_000/target_study/checklist.json` item 2 | weight 0.5 三项数值期望 |
| V11 | arxiv 2606.07591v3 Table 5 / §4.5 | 官方跑分与错误类别分布 |
| V12 | github.com/ymx10086/ResearchClaw README | 持久状态层、remediation tasks、contract/result bundle validation |
| V13 | github.com/aiming-lab/AutoResearchClaw README | 23 阶段管线、hard_validation、exec_fix、repair、PRM gates |
| V14 | TechCrunch 2026-06-30 | Claude Science fact-checker、图-代码-环境绑定 |
| V15 | `agent/huginn/bench/orchestrator.py:67-71` | RCB_DELIVERABLES 路径错位（背景缺陷） |
| V16 | git log `agent/huginn/cli/rcb_runner.py` | 2026-07-16 20:44 入库；两代 runner 时间线 |
