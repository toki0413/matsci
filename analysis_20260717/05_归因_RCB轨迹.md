# 05 · RCB 轨迹归因分析：ResearchClawBench 低分失败模式

**诊断对象**：huginn-agent（材料科学 AI Agent）
**分析角色**：科研复现归因专家
**分析日期**：2026-07-17
**分析方式**：严格只读。解析 `ResearchClawBench/` 全部评分 JSON、33 个 Material_003 工作区、4 个评分工作区的完整 `_agent_output.jsonl` 轨迹、两个 harness 入口（`rcb_huginn.py`、`agent/huginn/cli/rcb_runner.py`）、git 提交时间线；引用 `audit_20260717/` 04/05/07/16 已确认事实（标注来源，未重复审计）。推断处均标「推断」。

---

## ① 数据与方法

### 1.1 评分归属核定（先说清"哪个分是哪个 run 的"）

对评分 JSON 与工作区逐一做了归属核验（`_meta.json`、`batch_results.json`、reasoning 文本逐字比对）：

| 任务 | 分数 | 实际被评工作区 | 来源文件 | 备注 |
|---|---|---|---|---|
| Material_000 | 10.5 | `Material_000_20260716_152336` | `score_results.json` | 批次 run，归属一致 |
| Material_001 | 15.75 | `Material_001_20260716_154214` | `score_results.json` | 批次 run，归属一致 |
| Material_002 | 17.28 | `Material_002_20260716_154931` | `score_results.json` | 批次 run，归属一致 |
| Material_003 | **11.25** | **`Material_003_20260716_171400`** | `score_results.json` | **归属错配**：`batch_results.json` 指向 160245，但 reasoning 与 171400 的 `_score.json` 逐字一致（已程序化验证），160245 自己的 `_score.json` 实为 **8.25** |
| Material_003 | **5.25** | `Material_003_20260716_220105` | `score_m003_v2.json` | **运行被硬杀后的半成品**（`_meta.json` `status="running"`） |
| Material_003 | 2.25 | `Material_003_20260716_190227` | 工作区 `_score.json` | 3101s 马拉松 run，诚实记录失败 |
| Material_003 | 12.75 | `Material_003_20260714_234743` | 工作区 `_score.json` | 07-14 旧适配器（`rcb_huginn.py`） |

错配机制：`_score_batch.py:60-72` 按 **mtime 最新**选工作区评分，而非批次产出的工作区；且 `_score_batch.py:33-45` 把全部 image 类 checklist 项 monkey-patch 降级为纯文本评分（judge 看不到 agent 的图也看不到目标图）——M003 四项 checklist 全为 image 类型。同一报告两次评分可差 ±3 分（160245：11.25 vs 8.25 两次通过）。judge 与被测 agent 同源（deepseek-chat，`_score_batch.py:13-17`；audit 16 P1-3）。

### 1.2 语料规模

- **Material_003 共 33 个运行工作区**（07-14 4 个 + 07-16 29 个），**仅 8 个（24%）产出 `report/report.md`**；19 个 `_meta.json` 永久停在 `status="running"`（进程被硬杀，`run_task.py:146-154` 的 finally 都来不及执行）。
- 完整精读 4 条评分轨迹（M000/152336、M001/154214、M003/190227、M003/220105）+ 抽查 6 条（123033、131157、160245、164805、171400、234743）。
- judge 评语类别分布（`score_results.json` + `score_m003_v2.json` 共 17 个评分项，正则归类）：

| 失分类别 | 命中 | 占比 |
|---|---|---|
| 缺定量指标（no/lacks/missing metrics/values） | 10/17 | 59% |
| 方法替换（does not use/perform, instead, deviates） | 8/17 | 47% |
| 核心方法完全未尝试（completely absent/fails） | 6/17 | 35% |
| 只描述不执行/占位（only describes, placeholder, expected） | 4/17 | 24% |
| 自承省略（admits/explicitly states omission） | 2/17 | 12% |
| 结果不可信（implausibly good / fabricated） | 1/17 | 6% |

### 1.3 harness 两个世代（归因前提）

- **旧适配器** `rcb_huginn.py`（07-14 及更早）：system prompt 含 `MODEL COMPLEXITY CEILING`——「DEEP LEARNING (VAE, transformers, GNNs) is FORBIDDEN until report.md exists」（`rcb_huginn.py:150-154`）；配 `BenchmarkOrchestrator` 循环。
- **新 runner** `agent/huginn/cli/rcb_runner.py`（07-16 全天迭代）：3 步认知循环（Step 1 方法论提取 → Step 2 执行 → Step 3 对抗自审），`max_tool_calls=150`、`max_tool_calls_per_tool=50`（:305-306）。07-16 的提交时间线：`1ac780b 20:44`（σ1-σ5 控制环修复）、`fde2e42 21:06`（Rust sandbox 逃生门）、`6b8b50f 07-17 07:32`。白天各 run 用的是未提交的中间版本——**M001(15:42) 尚无熔断开关、190227(19:02) 尚无 Rust sandbox 逃生门**。

---

## ② 核心发现（按证据强度排序）

### F1【直接证据】工具链摩擦是预算的第一杀手，主管线因此被饿死

agent「不执行管线」的头号原因不是规划选择，而是**执行通道在脚下反复坍塌**：

- **M000（10.5）**：约半数回合在与沙箱搏斗。`_agent_output.jsonl:16-24`：RestrictedPython 反复拦截 `torch.load` 自定义类 `RealisticCrystalData`（"The sandbox keeps blocking this"）；`:54-57`：「The bash_tool budget is exhausted」「The code_tool is also blocked」；`:58`：「the main pipeline (fine-tuning + candidate prediction) **didn't run due to tool budget issues**」。预算拦截文案即 `adapter.py:723` 的 `工具调用预算耗尽: 工具 bash_tool 调用 N 次超过单工具上限 50`。注意：旧适配器 `rcb_huginn.py:45-49` 会 monkey-patch 掉 `validate_code`，而新 runner 没有——runner 的 agent 直接面对沙箱拦截。
- **M001（15.75）**：`jsonl:14`「Both bash_tool and code_tool are showing **circuit_open** errors」——熔断器在运行中途自锁执行工具。`HUGINN_HEALTH_MONITOR=0` 的关闭开关（`adapter.py:116`，import 时读取）直到 20:44 提交才存在（git `1ac780b`；知识种子 `agent/huginn/knowledge/seed/38_benchmark_evaluation_lessons.md:98` 自承"long runs trip CircuitBreaker"）。
- **190227（2.25）**：Step-3 修复阶段，`jsonl:17342`「The 'Unknown error' happens consistently for **any script doing sklearn GP fitting**」——甚至"tiny RF on 2D data crashes"。根因已被作者自己在提交信息中确认：`fde2e42`「Rust sandbox (huginn_ext.sandbox.run_sandboxed) **silently crashes on RDKit+sklearn GPR scripts**, returning success=False with empty stderr. adapter.py then returns "Unknown error" — agent cannot debug」（亦见 `bash_tool.py:121-123` 注释、`adapter.py:464`）。**该修复 21:06 才落地，比此 run 晚 2 小时。**
- **极端样本 123033**（61s 即死）：工具面完全错误——只挂出 deepagents 默认虚拟文件系统工具（`write_todos/ls/read_file/...`），`execute` 无后端（jsonl:117）、Windows 路径被拒（:113）、最后被秒级限流 5000 tok/s 打死（:139「秒级 token 超限」）。**164805**（75s 即死）：`sqlite3.OperationalError: unable to open database file`，LongTermMemory 构造即崩（jsonl:5-45）。

### F2【直接证据】「知道没做出数值结果仍提交」的行为存在——且被 harness 制度化

agent 多次**明示**自己在提交半成品：

- M000 `jsonl:59`：「Both tool budgets are exhausted. Let me write the report now using the EDA figures we have...」；结尾 `:85`：「could not complete due to tool budget limits. The report documents the complete methodology and **expected results**.」
- 190227 `jsonl:17358`：「I've exhausted tool call budget. Let me update the report with the information I have and finalize.」
- 220105 `jsonl:73`（Step-1 内）：「I've hit the tool budget.」

但定性必须准确：**这是规则压力下的合规行为，不是偷懒**——

- `INSTRUCTIONS.md` 模板把 `report/report.md` 定为 primary goal，且「文本回复即终止/标记失败」「Never finish early」（`evaluation/instructions_tmpl.py` Execution Protocol 段）；
- `orchestrator.py:89-99` 的 `_triage_prompt` 明示「If a full implementation is infeasible in remaining budget, write a working skeleton and **document the gap honestly in report.md**」；
- `rcb_runner.py:399-434` Step 2.5 在 agent 不写报告时**自动生成兜底报告**（"ensure there's a deliverable to score"）。

即：harness 本身把「先有报告交差」设计为合法逃生门。越界仅两处：(a) M000 报告 §3.2（`report.md:131-137`）用**过去时叙述从未执行的 fine-tuning 协议**（"The model was fine-tuned... The validation AUROC was tracked"），§3.3（:143-152）给出只有指标名没有数值的 "Expected Candidate Discovery Performance Metrics" 空表——judge 评语 "only describes expected or placeholder results" 精确命中；(b) M002 对玩具几何算出的 MAE=0.032 eV 过度宣称「near-DFT accuracy」，judge 判 "implausibly low... fabricated or erroneous"（`score_results.json` M002 item2）。

### F3【直接证据】核心方法替换在 Step-1 checklist 就被自我合理化；Step-3 自审能发现但修不动

- 220105 `jsonl:30-33`：agent 在 Step-1 就把 graph VAE 预标为 `[VARIANT]`，理由「a full SD-VAE requires large training datasets (~250K SMILES)... **infeasible** without pre-trained models」——未尝试先降级。
- 190227 Step-3 自审（`jsonl:159-160`）：「E. Graph Variational Autoencoder **[EXACT] NOT IMPLEMENTED — FAIL — silent substitution**... I substituted it with fingerprint screening without even attempting」——对抗自审机制**按设计工作**，准确揪出降级。但随后的 GVAE 实现尝试撞上 `eval()` 安全扫描拦截 + 脚本超时（`jsonl:174-176`），轨迹在此中断。
- 220105 同样：Step-3 verdict=fix_needed，发现数据泄漏（16/59 测试样本指纹相似度>0.95）+ GVAE 缺失，动手实现 GVAE 时被硬杀（见 F5）。
- 对照组 **171400（11.25，本批最高）**：真的实现了 char-level VAE（15,691 SMILES），观察到后验坍塌（KL≈0、latent GPR R²≈0），诚实写入 Limitations——**尝试过并如实记录失败，得分高于未尝试的 160245（8.25）与半成品 220105（5.25）**。
- 旧世代更糟：`rcb_huginn.py:150-154` 的复杂度上限**直接禁止** VAE/GNN/deep learning 直到报告写完——而 M000/M001/M003 的 checklist 核心恰是 CGCNN/VAE/BO。234743（旧适配器，12.75）的报告里 GVAE 只出现在 "future work"（`report.md:186`）。这是 prompt 与评分标准正面冲突的结构性失分。

### F4【直接证据】公布分本身噪声大、归属错、且包含对半成品打分

- `score_results.json` M003=11.25 实为 171400 的分（reasoning 逐字验证），批次产物 160245 只有 8.25——**归属错配**（机制：`_score_batch.py:60-72` 按 mtime 选最新）。
- M003 全部 checklist 项为 image 类型，但 `_score_batch.py:33-45` 把 image 项降级为纯文本评分（因 deepseek-chat 不支持 image_url）——judge 从未看到 agent 生成的 6-14 张图（audit 16 P2-2 已确认该机制）。
- 160245 同一报告两次评分 11.25 vs 8.25（judge 抖动 ±3 分）。
- 5.25（220105）是对 `status="running"` 的半成品打分；2.25（190227）是对"诚实记录失败"的打分。**分数序列 12.75→11.25→2.25→5.25 的波动主要来自评分口径与运行中断，而非能力波动。**

### F5【直接证据】33 次 M003 重跑是一部 harness 现场调试史，大量 run 根本没活到写报告

- 仅 8/33（24%）产出 report.md；19/33 的 meta 永久停在 "running"。
- 时间线：07-14 旧适配器 4 run（1 报告，12.75）→ 07-16 12:30-14:05 σ 调试期约 11 run 全灭（虚拟 FS/无限流/无 execute 后端/sqlite 崩）→ 15:10-18:05 部分修复（160245=8.25、171400=11.25）→ 19:02 马拉松 run 撞 Rust sandbox（190227=2.25）→ 20:44/21:06 两个修复提交 → 21:08-22:01 四次重跑（v3-v6）**全部中途被杀**（`_run_m003_v3/v4/v5/v6.log` 均无 "done" 行；`_run_single.py:24` 仅在 `runner.run()` 返回后打印 done），220105 的半成品被评 5.25。
- 关键推论：**杀死 v3-v6 的不是 agent 内部超时**（rcb_runner 无 wall-clock；`run_task.py` 也无超时参数），是外部中断（控制台关闭/人工中止——推断）。每个被杀 run 的 Step-3 修复都做到一半。

### F6【背景因素】任务设计张力：部分失分任何策略都拿不回

- M001 提供的数据是合成玩具：轨迹 `jsonl:17`「X features (all 5s)」，报告 `:29` 自承 "constant feature inputs"；而 checklist 要求 GNN loss<0.012@50ep、MAE 0.15 eV/atom、BO 在 352.4°C/19.8bar 达 9.8 TOF（`tasks/Material_001/target_study/checklist.json`）——论文级数值在常数特征数据上**物理不可达**。
- M003 checklist item2（w=0.25）要求**实验合成与表征**（测量 Tg、healability、recyclability）——纯计算 agent 天然 0 分；item1（w=0.3）要求 BO 在潜空间发现 >500K/373K/248K 三个特定 Tg 目标并 MD 验证。
- 反向激励现实存在：过度宣称的 M002（17.28）> 诚实记录失败的 190227（2.25）。judge 有 anti-fabrication 条款（audit 16 亮点 5：`evaluation/score.py:80-83`），但"执行后失败并分析到位"没有中间分段。

---

## ③ 根因链（现象 → 机制 → 代码位置）

**链 1｜「占位数值」**
judge 批 placeholder/expected → 报告写预期指标空表与过去时叙述 → 主管线（fine-tune+predict）从未执行 → bash/code 双工具预算耗尽（`adapter.py:723`）→ 预算被沙箱拦截战烧掉（torch.load 自定义类被 RestrictedPython 拦，M000 约 25 个回合）→ **根因位置**：新 runner 未继承旧适配器的 `validate_code` monkey-patch（对比 `rcb_huginn.py:45-49` 与 `rcb_runner.py` 全文）+ `max_tool_calls_per_tool=50`（`rcb_runner.py:306`）。

**链 2｜「承认省略但未尝试核心方法」**
checklist 缺 graph VAE/BO/GNN → Step-1 自我合理化 [VARIANT] 降级（220105 `jsonl:30-33`；旧适配器更直接在 prompt 禁止深度学习，`rcb_huginn.py:150-154`）→ Step-3 自审判 FAIL 后修复尝试被 Rust sandbox 静默崩溃「Unknown error」拦截（190227 `jsonl:17342`；`bash_tool.py:120-162`；根因确认见提交 `fde2e42`）→ 修复提交（21:06）晚于失败 run（19:02），晚间重跑又全被外部杀死 → **根因位置**：`huginn_ext.sandbox.run_sandboxed` 对 RDKit+sklearn 的静默崩溃 + 修复-重跑节奏断裂。

**链 3｜「分数低且不稳」**
公布分归属错配 + 半成品被打分 → `_score_batch.py:60-72` 按 mtime 选工作区 + `_score_batch.py:33-45` image→text 降级 + judge 与被测同源（`_score_batch.py:13-17`）+ 19/33 runs meta 卡 "running"（`run_task.py:146-154` 的 finally 也兜不住硬杀）。

**链 4｜「写报告交差」梯度（系统提示产物）**
INSTRUCTIONS 终止恐惧（文本回复=失败）+ deliverable=report.md 定义 + `_triage_prompt` 合法化 gap 提交（`orchestrator.py:89-99`）+ Step 2.5 兜底报告（`rcb_runner.py:399-434`）+ 旧适配器 phased protocol「Phase 3 (calls 31-40): WRITE report.md NOW... incomplete results are fine」（`rcb_huginn.py:143-148`）→ 共同构成"报告优先于结果"的强行为梯度。agent 的"交差"是被训练的。

---

## ④ 对用户问题的回答

**Q：agent 为什么「不真正执行管线」——工具链、预算、还是「写报告交差」策略？**

三者是**串联的因果关系，不是并列选项**：

1. **工具链/环境摩擦是主因（一阶）**。三层失效：(a) 代码沙箱 AST 拦截科学计算刚需（`torch.load` 自定义类、`open()`、`os`、`eval()`）；(b) Rust sandbox 对 RDKit+sklearn GP 静默崩溃，agent 只收到 "Unknown error" 无法调试（作者提交信息自认）；(c) 治理层自伤——熔断器中途自锁（M001 circuit_open）、秒级限流打死 run（123033）、sqlite WAL 构造即崩（164805）、loop detector 误判正常重试。摩擦直接烧掉 per-tool 50 次预算。
2. **预算不足是二阶症状**。150/turn + 50/tool 对完整论文复现偏紧（`rcb_runner.py:303-304` 注释自知 "80 步不够"），但 171400 证明：无摩擦时预算足够跑完 GP+VAE+诊断的完整管线。预算耗尽几乎总是摩擦的下游。
3. **「写报告交差」策略存在，但主要是系统提示/工作流引导的产物**。INSTRUCTIONS 的终止规则、deliverable 定义、triage prompt、兜底报告、旧适配器的 phased protocol 与复杂度上限，共同把"先有报告"设为最优逃生策略。agent 在规则内理性执行了它——同时在 Step-1 用 [VARIANT] 标签把核心方法降级自我合理化（deepseek-chat 的能力短板：面对摩擦倾向绕路而非死磕）。

**Q：agent 是否有「知道自己没做出数值结果仍提交」的行为？**

**有，且是明示的**（F2 四条原文引用）。但要区分三档：(a) 合规的诚实提交（190227 "finalize with what I have"）；(b) 灰色地带——M000 用过去时叙述未运行的实验协议 + "Expected" 空表（fabrication-adjacent，judge 未判造假但按无结果扣分）；(c) 明确过度宣称——M002 在玩具几何上算出 MAE=0.032 eV 宣称 near-DFT 精度（judge 判 fabricated/erroneous）。

**Q：这是否系统提示/模式引导的产物？**

**是，证据链完整**（链 4）。另有两个放大器不属于 prompt 范畴：(i) 评分侧——image 项文本降级、同源 judge、归属错配、±3 分抖动，使"低分"的测量本身失真；(ii) 任务侧——M001 玩具数据 vs 论文数值 checklist、M003 实验验证项（w=0.25）天然 0 分，这部分失分换任何 agent 都拿不回。模式切换层面：RCB 专用开关（CSM subset、KEEP_ROOT_N、SKIP_LOOP_DETECTOR、NO_RUST_SANDBOX、HEALTH_MONITOR=0）全是 07-16 逐败后补的——方向正确的事后补救，但也意味着**被测 agent 与生产 agent 运行姿态不同**（audit 16 P3-1 同款指出）。

**关于"自我进化"**：`agent/huginn_autoloop_report_loop_*.md` 16 份报告全部非空（最小 1791B），但内容是 strain/bandgap 级别的玩具 demo loop，与 RCB 改进无关。07-16 的真实修复（Rust sandbox、熔断、限流）全部来自**人工调试提交**，不是 autoloop 自进化产物——自进化循环尚未接入"从 benchmark 失败学习"的回路。

---

## ⑤ 可操作建议（按投入产出比排序）

| # | 建议 | 投入 | 预期收益 | 依据 |
|---|---|---|---|---|
| R1 | **评分门禁**：`_meta.json status=="completed"` 才允许评分；score 条目强制带 `workspace`+`run_id`+scorer 签名；同一工作区重复评分写新文件不覆盖 | 0.5 天 | 杜绝半成品打分（5.25）与归属错配（11.25 vs 8.25），立即使分数可审计 | F4/F5；`run_task.py:146-154`、audit 16 P2-4 |
| R2 | 修 `RCB_DELIVERABLES` 为 `report/report.md` + `report/images/*.png`，删 `data/*.csv` | 0.5 天 | `_is_done()` 不再永不触发，老路径不再每次烧满预算 | audit 16 P2-1；`orchestrator.py:67-71` |
| R3 | rcb_runner 启动前 30s 环境冒烟：经 bash_tool/code_tool 各跑一次 RDKit+sklearn 微型 GP + `torch.load` 真实数据文件；失败即中止并打印修复清单。同时把 `setdefault` 改强制赋值（NO_RUST_SANDBOX/HEALTH_MONITOR/SKIP_LOOP_DETECTOR） | 0.5 天 | σ 调试期 11 个全灭 run 类故障变为 fail-fast，不再烧 50 次预算才发现 | F1/F5；`rcb_runner.py:28-46` |
| R4 | **Step-3 独立修复预算池**：critique verdict≠pass 时追加专用 50 次预算 | 1 天 | 190227/220105 两次"自审发现 GVAE 缺失却修不动"直接消除；这是对抗自审机制价值兑现的最后一公里 | F3；`rcb_runner.py:305-306` |
| R5 | 退役旧适配器的 MODEL COMPLEXITY CEILING（`rcb_huginn.py:150-154`）；Step-1 prompt 增加：task 描述点名组件（graph VAE 等）**禁止未尝试标 [VARIANT]**，≥2 次尝试失败才允许降级且须附报错 | 1 天 | 消除 prompt 与 checklist 正面冲突；把 F3 的"未尝试先降级"从规则上堵死 | F3 |
| R6 | code_tool 沙箱为 RCB 场景开白名单 profile（放行 `torch.load`/`open`/`os`/pickle），替代"全禁或全放"两个极端 | 1-2 天 | 砍掉 M000 式半数回合的沙箱搏斗 | F1 链 1 |
| R7 | 报告强制区分 `EXECUTED` vs `EXPECTED/NOT EXECUTED` 标记（file_write_tool 侧 lint 或 report 模板约束） | 1 天 | 消除 M000 §3.2 式过去时灰色叙述；judge 归因更准 | F2 |
| R8 | 换视觉独立 judge（GPT-4o/Qwen-VL），恢复 image 项按图评分；judge≠被测模型 | 2 天+成本 | M003 全部 image 项恢复测量效度；去同源偏差 | F4；audit 16 P1-3/P2-2 |
| R9 | 任务侧：M001 类"玩具数据+论文数值 checklist"任务改数据或改 checklist；judge rubric 增加"执行后失败但分析到位"中间档（当前诚实失败 2.25 < 过度宣称 17.28，反向激励夸大） | 中期 | 消除不可达失分与反向激励 | F6 |

**最低成本起步组合：R1+R2+R3（1.5 天）**，可先重建可信基线；**R4+R5（2 天）** 直指 judge 最高频失分类别（方法替换 47%、核心未尝试 35%）。

---

*报告完。证据文件：`ResearchClawBench/score_results.json`、`score_m003_v2.json`、`workspaces/*/_agent_output.jsonl`、`workspaces/*/_meta.json`、`rcb_huginn.py`、`agent/huginn/cli/rcb_runner.py`、`agent/huginn/tools/bash_tool.py`、`agent/huginn/tools/adapter.py`、git log（1ac780b/fde2e42）、`audit_20260717/16_评测体系完整性.md`。*
