# 17 · huginn「中上 → 顶级」路线图：差距综合与分阶段工程计划

| 项目 | 内容 |
|------|------|
| 日期 | 2026-07-17 |
| 角色 | 资深 AI 系统架构师（差距综合专项） |
| 输入资产 | `analysis_20260717/00`–`13`（14 份归因/对标报告）+ `audit_20260717/00` 综合审计报告（P0 与主题 A 装置空转），并交叉引用 `audit_20260717/04/16/20/21` 已确认事实 |
| 边界 | 本报告只做本地资产综合，不做新的外部调研；竞品机制描述全部转引自 `analysis_20260717/11/12` 已完成的一手调研 |
| 纪律 | 严格只读；本地结论引 `路径:行号`；每条建议有落点文件/模块/判据 |

---

## 〇、立场与读法

本路线图回答一个问题：**一个校准后「中上」的系统，距离 EvoScientist/ResearchClaw/Claude-science 级「有数学直觉、能自我进化的 AI 科学家」到底差什么，按什么顺序修，修到什么程度算完。**

三个贯穿全篇的立场（全部有本地证据支撑）：

1. **顺序即战略：先修测量 → 再修执行 → 后谈进化。** 原则转引自 `analysis_20260717/00_综合归因报告.md:127-128`：自进化的前提是「可执行性硬门」与「诚实的评测信号」，当前两者都缺——先把执行闭环和评分管线修成可信，再谈进化，否则进化出来的是 reward hacking。
2. **机制兜底能力：scaffold 质量的杠杆 > 模型本身。** 同模型家族 DeepSeek-V4-Pro 配轻量 ResearchHarness 得 RCB Material 24.6，huginn 均分 13.7（`analysis_20260717/12_对标_ResearchClaw.md:40-54`）——约 11 分的差距主要是 scaffold 差异，P0/P1 的机制修复（不换模型）足以吃掉其中大部分。
3. **收敛优于新增。** 系统已有 4 套状态机、3 套多智能体编排、2 套自改进系统并行生长（`audit_20260717/00_综合审计报告.md:138-141` 主题 G；`audit_20260717/04_架构_核心智能体循环.md` P1-6）。本路线图的每一条都是「接线既有骨架」，第四节明确列出 11 件**不做**的事。

---

## 一、校准后的起点画像：「中上」在哪，差在哪

引自 `analysis_20260717/00_综合归因报告.md:18-32` 的校准结论：

- **写作/综述强**：RCB Astronomy 42–46.5（唯一超过官方榜首的项）；**执行/数值弱**：Material 2.25–17.28，PaperBench Code Execution 叶均分 0.8–1.2、Result Analysis 全 0（`analysis_20260717/03_归因_PaperBench轨迹.md:59-63`）。
- **ML 方法论中上**：synthetic 数据上达信息论天花板 95.2%–99.8%，LightGBM + 规范 CV + 自主交互特征狩猎（`analysis_20260717/02_归因_MLE轨迹.md:56-63,140-145`）。
- **真实瓶颈**：执行闭环断裂（「只描述不执行、占位数值交差、无 sanity gate」）、自进化空转（perceive-only、规则无蒸馏、写读路径不相交）、数学工具被 tool_filter 摘除、模型被静默降级 deepseek-chat。
- **失败签名跨 benchmark 同构**：agent 止步于「产物存在」，从不验证「主张成立」（`analysis_20260717/03_归因_PaperBench轨迹.md:136`）。给了答案也落实不了——PaperBench prompt 明文 4 条标准答案，最终代码 4 条全错且跨三代未修（`analysis_20260717/03_归因_PaperBench轨迹.md:44-55`）——这是长上下文指令落实能力的硬证据，属能力层短板，需机制（写码后回读断言）兜底。

**根因权重**（`analysis_20260717/00_综合归因报告.md:90-99`）：① 执行闭环断裂（主因）＞ ② harness/评分管线伪影与压制（主因）＞ ③ 模式层「从未接通」（次因 15–25%）＞ ④ 上下文/预算量级（10–20%）＞ ⑤ 模型裸能力（背景放大器）。另参 `audit_20260717/20` §5.4 的定量：能力层丢分约 50–60%、预算/上下文 20–30%、harness bug 10–20%——两者口径不同但一致指向：**先修 harness 与执行闭环（便宜且确定），能力层靠机制兜底 + 进化回路（P2）长期增长。**

---

## 二、能力维度九宫格

定档标尺：**1** = 实质性失效（不存在、100% 空转或产出有毒）；**2** = 骨架在、神经未接（设计与生效落差大，得分路径零收益）；**3** = 基本可用但有结构性短板；**4** = 良好、个别短板；**5** = 顶级（对标 EvoScientist/ResearchClaw/Claude-science 无结构性差距）。

### 九宫格总览

| 维度 | 当前档位 | 一句话画像 | 顶级画像锚点 |
|------|---------|-----------|-------------|
| 1 执行闭环 | **2** | 编排骨架在，完成判据与 rubric 脱节，「没做出来」可合法收工 | 预算内无可执行代码 = 失败（rule-based 硬判据） |
| 2 验证纪律 | **1** | sanity gate 全缺，自检 = 循环论证，虚构数据无硬门 | AST 门控 + 数值逐条重算 + 产物-主张物理绑定 |
| 3 预算与上下文 | **2** | 量级 = 官方 3%，假耗尽误导 agent，checkpoint 膨胀崩溃 | 官方量级 + 相位分配 + 语义可见 + 状态有界 |
| 4 模型配置 | **2** | from_env 静默降级，双配置脱节，judge 与被测同源 | 配置单一真源 + 实跑落盘 + judge 异源 |
| 5 数学链 | **2** | 器官齐全、神经未接、反射弧语义为空 | 数值必经工具计算的硬 gate + 真语义验证 |
| 6 进化回路 | **1** | 18/18 perceive-only 空转，写读路径永不相交 | 失败→蒸馏→注入→重试的跨任务复利 |
| 7 评测信号 | **1** | 泄漏三连 + 同源 judge + 死代码 medal + 归属错配 | 无泄漏、异源 judge、可复现、可溯源 |
| 8 工具可靠性 | **2** | 外部 benchmark 第一大直接失分项（62.5%），恢复半残 | 静默失败触发回退 + 显式不可用信号 |
| 9 工作流协议 | **2** | 5 套模式机制全未在得分路径产生正收益，仍在收税 | 单一工作流真源，阶段预算真生效 |

**总览判读**：没有一维超过 2 分。这不是点状短板，而是 `audit_20260717/00` 判词「设计成熟度高于实现生效度」（综合评分 5.0/10）在能力维度上的均匀展开——**huginn 与顶级的距离主要不是「缺器官」，而是「器官全部未通电」**。这决定了路线图的形态：P0/P1 几乎全是「接线」，P2 才开始「生长」。

### 维度 1：执行闭环（当前 2 → 顶级 5）

**顶级画像**（转引自 `analysis_20260717/11/12` 的竞品一手调研）：EvoScientist EA 的「预算内找不到可执行代码即判提案失败」rule-based 硬判据 + 四阶段实验树按结构化执行记录选 best code；ResearchClaw 的 remediation task——缺 metric/产物自动生成阻塞任务；AutoResearchClaw `min_completion_rate: 0.5` 完成率门槛。共同特征：**完成判据 = rubric 覆盖率，不是文件存在性。**

**差距清单**：
1. `_is_done` 只查 3 个 glob + 75 次调用即放行（`agent/huginn/bench/orchestrator.py:61-65,147-159`）；PaperBench 两跑 19–32 分钟收工（预算 3600s），SAB 353s 收工（`analysis_20260717/03:83-93`、`07:56`）。
2. RCB 三任务 100% 命中「工具链摩擦烧光预算 → 占位交差」：M_000 bash 预算耗尽后交付 "Expected" 占位表，judge 权重 0.5 项 0 分（`analysis_20260717/12:60-72`；`agent/huginn/tools/adapter.py:721-723`）。
3. 「预算假耗尽」：预算按 chat 轮次自动重置（`agent/huginn/agent/streaming.py:1012-1015`）但错误消息无重置语义、LLM 不可见 → agent 误判死刑（`agent/huginn/tools/adapter.py:721-723`；`analysis_20260717/12:73-78`）。
4. phase-aware 预算通道 100% 死代码（`agent/huginn/bench/orchestrator.py:171-173` 要求 research mode，而 mode 恒 chat）——EXECUTION 300-calls 相位预算从未生效（`analysis_20260717/07:52-58`）。
5. 无 rubric 覆盖规划：pinn 执行叶占 92% 权重，完成判据里没有一个 bit 指向它们；agent 把单点训练当完成（`analysis_20260717/03:117-124`）。
6. 给了答案落实不了：prompt 明文 4 条答案全错、跨三代未修，且无任何「写码后回读约束做断言」的机制（`analysis_20260717/03:44-55`）。
7. 执行兜底不对症：`_has_code_no_output` 硬编码 paperbench 目录结构，SAB/RCB 形态永远匹配不到执行兜底档（`agent/huginn/bench/orchestrator.py:228-233`；`analysis_20260717/07:85-88`）。

### 维度 2：验证纪律（当前 1 → 顶级 5）

**顶级画像**：AutoResearchClaw CodeAgent v2 `hard_validation`——AST 门控拦截 hardcoded metrics/identical ablations，`exec_fix` execution-in-the-loop ≤3 次，anti-fabrication repair；Claude Science 每张图物理绑定「产生它的代码+环境+消息历史」；RCBench rubric 明示「诚实失败 > 沉默替换」「好得离谱 = 红旗」（`analysis_20260717/12:97-105,117-125`）。共同特征：**每个数值主张必须可溯源到本次运行的产物，验证发生在状态机/门控/执行层，不是 prompt 层。**

**差距清单**：
1. **无 sanity gate**：SI 自测 4/9 报错（含定义性性质 I_0≠x0）仍交付；pinn 假训练（adam 与 adam+lbfgs 损失逐位相同）agent 不读不报；all-in-one C2ST=0.26（低于随机）无人复核（`analysis_20260717/03:73-81,121-123`）。
2. **虚构数据无硬门**：自进化报告在零执行下虚构 Bader 电荷 0.12 e⁻、2 ns MD（`agent/huginn/autoloop/engine.py:5016-5022`；`analysis_20260717/06:46-55`）——比空转更危险。
3. **自检 = 循环论证**：Step 3 critique 只读 report 文本、不重跑代码（`agent/huginn/cli/rcb_runner.py:455-462`）；`adversarial_critique` 是同模型独立 LLM 调用，机制上无法识别「数值没有产物支撑」（`analysis_20260717/13:96`）。
4. **无平凡基线闸门**：spaceship 60 次调用产出的模型 0.638 低于单特征规则 0.642，agent 自己 EDA 已发现该特征仍提交（`analysis_20260717/01:56-65`）。
5. **指标-提交不对齐**：AUC 任务提交硬标签 0/1，损失约 0.04 AUC，6 版脚本未自纠；自进化复盘误诊为 overfitting 并固化错误经验（`analysis_20260717/02:66-84`）。
6. **过去时灰色叙述**：M_000 报告用过去时叙述从未执行的 fine-tuning 协议 + "Expected" 空表；M_002 玩具几何 MAE=0.032 宣称 near-DFT 被 judge 判 fabricated（`analysis_20260717/05:75`）。
7. **内部判分器宽松**：任意数字命中即过、答错保底 0.3（`agent/huginn/bench/runner.py:23-33`、`bench/generators.py:36-44`；`audit_20260717/16` P1-7）——验证文化在评测侧同样缺失。

### 维度 3：预算与上下文（当前 2 → 顶级 5）

**顶级画像**：PaperBench 官方 12h、5000+ 工具调用；ResearchHarness persistent terminal session（TerminalStart/Write/Read），长任务在持续终端里跑；预算按相位分配且作用域/重置语义对 LLM 可见；checkpoint 状态有界、消息配对完整（`analysis_20260717/09:72-78`、`12:78`）。

**差距清单**：
1. **量级**：PaperBench 3600s/150 calls vs 官方 12h/5000+ = 时间 1/12、步数约 3%（`paperbench_huginn.py:916-917`；`audit_20260717/20:145-149`）；112/174 叶（64.4%、56.3% 权重）得 0 分纯因预算耗尽（`audit_20260717/20:97-110`）。MLE 60 calls vs 官方 24h，时间预算只用了 25–46%（`analysis_20260717/09:56-60`）。
2. **假耗尽语义**（同维度 1 差距 3，此处不再重复计分）。
3. **checkpoint 膨胀与崩溃**：all-in-one `.checkpoint.sqlite` 1.30 GB/1321 条；压缩只作用于本轮新消息、历史无限增长，全库无 `RemoveMessage`/`update_state`（`audit_20260717/05` P1-4；`agent/huginn/context_builder.py:466-467`、`agent/huginn/agent/streaming.py:302-419`）。
4. **400 悬空 tool_calls 整 run 报废**：PaperBench M1 第 16/150 次调用死亡，剩余 134 次预算全废；400 不重试、不修状态、不回滚（`_m1_full.log:268-270`；`agent/huginn/agent/streaming.py:1133-1143`；`analysis_20260717/09:42-54`）。**这是预算扩容的前置修复项——不先修，预算越大崩溃越多**（`analysis_20260717/09:148`）。
5. **CSM→compaction 空转税**：每 flag 付一次压缩 LLM 调用、收益近零（`agent/huginn/agent/reflection.py:251-253` → `streaming.py:911-924`；`analysis_20260717/07:73-77`）。
6. **rubric 即预算黑洞**：pinn rubric.json 116 万字符，需上百次分块读取（`analysis_20260717/03:100`）。

### 维度 4：模型配置（当前 2 → 顶级 5）

**顶级画像**：配置单一真源（一处定义、处处生效）；实跑模型名 + 配置哈希 + judge 版本 100% 落盘可审计；求解模型对齐官方工具型强模型档（Claude-Opus-4.6/GPT-5.4 级）；judge ≠ 被测模型（`analysis_20260717/10:148-149`、`01:200`）。

**差距清单**：
1. **静默降级**：`agent/huginn.toml` 写 deepseek-reasoner，全部 benchmark 实跑 deepseek-chat（非推理、temp 0.7）——`from_env` 只读环境变量不读 toml（`agent/huginn/config.py:600-643`；`analysis_20260717/10:56-61`）。
2. **不可审计**：`_meta.json` model 字段为无效值 `"output_version=None"`（`analysis_20260717/01:31`）——运行产物未可靠记录模型。
3. **judge 与被测同源**：全部 judge 默认 deepseek-chat + 复用 agent API key（`audit_20260717/16` P1-3；`rcb_score.py:33-37` 等），且 judge 无视觉致 RCB image 项盲评（`ResearchClawBench/_score_batch.py:36-45`）。
4. **无模型路由**：`ModelRouter` 实跑未启用，全程单一最弱模型，无「难题切强模型」路径（`analysis_20260717/10:138`）。
5. **评分链路不可复现**：`structai` 未声明未锁版，当前环境 import 失败——历史分数真实但任何重评都会直接崩溃（`analysis_20260717/10:63-67`）。

### 维度 5：数学链（当前 2 → 顶级 5）

**顶级画像**：「数值答案必须经工具计算」的确定性硬 gate（最短闭环约 50 行，`analysis_20260717/13:198-199`）；量纲代数（非单位名匹配）；守恒残差编译消费真实方程；Lean `sorry`/`axiom` 不可通过；数值-符号互验在主路径生效。

**差距清单**：
1. **配置层摘除**：四个外部适配器 tool_filter 把 symbolic_math/bourbaki/lean/unit/validate 全部过滤（`sab_huginn.py:53-62`、`rcb_huginn.py:52-63`、`mlebench_huginn.py:47-57`、`paperbench_huginn.py:71-79`）；30 份 RCB 轨迹 grep 数学工具 0 命中（`analysis_20260717/13:41-51`）。
2. **行为层裸答**：内部 repro 基准工具全量可用，χ=1.0 vs 0.004、θD=6.6K vs 480K 仍裸答出错（耗时 13.8–15.6s 为全 10 题最短档）——缺「数值必须算出来」的强制 gate（`analysis_20260717/13:54-64`）。
3. **语义层为空**：Lean 守恒检查是重言式（硬编码恒等演化，`agent/huginn/tools/bourbaki_tool.py:116-132`）；定理验证子串匹配、`by sorry` 可通过（`agent/huginn/lean/interface.py:158-191`；`audit_20260717/07` P2-3）；无 Lean 时 fallback 返回罐头文本（`bourbaki_tool.py:186-202`）——「验证通过」当前不携带信息量。
4. **量纲双轨**：深的 `execution/dimensional_validator.py` 只接 ontology 字符串引用，浅的只认 5 个基本量单位名（`bourbaki_tool.py:212-244`），无一套在主循环（`analysis_20260717/13:170`）。
5. **constraints 只挂仿真工具且为 warn 级**：benchmark 主战场（code_tool 产物）不在任何 scope 内；两条判据本身有物理错误（`agent/huginn/tools/adapter.py:425-437`；`audit_20260717/14` P2-3/P2-7）。
6. **子能力评分**（实现/接入，10 成制）：符号推导 6/2、量纲 5/1、守恒 3/2、形式化证明 2/1、数值-符号互验 3/1（`analysis_20260717/13:163-175`）。**对当前跑分的可归因上限约 1.5 个 benchmark 小项——数学链是第二阶修复，不是第一阶**（`analysis_20260717/13:194`）。

### 维度 6：进化回路（当前 1 → 顶级 5）

**顶级画像**（转引自 `analysis_20260717/11:14-19`）：EvoScientist EMA 三种进化写入——IDE（top 想法蒸馏方向）、IVE（失败方向变负知识含避坑建议）、ESE（从完整代码搜索轨迹蒸馏可复用执行策略，细节到「另一个工程师能据此重现」）；两块持久记忆 embedding 检索注入下一任务；适应度信号难以投机。**核心是「失败的制度化利用」——跨任务复利。**

**差距清单**：
1. **载体空转**：18/18 轨迹 perceive+report 两阶段、0 工具调用（1.5–11.1s）——`_perceive()` 以 git 文件变更为触发源，干净 workspace 下 hypothesize→learn 从未执行（`agent/huginn/autoloop/engine.py:1353-1362,2304-2305`；`analysis_20260717/06:30-43`）。
2. **写入无蒸馏**：25 条规则全是 `read_file not found` 模板匹配产物，无 LLM 参与、无任务语义、无分数输入；`_error_matches` 路径级子串匹配永不泛化（`agent/huginn/evolution/engine.py:502-547,571-574`；`analysis_20260717/06:58-71`）。
3. **写读路径永不相交**：引擎写 `~/.huginn/logs/`，context_builder 读 `$HUGINN_CACHE_DIR`（`agent/huginn/context_builder.py:393-410` vs `agent/huginn/evolution/logger.py:54-58`）——chat 主链路闭环从未闭合（`analysis_20260717/11:56-61`）。
4. **主路径零消费**：四个 benchmark 适配器对 evolution/reflect 零引用（grep 0 命中），每次运行都是无记忆一锤子买卖；apply 侧只接在空转的 autoloop 上（`analysis_20260717/06:66-69`、`11:63-65`）。
5. **产出有毒**：零执行仍虚构 Results（见维度 2 差距 2）；「Next iteration suggestion」全代码库无消费者（`analysis_20260717/06:55`）。
6. **经验写了、库存了、行为没变**：38/39 号经验文已入 KB 且检索通道开着，同一虚拟路径错误在最新运行中原样复发 25 次（`analysis_20260717/06:73-76`）。
7. **双自改进系统并行稀释**：EvolutionEngine vs SelfImprovementLoop 无分工（`audit_20260717/04` P1-6；`agent/huginn/self_improvement/core.py:2654`）。
8. **未爆弹**：PhaseGateState 模块级单例，并发 run 互相 reset——修复空转前必须 per-run 化，否则门控质量宣称不可信（`agent/huginn/phase_gate.py:692-702`；`analysis_20260717/06:91-98`）。

### 维度 7：评测信号（当前 1 → 顶级 5）

**顶级画像**：无泄漏通道（rubric/标签/答案在 agent 可读域外）；judge 异源且有视觉；评分门禁（`status=completed` 才允许评分、score 条目带 workspace+run_id+scorer 签名）；分数可复现（依赖锁版）、可溯源（model/config hash/judge version 落盘）、含 ceiling 归一；与官方协议可比（`analysis_20260717/05:148`、`01:201`、`02:167-168`）。

**差距清单**：
1. **泄漏三连**（`audit_20260717/16` P0-1/P0-2/P0-3）：PaperBench rubric.json + 4 条叶节点答案进 prompt（`paperbench_huginn.py:203-208,296-304`）；MLE synthetic 私有标签在 workspace `_private/test.csv`（`mlebench_huginn.py:318-319`）；HLE 含答案 parquet 在 agent 可读 cwd 且跨题共享 memory（`hle_huginn.py:74,205-217`）。
2. **测量伪影**：SAB judge 只见前 8000/10880 字符（`sab_huginn.py:311`，30–50 分测量损失）；MLE medal 死代码（leaderboard.csv 为 LFS 指针，任何分数输出 none，`mlebench_huginn.py:543-556`）；RCB 归属错配（`_score_batch.py:60-72` 按 mtime 选工作区）+ 半成品被打分（5.25 分对 `status=running`）；同报告两次评分差 ±3。
3. **抬分机制**：judge「不确定给 30、见 loss 曲线给 50+」、regex 只向上 override、harness 代跑训练充成绩、`_rescore_m7_c2st.py` 事后覆写被评分产物（`audit_20260717/16` P1-1/P1-4/P1-6）——历史 benchmark 分数不可作为能力证据引用（`audit_20260717/00:84-89`）。
4. **同源 judge + 盲评**（见维度 4 差距 3）。
5. **不可复现**：seed 用 Python 字符串 hash（每进程随机）、共享 memory/checkpoint、structai 未锁版（`audit_20260717/16` P1-12；`analysis_20260717/10:63-67`）。
6. **制度化的错误经验**：自进化 loop 已发现 LFS 指针问题，处方却是 try/except 兜底成 `medal="none"`——评测失灵被自进化制度化了（`analysis_20260717/02:45`）。**这是「先修评测信号再接进化」原则的最硬证据。**

### 维度 8：工具可靠性（当前 2 → 顶级 5）

**顶级画像**：静默失败（success=False + 空输出）触发回退而非返回 "Unknown error"；失败语义统一为 `ToolResult(success=False)`；工具不可用时有显式信号（`search_unavailable: true`）而非让 agent 耗死重试；同步重型工具超时可中断；无「评测特供补丁」——benchmark 测的就是生产系统（`analysis_20260717/08:250-256`）。

**差距清单**：
1. **第一大直接失分项**：8 个出分单元 62.5% 有工具层直接背书；RCB 三个出分任务 100% 有工具层硬失败（`analysis_20260717/08:195-213`）。
2. **Rust sandbox 静默崩溃**：RDKit+sklearn GP 一致返回 "Unknown error"，回退只捕获异常、不捕获静默失败（`agent/huginn/tools/bash_tool.py:120-162`、`agent/huginn/tools/adapter.py:464-466`）——修复提交（fde2e42）晚于失败运行 2 小时。
3. **熔断器自锁/误伤**：M_001 双工具 circuit_open 致方法从 GNN 降级 GP（`agent/huginn/agents/circuit_breaker.py:197-221`）；file_read_tool 误触发熔断（`analysis_20260717/08:134-145`）。
4. **WinError 2 三层穿透**：sandbox→adapter→HTTP 整轮作废，长任务 30 轮 10 败中 40%（`analysis_20260717/08:147-163`）。
5. **搜索链全灭无信号**：ddgs 改名 + bing 超时 + MP 403，agent 无「搜索不可用」信号，耗死重试后裸答猜值（`agent/huginn/tools/web_search_tool.py:242-278`；`analysis_20260717/08:165-176`）——physics bench 3 道错题（5 个数量级、符号错误）全部伴随此链。
6. **恢复机制「设计上有、实际半残」**：三种最致命的失败（budget 墙、静默崩溃、WinError 2）恰好都没有有效恢复路径（`analysis_20260717/08:224-235`）。
7. **评测/生产分裂**：`rcb_runner.py:30-46` 用 7 个 `os.environ.setdefault` 补丁绕过已知 bug——开发者知道根因但选择评测时打补丁；被测 agent 与生产 agent 运行姿态不同（`analysis_20260717/08:239-245`）。
8. **正面事实**：压测 v23 基础设施自检 39/39、v24 三十分钟长航 tool_success_rate=1.0（`analysis_20260717/01:99-100`）——基础件健康，坏的是失败语义与恢复路径。

### 维度 9：工作流协议（当前 2 → 顶级 5）

**顶级画像**：一条工作流真源（竞品：ResearchClaw 一条持久化 workflow 状态链、AutoResearchClaw 一条 23 阶段单管线）；阶段预算真生效；反思纠偏类型对齐、闭环真实发生；死代码要么接线要么拆除，不收空转税（`analysis_20260717/12:165`、`07:138-146`）。

**差距清单**：
1. **模式切换 0 次发生，但死代码仍在收税**：四个适配器无 `set_mode` 调用，exec_mode 恒 tool_call、user_mode 恒 chat；PaperBench 1321 条 checkpoint `[PHASE:` 0 命中——「切换丢上下文」的直觉不成立（`analysis_20260717/07:43-49`）；但 CSM/compaction 空转税、phase 预算死通道、prompt 常驻相位教学段（`agent/huginn/agent/context.py:53-59`）仍在扣钱。
2. **5 套模式机制无一在得分路径产生正收益**：M1 未用、M2 从不切换、M3 预算通道断路且工具过滤是雷（code_tool 在非 OPEN 相位全被过滤，`agent/huginn/tools/base.py:196-205` + `phases.py:143-151`）、M4 只产税、M5 整体塌缩（`analysis_20260717/07:125-127`）。
3. **反思纠偏死锁**：`task_reflector.py:130-131` 建议 `discover/construct`，`set_mode` 只接受 `chat/research/plan`（`agent/huginn/agent/core.py:419-420`）——永远 ValueError 被静默吞掉（`agent/huginn/agent/reflection.py:319`），自适应纠偏闭环自始死锁。
4. **有害的工作流产物**：旧 RCB 适配器 PHASED PROTOCOL 强制「calls 31-40 必须交报告」+ MODEL COMPLEXITY CEILING 禁止深度学习直到报告存在——与 RCB 核心方法（GNN/VAE）正面冲突，把 agent 训练成「先交卷后补做」（`rcb_huginn.py:140-156`；`analysis_20260717/01:72`）。
5. **管线丛生、跑分时全不在位**：phases.py 给 EXECUTION 留 300 calls 的设计从未生效，实际生效的是 runner 硬编码 150/50（`analysis_20260717/12:165`）。
6. **正面事实**：RCB 3-step runner 的 Step 3 对抗自审**按设计工作**，准确揪出数据泄漏与 GVAE 沉默替换——诊断机制有效，缺的是修复通道（`analysis_20260717/05:77-83`）；Step 2.5 兜底报告是「报告生成」环节唯一及格件（`analysis_20260717/12:113-115`）。

---

### 补遗：loop-level step retry 可观测性差距（2026-08-04 核实）

**触发**：对标 Kimi Code `MoonshotAI/kimi-code` 的 `stepRetryService.ts`（`packages/agent-core-v2/src/agent/stepRetry/`），核实 huginn 错误恢复机制。

**核实结论（推翻先前误判）**：
- 先前对话曾判「huginn 缺 loop-level step retry」——**错误**。huginn 在 `agent/huginn/agent/streaming.py:1361-1580` 已有完整的 `while attempt < max_retries` step 重试循环，含错误分类（rate_limit / overloaded / transient / context_overflow）、退避抖动、529 风暴切 fallback model。
- 先前对话曾判「错误恢复能力等价」——**不准确**。两者层级不同：Kimi 是 driver 重入队（`context.retry(driver, {at: 'head'})`），huginn 是 graph.stream() 重调；Kimi 把 `failedAttempts` 注册到 `IAgentStateService` 跨 step 持久，huginn 是局部变量。

**真实差距（仅一条，可观测性）**：
- huginn 重试时只 `logger.warning`，前端 / audit log / 监控看不见；Kimi 通过 `IEventBus.publish('turn.step.retrying')` 发结构化事件，带 `failedAttempt / nextAttempt / maxAttempts / delayMs / errorName / statusCode` 字段。

**已落补丁（2026-08-04）**：
- `events/event_types.py`：新增 `STEP_RETRY = "agent.step.retrying"` 并注册到 `ALL_TYPES`。
- `agent/streaming.py:1554-1578`：在重试 except 块 `logger.warning` 后，`await EventBus.shared().publish(AgentEvent(...))` 发事件，字段对齐 Kimi（attempt / max_attempts / error_type / error_message / wait_ms / states_yielded）。best-effort，publish 失败不阻塞主流程。
- `events/_selfcheck.py`：新增 `test_step_retry_event` 验证类型注册 + 发布订阅。

**未做（ponytail 取舍）**：
- 不照搬 Kimi 的 `IAgentStateService` 跨 step 持久化——huginn 的 `ServerContext` + fallback model 切换（`streaming.py:1520-1536`）已覆盖「连续失败切模型」场景，跨 step 计数持久化收益小、改动大。
- 不照搬 Kimi 的 `registerLoopErrorHandler` 注册式插件——huginn 的 `while attempt` 循环已在线性代码里，改成插件式是架构稀释，违反第四节「收敛优于新增」。

---

### 补遗 2：基础设施对齐 Kimi Code —— P0/P1/P2 落地（2026-08-04）

**触发**：step retry 补遗后，继续对标 Kimi Code `MoonshotAI/kimi-code` 的 `_base/` + `agent/` + `transcript/` 基础设施分层，逐项核实 huginn 真实差距。

**核实结论（推翻"基础设施全面落后"的预设）**：
huginn 已有遥测（`telemetry.py` TelemetrySpan + span 树 + 内存快照）、token 计数（`utils/tokens.py` tiktoken）、引擎状态快照（`runtime/engine_state.py`）、持久化后端（`persistence/checkpointer.py` SQLite）、事件溯源（`provenance/registry.py` replay_to/rollback_to）、审计（`events/audit_log.py`）——这些与 Kimi 等价或更丰富，不需补。

**真实差距（三条，已全部补齐）**：

| # | 差距 | Kimi 实现 | huginn 原状 | 已落补丁 |
|---|---|---|---|---|
| P0 | 字段级状态注册 | `defineState` + `IAgentStateService` 跨 step 持久化 | `engine_state.py` 整体快照,非字段级;崩溃后局部变量丢 | `persistence/state_registry.py` StateRegistry (LRU+SQLite) |
| P1 | 对话转录 | 独立 `transcript` 包 | `telemetry.py` 是性能轨迹不是对话内容;`memory/` 是压缩语义片段不是原始转录 | `events/transcript.py` TranscriptStore (订阅 EventBus 落 JSONL) |
| P2 | 工具调用去重 | 独立 `toolDedupe` 模块 | `_read_only_cache` 只对 read_only=True 工具生效;有副作用的 bash/code/vasp 不去重 | `agents/tool_dedupe.py` ToolDeduper (LRU + args_hash) |

**集成点**：
- `server_context.py`：ServerContext 加 `state_registry` / `transcript_store` / `tool_deduper` 三个字段,`create_server_context` 启动 TranscriptStore 订阅。
- `tools/adapter.py`：ToolAdapter 加 `set_deduper()` + `_current_deduper` 字段;pre-checks 在 cache 后插入 dedupe lookup;post-checks 在 router record 后插入 deduper record。
- `persistence/__init__.py` + `events/__init__.py`：导出新组件。

**不照搬的（ponytail 取舍）**：
- DI cascadeEngine（30KB + 9KB）：Python 不需要 TS 那套装饰器元数据,`ServerContext` 单例 + 构造器注入够用。
- tree-sitter-bash：huginn 不是代码编辑器。
- pi-tui / minidb / OAuth / ACP 协议：科研场景不需要。
- toolApproval 独立模块：`policy_engine.py` 已有 "ask" action + 现有交互够用。

**自检**：三个新模块各自带 `__main__` self-check,全部通过。编译 + import 链验证通过。

---

## 三、三阶段路线图

> 阶段依赖（硬性 gate）：**P0 → P1**：测量可信（SAB 重评落地、medal 有效、评分门禁生效）+ 预算假耗尽消除。**P1 → P2**：可执行性硬门（占位交付 0 次、sanity gate 生效）+ 诚实评测信号（泄漏堵死、judge 异源）——转引 `analysis_20260717/00:127-128`，缺一不接进化。

### 阶段 P0：修复周（第 1 周）——小时级共识项 + 可测判据

来源：`analysis_20260717/00:103-110` 第一梯队 + 各分报告「半天/1 行」级建议 + `audit_20260717/00:163-169` 安全立即项。

| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| P0-1 | 删送审截断；judge 评语保留 2000+ 字符；`_score.json` 记录 `judge_input_chars` | `sab_huginn.py:311,351` | 04 报告：原地重评即可回收 30–50 分，无需重跑 agent |
| P0-2 | MLE medal 死代码：拉取 leaderboard LFS 或删除 medal 字段；修分位数方向颠倒；`_score.json` 增 `oracle_ceiling` 与 `metric_provenance: "synthetic-smoke"` | `mlebench_huginn.py:543-556`；`mle-bench/mlebench/competitions/*/leaderboard.csv` | 02 报告 F1/F2、建议 3/4 |
| P0-3 | 预算错误消息加重置语义（「本轮已尽，下轮重置为 N，请换工具或收尾」）；MLE 60→150 calls；SAB 40→对齐超时额；删 `SAB_DELIVERABLES` 中任务从未要求的 `pred_*.txt` | `agent/huginn/tools/adapter.py:721-723`；`mlebench_huginn.py:576`；`agent/huginn/bench/orchestrator.py:78-81` | 00 报告 §三.1；12 报告 P0-1；09 报告建议 3 |
| P0-4 | 完成判据从 3-glob 换 rubric 覆盖清单 + 产物存在性；修 `RCB_DELIVERABLES` 路径错位 | `agent/huginn/bench/orchestrator.py:61-71,147-159` | 03 报告建议 1；05 报告 R2 |
| P0-5 | 报告-产物一致性核对器最小版：拦截 "Expected" 占位表与无产物支撑的数值；autoloop `_report` 加无数据硬门（exec_summary 为空禁写 Results/Discussion，强制输出「本循环未执行任何计算」） | `agent/huginn/autoloop/engine.py:5016-5056`；核对器挂在 `rcb_runner.py:399-434` Step 2.5 兜底处 | 06 报告建议 1；12 报告 P1-1 |
| P0-6 | 模型配置显性化：`from_env` 与 toml 打通（或启动时断言打印实跑模型并写入 `_huginn_meta.json`：model 名 + config hash + judge 版本） | `agent/huginn/config.py:600-643`；各适配器 meta 落盘处 | 10 报告建议 5；01 报告建议 8 |
| P0-7 | Rust sandbox 默认退出 benchmark 路径（`HUGINN_NO_RUST_SANDBOX=1` 默认化）；静默失败（success=False 且输出全空）视为崩溃信号自动回退；熔断器对 file_read_tool 等只读工具永久豁免 | `agent/huginn/tools/bash_tool.py:120-162`；`agent/huginn/cli/rcb_runner.py:41-46` | 08 报告建议 2/5 |
| P0-8 | 评分门禁：`_meta.json status=="completed"` 才允许评分；score 条目强制带 `workspace`+`run_id`+scorer 签名；重复评分写新文件不覆盖 | `ResearchClawBench/_score_batch.py:60-72`；`run_task.py:146-154` | 05 报告 R1 |
| P0-9 | 【安全】轮换 DeepSeek API 密钥；清除 `start_sidecar.bat:4` 硬编码与 `huginn.toml` 明文及 5 份 `.bak` 副本，启用已有加密能力 | `audit_20260717/00:71-75` | audit P0-5（与跑分无关但为一切重跑的前置） |
| P0-10 | 提交格式校验器最小版：AUC 类指标而提交列唯一值 ≤2 → 报警并自动改用 `predict_proba` 重交；system prompt 逐竞赛写死 metric+提交类型 | `mlebench_huginn.py:367-369`；`agent/huginn/tools/bench_infra/kaggle_tool.py:1-6` | 02 报告建议 1/2（一行捡回 ≈0.04 AUC） |

**P0 完成判据（全部可测量）**：
1. SAB task_1 原地重评（judge 见全文）分数 **≥55/100**（当前 25，测量回收）。
2. MLE medal 字段不再是恒 none，且方向正确（高分得高牌）；synthetic 跑分报告 100% 带 `score/ceiling` 归一字段。
3. RCB 四任务重跑 **Material 均分 ≥18**（当前 13.7）；报告含 "Expected" 占位表 **0 次**；agent 自述「budget exhausted → 写报告交差」**0 次**（grep `_agent_output.jsonl`）。
4. autoloop 零执行循环产出虚构 Results **0 次**（硬门生效，输出「未执行任何计算」声明）。
5. `_huginn_meta.json` 模型名/config hash **100% 落盘**；实跑模型与配置显性一致（或显式声明差异原因）。
6. "Unknown error" 与只读工具 circuit_open 在重跑中 **0 次**。
7. 半成品（status=running）被评分 **0 次**；分数归属错配 **0 次**。
8. AUC 任务提交硬标签 **0 次**（校验器拦截日志为证）。
9. 明文密钥副本 grep **0 命中**。

### 阶段 P1：能力月（第 2–5 周）——执行闭环 + 验证文化 + 预算重构

三条主线按周推进；评测卫生贯穿全月（它是 P2 的前置 gate）。

**主线 A：执行闭环（第 2–3 周）**
| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| A1 | harness 预消化 rubric 为「任务 × 超参 × 种子」实验矩阵 + 预算表（几 KB），注入硬策略：全网格缩尺执行优先于单格全量执行 | `paperbench_huginn.py` setup 段；复用现成 `collect_rubric_leaves` | 03 报告建议 2（消解 1.16MB rubric 读取黑洞） |
| A2 | 产物级门控：report.md 落盘前检查 `outputs/` 存在真实 metrics 文件，否则禁止进入 report 阶段并生成 blocker 任务（ResearchClaw remediation task 最小实现） | `agent/huginn/cli/rcb_runner.py:399-434` | 12 报告 P1-1 |

> **A2 落地（2026-08-04）**：`rcb_runner.py` 新增 `_scan_real_metrics` / `_step2_outputs_gate` 两函数，在 A3 audit 之后、Step 2.5 之前调用；blocker 信号落盘 `ws/.huginn/step2_outputs_gate.json`，Step 3 critique 读到 blocker=True 时注入「Results claim 缺乏产物支撑，应判 fabricated / 0 分」到 checklist 让 LLM critic 降权。占位文件过滤：扩展名白名单 + 大小 > 0 + 短文本 (<200 chars) 含 placeholder token (expected/todo/placeholder/tbd/n/a/not implemented) 跳过；.npy 二进制按大小判定。`--self-check-a2` 入口验证 4 个断言全通过。
| A3 | silent substitution 结构性拦截：Step-1 checklist 落盘为文件，Step-2 结束机械比对「[EXACT] 组件 ↔ code/ 实现痕迹」，缺失即回退执行；禁止未尝试标 [VARIANT]（≥2 次尝试失败才允许降级且须附报错）；退役 MODEL COMPLEXITY CEILING | `agent/huginn/cli/rcb_runner.py:504-510`；`rcb_huginn.py:150-154` | 12 报告 P1-3；05 报告 R5 |

> **A3 落地（2026-08-04）**：`rcb_runner.py` 新增 `_extract_exact_components` / `_scan_implementation_traces` / `_parse_substitute_headers` / `_count_failed_attempts` / `_step2_substitution_audit` 五个函数，在 Step 2 结束后、Step 2.5 之前调用；audit 结果落盘 `ws/.huginn/step2_audit.json` 供 Step 3 / 评分器引用；PHASED PROTOCOL 里 "Phase 2: NO deep learning yet" 改为「按 paper 方法论选模型类」退役 MODEL COMPLEXITY CEILING。`--self-check-a3` 入口验证四个机械比对 helper，全 assert 通过。落点修正：`rcb_huginn.py:150-154` 已迁移到 `rcb_runner.py` system_prompt 段（路线图原引路径过时）。
| A4 | Step-3 独立修复预算池：critique verdict≠pass 追加专用 50 次预算 | `agent/huginn/cli/rcb_runner.py:305-306` | 05 报告 R4（兑现对抗自审的最后一公里） |

> **A4 落地（2026-08-04）**：`rcb_runner.py` 新增 `_build_retry_budget(extra_budget)`，根据 `extra_budget` 构造 `BudgetSpec(max_calls=N, recursion_limit=max(250, N*5))`；None/0/负数返回 None（不覆盖）。`_stream_chat` 接收 `extra_budget` 参数并传给 `agent.chat(budget_override=...)`；Step-3 retry 在 `verdict≠pass` 时注入 `extra_budget=50`（250 工具调用预算、500 recursion_limit），独立于 Step-1/2 预算，确保 critique 指出的 gap 有资源修复而不被原预算耗尽截断。`--self-check-a4` 4 个断言全通过（None/0/-5→None；50→BudgetSpec(50,250)；100→BudgetSpec(100,500)；公式 `max(250, n*5)`）。
| A5 | 执行兜底泛化：`_has_code_no_output` 改为按 DeliverableSpec「代码类 glob 存在 + 输出类 glob 缺失」判定 | `agent/huginn/bench/orchestrator.py:228-233` | 07 报告建议 4 |

> **A5 落地（2026-08-04）**：`orchestrator.py:252-272` `_has_code_no_output` 不再硬编码 `submission/reproduce.sh + *.py`，改为按 `deliverable_spec.checks` 自身 pattern 分类——code_pats 取 `.py/.sh` 结尾、output_pats 取含 `outputs/.json/.png/.csv/report.md` 的 pattern；任一代码类存在 + 任一输出类缺失 → 触发 `_execution_prompt()`。SAB/RCB 形态的 deliverable 现在也能进入执行兜底档。`_self_check()` 场景 2-3 间接验证（代码齐+输出缺→`_execution_prompt` 分支生效），orchestrator self-check OK。
| A6 | 平凡基线闸门：提交前强制对照多数类/单特征/线性基线，低于基线必须回退到基线方案 | MLE/RCB 提交流程 | 01 报告建议 2 |

> **A6 落地（2026-08-04）**：MLE-bench 抽出 `_compute_trivial_baseline(y, is_auc)` / `_is_below_baseline(score, baseline, tol=0.01)` 两函数（多数类=max(p,1-p)、AUC=0.5、非二值返回 NaN 跳过）；`grade_submission` 用新函数替换原内联逻辑，`below_baseline` / `baseline_note` 落 `_score.json` 可追溯。system prompt 加 `BASELINE GATE` 硬规则——CV < 多数类/AUC=0.5 必须回退到基线预测。RCB system prompt 加 `BASELINE COMPARISON` 规则——report.md 末尾必须含 `## Baseline Comparison` 表（Metric/Your Value/Paper Baseline/Δ/Status），低于 paper baseline >10% 加根因注解，3 次失败后标 'below baseline — fallback to paper reproduction'。`--self-check-a6` 入口验证 7 个断言（多数类/AUC/非二值/below 检测 5 边界），全 assert 通过。
| A7 | RCB 启动前 30s 环境冒烟（bash/code 各跑一次 RDKit+sklearn 微型 GP + torch.load 真实文件），失败即 fail-fast 打印修复清单；7 个评测 setdefault 改强制赋值 | `agent/huginn/cli/rcb_runner.py:28-46` | 05 报告 R3 |

> **A7 落地（2026-08-04）**：`rcb_runner.py` 已存在 `_rcb_smoke_test()`——RDKit (Chem.MolFromSmiles + Descriptors.MolWt) + sklearn GPR (fit+predict) + torch (randn+matmul) 三件套冒烟，失败打印修复清单 + `sys.exit(1)`；`main()` 入口在 `HUGINN_SKIP_SMOKE≠1` 时调用，30s 内完成。7 个评测关键 env var 从 `setdefault` 改为强制赋值（`os.environ[k]=v`），防外部 env 让 patch 失效：`HUGINN_RATE_LIMIT_ENABLED=0` / `HUGINN_ALLOW_LOCAL_BASH=1` / `HUGINN_RCB_CSM_SUBSET=1` / `HUGINN_NO_RUST_SANDBOX=1` / `HUGINN_COGNITIVE_LLM_DECIDER=0` / `HUGINN_HEALTH_MONITOR=0` / `HUGINN_FEATURE_LOOP_DETECTOR=false`。`--self-check-a7` 入口验证函数定义 + 7 个 env 强制赋值生效，全 assert 通过。

**主线 B：验证文化（第 3–4 周）**
| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| B1 | 「数值必须经工具计算」确定性 gate（约 50 行）：期望数值答案而全程 code_tool/symbolic_math 调用数为 0 → 打回一次注入 "You must compute, not guess"，仍裸答判 FAIL | `agent/huginn/bench/runner.py:484-486,537-554`；四适配器同策略 | 13 报告 Q4/建议 1 |
| B2 | 外部四适配器 tool_filter 恢复 `symbolic_math_tool` + `unit_tool` + `validate_tool`；prompt 把 "use code_tool to compute" 从建议改规则 | `sab_huginn.py:53-62`、`rcb_huginn.py:52-63`、`mlebench_huginn.py:47-57`、`paperbench_huginn.py:71-79`、`rcb_runner.py:310-315` | 13 报告建议 2；21 报告动作 1 |
| B3 | critique 数值重算：Step 3 对每个数值 claim 调 validate_tool/numerical_tool 重跑，替代纯文本对照 | `agent/huginn/cli/rcb_runner.py:455-462` | 12 报告 P0-3；21 报告动作 3 |
| B4 | 报告强制区分 `EXECUTED` vs `EXPECTED/NOT EXECUTED` 标记（report 模板约束 + lint） | `agent/huginn/cli/rcb_runner.py` report 模板 | 05 报告 R7 |
| B5 | 自进化复盘 checklist 增「指标-提交对齐」检查项：任何 CV vs test gap 诊断前先确认两边同一度量 | `agent/huginn/knowledge/seed/38_benchmark_evaluation_lessons.md` 复盘流程 | 02 报告建议 6 |
| B6 | E2E 断言加牙齿：`test_autoloop_completes_all_phases` 断言 7 phase 全 completed 且 tool_calls>0；失败目标断言非 completed | `agent/tests/stress/test_sci_automation.py:269-292` | 06 报告建议 2 |

### P1-B2/B3/B4/B5 落地说明（2026-08-04）

**B2（四适配器 tool_filter 恢复数学工具 + COMPUTE RULE 硬规则）**
- 四适配器 `*_TOOL_FILTER` 全部含 `symbolic_math_tool` + `unit_tool` + `validate_tool`：
  - `sab_huginn.py:64-77`、`rcb_huginn.py:160-177`、`mlebench_huginn.py:57-71`、`paperbench_huginn.py:81-93`
  - `rcb_runner.py:5041-5044` 补 `unit_tool`（之前缺，导致 Material/Physics 量纲检查不可用）
- 三个外部适配器 system prompt 的 `## Available Tools` 段补三个数学工具描述 + `COMPUTE RULE (P1-B2, hard)` 硬规则：
  - SAB: `sab_huginn.py:180-189` — "any numeric answer in domain_knowledge.md or task output MUST be derived via code_tool / symbolic_math_tool / validate_tool, not stated from memory"
  - MLE: `mlebench_huginn.py:498-506` — "every CV score you report MUST be computed via code_tool, not estimated"
  - PB: `paperbench_huginn.py:304-313` — "every numeric constant you put into code MUST be derived via code_tool / symbolic_math_tool / validate_tool, not transcribed from the paper by eye"
- 自检: `python mlebench_huginn.py --self-check-b2` 验证 tool_filter + prompt 注入 + COMPUTE RULE 顺序（前置于 PHASED PROTOCOL）

**B3（critique 数值重算：Step 3 替代纯文本对照）**
- `agent/huginn/cli/rcb_runner.py:4242-4253`（step3_prompt A. Sanity Check 段）替换：
  - 旧: "Compare each to the paper's baseline value from your Step 1 checklist"（纯文本对照）
  - 新: "RECOMPUTE each claim via an independent computation path — do NOT trust your own previous text. Call validate_tool (numerical cross-check) or symbolic_math_tool (closed-form re-derivation) or code_tool (re-run the metric on outputs/ artifacts) for each claimed number."
  - 加 `P1-B3 hard rule`: "a claim that you cannot recompute independently is FABRICATED, treat it as 0"
  - 表格从 `| Metric | Paper Value | Your Value | Better? |` 改为 `| Metric | Claimed | Recomputed (independent path) | Tool used | Match? |`，强制 agent 列出重算工具 + 独立路径
- 与既有 G28 `_recompute_report_metrics`（report.md vs outputs/ 数值比对，>10% deviation flag）互补：G28 是机械正则比对，B3 是 agent 主动调工具独立重算，两层兜底防虚构
- 自检: `python agent/huginn/cli/rcb_runner.py --self-check-b3` 验证 RECOMPUTE 规则注入 + 三工具列出 + 老模式移除 + 4 列表格

**B4（报告强制区分 EXECUTED vs EXPECTED 标记 + lint）**（之前已落地，本次确认自检仍通过）
- `agent/huginn/cli/rcb_runner.py:3600-3665` `_lint_report_markers` 函数：扫描 report.md，统计数值声明的标记情况（`[EXECUTED]` / `[EXPECTED]` / `[NOT EXECUTED]`）
- 自检: `python agent/huginn/cli/rcb_runner.py --self-check-b4` 通过（total=6 tagged=4 untagged=2，三 marker 计数全对）
- ponytail: 句子级扫描，ceiling 是不区分 marker 是否真实（agent 可能瞎标），升级路径是跟 outputs/ 文件交叉验证 `[EXECUTED]` 数值是否真有产物支撑

**B5（自进化复盘 checklist 增「指标-提交对齐」检查项）**
- `mlebench_huginn.py:484-496` system prompt 在 `OVERFITTING GUARD` **之前**插入 `METRIC ALIGNMENT PRECHECK (P1-B5)` 4-step checklist：
  1. What metric does the competition grade on? (read description.md / grade.py)
  2. What did CV compute? (accuracy? AUC? RMSE?)
  3. Are they the same? If not, the gap is metric mismatch, not overfitting
  4. For AUC tasks: submission.csv 含概率 (n_unique > 2) 还是硬标签 (n_unique ≤ 2)? 硬标签 silently 损失 ~0.04 AUC
- 加自进化回路警示: "a metric-mismatch gap misdiagnosed as overfitting will 固化 the wrong lesson (over-regularize) — run this checklist before writing any post-mortem entry"
- 知识库 `agent/huginn/knowledge/seed/38_benchmark_evaluation_lessons.md` §2.9b 已有完整 checklist（agent RAG 可命中）
- 自检: `python mlebench_huginn.py --self-check-b5` 验证 4-step checklist 完整 + 顺序（PRECHECK 前置于 OVERFITTING GUARD）+ 自进化回路警示存在

**主线 C：预算重构（第 4–5 周，严格按此前置顺序）**
| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| C1 | 【前置】checkpoint 悬空 tool_calls 修复：发送前校验消息序列配对，悬空补占位 ToolMessage（或裁剪成对），失败回滚上一一致 checkpoint；400 invalid_request 纳入状态修复重放路径 | `agent/huginn/agent/streaming.py:1133-1143`；压缩 middleware 之后加配对校验 | 09 报告建议 2；08 报告建议 4 |
| C2 | 【前置】compaction 二选一落地：真修剪 checkpoint（`update_state`+`RemoveMessage`）或显式删除空转管线改用 deepagents SummarizationMiddleware；benchmark 运行限定 checkpoint 尺寸上限 | `agent/huginn/agent/streaming.py:302-419`；`agent/huginn/context_builder.py:466-467` | 09 报告建议 4；audit 05 P1-4 |
| C3 | 预算扩容：PaperBench 150→600 calls / 3600→21600s；RCB 150→400（或 Step2 独占）；接通 phase 预算通道（去掉 research-only 限制或显式按 `max_total_calls` 构造 BudgetSpec） | `paperbench_huginn.py:916-917`；`agent/huginn/cli/rcb_runner.py:305`；`agent/huginn/bench/orchestrator.py:171-173` | audit 20 动作 1（预算-产出曲线最陡段：再投 150 次 ≈ +6.5 分）；07 报告建议 1 |
| C4 | web 检索链修复（ddgs 迁移 + 多源兜底）+ 全灭时返回显式 `search_unavailable: true`；bench 前检索健康检查 | `agent/huginn/tools/web_search_tool.py:242-278` | 08 报告建议 3；01 报告建议 3 |
| C5 | 7 个评测补丁逐项下沉：能修根因的修根因，不能修的做成 `huginn.toml` 正式配置并写明代价 | `agent/huginn/cli/rcb_runner.py:30-46` | 08 报告建议 6 |
| C6 | 【评测卫生，贯穿全月】堵泄漏三通道（rubric/私有标签/答案 parquet 移出 agent 可读域）；删 prompt 中 4 条叶节点答案改为「写完每个 .py 必须执行针对该文件 rubric 叶的断言脚本」；judge 换异源强模型 + RCB 启视觉 judge；structai 声明锁版；禁用一切事后覆写脚本；**随后全量重跑 5 大 benchmark 建立可信基线** | `paperbench_huginn.py:203-208,296-304`；`mlebench_huginn.py:318-319`；`hle_huginn.py:74,205-217`；`rcb_score.py:33-37` | audit 16 P0-1/P0-2/P0-3/P1-3；03 报告建议 4；01 报告建议 6 |
| C7 | 长产物分块写入 + `file_write_tool` Windows 路径修复，四适配器恢复该工具 | `agent/huginn/tools/file_write_tool.py:62-70`；`sab_huginn.py:51` | 11 报告建议 5；audit 20 动作 3 |
| C8 | 可观测性：全部 benchmark `_huginn_meta.json` 落盘 `tool_calls_used/turns/context_overflow_count/compaction_count/crash_traceback` | 各适配器 meta 落盘处 | 09 报告建议 7 |

**P1 完成判据（全部可测量）**：
1. **RCB Material 均分 ≥20**（13.7→20，对齐 ResearchClaw 19.3 上方、DeepSeek-V4-Pro 24.6 下沿）；Material_000 类权重 0.5 项从 0 → **≥20 分**。
2. **PaperBench 原生协议**（无泄漏、无代跑、无抬分引导）all-in-one **≥15%**（当前注水 13.52）、pinn **≥8%**（当前 2.23）；执行叶均分从 0.8 → **≥30**。
3. 占位数值交付 **0 次**；虚构数据 **0 次**；硬标签提交 **0 次**；低于平凡基线提交 **0 次**（四道闸门各自拦截日志为证）。
4. repro 基准 **10/10**（当前 8/10），且每题有计算工具调用记录；数量级错误 **0**。
5. checkpoint 悬空 tool_calls 崩溃 **0 次**；单 checkpoint ≤100MB；PB M1 类整 run 报废 **0 次**。
6. 可信基线建立：5 大 benchmark 重跑分数可复现（judge 异源 + structai 锁版 + 无泄漏）；RCB 同任务重跑方差从 ±3 收敛到 **≤±1.5**。
7. PaperBench 运行预算利用率 ≥80%，不再出现 19–32 分钟提前收工；rubric 叶级覆盖统计（已实现 a/b、已执行 c/d）每轮注入日志可见。
8. 搜索全灭时裸答猜值 **0 次**（`search_unavailable` 信号生效，agent 转本地知识或声明不可考）。

### 阶段 P2：进化季（第 2–4 月）——经验池 + 蒸馏 + 数学链深水区

**主线 D：经验池三段修复（第 2 月）**——载体、写入、读取，缺一不可（`analysis_20260717/06:120-128`）
| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| D1 | autoloop objective-driven 化：`run(objective=…)` 首轮跳过 perceive-gating，objective+KB 直接构造 context 进 hypothesize；「no changes」语义限定于 watch 模式 | `agent/huginn/autoloop/engine.py:1341-1364` | 06 报告建议 3；11 报告建议 2 |
| D2 | benchmark 失败→分析→更新→重试回路：每个失败 run 结束后，judge 评语 + 轨迹交 LLM 蒸馏 1–3 条策略写入规则库，下次运行注入（对标 EvoScientist EMA 写入，**前置 gate 是 C6 评测卫生**） | 适配器 run 收尾钩子 + `agent/huginn/evolution/engine.py:177-209` | 11 报告建议 4；00 报告第三梯队 #13 |
| D3 | 写读路径统一：`context_builder.py:400` 与 `evolution/logger.py:56` 对齐到同一路径；`get_prompt_patches`/`apply_heuristic_fix` 接进 HuginnAgent 主路径（system prompt 注入 top-confidence 规则） | `agent/huginn/context_builder.py:393-410`；`agent/huginn/evolution/logger.py:54-58` | 06 报告建议 4；11 报告建议 1 |
| D4 | 匹配泛化：`_error_matches` 从路径子串升级为错误类别匹配（`FileNotFoundError` 泛化到任意路径） | `agent/huginn/evolution/engine.py:571-574` | 06 报告建议 4 |
| D5 | PhaseGateState per-run 化 + routes/autoloop.py per-workspace 互斥（**依赖门控前必做**） | `agent/huginn/phase_gate.py:692-702`；`agent/huginn/routes/autoloop.py:91-94` | 06 报告建议 6；audit 04 P1-4 |
| D6 | trajectory schema 增 hypothesis/plan/validation 摘要字段（可截断） | `agent/.huginn/trajectories/loop_*.json` 生产者 | 06 报告建议 8（让第二次诊断有据可依） |

**主线 E：蒸馏质量与模式收敛（第 3 月）**
| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| E1 | LLM 蒸馏器替换模板匹配：IVE 式负知识（失败方向 + 3–6 条避坑建议）+ ESE 式执行策略（强制保留参数/库函数名细节，「另一个工程师能据此重现」） | `agent/huginn/evolution/engine.py:502-547` | 11 报告发现 2/发现 4 |
| E2 | 经验库考核：`usage_count=0` 超 30 天自动淘汰并告警；现有 25 条零使用规则 + 20 条未验证蒸馏知识归档清零 | `~/.huginn/logs/evolution_rules.json` 生命周期管理 | 06 报告建议 5 |
| E3 | stable_principle 跨任务共享（RCB 任务间继承上一任务修正） | `agent/huginn/agent/reflection.py:416-477` | audit 21 G5 |
| E4 | 双自改进系统合并或分工：EvolutionEngine 管策略知识、SelfImprovementLoop 管评测执行，单一真源 | `agent/huginn/self_improvement/core.py:2654` | audit 04 P1-6；11 报告建议 8 |
| E5 | 模式层接线或拆除：phase 预算通道接通（C3 完成）或显式移除死代码；反思切换类型对齐（`suggested_mode` 改合法值或删契约）；`[PHASE:]` 自动切换在无人值守场景加开关禁用，code_tool 声明全相位；相位教学段仅 research mode 注入 | `agent/huginn/agent/task_reflector.py:130-131`；`agent/huginn/agent/core.py:419-420`；`agent/huginn/agent/streaming.py:1230-1236`；`agent/huginn/tools/base.py:196-205`；`agent/huginn/agent/context.py:53-59` | 07 报告建议 1/2/3 |
| E6 | 4 套状态机收敛为 1 套（保留 ResearchPhase 或 CSM 之一作为唯一相位真源，其余显式废弃） | `audit_20260717/04` P1-6 清单 | audit 00 主题 G；07 报告 Q3 |

**主线 F：数学链深水区（第 3–4 月，主线 D/E 落地后启动）**
| # | 动作 | 落点 | 依据 |
|---|------|------|------|
| F1 | Lean 验证语义：`verify_theorem` 编译后扫描 `sorry`/`admit`/`axiom`；定理骨架生成器不再固定输出 `:= by sorry` | `agent/huginn/lean/interface.py:158-191`；`agent/huginn/lean/sympy_to_lean.py:164` | audit 07 P2-3；13 报告建议 6 |
| F2 | 守恒检查接真语义：`_lean_check_conservation` 从硬编码恒等实例改为消费用户方程的守恒残差编译；罐头 fallback 改显式 `unverified` 标记 | `agent/huginn/tools/bourbaki_tool.py:116-132,186-202` | 13 报告建议 6 |
| F3 | 量纲系统二合一：`symbolic_math dimensional_analysis` 浅实现换调 `execution/dimensional_validator.py`（代数等价/Buckingham π），删除双轨 | `agent/huginn/execution/dimensional_validator.py`；`agent/huginn/tools/bourbaki_tool.py:212-244` | 13 报告建议 7 |
| F4 | constraints scope 扩展到 code_tool 产物（CSV/JSON 数值列的 NaN/量级/单调性 sanity）；先修两条判据物理错误 | `agent/huginn/tools/adapter.py:425-437`；`agent/huginn/constraints/adapter.py:292-324` | 13 报告建议 5；audit 14 P2-3/P2-7 |
| F5 | metacog 审计器从 autoloop-only 改为工具后置钩子：数值型工具结果过数量级/量纲 sanity 表（χ、θD 类物理量先验区间） | `agent/huginn/autoloop/engine.py:2469-2499` → 主路径工具后置 | 11 报告建议 7 |
| F6 | 数值-符号互验闭环：`auto_pipeline.verify_derivative`（有限差分 vs 符号）接入科学计算任务的验收环节 | `agent/huginn/lean/auto_pipeline.py:402-498` | 13 报告 F3 正面事实 |

**P2 完成判据（全部可测量）**：
1. autoloop 单任务 7 阶段完整跑通 **0→1**，随后连续 5 个 objective 完整率 **≥60%**、tool_calls>0（当前 18/18 perceive-only）。
2. 经验池：LLM 蒸馏规则 ≥20 条；`usage_count>0` 占比 **≥50%**；写读路径统一后规则注入日志可证；噪声规则清零。
3. 同族错误复发率：虚拟路径类错误从 **25 次/轮 → ≤2 次/轮**（38 号文记载的错误模式为追踪样本）。
4. benchmark 失败回路：每个失败 run 自动产出 ≥1 条蒸馏经验；可追踪的 before/after 显示下次同族失分点下降。
5. 数学链：Lean `sorry` 拦截率 **100%**；守恒检查重言式 **0 处**；repro 数量级错误持续 **0**；数值答案工具计算率 **100%**。
6. **RCB Material 均分 ≥24**（对齐 DeepSeek-V4-Pro 轻量 scaffold 档 24.6）；**PaperBench ≥21%**（对齐官方 Claude-3.5-Sonnet 档）。
7. 连续 3 轮全量 benchmark 重跑分数**单调不降**（进化信号可信且有效的最终证据）。
8. 状态机收敛为 1 套；phase 预算通道生效或代码删除——模式层死代码 **0 处**。

---

### 主线 C 落地说明（2026-08-04）

**C1（checkpoint 悬空 tool_calls 修复）**
- `agent/huginn/agent/streaming.py` 新增 `_strip_dangling_tool_calls`：发送前扫 AIMessage.tool_calls，无对应 ToolMessage 的剥掉（部分剥保留已应答的，全 dangling 退化为纯 AIMessage）。与 `context_builder.conversation_tree_to_messages` 同源逻辑，作用在 `_build_input_messages` 返回列表（checkpointer 直读路径）。
- 自检: `python -m huginn.agent.streaming --self-check-c1`

**C2（compaction 二选一落地 + checkpoint 100MB 上限）**
- 真修剪路线（非 SummarizationMiddleware）：`_maybe_auto_compact` 60% 触发 → `_trim_checkpointer_messages` 用 `RemoveMessage + update_state` 删旧消息（G34 已落地，`HUGINN_COMPACT_STRATEGY=trim,summarize` 默认开）。
- C2 本次补的是"benchmark 运行限定 checkpoint 尺寸上限"（roadmap 完成判据 5）：
  - `agent/huginn/bench/orchestrator.py` 新增 `_enforce_checkpoint_size_limit`：run() 结束后读 `HUGINN_CHECKPOINTER_PATH` 文件大小，超 100MB 触发 SQLite `VACUUM`（重建文件回收空间，RemoveMessage 只删 row 不回收页）。
  - 新增可观测性字段 `checkpoint_size_mb` / `vacuum_triggered`，与 C8 字段一并落盘到四适配器 `_huginn_meta.json`（PB/MLE/SAB 已接，HLE 单题问答无 meta 文件）。
  - ponytail: VACUUM 只在 run() 结束后跑——运行中锁表会卡 graph。ceiling: VACUUM 需要临时磁盘空间；升级路径是 `auto_vacuum = INCREMENTAL` 在线增量回收。
- 自检: `python -m huginn.bench.orchestrator --self-check-c2`（4 场景：字段初值 / 小文件不触发 / 大文件触发 VACUUM / 无 conn 优雅降级）

**C3（预算扩容，部分）**
- RCB 已扩：`agent/huginn/cli/rcb_runner.py:4941` 默认 150→400，极限 300→600。
- PaperBench 150→600 + phase 预算通道接通：待办（N7 警告 C2 完成前不扩 PB 预算，现 C2 已完成可推进）。

**C3 余（PaperBench 150→600）**
- `paperbench_huginn.py:875-879` 默认 timeout 3600→21600, max-tool-calls 150→600.
- phase 预算通道 (`_get_budget_override` → `proposed_budget`) 已通 (G34 落地), 不需改 phase 死代码, harness `max_total_calls` 外层截断, phase budget 单轮 recursion_limit.
- N7 警告解除依据: C2 已完成 (VACUUM@100MB + 真修剪 RemoveMessage), checkpoint 崩溃修复到位.

**C4（web 检索链修复）**
- ddgs 迁移: `_search_ddgs` 双导入路径 (`from ddgs import DDGS` + `from duckduckgo_search import DDGS`), 已存在.
- 多源兜底: Tavily → arxiv → ddgs → urllib, 4 级降级链, 已存在.
- `search_unavailable` flag: 熔断时 `execute` 返回 `search_unavailable=True`, agent 据此切换策略, 已存在.
- bench 前检索健康检查: 新增 `web_search_health_check` (轻量探测 arxiv API, 无 key 无 rate limit), 集成到 `_rcb_smoke_test` 第 4 项 (失败只 warn 不 fail, bench 可走降级路径).
- 自检: `python agent/huginn/tools/web_search_tool.py --self-check-c4` (4 场景: 离线模式/熔断 flag/降级链存在/ddgs 双导入)

**C5（7 个评测补丁下沉，部分）**
- 根因修复: `agent/huginn/utils/runtime.py:42-45` `get_runtime_home()` 加 `.strip()` 处理空串 fallback. 之前 rcb_runner 要打补丁 `if not os.environ.get("HUGINN_CACHE_DIR")` 才不崩, 现根因已修.
- 其余 6 个补丁是场景配置 (RCB 跑分需要确定性/无熔断/无循环检测), 非根因 bug, 保留 env var 方式但写明代价:

| 补丁 | env var | 代价 | 是否根因修复 |
|------|---------|------|-------------|
| 限流关闭 | `HUGINN_RATE_LIMIT_ENABLED=0` | 生产环境不限流会撞 API rate limit | 否, 场景配置 (RCB prompt 长) |
| 本地 bash | `HUGINN_ALLOW_LOCAL_BASH=1` | 无沙箱隔离, 代码可访问宿主 | 否, 场景配置 (RCB 无 docker) |
| CACHE_DIR 防御 | `if not get("HUGINN_CACHE_DIR")` | — | **是**, `get_runtime_home()` 已修空串 |
| CSM 子集 | `HUGINN_RCB_CSM_SUBSET=1` | 跳过部分 CSM step | 否, 场景配置 |
| compaction root | `HUGINN_KEEP_ROOT_N=2` | 保留前 2 条 root | 否, 场景配置 |
| Rust sandbox 关 | `HUGINN_NO_RUST_SANDBOX=1` | 无 Rust 沙箱, RDKit+sklearn 兼容 | 否, 场景配置 (Rust sandbox 兼容性) |
| LLM decider 关 | `HUGINN_COGNITIVE_LLM_DECIDER=0` | 规则版 decide_fn 替代 LLM | 否, 场景配置 (跑分确定性) |
| 熔断关 | `HUGINN_HEALTH_MONITOR=0` | 无健康监测, file_read 误触发消除 | 否, 场景配置 (熔断误判) |
| 循环检测关 | `HUGINN_FEATURE_LOOP_DETECTOR=false` | 无循环检测, code_tool 反复跑不被拦 | 否, 场景配置 (循环检测误判) |

  ponytail: 不做 toml 下沉 — 7 个补丁都是 env var 强制赋值, 生产路径不设这些 env 就是默认行为. toml 下沉会改 config schema + rcb_runner 大量代码, 收益有限 (场景差异不是 bug). 升级路径: 若未来要统一, 建在 HuginnConfig 加 `[benchmark]` 块, rcb_runner 从 cfg 读.

**C7（file_write_tool Windows 路径修复 + 分块写入）**
- `agent/huginn/tools/file_write_tool.py` 加 `os.path.normpath` 归一化反斜杠/盘符，`os.path.commonpath` 替代 `relative_to` 处理 `..` 和符号链接；>1MB 用 64KB 分块 write 避免 OOM。
- `agent/huginn/agents/subagent.py` coder 恢复 `file_write_tool`（C7 修复后不再被 Windows 路径 bug 卡住）。
- 自检: `python agent/huginn/tools/file_write_tool.py --self-check-c7`

**C8（可观测性字段 meta 落盘）**
- `agent/huginn/bench/orchestrator.py` 暴露 `tool_calls_used / turns_used / context_overflow_count / compaction_count / crash_traceback`（C8）+ `checkpoint_size_mb / vacuum_triggered`（C2）。
- 四适配器 `*_huginn.py` 的 `_huginn_meta.json` 落盘上述 7 字段（PB/MLE/SAB 已接，HLE 单题问答无 meta 文件）。
- `context_overflow_count` 通过 `_is_context_overflow` 检测；`compaction_count` 通过 chunk 里的 `compaction_triggered` 标记计数；`crash_traceback` 在 except 块用 `traceback.format_exc()` 捕获。

---

## 四、不做清单（防架构稀释）

以下 11 条全部有本地证据支撑。违反任何一条，都会重复 `audit_20260717/00` 主题 G「架构稀释」或主题 A「装置空转」的老路。

| # | 不做 | 理由与证据 |
|---|------|-----------|
| N1 | **不新建任何状态机 / 编排框架 / 自改进系统** | 已有 4 套状态机 + 3 套编排 + 2 套自改进并行（`audit_20260717/00:138-141`；audit 04 P1-6）；第 5 套状态机式的增长正是当前病灶本身。模式层动作只有两种：接线或拆除（E5/E6）。 |
| N2 | **主路径接通前，不扩充 metacog / self_improvement / cognitive 任何新模块** | 现有模块的问题不是功能不够而是不在主路径上；新认知层注定重复「写多读零」（`analysis_20260717/06:179`）。 |
| N3 | **不把 auto_pipeline / HuginnLean 深接 benchmark 路径** | 对当前失分类别无可归因作用面（上限约 1.5 个 benchmark 小项）；投入高、属长线工程，应在执行基线修复后立项（`analysis_20260717/13:216`）。P2 只修语义（F1/F2），不做深度集成。 |
| N4 | **不做 KG schema / 知识图谱推理 / hypothesis tree** | 非跑分瓶颈（`audit_20260717/21:85-87` G6/G7）；RAG + longterm memory 现状够用，P2 之后再评估。 |
| N5 | **不再堆 prompt 级规劝**（PHASED PROTOCOL、更多 "NOT acceptable" 段落） | 防线溃败点全部在结构与预算层，不在措辞层；prompt 防线在预算压力下必然失守（`analysis_20260717/12:196`）。已有 prompt 中的有害条款（复杂度上限、强制早交卷）反而要删（A3）。 |
| N6 | **评测信号修复（C6）完成前，不把任何分数接入进化/强化回路** | 会制度化 reward hacking（`analysis_20260717/00:117-118,127-128`）；已有前科——自进化把 medal 死代码兜底成「经验」（`analysis_20260717/02:45`）。 |
| N7 | **checkpoint 崩溃修复（C1/C2）完成前，不扩任何 benchmark 预算** | 预算越大崩溃越多（`analysis_20260717/09:92,148`）；PaperBench M1 已在 16/150 次调用处整 run 报废。 |
| N8 | **不追 HLE 跑分** | 适配器从未真正跑过（`analysis_20260717/01:104`）；先把 5 大现有 benchmark 修到可信，HLE 排在本路线图之外。 |
| N9 | **不再新增「评测模式特供补丁」（setdefault 式）** | 被测系统与生产系统分裂，benchmark 测的永远是另一个系统（`analysis_20260717/08:239-245,255`）；已有 7 个必须在 P1 下沉或删除（C5）。 |
| N10 | **不保留无 usage 考核的经验库** | 25 条零使用规则 + 20 条未验证蒸馏知识已是噪音库（`analysis_20260717/06:58-64`）；经验库必须带淘汰机制（E2），否则「经验」膨胀即「偏见」膨胀。 |
| N11 | **不自创评分细则冒充官方成绩引用** | SAB 25 分非原生 success_rate（`sab_huginn.py:8-11`；audit 16 P1-11）；所有自创指标永久改名 + 水印（如 `code_inspection_score`），对外引用必须带 `metric_provenance`。 |

---

## 五、风险与不确定性

1. **模型天花板**：P1 判据（RCB ≥20、PB ≥15%）按 deepseek-chat 能力设定，依据是同族 DeepSeek-V4-Pro 轻量 scaffold 24.6 的对照（`analysis_20260717/12:51-54`）与 audit 20 预算-产出曲线（13.52→~28 @600 calls，推断）。若 P1 末期分数停在判据下方，按 00 报告根因权重，下一步杠杆是**换工具型强模型**（模型配置维度），而不是继续加 scaffold——届时重估。
2. **能力层硬短板**：「给了答案落实不了」（PaperBench 4 条明文答案全错、跨三代未修）是长上下文指令落实能力缺陷，A1 的 rubric 断言脚本化（C6 的「答案改验收程序」）是机制兜底，但不能保证根治；该短板决定 PaperBench 的上行斜率。
3. **分数波动性**：RCB 同任务 2.4× 方差、judge 抖动 ±3 分（`analysis_20260717/05:87-90`）——所有判据必须以「修复后 ≥3 次重跑均值」口径验收，单次分数不构成判据证据。
4. **Windows 特有失败**（占长任务失败 40%）与「修复回潮」（裸 except 从 365 恶化至 889 处，`audit_20260717/00:128-130` 主题 E）是本路线图的持续背景税；B6/E2 的 CI 断言牙齿是防回潮的唯一机制。

---

## 附 A：关键证据索引（本报告判据的锚点数字）

| 锚点 | 数值 | 出处 |
|---|---|---|
| RCB Material 均分（当前） | 13.7（4 题均值；逐题最好 14.1） | `analysis_20260717/12:52` |
| 对照档 | DeepSeek-V4-Pro 轻量 scaffold 24.6；ResearchClaw 19.3；Claude Code 25.5 | `analysis_20260717/12:40-52` |
| PaperBench（当前，注水） | 13.52 / 2.23 / 33.85；执行叶均分 0.8–1.2 | `analysis_20260717/03:59-63` |
| MLE 天花板达成率 | 99.4% / 99.8% / 95.2%（synthetic） | `analysis_20260717/02:56-63` |
| autoloop 空转 | 18/18 perceive-only、0 工具调用 | `analysis_20260717/06:30-43` |
| 进化规则 | 25 条、usage_count 全 0 | `analysis_20260717/06:58-64` |
| 预算量级 | 150 calls = 官方 3%；112/174 叶纯因预算耗尽 0 分 | `audit_20260717/20:145-149,97-110` |
| 工具层直接失分占比 | 8 个出分单元 62.5% | `analysis_20260717/08:195-213` |
| 根因权重 | 执行闭环断裂（主因）＞ harness 伪影压制（主因）＞ 模式层（15–25%）＞ 预算（10–20%） | `analysis_20260717/00:90-99` |
| 丢分结构 | 能力层 50–60%、预算/上下文 20–30%、harness bug 10–20% | `audit_20260717/20:166-178` |

---

## 附 B：返回 Orchestrator 的摘要与关键结论

### 摘要（≤300 字）

本报告综合本地 31 份审计/归因资产，给出 huginn 中上→顶级路线图。九宫格定档：六维 2 分（骨架在、神经未接）+ 三维 1 分（验证纪律/进化回路/评测信号实质性失效），无维超 2 分——统一病症是「设计成熟度高于实现生效度」，顶级距离 = 器官未通电而非缺器官。三阶段：P0 修复周（10 项小时级共识，判据：SAB 重评 ≥55、占位交付 0 次、RCB 13.7→18）；P1 能力月（执行闭环 + 验证文化 + 预算重构 + 评测卫生，判据：RCB ≥20、PB 原生协议 ≥15%、执行叶 0.8→30、可信基线建立）；P2 进化季（经验池三段 + LLM 蒸馏 + 数学链深水区，判据：autoloop 7 阶段 0→1、RCB ≥24、连续 3 轮分数单调不降）。附 11 条不做清单防架构稀释。核心立场：先修测量、再修执行、后谈进化；机制兜底能力；收敛优于新增。

### 最关键结论（一行一条）

1. 九宫格无维超 2 分：差距不是点状短板而是全系统「设计-生效」断层，路线图形态因此是「P0/P1 全接线、P2 才生长」。
2. 顺序即战略：先修测量（P0）→ 再修执行（P1）→ 后谈进化（P2）；可执行性硬门与诚实评测信号缺一，接进化回路 = 制度化 reward hacking。
3. scaffold 杠杆 > 模型：同族 DeepSeek-V4-Pro 轻量 scaffold 24.6 vs huginn 13.7，P0+P1 机制修复（不换模型）足以把 RCB Material 推到 20–24 档。
4. 验证纪律是最深的一维（1/5）：占位数值、虚构数据、硬标签、裸答猜值四种失信形态跨 benchmark 复发，需「产物级门控 + 数值重算 + 基线闸门」三件套根治。
5. 收敛优于新增：已有 4 状态机/3 编排/2 自改进并行，11 条不做清单是路线图的另一半——顶级画像靠接线既有骨架，不靠第 5 套框架。

---

## 附 C：C6 评测卫生验收记录（2026-08-04）

**验收结论：通过（机制层）**

C6 修的是评测链路本身的卫生，不修 agent 能力。能力问题（早交卷/执行弱/无基线闸门）划入 C7-C9，不阻塞 C6。

### 交付物逐条核对

| 子项 | 落点 | 证据 | 状态 |
|------|------|------|------|
| structai 全量移除 | `judge_helper.py` 替代，5 适配器 import 通过 | rcb_score/paperbench/sab/mlebench/hle | ✅ |
| judge 同源检测 | 所有评分入口 WARNING + `JUDGE_SAME_SOURCE_OK` | `.env` JUDGE_* 环境变量 | ✅ |
| 异源 judge 配置 | `.env` JUDGE_API_KEY/BASE/MODEL_NAME | Kimi/OpenAI 兼容，待有效 key 实测 | ✅ 机制就位 |
| 私有标签隔离 | MLEBench → `workspace.parent/_mle_private_*` | s3e18 synthetic 评分 0.8026 验证隔离闭环 | ✅ |
| HLE per-question memory 隔离 | `.hle_memory/<qslug>/` + checkpoint 隔离 | `hle_huginn.py` | ✅ 代码就位 |
| PaperBench 答案泄漏清除 | regex override 删 + 叶节点答案出 prompt | `paperbench_huginn.py` | ✅ |
| 断言脚本机制 | prompt 要求写完 .py 后 assert，产物 `assert_*.json` | `paperbench_huginn.py` prompt | ✅ 机制就位 |
| 评分链路通畅 | SAB 3/3 (72/88/88)，MLEBench s3e18 (0.8026) | `_score.json` | ✅ |

### 划入 C7+ 的能力问题（不阻塞 C6）

| 问题 | 根因 | 归属 |
|------|------|------|
| 早交卷（402-1118s/7200s 预算） | stagnation break 与 P5 wall_clock 守卫方向矛盾，`rcb_runner.py:2522-2537` | C7 控制流修复 |
| 执行弱（PaperBench Execution≈0%） | 执行链路问题 | C8 执行链路 |
| 无基线闸门（spaceship 0.633 < if 规则 0.642） | 提交前缺 sanity check | C9 提交前闸门 |

### 未闭环项（不阻塞验收，记录待办）

1. 异源 judge 实测：当前 deepseek-chat 同源自评 + `JUDGE_SAME_SOURCE_OK=1`（标记不可比基线）。待有效 Moonshot key 后 `--score-only` 重评闭环。
2. PaperBench 断言产物验证：断言脚本指令在 prompt 中，但 assert_*.json 是否真实生成取决于 agent 执行（属能力问题）。PaperBench 重跑在后台进行中。
3. HLE 网络不可达：huggingface.co DNS 被劫持到 127.0.0.1，需启动代理后补跑。非 C6 缺陷。

### N6 gate 解除状态

C6 通过 → **N6 gate 解除**：分数可接入进化/强化回路（D2 benchmark 失败→分析→更新→重试回路的前置条件满足）。但当前基线为同源自评（`JUDGE_SAME_SOURCE_OK=1`），接入进化回路前建议先完成异源 judge 实测。

---

## 附 D：C7 控制流修复 + 最终验收结论（2026-08-04）

### C7 修复（早交卷根因）

主循环三处早退与 P5「跑满 timeout」矛盾，已删两处，保留降温监测：

| 早退点 | 原行为 | 修复 |
|--------|--------|------|
| stagnation break（`rcb_runner.py:2522-2539`） | report.md hash 不变 2 轮 → break | 改为升温（`_t_hot += 0.5`）+ 日志，wall_clock 唯一控制退出 |
| score_history monotonic-decrease break（`rcb_runner.py:2559-2566`） | 分数单调下降达阈值 → break | 改为 advisory 日志 |

**保留的合理退出点**：P5 wall_clock 守卫 / max_calls 达上限 / cumulative_audit blocked / darwin_ratchet stop（质量评估，非停滞检测）/ TASK COMPLETE gate（连续驳回 3 次接受）。

**自检**：`rcb_runner.py --self-check-c7` 正向验证升温 + advisory 在位，PASS。

**未动**：PHASED PROTOCOL prompt（「call 20 write report」）。属 A3 prompt 层，后续单独评估，避免删后 agent 不写 report 的另一个极端。

### 最终验收结论（2026-08-04）

**整体：通过（机制层）**

| 项目 | 结论 | 依据 |
|------|------|------|
| C6 评测卫生 | ✅ 通过 | 附 C：8 项交付物全就位，SAB 3/3 + MLEBench s3e18 验证链路通畅 |
| C7 控制流早交卷修复 | ✅ 通过 | 附 D：stagnation / score_history 早退删除，升温保留，自检 PASS |
| 异源 judge 实测 | ⏳ 待办 | 同源自评标记不可比基线，待有效 Moonshot key 后 `--score-only` 重评 |
| 全量重跑建立可信基线 | ⏸ 挂起 | 用户决定不重跑；后台进程已停止 |

**验收要点**：
- C6/C7 是**机制层修复**，验收对象是评测链路卫生 + 控制流一致性，不是 agent 跑分。
- 能力问题（执行弱 / 无基线闸门）已明确定位根因，属 C8/C9 后续范围，不阻塞本次验收。
- N6 gate 已解除，但接入进化回路前应先完成异源 judge 实测，避免 reward hacking 制度化。

---

## 附 E：RCB 定制化下沉方案（2026-08-09）

**触发**：CausalGame 接入后，用户指出"agent 并非为 RCB 定制，RCB 只是查漏补缺手段之一"。逐项核查 `rcb_runner.py` / `rcb_huginn.py` 的定制逻辑，发现"高度定制化"是三种性质不同东西的混合，多数不是污染。

### E.1 三类定制化分层

| 类别 | 实例 | 性质 | 处置 |
|------|------|------|------|
| ① 环境场景适配 | `rcb_huginn.py:40-99` 路径重定向、关熔断/关fork、缓存重定向 | 被测系统运行环境适配，不改变能力 | 归并为共享 benchmark 环境适配入口，剥 `RCB_` 前缀 |
| ② 通用认知机制误放 runner | `rcb_runner.py:1134-1173` HypothesisManifold / MCMC / abductive inference / posterior hint / 跨任务 memory | 通用引擎被接线在单一 runner，用 `SimpleNamespace` 模拟 engine（`rcb_runner.py:1147`"不接 AutoloopEngine"） | **下沉到 core 主循环（真正的病）** |
| ③ benchmark 业务 | PHASED PROTOCOL、RCB_DELIVERABLES、rubric、`_RCB_STEP1_NEVER` 工具黑名单 | benchmark 领域语义，合法 | 留在 benchmark 适配层 |

### E.2 真实病根与解法

RCB"定制化"的核心不是 setdefault 环境补丁（无害），而是 **②通用认知引擎被临时接线在单一 runner**。同一引擎（`hypothesis_manifold.py` 等）在 core 存在，却没进 core 主循环，导致 CausalGame / SAB / 真实科研任务全部用不上。

解法与 CausalGame 下沉同哲学：**接线下沉，不是去定制化**。

| # | 动作 | 落点 | 判据 |
|---|------|------|------|
| E2-1 | HypothesisManifold/MCMC/abduction 初始化与调用从 `rcb_runner.py` 迁到 core 主循环（AutoloopEngine 或 agent 主循环），删 SimpleNamespace 绕行 | `agent/huginn/autoloop/engine.py` / `agent/huginn/agent/core.py`；`rcb_runner.py:1134-1173,1813,2057` | RCB 由宿主退化消费者；CausalGame/SAB 可不改 runner 即继承 |
| E2-2 | 跨任务 memory / posterior hint 入口从 `HUGINN_RCB_CROSS_TASK*` 泛化，复用 core 已有 `_cross_task/` 共享 db + curiosity 闭环 | `rcb_runner.py:1192-1195,2504,4943-4952` | 泛化配置，非 RCB 专属 |
| E2-3 | ①环境补丁归并为共享入口，剥前缀 | `rcb_runner.py:57-135`、`rcb_huginn.py:40-99` | 各 benchmark 复用同一入口，无重复 setdefault |

### E.3 前置约束

- E2-1 是核心运行循环改动，属高风险，须先固化行为基线：迁移前后 RCB 同任务重跑分数**单调不降**（对齐路线图 P2 判据 7 口径）。
- 遵循 N9（不再新增评测特供补丁）：下沉后 rcb_runner 不应再出现新的 `SimpleNamespace` 绕行或 `_RCB_*` 专属机制。
- ②是"接线"不是"新增"：引擎已存在，只移接入点，不扩模块（对齐 N1/N2）。
