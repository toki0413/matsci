# 16 · 数学直觉与自我进化：SOTA 实现路径调研与 huginn 接入方案

- **诊断对象**：huginn-agent（项目根 `C:\Users\wanzh\Desktop\matsci-agent`）
- **专项角色**：神经符号 / 自进化系统研究员
- **日期**：2026-07-17
- **任务**：调研「数学直觉」与「自我进化」的 SOTA 实现路径（FunSearch、AlphaEvolve、AlphaProof/AlphaGeometry、Lean 生态自动定理证明 agent、神经符号科学发现、经验记忆机制），为 huginn 设计可落地接入方案。
- **方法**：外部调研（kimi_search_v2，2026-07-17，全部结论附论文名/URL+时间）+ 本地只读代码核实（引用 `路径:行号`，关键行号均已现场复核，非转引）。本地归因结论直接引用 `analysis_20260717/00、06、11、13` 四份报告并标注。
- **边界**：只读；未修改任何项目文件；本报告为唯一新建文件。

---

## ① 数据与方法

### 外部调研对象（全部于本轮检索确认，关键数字均来自下列一手来源或其转述页）

| 系统 | 来源 | 时间 | 本报告采用的要点 |
|---|---|---|---|
| FunSearch | Romera-Paredes et al., *Nature* 625:468-475，arXiv:2309.02840；PubMed PMID 38096900；DeepMind blog；github.com/google-deepmind/funsearch | 2023-12-14 在线 / 2024-01 刊出 | 程序空间搜索（进化函数而非解）、冻结评估器、skeleton 只进化关键逻辑、best-shot prompting（k=2）、岛屿进化+签名聚类保多样性；cap set 20 年最大下界改进 |
| AlphaEvolve | Novikov et al., DeepMind Technical Report，白皮书 PDF（storage.googleapis.com/deepmind-media/.../AlphaEvolve.pdf）；arXiv:2604.26275 引述 | 2025-05 | Gemini Flash+Pro 集成做变异、自动评估器打分、进化整个代码库（diff 级）；4×4 复数矩阵乘法 48 次标量乘法（改进 Strassen 1969 结果）、50 个开放数学问题中 20% 被推进、回收 Google 0.7% 算力、FlashAttention 加速 23% |
| AlphaProof / AlphaGeometry 2 | DeepMind blog「AI achieves silver-medal standard…」；Nature 2025 方法论（Hubert et al. 2025）；theorempath.com 综述页（2026-07 检索） | 2024-07-25 宣布；2025-11 刊出 | RL + Lean 4 kernel 二元确定性奖励；autoformalization 产合成数据；测试时 RL（对目标题生成变体先训后证）；合计 4/6 题 28/42 分（IMO 2024 银牌当量）；AlphaGeometry 2 为神经提议器+符号演绎引擎 |
| DeepSeek-Prover-V2 | Ren et al., arXiv:2504.21801；综述 arXiv:2606.08728 §VI-D 转述 | 2025-04 | 子目标分解：DeepSeek-V3 生成自然语言证明草图+含 sorry 占位符的 Lean 模板，7B prover 递归解子目标；MiniF2F-test 88.9%（pass@8192）、PutnamBench 49/658 |
| Kimina-Prover | Wang et al. 2025（Project Numina）；综述 arXiv:2607.07779 §4.1 表 | 2025-04/07 | 结构化推理模式（自然语言推理与 tactic 块交错）+ 大规模 RL（Qwen2.5-72B）；MiniF2F 92.2% |
| Hilbert / Numina-Lean-Agent / Seed-Prover | arXiv:2509.22819（Hilbert）；arXiv:2603.20405（Numina-Lean-Agent 案例）；arXiv:2507.23726（Seed-Prover）；SOTA 表 arXiv:2607.07779 | 2025-07 ~ 2025-09 | 多智能体编排（reasoner+prover+verifier+retriever）递归子目标分解，MiniF2F 99.2%；Numina-Lean-Agent 用 Claude+MCP Lean server 解全部 12 道 Putnam 2025；Seed-Prover 99.6% |
| LLM-SR / LaSR / SGA / DrSR | LLM-SR：Shojaee et al., arXiv:2404.18400；LaSR/SGA/DrSR 由 arXiv:2602.13021 §2.3 与 ICML25 论文引述确认 | 2024-04 起 | 方程发现=程序进化+LLM 先验；评估域含材料应力行为；SGA 为双层优化（LLM 提假设+物理仿真验证）；DrSR 双推理（数据洞察+反思反馈） |
| Voyager | Wang et al., arXiv:2305.16291（TMLR 2024）；多篇 2025-2026 论文引述确认三组件 | 2023-05 | 自动课程表、**可执行代码技能库**（入库前经执行反馈+自验证）、embedding 检索复用、迭代提示机制 |
| Reflexion | Shinn et al., NeurIPS 2023，arXiv:2303.11366 | 2023-03/09 | Actor/Evaluator/Self-Reflection 三角色 + episodic memory 的「语言强化学习」；HumanEval pass@1 91%（对照 GPT-4 80%）、AlfWorld 97% vs 75% |
| ExpeL | Zhao et al., AAAI 2024 (Oral)，arXiv:2308.10144；项目页 andrewzh112.github.io/expel | 2023-08 / 2024-03 | 经验池收集成败轨迹 → insight 抽取（ADD/UPVOTE/DOWNVOTE/EDIT 四算子）→ 推理时召回 insight+相似成功轨迹；跨任务迁移 HotpotQA→FEVER 70% SR |
| EvoScientist | arXiv:2603.08127（本地报告 11 已调研） | 2026-03 | EMA 三种写入（IDE/IVE/ESE）+ embedding 检索闭环；「预算内无可执行代码=提案失败」rule-based 硬判据 |
| CodeEvolve | arXiv:2510.14150 | 2025-10 | AlphaEvolve 的开源可操作化：岛屿 GA + 加权 LLM 集成 + inspiration 交叉 + meta-prompting + 深度定向精炼 |

### 本地证据（本轮现场复核行号）

| 事实 | 位置 |
|---|---|
| 四适配器 tool_filter 摘除全部数学工具 | `sab_huginn.py:53-62`（SAB_TOOL_FILTER 8 项，无 symbolic_math/bourbaki/lean/unit/validate） |
| Lean 守恒检查是硬编码恒等重言式（`myEvolution := id`、`invariant := id`），与用户方程无关 | `agent/huginn/tools/bourbaki_tool.py:116-132`（本轮逐行复核） |
| 定理验证=编译通过+定理名子串匹配，无 sorry/axiom 扫描 | `agent/huginn/lean/interface.py:158-191`（本轮复核，docstring 自承 "thin wrapper around lake build plus a regex check"） |
| 定理骨架固定输出 `:= by sorry` | `agent/huginn/lean/sympy_to_lean.py:164`（报告 13 F3） |
| 数学验证钩子只活在 autoloop 且要求结构化 `equations` 字段 | `agent/huginn/autoloop/engine.py:4128-4209`（本轮复核） |
| 规则生成是手写 if-else 模板匹配（VASP/Gaussian/LAMMPS 关键词），全程无 LLM | `agent/huginn/evolution/engine.py:502-547`（本轮复核） |
| 规则匹配是双向子串匹配，路径级不泛化 | `agent/huginn/evolution/engine.py:571-574`（本轮复核，注释自承 "Simple substring matching"） |
| 规则写入在 reflection 主路径、读取只在空转的 autoloop | `agent/huginn/agent/reflection.py:200-230`（本轮复核）；`autoloop/engine.py:3236`（报告 06 F3） |
| 写入路径 `~/.huginn/logs/` 与读取路径 `$HUGINN_CACHE_DIR/evolution_rules.json` 永不相交 | `agent/huginn/evolution/logger.py:54-58` vs `agent/huginn/context_builder.py:403-404`（本轮均复核） |
| 蒸馏器也是模板拼接（`_generate_error_lesson`/`_find_common_params`，无 LLM），仅 Jaccard 去重 | `agent/huginn/evolution/knowledge_distiller.py:87-195`（本轮复核）；schema 字段 :25-45 |
| bench system prompt 只有建议性「Use code_tool to compute」，无强制 gate | `agent/huginn/bench/runner.py:537-554`（本轮复核） |
| repro 基准 χ=1.0 / θD=6.6K 数量级错误发生在工具全量可用时 | `bench_repro.log`（报告 13 F2） |
| autoloop 18/18 轨迹 perceive+report、0 工具调用；`_report` 无数据硬门虚构结果 | 报告 06 F1/F2（`autoloop/engine.py:1353-1362, 2304, 5016-5022`） |
| 完整代数量纲系统 `dimensional_validator.py` 只被 ontology 字符串引用 | `agent/huginn/execution/dimensional_validator.py`（存在性本轮确认；引用面见报告 13 F5） |
| 评测信号有毒（rubric 泄漏、judge 同源、保底分） | 报告 11 发现 7 / `audit_20260717/16` P0-1/P1-3/P1-6/P1-7 |

---

## ② SOTA 横向综合：五条不变量

把上述系统摆在一起，「数学直觉」与「自进化」的 SOTA 实现共享五条工程不变量——huginn 的接入设计逐条对标：

1. **进化/学习只作用于「可被自动评估的产物」**。FunSearch 进化的是程序且评估器冻结（Nature 2024）；AlphaEvolve 进化代码库 diff、评估器自动打分（DeepMind TR 2025-05）；AlphaProof 的奖励是 Lean kernel 二元判定。反例正是 huginn 现状：进化的「产物」是模板字符串规则，评估器缺失，于是 25 条规则 usage_count 全 0 也无从区分好坏（`~/.huginn/logs/evolution_rules.json`，报告 06 F3）。
2. **奖励信号必须二元、确定、快速、难投机**。Lean kernel、单元测试、冻结评估器皆是。「judge 同源 LLM + 保底分 + 答案泄漏」的 huginn 评测面（`audit_20260717/16`）若直接接入进化回路，将制度化 reward hacking——EvoScientist 的失败判据之所以有效，正因为「可执行性」难以投机（报告 11 发现 7）。
3. **记忆入库前必须有验证门**。Voyager 的技能代码须经执行反馈+自验证才入库（arXiv:2305.16291）；ExpeL 的 insight 靠 UPVOTE/DOWNVOTE 算子做群体一致性筛选（AAAI 2024）；ESE 强制「另一个工程师能据此重现」的细节级别（arXiv:2603.08127）。huginn 的 `KnowledgeDistiller` 无验证门，`verification_status` 恒 "unverified"（`knowledge_distiller.py:38`）。
4. **检索注入是闭环的最后一公里**。Voyager embedding 检索技能、ExpeL 按任务相似度召回成功轨迹、EvoScientist mxbai-embed-large top-k 注入——全都是「写入→索引→按需注入→用后归因」的完整链路。huginn 断在两处：写读路径不相交（`logger.py:56` vs `context_builder.py:403-404`），且注入后无归因（无 injected_ids 记录，usage_count 永远不涨）。
5. **多样性维护防止早熟收敛**。FunSearch 岛屿+签名聚类、AlphaEvolve MAP-Elites 式行为多样性（arXiv:2602.22425 §2.1 引述）。huginn 的「经验」25 条全是同一错误的变体（报告 06 F3）——没有多样性机制，也没有去重之外的结构性组织。

另有一条 2025 后的趋势判断，直接决定 huginn 该走哪条路：**定理证明的前沿正从「自训专用 prover」转向「通用前沿模型+agentic scaffold 围绕现成 Lean server」**——Numina-Lean-Agent 用 Claude+MCP Lean server 解全部 12 道 Putnam 2025（arXiv:2603.20405）；Hilbert 用 Gemini 2.5 当 reasoner、DeepSeek-Prover-V2-7B/Goedel-32B 当 prover、Kimina Lean Server 当 verifier、mathlib 检索当 retriever 达 MiniF2F 99.2%（arXiv:2509.22819 / 2607.07779）；综述结论「frontier NTP 越来越多建立在 verifier-grounded 数据生成、分解与搜索时扩展之上，而非静态监督学习」（arXiv:2605.30914）。**对 huginn 的含义：不要自训 prover，把 Lean 当 verifier 服务编排。**

---

## ③ 数学直觉的可工程化分层（L0→L4）

分层原则：每一层只依赖下层的可信输出；**上层在下层未达标前不产生任何信息增量**（这是 huginn 当前「语义为空」的根治原则——`bourbaki_tool.py:116-132` 的重言式恰恰是在 L0/L1 缺失时直接跳到 L3 的结果）。每层给出：定义 → SOTA 参照 → huginn 现状（file:line）→ 最小可行实现（MVI，含落点）→ 验收判据 → 收益标注（【直接收益】= 对当前 benchmark 分数有可归因作用；【中期收益】= 修复执行链后兑现；【长期投资】= 用户核心目标的能力地基，当前不直接涨分）。

### L0 计算纪律（computation discipline）：任何数值结论必须携带计算证据

- **定义**：agent 提交的每个数值答案必须能指认产生它的工具调用；纯文本裸答在 harness 层被拒收。
- **SOTA 参照**：这不是某个具体系统，而是全部 SOTA 的共同前提——FunSearch 的整个设计动机就是「LLM 会编造，所以只信评估器跑出来的分」（Nature 2024，PMID 38096900）；Reflexion 的 Evaluator 角色同理（NeurIPS 2023）。LLM 算术不可信是所有验证优先系统的公理。
- **huginn 现状**：工具被 filter 摘除（`sab_huginn.py:53-62` 等四个适配器）；内部 bench 工具全量可用仍裸答出错，prompt 只有建议无 gate（`bench/runner.py:537-554`）；χ=1.0（照抄题面常数）、θD=6.6K（丢指数）两题耗时全 10 题最短——未走任何计算工具的时间特征（报告 13 F2）。
- **MVI**（~50 行，报告 13 Q4 方案的工程化）：
  1. `bench/runner.py` `_run_task` 在 `task.evaluate(output)`（:493）前插入确定性 gate：若 evaluator 含数值提取（`_extract_number` 类）且本次会话 `code_tool`/`symbolic_math_tool` 调用计数为 0 → (a) 第一次打回并注入 "You must compute, not guess. Call code_tool now."；(b) 仍裸答判 FAIL 并记录 `gate=compute_missing`。
  2. 四个外部适配器 filter 各加 `symbolic_math_tool` + `unit_tool` 两行；各 system prompt 把 "Use code_tool to compute" 从建议改为规则："Any numeric final answer MUST be produced by a tool call; state the tool call id next to the number."
  3. trajectory schema 增加 `compute_evidence: [{answer_value, tool_call_id}]` 字段——这是后续 L1/L2 与自进化经验池的统一证据格式。
- **验收判据**：repro bench 10/10；人为注入「照抄常数」「丢指数」两种扰动各 5 次，gate 拦截率 100%；trajectory 中每个数值答案均有 `compute_evidence`。
- **收益**：【直接收益】——repro 2 题翻转（报告 13 Q3 表：上限 80%→100%），并对全 benchmark 形成通用反裸答约束。

### L1 量纲 / 守恒 sanity：把「物理合理性」从 warn 级 advisory 升级为 block 级 post-condition

- **定义**：数值结果（工具产物与最终答案）在提交前过量纲一致性检查与物理量先验区间检查；违例 block 并回注错误类别。
- **SOTA 参照**：AI Feynman 把量纲一致性作为符号回归的硬剪枝（Udrescu & Tegmark, *Science Advances* 2020）；SGA 用物理仿真做假设验证的双层优化（Ma et al. 2024，arXiv:2602.13021 §2.3 引述）；PG-SR 的 prior constraint checker 在进化早期拦截伪方程并防止其作为上下文污染后代（arXiv:2602.13021）——「错误产物不得进入记忆/上下文」与 huginn 的经验池毒性问题（报告 06 F2 虚构报告）同构。
- **huginn 现状**：两套量纲系统双轨且都未进主循环——`execution/dimensional_validator.py`（代数量纲+Buckingham π+SymPy 推断，深）仅被 ontology 字符串引用；`bourbaki_tool.py:212-244` fallback 只认 5 个基本量单位名（浅）。constraints 已接线但只挂在仿真工具、warn 级、且两条判据本身有物理错误（`tools/adapter.py:425-437`；`audit_20260717/14` P2-7/P2-3）。
- **MVI**：
  1. **双轨合并**：`symbolic_math` 的 `dimensional_analysis` action 改为调用 `execution/dimensional_validator.py`，删除 `bourbaki_tool.py:212-244` 的浅 fallback（或显式标注 `unverified`）。
  2. **scope 扩展**：给 `code_tool` 产物（CSV/JSON 数值列）注册新 constraint scope——NaN/无穷、量级先验表（如磁化率 χ∈[1e-6, 1e-1]、Debye 温度 θD∈[10, 2000] K、形成能∈[-10, 10] eV/atom，按材料科学常用量建 20-30 条）、单调性/守恒残差（如概率列和=1±1e-6）。落点：`constraints/adapter.py` 默认库（:292-324，报告 13 F4）新增 scope 注册；`tools/adapter.py:425-437` 现有评估通路不变。
  3. **级别分级**：L0 的 `compute_evidence` 携带的量若违反量级先验 → block（这是拦截 θD=6.6K 的机制）；量纲不一致但量级合理 → warn 并要求 agent 显式确认单位制。
  4. 先修 `audit_20260717/14` P2-7/P2-3 两条错误判据与 `physics_auditor.py` 压力单位 bug——**判据错了，gate 越强伤害越大**。
- **验收判据**：χ=1.0/θD=6.6K 两类注入错误 100% block；repro 现存 8 道 PASS 题零误伤（false positive = 0 是硬指标，误伤会逼 agent 学会绕过 gate）；所有 block 事件写入 trajectory 供经验池消费。
- **收益**：【直接收益】（repro 类与 RCB 定量声明的辅助防线），同时是 L2-L4 的可信输入前提。

### L2 数值-符号互验：同一结论必须过两个独立通道

- **定义**：关键数值/公式结论须同时有符号通道（sympy 推导）与数值通道（code_tool 计算/有限差分/数据拟合），两通道一致才允许提交；不一致触发复核而非静默选一。
- **SOTA 参照**：DeepSeekMath-V2 训练 generator-verifier 对、按推理步发奖励，Putnam 2024 达 118/120（arXiv:2607.06820 §2.1 引述）；DSP「draft–sketch–prove」把非形式草图与形式证明互相锚定（DSP+ MiniF2F 83.6%，arXiv:2607.07779 §4.1）；FunSearch 的冻结评估器本质是「程序输出 vs 目标函数」的互验。
- **huginn 现状**：`auto_pipeline.py:402-498` 的 `verify_derivative`（有限差分 vs 符号导数）方向正确但只被 autoloop `_run_math_validation` 调用，且要求上游结果携带结构化 `"equations"` 字段才触发——benchmark 路径零交集（`autoloop/engine.py:4128-4209`；报告 13 F5）。RCB 的 sanity 防线是纯 LLM 文本对照，机制上识别不了「数值无产物支撑」（`rcb_runner.py:493-499`，报告 13 F6）。
- **MVI**：
  1. **抽工具**：把 `verify_derivative` 从 autoloop 钩子里抽出为独立 `verify_tool`（注册进主 registry，进四个适配器 filter），输入改为「表达式+采样点+数值结果」三元组，不再依赖 autoloop 的结构化字段。
  2. **双通道协议**：对 L0 `compute_evidence` 中标记为 `key_result` 的量，要求 `symbolic_channel`（sympy 化简/解析求值）与 `numeric_channel`（code_tool 数值）相对偏差 < 容差；不满足则注入复核指令（而非自动选边）。
  3. **报告-产物一致性核对器**（报告 13 #3 的方案）：确定性脚本提取 report.md 数值声明，与 `outputs/` 产物实际值比对，不一致项注入 RCB Step 3 prompt 标红——替换 `rcb_runner.py:493-499` 的纯文本 sanity 数值部分。
- **验收判据**：双通道不一致事件在 telemetry 可见且收敛；PaperBench C2ST/MCMC 类数值叶的「算错而不自知」在复核注入后能被 agent 自己发现并修正（以轨迹为证）；RCB 占位数值声明（"Expected …"）100% 被核对器标红。
- **收益**：【中期收益】——PaperBench 数值叶与 RCB 定量项的反事实收益以执行链修复（第一/二梯队）为前置（报告 13 Q3）。

### L3 形式化证明：让「验证通过」携带信息量

- **定义**：物理命题（守恒律、稳定性判据、本构关系性质）编译为 Lean 4 定理，由 kernel 判定真伪；`sorry`/`axiom`/`admit` 一律视为未证。
- **SOTA 参照**：AlphaProof（RL+Lean kernel 奖励，IMO 2024 银牌当量，DeepMind blog 2024-07 / Nature 2025）；DeepSeek-Prover-V2 子目标分解（arXiv:2504.21801，2025-04）；Kimina-Prover 结构化推理（MiniF2F 92.2%）；**以及对 huginn 最要紧的 agentic 路线**：Hilbert 多智能体编排（reasoner+prover+verifier+retriever，MiniF2F 99.2%，arXiv:2509.22819）与 Numina-Lean-Agent（Claude+MCP Lean server 解全部 12 道 Putnam 2025，arXiv:2603.20405）——不训 prover，把 Lean 当 verifier 服务，用通用模型做 draft-sketch-prove。
- **huginn 现状**：工具链就绪（lean4 v4.16.0、HuginnLean 11 模块、auto_pipeline 1020 行）但验证语义为空——`verify_theorem` 只做「编译通过+定理名子串匹配」（`lean/interface.py:158-191`），骨架固定 `:= by sorry`（`lean/sympy_to_lean.py:164`），守恒检查是硬编码恒等重言式（`bourbaki_tool.py:116-132`）。「验证通过」当前不携带信息量（报告 13 F3）。
- **MVI（严格按序，前一步是后一步的前提）**：
  1. **诚信修复（即刻，1-2 天）**：`verify_theorem` 编译后对源文件做 `sorry`/`admit`/`axiom` 扫描，命中即判 FAIL；定理存在性从子串匹配改为 Lean declaration 解析（`audit_20260717/07` P2-3 建议）；`sympy_to_lean.py:164` 的 sorry 骨架产物必须标记 `unproven` 且下游不得当作已证结论引用。**这一步不涨分，但它让「验证通过」第一次不等于「什么都没说」——是全部后续工作的语义地基。**
  2. **重言式替换（周级）**：`_lean_check_conservation`（`bourbaki_tool.py:116-132`）从硬编码恒等实例改为消费用户方程：sympy 推导守恒残差（d/dt(invariant)=0 或散度形式）→ 经 `auto_pipeline` 的 SymPy→Lean Float 编译 → kernel 判定；无 Lean 时 fallback 必须显式返回 `unverified` 而非罐头文本（:186-202 的 `buckingham_pi` 恒返 "[Re, Fr, We]" 之类改为 honest refusal）。
  3. **verifier 服务化（月级，对标 Hilbert/Numina-Lean-Agent）**：接入现成 Lean server（Kimina Lean Server 或 lean-lsp-mcp，均 2025 年开源）作为 verify backend；证明搜索用「通用 reasoner 出 sketch（含 sorry 子目标）→ 子目标递归求解（DSP/Prover-V2 模式）→ server 判定」的编排，**不自训 prover**。mathlib 检索可复用 Hilbert 的 mathlib_informal embedding 方案（arXiv:2509.22819 §3.1）。
- **验收判据**：负样本测试——对已知的错误方程（如量纲不守恒的「守恒律」）必须 FAIL；对 HuginnLean 11 模块中真定理 PASS；注入含 sorry 的证明必 FAIL（防回归）。
- **收益**：【长期投资】（报告 13 已定性：对现有失分类别无直接作用）；其中第 1 步属诚信修复，应即刻做，但理由不是分数而是「不再说谎」。

### L4 猜想生成：FunSearch 式程序进化（只在 L0-L2 落地后立项）

- **定义**：在「评估器完备」的问题上，让 LLM 作为变异算子进化候选程序/公式，产出可解释的新构造。
- **SOTA 参照**：FunSearch（冻结评估器+skeleton 只进化关键逻辑+best-shot k=2+岛屿+签名聚类，cap set 20 年最大改进，Nature 2024）；AlphaEvolve（代码库级 diff 进化，4×4 复数矩阵乘法 48 次标量乘法改进 Strassen、50 个开放问题推进 20%，DeepMind TR 2025-05）；LLM-SR 把方程发现做成程序进化且评估域含材料应力行为（arXiv:2404.18400）；CodeEvolve 给出开源可操作化（arXiv:2510.14150）。
- **huginn 现状**：`autoloop/conjecture.py` 存在但整个 autoloop perceive-only 空转（报告 06 F1）；`ml/transfer_registry.py` 仅被 conjecture 与 deli_research 引用（报告 13 F5）——有器官无循环。
- **MVI（FunSearch-mini，明确限定在 benchmark 之外的「评估器完备」场景）**：
  1. 选 1 个材料问题做试点：从应力-应变/相图数据族中重发现经验公式（对标 LLM-SR 的材料域实验）。评估器冻结 = NMSE + L1 量纲一致性（硬剪枝）+ 复杂度惩罚（对表达式节点数）。
  2. skeleton：固定数据加载与评估逻辑，只进化 `priority`/`score` 式关键函数（FunSearch 的核心工程决策：搜索空间收敛到关键想法）。
  3. programs DB：sqlite 表 `(program, signature, score, island, parent_ids)`；island=4，签名聚类采样，best-shot k=2，prompt 注入 top 程序全文——直接照搬 FunSearch 开源实现的参数骨架（github.com/google-deepmind/funsearch）。
  4. 全部产物（含失败代际）写入自进化经验池（见 ④ Step 1 的 episode 表）——让 L4 同时成为经验池的高质量数据源。
- **验收判据**：在 LLM-SRBench 材料子集（含 Material Science 任务，arXiv:2605.03101 使用过该子集）上 NMSE 不劣于 PySR baseline；或从合成数据重发现一个已知经验定律（如 Hall-Petch）。**达不到即说明 LLM 变异算子质量不足，回退到 L2 层继续攒证据，不在 L4 空耗。**
- **收益**：【长期投资】——这是用户「有数学直觉的 AI 科学家」目标的能力证明形态，但对当前四个 benchmark 分数零直接作用；在 L0-L2 与自进化 Step 1-2 落地前立项即重演「重言式悲剧」。

### 分层小结

| 层 | 一句话 | SOTA 锚点 | huginn 落点 | 投入 | 收益 |
|---|---|---|---|---|---|
| L0 | 数值必须算出来 | FunSearch/Reflexion 的评估器公理 | `bench/runner.py` gate + 四适配器 filter | ~50 行+1 天 | 【直接】 |
| L1 | 量纲/量级必须合理 | AI Feynman、PG-SR 约束检查 | `constraints` scope 扩展+双轨合并 | 天级 | 【直接】 |
| L2 | 两个通道必须一致 | DeepSeekMath-V2、DSP | `verify_tool` 抽取+报告核对器 | 天-周级 | 【中期】 |
| L3 | 证明必须无 sorry | AlphaProof、Prover-V2、Hilbert（编排路线） | `interface.py` 诚信修复→重言式替换→Lean server 编排 | 天级→月级 | 【长期】（第 1 步为诚信修复，即刻） |
| L4 | 评估器完备处进化 | FunSearch、AlphaEvolve、LLM-SR | FunSearch-mini 试点 | 周-月级 | 【长期】 |

---

## ④ 自进化：从空转到 FunSearch/Voyager 式的三步落地

**前置原则（不可协商，来自报告 00 第六节与报告 11 发现 7）**：评测信号修复（`audit_20260717/16` P0-1/P0-2/P1-3/P1-6/P1-7）必须先于 Step 2——否则蒸馏管线蒸馏出的是「读泄漏答案/迎合宽松 judge」的作弊策略，reward hacking 被制度化。Step 1 与 Step 3 的路径修复不依赖此前置，可即刻动工。

### Step 1：经验池结构（experience pool）——统一 schema，硬门准入

**对标**：Voyager 技能库（可执行+验证后入库，arXiv:2305.16291）、ExpeL 经验池（成败轨迹对，AAAI 2024）、EvoScientist 双记忆 M_I/M_E（arXiv:2603.08127）。

**huginn 现状**：三类产物各有 schema 但都不合格——`evolution_rules.json`（25 条模板规则，usage_count 全 0）、`distilled_knowledge.json`（20 条，verification_status 全 "unverified"）、`evolved_skills.json`（1 条空泛模板）；且 18/18 空转轨迹这类「零执行记录」也能产出报告（报告 06 F1/F2）。

**设计**：sqlite 单库 `~/.huginn/experience/experience.db`，三张表：

1. `episode`（每次任务执行一条，**硬门**）：`episode_id, task_id, objective, adapter, model_alias, trajectory_ref, tool_calls_count, score, judge_reasons_json, artifacts_manifest_json, compute_evidence_json, valid`。
   - **准入判据**：`tool_calls_count=0` 且无产物文件 → `valid=false`，不进入蒸馏候选——从 schema 层灭绝「空转轨迹进经验池」（报告 06 F1 的结构性修复）。
   - 触发点：四个外部适配器收分处 + `bench/runner.py:507-512` 的 `log_conversation` 同点扩展。
2. `insight`（蒸馏产物，复用扩展现有 schema）：在 `DistilledKnowledge`（`evolution/knowledge_distiller.py:25-45`）基础上加三字段：`trigger_condition`（何种情境适用，可机判）、`expected_outcome`（应用后应观察到什么，可机判）、`vote`（UPVOTE/DOWNVOTE 累计，ExpeL 算子）。**这两个可机判字段是「可证伪性」的载体——没有它们的 insight 就是 25 条 read_file 规则那种不可证伪的噪声。**
3. `skill`（可执行经验，对标 Voyager）：`skill_id, name, code, self_check_test, language, usage_count, success_count, last_used_at`。入库条件：`self_check_test` 在 sandbox 实跑通过（Voyager 自验证的最小对应物）。

**验收判据**：每次 benchmark 运行必然产生 1 条 `episode`（缺则 CI 告警）；18/18 空转那类轨迹重放后被标 `valid=false`；`insight` 表拒绝 `trigger_condition` 为空的写入。

**收益**：【直接收益】的一部分——episode 硬门同时就是 RCB「虚构 Results」的结构性防线（与报告 06 #1 的 `_report` 数据硬门互补：一个在生成侧拦截，一个在记忆侧拒收）。

### Step 2：LLM 蒸馏管线（distillation pipeline）——ExpeL 算子 + ESE 细节强制 + IVE 负知识

**对标**：ExpeL 的 insight 抽取（对成败轨迹对执行 ADD/UPVOTE/DOWNVOTE/EDIT，arXiv:2308.10144）；EvoScientist EMA 的 ESE（强制保留参数/库函数名细节，"另一个工程师能据此重现"）与 IVE（失败方向→含 3-6 条避坑建议的负知识）（arXiv:2603.08127，报告 11）；Reflexion 的 Self-Reflection 角色（把稀疏奖励转成语言反馈，NeurIPS 2023）。

**huginn 现状**：规则生成是手写 if-else 模板（`evolution/engine.py:502-547`），蒸馏器是字符串拼接（`knowledge_distiller.py:110-195`），全程无 LLM、无任务语义、无分数输入（报告 11 发现 2）。

**设计**：

1. **触发**：(a) 每个 `episode` 落库且 `score` 到手即触发单轨迹蒸馏（Reflexion 式：judge 评语+轨迹→1-3 条候选 insight）；(b) 每累积 5 条同 `task_id` 族 episode 触发跨任务蒸馏（ExpeL 式：成败轨迹对→算子操作现有 insight 列表）。
2. **管线**（单次 LLM 调用，结构化输出 JSON）：
   - 输入：轨迹摘要（工具调用序列+错误+最终产物清单）、`score` 与 `judge_reasons`、现有 insight 列表（按 vote 排序，top 30）；
   - 输出：算子序列 `[{op: ADD|UPVOTE|DOWNVOTE|EDIT, target_id?, content, trigger_condition, expected_outcome, source_evidence: [episode_ids]}]`；
   - **ESE 细节检查**（确定性后处理）：ADD/EDIT 的 `content` 若不含任何具体参数名/库函数名/错误类别串 → 拒收并打回一次（对标 "另一个工程师能据此重现" 的机判近似）；
   - **IVE 负知识通道**：`score` 低于任务族中位数的 episode 必须额外产出 `avoidance` 条目（`trigger_condition` + 3-6 条避坑建议），写入 insight 表并标记 `polarity=negative`；
   - `source_evidence` 为空 → 拒收（可溯源是硬要求，对标 FunSearch 程序可解释性精神）。
3. **落点**：`evolution/engine.py:_generate_heuristic_fix`（:502-547）与 `knowledge_distiller._generate_error_lesson/_find_common_params` 的模板路径整体替换为上述 LLM 管线；保留 `knowledge_distiller.py:87-108` 的 Jaccard 语义去重作为入库前第一道（<500 条目够用，其 ponytail 注释亦如此预期）；`_error_matches`（`evolution/engine.py:571-574`）从路径级双向子串改为「异常类别+工具名」匹配（`FileNotFoundError` 泛化到任意路径——这是让「同一个教训不被学 25 次」的匹配侧修复，报告 06 F3）。
4. **考核**（报告 06 #5 的机制化）：`usage_count=0` 超 30 天自动 DOWNVOTE 并归档；`vote<0` 的 insight 不进入注入候选。现有 25 条零使用规则+20 条未验证蒸馏知识归档清零，作为管线首日输入重蒸。

**验收判据**：25 条同构 read_file 规则经重蒸后收敛为 1-2 条类别级 insight（FileNotFoundError 泛化）；新 insight 的 `trigger_condition`/`expected_outcome` 非空率 100%；蒸馏后 30 天内 `usage_count>0` 的 insight 占比 > 30%（低于此值说明蒸馏质量或检索质量不达标，停下来修而不是继续堆量）。

**收益**：【中期收益】——跨任务复利的起点（EvoScientist 的跑分优势核心，报告 11 发现 4）；对单跑分数无即刻作用。

### Step 3：回灌 prompt 的路径（re-injection）——先通、再准、后归因

**对标**：Voyager embedding 检索技能注入（arXiv:2305.16291）、ExpeL 推理时召回 insight+最相似成功轨迹（AAAI 2024）、EvoScientist mxbai-embed-large top-k 注入（报告 11）。

**huginn 现状**：写入 `~/.huginn/logs/evolution_rules.json`（`evolution/logger.py:56`）与读取 `$HUGINN_CACHE_DIR/evolution_rules.json`（`context_builder.py:403-404`）永不相交（报告 11 发现 3，本轮双路径复核确认）；注入后无归因字段，usage_count 恒 0。

**设计（三段，按序）**：

1. **通**（一行级，报告 11 #1）：统一写读路径。建议两侧都走 `~/.huginn/logs/`（写入侧已是事实标准，读侧 `context_builder.py:403-404` 改两行）。改完既有 25 条规则即刻可见——预期价值有限（内容是噪声），但它让「写读不相交」这个一类 bug 先归零。
2. **准**（天级）：`build_evolution_rules`（`context_builder.py:390-419`）从「全量注入」改为 top-k 检索注入——按当前 task 的 tags/工具面与 insight 的 `trigger_condition` 做匹配（类别级精确匹配优先），加 embedding 相似度兜底，取 3-5 条 insight（`polarity` 正负都取，负知识以 "Avoid:" 前缀注入）+ 1 条最相似 `valid` episode 的摘要（ExpeL 的 trajectory recollection 对应物）。注入位置：system prompt 尾部独立 `## Learned lessons` 段，与 KB 注入（`context_builder.py:157-218` 的 top-5 向量检索通道，报告 06 F4）并列但分开标注来源。
3. **归因**（天级，Reflexion 闭环的最小对应物）：每次注入把 `injected_insight_ids` 写入 trajectory；本轮执行结束后按 `trigger_condition` 是否命中+`expected_outcome` 是否达成自动回填——命中且成功 → 对应 insight `usage_count++` 且 `vote+1`；命中但失败 → `vote-1`；未命中不计。这把「软提示是否压倒路径猜测习惯」（报告 06 F4 的未决推断）变成可观测数据。

**验收判据**：虚拟路径错误（`read_file("/paper/paper.pdf")` 式，报告 06 F4 记载复发 25+ 次）在负知识注入后的后续运行中 0 复发；trajectory 100% 含 `injected_insight_ids`；每条 insight 的 `usage_count` 与真实注入记录对账一致。

**收益**：【直接收益】（路径修复让既有规则可见，边际但非零；更重要的是它把「人工 prompt 修补」这条当前唯一有效的进化通道（报告 06 机制 5）替换为自动回路，且避免 spaceship-titanic 0.6833→0.638 那种人工教训回退——因为每条 insight 都有 `expected_outcome` 归因）。

---

## ⑤ 收益分类汇总：哪些直接涨分，哪些是长期投资

| 项目 | 落点 | 对当前 benchmark 分数 | 性质 |
|---|---|---|---|
| L0 计算纪律 gate | `bench/runner.py` + 四适配器 filter | **直接**：repro 2 题翻转；全 benchmark 反裸答 | 直接收益 |
| L1 量纲/量级 block 级检查 | `constraints` scope 扩展 | **直接**（辅助）：repro 类防错 + RCB 定量声明防线 | 直接收益 |
| Step 1 episode 硬门 | 适配器收分处 + `experience.db` | **直接**（间接路径）：与 `_report` 数据硬门互补，防虚构 Results（RCB judge 最重扣分项的结构性防线） | 直接收益 |
| Step 3 路径修复 + top-k 注入 | `context_builder.py:403-404` 等 | **直接**（边际）：既有规则可见；虚拟路径类复发错误归零 | 直接收益 |
| L2 数值-符号互验 | `verify_tool` + 报告核对器 | 执行链修复后兑现：PaperBench 数值叶、RCB 占位数值 | 中期收益 |
| Step 2 LLM 蒸馏管线 | `evolution/engine.py:502-547` 替换 | 跨任务复利，N 次运行后单调改善；**前置=评测信号修复** | 中期收益 |
| L3-1 sorry 扫描 | `lean/interface.py:158-191` | 不涨分；让「验证通过」开始携带信息量 | 诚信修复（即刻） |
| L3-2/3-3 重言式替换 + Lean server 编排 | `bourbaki_tool.py:116-132`、外部 server | 当前失分类别无可归因作用面（报告 13 Q3） | 长期投资 |
| L4 FunSearch-mini | 新模块（benchmark 外） | 零直接作用；用户核心目标的能力证明 | 长期投资 |
| EvoScientist 完整对标（idea 树+Elo、EMA 全量） | 后续立项 | 需要 Step 1-2 与可信评测信号全部就绪 | 长期投资 |

与三梯队修复共识（报告 00 第六节）的对应：L0/L1、Step 3-1 属第一梯队（小时-天级立见分）；L2、Step 1/3-2/3-3 属第二梯队（天级，修复执行链）；Step 2、L3、L4 属第三梯队（周-月级，接通核心目标），其中 Step 2 与 L3-3/L4 共享「评测信号可信」这一前置。

## ⑥ 风险与前置条件（按阻塞性排序）

1. **评测信号有毒 → Step 2 绝对前置**（`audit_20260717/16` P0-1/P0-2/P1-3/P1-6/P1-7）：rubric 与标签在 agent 可达范围内、judge 与被测同源 deepseek-chat、判分器保底分。不修复则蒸馏管线产出作弊策略，且制度化后比人工修补的回退（spaceship-titanic 0.6833→0.638）更难察觉。
2. **模型静默降级**（报告 00 三.2：`from_env` 忽略 toml 里的 deepseek-reasoner）：L2 双通道复核、Step 2 蒸馏、L3 sketch 生成都需要推理档模型；deepseek-chat（非推理、temp 0.7）会让蒸馏质量与互验质量同时失真。
3. **PhaseGateState 单例串扰**（报告 06 F6）：本批未起效，但 Step 1 让循环真正跑起来后，并发 run 互相 `reset_runtime()` 会使门控放行不可信——在任何「依赖门控保证质量」的宣称之前必须 per-run 化。
4. **判据错误先修**（`audit_20260717/14` P2-7/P2-3、physics_auditor 压力单位 bug）：L1 升级为 block 级后，错误判据的杀伤力同步放大。
5. **蒸馏噪声管控**：ESE 细节检查+可证伪字段+30 天零使用淘汰，三者缺一，经验库会在数月内膨胀成第二个「25 条 read_file 规则」噪声库。
6. **L4 的纪律**：达不到验收判据（不劣于 PySR）即回退，不在 L4 空耗——FunSearch 的前提是评估器完备，huginn 目前只在合成/重发现场景满足，不应对真实开放问题直接开 L4。

---

*报告完。分析人：神经符号/自进化系统研究员（SOTA 调研与接入设计专项）。*
