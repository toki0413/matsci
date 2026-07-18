# 11 · 对标 EvoScientist：huginn 距离「能自我进化、有数学直觉的 AI 科学家」差在哪

- **诊断对象**：huginn-agent（`C:\Users\wanzh\Desktop\matsci-agent`）
- **角色**：AI 科学家系统研究员
- **日期**：2026-07-17
- **方法**：严格只读。外部调研（EvoScientist 论文 arXiv:2603.08127 及 AI Scientist 系工作）+ 本地代码/评分/轨迹证据核对。引用格式 `相对路径:行号`；推断均已标注。

---

## ① 数据与方法

**外部调研**（kimi_search_v2 + 论文页面，2026-07-17 检索）：

- EvoScientist（arXiv:2603.08127，华为，2026-03）：三智能体（RA 研究 / EA 工程 / EMA 进化管理）+ 两块持久记忆（ideation memory M_I / experimentation memory M_E）。核心机制：
  1. **RA 的 idea tree search**（propose–review–refine 树）+ **Elo 锦标赛排序**，top-3 想法作为方向证据；
  2. **EA 的四阶段实验树搜索**（初始实现→调参→方法实现→消融），每阶段「生成代码→执行→检查→失败修正」循环，按结构化执行记录选 best code；**预算内找不到可执行代码即判提案失败（rule-based 硬判据）**；
  3. **EMA 的三种进化写入**：IDE（从 top 想法蒸馏可行方向）、IVE（失败方向变负知识，含 3-6 条避坑建议）、ESE（从完整代码搜索轨迹蒸馏可复用执行策略，强制保留参数/库函数名细节，要求"另一个工程师能据此重现"）；
  4. **记忆检索闭环**：两块记忆用 embedding（mxbai-embed-large）余弦 top-k 检索，注入下一次任务的 RA/EA。
- 旁证：Sakana AI Scientist（生成-评审闭环）、AgentRxiv（自我引用改进）、EvoGens/PiEvo（想法种群的突变/交叉算子）——共同点是**进化作用于"结构化的历史产物"，且有显式的适应度信号**。

**本地证据**：

- 代码：`agent/huginn/evolution/`（engine.py 587 行）、`self_improvement/core.py`（2688 行）、`metacog/`（9 模块）、`cognitive_engine.py`（548 行）、`cognitive_primitives.py`、`bourbaki_env.py`（219 行）、`autoloop/engine.py`（7185 行）。
- 持久化状态（用户目录，只读核对）：`~/.huginn/logs/evolution_rules.json`（25 条）、`evolved_skills.json`（1 条）、`evolution_history.json`（4 个 cycle）。
- 运行痕迹：`agent/.huginn/trajectories/loop_*.json`（18 条）、`agent/huginn_autoloop_report_loop_*.md`（16 份）。
- 成绩：`workspaces/mlebench/*/_score.json`、`workspaces/sab/task_1/_score.json`、`ResearchClawBench/score_results.json`。
- 已确认审计事实：`audit_20260717/04`（P1-6 双自改进系统并行等）、`audit_20260717/05`（降级链空转）、`audit_20260717/07`（执行层失效点）、`audit_20260717/16`（评测泄漏与判分失真）。

---

## ② 核心发现（按证据强度排序）

### 发现 1：autoloop「自进化主循环」在全部 18 条轨迹中从未跑过实验 —— 只有 perceive + report（证据强度：决定性）

18 条轨迹的 span 摘要全部为 `phase:perceive ×5 + phase:report ×1`，无一条出现 hypothesize/plan/execute/validate/learn。例：`agent/.huginn/trajectories/loop_d0906c41.json` 的 summary：`{"phase:perceive": {"count": 5, "duration_ms": 56.33}, "phase:report": {"count": 1, "duration_ms": 5837.53}}`——整个"自主研究循环"总时长约 6 秒。对应报告 `agent/huginn_autoloop_report_loop_16c3481f.md` 自述 "Total Time: 2.6s"，阶段表 5 行全是 perceive；`loop_2f1d2a47` 报告仅 1.7 KB（近似空报告）。

机制定位：`autoloop/engine.py:1353-1362` —— perceive 返回 falsy 时 `continue` 跳过本轮全部后续阶段；而 `_perceive_legacy`（engine.py:2347-2390）以 `git status --short` + 最近 1 小时 `.log` 错误扫描为触发源，workspace 干净时返回 `None`。即 autoloop 是**文件变更驱动的事件循环**，不是任务驱动：给它一个研究 objective、一个干净 workspace，它就空转 5 轮然后写报告。

**这是「有进化骨架、无进化运行」的第一证据：自我改进的载体进程本身没有启动过。**

### 发现 2：EvolutionEngine 实际产出的"知识"是噪声级模板规则，且使用计数全为 0（证据强度：决定性，持久化文件实测）

`~/.huginn/logs/evolution_rules.json` 实测 25 条规则，**全部**是同一形态：

```
trigger: "read_file|Error: File '/related_work/paper_000.pdf' not found"
action:  {"files": "check_existence", "paths": "verify"}
usage_count: 0（25/25 全为 0）
```

`evolved_skills.json` 仅 1 条 "Unknown Workflow (general)"，usage_count=0。`evolution_history.json` 4 个 cycle（2026-06-26 ×3、2026-07-13 ×1）的 new_failure_rules/new_prompt_patches/new_reward_* 几乎全 0。

机制定位：`evolution/engine.py:177-209`（evolve_from_failures）只对 tool 错误字符串做模板匹配（`_generate_heuristic_fix` :502-547 是手写的 VASP/Gaussian/LAMMPS 关键词 if-else）；**全程无 LLM 参与蒸馏、无任务语义、无 benchmark 分数输入**。对比 EvoScientist 的 EMA：IVE/ESE 都是 LLM 从完整轨迹蒸馏结构化知识。huginn 的"进化写入"缺失了最关键的生效件——**一个能从轨迹中提炼语义的 LLM 蒸馏器**。`evolve_from_rewards`（engine.py:288-378）虽有连续奖励通道设计，实测产出 0 条。

### 发现 3：进化成果注入 prompt 的「最后一公里」路径断裂 —— 写入与读取指向两个永不相交的文件（证据强度：决定性，双路径实测核对）

- **写入方**：`EvolutionEngine` 默认经 `ExecutionLogger()` 持久化到 `Path.home()/".huginn"/"logs"/evolution_rules.json`（`evolution/logger.py:54-58`）——实测存在，11.7 KB。
- **读取方**：`context_builder.build_evolution_rules()`（`agent/huginn/context_builder.py:393-410`）读 `Path(os.environ.get("HUGINN_CACHE_DIR", ".huginn")) / "evolution_rules.json"`。
- 核对三种配置：默认（cwd 下 `.huginn/evolution_rules.json`——不存在，实测）；`rcb_runner.py:35` 设 `HUGINN_CACHE_DIR=~/.huginn`（则读 `~/.huginn/evolution_rules.json`——不存在，实测）；任何配置下读取路径都不含 `logs/` 子目录。

后果：chat 主链路的"tool fails → rule learned → next call benefits"闭环（context_builder.py:393 的 docstring 自述）**从未闭合过**。唯一真实读到规则的是 autoloop（engine.py:6094-6102 直接持有 engine 对象调 `get_relevant_skills`/`get_prompt_patches`）——但 autoloop 本身空转（发现 1），且规则内容是无意义的 read_file 模板（发现 2）。

### 发现 4：四个外部 benchmark 适配器完全不经过进化/反思/元认知基础设施 —— 每次运行都是无记忆的一锤子买卖（证据强度：决定性，grep 全量核对）

对 `mlebench_huginn.py`、`sab_huginn.py`、`rcb_huginn.py`、`paperbench_huginn.py` grep `evolution|reflect|get_prompt_patches|apply_heuristic`：**零命中**。benchmark 失败（MLE no medal、SAB 25 分、RCB 10.5 分）之后，没有任何"judge 评语 → 失败分析 → 策略更新 → 重试"的回路。EvoScientist 的跑分优势恰恰来自这个回路（IVE 把失败方向写回记忆抑制重复；ESE 把成功执行策略迁移到下一任务）。huginn 的 18 条轨迹也佐证：没有任何一条轨迹的 objective 是 benchmark 任务——benchmark 运行与进化系统是**物理隔离的两个世界**。

### 发现 5：「数学直觉」层的三个生效件全部缺位（证据强度：高）

- `cognitive_primitives.py` 全文件 1 行：`# deprecated: adversarial_critique 直接塞 rcb_runner.py 了`（audit 04 P3-1 确认全仓 0 引用）。
- `bourbaki_env.py` 219 行是 **Lean 4 安装器**（elan 下载/lake build 封装），不含任何数学推理；真正的定理验证在 `lean/interface.py:182-189` 只做**定理名子串匹配**——`by sorry` 占位证明可通过验证，全库无 `sorry`/`axiom` 扫描（audit 07 P2-3 确认）。
- `cognitive_engine.py:12-17` docstring 自述："This is NOT a new attention mechanism — it's a cognitive layer that shapes what the LLM pays attention to **via prompt engineering**"。S0-S7 状态机只是 prompt 策略切换，且 CSM 迁移失败静默吞掉（`agent/core.py:440-445`，audit 04 P1-6）。
- 直接后果可由 repro 基准证伪：χ=1.0 vs 文献 0.004、θD=6.6K vs 480K 这类**数量级级物理错误**（2/10 失败项）产出路径上没有任何量纲/数量级守卫拦截——`metacog/` 的 completion_auditor、equivalence_auditor 等审计器全部以 lazy import 挂在 autoloop 内部（`autoloop/engine.py:2469-2499`），benchmark 与 chat 执行路径不经过它们（对 agent/ 主链路 grep metacog，仅 reflection.py 的 SignalHub 一处）。

### 发现 6：执行缺口的直接失分机制 —— RCB「只描述不执行」与 SAB「代码截断」都源于 harness 没有"可执行性硬门"（证据强度：高，judge 原文）

- SAB task_1（25/100，`workspaces/sab/task_1/_score.json`）：judge 原文 "The code is incomplete (truncated mid-function), missing the main execution block... It does not save predictions to the required output file"，breakdown output=0/20。适配器侧证据：`sab_huginn.py:51` ponytail 注释自承 "不含 file_write_tool/file_edit_tool — 它们在 Windows 上路径解析有 bug"，:156-158 改让 agent 用 `code_tool` 里 `open()` 写整个程序。**单次 LLM 输出写 300+ 行程序，撞上输出上限即截断，harness 无分块写入工具、无写后 `py_compile`/运行校验门**。对照 EvoScientist EA：「预算内无可执行代码 = 失败」是 rule-based 硬判据，执行记录结构化后驱动下一轮修正——huginn 截断的代码被原样送交 judge，没有任何环节发现"它根本跑不了"。
- RCB（`ResearchClawBench/score_results.json`，Material_000=10.5 / Material_001=15.75 / Material_002=17.28 / Material_003=11.25；注：与任务书快照 5.25 不同，系多次重跑中的一次——audit 16 P1-12 确认同题重跑 2.25–46.5 大幅波动）：judge 原文 "the report does not provide any actual model performance metrics (AUROC, precision, recall) from running the pipeline — **it only describes expected or placeholder results**"。agent 生成了"像论文的报告"但没有真正训练模型。机制同上：无执行强制门 + RCB deliverable 判据路径不一致导致完成信号失真（audit 16 P2-1）+ 超时体系对同步工具失效（audit 07 P1-1，长训练调用会冻结事件循环，agent 倾向"描述"而非"执行"是理性避坑——推断）。

### 发现 7：评测信号本身有毒 —— 以此做进化信号会进化出应试作弊（证据强度：高，audit 16 已确认事实）

`audit_20260717/16` 确认：PaperBench 把 rubric 与具体叶节点答案写进 agent prompt（P0-1，`paperbench_huginn.py:296-304`）；MLE-bench synthetic 私有标签就在 workspace `_private/test.csv`（P0-2）；全部 judge 与被测模型同源 deepseek-chat（P1-3）；PaperBench judge "不确定给 30、看到 loss 曲线给 50+"（P1-6）；自建判分器"任意数字命中即过、答错给 0.3 保底"（P1-7）。这意味着两层损失：(a) 现有分数不是有效测量；(b) 更致命——**若把这些分数接入进化回路（发现 4 的修复方向），进化压力会优选"读泄漏答案/迎合宽松 judge"的策略**，即 reward hacking 的制度化。EvoScientist 的进化能生效，前提是它的失败判据（可执行性、与 baseline 比较）是难以投机的。

---

## ③ 根因链（现象 → 机制 → 代码位置）

**链 A：MLE-bench 三任务全 no medal**
现象：单次提交分数 0.758/0.638/0.745，medal=none（`workspaces/mlebench/*/_score.json`）。
机制：单次尝试，无迭代改进回路；历届 Kaggle 方案的经验无处沉淀；下一次运行从零开始。
代码：`mlebench_huginn.py` 无任何 evolution/reflect 引用（grep 零命中）；奖牌逻辑本身颠倒但因 leaderboard.csv 是 LFS 指针未爆（audit 16 P1-10）。

**链 B：SAB 25/100**
现象：代码函数中间截断、缺主执行块、未保存预测文件（`_score.json` judge 原文）。
机制：长产物单点写入 + 无可执行性校验门。
代码：`sab_huginn.py:51`（file_write_tool 因 Windows bug 被移除）→ :156-158（改用 code_tool open() 一次性写全文）→ 无续写/无 py_compile 门；判分侧 `sab_huginn.py:311` 还把截断代码再截到 8000 字符送审。

**链 C：RCB 10.5/15.75/17.28/11.25**
现象：judge 高频批语 "only describes expected or placeholder results"（score_results.json Material_000 item 0 reasoning 原文）。
机制：agent 选择"写报告"而非"跑实验"无人纠正；完成判据永不触发导致烧满预算。
代码：交付判据路径不一致（audit 16 P2-1，`bench/orchestrator.py:67-71` vs `rcb_huginn.py:122-125`）；重型执行路径故障（audit 07 P1-1/P1-3）；无"报告数值必须来自本次运行产物"的校验。

**链 D：repro 基准 2/10 数量级级物理错误**
现象：χ=1.0 vs 0.004；θD=6.6K vs 480K。
机制：物理 sanity 检查（metacog 审计器族）只接在空转的 autoloop 内，主执行路径无守卫；数值判分器又宽松到"任意数字命中即过"。
代码：`autoloop/engine.py:2469-2499`（lazy import，autoloop-only）；`bench/runner.py:23-33`（audit 16 P1-7）。

**链 E：自我进化名存实亡（总根因）**
现象：16 份 autoloop 报告内容近似（5×perceive）、25 条进化规则 usage_count=0、4 个 evolution cycle 全 0 产出。
机制：进化循环三段全部断线——**载体**（autoloop 事件驱动空转，engine.py:1353-1362 + :2347-2390）、**写入**（模板匹配代替 LLM 蒸馏，evolution/engine.py:502-547）、**读取**（路径 bug，context_builder.py:393-410 vs evolution/logger.py:54-58）。另有两套自改进系统并行稀释（audit 04 P1-6：EvolutionEngine vs SelfImprovementLoop，`self_improvement/core.py:2654`，其 evaluator 仅为 keyword/numeric 匹配 :23-55）。

---

## ④ 对用户问题的回答

**Q：EvoScientist 类系统的跑分优势来自哪几个机制（按优先级）？**

1. **失败的制度化利用（IVE/ESE）**：失败方向变负知识抑制重复、成功轨迹蒸馏为可迁移执行策略——这是"跨任务复利"，跑分随任务数单调改善；
2. **可执行性硬判据 + 分阶段 best-code 选择**：产出物必须真的能跑，执行记录结构化驱动修正——直接消除 SAB/RCB 式失分；
3. **idea 质量控制（树搜索 + Elo 锦标赛）**：减少把预算烧在不可行方向上；
4. **结构化蒸馏 prompt 保证记忆质量**（"另一个工程师能重现"级别的细节）；
5. **embedding 检索注入下一任务**——闭环的最后一公里。

**Q：huginn 有这些机制的骨架但缺什么「生效件」？**

| 骨架（存在） | 缺的生效件 | 证据 |
|---|---|---|
| `evolution/engine.py` 规则库+技能库（≈M_I/M_E） | LLM 轨迹蒸馏器；规则被实际调用（usage_count=0）；chat 路径注入（路径 bug） | 发现 2、3 |
| `autoloop/engine.py` perceive→learn 七阶段（≈RA+EA pipeline） | 任务驱动的启动条件：objective 存在时首轮无条件进 hypothesize | 发现 1（engine.py:1353-1362） |
| `metacog/` 9 个审计器（≈评审闭环） | 接入主执行路径（现为 autoloop-only lazy import）；`lean/interface.py:182-189` 的真验证（ sorry 拦截） | 发现 5 |
| `bench/` + 5 个外部适配器（≈适应度信号） | 无泄漏、无保底分的可信判分；judge 与被测异源 | 发现 7（audit 16） |
| `self_improvement/core.py` SelfImprovementLoop | 与 EvolutionEngine 合并或分工；evaluator 接真实执行结果而非关键词 | audit 04 P1-6 |

**Q：哪些差距解释了跑分差距？**

- 直接解释 SAB 25 分：无"写后必验证可运行"硬门（链 B）；
- 直接解释 RCB「只描述不执行」：无执行强制 + 完成判据失灵（链 C）；
- 直接解释 repro 数量级错误：物理守卫不在执行路径（链 D）；
- 解释"跑了这么多 benchmark 却不进步"：benchmark 与进化系统物理隔离 + 进化三段断线（链 A/E）。

---

## ⑤ 可操作建议（按投入产出比排序）

1. **修进化规则读取路径 bug**（约 1 行级）：`context_builder.py:400` 的规则路径与 `evolution/logger.py:56` 对齐（统一走 `~/.huginn/logs/` 或统一走 `$HUGINN_CACHE_DIR`）。立刻让既有 25 条规则可见——但要预期其价值有限（发现 2）。
2. **修 autoloop 空转**（小改）：`engine.py:1353` 的 `if not phase.result: continue` 增加例外——`self._iteration == 1 and self._objective` 时强制进入 hypothesize。让 16 份报告之后第 17 次运行第一次真正跑完七阶段。
3. **给 SAB/RCB/PaperBench 类适配器加"可执行性硬门"**（中改，直接涨分）：代码产物必须过 `py_compile` + 冒烟执行 + 预期输出文件存在且非空，才允许结束；报告类任务要求关键数值能溯源到本次运行的产物文件。对应 EvoScientist 的 rule-based 失败判据，是投入产出比最高的一项。
4. **建最小 EMA 回路**（中改）：每次 benchmark 结束后，把 judge 评语 + 轨迹交给 LLM 蒸馏 1-3 条策略写入 evolution_rules（替换现有模板匹配），下次运行注入。**前置条件是先做第 6 条**，否则蒸馏出的是作弊策略。
5. **长产物分块写入 + file_write_tool Windows 路径修复**（中改）：消除 `sab_huginn.py:51` ponytail 妥协，代码生成按模块分次写入再拼接。
6. **评测面先行修复**（前置项，audit 16 P0-1/P0-2/P1-3/P1-6）：rubric/标签移出 agent 可达范围、judge 换异源强模型、删除保底分。分数可信之前，一切"以分数为进化信号"的工作都是负资产。
7. **metacog 审计器从 autoloop-only 改为工具后置钩子**（中大改）：数值型工具结果过数量级/量纲 sanity 表（χ、θD 类物理量各给先验区间），拦截链 D 类错误。
8. **合并双自改进系统**（架构项，audit 04 P1-6）：EvolutionEngine 管"策略知识"（对标 M_I/M_E），SelfImprovementLoop 管"评测执行"，明确单一真源，停止平行生长。

---

*报告完。诊断人：AI 科学家系统研究员（对标 EvoScientist 专项）。*
