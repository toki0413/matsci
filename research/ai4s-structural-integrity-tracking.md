# AI4S 结构完整性跟踪报告：从「能跑」到「可信」

日期：2026-10-31
范围：Huginn 科研智能体（`/workspace/agent`）的结构性缺陷台账、四轨修复、跟踪指标与未闭合风险
性质：跟踪基线文档。所有仓库内数字均来自当前工作树快照；外部结论均标注来源与确定性等级。

---

## 摘要

AI4S 的终极目标不是「生成看起来像论文的产物」，而是让机器在**假设—实验—验证—迭代**闭环中产出**可验证、可复现、可审计、可归因**的科学结论。外部证据显示，当前自主科研系统最普遍的失效恰好不在「生成」，而在「可信」：MLReplicate 在 6 个前沿系统、45 篇生成稿件上发现，人工评审在所有系统中都识别出方法缺陷、实验结果幻觉与复现失败，且 59% 被自动评审接收的稿件包含编造或无支撑的论断 [cite:5]。

本仓库把这类失效拆成 **11 个结构缺陷类**（A–L），每类挂可执行探针，并以四轨修复（契约进数据流、默认拒绝、文档断言、执行覆盖度量）把「可验证性」变成机器判据。当前状态：四轨代码与守护测试已落地，治理台账与打包镜像已收敛；剩余的关键缺口是**真值验证**（ws35 真跑与冻结真值审计，受凭证 401 阻塞）与**全量回归收口**。本报告给出三层跟踪机制（门 / 覆盖 / 真值）、节奏、升级规则与当前登记表。

---

## 1. AI4S 终极目标的界定

### 1.1 文献口径

「AI for Science 是科学研究的第五范式」这一判断可追溯到微软剑桥研究院 Chris Bishop 等人的表述，中国科学院院刊将其概括为「利用 AI 和机器猜想进行科学发现的新方法」，与前四种范式不同，它不只依赖已有数据与方程，还能通过机器学习推断未知规律 [cite:3]。Agent4S 进一步指出，当前 AI4S 主要充当分析工具，并未解决研究流程的核心低效；真正的第五范式是 LLM 驱动的智能体自动化**整个研究流程**，并给出 L1–L5 的五级路线，从简单任务自动化到完全自主、协作的「AI 科学家」[cite:1]。另一份综述把自主科学发现建模为「五项核心能力 + 四阶段动态工作流」，并把智能体科学定位为 AI for Science 的一个演进阶段：从部分辅助走向完整的科学能动性 [cite:2]。

需要注意的是能力分布的现状判断：一份 2026 年的综述性预印本指出，多数当前研究最多达到 L3 自动化，只有少数达到 L4（人类参与关键决策的专家级服务）[cite:4]。该结论来自预印本，确定性等级为「多数意见」，不是同行评审共识。

### 1.2 本报告的操作化定义

把上述叙事压缩成可判定的四条轴，供工程度量使用（此为工程定义，不是文献原话）：

| 轴 | 判定问题 | 对应失败模式 |
|---|---|---|
| 自主性 | 闭环能否在无人工兜底时推进到验收 | 自评收工、空转、任务漂移 |
| 可验证性 | 完成与否是否由外部确定性判据裁决 | 散文契约、advisory 门、存在性假绿 |
| 可复现性 | 产物能否由脚本重跑并逐字节/逐数值复现 | 手写结果、换数据、丢种子 |
| 可审计性 | 失败与降级是否显式可见并可追溯 | 宽捕获吞异常、供应商失败不可见 |

这四轴是后文缺陷分类与跟踪指标的判据来源。

---

## 2. 外部证据：自主科研系统的共性失效

### 2.1 复现性缺口

MLReplicate 用 ICML 2025 的杰出论文重构标准化输入，评测 6 个自主研究系统，得到 45 篇生成稿件（3 次实验失败）。其中 37 篇有效投稿里自动评审接收 10 篇，另有 8 篇因页数不足被 desk-reject；但人工评审在所有系统中都识别出方法缺陷、幻觉实验结果与复现失败，且 59% 的自动接收稿件含编造或无支撑论断 [cite:5]。同一研究还发现，token 预算与算力成本都不能预测产出质量：最便宜的系统在人工评估中优于最耗资源的系统，输入 token 相差 38 倍 [cite:5]。

这对工程的直接含义是：**「更多算力」不是可信度的解法，工作流结构才是**。本仓库把验收权从模型自评移到契约门，正是对这一结论的工程回应。

### 2.2 评测基准的演进

AstaBench 提供 2400+ 问题，覆盖完整科学发现流程与多学科，并配套可控制、可复现的搜索环境，用于整体度量智能体的科研能力 [cite:6]。TruthInsightBench 则用 40 个来自 40 篇同行评审研究的盲任务，覆盖 10 个学科，只给中性目标与冻结数据，由固定 LLM 评判智能体自述论断的「证据成熟度」[cite:7]。这两个基准的共同转向是：**从「答案对不对」转向「论断是否有证据支撑」**。

### 2.3 验证瓶颈的规模证据

DeepMind 的 GNoME 预测了约 220 万种新材料 [cite:8]；二级来源称其中约 38.1 万被列为高概率可合成，而截至论文发表时外部实验室实际合成 736 种 [cite:11]（后者为 P2 来源，标记 [部分未核验]）。这个漏斗说明：当生成能力提升两个数量级后，**瓶颈转移到验证与复现**。自驱动实验室的闭环（机器人 + 高通量实验 + 多模态测量 + AI 决策）正是为解这个瓶颈而生 [cite:9][cite:10]。

---

## 3. 缺陷类与 AI4S 能力的映射

仓库缺陷台账（`agent/tests/loops/test_structural_guards.py` 的 `_DEFECT_LEDGER`）按类记账而非按实例记账，规则是同 class 第 2 次复发起禁止实例补丁，必须先有结构性修复与可执行探针。当前 11 类的映射如下：

| 类 | 症状（台账原文摘要） | 复发次数 | 被侵蚀的轴 | 外部对应 |
|---|---|---|---|---|
| A-spec-undecidable | 交付物/键名/行 schema 只是散文，引擎无法判定 | 4 | 可验证性 | 契约缺失导致评审口径漂移 [cite:5] |
| B-acceptance-authority-misplaced | 完成与否由同一模型自评，说 stop 就收工 | 2 | 自主性 | 自动评审误接收 [cite:5] |
| C-gate-strength-asymmetry | 门写着 advisory 实际不阻断；progress 只数动作 | 2 | 可验证性 | 形式合规但实质未验证 |
| D-probe-fidelity-gap | 测试桩自带内核没有的属性 → 桩绿真跑走死分支 | 2 | 可审计性 | 基准与真实环境脱节 [cite:6] |
| E-failure-unobservable | 超时静默重试、挂了只看到退出码 | 2 | 可审计性 | 失败不可归因 |
| F-existence-gate-fake-green | 文件在 ≠ 结果真；空壳 JSON 判绿 | 3 | 可验证性 | 存在性检查替代正确性检查 |
| G-spec-truncated-out-of-prompt | 契约被 prompt 预算截断 → 模型自造假设 | 2 | 可验证性 | 长上下文丢约束 |
| H-idle-bypasses-acceptance | 空转策略与门信号脱钩，验收红仍 skip 到耗尽 | 2 | 自主性 | 轮次预算被无效消耗 |
| I-objective-hijacked | 已写死协议被 hypothesize/pivot 覆盖，轮次烧在换假设 | 2 | 自主性 | 任务漂移 |
| K-artifact-fabricated-not-reproducible | 文件键名齐全但数值手写，或违反技术条款 | 3 | 可复现性 | 幻觉实验结果 [cite:5] |
| L-provider-hard-failure-masked | 400/401 被宽捕获吞成空产出，循环空转 | 3 | 可审计性 | 外部依赖故障静默降级 |

其中 A、F、K 三类直接对应外部证据里最贵的失效：**看起来完成了，其实没有**。L 类与本次 ws35 阻塞（凭证 401）同源：供应商硬失败必须落闩锁终止整轮，而不是逐调用面兜底。

---

## 4. 已落地的四轨与可度量证据

### 4.1 A 轨：契约进数据流

`contract.json`（schema `huginn.task-contract/v1`）成为唯一权威：清单、键名、假设标签、`loop_log` 行数与字段、静态约束全部来自契约，objective 散文不再被正则解析。启动时缺契约且未显式设 `HUGINN_ALLOW_UNCONTRACTED=1` 即拒绝启动（exit 2）；`results.json` 必须写入 `contract_sha256`，门用同一份字节复核。两个真实任务夹具锁住「同门不同题」：ws35（MgH2 元分析）通过，ws34（Fick 扩散反演）因 `loop_log` 字段缺失按预期 FAIL。

### 4.2 B 轨：默认拒绝

- local bash 默认拒绝：仅 `HUGINN_ALLOW_LOCAL_BASH=1` 且（Landlock 可用或显式 `HUGINN_UNSAFE_ENV=1`）才放行；降级必须在 `results.json` 记 `unsafe: true`。
- 审批三档 `deny|workspace|all`，默认 `workspace`：工作区外写/执行 DENY（`matched_rules=["workspace:deny"]`）。
- 子进程环境剥离：匹配 `API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL` 的变量不进 bash 子进程环境，实测 `env` 输出不含 `INTERNLM_API_KEY`。

### 4.3 C 轨：文档被断言或删除

四个目标文档首行必须为 `generated` / `manual` 之一，无标记即 FAIL；`generated` 文档由测试重跑生成命令并做字节比对；`manual` 文档做 env 键交叉检查（文档里出现的 `HUGINN_*` 必须在 `env_schema.SCHEMA` 登记）。当前 `docs/env-contract.md` 覆盖 330 个环境变量。该机制在本会话抓到一处真实漂移：源码行号位移与新增 `HUGINN_TRACE` 未重生成，导致字节比对失败；重生成后恢复。

### 4.4 D 轨：执行覆盖度量

`HUGINN_TRACE=1` 时用 `sys.setprofile` 按文件计数，落 `trace.json`；对照 `research/huginn_module_census.csv` 列出「KEEP 但零执行」。普查共 123 个顶层模块，其中 KEEP 97 个（工作流内 51 + 可达 46），MOVE 17、SUSPECT 5、DELETE 2、ENTRY 2。度量口径的关键约束是：**不产生 trace 时不得声称已度量**。

### 4.5 治理面现状

`docs/governance-state.md` 由 `python -m huginn.cli.governance_audit` 确定性生成，覆盖 7 个工作流、17 个 arch 门、2 个 CLI 门、5 个本地门、21 个工程门、19 个快速失败门、122 条工作流命名 step 下限、19 个能力锚点、13 个 prompt 段；镜像漂移预算为 0/0/0（实测 0）。预算只减不增，任何放宽都是需要显式修改 `BUDGET_SPECS` 的发布决策。

---

## 5. 跟踪机制：三层闸门与升级规则

### 5.1 三层结构

| 层 | 频率 | 输入 | 判定 | 失败后果 |
|---|---|---|---|---|
| L1 门 | 每个 PR / 每次提交 | 契约、安全默认、文档、治理登记、镜像 | 确定性测试全绿 | 阻断合并 |
| L2 覆盖 | 每次真实运行 | `HUGINN_TRACE=1` 的 `trace.json` + census | 「KEEP 但零执行」清单可解释 | 该次运行不得宣称已度量 |
| L3 真值 | 每批次 / 每个真实任务 | ws35/ws34 产物与冻结真值 | 数值容差、假设 supported/refuted、`loop_log` 行数与字段 | 任务判定为未完成 |

三层缺一不可：L1 保证「流程可信」，L2 保证「代码真的被执行」，L3 保证「结果真的对」。外部证据里 59% 的编造论断能通过自动评审，正是因为只有 L1 的形态检查、没有 L3 的真值对账 [cite:5]。

### 5.2 节奏

- **PR 级**：`tests/test_task_contract.py`、`tests/test_security_defaults.py`、`tests/test_doc_assertions.py`、`tests/test_trace.py`、`tests/loops/test_structural_guards.py`、`tests/test_publish_mirror_drift.py`、`tests/test_arch_gate_registry.py`。
- **日级**：`governance_audit` 重生成比对 + 镜像漂移 check。
- **批次级**：真实任务（ws35）真跑 + 冻结真值审计；trace 覆盖率报告。

### 5.3 升级规则（硬规则，不靠自觉）

1. 同一缺陷类第 2 次复发 → 禁止实例补丁，必须先改判据/契约并挂可执行探针（`test_repeated_defect_class_requires_structural_fix_and_probe` 已把此规则变成门）。
2. 预算只减不增；上调必须走 `BUDGET_SPECS` 显式修改。
3. 默认拒绝优先：任何新能力默认关闭，开启需显式开关 + 可审计标记。
4. 无 trace 不得声称已度量；无真值审计不得声称任务完成。
5. 供应商硬失败（配额/认证）必须终止整轮并落原因，不允许规则兜底空转。

---

## 6. 当前跟踪登记表（本会话实测）

| 项 | 状态 | 证据 | 下一动作 |
|---|---|---|---|
| 环境可用性 | 已恢复 | `/workspace/venv` 重建，`uv pip install -e ".[dev]"` 成功 | 保持锁文件口径 |
| pytest 生态兼容 | 已修复 | pytest 9 移除 `Function._nodeid`，pytest-benchmark 5.x 不兼容；已锁 `pytest>=7,<9`，实测 8.4.2 下 benchmark 31 passed | 观察 upstream 适配后解锁 |
| 审批旧开关映射 | 已修复 | `HUGINN_APPROVE_MODE` 声明了默认值导致 `env_str` 永远返回 `workspace`，旧开关分支走不到；改用 `env_is_set` 分流 | 回归通过 |
| 文档漂移 | 已修复 | `env-contract.md` 缺 `HUGINN_TRACE` 且行号位移；重生成后 330 键 | 日级重生成 |
| 镜像漂移 | 0 | `scripts/mcpb_vendor.py --write` 同步 `permissions.py`；漂移门 2 passed | 每次改动 huginn/ 后同步 |
| 全量回归（T13） | 进行中 | `pytest tests/ -o addopts="" -q`（pytest 8.4.2，后台） | 全绿后勾 T13 |
| ws35 真跑（50 轮） | 阻塞 | 书生凭证 401，无法产生真实 `results.json` | 凭证恢复后重跑 |
| 冻结真值审计 | 待做 | 依赖 ws35 产物 | 数值容差 + 假设判定 + 行数 |
| 零引用文件清理（57） | 未启动 | spec 中 deferred | 单独批次逐条确认 |
| MOVE 模块迁移（17） | 未启动 | spec 中 deferred | 不动 |

---

## 7. 结论与优先级

1. **P0｜真值验证**：没有 ws35 真跑与冻结真值审计，A 轨与 K 类只能证明「门变严了」，不能证明「结果可信」。这是当前唯一会阻断 AI4S 目标达成的缺口。
2. **P1｜全量回归收口**：环境重建暴露出 pytest 主版本漂移这一隐性风险；锁版本后全量结果才是可复现的基线。
3. **P2｜trace 常态化**：把 `HUGINN_TRACE=1` 设为真实运行的默认，使「KEEP 但零执行」持续可见，而不是一次性体检。
4. **P3｜外部依赖失败可见化**：L 类闩锁已落地，但需在真实 provider 故障场景中复验「终止而非空转」。

外部证据给出的方向一致：自主科研系统的可信度由**工作流结构**决定，而不是算力规模 [cite:5]。把验收权交给契约、把失败暴露成事件、把真值对账变成门，这三件事比增加模型档位更接近 AI4S 的终极目标。

---

## 8. 局限

- 外部文献中含 2026 年预印本（[cite:4][cite:5][cite:6][cite:7]），未经同行评审的部分按「多数意见/单一来源」标注，不等同于共识。
- GNoME 的「38.1 万可合成、736 已合成」来自二级来源 [cite:11]，标记为部分未核验。
- 本报告未覆盖 57 个零引用文件的删除决策与 17 个 MOVE 模块的迁移，二者仍是未启动的独立批次。
- 仓库内数字随工作树变化；重生成口径为 `docs/governance-state.md` 与 `scripts/doc_assertions.py --regen`。

---

## Sources

- [cite:1] Agent4S: The Transformation of Research Paradigms from the Perspective of Large Language Models — https://arxiv.org/abs/2506.23692
- [cite:2] From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery — https://arxiv.org/abs/2508.14111
- [cite:3] AI for Science：智能化科学设施变革基础研究（中国科学院院刊）— http://bulletin.cas.cn/BCAS_CH/doi/10.16418/j.issn.1000-3045.20230820004
- [cite:4] Deep Research of Deep Research: From Transformer to Agent, From AI to AI for Science — https://arxiv.org/html/2603.28361
- [cite:5] MLReplicate: Benchmarking Autonomous Research Systems for Machine Learning Reproducibility — https://arxiv.org/abs/2605.16616
- [cite:6] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite（ICLR 2026）— https://proceedings.iclr.cc/paper_files/paper/2026/file/b2ce9568dbb559aefc8c98ca5b5314ce-Paper-Conference.pdf
- [cite:7] TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents — https://arxiv.org/html/2609.05079v1
- [cite:8] DeepMind predicts millions of new materials（Science）— https://www.science.org/doi/pdf/10.1126/science.adn2116
- [cite:9] 'Self-driving' labs power up（Nature）— https://media.nature.com/original/magazine-assets/d41586-026-00974-2/d41586-026-00974-2.pdf
- [cite:10] The lab that learns（Science）— https://www.science.org/doi/10.1126/science.aee2448
- [cite:11] AI for Science in 2026: Real Breakthroughs vs. the Hype（二级来源）— https://www.fatherofai.in/blog/ai-for-science-2026-breakthroughs-vs-hype/
