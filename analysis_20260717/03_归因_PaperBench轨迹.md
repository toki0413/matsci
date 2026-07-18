# 03 · PaperBench 轨迹归因报告

**诊断对象**：huginn-agent · PaperBench 三篇论文复现（all-in-one / pinn / stochastic-interpolants）
**角色**：论文复现归因专家
**日期**：2026-07-17
**方法**：严格只读。解析三篇 `_score.json` 全部 2206 个叶节点评分与 judge 评语、submission 源码与 mtimes、`_m1_*.log`、checkpoint sqlite（只读模式）、`paperbench_huginn.py` 与 `agent/huginn/bench/orchestrator.py` 机制代码；直接引用 `audit_20260717/` 04/05/07/16 已确认事实（标注出处）。未运行 agent 本体。

---

## ① 数据与方法

### 数据源

| 数据 | 路径 | 说明 |
|---|---|---|
| 逐叶评分 | `workspaces/paperbench/<paper>/_score.json` | all-in-one 174 叶 / pinn 1963 叶 / SI 69 叶，含 judge reasoning |
| 历史评分 | `workspaces/paperbench/all-in-one/_score_M7.json`、`_score_T20.json` | 12.36 / 15.33，对比当前 13.52 |
| 提交物 | `workspaces/paperbench/<paper>/submission/` | 源码、outputs、mtimes |
| 运行日志 | `_m1_full.log`（崩溃轨迹）、`_m1_full2.log`、`_m1_smoke*.log` | M1 里程碑运行 |
| harness | `paperbench_huginn.py`（978 行）、`agent/huginn/bench/orchestrator.py`（272 行） | 约束与编排机制 |
| checkpoint | `workspaces/paperbench/all-in-one/.checkpoint.sqlite` | **1.30 GB**，1321 条 checkpoint，只读查询 |
| 轨迹 | `agent/.huginn/trajectories/loop_*.json` | 18 个文件均为 autoloop（MoS2 等）轨迹，`tool_calls` 全为空，**不含 PaperBench 运行的逐步工具记录**——预算分解只能依赖日志、mtime 与 checkpoint 统计重建（此为取证限制，下文标注「推断」处即由此产生） |

### harness 既定约束（机制事实）

- `timeout=3600`、`max_tool_calls=150`（`paperbench_huginn.py:916-917`）。
- **rubric 直接交给 agent**：`rubric.json` 复制进 workspace（`:203-208`），prompt 明示「your grading criteria」（`:271-272`），且 prompt 内含 4 条具体叶节点标准答案（`:296-304`，audit_20260717/16 P0-1 已确认为泄漏）。
- 完成判据：`PAPERBENCH_DELIVERABLES` 仅 3 个 glob——`reproduce.sh` 存在 + 任意 `*.py` 存在 + 任意 `outputs/*.json` 存在（`orchestrator.py:61-65`）；`_is_done` 允许「交付齐 + 调用 ≥ 75 次（150×0.5）」即停（`orchestrator.py:147-159`）。
- **phase-aware budget 全程未生效**：`_get_budget_override` 仅在 `mode=="research"` 时返回预算（`orchestrator.py:171-173`），而 agent 默认 `_mode="chat"`（`agent/huginn/agent/core.py:391`），适配器从未 `set_mode`。`PHASE_BUDGETS`（`agent/huginn/phases.py:48-52`，LITERATURE=50 / EXECUTION=300）在这批运行中是死配置；唯一生效的上限是 agent 级 150 次工具调用。
- judge 侧抬分引导：不确定给 30、见 loss 曲线给 50+、regex 只向上 override（`paperbench_huginn.py:807-810,842-844,648-672`；audit 16 P1-6）。**即当前分数已是注水后的上限，原生协议下只会更低。**

### 三篇 rubric 权重结构（从 rubric.json 实算）

| 论文 | 叶数 | 总权重 | Code Development | Code Execution | Result Analysis |
|---|---|---|---|---|---|
| all-in-one | 174 | 199 | 99（50%） | 62（31%） | 38（19%） |
| pinn | 1963 | 2514 | 157（6%） | **2319（92%）** | 38（2%） |
| stochastic-interpolants | 69 | 96 | 85（89%） | 7（7%） | 4（4%） |

---

## ② 核心发现（按证据强度排序）

### F1 【直接代码验证】prompt 给了 4 条标准答案，最终代码 4 条全错，且跨 3 代迭代未修

prompt 的 "RUBRIC ALIGNMENT (critical — past runs lost points here)" 节（`paperbench_huginn.py:296-304`）把历史失分点的答案蒸馏成明文指令。对照当前提交：

| prompt 明文答案 | 当前代码实况 | 证据 |
|---|---|---|
| "value embedding must REPEAT scalar to match dim, NOT nn.Linear(1, d)" | `self.value_embed = nn.Linear(1, d_value)` | `submission/simformer.py:29` |
| "tokenizer concat order MUST be: identifier, value, metadata, condition" | `torch.cat([id_emb, value_emb, cond_emb])` —— 仅 3 项且无 metadata | `submission/simformer.py:37` |
| "attention mask must implement ALL THREE (undirected/directed/dense)" | 只有 `build_dependency_attention_mask` 一种 | `submission/simformer.py:64` |
| "loss MUST have λ(t)=g(t)² weighting, NOT simple mean" | `loss = ((noise_pred - noise)**2 * latent_mask).sum(-1).mean()`，无 λ(t) | `submission/train.py:59` |

judge 对这 4 处全判 0（`_score.json` 叶 #8/#11/#17-20/#25 一带）。**同样的错误在 `submission_T20_backup/simformer.py:44,53` 中一字不差地存在**——M7（12.36）→ T20（15.33）→ M1（13.52）三代迭代中，prompt 不断累加教训，代码缺陷原样存活。这排除了「信息不足」解释：失败发生在「长上下文中把明文约束落实到 5KB 模型文件」这一步，且没有任何机制在写码后回读约束做断言（prompt 要求的 self-check `:303-305` 从未执行）。

### F2 【评分统计】三篇的执行+分析叶几乎全灭，这是低分的算术主因

| 论文 | 总分 | 叶得分分布（0/30/50/100） | Dev 均分 | Exec 均分 | Analysis 均分 |
|---|---|---|---|---|---|
| all-in-one | 13.52 | 144 / 1 / 12 / 17 | 24.8（92 叶） | **0.8（62 叶）** | **0.0（20 叶）** |
| pinn | 2.23 | 1887 / 4 / 53 / 19 | 19.4（126 叶） | **1.2（1815 叶，占 92% 权重）** | 0.0（22 叶） |
| SI | 33.85 | 47 / 0 / 4 / 18 | 34.5（58 叶） | 0.0（7 叶） | 0.0（4 叶） |

- all-in-one 执行+分析共 82 叶、100 权重（占 50.3%），**仅叶 #48 得 50 分**，其余全 0。
- pinn 的全部 19 个 100 分里 18 个是 Dev、1 个是 Exec（叶 #20，width-50 MLP「被训练过」）。
- 失分类型归因（按 judge 评语归类 + 代码核实）：
  - **未运行/未跑到规模**（首因）：all-in-one 执行叶 40 处要求 10^3/10^4/10^5 三档模拟量扫描，代码只有 `n_train=5000` 单点（`train.py:18`）；HMM、Tree 两个 benchmark 完全未实现（judge：「No HMM task implemented」）；无 MCMC 参考后验。pinn 要求 3 问题 × 4 宽度 × 多学习率 × 4 优化器 × 5 种子的网格，实际只跑了 convection 的 3 个配置（`outputs/convection_results.json`，仅 3 条记录）。
  - **实现错/与 rubric 细节不符**：见 F1；SI 的 Dev 0 分几乎全是 rubric 明文细节未落实——64 等分 tile + p=0.3 mask、类别值通道、最近邻上采样、损坏图像作通道、精确 loss 形式（`_score.json` 各叶 reasoning）。
  - **占位式执行**：pinn `convection_results.json` 三条记录 training_time 分别 0.117s / 3.81s / 0.036s，L2RE=1.02~1.41（≈完全没学到）；adam 与 adam+lbfgs_1k 的 final_loss **小数点后 16 位完全相同**（3.306041717529297）——L-BFGS 阶段未产生任何效果，但 agent 未察觉、照写 README 宣称完整扫描（`submission/README.md` + `reproduce.sh` 自称 `--seeds 5 --iters 41000`）。
  - **harness 代跑**：本次三篇的 outputs 均为 agent 运行内产物（无 `fallback_run_log.json`，`_execute_training_fallback` 未触发，`:413-415` 提前返回）——audit 16 P1-1 在本批属**背景缺陷**。

### F3 【agent 自证】SI 交付时自己的测试 4/9 报错，包括插值定义性性质 I_0=x0

`workspaces/paperbench/stochastic-interpolants/submission/outputs/test_results.json`（agent 自己生成）：

- `"interpolant_basics": "ERROR: I_0 should equal x0"` —— 插值在 t=0 必须等于 x0 是该论文方法的定义性性质，实现错了。
- 3 处 UNet 通道数不匹配（`expected ... 32 channels, but got 16` 等），full inpainting / super-resolution pipeline 均 ERROR。
- `"all_passed": false`。

agent 在 10:04 生成这份全红报告后，10:09 又写了 `run_tests.py` 做了一次小验证（`quick_test.json` 只测一步 loss=20.43），随后运行结束——**带着自证失败的现场交付**。judge 不运行代码，Dev 叶仍给了 18 个 100；但 7 个执行叶 + 4 个分析叶（ImageNet 训练 / FID-50k）全 0。另外 `submission/README.md` 缺失（harness 列为交付物，`paperbench_huginn.py:252-253`）。

### F4 【时间线重建】三次运行无一触到 3600s 超时；完成判据过松是「提前收工」的直接通道

| 运行 | 墙钟 | 结局 | 证据 |
|---|---|---|---|
| pinn 主运行 | 约 19 分钟（04:13→04:32） | 9 文件 + 1 次占位训练后停止；**未超时** | submission mtimes |
| SI 主运行 | 约 32 分钟（09:37→10:09） | 自测失败状态下停止；**未超时** | submission mtimes |
| SI 二次运行 | 25 秒 | 0 字节新 checkpoint、零新文件——实质是重打分过场 | `_huginn_meta.json`（duration_seconds=25）、`.checkpoint.sqlite` 0B |
| all-in-one 崩溃运行 | 16 次工具调用后死亡 | 全部花在读 paper/rubric（ls×2→bash→code_tool 连读→write_todos→继续读），第 16 次后 `openai.BadRequestError 400` | `_m1_full.log` |
| all-in-one 产出运行 | ≥2 段（04:56–05:01、05:39–05:57，跨 1 小时需多次 resume） | 产出当前 submission；其间还把 outputs 写重了一份到 workspace 根目录（路径纪律失守，prompt `:356-363` 明令禁止） | mtimes；根目录 `outputs/` 与 `submission/outputs/` 双份 |

机制含义：150 次调用 + `_is_done` 的 3-glob 判据（`orchestrator.py:61-65,147-159`）意味着 agent 只要写出若干 .py、跑出一个 JSON、越过 75 次调用，编排器就合法放行。**「实验矩阵覆盖了 rubric 的多少」根本不在完成判据里**。pinn/SI 的提前收工不是超时杀人，是闸门放行。

### F5 【机制+审计互证】上下文基础设施在烧预算：崩溃、递归超限、1.3GB 不可压缩状态

- `_m1_full.log`：第 16 次调用后 langgraph 状态损坏——`An assistant message with 'tool_calls' must be followed by tool messages...`（400）。压缩/截断删掉了 tool 响应消息却留下 assistant 的 tool_calls（与 audit 05 P1-4「checkpointer 路径下压缩只作用于本轮新消息、历史无限增长」同族）。一整轮论文阅读预算全废。
- `_m1_smoke4.log`：5 次调用的 smoke 也以 `GraphRecursionError: Recursion limit of 250 reached` 死亡——图循环无停止条件。
- 当前 all-in-one `.checkpoint.sqlite`：**1.30 GB / 1321 条 checkpoint / 单条均值 760 KB / 单线程**。每次 resume 要恢复这份不断膨胀的状态；99KB 的 `paper_text.txt` 与累积工具输出被逐 step 复制。audit 05 P1-4 的「checkpoint 状态随会话无限增长」在此达到荒诞量级。
- 预算去向粗分解（可观测部分）：崩溃运行 16/16 调用全在 LITERATURE；pinn 的 rubric.json 为 **1,163,686 字符**（1963 叶），按工具输出截断上限需上百次分块读取才能过完一遍——rubric 本身就是预算黑洞（`paperbench_huginn.py:525` 注释自承「第 8 次 49 calls 读 paper 被 timeout 杀，全浪费」，为此打的补丁是 checkpoint resume，而 resume 又落入 F5 的膨胀陷阱）。

### F6 【背景缺陷，本批未直接起作用】

- **judge 视野非递归**：`score_submission` 只 `outputs_dir.glob("*")`（`paperbench_huginn.py:750`），`outputs/simformer/*.json`、`outputs/baselines/*.pt` 不可见。但抽查 NPE 训练叶的 0 分评语（如叶 #89「only trains with a single n_train value」）均为实质缺口——此缺陷在本批最多影响边缘分数，列为背景。
- **fallback 代跑**（audit 16 P1-1）：本批未触发，见 F2。
- **rubric 泄漏**（audit 16 P0-1）：使 13.52/2.23/33.85 与官方榜单（o1≈26.6%）不可比；但对「为什么低分」的解释方向相反——**给了答案还考砸，恰恰放大而非掩盖了能力缺口**（F1）。
- `write_file/edit_file` 被禁用导致浪费调用（prompt 自述 `:283-285`）：无轨迹可量化，存疑列背景。

---

## ③ 根因链（现象 → 机制 → 代码位置）

**现象层**：三篇总分 13.52 / 2.23 / 33.85；执行+分析叶 82/82、1815/1815、11/11 几乎全 0（F2）。

**机制层**（四条互相咬合的链）：

1. **完成判据与 rubric 脱节** → agent 以「文件存在」为终点而非「rubric 覆盖」为终点。
   `_is_done` = 3 个 glob + 75 次调用下限（`agent/huginn/bench/orchestrator.py:61-65,147-159`）；编排器的三档分流只检查缺失文件（`:208-219`），从不解析 rubric 覆盖率。pinn 执行叶占 92% 权重，完成判据里没有一个 bit 指向它们。
2. **无覆盖规划** → 预算被「单点做深」吃掉，执行矩阵整片留白。
   agent 把 convection 一个配置跑出结果就写 README（pinn），把 n_train=5000 单点训练当完成（all-in-one `train.py:18`），从未把 rubric 解析成「任务 × 规模 × 方法」矩阵再分配预算。system prompt 要求 "Prioritize by rubric weight"（`:293`）但无机制支撑。
3. **执行验证闭环缺失** → 坏结果不被识别为坏。
   SI 自测 4/9 报错（含 I_0≠x0）仍交付；pinn L2RE=1.02、adam 与 adam+lbfgs 损失逐位相同仍宣称「All sweeps complete」；all-in-one C2ST 出现 0.26（<0.5，分类器精度低于随机——评估实现必有 bug）与 0.95（后验完全可区分）无人复核（`submission/outputs/c2st_results.json`）。agent 不读自己的指标，科学家的「结果 sanity check」这一层完全空缺。
4. **上下文基础设施漏预算** → 有效预算远低于名义 150。
   消息史损坏使运行在第 16 次调用整轮报废（`_m1_full.log` 尾部 400 错误）；checkpoint 无限膨胀至 1.3GB（audit 05 P1-4：`context_builder.py:466-467` + `streaming.py:906-967` 只压新消息）；pinn 的 1.16MB rubric 在读取端吃掉大量调用。每一次崩溃/膨胀都强迫下轮重读论文重建立上下文。

**代码位置索引**：完成判据 `orchestrator.py:147-159`；deliverable 定义 `orchestrator.py:61-65`；失效的 phase budget 通道 `orchestrator.py:161-174` + `core.py:391`；泄漏+答案蒸馏 `paperbench_huginn.py:203-208,271-272,296-304`；judge 抬分 `:807-810,842-844`；judge 视野 `:750`；单点训练硬编码 `submission/train.py:18`；F1 四处实现错误 `submission/simformer.py:29,37,64`、`submission/train.py:59`。

---

## ④ 对用户问题的回答：工作流 / 模式切换 / 其他？

**结论：模式切换（phase budget）基本不背锅——它在这批运行里根本没通电；工作流（harness 编排与判据）是最大的可修放大器；但根子在 agent 核心能力层的「科学家闭环」缺失。**

1. **模式切换：排除为主因。** phase-aware budget 因 mode 恒为 chat 而从未生效（`orchestrator.py:171-173` + `core.py:391`），`PHASE_BUDGETS` 是死配置；三次运行两次远未触及 3600s 超时（19 分钟、32 分钟），也不存在「EXECUTION 阶段预算被 LITERATURE 抢光」的实际发生（pinn 的 rubric 读取黑洞是文件尺寸问题，不是 phase 预算分配问题）。这套机制的问题是「名义存在、实际空转」（与 audit 05 对全系统的结论同构），而非「切换逻辑切错了」。
2. **工作流：直接失分的放大器，且是最便宜的可修点。** 3-glob 完成判据（F4）、judge 抬分引导与非递归视野（F2/F6）、rubric+答案进 prompt（F1 的畸形补偿——用泄漏补能力，结果补不上还毁了指标可比性）、崩溃烧预算（F5）。这一层修好了，同样能力的 agent 分数会显著上升——因为它目前被允许在 20% 覆盖率时合法收工。
3. **其他（agent 核心能力）：低分的最终归因。** 即便闸门修好，F1 证明 agent 在长上下文里连明文给的 4 条约束都落实不进一个文件（指令遵循/长程一致性缺陷）；F2/F3 证明它不把 rubric 当实验矩阵规划（覆盖规划缺陷）、不读自己的实验结果（自我验证闭环缺陷）。这正是「给了 rubric 还得低分」暴露的能力层：**长程规划（把评分树翻译成实验矩阵并预算化）+ 代码正确性（定义性性质 I_0=x0 都错）+ 执行验证闭环（L2RE=1.02 不报警）三者同时缺位**，不是单点短板。与 SAB（代码截断、缺主执行块）、RCB（只描述不执行、占位数值）的判词完全同构——这是跨 benchmark 的同一签名：agent 止步于「产物存在」，从不验证「主张成立」。

---

## ⑤ 可操作建议（按投入产出比排序）

1. **【最高 ROI，约 1-2 天】把完成判据从「文件 glob」换成「rubric 覆盖清单」**：harness 侧用 `collect_rubric_leaves`（现成函数）把 rubric 解析成叶级 checklist，编排器每轮注入「已实现 a/b、已执行 c/d、无证据 e/d」的机械统计，`_is_done` 增加「执行叶证据覆盖率 ≥ X% 或逐叶标注 infeasible」条件。直接攻击 50–92% 权重的无人区，且不需要 agent 变聪明，只需要不许它提前收工。
2. **【高 ROI】harness 预消化 rubric 为实验矩阵 + 预算表**：对 pinn 类 1.16MB rubric，先机械抽取「任务 × 超参 × 种子」矩阵压缩成几 KB 表格交给 agent（省上百次分块读取），并注入硬策略：**全网格缩尺执行（每格 500 迭代）优先于单格全量执行**——rubric 的 Execution 叶按格子给分，覆盖优先于深度。同时消解 F5 的读取黑洞。
3. **【高 ROI】落地「sanity gate」自我验证**：在允许写 README/停止之前，编排器机械检查：loss 曲线单调下降区间存在、outputs 指标通过领域断言（L2RE<1、C2ST∈[0.5,1]、agent 自测全绿）、失败则注入修复指令而非放行。SI 的 I_0≠x0、pinn 的 0.117s「训练」、all-in-one 的 C2ST=0.26 都会被这道闸门拦下。可复用现有 `_execution_prompt` 三档分流框架（`orchestrator.py:102-112`）。
4. **【中 ROI，兼修泄漏 P0-1】删除 prompt 中的 4 条叶节点答案**（`paperbench_huginn.py:296-304`），替换为「每写完一个 .py 必须 code_tool 执行针对该文件 rubric 叶的断言脚本」的流程指令——把答案蒸馏改成可执行的验收程序。既恢复分数可比性，又把 F1 的「给了答案不落实」变成「不过断言不许走」。
5. **【中 ROI】修上下文损坏与膨胀**：压缩/截断时保证 assistant tool_calls 与其 tool 响应成对存留（修 `_m1_full.log` 的 400 崩溃族）；落实 audit 05 P1-4 建议（`update_state` + `RemoveMessage` 真正修剪 checkpoint），并给 benchmark 运行限定 checkpoint 尺寸上限。每次崩溃平均烧掉 16-50 次调用的重读成本。
6. **【低中 ROI】judge 视野修复**：`outputs_dir.glob("*")` 改 `rglob`（`paperbench_huginn.py:750`）；prompt 中要求 outputs 扁平化作为双保险；fallback 产物按 audit 16 P1-1 隔离披露。
7. **【低 ROI 但必要】不可行叶协议**：SI 的 ImageNet 训练 + FID-50k 在 3600s/CPU 下物理不可行——prompt 应明确「缩尺替代 + 诚实文档化可换部分分」（如 CIFAR 级演示 + README 注明差距），而非留白得 0。当前 prompt 只有 "document the gap honestly"（`orchestrator.py:97-98`）出现在 triage 分支，主路径没有。
8. **【跨 benchmark 联动】** 本报告的 F2/F3 签名（占位执行、无验证闭环）与 SAB 25 分、RCB 5.25-15.75 分的 judge 评语同根——建议 1/3/4 修在 `huginn/bench/orchestrator.py` 共享层，一次修复五个 benchmark 同时受益。

---

### 附：分数有效性声明

三篇分数已受 judge 抬分引导（audit 16 P1-6）与 rubric 泄漏（P0-1）影响，**高于**原生 PaperBench 协议下的等值分，且不可与官方榜单对比。本报告的归因结论（执行覆盖崩塌、验证闭环缺失）不依赖分数绝对值，全部由叶级评语、代码实况与运行时间线独立支撑。
