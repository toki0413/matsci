# Huginn 设计论据（Design Rationale）

> 目的：回答两个"为什么"——**我们到底怎么设计的、为什么这么设计**。
> 读者对象：接手本项目的人、合作者、以及任何想判断"这个 agent 是否值得信任/可演化"的人。
>
> 本文不重复契约清单（那些在 docs/ 下的 `-contract.md` 契约文件，由代码自动再生、不会漂移）；
> 本文只讲**设计的动机、取舍、代价与诚实边界**。技术现状事实以 [tech-spec.md](tech-spec.md) 为准，
> 直连模块清单以 [architecture.md](architecture.md) 为准。

---

## 0. 一句话定位（先对齐认知）

**Huginn 不是一个"建立在数学语言之上"的符号系统，也不是一个普通函数调用壳；**
它是一个 **以 LLM 推理循环为地基、用"数学做真算工具 + 结构审计层"、用"朗兰兹函子性作为设计哲学隐喻"** 才能自洽的
自驱式材料科学 agent。

- **数学 = 工具箱 + 审计层，不是基底语言。** 我们在主循环里真实计算同调不变量、Čech sheaf 上同调、Hodge 签名、
  持续同调、因果模型、Lean 证明——但它们**补强** LLM 判断（advisory），**不取代** LLM 推理，更不是承载推理本身的语言。
- **朗兰兹 = 设计精神（分离域 / 结构保持 / 只在自然边界翻译），不是字面的朗兰兹纲领。**
  没有任何 Galois 表示、自守形式、L-函数。它是拿来约束架构的一种思想，详见 §5.4。

如果你未来的文档或对外表述里出现"符合朗兰兹纲领"，请先读 §5.4 —— 按我们的定位，那句话是**隐喻**，不是数学主张。

---

## 1. 为什么存在这个 agent（核心动机）

材料科学里，一个研究目标不是"跑一个脚本"，而是一条**分叉的探索路径**：有的方向可行、有的撞墙、有的推翻了子假设。
传统 workflow（定死的 DAG）无法表达这种"自主分叉—试错—学习—再试"。

所以我们不写 workflow，写一个 **认知回路（Cognitive Loop）**，目标是让 agent：

1. 有**主动性**：不是每次都被 query 推着走，而能在 objective 下自我提出假设、执行、验证、迭代（见 [autoloop/](../huginn/autoloop/)）。
2. **可演化**：不只是"LLM 在跑"，而是**系统本身的行为（prompt 模板 / workflow / phase / tool 白名单）可以被改进**（见 [harness_evolution_spec.md](harness_evolution_spec.md) 的 H0–H5）。
3. **对错误诚实**：当它自己不确认某个结论时，能区分"我知道 / 我推测 / 我没把握"，而不是把不确定当定论输出（见 §5.3 CLAIM 层）。

这三个动机分别对应三根支柱：**Agent 循环**、**Harness 可演化**、**Knowledge 可审计**。

---

## 2. 分层架构总览（真实代码，不看图）

简化成四层（详细职责在 [architecture.md](architecture.md)，这里只看"每层解决什么问题、为什么存在"）：

```
┌──────────────────────────────────────────────────────────────┐
│ Entry: CLI (~40 cmd) · API (FastAPI /v1, WS/SSE)             │
├──────────────────────────────────────────────────────────────┤
│ ① Agent / Autoloop 层   —— 决定"下一步做什么"                │
│    agent/  core loop、session、streaming、reflection          │
│    autoloop/ CognitiveLoop(observe/decide/execute/reflect)    │
│    agents/  多 agent：orchestrator、subagent、swarm、team      │
├──────────────────────────────────────────────────────────────┤
│ ② Capability 层        —— 决定"怎么做"                        │
│    tools/(~178 + Registry) · skills/(presets + .md)           │
│    memory/ 3-tier · evolution/ · knowledge/ · causal/         │
├──────────────────────────────────────────────────────────────┤
│ ③ Metacog / 数学审计层 —— 决定"判断得可不可信"                 │
│    metacog/ 同调、sheaf H¹、Hodge、范畴 functor、mental imagery│
├──────────────────────────────────────────────────────────────┤
│ ④ Runtime / 安全 / 状态   —— 决定"能不能在真实环境里跑"         │
│    runtime/ · security/ · events/ · persistence/ · hpc/       │
└──────────────────────────────────────────────────────────────┘
```

**为什么这样分：**
- ①②分开是因为"策略"和"能力"是两种不同的东西，混在一起会导致"换套能力就要改推理逻辑"。
- ③独立成层，是为了让**结构审计**（数学不变量）能和**语义推理**（LLM）互不污染：审计结果作为 advisory 注入 prompt（见 [engine_observe.py:1026](../huginn/autoloop/engine_observe.py)），而不去改推理内核。
- ④兜住真实环境的约束（安全、持久化、远程执行），让①②③可以"天真"地研究而不必担心炸掉生产。

---

## 3. 设计原则（我们从哪几条原则推出来的）

来自 [architecture.md](architecture.md#Design-Principles) 与 [harness_evolution_spec.md](harness_evolution_spec.md#设计原则)，合并提炼为 7 条：

1. **优雅降级（graceful degradation）**：每个组件带 mock/fallback 路径。没有 chromadb 也能用离线的 KB；GUDHI 没装也能退 networkx 风格 β₀/β₁。这让研究不必等基础设施。
2. **安全默认（fail-closed）**：工具元数据默认只读/非破坏/需确认；密钥只进内存、逐 item salt。（[security/default_policy.yaml](../huginn/security/default_policy.yaml)）
3. **模块化 + 类型安全**：Pydantic 统一 I/O；组件可独立使用。
4. **可测防回归**：session 级共享 app + autouse 护栏保证 `ToolRegistry` 测试间逐位一致；覆盖率门禁 60。
5. **文件系统即记忆**：harness 变体 / patch / 归档存文件（`.huginn/...`），不塞 context window。
   这是权限外记忆，避免"所有事都挤进 LLM 上下文"。
6. **不破坏控制流以换取回退成本 = 删一个目录**：CognitiveLoop 的 4 个钩子（observe/decide/execute/reflect）签名稳定，
   演化发生在钩子**内部**。所以升一次级，最坏回退成本是一个目录，而不是一次大重构。
7. **数学做真算审计、做隐喻，不做装饰**：每个数学模块要么真的算出结构不变量喂给判断，要么被明确标注为 research/实验层；
   不允许"借个高大上名字但什么都没算"的东西混进主循环（这是我们今天修掉的那批过期注释背后的规矩）。

---

## 4. 0→1→5：我们怎么递进（演化路线）

一个"看起来很大的功能"（比如科学结论超图、harness 可演化）不是一次写完，而是按**阶段递进**落地。
harness 侧有明确的 H0→H5 路线（见 [harness_evolution_spec.md](harness_evolution_spec.md#执行顺序)）：

- **H0 stable_principles 接入 prompt** —— 最小修复"PM 层在 autoloop 不通"。
- **H1 prompt template patch** —— agent 能改自己的 prompt 模板 block（自改进的最小闭环）。
- **H2 workflow 演化搜索** —— 对同一 objective 生成 N 个 workflow 变体，用物理奖励做 bandit 选优。
- **H3 联合优化** —— prompt block + workflow 参数联合 bandit（不含 model 维度）。
- **H4 phase 行为体可演化** —— 把 phase 方法体 / subagent spec 抽成 PhaseSpec，agent 可改行为但不失控。
- **H5 unified LLM client / tool dispatch** —— 把 LLM 调用、工具白名单真正统一（已部分落地）。

**为什么递进而不是一步到位：** 每一步都是一个独立可验证的闭环（selfcheck 过了、能跑 3 轮 autoloop 不崩才继续），
且前一步是后一步的地基。这样任何一步失败，代价只停留在那一层，不会牵一发动全身。
新功能（如 CLAIM 超图）同样按 "0 骨架 → 1 单点 → 5 全链路" 的节奏递进，而不是一次性堆完。

---

## 5. 每层的 How + Why（核心）

### 5.1 Agent 循环（agent/ + autoloop/）

- **How**：`CognitiveLoop` 提供 4 个控制流钩子；`AutoloopEngine` 在其内实现 7 个 phase
  （perceive/hypothesize/plan/execute/validate/learn/report）；每次迭代都有 `_metacog_topology_audit`
  对假设图跑结构审计。
- **Why 要 7 phase 而不是一步**：把"产生想法"和"验证想法"分开，才能在「想法—结果」之间建立因果日志，
  否则 agent 无法判断"我上次改进到底有没有起作用"。
- **Why advisory 数学**：见 §5.2。

### 5.2 Metacog 数学审计层（我们重点核对过的一层）

**How（已核对源码，非装饰）：**
- 单纯同调/持续同调、sheaf H¹、Hodge 签名、范畴 functor、cognitive map、mental imagery 都已**接进主循环**，
  不是挂在实验目录里（[simplicial_homology.py](../huginn/metacog/simplicial_homology.py)、
  [sheaf_cohomology.py](../huginn/metacog/sheaf_cohomology.py)、[hypothesis_loop.py](../huginn/autoloop/hypothesis_loop.py)）。
- 结果经 [engine_observe.py:1026](../huginn/autoloop/engine_observe.py) **回灌 prompt**，作为"这个假设结构上自洽吗"的提示。

**Why advisory（关键取舍）：**
- 数学不变量是**结构上的**诚实信号（"多源证据全局不一致 H¹≠0"、"旋转模式无对应流形 β_rotation=0"），
- 但材料科学判断还需要**语义上的**诚实（某个物相是否合理、某篇文献是否可信），这只能交给 LLM。
- 所以两者互补：**数学给结构边界，LLM 给语义判断**，谁也不替代谁。把数学当硬门控会过拟合、把数学当装饰则浪费。
- **诚实边界**：部分更强力的拓扑/范畴模块在代码里仍标"research 层"（如
  [experimental/persistence_landscape.py](../huginn/experimental/persistence_landscape.py)）；那些是未来的 hook，不是今天的承诺。

### 5.3 Knowledge / CLAIM 层 —— "debug 人类已有的科学认知"

这是本项目区别于"知识库 = 文档检索"的地方。

- **How**：把文献结论提升为知识图谱里的 CLAIM 一等节点（[kg/entities.py](../huginn/kg/entities.py)、
  [kg/graph.py](../huginn/kg/graph.py)）；用超图表达"结论 ← n 元前提"的 AND/OR 依赖（[kg/hypergraph.py](../huginn/kg/hypergraph.py)）；
  用 sheaf H¹ 检测多源证据全局矛盾（可溯源到 metacog）；用 `ClaimAuditor` 编排"注册→冲突检测→挑战传播→自指环审计"（[kg/claim_audit.py](../huginn/kg/claim_audit.py)）。
- **Why**：人类已有科学认知本身不是一部永远正确、彼此自洽的数据库。一个结论建立在更早的假设上；不同实验证据强度不同；
  一个几十年前的源头判断在今天的理论框架下可能需要重新解释。我们希望当一篇新文献进来时，系统回答的不只是"数据库多了一篇"，
  而是"它支持了什么 / 挑战了什么 / 改变了哪些结论的可信度 / 改了谁的适用边界 / 如果它挑战源头，哪些下游知识要一起复查"——
  **像 debug 软件一样去 debug 人类已有的科学认知**。
- **落地**：存疑结论打 `contested` 标签，在 RAG 检索（[perception/rag_bridge.py](../huginn/perception/rag_bridge.py)）
  和 context 构建（[context_builder.py](../huginn/context_builder.py)）时注入提醒，让模型不会把有争议的知识当定论使用。
- **诚实边界**：这仍是"审计 + 提示"层；它**不产生**新的科学事实，只让 agent 对"证据是否打架、结论是否被挑战"更诚实。

### 5.4 存储隔离 —— 我们的"朗兰兹函子性"究竟指什么

**How**：三种记忆**
`memory`（时序语义）、`knowledge`（向量空间）、`kg`（图拓扑）**天然是三种不同的数学结构。

**Why 不全局打通**（这是最容易做错的地方）：
- 如果把 memory 硬转成向量去跟 KB 比，会丢失时序结构；
- 如果把 KG 压成向量嵌入，会丢失拓扑（环路、依赖、n 元合取）；
- 所以**不强迫三层全局互译**。

**"朗兰兹函子性"在这里的含义（请务必这样理解）：**
> 设计备忘录 [.omm/information-flow/context.md](../.omm/information-flow/context.md) 借用一个思想：
> 当两类结构之间存在"理应保结构"的对应时，翻译只在**自然边界**发生，且**必须保持结构**。
> 最短路径：噪音过滤 → 矛盾时回 grounded → 再结构化，而不是"把所有东西都塞进一个 embeddings 空间"。

这是一个**设计哲学隐喻**，不是字面的朗兰兹数学。它的可操作产物是三条规则：
1. 三库各自用自己的原生结构存储（不清洗掉自己的结构）；
2. 跨库翻译只发生在明确的边界点（如 ContextBuilder 拼 prompt 时）；
3. 翻译要保结构（time→time、graph→graph），不降维到单一向量糊在一起。

如果你要把这句"符合朗兰兹纲领"写进任何正式文档：请统一改成 **"借鉴朗兰兹函子性的设计精神（分离域 / 结构保持 / 只在自然边界翻译）"**。

---

## 6. 关键决策条目（ADR 摘要：背景 → 选项 → 为什么选它 → 代价）

### D-1 数学审计做 advisory 而非硬门控
- **背景**：想让"结构不一致"阻止 agent 前进。
- **选项**：硬拒绝 vs advisory 提示。
- **选它**：硬拒绝会在长尾（真实材料里本就允许某些 H¹≠0 的近似）上误杀；提示则保留 LLM 裁量。
- **代价**：advisory 只"影响判断"，不保证 agent 一定采纳 → 主循环里数学结果可能只是参考。

### D-2 file-system-as-memory 存 harness 变体，不塞 context window
- **背景**：自改进的 patch/variant 很多，全进 prompt 必爆。
- **选它**：变体存 `.huginn/` 文件，只在需要时读回。
- **代价**：需要额外的惰性加载单例与持久化逻辑；有目录漂移风险。

### D-3 不直接让 agent 改 engine.py 源码
- **背景**：自改进最彻底是让 agent 改代码，但我们用 spec 覆盖 + fallback。
- **选它**：让 LLM 改源码风险过高（一旦改坏无法自动回退）；spec 覆盖能拿到 80% 收益且回退成本 = 删目录。
- **代价**：比"改源码"的表达力低；复杂行为改不了（这是刻意的诚实边界）。

### D-4 CLAIM 超图用 n 元超边而非二元图
- **背景**：一个结论往往依赖"前提 A 且前提 B 且（前提 C 或 D）"。
- **选它**：二元图无法表达合取/析取语义；超边（[kg/hypergraph.py](../huginn/kg/hypergraph.py)）能表达 AND/OR。
- **代价**：超图比图复杂，挑战传播/环检测实现量更大。

### D-5 契约文档自动再生、不手写
- **背景**：手写契约一定会在代码 drift 后过期（今天那批"研究探索层"过期注释就是活例子）。
- **选它**：env/events/tools/routes 等由代码生成，永不漂移。
- **代价**：契约"全面"但**不解释为什么**——所以需要本文档补上设计论据那一半。

---

## 7. 诚实的边界（别人一眼容易高估、你接手要记住的）

1. **数学层 ≠ 全部接好**：sheaf 框架、simplicial Betti、persistence、范畴 functor 已接主循环（advisory）；
   但 `hodge_signature` 只是**图论近似**（β₁≈E−V+C + 度熵，非真 Hodge 分解，[topology_lens.py:227](../huginn/metacog/topology_lens.py#L227)），
   persistence_landscape、topology_protocol 仍是 research 层。
   **判断一个模块胆子以代码调用点为准，别以模块头 docstring 为准**（我们这次就因此踩过一次）。
2. **数学不是基底语言**：地基是 LLM 推理循环；数学是工具+审计层。
3. **朗兰兹是隐喻**：字面无朗兰兹数学。
4. **CLAIM/contested 是审计+提示**：不产生新科学事实。
5. **部分机制是 staging 理论稿**：如 `reward_design.md`（建设中的设计，别当已实现）。
6. **advisory 不阻断**：数学结果默认只影响判断，不强行阻断假设。

### 7.1 对标语：若要"认真学习朗兰兹函子性精神"，现状缺口在哪（附源码锚点）

下表是逐字核对源码后的真实状态，用于把"朗兰兹=隐喻"这句话落到可执行层面。每个状态都带文件行号，读者可直接验证。

| 朗兰兹元素 | 我们的对应物 | 核对后的真实状态 | 实质缺口 |
|---|---|---|---|
| **范畴 / 函子** | Category / Morphism / Functor + `verify_functor` | 有**真范畴定义**（[category_functor.py:53](../huginn/metacog/category_functor.py#L53-L120)）；四层验证含保交换图（[404](../huginn/metacog/category_functor.py#L404-L550)）。但主循环只调 `get_category` 贴一句"两域结构同构"提示（[hypothesis_loop.py:2687](../huginn/autoloop/hypothesis_loop.py#L2687-L2699)），**`propose_functor`/`verify_functor`/`transfer_hypothesis` 生产路径无调用**，仅模块自检/eval 用到 | 三个 category **结构同构**（[139](../huginn/metacog/category_functor.py#L139-L141)）；且验证/迁移函数未接主循环，真 functor 应用停留在自检，未进入决策 |
| **自然变换 / 相干性** | 多翻译路径结果应一致 | 无实现 | 无 coherence 检查；`transfer_hypothesis` 是文本替换且未接生产（[556](../huginn/metacog/category_functor.py#L556-L598)） |
| **保持结构的可检查判据** | "只在自然边界翻译、保持结构" | .omm 声明 + 唯一 cross-ref 确实保文本结构（[context_builder.py:211](../huginn/context_builder.py#L211)） | "保结构"未形式化成可检查资产；唯一的实例是 KB↔Memory 文本层 cross-ref |
| **对偶 / 谱侧（Langlands dual）** | 跨库比较 | 无每域对偶重表述 | 跨库比较是 ad hoc（文本层）；无经匹配"对偶不变量" |
| **局部 ↔ 全局** | sheaf / Čech H¹ | **框架真搭了** C⁰/C¹/C² + δ⁰/δ¹（[sheaf_cohomology.py:266](../huginn/metacog/sheaf_cohomology.py#L266-L336)）；但模块自承 constant sheaf 下真 H¹ 检测不到 pairwise 冲突，主力是 Layer2 proxy（[21](../huginn/metacog/sheaf_cohomology.py#L21-L32)） | 真 Čech H¹ 在 constant sheaf 下主循环常为 0；non-constant sheaf / monodromy 未实现 |
| **迹公式 / 全局恒等式** | 全局一致性校验 | 无 | 无 "path-sum = eigenvalue-sum" 这类的可验证全局恒等式 |
| **等变 / 表示理论** | 结构识别应不随视角旋转 | `hodge_signature` 是 β₁≈E−V+C + 度熵（[topology_lens.py:227](../huginn/metacog/topology_lens.py#L227-L293)），**无表示、无等变**；且其 β₁ **直接进 Darwin 打分**（topology_richness→停滞/棘轮判断，[cognitive_loop.py:685](../huginn/autoloop/cognitive_loop.py#L685-L707)），是**决策输入而非纯 advisory** | 需先接真 Betti（gudhi）才有讨论等变的基础；当前能进决策的只是"近似 β₁"这一维 |
| **元层（L-群）** | 库间函子作为被研究对象 | 无 | 无"把库间翻译当作应被证明/研究的对象"的三级结构 |

> **反向发现**：跨库结构同步并非不存在——`_sync_simplicials_to_kg`（[hypothesis_loop.py:2318](../huginn/autoloop/hypothesis_loop.py#L2318)）把假设图的超图命题写回 KG 成 hyperedge，是真实的数据层结构继承通道，与".omm 只在自然边界翻译"声明互补（都在自然边界、且保结构），不是矛盾。

> 读表须知：本表"状态"列每个论断都来自源码锚点，不涉及主观判断，欢迎读者用行号复核；我们这次就因相信模块 docstring 而非调用点，误判过一次。

---

## 8. 如何用本文

- 想理解**全局 why** → 读本文。
- 想了解**每层谁是什么** → 读 [architecture.md](architecture.md)。
- 想看**每个端点/工具/枚举的事实** → 读 docs/ 下的 `-contract.md` 契约文件（自动再生、不会漂移）。
- 想**上手改代码** → 读 [HOW_TO_READ.md](HOW_TO_READ.md)。
- 想看**旧的 over-review 备注在哪** → [INDEX.md](INDEX.md) 的状态标记。