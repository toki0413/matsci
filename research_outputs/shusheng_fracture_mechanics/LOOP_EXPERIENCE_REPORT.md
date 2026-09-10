# Huginn × 书生 长程自主科研循环：经验沉淀报告

> 面向 IJF 2026 综述《Outstanding issues and emerging frontiers in fracture mechanics》
> (杨卫 / 冯西桥 / 高华健)，将 7 个可计算开放问题转译为真实数值实验，
> 由书生(Intern-S2)驱动 Huginn 自主循环完成跨批次长程科研。
> 本报告不完全复述"结果"(见 `research_report.md` / `research_taste.md` / `cycle*.md`)，
> 而沉淀**方法 / 循环机制 / 品味提炼 / 工程问题**这四层经验。

---

## 0. 一句话结论

把 200+ 轮自主循环的经验固化后证明：**"研究品味"是可以被结构化提炼、语义化记忆、并反向长出可证伪新问题的**；
而这个能力不依赖某个特定大模型，而是由 Huginn 的**(提出 → 求解 → 反身 → 沉淀 → 再提出)**闭环承载，
书生在其中扮演"决策入场者"，agent 自身保持工具链与记忆的中立。这正是主办方要求的
"基于书生大模型 + 任意 harness + MCP + Agent Skills + 多智能体 + 长周期记忆"的工作流循环验证。

---

## 1. 循环架构与自主性

### 1.1 我们跑的不是单次脚本，而是"长程自主循环"

每一项(cycle)内部都是书生自己"观察 → 推理 → 行动"的多轮编排，不打断：

```
书生自主决策层(非干跑)
   ├─ 生成 goals / 假说 / 扫描配置 / 报告写作（client ≠ None）
   ├─ 选择并调用 probe_* 数值工具（沙箱执行真实物理计算）
   ├─ 生成自编实验码 → code_lab 沙箱运行 → 取回真实数值
   └─ 多角色复核：主研究员成文 + CriticAgent 反对立场复核
        │
Huginn harness 层(保持中立、不偏袒模型)
   ├─ 分层流式结算(分层判定, 防"被 encode 出正确")
   ├─ 层间重规划 + Pareto 剪枝(存留多证据非支配前沿)
   ├─ claim grounding 门禁(数值主张必须回流溯源, 防编造)
   ├─ 判断护栏(软提示引导语境感知决策)
   └─ 三记忆循环(persona / 会话 / 长程知识) + taste 语义 RAG
```

### 1.2 agent 的"独立性"如何守住

主办方要求基于书生，但我们同时坚持 agent 不绑死在某一个模型上(agent 不为书生设计)。做法：

- **对话层可插拔**：`client` 传什么模型，书生就长程驱动什么；`--dry` 走确定性综合，无 API key 也能整条管线跑通。
- **工具面中立**：probe_* 数值工具、code_lab、claim grounding、分层/Pareto/三记忆 都属于 Huginn 原生机制，
  与书生无关；书生只是"借用"这些通用能力。
- **品味库可迁移**：taste 沉淀层不要求特定 VL/偏模型，语义召回用自包含 ST/ONNX 编码器，换域直接复用。

结论：这是 **书生入场、harness 托底** 的工作流循环，二者职责清晰，agent 能力可跨模型/跨域复用。

---

## 2. 核心机制：把"提出问题"本身变成可训练、可记忆、可迁移的能力

### 2.1 七大类提出机制标签(taste taxonomy)

对 IJF 综述的每个开放问题，不只看"它是什么"，而反身回答"**它是怎么被提出来的**"，
并归纳为可复用机制标签：

| 标签 | 含义 | 断裂域实例 |
|------|------|-----------|
| `PARADOX` | 数学解与物理现实直接冲突 | ON1 界面裂纹互穿：ε≠0 时 K 场预言裂纹面穿透 |
| `ASSUMPTION_FAIL` | 某个理想化假设在现实边界失效 | ON2 脆韧/Rice-Thomson；ON4 Weibull 独立缺陷；ON6 动态禁区；ON7 K-dominance |
| `SCALE_BREAK` | 尺度耦合导致连续性断裂 | ON3 纳尺度缺陷容差 1/√a 失效；ON5 桥联层级结构 |

这份 taxonomy 就是"品味"的骨架：它把"顶尖学者为什么会问出这个问题"解码成若干可复用的发问模式。

### 2.2 从"机理"长出"新问题"(taste → new questions)

书生依品味自生成的白名单之外、可证伪、可在纯数值实验室落地的新问题，均带**明确预言 + 拟用实验**，例如：

- 机制 `ASSUMPTION_FAIL` → **Rayleigh 势垒陡峭度**：`-dln g/d(v/cR)@0.95` 是否随 vcR 单调下降？
  预言在 vcR≈0.6 出现极小值 → 拟用有限元动态断裂模拟扫 vcR。
- 机制 `SCALE_BREAK` → **泊松比对势垒陡峭度**是否引起 >±1% 波动？预言 <0.5% → 拟固定 vcR=0.95 扫 ν。
- 机制 `ASSUMPTION_FAIL` → **桥联增益饱和**：σ0/σy 从 0.3→0.7 是否线性？预言 σ0/σy≈0.5 拐点 → 内聚区模型扫 σ0/σy。

关键：这些新问题**不是**空想，而是复用了已沉淀的品味模式 + 上一轮真实数值缺口，形成闭环。

### 2.3 品味记忆跨批次积累(长周期记忆落地)

- 三记忆循环之外设立**品味专属语义 KB(taste_kb)**，一条"决策启发式 / 机理 / 新问题"入库。
- 用 **embedder 指纹**保护语义空间：ST(384, 多语言) ↔ ONNX 切换时自动重建，同空间则跨批次**继续积累**(44 → 104 条)，不误清库。
- 端到端闭合断点B：每次反身提炼前，`_recall_taste(...)` 召回已沉淀品味注入本次，
  保证 **新品味从旧品味上长出**，而不是每次从零发明。

---

## 3. 质量护栏：真实、可复现、不编造

- **全部数值来自真实物理计算**：probe_* 数学/内聚区/蒙特卡洛 + 书生成码沙箱执行，无一行编造。
  例：界面振荡 |ε| ∈ [0.0157, 0.0723]，纳缺陷临界 a* ∈ [0.30, 2.59] nm，Weibull 斜率 0.2483 vs 理论 0.25(R²=0.9996)，
  桥联增益至 20.4%，声学缺口 7.8%，KIC 门槛跨度 5.27(log)、门槛/塑性区比 46.6–47.1。
- **claim grounding 门禁**：任何"数值主张"必须能回流到某次实验结果；本轮批次 6 个 cycle 全部 pass，
  早期 `n_flaws` 因把配置参数当主张而误报 `needs_grounding` 的问题已修复(结果标量单独放 summary，不含配置列表)。
- **CriticAgent 对立复核**：每个 cycle 由审稿副体持反对立场挑刺(未报误差 / 跨条件强推 / 选择性使用数据…)，
  逼主研究员收窄措辞，而不是自说自话。

---

## 4. 关键工程问题与本轮根治(root-cause, 非打补丁)

| 症状 | 根因 | 根治 |
|------|------|------|
| taste_kb 每进程被清空，跨批次不积累 | `_get_taste_kb` 无条件按 source 清库 | collection metadata 记 embedder 指纹，同空间保留续积累，异空间才重建 |
| 语义编码静默降级浓/英文空间不一 | `_taste_embed` 被动依赖 store 的 `_st` 惰性加载 | 显式本地 ST 单例(本地快照, 不联网) + ONNX 兜底 |
| ST 权重下载失败(镜像 401) | hf-mirror Xet 协议 401 | `HF_HUB_DISABLE_XET=1` 传统通道重下，`local_files_only` 直读快照 |
| 批处理后台启动即崩 | nohup 环境丢了 numpy/scipy/openai | 逐个补齐到 `python` 解释器，`env` 前缀注入 API_KEY + HF 镜像再 nohup |
| claim 门禁误报 `n_flaws 配置值未落地` | summary 混入配置参数被当成主张 | summary 只放结果标量(斜率/R²)，不放配置列表 |
| course7 门禁 `needs_grounding`(-0.95) + S2/S3 数据全同 | `_dim_canon` 把 `vcR→barrier`，但 barrier 分支只用 ν 扫描 → 书生提的 vcR 假设空转、S2/S3 都落 nu | 为 vcR 建独立扫描维(真扫 v/cR∈[0.1..0.98] 的锐度)，补 `_DIM_VALUES`/`_dim_objective`；修复后门禁 `pass`、S1 单调锐度直接证伪"非单调"假设 |
| 书生成码因缺 matplotlib 整体回退 | 沙箱环境未装 matplotlib | `pip install matplotlib`；沙箱前置 `MPLBACKEND=Agg` 防无头崩溃 |
| 书生成码被拦"import typing 不在白名单" | `_ALLOWED_IMPORTS` 无纯类型标准库 | 两处白名单补 typing/typing_extensions/dataclasses/itertools/functools/collections/copy/decimal/fractions/operator(纯标准库无 IO/网络) |
| 书生成码"必须返回 dict / 需要 summary"被拒 | 书生常 `return {"Kc_K0": 1.4142}` 裸数值 dict 而非标准包装 → 真实数值被丢 | `_coerce_author_result` 宽宥"裸数值 dict": 纯数值标量键→objectives、其余进 summary，success=True，**不合成任何数**；纯字符串/不可序列化仍拒 |
| 书生成码运行时崩溃(KeyError:'materials' / SyntaxError) | 一二稿代码有 bug，被 schema 通过后自己执行失败 | `_AUTHOR_MAX_RETRY=2` agentic 重试: 失败时把真实 err(KeyError/SyntaxError/流程)回流给书生重修一版再执行，全败才回退白名单 |

---

### 4.1 方案2/倾斜：书生成码"带错修错"agentic 重试 + 走书生擅长路径(cycle16-22 实证)

书生成码的失败形态随修复逐层外移：依赖缺失 → import 白名单 → 返回契约(schema) → **代码本身的质量**。
这一层不再是 harness 能修的路，而是模型生成能力面。为此给 `_try_author_code` 加 agentic 重试：

```
第1次: 书生写代码 → sandbox_run 执行
  ├─ 成功 → 采纳(进入 objectives/Pareto/门禁)
  └─ 失败 → 把真实 err(如 KeyError: 'materials' / SyntaxError: ...)
          拼进下一轮 LLM user: "你上一版代码执行报错如下: {err}, 请只输出修正后的完整代码"
第2次: 书生带错修一版(可选, _AUTHOR_MAX_RETRY=2) → 再跑
全败 → 回退白名单扫描(不阻塞循环, 不伪造)
```

要点:
- **只回流真实 err**, 不喂模型伪造的"建议"——书生自己根据错误修代码, harness 只做执行器。
- **诚实边界不变**: 重试成功的数值仍是真实计算; 失败依然如实回退, 不因重试而放水 schema(IP/序列化/非 dict 依旧拒绝)。
- 这本质是把 Code-as-World 的 `CompareAndDiagnose→Δ→局部修订` 从"世界假说"层平移到"书生的实验代码"层: 每一次 err 都是一次可验证的 Δ, 迭代到预算(_AUTHOR_MAX_RETRY)耗尽即停。

> **cycle17-22 实证(倾斜)"**：
> - **重试预算调低**: `_AUTHOR_MAX_RETRY` 2→1, 书生成码失败一次即回退白名单扫描, 把 API
>   火力让给书生更擅长的路径(scan/probe/闭式核验), 不在一处代码上死磕两轮。
> - **失败形态漂移**: 6 轮观察到的 err 涵盖 `TypeError('float' not iterable)` / `KeyError('c1')`
>   / `ValueError(Material 0 not in database)` / `SyntaxError(括号未闭合/f-string unterminated)`;
>   书生成码全在重试 1 次后回退, 未恢复出可运行代码 —— 印证 **Intern-S2 当前能力画像**:
>   数值实验设计/解读强, 手写可运行实验代码弱。
> - **scan/probe 主导**: 倾斜后书生在 goal 显式引用 `probe_flaw(material)`/`probe_barrier(nu)`
>   等探针核验开放问题, 扫描覆盖 nu/bridge/barrier/material/vcR, 门禁一贯 pass。
> - **教训**: 别把模型往它不擅长的路径上硬推; harness 保持中立, 提供多条可证伪通道
>   (scan/probe/Symbolic/code) 让模型选最顺手, 各通道数值都经 claim grounding 门禁。

---

### 4.2 关键纠偏：从"预置领域内核"到"通用工具面"(泛化能力的分水岭)

跨域沉淀时我们曾纠结：每开一个新科学域，是否都要**先手写一批域内核**(Landau 自由能、
Dundurs 界面参数、Weibull 弱链…)，才能驱动深研？这看似"负责任"，实则把 agent 的泛化
能力做死成"查表式换域"—— 换个不相干学科又要从头写一遍内核，泛化无从谈起。

**纠偏结论(本轮定型)**：huginn 不给任何域预置物理内核，改为暴露**域无关通用工具面**，
让书生自主建模。具体落地(以 quantum_critical 第三域为证，与固体力学零族)：

- **通用科学仪器**：`integrate / root / minimize / curve_fit / ode` 等由 scipy 真实计算的
  原语，不走 ToolRegistry 的 fake-echo，能经 `resolve_diagnostic_tools` 被研究中台正确解析、
  调出真实数值(如 ∫θ³dθ=4.0、√2≈1.4142)。
- **书生成码为主**：`_try_author_code` 让书生亲手写 `def run(cfg)` 自主建模(Landau/Curie-Weiss/
  临界指数/相图拓扑全由书生决定)，code_lab 沙箱执行 → 数值进 trace 供门禁核对；失败带错
  回流修一版，预算 1，全败回退白名单自检(不伪造)。
- **不加预置内核**：占位实验仅做"工具面自检"(无物理内核的积分/求根)，证真仪器可用，
  不替书生解题。

> **泛化判断**：这套"通用工具面 + 书生成码"与断裂域(扫描/probe 主导)的本质一致——两者
> 都在验证**同一套 huginn 机制零改动搬到陌生域**。断言越少预置、越依赖通用原语，
> agent 越能跨学科复用；预置内核越多，越是"为这个域造专用电梯"，泛化能力越差。

---

### 4.3 架构红线：agent 不构建模型权重(研究 agent 机制/泛化的定位边界)

对照外部项目(Code-as-World / VibeWorlding-Gym 源码级)后定下的边界，明确本项目**不追求
"agent 让模型变强"，而专注"agent 机制本身能否跨域复用"**：

- **权重归外部**：agent(书生+harness)在运行中**不访问、也不应访问**底层模型权重。它产出
  的是决策轨迹、工具返回、门禁/验证器 reward 等**输入**；权重的更新(如 GRPO θ←θ+lr·∇J)
  只能由**外部训练器**基于这些轨迹/reward 完成。VibeWorlding 的 agent 同样不碰权重——
  它只产出 final_map+reward，权重由其 verl/GRPO 训练器改，两者靠数据管道(其 FileRPC +
  broker 桥)隔离。
- **泛化是主线且须可证伪**：核心命题 = "**同一套 huginn 机制零改动搬到陌生域**"。不能停在
  "跑通 3 个域"的口头断言，要用可证伪数值量化(跨域零改动复用度)。
- **中立与诚实优先**：不引入训练绑定，守住模型无关 + `--dry` 确定性 + 门禁零伪造这三条护城河。
  训练(若未来要做)只作为外挂在数据层的可选后端，永远不内联进 agent 推理路径。

> 附带教训(来自源码级复核)：top 项目的 paper/blog 架构图 ≠ 开源仓库实现。Code-as-World
> 的开源仓库只含**单轮 batch 评估**(evaluation.py)与**确定性仿真执行**(simulation.py)，
> 并无其所宣称的"iterative agentic discovery loop"。判断外部方法能否借鉴，须**读源码**，
> 不能只信架构图/文档。

---

### 4.4 泛化基线实测：跨域零改动复用度(cross-domain reuse score)

把"同一套机制零改动搬 N 域"从口头断言变成**可证伪数值**：`tests/test_cross_domain_reuse.py`
用**同一个** harness 机制(严格契约包装 / objectives 提取 / 门禁可落地 / 多证据)驱动 3 个
零族域(rigidity 自治轨迹式 / fracture 白名单扫描式 / quantum 通用工具面自检式)。

- **当前基线 reuse_score = 0.9167**(11/12)。fracture、quantum_critical 四个机制 stage 全
  满格；机制级 stage(objectives 提取/门禁可落地/多证据)在 3 域全部通过。
- **唯一扣分项(诚实 gap)**：老域 **rigidity** 不遵守统一 `{success,summary,objectives}`
  契约——它顶层 `import torch`, 返回裸数值 dict / 整数键逐族 dict(`exp_constraint_dimension`
  返回 `{0:{...},1:{...},2:{...}}`)。harness 用 `_coerce_author_result` 在边界能宽容其部分
  (X1 抽出 objectives), 但域自身不统一契约 → 严格包装 stage 只得 0 分。
- **意义**：这不是"造一个假满分"的表演, 而是暴露了真实的待对齐清单——**零改动复用不是
  无条件满格**, 老域/旧风格域需要做契约对齐(统一 `{summary,objectives}` 包装、扁平化
  objectives)才能从 0.92 → 1.0。这正是"泛化能力"要量化的东西。

---

## 5. 可迁移性与复用路径(这份品味不只在断裂力学成立)

- **跨域复用**：taste taxonomy(`PARADOX / ASSUMPTION_FAIL / SCALE_BREAK`…)自带非断裂类比例子
  (如稀薄气体→连续介质失效→分子动力学/格子玻兹曼)，可直接作为其他科学域的"发问词典"。
- **换域即换工具面**：把 probe_* 数值工具替换为新域的科学计算，其余(分层/门禁/Pareto/三记忆/taste/RAG)骨架不动。
- **给后续批次的配方**：
  1. 读综述/文献 → 用 taste taxonomy 反身标注"每个前沿问题如何被提出"；
  2. 抽取**可计算 + 可证伪**的子集 → 每问题一个独立证据分支；
  3. 书生全流程决策 + Huginn 中立托底 → 多轮自主循环；
  4. claim grounding 守"不编造"，CriticAgent 守"不虚浮"；
  5. 提炼品味 → 语义 RAG 入库 → 下一轮从旧品味长出新问题。

---

## 6. 面向主办方的三点定位

1. **工作流循环(核心交付已验证)**：完整验证"书生决策入场 + huginn harness 长程自主 + MCP/Agent Skills/多智能体/长周期记忆"的循环可自动秒级推进，无手动步骤。
2. **能力不绑定单模型(独立性)**：书生驱动、agent 中立，`--dry` 无 key 也能跑，跨域跨模型可复用。
3. **把"提出好问题"做成可积累资产**：taste taxonomy + 语义品味库 = 让模型不仅会解问题，还会**像好科学家一样提出问题**，且随批次越积越强(44→104 条)。

> 待办/开放项：`audit.score_usage / governance.external_verify` 属软门禁非落地项；
> 后续可把 taste taxonomy 固化为 Agent Skill 供任意域一键调用。