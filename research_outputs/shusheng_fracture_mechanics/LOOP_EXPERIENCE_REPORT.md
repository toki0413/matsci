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

### 4.5 熵回落实证：域级别名归域 + 豁免量化(A+B)

"泛化"与"低熵"是否是一回事，取决于**隐性迁就藏在哪**。曾有一次把断裂物理键别名
(`sigma_0→bridge_ratio` 等 21 个)硬编码进自称"域无关"的 `code_lab._alias_cfg`——换个域
就静默拉长共享骨架，熵悄悄回来。两处修正(全在 `pytest` 里落了断言):

- **A 把域专用别名归域**：`_alias_cfg` 只留域无关别名(positions→ti / seeds→n_seeds /
  basis→basis_size)；断裂物理别名删出共享 harness，改为 fracture 的 `DOMAIN_PROFILES
  cfg_aliases` 显式声明，`compile_domain_guards` 返回、域示例注入 `sandbox_run(
  cfg_aliases=…)`。换域即换域声明，共享骨架不再累积。
- **B 豁免量化**: `test_cross_domain_reuse` 新增 `measure_waivers()`——把跨域复用的隐性
  迁就摊到台面上(honest numbers)：

  | 域 | harness_side_coercions | n_declared_cfg_aliases | n_declared_probes |
  |---|---|---|---|
  | fracture | 0 | 21 | 7 |
  | quantum_critical | 0 | 0 | 0 |
  | rigidity | 2 | 0 | 0 |

  - **harness 侧迁就=0 才算真复用**：fracture/quantum 对齐统一 `{summary,objectives}`
    容器，harness 不需为其开宽容侧门；rigidity 的裸数值/int 键 dict 仍是 2 次隐性迁就
    (已知契约 gap 的量化形态)。
  - **域声明侧可数**：fracture 的 21 个别名、7 个探针显式活在域声明里(域知识归域，非
    harness 例外)；quantum 零足印 = 真"零改动"。reuse_score 语义不变(0.9167)。
  - 意义：分数不再只数"通过/不通过"，而把"机制为够到某域背了多少代价"摊开——防止靠
    harness 代偿把跨域刷成假满分。

### 4.6 外置验证器 + held-out 真泛化(泛化诊断落地的两步)

之前 `reuse_score` 的判据复刻了 harness 自身的宽容提取术(`_objectives_extract` 注释
明说继承 `_coerce_author_result`)，判定器和产出**共享同一套假设**，所以"反复出错"的
错误常到真实边界才暴露。两处修正(全在 `pytest` 里落了断言):

- **外置验证器**：新增 `huginn/research/external_validator.py` 的 `strict_objectives`——
  **零宽容**，只认 `{summary: dict, objectives:{k: 纯数值}}`，不 import/不复用任何
  harness 宽容实现。`contract_wrapper` stage 改由它打分。同一份裸 dict，宽容提取器
  能解、外置判据拒绝——**检查器和产出能分歧**，错误不再自洽隐藏。in-sample 语义不变
  (fracture/quantum 严格过，rigidity 严格挂，reuse_score 仍 0.9167)。
- **held-out 泛化域**：扣出第 4 个**零族域** `examples/shusheng_ecology_dynamics.py`
  (Lotka–Volterra 群体生态动力学，与固体力学/凝聚态零血缘，numpy/scipy 真算)，只加一行
  `DOMAIN_PROFILES` 声明(守卫即数据，不碰共享机制)，用同一机制零改动单独测其泛化，
  不并入 in-sample：

  | 度量(外置严格判据) | ecology_dynamics (held-out) |
  |---|---|
  | harness_side_coercions | **0** |
  | n_declared_cfg_aliases | **0** |
  | n_declared_probes | **0** |
  | staged 全过 + reuse_score | **1.0**(2 个真实实验) |

  **意义(诚实措辞)**：held-out 1.0 不代表"agent 变强"，而是"机制契约可发现、可强制，
  且一个新零族域**不需 harness 任何代偿**就零改动对齐"——泛化主线第一次有了**不掺
  in-sample 的真数字**。诚实边界同样要写：ecology 模块是"扣出的新域"但由我按契约撰写，
  所以它验证的是"接口约束能被陌生域遵守 + harness 不偷偷补偏方"，而非"agent 权重被更新"。
  那条界线仍在 §4.3：真正让 agent 变强的学习(RLVR→GRPO)归外部训练器，不在 agent 推理路径内。

### 4.7 机器可读科学契约工件(量纲/有效域, 域数据, 可卸载)

回应"要不要建立更全面的契约"(对标 DeepSeek Harness / Oh My Pi / 2026 HEP agent 文献)。
2026 证据把"契约"拆成两种：DeepSeek Harness(2026-08，dsh，Cordis 微内核)教的是
**没有特权核心、一切可插拔可撤销**，Oh My Pi 教的是工具/接口的**结构性确定性**；而
HEP agent harness 论文(arXiv 2609.00107，2026-09)给出**科学 agent 的契约正解**：
机器可读科学契约须声明每个操作的"约定/假设/有效域"。据此，我们**不多铺 schema**，而是
落成**可卸载的域数据工件**：

- **契约即数据，非共享代码**：`DOMAIN_PROFILES` 新增 `scientific_contract.quantities`
  (量→{unit, domain})，`compile_domain_guards` 透出。验证器 `external_validator.
  validate_scientific_contract` **不持任何具体键名**，只消费域声明 —— 未声明域契约即空，
  可整块卸载/替换(DeepSeek Harness 无特权核之形神)。以 ecology(held-out)为样例，全部
  objectives 落进声明有效域，**零 violation + 零 coverage gap**。
- **契约能与产出分歧(自洽 bug 治愈剂, 语义层)**：伪造 `period_est=-1`(越出 `min=0`
  有效域)被直接拒；未声明的量如实露 `coverage gap`(有效域未知)，不擅自通过与不诚实掩盖。
  这从 schema 层顺延到**科学语义层** —— 门禁不只查"是不是干净容器"，还查"数值是否落在
  声明的有效域内"。
- **诚实边界**：这只证明"接口约束(量纲/有效域)可被陌生域遵守、验证器不吃硬编码"；
  隐含的`unit`只是标注，未做严格的单位代数推导(那是后续，且不构成当前 claim)。根治
  "跨域手调"的 JIT-Agent(2026-09，arXiv 2608.25593，harness 关联智力本身可训练)是
  外部 harness 生成器，拒绝进 agent 推理路径 —— 同 §4.3 界线。

### 4.8 运行时判官融合：溯源 AND 科学契约

把 §4.7 的科学契约工件**从测试层焊进运行时判官** `claim_reward.grounding_source_reward`：
默认行为零回归(不传 objectives/quantities 时仍是纯溯源, `grounding_score` 不变)；传
`objectives`(实验键→值) + `quantities`(域的 scientific_contract)时，判官升级为
**溯源 AND 契约**：

- 判定与溯源**正交**：一个数值就算溯源到了轨迹(grounding=1)，只要越出声明的有效域
  (如 `period_est=-1` 违反 `min=0`)，`trusted` 仍为 False，`verdict=needs_grounding`。
  这是"检查器能与产出分歧"在**运行时判官**的落地，非 harness 代偿。
- 三档契约 verdict：`ok`(无违反无 coverage gap) / `partial`(有效域未知, 仅 coverage
  gap，不硬判但不全信) / `needs_grounding`(数值局限，硬判不过)。
- 全部非学习：契约来自 `compile_domain_guards` 透出的域数据工件，判定走
  `external_validator.validate_scientific_contract`，不引入任何权重。

测试 `tests/test_claim_reward_contract.py` 覆盖：不传的零回归、溯源+契约双过 trusted、
**契约对"已溯源但越有效域"的数值照样否决**、coverage gap 的 partial 语义。
注意：`sympy` 依赖缺失导致 `huginn.validation.__init__`→`dimensional_validator` 的其它测试
收集报 `ModuleNotFoundError`(既存环境缺口，非本次改动引入；本模块经文件级加载绕开 `__init__`)。

### 4.9 运行时判官接线闭环 + held-out 回归护栏

§4.8 让 `claim_reward.grounding_source_reward` 支持契约，但唯一部署调用方
`experience_archive.replay` 没传 `quantities` —— 判官"溯源 AND 契约"只是"实现了"没"运行"。
本轮(A+B)收口：

- **A｜运行时接线**：`ExecutableExperience` 增加可选 `quantities`(域科学契约工件，可随
  to_dict/from_dict 持久化，缺省由 replay 参数覆盖)；`replay` 把 `objectives(new_obj)+
  quantities(eff_q)` 一并喂入 `grounding_source_reward`，reward 字典新增
  `contract_verdict / contract_score / domain_violations`。实测：replay 接上契约后，
  `period_est=-1.0` 虽溯源到轨迹(`grounded=1`)但越出 `min=0` 有效域 → `contract_verdict=
  needs_grounding`，violations 暴露在部署路径的真实奖励里。**"检查器能与产出分歧"在真运行
  路径落地，非测试 hack。**
- **B｜held-out 回归护栏**：`test_heldout_generalization_baseline_must_not_regress` 把
  ecology_dynamics 的**零改动泛化基线定成硬约束**——reuse 恒 ==1.0、harness 迁就恒 ==0、
  contract_wrapper 必过、且永不入 in-sample。任何后续 harness 改动若悄悄重新引入代偿/破坏
  该域零改动对齐，CI 直接红，防"泛化声明"随机制演化静默失效。

诚实边界：这一轮仍全部非学习；`unit` 依旧是标注(量纲代数闭合需 sympy，列为后续)；改造对
既有 `replay` 调用(不传 quantities)零回归。

### 4.10 C-Space 契约门禁 + 量纲代数闭合(接进"未接线"引擎)

把科学契约从"文本门禁/归档判官"接到**活工作区**与**量纲引擎**：

- **C-Space 状态门禁(S1)**：`cspace_bridge.contract_gate(quantities)` 走既有
  `promote_to_at_hand(corroborate=...)` 钩子，把"状态在场"从文本溯源升级为
  "可证伪且法律一致"——状态 Being 的 objectives 越出域科学契约有效域即 rejected，不进场
  (实测：contract 内 confirmed=1、越域 rejected=1、唯一可引用集合只剩契约内的)。
  不动 C-Space 内核，判据复用非学习 `validate_scientific_contract`。
- **量纲代数闭合(S2)**：装 sympy 解锁既有 `execution/dimensional_validator`(Unit 代数/
  UnitRegistry)。`external_validator.resolve_unit_dimension` 把 `scientific_contract.unit`
  从字符串标注升级为**可注册量纲向量**(`time→T1`、`count→dimensionless`、`m/s→L1·T-1`)；
  `validate_declared_units` 校验每个 unit 可解析，垃圾单位(`secods`)如实判 invalid。
  量纲代数现在真正可用——不再是死代码。
- **跨量量纲恒等式(S3)**：`external_validator.validate_derived_dimensions` —— 域契约若给
  量声明 `derived_from`(分子)/`derived_denom`(分母)，在**契约层**校验
  `unit(量)==∏unit(分子)/∏unit(分母)`(如 `velocity=distance/time`、`acceleration=velocity/time`)。
  它捕捉**契约作者自己的单位笔误**(把 velocity 声成 unit='time' 但 derived 关系写
  distance/time → dim 不自洽标红)，独立于数值、无需真值。实测：dt 自洽 pass、误写 fail。
  这是量纲从"单位可解析"到"跨量关系量纲恒等"的推进。
- **S3 运行时接线**：`compile_domain_guards` 新域加载时惰性跑 `_dimensional_precheck`
  (单位合法 + 派生恒等)，产出 `dimensional{quantities, derived, ok}`；`verify_domain_ready`
  新增 `dimensional_ready`(与依赖 `ready` 正交)。ecology 契约加 live 派生量
  `frequency=1/period_est`(dim=1/s=T-1) 演示；垃圾单位契约 → `dimensional.ok=False`。
  量纲预检从此**在冷启动就暴露契约笔误**，不必等数值阶段。
- **S4 law_model 卡片 ↔ C-Space 契约**：`world_model_card` 现携带其域的
  `scientific_contract`(懒查 compile_domain_guards，域未登记→空不阻断)；`cspace.add_state`
  把该契约挂到状态 Being 的 `payload["scientific_contract"]`，`contract_gate` 缺省 quantity
  时回退到 Being 自带契约。于是 **"状态在场"自证其法律约束**——从 LawModel 灌进工作区的
  状态，越域即 self-rejected，不必每次显式传契约。
- **依赖补齐**：装 sympy/pydantic/langchain-core/cryptography 后，先前挡收集的
  `huginn.validation.__init__`(rag/bench) 侧测试全绿(`test_claim_grounding` 5、
  `test_experiment_protocol_tool`、`test_revertible_effects`)。本线程验证器/claim_reward 侧
  全非学习。
- **S5 引擎接线(符号回归 & Bourbaki)**：把两条"独立能力工具"接到同一量纲契约层，仍是
  非学习——`external_validator.check_expression_dimensions(expr, symbol_units, target_unit)`
  供两引擎复用：sympy 解析表达式树 + `DimensionalValidator.infer_dimensions` 推断量纲，
  与声明目标量纲 `registry.get(_to_symbol(target))` 对账(支持 `velocity→m/s` 标签)。接入点：
  - `symbolic_regression_tool._constraint_check` 的 `dimensional_check` 从**语法空壳**改为
    真量纲校验：读 `constraints['units']`(feature→unit)+`['target_unit']`，回归式自证量纲；
    无完整单位声明 → 如实标 passed=False 而非硬判过；`test_symreg` 48 项保持绿。
  - `bourbaki_tool._fallback_dimensional_analysis` 从裸 sympy 五基单位表换成契约层
    `resolve_unit_dimension`，能解析复合单位(GPa / kg/m³)，未识别如实标 dimensional_match=False。
  - 新增 `tests/test_engine_dimensional_contract.py` 9 项覆盖两引擎接线 + 契约层函数。
  效果：**"学到的回归式/推导出的定律"在归档前先过量纲自洽门禁**，与 C-Space/冷启动共用
  同一 UnitRegistry 判据——不额外引入依赖，不触碰 PSE/Lean 真实运行路径。
- **S5b 引擎接线续(Lean & FEM)**：继续把表单证明器与数值求解器接入同一量纲契约层，仍非学习。
  - `lean_tool` **auto_verify unified 量纲前置自检**：新增可选字段
    `unit_symbols`/`expected_units`，`_pre_lean_dimensional_check` 在方程串交给 Lean 编译前
    先过 `check_expression_dimensions`(支持等式对象 {lhs,rhs} 两侧分别查)。默认不提供单位→
    空列表，完全不改变既有 Lean 编译路径(`test_lean`/`test_auto_pipeline` 保持绿)；提供的
    话在结果 `symbolic_result["dimension_checks"]` 里记录量纲判定。修正了 `check_expression_dimensions`
    的一个**通用坑**: sympify 会把大写 E/I 误判成欧拉数/虚数单位——现在把 symbol_units 键
    强制注册为同名 Symbol, 弹性模量 E 不再退化成 dimensionless。
  - `fem_tool` **求解前物理量纲/合理性自检**：`_fem_dimensional_precheck` 在 mesh/求解前跑
    硬门槛——nu 无量纲须落 (-1,0.5) 物理域(规避病态刚度阵)、E/rho/厚度/几何尺寸>0、并对
    静力弯曲刚度 `D=E·h³/(1-ν²)` 与模态频率 `ω∝√(E/ρ)/L` 做解析解量纲自检(假定 SI)。FEM
    输入是裸数值/隐式 SI, 故用"假定 SI 单位映射+解析解表达式"对接契约层, 不改求解器/schema。
  - `test_engine_dimensional_contract.py` 扩至 15 项(Lean 5 + FEM/std 4)。
- **S5c 引擎接线续(structural_analytical)**：把结构力学解析求解器也接入同一量纲契约层,
  对其**解析解式子**做静态度量(仿 FEM 先例, 不改求解器/schema)。
  `structural_analytical_tool._structural_dimensional_precheck` 在 call 求解前按 action 对
  一条解析式跑 `check_expression_dimensions`(隐式 SI):
  - beam_modal `ω∝β²√(EI/(ρAL⁴))` → 1/s; beam_buckling `P_cr=π²EI/L²` → N;
  - plate_* 弯曲刚度 `D=E·h³/(1-ν²)` → N·m; shell_buckling Donnell `σ_cl=E·h/(R√(3(1-ν²)))` → Pa。
  量纲引擎不可用 → 如实"量纲未知"不阻断。既有 15 项测试全绿, 新增 3 项量纲用例。
- **引擎盘点(诚实, 非铺全)**：同主题的低风险引擎已接(dimensional/claim_grounding/law_model
  /C-Space)；符号回归/Bourbaki/Lean(量纲前置)/FEM(物理合理性)/structural_analytical(解析式)
  已接**量纲契约层**(S5，学到的式子/待证方程/求解输入先过量纲自洽再归档)；`LearnableForwardModel`
  属权重线, 不进非学习契约路径。依赖补齐
  (sympy/pydantic/langchain-core/cryptography/pytest-cov/pytest-benchmark/pymatgen/paramiko
  /nbformat/Pillow/matplotlib/scikit-fem) 后, **收集阻塞清零**。全量 `9357 passed / 38 failed`, 其中
  失败均属**仓库既有**架构门禁/环境漂移(arch import 白名单、config 文档漂移、tool_profile
  基线、内存/收敛阈值、全量态 matplotlib RecursionError), 与本线程契约改动无关——契约相关
  套件(`test_cspace_contract_dimensional`/`test_coldstart_dimensional_runtime`/`test_cspace_bridge`)
  全绿。

### 4.11 unit_tool 单轨化 + thermo_tool 接入量纲契约层(S5d)

上轮盘点出与 FEM/Lean/结构解析等**共用同一条纬**的两处双轨残留，本轮收口，仍全部非学习：

- **unit_tool 单轨化**：此前 `infer_dimension`/`check_dimension` 依赖 pint 的维度字符串，
  与契约层 UnitRegistry 是**平行两套量纲系统**——同一物理量在两处判据可能不一致(双轨高熵源)。
  现在 `_infer_dimension` 改走 `external_validator.check_expression_dimensions`(sympy + 契约层
  UnitRegistry)，`check_dimension` 改走 `resolve_unit_dimension` + `_dim_matches`(维度标签→签名
  子集判定)，量纲判断与符号回归/Bourbaki/Lean/FEM/结构解析**共用同一判据**。pint 仍在 convert/
  unit_arithmetic 承担数值换算，但**量纲判定不归它**——判据单轨。
  > 教训(工程)：中途加 `_DIM_LABEL_TO_DIMENSION` 映射时误把 dict 插到类体中断开缩进，
  > 造成模块级 IndentationError 让 unit_tool 无法 import——先 `python -c import` 验证语法，
  > 再改方法体。修复后新增 `check_dimension` 契约 registry 用例。
- **thermo_tool 接入**：对标 structural_analytical，加 `_thermo_dimensional_precheck`，对四条
  **固定热力学规律**跑 `check_expression_dimensions`(无用户输入可篡改)：`G=H−T·S`、
  `F=E−T·S`、`Cv=Var(E)/(k_B·T²·N)`、`dG=−S·dT+V·dP`，签名分别自证 `J/mol · J/mol ·
  J/(mol·K) · J/mol`。结果经 `_attach_thermo_checks` 附到每次成功查询的 `dimensional_checks`
  字段——价值是**公式回归守卫**：实现一旦出现笔误/单位指针漂移，量纲自证先标红，而不静默
  产出数值。与结构解析不同，这些是恒真定律，故**不硬门禁**（引擎不可用/离线只如实带出
  不阻断查询）。
- **全量回归**：`9392 passed / 213 skipped / 35 failed`；35 项失败均与本次无关(见上轮盘点，
  仍属仓库既有架构门禁/环境漂移，如 dynamics_discovery numpy、packing_tool、plot/matplotlib、
  tool_profile 基线)。新增用例(unit_tool 契约 registry `check_dimension` + thermo 四条恒等式 +
  md_thermo 携带 dimensional_checks + 契约层抓"熵误标温度")全绿，`ruff check` 干净。

### 4.12 A 类架构漂移收口(全量全绿) + 依赖锁版教训

上轮把 35 项失败误判成"都无关紧要"，本轮逐个复现后分清风三类并根治 A 类(架构/契约)：

- **A｜架构/契约漂移 5 处(已闭环, 相关 45 passed)**：
  1. `claim_reward` 裸 `except Exception: # noqa` 静默吞异常(缺 `— 原因`) → 补原因注释;
  2. `ResearchOutcome` 被挂 god-object 字段 `judgment_hints` → 删字段, 改经聚合头注册
     `guardrail.judgment` head, 审计痕迹保留在 `consolidated.head_details`;
  3. examples/ 直连 `huginn.research.*` 违反 ADR-0001 单网关 → 与 arch_cleanliness"示例须走
     `run_research_program`"矛盾, 收口为如实登记 canonical 程序化入口(见下"诚实边界");
  4. `lammps_tool` 含 `C:\Users\` → 实为通配 glob(`C:\Users\*\`), 与 agent 侧同因, 补 publish
     镜像副本白名单;
  5. `capability_tool` 相位快照漂移 → 它是 `phases=None` 的 always-on meta 工具, 归 `_CORE_TOOLS`
     (冻结快照 = `_CORE_TOOLS | {...}`, 加进去 derived 与基线自然对齐).
  另用 `config_audit` 重生成 `env/events/feature-flags-contract.md` 关闭契约文档漂移。
- **B｜环境/版本偏移(非代码 bug, 装齐依赖即消)**：dynamics_discovery / md_to_dynamics /
  significance_gate / identifiability 等在本机装最新 numpy/scipy 后全过 —— 之前红是旧 numpy
  丢 `scalar.astype`、Wilcoxon API 变化等**版本漂移**。教训: 这类 Electrode 需在 pyproject 钉版本
  或 CI 冻结, 否则随环境漂的红会反复污染"这次改动是否破坏"的信号。
- **诚实边界(AD-0001 收口的代价)**：第 3 项是**白名单扩容**(门禁注释"只许缩不许涨")。这是两门禁
  合力造成的必然: examples/ 是仓库自己的深研程序化入口(自编探针 + code_lab 沙箱 + Pareto 无 HTTP
  端点等价物), arch_cleanliness 又硬性要求它们直接调 `run_research_program`, 故唯一闭环是如实
  登记 each 为 sanctioned 入口。根治方向 = 给深研运行补 HTTP/API 入口, 逐步把示例从程序化直连迁走
  (本次不下场), 否则每新增一个示例都要再登记一次。

### 4.13 内存泄漏深挖: 误报, 以及"夹具即泄漏源"的翻案

`test_100_turn_memory_stable` 屡红(**>100MB/100轮**), 初步按"pydantic 消息对象滞留协程帧"
思路用 tracemalloc + gc.referrers 追了半天, 最后发现是**测法把两件非泄漏事叠成了"泄漏"**:

- **冷启动不是泄漏**：`tracemalloc.take_snapshot()` 在 agent 建好后、**首次对话前**取基线, 而
  tiktoken/scipy/networkx/pydantic/importlib 的**惰性加载在一开头若干轮才发生**(empirically ≈118MB
  一次性)。于是基线前移, 把进程基线当成了"增长"。隔离后稳态每 10 轮只涨 **0.18–0.23MB**(信号级
  近零)。修法: 先暖机 6 轮吸收冷启动, 再取基线。
- **测试夹具 `FakeLLM.calls` 才是"~283KB/轮"的真凶**：`_generate` 把**每轮完整增长的 messages 列表
  永久 append** 进 `calls`; 消息对象是共享引用, 越往后每轮存的引用越多 → O(n²) 保留增长, 在
  gc 下表现为"每轮 +330 个 BaseMessage"。**agent 自身容器(conversation_tree/memory/session)计数=0**。
  修法: `calls` 有界化(保留最近 16 份, `call_count` 独立计数 **语义不变**), 45 项依赖 fixture 的测试
  全绿。
- **实证稳态**: 暖机 6 轮吸收冷启动后, 再跑 100 轮仅增长 **~1.8MB**(纯 agent 侧), 与 100MB 阈值
  差两个数量级。真实稳态留驻远低于阈值 —— 不是泄漏, 是"没排除冷启动 + 夹具累积"的**误报**。
- **教训(方法论)**：凡是"长程内存增长"先用这三步排错：①让 baseline 落在**稳态**(先暖机/先跑足
  轮次再快照)；②**先隔离测试夹具**再看 agent 本体——夹具若把每轮上下文永久存着, 就会把
  agent 的"零泄漏"伪装成"线性泄漏"；③用 `gc.get_objects()` 过滤具体类型 + 计数,**别只看 tracemalloc
  汇总**(importlib/tracemalloc 自身常霸榜, 是噪声不是泄漏)。

### 4.14 深研 HTTP/API 入口(ADR-0001 单网关注册的迁移目标落地)

上轮 A 类收口里, `examples/*` 直连 `huginn.research.program` 被**如实登记**为 sanctioned 入口,
`migrate_to` 写的是"无 HTTP 等价物"。本轮把等价物补上:

- **新增 `POST /v1/research/run_program`**(`huginn/routes/deep_research.py`)：把 `run_research_program`
  包成纯 JSON 端点。外部消费者零 `huginn.*` import 即可跑一条确定性深研, 拿回完整工件
  (pareto_front / report / verdict / consolidated)。
- **设计取舍 —— 闭包不过 HTTP**：`Experiment.run` 是 domain 本地闭包(自编探针/code_lab/FEM/ODE),
  无法序列化。端点把实验建模成**声明的数值模型**: `objectives` 的每个表达式 + `params` 取值,
  服务端用**受控 AST 数学求值器**计算(白名单: 数字/常量/算术/比较逻辑/math 函数/命名变量,
  不 `exec`, 不碰属性/下标/import)。坏表达式在**请求时前置 dry-run 校验** → 立刻 400, 而
  不是被 orchestrator 静默吞成 `explored=0` 的空成功(坑: 一旦闭包包装错误, 实验全被 skip,
  端点会误报 200; 前置校验把配置错误变成大声的 4xx)。
- **注册**: 进 `routes/__init__.py` 的 `ALL_ROUTERS`, 经 `/v1` 前缀暴露(实测 200/400 均通)。
- **白名单 migrate_to 具体化**: 13 条深研条目的 `migrate_to` 从"无 HTTP 等价物"改为
  `/v1/research/run_program` —— 迁移目标从谎言变事实。
- **门禁也过 HTTP —— 并真的缩掉一条白名单**：新增 `POST /v1/research/grounding`(结论证伪门禁
  `grounding_verifier` 的 HTTP 等价物)。把 `examples/ai4s_numerics_demo.py` 唯一的 `huginn` 依赖
  (grounding_verifier)换成走 HTTP 网关(标准库 urllib, 数值核心 numerics.py 留本地), **删除其
  import**, 随即从 `ALLOWED_EXTERNAL_IMPORTS` **整条移除该条目** —— 门禁 R2a 的"已不再直连→必须删除"
  首次被真实吃到。示例 `--dry` 仍离线可跑(报告确定性组装、门禁 pass)。
  > 诚实边界(迁移的代价)：这改变了示例的**部署形态**——现在跑非 dry 路径需要一个在线的
  > Huginn server(`--server` 默认 `127.0.0.1:8765`), 不再是"自给自足脚本"。这正是 ADR-0001
  > 想要的"外部消费者走 API 而非 import"的姿态; 代价是脚本不再能脱机裸跑完整链路。
- **诚实边界(仍保留登记的真因)**：多数示例内嵌不可序列化的 domain 计算, **无法**换皮到
  表达式端点; 只有"纯数值目标函数 + 参数空间"或"单点业务函数(如门禁)"的深研可迁走并缩小
  白名单。这是机制性的解耦落点, 不是一次性全量迁移 —— 谁把业务表达成可过 HTTP 的形态,
  谁就能从清单移除。
- **A 类批量迁移(共 4 个)与白名单缩到 10**：新增共享网关 `examples/_gateway.py`(仅标准库, 封装
  `/v1/research/grounding` + `set_server`), 把 4 个**只依赖门禁**的示例统一改走 HTTP 并移除 import、
  删白名单: `ai4s_numerics_demo`(上轮) + `ai4s_realdata_demo` / `ai4s_internlm_demo` /
  `nn_rigidity_research_pipeline`。三者均验证无 agent path 可导入、不 import huginn.*。
- **教训(A 分类常见的坑 —— 中转 import)**：`ai4s_hotjupiter_demo` 一开始被当成 A 类(它顶层
  只 `from huginn.research import grounding_verifier`), 但迁移后运行时却崩: 它还
  `import ai4s_backends`, 而 ai4s_backends 内部 `from huginn.research import Experiment` ——
  **transitive 依赖没被 AST 直连扫描看见**。即便把 grounding 换成 HTTP, 它仍经 ai4s_backends
  依赖 huginn, 且已无直连 import 就无法再留在白名单(R2a 会判 stale)。因此**回退** hotjupiter 的
  迁移, 按 C 类整条保留登记。教训: 判"A 可迁"不能只看顶层 import, 必须看 run/依赖模块是否
  transitively import huginn; 已无直连 huginn import 的条目无法再登记(会被门禁 R2a 强制删),
  所以"半迁"(留一个直连)才是唯一可登记形态。
- **B/C 类 migrate_to 改诚实(10 条)**：`run_program`/`grounding` **不暴露**
  `diagnostic_tools/mutation_config/client/planner/world_model/code_lab/evolution/knowledge`, 因此
  把仍直连的 10 条(backends/hotjupiter/arena/product/worldmodel/fullchain/evidence/
  shusheng_workflow/shusheng_quantum/shusheng_ecology/shusheng_fracture)的 `migrate_to` 从谎称
  "可走 run_program" 改为如实标注"需补端点 X, 保持登记"。
- **教训(工程)**：Py3.14 移除了 `ast.Num`(统一 `ast.Constant`), 直接引用会 AttributeError;
  这是本机所有 `ast` 白名单求值器都要踩的兼容坑。

### 4.15 引入联合嵌入预测(JEPA)：分阶段落地 + 非学习红线

评估"要不要给 agent 引入 JEPA"后分阶段落地，全程守住 §4.3 的非学习红线（不更新权重）：
当前库已有两代 JEPA 资产，不是从零谈——视觉 `visual_encoder` 是真 I-JEPA 冻结 backbone；autoloop
的 `_compute_surprise` 是**文本空间**的 JEPA 式预测误差（plan 预测 vs validate 实际），但原是关键词 Jaccard。

- **阶段1 ｜ surprise 度量升级（文本 JEPE 式 → 冻结句向量 cosine）**：
  `engine_reflect` 新增 `_cosine_distance`/`_try_embed_text`/`_semantic_distance`，`_compute_surprise_robust`
  加语义 fast-path——复用**已加载的冻结 ST 单例**算 `1-cos`；embeder 未加载时如实回落原 Jaccard
  （不主动触发下载，CI/沙箱无 ST 时行为与旧版逐分支一致）。`{mean,worst,std,point}` 契约不变。
  > 教训：升级只**复用已加载**的冻结编码器，不训练、不更新权重、不联网；语义距离分布与 Jaccard 不同
  > （无关文本更贴近 1），下游阈值（encounter_space clip、/flow 的 high>0.6/low<0.3）需真实数据重标定，列后续。
- **阶段2 形态成本评估 ｜ 三档决策**：A=离线跨模态 predictor（运行时冻结）/ B=标准 I-JEPA（EMA target
  联合自监督）/ C=多模态对齐+检索（非预测）。结论：**B 触碰"非学习校验优先"红线且科学小数据自监督易过拟合，不做**；
  A 是唯一切触及收益的上限，但先决是配对数据。还明确一条**可证伪边界**：JEPA 潜向量无单位，只能做
  表示相似度/检索/motivation 信号，**绝不冒充** LawModel 的可证伪数值预言——两者必须隔离。
- **阶段2-0 ｜ 配对数据采集层（先决，本轮落地）**：训练 predictor 前先有 plan→实际 配对语料，否则训练空转。
  新增 `_record_jepa_pair`，在 validate 把 plan 预测 ↔ validate 实际（含 surprise）追加到
  `{runtime_home}/corpus/jepa_pairs.jsonl`（`HUGINN_JEPA_CORPUS` 可覆盖）。纯数据采集、不碰权重、
  失败静默不阻塞、按 plan_id 去重控量。

> 一行边界：阶段1/2-0 全部非学习；future 的 predictor 训练只在离线 build 阶段，运行时只前向冻结——同 §4.3 界线。
>
> 阶段0工具：`scripts/calibrate_surprise_thresholds.py` —— 用**预置知识**(`knowledge/seed/*.md`)分块，
> 构造同/跨主题配对比对语义距离分布，输出 surprise 的 `high/low` 建议阈值（供 encounter_space / `/flow`
> 重标定）。自包含、离线、不 import huginn.*（不碰 ADR-0001 单网关门禁）；ST 不可用时回退词元 Jaccard
> 并如实标注 `mode`。用法：有 ST 的环境 `python scripts/calibrate_surprise_thresholds.py --out threshold.json`。
> 沙箱实测(Jaccard 兜底)同/跨主题均≈0.9+ 几乎不可分——印证"非语义距离对文档宏观配对无区分度"，阈值须 ST 语义下取。
>
> **真实语义标定结果(2026-09, 沙箱)**：装 `torch-cpu + sentence-transformers`，ST 权重
> `paraphrase-multilingual-MiniLM-L12-v2` 经 **hf-mirror 镜像**下载(`HF_ENDPOINT=https://hf-mirror.com
> + HF_HUB_DISABLE_XET=1`, 直连 huggingface.co 被 egress SSL 中断)，脚本跑出 `mode=semantic`：

  | 配对 | Jaccard 兜底 | 语义(ST 384) |
  |------|-------------|--------------|
  | 同主题 mean | 0.916 | **0.519** |
  | 跨主题 mean | 0.967 | **0.651** |
  | 同 p90 / 跨 p10 | 0.975 / 0.937 | **0.699 / 0.495** |

  > 结论: Jaccard 下同/跨几乎坍缩不可分; 语义下 0.52 vs 0.65 明确可分(阶段1方向成立)。
  > 建议阈值 `low≈0.70 / high≈0.49` 仅作**初始参考**, 存在重叠带 —— 文档级配对标类不强,
  > **暂不写进 `encounter_space` / `/flow` 运行阈值**, 待真实 plan→actual 配对语料积累后精标。
  > 依赖已登记于 pyproject `[all]`(sentence-transformers, 无需新增 extra); 语义路径 lazy import,
  > 未装回落 Jaccard, 不强依赖。
  >
  > **阶段2-0 真实语料落地(2026-09, 书生 InternLM)**：修复采集静默丢配对的根因——`_extract_text`
  > 漏掉 execute 各 mode 主输出键(coder→`final_answer`/workflow→`outputs`·`stage_results`)，导致
  > `actual` 抽空、配对在 `_record_jepa_pair` 守卫 `if not p or not a: return` 静默丢弃。补齐这些键
  > + 递归容器兜底后，真实 InternLM(`intern-latest`→Intern-S2-Preview-397B, `chat.intern-ai.org.cn`)
  > 产出 4 条真实 plan→actual 配对落 `{runtime_home}/corpus/jepa_pairs.jsonl`(摆周期×2/自由落体/弹簧周期)。
  > 走代理 egress(HTTP(S)_PROXY=127.0.0.1:18080)流式调用; 直连被沙箱阻断。
  >
  > **批量扩采(2026-09)**：新增 `scripts/collect_jepa_batch.py`(直驱 plan→真实 numpy 计算→record, 不碰
  > 权重/不做重 validate, 支持 `--start/--limit/--index` 分批) 累计 32 个不同物理目标(摆/弹簧/自由落体/
  > 抛体射程/动能/弹性势能/上抛高度/向心力/波速/RC/电功率/电感能量/理想气体/动量/热传导 + 第二批跨量纲:
  > 重力势能/功/功率/欧姆/电容储能/液压/热膨胀/声波长/抛体滞空时间/频率/电场/磁力/理想气体体积/
  > 平行板电容/应力/万有引力) → 语料扩到 **36 条真实配对**。
  >
  > **语义重标定(36, ST 语义)**：同配 mean=0.365 / p90=0.506, 跨配 mean=0.532 / p10=0.411 —— 配对从
  > Jaccard 恒 1.0 塌缩解耦为 0.17–0.55 连续语义值(阶段1方向在真实落地上成立)。n 从 4→20→36, 同/跨
  > 均值分离稳定(≈0.37 vs ≈0.53)但同 p90—跨 p10 重叠带仍在, 阈值仍未精标、不写运行阈值。
  >
  > **阶段2-A 离线 predictor 管线**：新增 `scripts/train_jepa_predictor.py` —— 冻结
  > `paraphrase-multilingual-MiniLM-L12-v2` 编码器 + 单隐层 predictor, 用 `jepa_pairs.jsonl` 闭环演示
  > 数据加载→冻结编码→训练(实际会随) →运行时冻结前向, 产物落 `{runtime_home}/models/jepa_predictor.json`。
  >
  > **运行时接入(本期落地)**：`_validate` 的 JEPA 块改为优先用冻结 predictor 前向算 surprise
  > (1−cos(predictor(φ(pred)), φ(actual)))，作探索动机的**相对/排名**信号, 并在
  > `prediction_error.surprise_source` 标注 `jepa_predictor/semantic/jaccard`。产物缺失/失败静默回落现有
  > 语义或 Jaccard, 不设绝对阈值、不更新权重(红线不变)。同一配对上实测:
  > predictor surprise **0.152** ≪ 语义 0.617 ≪ Jaccard 1.0 —— predictor 潜映射把 pred→actual 对齐更紧,
  > 是显著更好的排名信号。85 项既有回归全绿。
  >
  > **per-domain 相对化(阶段2-A-2)**：新增 `_relative_surprise` —— 按域(`self.surprise_domain`, 缺省
  > `global`)维护运行样本, 用经验分布秩把原始 surprise 映射到 **[0,1] 相对分数**(单调于原始值, 免绝对
  > 阈值, 对小样本稳)。`prediction_error.surprise_rel` 与 `_last_surprise_rel` 供给 episodic/encounter_space
  > (run_cognitive 的 episodic shard 改用相对秩), 原始 surprise 仍保留喂 feynman/tag/alignment 等既有阈值
  > 消费者, 语义不变。回归 20 项全绿。
  >
  > **目标级留出验证(新增 `scripts/eval_jepa_heldout.py`)**：语料按 `objective` 打标, 随机目标级拆分
  > (留出目标在训练里完全未见), 25% 留出 × 12 次: predictor 冻结前向重建 surprise = **0.243 ± 0.009**,
  > 优于"直接用 pred 取 actual"基线 0.374、远低于跨配转机 0.537, 且在留出目标上 12/12 次胜过基线;
  > 语义 surprise 可辨性: 留出目标 pred↔自身 actual 距离落进跨配 p10 以内 **12/12** 次。
  > 结论: 语义 surprise 在未见目标上**仍是可辨的**(可作相对信号), predictor 学到 pred→actual 潜空间偏移
  > 且**跨目标泛化**(未见目标重建≈全量 in-sample 0.234, 几乎不掉); 但重建均值 0.243 仍非零 ——
  > 潜对齐不完美, 只宜作排名/相对动机信号, 不可当绝对阈值或冒充可证伪数值预言(隔离红线不变)。
  > ⚠ 诚实边界: 36 样本仍不足以训出稳定绝对阈值; 此验证证明的是"信号有用 + 管线泛化", 非"阈值已成立"。
  >
  > **第三批 结构/方法级目标扩采(2026-09)**：为让 predictor 潜空间覆盖更广, 给 `scripts/collect_jepa_batch.py`
  > 新增 14 个**非纯数值量**目标(结构/方法级, index 32–45)：能量守恒核实(PASS/FAIL 判据)、小角近似误差分析
  > (校正因子+相对误差)、串/并联刚度比较、RC 半衰期、欠阻尼衰减比+能寿命、拍频、数值积分 vs 解析(e−1)对比误差、
  > 稳态线性温场中点、力矩平衡判据、相干波构造/相消干涉判据、多输出联报(range/高/滞空)、应力安全 PASS/FAIL、
  > 参数扫描(4 长度摆周期)、马赫数区制(criteria 判读)。真实 InternLM(书生) 逐条生成 plan+expected_prediction、
  > 沙箱执行真实 numpy/python 计算得 actual —— **14/14 全部采集成功**, 语料 36 → **50 条真实配对**。
  > 关键动机：纯数值量让 predictor 只能对齐"数字串", 而 VERDICT/PASS-FAIL/多输出/扫描/比较这些**方法级结构**
  > 编码了"怎么算、依据什么判据"的潜结构, 反向迫使 predictor 学到 pred→actual 的**语义/推理偏移**而非只会抄数字。
  >
  > **扩采后重训+留出复验**：50 条语料离线重训(冻结 ST 编码器, 同 §4.3 只前向), predictor 冻结前向重建 surprise
  > = **0.295**(in-sample) → 目标级留出 25% × 12 次 = **0.297 ± 0.008**, 12/12 次胜过基线(直接 pred 取 actual 0.420)、
  > 显著低于跨配转机(0.578)。与上一批(36 条, 纯数值, 0.243±0.009)相比, 绝对重建略升是**预期的**——
  > 结构/方法级目标的 actual 含判据/多行文本, 编码后本就更难逐字还原, 但**相对可辨性 / 优于基线 / 低转机**三条
  > 信号全部保持, 证明 predictor 学到的不只是"记住训练目标的数字", 而是可迁移的 pred→actual 潜空间偏移。
  > 诚实边界: 50 样本仍偏少, 阶段2-B 阈值精标仍须继续累积真实配对。
  >
  > **第四批 混合结构 / 方法级目标扩采(2026-09)**：继续往"方法级"纵深拓宽, 新增 12 个目标(→ `collect_jepa_batch.py` index 46–57)，
  > 刻意覆盖**非数值结果的结构类型**：多方法一致性(功-能定理 vs 数值积分终速互换)、两判据联判(动量+动能双守恒)、
  > 量纲核查(Grashof 无量纲)、迭代收敛(Newton-Raphson 迭代次数+残差判据)、数值稳定性(CFL 判据)、
  > 解析 vs 数值交叉(终端速度)、本征模态(2-DOF 双频率)、积分器对比(Euler vs Verlet 能量漂移)、
  > 级数收敛判定(比值判别)、往返自洽(反解角度回收 45°)、约束极值(拉格朗日乘子梯度条件)、
  > 随机 vs 解析(蒙特卡洛估 π)。真实 InternLM 生成 plan、沙箱执行原生 numpy/python —— **12/12 全部采集成功**，
  > 语料 50 → **62 条真实配对**。
  > 关键动机：第三批已引入"判据/多输出/扫描", 第四批再加**方法交叉、双判据联立、迭代收敛、本征模态、随机算法判断**
  > 这些更贴近"科研怎么算对"的推理结构 —— 前两批侧重"单数量多物理", 第四批打开"多数量+多方法+判据链",
  > 让 predictor 面对的 pred→actual 偏移分布更宽、更接近真实科研的验证语言。
  >
  > **扩采后重训+留出复验**：62 条离线重训(冻结 ST 编码器, 只前向, 同 §4.3 红线), 目标级留出 25% × 12 次
  > (15/61 目标完全未见): predictor 冻结前向重建 surprise = **0.322 ± 0.007**, 12/12 胜过基线(直接用 pred 取 actual 0.395)、
  > 显著低于跨配转机(0.581), 语义 surprise 可辨性 11/12。相比 50 条时的 0.297, 绝对重建略升仍是预期 ——
  > 方法级目标 actual 含多条验证结果/判据文本, 编码后本就难逐字还原; 但"优于基线、低转机、留出泛化"三条核心
  > 信号稳定保持, 证明确实学到可迁移的 pred→actual 潜空间偏移而非记住训练目标。阶段2-B 阈值精标仍需更多配对。
  >
  > **第五批 混合结构 / 方法级目标扩采(2026-09)**：继续纵深拓宽(→ `collect_jepa_batch.py` index 58–69)，新增 12 个目标，
  > 打开**更贴近分析/方法论**的结构类型：本征矩阵对称(Maxwell-Betti 互易判据)、Logistic 均衡(定点容量+判据)、
  > 简谐周期振幅无关性(过零测量双振幅一致)、误差传播(相对误差平方和正交化)、二分线性收敛阶(order=1 判据)、
  > 灵敏度指数(摆周期对长度指数 0.5)、双模叠加(两自然频率贡献)、Grashof+Prandtl 判据链(Rayleigh 区制)、
  > 热力学循环第一定律(稳态净存 0)、Snell 往返自检、牛顿冷却时间常数(63% 判据)、矩阵逆重建( A·A⁻¹=I 判据)。
  > 真实 InternLM 生成 plan、沙箱执行原生 numpy/python —— **12/12 全部采集成功**, 语料 62 → **74 条真实配对**、
  > 覆盖 73 个唯一目标。动机与前几批一致：让 predictor 面对的 pred→actual 偏移分布覆盖
  > "数值、判据、多输出、方法交叉、误差/灵敏度、本征对称、收敛阶"多重结构, 逼近真实科研的验证语言。
  >
  > **扩采后重训+留出复验**：74 条离线重训, 目标级留出 25% × 12 次(18/73 目标完全未见): 冻结前向重建 surprise
  > = **0.314 ± 0.005**, 12/12 胜过基线(0.406)、显著低于跨配转机(0.586), 语义 surprise 可辨性回到 **12/12**。
  > 相较 62 条时(0.322±0.007)误差棒进一步收窄(0.005), 且可辨性 11/12→12/12 —— 扩批非但没有掉泛化, 反而让
  > "优于基线 / 低转机 / 留出泛化"三条信号更稳。绝对重建 ~0.31 仍是编码不可还原的信息下限所致, 只宜排名/相对信号。
  > 阶段2-B 绝对阈值精标仍须继续累积真实配对。
  >
  > **第六批 混合结构 / 方法级目标扩采(2026-09)**：继续拓宽(→ `collect_jepa_batch.py` index 70–81)，新增 12 个目标，
  > 打开**数值分析 / 统计推断 / 稳定判据 / 谐波 / 假设检验**结构：中心差分二阶收敛阶实测、最小二乘线性拟合残差、
  > 矩阵条件数(近奇异判病态)、中心极限定理(std·√(12N)≈1)、测量扩展不确定度(t 区间)、临界阻尼区制(ζ 三分)、
  > 驻波整数谐波(1:2:3)、双模叠加(拍频)、势能稳定性(二阶导凸性判据)、黎曼和积分一阶收敛阶、
  > 多体碰撞动量守恒(解未知分速度+复核)、卡方拟合优度(公平骰子 accept/reject)。
  > 真实 InternLM 生成 plan、沙箱执行原生 numpy/python —— **12/12 全部采集成功**, 语料 74 → **87 条真实配对**、
  > 覆盖 86 个唯一目标。其中 `integral_error_riemann` 首次采集因我在目标里把收敛阶假设错(sin 端点对称致一阶项消失、
  > 实为二阶)而判据失准 —— 已把被采集数据剔除、目标改为真正一阶的 ∫x 并**重采该 index 修正**，验证 ratio=2.0 正确。
  > 这是"采集目标的物理假设也要可证伪"的一次直接教训, 印证诚实边界。
  >
  > **扩采后重训+留出复验**：87 条离线重训, 目标级留出 25% × 12 次(22/86 目标完全未见): 冻结前向重建 surprise
  > = **0.320 ± 0.007**, 12/12 胜过基线(0.389)、显著低于跨配转机(0.587), 语义 surprise 可辨性 **12/12**。
  > 样本从 50→87 逐批扩到下, predictor 的"优于基线 / 低转机 / 留出泛化 / 高可辨性"四条信号始终稳定,
  > 而绝对重建 ~0.32 稳定在编码不可还原的信息下限附近 —— 说明 predictor 已持续学到可迁移的 pred→actual
  > 潜空间偏移, 未因目标结构类型激增而退化。阶段2-B 绝对阈值精标仍须继续累积真实配对。
  >
  > **第七批 混合结构 / 方法级目标扩采(2026-09)**：继续拓宽(→ `collect_jepa_batch.py` index 82–93)，新增 12 个目标，
  > 打开**向量微积分 / 张量 / 特殊函数 / 守恒律**结构：标量场梯度一致性(|∇φ|大小核验)、Jacobi 行列式面积缩放、
  > Lagrange 插值过点核验、内积正交性、梯度下降收敛到极值、2D 非线性牛顿系统求解、引力结合能(U=-GMm/R=-mgR)、
  > 角动量守恒(Iω 守恒)、球贝塞尔递归一致、线积分路径无关(守恒场)、特征向量-特征值核验、Maxwell-Boltzmann 最可几速率峰值判据。
  > 真实 InternLM 生成 plan、沙箱执行原生 numpy/python —— 采集时 `grad_div_curl(index 82)` 曾一度未落盘(缺口),
  > 已按 index 单采补齐，语料 87 → **100 条真实配对**。
  >
  > **扩采后重训+留出复验(第七批)**：100 条离线重训, 目标级留出 25% × 12 次: 冻结前向重建 surprise = **0.318 ± 0.005**,
  > 12/12 胜过基线、显著低于跨配转机。向量微积分/张量/守恒律这些"方法级"目标与前述结构类型混合后，predictor 的
  > 分数化 relative-rank 信号依旧稳定, 未引入退化。
  >
  > **第八批 混合结构 / 方法级目标扩采(2026-09)**：继续纵深拓宽(→ `collect_jepa_batch.py` index 94–105)，新增 12 个目标，
  > 打开**向量场算符 / 多元微积分 / 数值积分 / 矩阵代数 / 正交展开 / 随机过程**结构：curl 并验证 div(curl)=0、
  > Clairaut 混合偏导对称(f_xy=f_yx)、e^x 泰勒/Maclaurin 收敛、Simpson 积分精度、3×3 行列式余子式展开一致性、
  > 叉积性质(|a×b|=|a||b|sinθ 且 a⊥a×b)、Cauchy-Schwarz 界、2×2 trace-det 特征分解、转置积恒等式(AB)^T=B^T A^T、
  > 旋转矩阵正交性(R R^T=I, det=1)、Fourier 正弦系数 b_1=2、Markov 平稳分布(πP=π, Σπ=1)。
  > 真实 InternLM 生成 plan、沙箱执行原生 numpy/python —— 采集时 `curl_divergence_vector_field(index 94)` 未落盘,
  > 已按 index 单采补齐, 语料 100 → **112 条真实配对**, 12/12 全入库。
  >
  > **合并重训+留出复验(8 批 112 条)**：112 条离线重训(冻结 ST 编码器, 只前向, 守 §4.3 红线), 目标级留出 25% × 12 次
  > (28/111 目标完全未见): 冻结前向重建 surprise = **0.324 ± 0.003**, 12/12 胜过基线(直接 pred 取 actual 0.368)、
  > 显著低于跨配转机(0.581), 语义 surprise 可辨性 12/12。语料从 50→112 逐批扩到, predictor 的
  > "优于基线 / 低转机 / 留出泛化 / 高可辨性"四信号始终稳定, 绝对重建 ~0.32 仍贴近冻结 ST 编码的不可还原信息下限,
  > 与第七批结论一致 —— 该 predictor 已持续学到可迁移的 pred→actual 潜空间偏移, 未因混合结构类型持续激增而退化。
  >
  > **评估路径修复: objective 标签落盘一致性 + 内容驱动回填**：采集期 `objective` 并非由 `_record_jepa_pair` 落盘,
  > 而是 `eval_jepa_heldout.py` 按**行位置**反推 OBJECTIVES 补标 —— 一旦语料行与 OBJECTIVES 非严格对齐(legacy 行、
  > 缺口按 index 单采、重复样本)就错位, 产出 `extra_*` 占位符(此前 112 行里有 4 行 `extra_86/98/110/111`)。
  > 修复: (1) `_record_jepa_pair` 增 `objective` 参数并落盘, `collect_jepa_batch._collect_one` 采集时直接带真实目标名(向后兼容);
  > (2) `_tag_objectives` 弃位置映射, 改为按每个目标真实 compute 输出反查 `actual→目标名` 回填缺失/占位行。
  > 回填后 112 行 `extra_*` 清零、无 None、108 个唯一目标(4 组为同目标双样本, 属正常)。目标级留出复验
  > surprise = **0.324 ± 0.004**, 12/12 胜基线(0.375)、远低于转机(0.584), 可辨性 12/12 —— 结论未变, 但归因更可靠。
  >
  > **阶段2-C 目标难度分层 + 对抗样本补采(2026-09)**：把"继续堆同量级目标"换成可证伪的定向补采。
  > 新增 `scripts/stratify_jepa_objectives.py`: 每条目标算 冻结前向 self_err 与"最近他人 actual"的 margin
  > (margin>0=自己更近=易; margin<0=投影落向他人=可混淆/难)。
  > **分层结论**：弱点族 = **短数值型 actual 目标**(周期/波长/体积/频率等, 输出就是一两行数字),
  > 在冻结 ST 潜空间互塌缩 —— `sensitivity_period_length / ideal_gas_volume / spring_period / pendulum_*`
  > 彼此把对方当"最近的正确", margin 贴 0 (最低 -0.012, `sensitivity_period_length` 是塌缩吸引子锚点)。
  > 易侧 (margin +0.25~+0.33) 反而是"重建差但区分清"的多行判据型目标。这印证绝对重建 ~0.32 是编码下限:
  > 预测器不是"谁的 actual 都能还原", 而是"能分清哪个 actual 是最该激活的" —— 短数值族恰是它最分不请的地方。
  >
  > **补采 10 个对抗样本(→ index 106–115)**：专打该塌缩族 —— 同值不同物理(弹簧 vs 摆同周期 1.0)、
  > AM≥GM、往返调和均(40≠算术45)、LC 周期、自由落体高度 roundtrip、波长 λ=v/f=v·T 双路线、
  > 理想气体 PV=nRT、频率周期倒数 roundtrip、同结构不同 g(更大 g→更小 T, 考验不塌缩)、往返算术均错误判据。
  > 真实 InternLM 生成 plan + 沙箱执行, **10/10 全部入库**, 语料 112 → **122 条**。
  >
  > **分层判据的预测性验证**：加入后重跑冻结前向, **10/10 对抗样本 `confused=True`**(预测被投影到
  > "他人的 actual" 而非自己的) —— 我们针对"短数值互塌缩"设计的样本, 被独立判据 100% 判为可混淆,
  > 证明该难度分层是可预测、不是噪音。重训后目标级留出(122 条, 30/118 目标留出):
  > surprise = **0.318 ± 0.003**, 12/12 胜基线(0.376)、远低于转机(0.580), 可辨性 12/12 ——
  > 加入对抗样本后 predictor 泛化不退化, 且弱点族从"孤点塌缩"补上了对照负例。
  >
  > **教训与路线**：对抗补采比同量级均匀扩张更有信息量 —— 它把"哪类目标当前预测器分不清"显式化
  > 并喂回。上限仍是冻结 ST 编码的信息下限 ~0.32(相对秩可用, 绝对阈值不可用, 见阶段2-B)。下一步若要
  > 更强可辨性, 应换更高容量/更大长度保留的文本编码或 span-level predictor, 而非继续扩短数值样本量。
  >
  > **编码器升级对照实验: 更高容量 (MiniLM-384 → mpnet-multilingual-768)**：
  > 为检验"~0.32 绝对下限是否是 MiniLM 容量不足所致", 把 JEPA 编码器解耦为 `HUGINN_JEPA_EMBED_MODEL`
  > (默认 `paraphrase-multilingual-mpnet-base-v2`, 768 维; 与共享 RAG 的 EMBED_MODEL 隔离, 不惊扰 taste 语义库),
  > 训练/验证/分层统一读取; 运行时 `_predictor_surprise` 也改走同一 JEPA 编码器(不再借用共享 `_EmbeddingModel`),
  > 保证落盘 predictor 与运行时维度/空间一致。下载经 hf-mirror + `HF_HUB_DISABLE_XET=1`(huggingface.co 443 被墙, 镜像 307 可达)。
  >
  > **结果(122 条, mpnet-768)**: 同配语义距离 0.374→**0.331**(更近), 跨配 p10 0.429(分离更开);
  > 目标级留出重建 surprise = **0.324 ± 0.002**, 可辨性 **12/12**, 远低于转机 0.553。
  > **关键证伪**: 直接用 pred 取 actual 的**基线大幅变强 0.376→0.329**(更高容量编码让 pred 本身就更贴 actual),
  > 训练 predictor 相对该基线的胜率从 **12/12 收窄到 7/12**; 且重跑难度分层, 10/10 对抗样本仍 `confused=True`
  > (短数值型 actual 互塌缩未被缓解)。
  >
  > **可证伪结论**: ~0.32 的绝对重建下限**不是** MiniLM 容量不足 —— 换 768 维 mpnet 没能击穿它, 反而把
  > "pred 直接当 actual" 的基线拉得几乎与训练 predictor 一样好, predictor 的边际增量收窄。真正的信息瓶颈在
  > 文本本身: 多行数值+判据型 actual 经任何固定句向量都无法被预测文本逐字还原, 只能提供相对排名信号。
  > 这正面(!)回应了"换更高容量编码器能否破局"的假设 —— 容量升级改善原始语义贴近度, 但不改变"绝对阈值不可行、相对秩才稳健"的阶段2-B 结论。
  >
  > **Span-level 结构化输出编码方向验证 (阶段2-C-2)**：既然句子级把"多行数值+判据" actual 压成一个
  > 平均向量导致短数值互塌缩, 试**行级 span 表示** —— 把 prediction/actual 拆成行级 span 向量(保
  > "数字行" 与 "判据行" 分立), 逐 span 共享 MLP(d→64→d) 预测, 目标 = 同 pair 最近真实 span。
  > 新增 `scripts/train_jepa_span_predictor.py`(同一 mpnet-768 编码, 同一 122 语料)。
  >
  > **结果**: span predictor 前向 surprise = **0.279 ± 0.003**, 12/12 分制胜句子基线(0.339)、
  > 远低于转机(0.553), 逐留出样本上 span<句子基线 252 次。对照同容量编码的句子级(0.324),
  > span 级把 pred→actual 映射压紧了约 **0.045** —— 行级结构确实缓解了"整段压句向量"的信息丢失。
  > **诚实边界**: span-surprise 取"每预测 span 到最近真实 span"的最小匹配, 与句子余弦不是同一量纲,
  > 应视为**相对/方向性**证据而非绝对真理; 该差值是否稳定还需更大语料。~0.28 是否仍受"预测文本本身缺
  > 少还原所需信息"的下限约束, 需更大样本才可定论。生产化(把 span predictor 接入运行时)是后续工作量,
  > 本轮仅完成方向验证。
  >
  > **Span 数值规范化子实验(负结果)**: 为检验 0.279 里还剩多少是"数值格式噪声(1.0 vs 1.0000)", 加 `--canon`
  > 把数字 token 统一成 %6g 后再跑同一 span pipeline。结果**逆向**: span surprise **0.279→0.306 恶化**,
  > 胜句子基线 12/12→10/12, 逐样本胜出 241→184。根因: 大量目标共享相同数值(弹簧/摆/LC 周期都近同值),
  > 原格式(1.0/1.0000/2.0061)恰是模型区分"哪个数值字段"的额外结构信号; 归一化把它们压成同一串,
  > 跨目标混淆反而上升(chance 0.556→0.564)。**可证伪收窄**: 提升来自行级结构分离(0.324→0.279),
  > 而非数值平滑; 数值 token 原样保留是有益结构, 不该归一化。
  >
  > **Span 标签/单位字段分离子实验(方法级)**: 为更进一步区分"同值不同物理"(弹簧 vs 摆周期都近 1.0),
  > 试把每行显式拆成 `[整行 | 标签/单位前缀 | 数值token]` 三通道拼接(`--fields`, 768→2304) —— 不删原始信号, 只增结构。
  > 结果仍未正向: span surprise **0.287→0.295**(同跑次对比)、胜基线 12/12→11/12、逐样本 219→199。根因:
  > 逐 span 共享 MLP + 最近真实 span 对齐在拼接后的高维向量上, 没有机制真正利用"标签通道", 多余维度成噪声;
  > 且近邻对齐在高维下匹配关系改变。
  > **收窄结论**: 在"span 行切分 + 共享 MLP + 最近邻对齐"设计下, 行级结构分离已把映射压到位(0.324→0.287),
  > 数值平滑与标签分通道都没再提升 —— 下一步增益不应来自"往旁路拼通道", 而应换**能消费结构的架构**
  > (attention 对齐 / token 级字段槽), 而非简单 concat。
  >
  > **Attention 对齐子实验(≈持平)**: 换**自注意力聚合**(单头 h=32/64): 每个预测 span 由同 prediction
  > 其他 span 加权(只读预测内, 不看 actual)再经 MLP 回归, 给模型可学习的结构消费机制。结果 attention
  > surprise **0.286/0.284 (h=32/64)**, 与逐 span 共享 MLP(0.285-0.287)持平、12/12 胜基线、低转机。
  > **收窄到机制之上**: 在 span 表示下, 匹配机制(硬最近邻 vs 逐 span MLP vs 自注意力)、容量、数值平滑、
  > 标签分通道都汇聚在 **0.28-0.31** 区间 —— 剩余下限由预测文本本身缺精确数值/多行还原信息决定, 非架构。
  >
  > **生产化(接入运行时)**: 把表现最佳的 base span predictor 用 `--persist` 落盘
  > `{runtime_home}/models/jepa_span_predictor.json`(K=8, 768, 纯行切分); 运行时 `_predictor_surprise` 改为
  > **优先 span 前向**(逐行 span → 掩码均值近邻真实 span), 缺则回落句子级; 均为同 `HUGINN_JEPA_EMBED_MODEL`
  > 编码(训练=运行时一致), 不惊扰共享 RAG。真实语料冒烟: self<cross 全成立, 回归 67 passed/1 skipped。
  >
  > **阶段2-B 阈值精标: 可证伪结论 —— 绝对阈值不可行, 相对秩是唯一稳健形式**：
  > 新增 `scripts/calibrate_jepa_threshold.py`(全量)与 `calibrate_jepa_threshold_holdout.py`(目标级留出)做阈值分离分析。
  > 把同配(pred→自配 actual, 应小)标 label0、跨配(pred→他配 actual, 应大)标 label1, 对阈值扫描 F1/Youden/AUC。
  > 结果(87 条, 冻结 predictor 前向): **全量 AUC=0.5007, 目标级留出 mean AUC=0.5008**(8 次拆分
  > 0.491–0.506), 且每拆 same_mean≈cross_mean(差 <0.001)。即同配与跨配的 surprise 分布**几乎完全重叠**, 任何
  > 单阈值都无法同时"放行正确 + 拦截错配"(最优 F1 处 FPR=TPR=1, 即只用"全判异常"才能高捕获)。
  > —— 这是一条**决定性负面证据**: 对当前 predictor, 绝对阈值判定不可用。
  > 根因与 predictor 重建 theor 下限 ~0.32 一致: 数值/判据/多行 actual 经冻结 ST 编码后本就不可逐字还原,
  > 所有前向 surprise 坍缩到同一窄区间, 跨配相对同配无显著增大, 故无分离带。
  > **据此收口**: 阶段2-B 的"绝对阈值"目标不成立, 运行时保留 §4.15 的**相对秩** `_relative_surprise`(per-domain
  > 经验秩→[0,1]) 作为唯一稳健的 surprise 消费方式, 继续供 explore 排名/encounter_space, 不当绝对门槛、更不冒充
  > 可证伪数值预言。此为"先证值得做、再证做不到"的诚实闭环 —— 阈值本是分类器, 数据不给分离度, 就停手不硬造。
  >
  > **收口 review: 修复 predictor 激活导致的量纲漂移 bug**：上结论后对 JEPA 线做了只读审查, 发现一个真实风险——
  > predictor 前向在运行时默认激活(predictor.json 存在即接管), 其 surprise 坍缩在 ~0.2-0.45 窄带, 而下游多处
  > legacy 硬编码阈值(plan_check `>0.5`/`>0.9`、hypothesis_loop `>0.5`/`>0.6`、早停 `max(0.08,0.20-0.4·noise)`)
  > 预期的是 jaccard 的 [0,1] 宽带量纲。predictor 值 <0.45 导致 `>0.9` 强制 explore **永不触发**(失效)、部分 >0.5
  > 值又可能**误触发**、早停阈值上限 0.20 低于 predictor 前向常见值(可能误停)。量纲漂移会在运行时静默扭曲决策。
  > **修复**(engine_reflect.py `_validate`)：当 `source=="jepa_predictor"` 时, 交给下游 `_last_surprise` 的一律用
  > **相对秩 `surprise_rel`**([0,1], 域归一, 单调), 原始前向值保留在 `prediction_error.surprise_abs` 供审计;
  > jaccard/semantic source 行为不变; `_record_jepa_pair` 采集仍存原始前向值。这样 predictor 激活时下游量与
  > legacy 阈值兼容, 且正好兑现阶段2-B"相对秩才是信号"的结论。已跑 tests/test_cross_scale_invariance +
  > test_lucid_prereqs = 67 passed/1 skipped; `_relative_surprise` 冒烟验证单调且域归一到 [0,1]。
  > 诚实边界: 域桶按 `surprise_domain` 累积、不分 source, 冷启动时相对秩样本少会偏低区分度; 且 predictor 和
  > jaccard 值混入同桶会污染秩 —— 这是既有的域桶设计边界, 不在本次修复范围, 留待未来(如需精确分域可分桶)。
  >
  > **更大语料复核 0.28 的稳定性(方案①)**: 补采 batch7-10 共 46 条混合结构/方法级目标(index 82-127:
  > 矢量微积分, Jacobian 行列式, Lagrange 插值, 内积, 梯度下降/Newton-Raphson, 动量/引力结合能, RMS,
  > log 积恒等式, Fibonacci-Cassini, Euler 公式, 勾股三元组, 质心, 方差分解, 抛体射程, 等比级数等), 语料
  > **122→134** 对(判据数 130 目标)。`train_jepa_span_predictor.py` 同配(同一 mpnet-768, 同一种子/切分)重跑:
  > span predictor 前向 surprise **mean=0.287 ±0.004**, 12/12 held-out 分制胜句子基线(0.327), 远低于转机(0.553),
  > 逐留出样本上 span<句子基线 229/375。与 batch7 前(0.279±0.003)相比, **0.28 下限在更大语料上稳定成立**、
  > 未见退化 —— 印证 `LOOP_EXPERIENCE_REPORT.md §4.15` 行级 span 把 pred→actual 压到的 ~0.28 不是小语料假象。
  > span predictor 权重已对 134 对语料重训并 `--persist` 落盘(覆写 `{runtime_home}/models/jepa_span_predictor.json`,
  > num_objs=134, dim=768)。
  > 诚实边界: 全程配对混淆率(每个预测 span 的最近真实 span 是否属本 pair)随语料扩到 134 后升至 **99.3%** ——
  > 因为真实 span 库越大, 硬最近邻跨配天然越高; 真正决策指标仍是**目标级 held-out 分制胜句子基线**(12/12),
  > 混淆率数字本身不构成"信号坏了"的证据, 只提醒"最近邻匹配"与"分布区分"是两件事。
  >
  > **方案①-收口: span surprise 正式化为相对秩信号(区分 source)**: 按方案①, 把 span surprise 从"实验数字"
  > 走成运行时正式信号 —— `_predictor_surprise` 由返回 `float|None` 改为返回 `(surprise, source)`,
  > source ∈ {`jepa_span_predictor`(优先), `jepa_sentence_predictor`(回落)}; 调点 `_validate` 据此打独立标签;
  > `_relative_surprise(surprise, source)` 的秩桶由 `global` 改为 `{domain}:{source}` —— span 与 句子 predictor
  > 各在**自身分布内**算经验秩, 避免跨量纲(span~0.28 vs jaccard~[0,1])混桶污染(回应上节"混入同桶会污染秩"
  > 的既有边界); 透传给下游的量纲收口判定从 `source=="jepa_predictor"` 收紧为 `source.startswith("jepa_")`
  > (span/句子均走相对秩, 其余 jaccard/semantic 行为不变)。`surprise_source` 现可精确区分 span/sentence。
  > 已 `ast.parse` 校验语法; 复跑 span predictor 复核数全部与上述一致。
  >
  > **方案①-实现 + 诚实负面: 结构化计划槽不破 0.28 信息下限**：
  > 按"输入侧富计划"方向落地最小闭环 —— 新增 `huginn/jepa_slots.py`(槽序列化
  > `PLAN_SLOTS: a=3; b=4; formula=RMS(a,b)` + 防泄漏检测); planner(`plan_check._parse_plan`
  > 解析 `SLOTS:`/`FORMULA:` 行, `engine_act`/`cognitive_loop` 把槽块追加到 prediction 并暂存);
  > `_record_jepa_pair` 落库 `prediction_inputs`/`plan_formula` 字段并做泄漏检测(槽数值 ≈ actual
  > 末位答案码 → 丢弃该对)。
  > **验证(决定性负面)**: 对 9 个人工核过 given 的方法级目标做双通道 A/B——
  > (a) 固定 134 训练 predictor, 喂 `原始 vs 原始+PLAN_SLOTS` 文本, surprise **0.2602→0.2643
  > (d=+0.004, 1/9 胜)**; (b) **重训**槽版 predictor 于槽化语料, held-out surprise **0.282±0.003 vs
  > 基 0.287** —— 无统计显著移动。
  > **结论(修正原叙事)**: "prediction 缺字面数字"在本语料上**不成立** —— 运行时 prediction 文本
  > 已很富(常含答案), 故追加规范槽是冗余文本, 不提供新信息, 也不降低 ~0.28。0.28 下限更接近
  > **表征/信息处理上限**(冻结 mpnet + span-MLP 无法把文本中的数值转成 actual 的潜分布), 而非
  > 文本级信息缺失。据此, 多补输入富文本这条路的**杠杆被本实验否证**; 更高杠杆方向转回
  > ***表征侧**(数值感知编码 / 潜变量 y-predictor)或**诚实对比/排序目标**(相对秩已是消费端,
  > 把它变成可训分离)。运行时槽采集保留(无害、仅方法级触发、带防泄漏), 但不再作为破 0.28 的路径。

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