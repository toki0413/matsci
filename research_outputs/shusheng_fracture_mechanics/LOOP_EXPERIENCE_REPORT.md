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

## 6.5 方案2：书生成码"带错修错"agentic 重试(cycle16 后新增)

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
- 这本质是把 Code-as-World 的 `CompareAndDiagnose→Δ→局部修订` 从"世界假说"层平移到"书生的实验代码"层: 每一次 err 都是一次可验证的 Δ, 迭代到预算(2次)耗尽即停。

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