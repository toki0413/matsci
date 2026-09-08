# C-Space (Capability Workspace) — 外部化全局工作区的设计规格

> 对齐: Anthropic 《A global workspace in language models》的 J-Space (GWT) 与"空间/状态工作区"
> (S-Space) 的设想。把"agent 脑子里有哪些概念/状态在**场**"从隐式隐含, 变成**可读、可控、
> 可推理、可证伪**的**外部持久工作区** —— 这是产品能力, 不是论文复刻(见诚实边界)。

## 1. 动机
长程科研 agent 的通病是"嘴上说过的东西转眼不记得、没法证明它在场"。J-Space 给我们的启发不是
"找模型内部激活", 而是: 一个 agent 推理时, 需要一小撮**它当前在想的东西**持续在场、可被读、
可被控、可被推理引用。我们把这个工作区**外置**到代码层, 让它跟我们的知识图(kg)、世界状态
(LawModel)、交互可解释(interaction_explain)、声明门禁(claim_grounding) 咬合成一条线。

## 2. 三大"在场"(being-at-hand) 通道
工作区里的每个原子叫 **Being**, 有三类, 对应两种工作区的合流(J-概念 + S-状态)再叠加**可读性**:

| kind | 直觉对应 | 证据来源(必须可证伪) |
|---|---|---|
| `concept`   | J-Space 概念在场 | `kg` 节点 / 命题 + trace 里的 grounding 数值 |
| `state`     | S-Space 状态在场 | `research.law_model.world_model_card` + reconcile 可对账 |
| `interaction` | J-Space 可读激活 | `research.interaction_explain` 交互基元(Σφ, 结构性) |

## 3. 四项能力 (对应 J-Space 的操作性: report / control / reason / audit)
- **report(读)**: `probe(text)` 按词关联做**激活扩散(spreading activation)**, 返回点亮排序。
- **control(控)**: `pin(id)` 钉住(内置在每次读里点亮)、`suppress(id)` 压制(soft 降维)——即
  J-Space 的"hypnotize / 熄灭"。
- **reason(推理引用)**: `readout()` 把点亮 Beings 折叠成一段紧凑上下文, 供提示/决策直接引用。
- **audit(可证伪门禁)**: `broadcast(statement)` 必须过 `claim_grounding`(声明门禁)才落 `trace`;
  **未落凭据的 Being(falsifiable=False)即使被 probe 命中也绝不"确认在场"** —— 这是对 J-Space
  "无证据不为在场"的治理, 也是"幻觉不因在场而变得可信"。

## 4. 治理规则(机械可测)
1. 每个 Being 必须带 `source` 证据指针; 缺证据 → `falsifiable=False` → 读时激活恒为 0。
2. 每次读(report)把"点亮"建立在**证据可链接**的 Beings 上; 关联激活只做召回助手, 不替代证据。
3. 广播/进 trace 必须过声明门禁; 未落地数值不得成为"结论在场"。

## 5. 与既有组件的关系(复用不新起)
- 复用 `grounding_verifier()`(program.py 单一实现)、`world_model_card`、`interaction_trace`。
- **不另造存储**: 持久化复用 `persistence/checkpointer`+`kg`; 本工作区是内存会话态 + trace 挂载。
- 入口唯一: 所有"读工作区 / 控在场 / 广播"统一走 `huginn.research.cspace.CSpace`, 不内联再造。

## 6. 诚实边界
- 这是**外部/近似**模拟外部化工作区空间, 不声称"读出模型内部激活"(那是 J-lens 的事)。
- 激活扩散是启发式召回, 不是真实内建 Jacobian; 只把它当**证据索引 + 召回**, 不当"模型在想"。
- S-Space 若有官方定义, 以官方为准; 本规格用"状态在场"作为其务实近似。