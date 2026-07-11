# 信息流 — 三个关键性质

## 1. 有向性

信息在系统中沿固定方向流动，形成有向循环：

```
tool_call → tool_result → _learn → memory/KB/KG → _build_*_text → prompt → tool_call
```

这个循环不是对称的。每一环的角色不同：

- **tool_call** 是 action：LLM 通过工具调用改变环境状态（跑 VASP、执行 LAMMPS、写入文件）
- **tool_result** 是 perception：环境把执行结果回传给 agent，这是 agent 感知世界的唯一通道
- **_learn** 是固化：把感知到的东西写入三个仓库，让经验不会随 turn 消失
- **_build_*_text** 是回流：仓库内容在下一轮被重新检索、格式化、注入 prompt
- **prompt** 是决策依据：LLM 基于回流的信息做出下一轮 tool_call

箭头方向不可逆转。tool_result 不会自发产生 tool_call（需要 LLM 推理），memory 不会自发修改 prompt（需要 ContextBuilder 主动检索）。这意味着信息流是因果有序的：感知先于固化，固化先于回流，回流先于行动。

有向性的直接后果是反馈延迟：本轮 tool 产生的信息要到下一轮 prompt 才能影响决策。Agent 无法在同一轮内"看到"自己刚写入 memory 的内容。这个延迟是结构性的，不是性能问题——它保证了每轮决策基于稳定快照而非竞态状态。

## 2. 多尺度

同一套信息流模式（hypothesize → execute → validate → learn）在三个尺度上同构出现：

### Phase 层（Autoloop Engine）
七阶段自主循环：Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report。每个阶段是一个完整的认知动作，阶段间通过 surprise 信号和 PhaseGate 传递信息。这是最粗粒度的信息流——一个完整循环可能跨越数十次工具调用。

### Stage 层（Deli AutoResearch）
学术研究管线的九阶段：Topic → Search → Gap → Outline → Draft → CitationVerify → PeerReview → Revision → Final。每个阶段有对应的 hypothesize（gap 分析）、execute（起草）、validate（引用验证/同行评审）、learn（修订）。阶段间通过 integrity gate 传递信息，不通过则打回重做。

### Tool 层（symbolic_math 等）
单个工具内部的动作序列：PDE 工具的 classify（分类）→ separation/characteristics（推导）→ discretize（离散化）→ constraint_check（约束检查）。这是最细粒度的信息流——一次工具调用内部就完成了假设-执行-验证的微循环。

三个尺度不是嵌套关系，而是分形同构。Phase 层的 surprise 信号在 Tool 层没有直接对应物，但 hypothesize→execute→validate→learn 的结构在每个尺度都完整存在。这种同构不是刻意设计的结果，而是研究活动本身的结构特征——任何尺度的研究都需要先猜、再做、再验、再记。

跨尺度不变性测试（`test_cross_scale_invariance.py`）验证了这个性质：三个尺度在四个环节上都为 True。

## 3. 非平衡稳态

系统的信息流不是一次性管线，而是一个持续运转的循环。pause/resume 不打断信息流，goal 持久化维持稳态。

### pause/resume 不打断
GoalStore 支持 active → paused → active 的状态转换。pause 时，已有的 memory/KB/KG 内容全部保留，goal 对象本身持久化在 goals.json 中。resume 时，ContextBuilder.build_goal_text() 重新注入 goal 文本，AutoloopEngine 从上次的 iteration 计数继续。信息流没有断点——pause 只是暂停了新信息的产生，已存储的信息仍然完整。

这和"关机再开机"本质相同：系统的状态不在内存里，而在持久化层。memory 在 SQLite，KB 在 ChromaDB，KG 在 project_kg.json，goal 在 goals.json。这些文件就是系统的稳态载体。

### goal 持久化维持稳态
GoalStore 跨 session 存在。上一个 session 设定的 goal，下一个 session 打开时仍然 active。build_goal_text() 每轮注入 "Persistent Goal (iter N)" 文本，让 LLM 始终知道目标是什么、进行到第几轮。这个持续注入是稳态的关键：没有它，每个新 session 会忘记自己在做什么，信息流会从零开始而非从上次的状态继续。

非平衡稳态的含义是：系统不趋向某个固定点然后停止，而是在 goal 的驱动下持续循环。goal 完成时 status 变为 completed，循环停止；goal 未完成时，每轮 learn 后 GoalJudge 检查是否达成，未达成则继续。这是一种动态平衡——信息流在 goal 的约束下持续运转，直到 goal 被满足或用户主动暂停。
