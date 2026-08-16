# Research Note: Physical RSI 与视频世界模型可解释性对 Huginn 的启发

> 状态: **report**（研究参考，非实现契约；据此驱动的设计变更需另立 spec/plan）
> 日期: 2026-08-16
> 关联: `autoloop` / `evolve` / 知识蒸馏 / `diagnose` / `failed_direction` /
>       Lean 4 形式化 / 两条控制轴（ModelTier + ThinkingIntensity）/ 成本叙事

本备忘记录两篇研究对 Huginn Agent 的启发，供后续架构决策参考。**不杜撰、不承诺实现**，
每条启发标注"可直接落地 / 需设计 / 仅认知参考"。

---

## 1. Physical RSI 框架（MirroS）

> 核心主张：数字世界 RSI 是**单轴进化**（环境已知 → 只改进 Actor/策略）；物理世界必须
> **双轴耦合**——Actor 与 **World Model（对域的理解/仿真模型）** 同步迭代。外环由
> "OOD 意外"驱动，八步进化；内环做"先判认知、再判动作"的诊断。

### 1.1 与我们 agent 的对应

| Physical RSI 概念 | Huginn 现状 | 启发 | 可落地性 |
|---|---|---|---|
| 单轴 vs 双轴进化 | `autoloop`/`evolve`/蒸馏是单轴（改进假设/策略，环境视为已知） | 缺**第二根轴**：对"域模型"（材料/物理/仿真正确性）的自改进 | 需设计 |
| Environment Gap vs Skill Gap 诊断 | `diagnose`（计算化学/MD 错误诊断）+ `failed_direction` 记忆类型 | 失败归因应显式拆出"域模型问题"路径，让 `evolve` 知道该更新认知还是策略 | **已落地**（见 §1.2） |
| OOD 意外 → 进化燃料 | `verification_status` 蒸馏闭环 | 意外应触发**认知更新**而非仅策略更新 | 需设计 |

### 1.2 落地进展

- **已落地（2026-08-16）**：`FailedDirectionRecord` 新增 `gap_type` 归因字段
  （`environment_gap` / `skill_gap` / `unknown`），贯穿 `FailedDirectionStore.record()`
  写入与 `query()` 过滤，P12 typed 与 legacy 两条路径均支持，随 `strategy:`
  /`math_concept:` 同一 content+tag 模式存储。selfcheck 与 `tests/test_evolution_modules.py`
  （118 passed）覆盖。域模型侧 `diagnose_tool` 未改（其为 skill 侧检索工具，硬塞规则归因脆弱）。
- 待推进：`evolve` 消费 `gap_type` —— 当 `environment_gap` 命中时产出对域模型/仿真假设的修订，
  而非仅产出新策略。

---

## 2. Interpreting Physics in Video World Models（Meta FAIR, ICML 2026）

> 核心发现：视频世界模型**不用紧凑分解的物理引擎状态变量**，而是用**高维分布式、类脑表示**
> （方向编码在 40–80 维圆形种群码里）；物理在中间层"Physics Emergence Zone"(PEZ) 才线性可及，
> 且在输出层退化。相关工作用 CAV 在 PEZ 层注入方向向量即可无训练地改变物理判断（steering）。

### 2.1 与我们 agent 的关系

| 论文概念 | 对 Huginn 的意义 | 可落地性 |
|---|---|---|
| 分布式表示足可"预测" | 反衬我们**因子化/符号化**路线的价值：要的是可解释的科学发现（紧致定律），非黑箱预测 | 认知参考 |
| Compact 状态 vs 分布表示 | Lean 4 形式化是我们的"可解释性窗口"，是差异化优势而非劣势 | 认知参考 |
| 关键层可干预（steering） | 借鉴：给"物理/域推理关键层"加可观测、可干预旋钮，而非只在输入端调 prompt；呼应我们两条控制轴 | 需设计 |
| Physics Emergence Zone (PEZ) | 在 trace 里定位"域知识从哪一层开始起作用"，把观测/审计（HOOKS、成本叙事）对准关键区 | 需设计 |

### 2.2 建议动作（若推进）

- 在观测/审计设计中，识别并聚焦"域推理关键区"，而非平均撒网。
- 把两条控制轴（ModelTier/ThinkingIntensity）表述为"可干预的推理 steering"，供后续
  文档与 UI 文案复用。

---

## 3. 一句话总结

- **Physical RSI** → 给 `evolve` 加第二根轴：学习"世界模型"而非只学策略，失败时先判
  "认知缺口"还是"技能缺口"。
- **世界模型可解释性论文** → 坚定"因子化 + Lean 4 形式化"立足点（可解释发现 vs 分布式
  黑箱预测），并借鉴"关键层可干预(steering)"来设计控制轴与观测点。

## Source（原文）

- MirroS: Building Physical RSI Beyond the Known World
- Joseph et al., *Interpreting Physics in Video World Models*, arXiv:2602.07050
- Alam et al., *Causal Physics Steering in Video World Models via CAVs*, CVPRW 2026
- RSI-Bench: Multi-Axis Benchmark for Recursive Self-Improvement