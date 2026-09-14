# Research Note: RSI/AGI 差距复评 —— 三条质变缺口补齐之后

> **状态**: report · **日期**: 2026-09-14 · **性质**: 本轮三条质变缺口（M-R1 / A1 / A3）落地后的独立复评, 承接 `third-party-audit-final.md` 与 `physical-rsi-and-world-model-interpretability.md`（dual-axis RSI）的缺口列表, 重新核对"离 RSI / AGI 还有多远"。

## 结论（TL;DR）

在补齐"递归自举 / 复合加速 / 直接自改源码"三道质变缺口后, Huginn 从"单层非递归改进器"迈入了一个**有护栏的、双环（improver→strategist→source）+ 真实 r_phys 地面真值验收 + 时空可组合**的递归改进层。这真实地缩小了与 RSI 的距离, 但**离自足 RSI / AGI 仍有四条本质缺口**, 未随本次工作消失。下方逐条核对。

---

## 一、本次已落的差距（对照 previous audit 的缺口清单）

| 原缺口 | 现状（代码锚点） | 状态 |
|---|---|---|
| 改进器单层非递归 | `MetaImprover` 把 improver_prompt/阈值提升为一等对象; `maybe_propose`→`evaluate`→`maybe_promote` 走 SignificanceGate+OODHoldout, 仅 GREEN 换代（[meta_improver.py](file:///workspace/agent/huginn/harness/meta_improver.py)） | ✅ 补上（递归深度 1）|
| 改进器如何改进（策略层）不可变 | strategist champion 让"生成改进器候选的策略模板"本身可被改进, 只保留 meta² 模板写死（A1 边界） | ✅ 补上（递归深度 2）|
| STOP 复合加速无从验证 | `CompoundingTracker` 滚动窗口斜率/成本趋势/滞回/死锁; `is_compounding` 只判"不退化" | ✅ 补上（复合验证）|
| Goodhart（代理分可被 gaming）| `BehavioralFidelity`（真实采纳率锚）+ `RPhysTrack` 真实 r_phys 归因验收（Mann-Whitney U 地面真值）+ `RandomizedControl` 真实验收采样 + holdout 子集 | ✅ 大幅缓解 |
| 论文 2609.03621 可验证工作流 | `VerifiableGate.battery()` 对真实 `world_model` 全部能力动作做约束+前向验证 | ✅ 接线真实模型 |
| 直接自改源码（第三大缺口）| `huginn/harness/source_patch.py`: 候选源码补丁经 compile+anchor+定义验证 → 运行时 monkeypatch, 可逆、默认关; MetaImprover source 环 | ✅ 补上（v0 运行时级）|
| 时空可组合 | `RevertibleContext` 换件补偿（时间）+ `CoEffectRegistry` 依赖图（空间） | ✅ |
| 全链路可验证 | 引擎入口冒烟（`EngineReflect._generate_next_loop_directive`）全开关驱动三环 | ✅ |
| 回归地基 | `dev` extra 补齐科学栈, `tests/` 从"81 文件收集 error"恢复全量可收集 | ✅ |

## 二、仍真实存在的四道本质缺口（未随本次消失）

1. **世界不是开放的、r_phys 仍是标量 grader**。所有"改进"最终关回到一个可被下游 grader gaming 的标量 `r_phys`; 改进后的 agent 并没有在一个**开放、持续、自带评价器的世界**里自足运行。AGI 的关键不在"能改自己", 而在"活在世界里并持续被世界校验"。
2. **递归深度是被硬封顶的**。A1 明确 `只一层`、strategist 的 meta² 模板写死、`不做无限递归`——这是**安全优先的刻意的封顶**, 不是能力溢出。自足 RSI 需要受控的、随经验增长而加深的能力, 而当前深度是常数。
3. **自改能力被钉死在自己的节奏常量上**。A3 v0 的目标 `_SOURCE_TARGET` 硬编码为 `meta_improver._PROPOSE_EVERY_N`; 且只做运行时 monkeypatch（进程内, 不跨 session）。"能改任意函数/架构/跨 session 固化"未做。
4. **验证器本身不被自改进**。`VerifiableGate` 的能力电池、`world_model`（aspirate/dispense/mix/aliquot 的朴素规则）是**固定资产**; 改进器改 prompt、改节奏, 但"如何验证"不被进化。真正的元自举要连验证手段一起进化。
5. **双轴世界模型自改进（Physical RSI）未接**。对照 physical-rsi 笔记: Huginn 仍是工件轴（改策略/假设）; "域模型/物理正确性"这根第二轴只走了数据通道, 未闭环到自改进里。

## 三、客观评分（0 无 → 5 自足）

| 维度 | 年前(补齐前) | 现在 | 与自足 RSI 的差距 |
|---|---|---|---|
| 递归自举 | 0 | 3（两层, 有护栏）| 深度恒定、封顶 |
| 复合加速验证 | 0 | 3（滚动窗口 + 斜率）| 只判"不退化", 未证真实加速 |
| 直接自改源码 | 0 | 2（运行时, 单目标）| 非任意, 不跨 session |
| Goodhart 治理 | 1 | 4（多锚 + 真值）| r_phys 本身可被游戏 |
| 开放世界自足运行 | 0 | 1（有引擎冒烟, 无实机长跑）| **最大缺口** |

## 四、下一步的杠杆（与用户此前的效益比分析一致）

1. **低成本焊接**: 把 `world_model` 从朴素规则升级为数据驱动（第二轴雏形）+ 让 `VerifiableGate` 验证"agent 真实产出的实验"而非代表性电池。
2. **提升 A3 覆盖**: source 目标可配置化/多目标; 讨论跨 session 持久化的安全前提（需回归地基——已补）。
3. **世界闭环**: 接一个真实的、带协调净评价器的目标域（compute-heavy）, 让 `HUGINN_META_ABLATION` 真跑起来。

> 一句话: 三道缺口让 Huginn 成为**有严格护栏、可验证、两层递归、能自改自身节奏**的改进系统; 它离"自足递归自我改进"还差"开放世界持续被世界校验"这最后一根轴。