# Spec: deep_think → CSpace 桥（外部草稿 → 可证伪工作区）

> 状态: **Design / 待实现**。目的: 把 external-thinking 的"写"通道（deep_think 把思维倾入
> reasoning_trace）与 C-Space 的"读 + 门禁"通道（`huginn/research/cspace.py`）接成一条闭环:
> **思维草稿永远只是"候选在场"，只有被证据证实才 promote 为可引用的在场**。
> 这是"幻觉不因在场而变得可信"在推理仓库上的落地，也是对 Qoder/Harness 五维里
> "Learning Capture" 的个体层具体化。

## 1. 背景与动机
- **写通道（已有）**: `huginn.tools.deep_think_tool.DeepThinkTool` 让模型在行动前把分析显式
  写出，扁平 `context.memory_manager.add_reasoning(text)` 落 `session.reasoning_trace`，
  结构化 `add_reasoning_record(ReasoningRecord)` 落更细粒度 record(phase/claim/evidence/
  estimate/uncertainty/plan)。原生 `reasoning_content` 也汇入同一 trace（双通道并存）。
- **读通道（已有）**: `huginn.research.cspace.CSpace` 提供 `probe/pin/suppress/readout/broadcast`，
  且 `run_research_program(workspace=)` 已把报告经工作区广播门禁。
- **断口**: 两个通道目前不连通 —— 模型"写下来的思考"从不自动进入可读、可证伪的工作区；
  工作区也不消费 reasoning_trace。结果: 思维草稿要么堆在 trace 里不被复用，要么被报告
  直接引用却无证伪背书。

## 2. 核心原则（一条红线）
> **deep_think 草稿 ≠ 在场。草稿是候选（falsifiable=False），进入工作区即挂起
> （pending），只有被证据证实（promote）才能成为可引用在场。**

这保证: 模型可以随便"想"，但"想的东西能不能当结论/证据用"由证据门禁说了算，不由模型
"想到"了就算。

## 3. 数据模型
复用 `CSpace.Being`，仅新增一个派生字段 `phase`（来自 `ReasoningPhase`）与身份标记:

```
Being(
  id        : str,             # 稳定: "dt_" + sha1(origin+phase+claim)[:12]
  kind      : "concept",       # 三类在场之一(草稿作概念在场)
  payload   : {phase, claim, estimate, uncertainty, plan},
  source    : str,             # 指向 reasoning_trace 中该条目的指针(证据追溯)
  falsifiable : bool,          # 默认 False → 候选; promote 后 True → 在场
  activation : 0.0, pinned:False, suppressed:False,
  phase     : str,             # think|plan|pre_action|reflect
)
```

**三态机**：
- `candidate`  : falsifiable=False —— 刚入桥，出现在 `readout().pending_no_source`。
- `confirmed`  : falsifiable=True —— 通过 promote 门禁，可被 probe 点亮、进 readout。
- `rejected`   : 从 trace 移除或标 `suppressed` —— 低质/空/套话，或不具证伪前景（记计数）。

## 4. 双向接线图
```
模型 → deep_think(analysis/phase/claim/evidence/estimate/uncertainty/plan)
        └─ add_reasoning(text)──────────────► session.reasoning_trace
        └─ add_reasoning_record(record) ──────► reasoning records
                                              │
                              cspace_bridge.ingest_reasoning_trace(cspace)
                                              ▼
                                    candidate Beings (falsifiable=False, pending)
                                              │
                     cspace_bridge.promote_to_at_hand(cspace, id, corroborate=…)
                                              ▼
                              confirmed Being (falsifiable=True) → probe/readout/report
```
报告/决策只能引用 **confirmed** 在场；`estimate`（定量预判）在后续真实执行到达时，
用 reconcile 式对账把 pre_action 预言升级为强在场。

## 5. 契约（新模块 `huginn/research/cspace_bridge.py`，进 research 接缝）
```python
def enqueue_deliberation(cspace, record, *, origin="deep_think") -> Being: ...
    # 从 ReasoningRecord(或 flat text) 建 Being, falsifiable=False; 幂等(同 id 不重复).

def ingest_reasoning_trace(cspace, records: list[ReasoningRecord | str],
                           *, origin="deep_think") -> list[Being]: ...
    # 批量入桥: 每条 → candidate; 空/纯空白跳过并计数 rejected_empty.
    # 需要从 memory_manager 读回 records 的访问器: 若缺, 新增
    # `memory_manager.iter_reasoning_records()`(只读, 不复制敏感全文入上下文).

def promote_to_at_hand(cspace, being_id, *, 
                       corroborate: Callable[[Being, CSpace], bool] | None = None,
                       verify=None) -> dict: ...
    # 门禁: falsifiable=True 仅当 corroborate 通过——默认策略:
    #   (a) 该 Being 的 claim/estimate 数值在 cspace.trace 有真值(过 broadcast/verify), 或
    #   (b) 该 Being 带可对账的 estimate, 且已由真实执行的 reconcile 数值证实。
    # 通过: falsifiable=True(confirmed); 否则保持 candidate 并计数 pending.

def reject(cspace, being_id, *, reason="") -> None: ...
    # 标 suppressed 或移除; 维护 rejected_counts 供治理账本统计 "套话率"。

def confirmed_at_hand(cspace) -> list[Being]: ...
    # 便捷: readout().at_hand(仅 falsifiable=True)—— 报告/决策唯一可引用集合。
```
`verify` 复用 `claim_grounding`（与 `cspace.broadcast` 同一实现），不另造门禁。

## 6. 门禁语义（promote 的判定）
- **定性 claim**（think/reflect）: 只有当其**数值化断言能在 trace 中找到对应真值**时才 confirmed，
  否则永不 promoted（说明性是软引用，不是证据在场）。
- **定量预判**（pre_action 的 `estimate`）: 这是桥的“明星路径”—— estimate 是数字，天然可证伪；
  后续真实执行给出 actual 时，用 `law_model.reconcile` 式数值对账，|Δ| 在容差内 → promoted 为
  强在场；不符 → 如实 rejected/falsified，进入治理账本。
- **promote 失败不进报告**: 报告阶段只读 `confirmed_at_hand`，绝不因草稿里有它就把未证实
  断言当结论。

## 7. 与既有组件的集成
- **写侧薄钩**: 不进 `deep_think_tool` 内部耦合；由持有 CSpace 的运行时在
  `memory_manager.add_reasoning_record` 后调用 `ingest_reasoning_trace`（桥从内存读回，
  工具本身行为不变，向后兼容）。
- **读侧复用**: 完全走 `CSpace.probe/pin/suppress/readout/broadcast`；`promote` 用
  `cspace.broadcast` 同源的 `grounding_verifier`。
- **管线侧**: `run_research_program(workspace=)` 已接入报告场——桥的 confirmed 在场
  与存活结论并列，都被工作区广播门禁约束。

## 8. 失败模式（fail-open，不阻断主流程）
| 情形 | 处理 |
|---|---|
| `memory_manager` 为 None | bridge 入桥为 no-op（不抛） |
| deep_think 空/套话 | 入桥即 `reject_empty`，记计数，不进任何 confirmed |
| 草稿 id 撞车(同 claim 重复想) | 幂等合并，不重复入场 |
| 读回 records 的访问器缺位 | 回到“无桥”行为（无增强不回归）—— flag 兜底为空 |
| estimate 无后续真值 | 保持 candidate；超龄由治理账本/蒸馏自然淘汰 |

## 9. 隐私 / 上下文
- **草稿不回显**: 原始 trace 全文不进入报告上下文；只暴露 `confirmed_at_hand` 的精选在场
  （id/kind/activation/source），防上下文刷屏、也防原生推理泄漏。
- 安全默认对齐 `mcp_export` 只读语义：bridge 全程只读，不起副作用。

## 10. 测试计划（`tests/test_cspace_bridge.py`）
1. `enqueue` → candidate(falsifiable=False) 出现在 `pending_no_source`，不在 `at_hand`。
2. `promote_to_at_hand` 无证据 → 保持 candidate；带 trace 真值 → confirmed 进 `at_hand`。
3. `estimate` 路径: pre_action 预言 + 真实执行 reconcile 数值吻合 → promoted；偏差 → rejected。
4. `ingest_reasoning_trace` 批处理: 幂等去重、空草稿计空、id 稳定。
5. 断口闭环: 构造假模型 `deep_think` → trace → ingest → promote → `run_research_program`
   报告只引用 confirmed（与既有 workspace 门禁协同）。
6. fail-open: memory=None 无异常。

## 11. Deferred
- LLM 对草稿的**质量打分**（deviation/novelty）—— 当前交给门禁与蒸馏自然过滤，不硬编码。
- provider 级 `forceReasoningOff` 强制双通道并存策略维持现状（见 external-thinking spec）。

## 12. 与治理/五维的关系
- 这是 **Learning Capture** 的个体层实现：草稿→证实→可复用在场，是"教训→资产"的微观种子。
- confirmed 在场的 `source` 指针保持到 `task_episode`（后续补统一任务实录身份），即可被
  团队/组织层治理账本跨项目统计——即上一轮谈的“任务实录 + 共享治理账本”的第一步落脚点。

## 里程碑
- M1: `cspace_bridge.py` 核心（enqueue/ingest/promote/reject/confirmed）+ 单测 1–4。
- M2: `memory_manager.iter_reasoning_records()` + deep_think 后薄钩接线，测试 5。
- M3: reconcile 式 estimate 证实路径 + 治理计数暴露，测试 3 完整。