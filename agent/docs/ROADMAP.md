# ROADMAP — AutoloopEngine 方法族去 mixin化

目标：把 `AutoloopEngine` 的多继承 mixin 方法族逐步拆成独立协作对象（组合 + 薄委托），消除 god-class。由小到大、状态写最少者优先，每个阶段独立验证（引擎测试 + arch 门禁）后再进下一个。

- [x] M1: MathValidationMixin → MathValidator（387 行，3 方法）
- [x] M2: EnginePerceiveMixin → EnginePerceive（535 行，15 方法）
- [x] M3: VisualInspectMixin → VisualInspect（636 行，6 方法）
- [x] M4: EngineActMixin → EngineAct（926 行，14 方法）
- [x] M5: EngineControlMixin → EngineControl（862 行，20 方法）
- [x] M6: PlanCheckMixin → PlanCheck（1169 行，22 方法）
- [x] M7: EngineObserveMixin → EngineObserve（1560 行）
- [x] M8: EngineReflectMixin → EngineReflect（3469 行）
- [x] M9: HypothesisMixin → HypothesisLoop（2969 行）
- [x] M10: CognitiveLoopMixin → CognitiveRunner（3591 行，命名替换避免与同模块编排类 CognitiveLoop 冲突）

行数为拆分汇总实测量。每个里程碑定义与方法族契约见 `docs/staging/specs/2026-09-13-engine-decompose-mixin.md`。

---

## 独立轨道：RSI 递归改进层（harness 之上，非 mixin 工作）

去 mixin 目标是消除 god-class；本轨道目标是补「递归自我改进」——在已落地的单层 harness 改进器（H1 prompt_patch 等）之上加一层 meta-improver，让「改进器自身如何改进」也可被改进、可被 gate 验收。与去 mixin 无依赖。

- [x] M-R1: Recursive (Meta-)Improver — 改进器自身的 improver_prompt/阈值成为可改进对象，复用 SignificanceGate / OODHoldout / AdoptionGate 做验收，仅 GREEN 才全局换用改进器配置（默认关）

契约与环境见 `docs/staging/specs/2026-09-13-recursive-improver.md`。

---

## 独立轨道：结构完整性修复（反转默认）

目标：把验收从「事后读产物 + 运行时解析散文」搬到「数据流内绑定」；把安全默认从「允许」反转为「拒绝」；把文档从「生成物」变成「被测试断言」；把度量从「设计覆盖」换成「执行覆盖」。四轨独立、可分别验收。

- [ ] M-INT1: 契约进数据流（`contract.json` + sha 绑定，删除散文解析）
- [ ] M-INT2: 默认拒绝（local bash / 审批 / 子进程 env 三条基线）
- [ ] M-INT3: 文档断言（generated / manual / 无标记即失败）
- [ ] M-INT4: 执行覆盖（`HUGINN_TRACE=1` → trace.json 对照 census）

契约见 `docs/staging/specs/2026-10-31-structural-integrity-repairs.md`，计划见 `docs/staging/plans/2026-10-31-structural-integrity-repairs.md`。