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
- [x] A1: Recursive Compounding（strategist 可演化 + 复合护栏）— 在 M-R1 之上叠一层 strategist，让「改进器如何改进」也可被改进；用 CompoundingTracker 复合验收、BehavioralFidelity 保真锚、RPhysTrack 端到端真实 r_phys 归因验收（Mann-Whitney U 地面真值通道）治 Goodhart、VerifiableGate（结合 arXiv:2609.03621 可计算实验室表示）把换件锚到状态化仿真验证；死锁/滞回/随机化对照/RevertibleContext+CoEffectRegistry 时空可组合。默认关。

契约与环境：
- M-R1：`docs/staging/specs/2026-09-13-recursive-improver.md`
- A1：`docs/staging/specs/2026-09-13-recursive-compounding-design.md`