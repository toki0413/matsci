# ROADMAP — AutoloopEngine 方法族去 mixin化

目标：把 `AutoloopEngine` 的多继承 mixin 方法族逐步拆成独立协作对象（组合 + 薄委托），消除 god-class。由小到大、状态写最少者优先，每个阶段独立验证（引擎测试 + arch 门禁）后再进下一个。

- [x] M1: MathValidationMixin → MathValidator（387 行，3 方法）
- [ ] M2: EnginePerceiveMixin → EnginePerceive（535 行）
- [ ] M3: VisualInspectMixin → VisualInspect（636 行）
- [ ] M4: EngineActMixin → EngineAct（926 行）
- [ ] M5: EngineControlMixin → EngineControl（862 行）
- [ ] M6: PlanCheckMixin → PlanCheck（1169 行）
- [ ] M7: EngineObserveMixin → EngineObserve（1560 行）
- [ ] M8: EngineReflectMixin → EngineReflect（3469 行）
- [ ] M9: HypothesisMixin → HypothesisLoop（2969 行）
- [ ] M10: CognitiveLoopMixin → CognitiveLoop（3591 行）

行数为拆分汇总实测量。每个里程碑定义与方法族契约见 `docs/staging/specs/2026-09-13-engine-decompose-mixin.md`。