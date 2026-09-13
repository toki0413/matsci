# Spec: AutoloopEngine 方法族去 mixin化（M1 首个阶段=MathValidationMixin）

**日期**: 2026-09-13
**milestone**: M1（docs/ROADMAP.md）

## 目标（一句话）
把 `AutoloopEngine` 的多继承 mixin 方法族（**16,104 行 / 10 个文件**）逐步拆成**独立协作对象**，引擎经组合持有实例、保留薄委托方法，最终不再靠多继承把整个认知中枢堆成 god-class。本 spec 覆盖**模式定义 + 全 10 mixin 分阶段路线 + 阶段1（MathValidationMixin，387 行）落地**。

## 背景证据（实测，非推断）
- mixin 文件体量：`cognitive_loop.py`(3591)、`engine_reflect.py`(3469)、`hypothesis_loop.py`(2969)、`engine_observe.py`(1560)、`plan_check.py`(1169)、`engine_act.py`(926)、`engine_control.py`(862)、`visual_inspect.py`(636)、`engine_perceive.py`(535)、`math_validation.py`(387)。
- `AutoloopEngine` 由 10 个 mixin 多继承（engine.py），全部通过 `self` 直接读写引擎状态 `self._xxx` / `self.signals` / `self.model` 等。
- `MathValidationMixin` 仅 3 方法、只碰 `self.workspace/settings/_query_kb_reference`；其唯一调用点 `engine_reflect.py:133 self._run_math_validation(...)`。

## 模式（去 mixin pattern）
1. 每个 `XxxMixin` → `Xxx`（普通类），构造器收引擎（duck-typed `engine`）或显式依赖，暴露 `run(...)` 等方法。
2. 引擎 __init__ 持 `self.<xxx> = Xxx(self)`；并保留同名**薄委托方法**（如 `async def _run_math_validation(self, x): return await self._math_validator.run(x)`）→ 既有 `self.method()` 调用点**零改动**。
3. 从 `AutoloopEngine` 继承列表移除该 mixin。
4. 每个阶段独立验证（引擎测试 + arch 门禁绿）后再进下一阶段。

## decisions（阶段路线）
- **阶段1（本 spec 实施）**：`MathValidationMixin` → `MathValidator`。 ✅ 已落地（2026-09-13）
  - `MathValidator.__init__(self, engine)` duck-typed 读 `engine.workspace/settings`、调 `engine._query_kb_reference`、读 `engine._last_execution_result`；暴露 `async run(execution_result)`（=原 `_run_math_validation`）、`async collect_math_evidence(...)`、`def verify_via_gp(...)`。
  - `engine.py`: 移出 base 列表（MathValidator 不再继承）、`__init__` 加 `self._math_validator = MathValidator(self)`、保留 3 个薄委托方法 `_run_math_validation` / `_collect_math_evidence` / `_verify_via_gp` → 调用点 engine_reflect.py:133/153 零改动。
  - 证据：`tests/test_engine_decomposed.py` 5 项 + `test_math_validation.py` + `test_autoloop_engine.py` + arch 门禁全绿。
- **后续阶段（各自横轮，逐个 PR）**：`EnginePerceive`(535)→`VisualInspect`(636)→`EngineAct`(926)→`EngineControl`(862)→`PlanCheck`(1169)→`EngineObserve`(1560)→`EngineReflect`(3469)→`HypothesisLoop`(2969)→`CognitiveLoop`(3591)。由小到大、状态写最少者优先。

## contract（阶段1 接口）
- 新 `huginn/autoloop/math_validation.py::MathValidator`：`__init__(self, engine)`（duck-typed，读 `engine.workspace/settings`、调 `engine._query_kb_reference`）；`async run(execution_result) -> dict`（等价原 `_run_math_validation`）。
- 引擎：base 列表移除 `MathValidationMixin`；`__init__` 加 `self._math_validator = MathValidator(self)`；保留 `async def _run_math_validation(self, execution_result): return await self._math_validator.run(execution_result)`。

## invariant
- `engine._run_math_validation(x)` 与重构前输出一致（委托等价）。
- `engine_reflect.py:133` 调用点不改。
- 测试：`engine_state.signals` / arch 门禁 / `test_autoloop_engine` 全绿。

## test（证明）
- `tests/test_engine_decomposed.py`：
  1. `test_autoloop_engine_no_longer_inherits_math_mixin`：`MathValidationMixin not in AutoloopEngine.__bases__`，且引擎有 `_math_validator` 且是 `MathValidator`。
  2. `test_math_validator_transfers_state`：`MathValidator(engine)._run...`?——改为：构造 `MathValidator` 收一 stub engine，`run({...})` 返回 dict 且暂不抛（守恒/变分缺工具时只记 *_error）。
  - 回归：`test_autoloop_engine` + arch 门禁。

## deferred（暂不定）
- 不在此阶段碰其余 9 个 mixin；不删委托方法（删桥需等阶段路线走完后统一评估）。
- 大 mixin（cognitive_loop/engine_reflect/hypothesis_loop）是否内部再拆子对象，各阶段自行定。

## 范围（out of scope）
- 不改 math_validation 的任何校验逻辑分支；只改"归属形式"（mixin → 组合）。
- 不动其它 mixin 与 engine 其它方法。