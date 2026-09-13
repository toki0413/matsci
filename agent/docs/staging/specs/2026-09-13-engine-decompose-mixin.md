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
- **阶段2（本 spec 后续）**：`EnginePerceiveMixin` → `EnginePerceive`。 ✅ 已落地（2026-09-13）
  - 15 个 perceive/上下文构建方法下沉为普通类 `EnginePerceive(engine)`。
  - **属性转发**设计：`__getattr__`/`__setattr__` 把未定义属性读写转发到 engine → 方法体零改动、`_kb/_perception/_persona_manager` 等引擎级共享缓存留在 engine 不复制（缓存共享、行为完全等价）。
  - `engine.py`: 移出 base 列表、`__init__` 加 `self._engine_perceiver = EnginePerceive(self)`、保留 13 个薄委托方法 → 调用点 plan_check/engine_observe/cognitive_loop/engine_reflect 零改动。
  - 证据：`test_engine_decomposed.py` 阶段2 4 项 + `test_autoloop_engine.py` + arch 门禁全绿。
- **阶段3（本 spec 后续）**：`VisualInspectMixin` → `VisualInspect`。 ✅ 已落地（2026-09-13）
  - 6 个 visual_inspect 方法下沉为普通类 `VisualInspect(engine)`。方法体只用引擎字段(_last_visual_context/_visual_base64/_last_visual_base64)+同级方法、不调引擎方法 → **只读转发**: `__getattr__` 把未定义私有字段读转发到 engine(object.__getattribute__ 终止 engine==self 时递归); `__setattr__` engine!=self 转发到引擎(清除陈旧 base64 写回一致), engine==self(mock/selfcheck)直写实例避免递归.
  - `engine.py`: 移出 base 列表、`__init__` 加 `self._visual_inspector = VisualInspect(self)`、保留 7 个薄委托方法 → 调用点 engine_act.py:337/phase_spec 零改动。gate 测试子类继承 mock 同步改 object.__setattr__。
  - 证据：`test_engine_decomposed.py` 阶段3 4 项 + `test_visual_inspect.py` + `test_visual_inspect_gate.py` + autoloop/arch 全绿。
- **附带修复（历史遗留）**：`tests/test_krcl_plan_check.py::_make_engine` 用 `__new__` 绕过 `__init__` 却未注入 `signals`, 触发类级 SignalBridge `_set` → AttributeError。补 `eng.signals = EngineSignals()` 后 58 failed → 58 passed（与 M3 一并落地）。
- **阶段4（本 spec 后续）**：`EngineActMixin` → `EngineAct`。 ✅ 已落地（2026-09-13）
  - 14 个 plan/execute/llm_chat 方法下沉为普通类 `EngineAct(engine)`。方法体大量读写引擎状态(字段+方法: _grill_active/_current_prediction/model/model_router) → **全属性转发**: `__getattr__` 把未定义属性读转发到 engine(object.__getattribute__ 防 engine==self 递归), `__setattr__` 转发写(engine is self 时直写实例防自递归)。字段/方法留引擎不复制。
  - `engine.py`: 移出 base 列表、`__init__` 加 `self._engine_actor = EngineAct(self)`、保留 14 个薄委托方法 → 调用点（_llm_chat 被 reflect/hypothesis/plan_check 共用, _execute 被 cognitive_loop, _execute_skill 被 skill_tool）零改动。
  - 证据：`test_engine_decomposed.py` 阶段4 4 项 + autoloop/arch/krcl 全绿。
- **阶段5（本 spec 后续）**：`EngineControlMixin` → `EngineControl`。 ✅ 已落地（2026-09-13）
  - 20 个循环控制/checkpoint/状态持久化/澄清交互/事件总线方法下沉为普通类 `EngineControl(engine)` → 全属性转发(__getattr__ 读转发 / __setattr__ 条件写), 字段/方法留引擎。
  - `engine.py`: 移出 base、`__init__` 加 `self._engine_controller = EngineControl(self)`、保留 20 个薄委托方法 → 调用点 cognitive_loop/engine_act/plan_check/engine_reflect 零改动。
  - `test_budget_gui_approval.py` 由 `EngineControlMixin._maybe_run_budget_approval(engine)` → `EngineControl(engine)._...`, `_attach_budget_helpers/_human_decide_result` 同步改组合构造。
  - 证据：`test_engine_decomposed.py` 阶段5 4 项 + `test_budget_gui_approval.py` + autoloop/arch/krcl 全绿。
- **阶段6（本 spec 后续）**：`PlanCheckMixin` → `PlanCheck`。 ✅ 已落地（2026-09-13）
  - 22 个 plan_check/plan-prompt 方法下沉为普通类 `PlanCheck(engine)` → 全属性转发(__getattr__ 读转发 / __setattr__ 条件写), 字段/方法留引擎。
  - **own-method 覆写槽**：PlanCheck 定义自身方法名集合 `_OWN_ATTRS`，`__setattr__` 命中该集合(如测试 mock `_plan_check`)时写入本对象实例 dict（而非转发回引擎）→ 协调器 `_plan_check_and_refine` 内部调同名步骤时能覆盖。`_plan_check_history/_last_result/warnings/patterns` 等引擎状态字段不在集合 → 仍转发回引擎。
  - `engine.py`: 移出 base、`__init__` 加 `self._plan_checker = PlanCheck(self)`、保留 22 个薄委托方法 → 调用点 engine_act._plan / cognitive_loop / engine_observe 零改动。
  - `test_krcl_plan_check.py`: mock 边界从引擎移到 checker（`eng._plan_check` → `eng._plan_checker._plan_check`）——因 `_plan_check_and_refine` 是 PlanCheck 自身方法，引擎实例覆盖不再拦截内部同名步骤；`_make_engine` 挂 `_plan_checker` + checker 上 mock 持久化。
  - 附带修复（历史遗留）：`test_lucid_prereqs.py::_make_engine_with_graph` 与 `test_math_prompt_injection.py` fixture 用 `__new__` 绕过 `__init__` 却缺 `signals`，补 `engine.signals = EngineSignals()`（SignalBridge 前置依赖）。
  - 证据：`test_engine_decomposed.py` 阶段6 5 项 + `test_krcl_plan_check.py` 58 项 + `test_lucid_prereqs.py` plan-routing 类 + autoloop 回归无新增失败。
- **阶段7（本 spec 后续）**：`EngineObserveMixin` → `EngineObserve`。 ✅ 已落地（2026-09-13）
  - 36 个 prompt 拼装 + 元认知方法 (含 `_build_hypothesis_prompt` / `_apply_block_patches` / `_trim_to_budget` / `_build_pmk_block` / metacog getters/checkers) 下沉为普通类 `EngineObserve(engine)` → 全属性转发 + own-method 覆写槽 `_OWN_ATTRS`。
  - **类常量桥**：`_MATH_DEPTH_PROMPT_BLOCK` / `_PROMPT_BUDGET` / `_PROMPT_BUDGET_BY_PHASE` / `_IMAGINATION_PROMPT_BLOCK` 需保持 `AutoloopEngine._X` 类级访问与 plan_check 转发读取 → 在 engine 加同名字段引用观察对象类常量。
  - **duck-typed 类方法兼容**：`trigger_isomorphic_anomaly_hypothesis` / `trigger_alignment_surprise_hypothesis` 被 `metacog/hypothesis_manifold`、`cli/rcb_cognition` 当纯类方法调用（传 stub engine，无 `_engine_observer`）→ 委托方法用 `getattr(self, "_engine_observer", None)`，缺时回退 `EngineObserve.<method>(self, ...)` 以维持 stub 兼容（两方法只读 `_hypothesize/_last_hypothesis/_last_raw_hypothesis`）。
  - `engine.py`: 移出 base、`__init__` 加 `self._engine_observer = EngineObserve(self)`、保留 36 个薄委托方法 → 调用点 cognitive_loop/engine_reflect/hypothesis_loop/engine_act/plan_check 零改动。
  - 测试迁移：`test_hypothesis_loop` TestTopologyPromptInjection 由 `object.__new__(EngineObserveMixin)` → `AutoloopEngine.__new__` + `_engine_observer`；`test_world_state` 由 `_Dummy()` → `_Dummy(object())`（stub engine，方法经 getattr default 走转发）。
  - 证据：`test_engine_decomposed.py` 阶段7 5 项 + `test_world_state` + `test_hypothesis_loop` + `test_autoloop_engine/eventsourcing/phase_gate/cognitive` + arch 门禁全绿。
- **后续阶段（各自横轮，逐个 PR）**：`EngineReflect`(3469)→`HypothesisLoop`(2969)→`CognitiveLoop`(3591)。由小到大、状态写最少者优先。

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