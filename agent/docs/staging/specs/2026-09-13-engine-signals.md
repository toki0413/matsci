# Spec: EngineSignals 收敛 AutoloopEngine 环信号

**日期**: 2026-09-13
**milestone**: M1（见 docs/ROADMAP.md，若已建）

## 目标（一句话）
把 `AutoloopEngine` 上被 10 个 mixin 与外部（`runtime/engine_state` 等）交错的 ~28 个**纯环信号**字段收敛进一个显式 `EngineSignals` dataclass，引擎持 `self.signals`；策略=包 + 属性桥 + TDD，持久化同步切到 signals snapshot。

## 背景证据（实测，非推断）
- 信号字段（`_next_phase_hint / _last_surprise / _surprise_history / _darwin_best_score / _consecutive_failures / _refine_count / _pivot_count` 等）被 **112 处 / 15 文件** 读写；最重：`cognitive_loop.py`(39)、`runtime/engine_state.py`(23)、`engine.py`(10)、`engine_reflect.py`(7)、`plan_check.py`(5)。
- `runtime/engine_state.py::_ENGINE_FIELDS` 用 `getattr/setattr` duck-typing 持久化，当前字段里混有环信号与基建态（`_budget_* / _last_persona / _mcmc_* / _visual_*`）。

## contract（EngineSignals 接口）
- 新 dataclass `huginn/autoloop/signals.py::EngineSignals`，纯数据、无副作用、可序列化：
  - `to_snapshot() -> dict`（`asdict` 化，供持久化/audit）
  - `from_snapshot(d: dict) -> EngineSignals`（容忍缺字段→默认值，兼容旧格式）
  - `apply_to_engine(engine) -> None`（把信号写回引擎，经属性桥）
- 引擎持 `self.signals`；对纳入的每一字段保留同名 `self._<field>` **读/写 property 桥**，转到 `self.signals.<field>`。
- `AutoloopEngine.__init__` 由 4 个 `_init_*` 分片改为 `self.signals = EngineSignals(...)` 一处初始化。

## decisions（字段归属）
**纳入 EngineSignals（纯环信号）**
- G1 失败/回退：`_consecutive_failures`、`_consecutive_failures_by_type`、`_validate_window`、`_refine_count`、`_pivot_count`、`_next_phase_hint`、`_refined_hypothesis`
- G2 演化信号：`_last_surprise`、`_surprise_history`、`_darwin_best_score`、`_darwin_stagnation`、`_darwin_last_score`、`_darwin_belief_mu`、`_darwin_belief_sigma2`、`_last_hypothesis_confidence`、`_last_hypothesis_evidence_strength`、`_evals_history`、`_scene_tag_extra_keywords`
- G3 计划检查：`_plan_check_history`、`_plan_check_last_result`、`_plan_check_warnings`、`_plan_check_patterns`
- G4 迭代/点名：`_iteration`、`_should_stop`、`_current_phase`、`_grill_active`、`_grill_turns`、`_last_visual_context`、`_last_rule_hit_id`

**留在引擎（基建态，不迁）**
- 服务句柄/懒实例：`model / kg / memory / workflow_engine / explorer / coder / verification_model / model_router / metacog 各懒实例 / phase_gate_hook`
- 其它持久化态：`_budget_rejects / _budget_degraded / _last_persona / _mcmc_* / _visual_primitives_history / _image_embeddings / _token_budget`
- 事件/通道句柄：`_event_bus / _side_channel / _goal_scheduler / progress_tracker / _wake_scheduler`

## persistence 决策（engine_state.py，一次到位）
- `_ENGINE_FIELDS` 保留；新增集合 `_SIGNAL_FIELDS`（=纳入清单）。
- `_snapshot_engine`：`_SIGNAL_FIELDS` 从 `engine.signals` 的 snapshot 聚合，其余字段照旧 `getattr(engine, f)`。
- `apply_state_to_engine`：`_SIGNAL_FIELDS` 写 `engine.signals`（`from_snapshot`），其余照旧 `setattr`。
- 磁盘格式兼容：`EngineState` 内新增 `signals: dict`（用 `EngineSignals.to_snapshot()` 产出）承载环信号；旧 snapshot 缺 `signals` 时回退默认（load 容忍）。
- `_FakeEngine` stub 增 `signals`，selfcheck 场景 1/5/7 相应扩断言。

## invariant
- 所有既有 `self._<field>` 读点在迁移完成前**必须**仍返回原值（属性桥保证），112 处旧读点不改。
- `engine_state` save→load→apply 逐字段 round-trip 一致（含环信号）。
- 任何迁移后 `test_arch_single_gateway / test_arch_cleanliness / autoloop 相关门禁` 全绿。
- mixin 读取行为零变化；只动"存储位置 + 持久化读法"，不改任何逻辑分支。

## failure（失败模式与降级）
- 属性桥里若某字段尚未建立（顺序迁移中途），`getattr` 抛 AttributeError → 迁移必须一次合入（字段与桥同 PR 落），不允许半迁移态上线。
- `from_snapshot` 对未知/缺失键走默认值（新增字段不破老 snapshot），与现有 `load_engine_state` 的缺省语义一致。

## test（证明")
- `tests/test_engine_signals.py`：
  1. `test_signals_snapshot_roundtrip`：构造 EngineSignals 填非默认值 → `to_snapshot` → `from_snapshot` → 断言逐字段等于首值（含 tuple/list 嵌套，如 `_validate_window`）。
  2. `test_property_bridge_preserves_reads`：挂到 `_FakeEngine`+signals，断言 `eng._iteration / _last_surprise / _next_phase_hint / _validate_window` 读回与 `self.signals` 一致，写也一致。
  3. `test_engine_state_signals_persist`：`save_engine_state` → `load_engine_state` → `apply_state_to_engine`，环信号与基建态都 round-trip（扩现有 engine_state 自检场景）。
- 回归：`python -m pytest tests/test_arch_*.py tests/test_autoloop_*.py tests/test_engine_?.py`（以仓库实际存在的 autoloop/engine 测试名为准）。

## deferred（暂不定）
- 不在此 PR 全量把 112 处旧 `self._xxx` 逐个改成 `self.signals.x`（属性桥承接，后续按点在独立 PR 迁移）。
- 不合并 `_budget_* / _mcmc_* / _visual_*` 进 signals（它们各自是独立关注点，后续各开 bundle）。
- 不引入 `EngineState` 之外的新序列化框架（沿用 asdict+JSON）。

## 范围（out of scope）
- 不改任何 mixin 的业务逻辑分支；只动存取位置与持久化读法。
- 不改 `EngineState` 的其余字段语义；`EngineSignals` 是它的一个子槽，非替换。

## Status（2026-09-13 实施完成）
- 已落地：`huginn/autoloop/signals.py::EngineSignals`(29 字段, `SIGNAL_NAMES` 单一事实源) + `to_snapshot/from_snapshot/apply_to_engine`。
- 已接线：`AutoloopEngine.__init__` 首行 `self.signals = EngineSignals()`;类尾 29 个属性桥(读/写都落 signals)。
- 持久化一次到位：`ENGINE_STATE.signals: dict` 槽 + `_snapshot_engine/load/apply` 走 `EngineSignals` snapshot(旧格式缺 `signals` 容忍)。
- 验证：`tests/test_engine_signals.py`(4 用例)+ signals/engine_state 自检 + `test_autoloop_engine` + arch 门禁 全绿(29 passed)。