"""Engine state persistence — crash-safe resume for AutoloopEngine.

借鉴 InternAgent Table 1 "Persistence Running" 打钩项: 把 ~15 个运行时字段
原子写到 <workspace>/.huginn/engine_state/<run_id>.json, crash 后通过
resume_from_state=<run_id> 恢复, 维持 persona-memory-knowledge 循环连续.

HUGINN_USE_PERSISTENCE=1 启用, 默认 off. Flag off 时 save/load 完全跳过,
现有 run_cognitive 行为 100% 不变.

Layout:
  <workspace>/.huginn/engine_state/<run_id>.json       — EngineState
  <workspace>/.huginn/hypothesis_graph_<run_id>.json   — HypothesisGraph snapshot
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huginn.utils.common import atomic_write_json

# 跟随 P12/P13/P14 的 env flag 风格, 默认 off.
# ponytail: 不读 settings, 不加配置文件 — env var 跟现有 flag 一致.
_PERSISTENCE_FLAG = "HUGINN_USE_PERSISTENCE"
_SAVE_EVERY_FLAG = "HUGINN_ENGINE_STATE_SAVE_EVERY"
_DEFAULT_SAVE_EVERY = 10


def use_persistence() -> bool:
    """HUGINN_USE_PERSISTENCE=1 才开. 默认 off."""
    return os.environ.get(_PERSISTENCE_FLAG, "0") == "1"


def save_every_steps() -> int:
    """每 N 步触发一次 save. 默认 10, 0 表示禁用周期 save (仍保留 refute/pivot save)."""
    try:
        return max(0, int(os.environ.get(_SAVE_EVERY_FLAG, str(_DEFAULT_SAVE_EVERY))))
    except ValueError:
        return _DEFAULT_SAVE_EVERY


# 13 个 engine 运行时字段 + run_id + saved_at = 15
# ponytail: 字段名前缀保留下划线跟 engine 实例属性对齐, 加载时直接 setattr.
_ENGINE_FIELDS: tuple[str, ...] = (
    "_iteration",
    "_consecutive_failures",
    "_refine_count",
    "_pivot_count",
    "_next_phase_hint",
    "_refined_hypothesis",
    "_plan_check_history",
    "_plan_check_patterns",
    "_last_persona",
    "_last_surprise",
    "_evals_history",
    "_budget_rejects",
    "_budget_degraded",
)


@dataclass
class EngineState:
    """AutoloopEngine 运行时状态的快照.

    13 个 engine 实例字段 + cognitive_maps + run_id + saved_at = 16 字段.
    dataclass + asdict 直接 JSON 序列化, 不上 pickle — 跨版本/跨进程更稳.
    """

    _iteration: int = 0
    _consecutive_failures: int = 0
    _refine_count: int = 0
    _pivot_count: int = 0
    _next_phase_hint: str | None = None
    _refined_hypothesis: str | None = None
    _plan_check_history: list[dict[str, Any]] = field(default_factory=list)
    _plan_check_patterns: list[dict[str, Any]] = field(default_factory=list)
    _last_persona: str | None = None
    _last_surprise: float = 0.0
    _evals_history: list[Any] = field(default_factory=list)
    _budget_rejects: dict[str, int] = field(default_factory=dict)
    _budget_degraded: bool = False
    # P1: Structure Cognitive Map 持久化 (HUGINN_USE_COGNITIVE_MAP=1 时填充).
    # 不是 engine 实例属性, 不在 _ENGINE_FIELDS 里, save 时单独从 tool registry 拉.
    cognitive_maps: dict[str, dict] = field(default_factory=dict)
    # meta
    run_id: str = ""
    saved_at: float = 0.0


def _engine_state_dir(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / ".huginn" / "engine_state"


def _engine_state_path(workspace: str | Path, run_id: str) -> Path:
    return _engine_state_dir(workspace) / f"{run_id}.json"


def _hypothesis_graph_path(workspace: str | Path, run_id: str) -> Path:
    return Path(workspace).resolve() / ".huginn" / f"hypothesis_graph_{run_id}.json"


def _snapshot_engine(engine: Any, run_id: str) -> EngineState:
    """从 engine 实例读 13 字段 + run_id + saved_at, 返回 EngineState.

    getattr 缺失字段时用 dataclass 默认值, 兼容旧 engine 不带某些字段的场景.
    ponytail: 不做 isinstance(engine, AutoloopEngine) 检查 — duck typing 够.
    """
    defaults = EngineState()
    kwargs: dict[str, Any] = {"run_id": run_id, "saved_at": time.time()}
    for f in _ENGINE_FIELDS:
        kwargs[f] = getattr(engine, f, getattr(defaults, f))
    return EngineState(**kwargs)


def save_engine_state(
    engine: Any, run_id: str, workspace: str | Path
) -> "EngineState | None":
    """原子写 EngineState + 同步 hypothesis_graph.

    返回保存的 EngineState, 失败 (flag off / IO 异常) 返回 None.
    flag off 时直接 return None, 不碰磁盘.
    """
    if not use_persistence():
        return None
    if not run_id:
        return None
    try:
        state = _snapshot_engine(engine, run_id)
        # P1: Structure Cognitive Map — flag on 时从 tool registry 拉活跃 map.
        # 跟 hypothesis_graph 同范式: try/except, 失败不阻塞 engine_state save.
        if os.environ.get("HUGINN_USE_COGNITIVE_MAP", "0") == "1":
            try:
                from huginn.tools import structure_cognitive_map_tool as _cm
                state.cognitive_maps = {
                    mid: m.to_engine_state_dict() for mid, m in _cm._MAPS.items()
                }
            except Exception:
                import logging
                logging.getLogger(__name__).debug(
                    "cognitive_maps serialization failed (non-fatal)", exc_info=True,
                )
        atomic_write_json(_engine_state_path(workspace, run_id), asdict(state))
        # hypothesis_graph 同步落盘 — refuted/supported 状态跨 session 必须保留.
        # graph.save 自己处理 flag off (返 None) 和异常 (返 None).
        graph = getattr(engine, "hypothesis_graph", None)
        if graph is not None and hasattr(graph, "save"):
            try:
                graph.save(_hypothesis_graph_path(workspace, run_id))
            except Exception:
                # graph save 失败不阻塞 engine_state save — engine_state 自身可独立恢复
                import logging
                logging.getLogger(__name__).debug(
                    "hypothesis_graph.save failed (non-fatal)", exc_info=True,
                )
        return state
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "save_engine_state failed for run_id=%s", run_id, exc_info=True,
        )
        return None


def load_engine_state(
    run_id: str, workspace: str | Path
) -> "EngineState | None":
    """读 <workspace>/.huginn/engine_state/<run_id>.json 反序列化.

    文件不存在 / flag off / 解析失败均返回 None.
    """
    if not use_persistence():
        return None
    if not run_id:
        return None
    path = _engine_state_path(workspace, run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 字段缺失时用 dataclass 默认值, 兼容旧格式 (新增字段不破老 snapshot)
        defaults = EngineState()
        kwargs = {f: data.get(f, getattr(defaults, f)) for f in _ENGINE_FIELDS}
        # cognitive_maps 不在 _ENGINE_FIELDS 里 (不是 engine 实例属性), 单独读.
        kwargs["cognitive_maps"] = data.get("cognitive_maps", {})
        kwargs["run_id"] = data.get("run_id", run_id)
        kwargs["saved_at"] = data.get("saved_at", 0.0)
        return EngineState(**kwargs)
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "load_engine_state failed for run_id=%s", run_id, exc_info=True,
        )
        return None


def apply_state_to_engine(state: EngineState, engine: Any) -> None:
    """把 EngineState 字段写回 engine 实例 (setattr).

    用于 __init__ resume_from_state 流程. 不写 run_id / saved_at (meta).
    ponytail: 不判断 engine 类型, duck typing; 字段名跟 engine 实例属性对齐.
    """
    for f in _ENGINE_FIELDS:
        try:
            setattr(engine, f, getattr(state, f))
        except Exception:
            # 某些字段可能是 property / read-only, 跳过不阻塞其他字段
            import logging
            logging.getLogger(__name__).debug(
                "apply_state_to_engine setattr %s failed (non-fatal)", f,
                exc_info=True,
            )


def engine_state_digest(state: EngineState) -> str:
    """EngineState 的 8 位短 hash, 给 Checkpoint.engine_state_digest 用.

    ponytail: sha256 前 8 位, 跟 objective_hash 同范式. 不做加密用.
    """
    import hashlib

    payload = json.dumps(asdict(state), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


# ── selfcheck ──────────────────────────────────────────────────────────────


def _selfcheck() -> None:
    """3 场景: save+load round-trip / load missing / atomic write 不留 partial."""
    import os
    import shutil
    import tempfile as _tf

    # 强制开 flag — selfcheck 不受 env 影响
    prev = os.environ.get(_PERSISTENCE_FLAG)
    os.environ[_PERSISTENCE_FLAG] = "1"
    ws = Path(_tf.mkdtemp(prefix="huginn_es_test_")) / "ws"
    ws.mkdir()
    try:
        # ── 场景 1: save → load round-trip ──────────────────────────
        class _FakeEngine:
            """最小 engine stub, 只暴露 13 个字段 + hypothesis_graph."""

            def __init__(self):
                self._iteration = 7
                self._consecutive_failures = 3
                self._refine_count = 2
                self._pivot_count = 1
                self._next_phase_hint = "execute"
                self._refined_hypothesis = "h_new"
                self._plan_check_history = [{"phase": "plan", "ok": False}]
                self._plan_check_patterns = [{"pat": "missing_test"}]
                self._last_persona = "domain_skeptic"
                self._last_surprise = 0.42
                self._evals_history = [{"step": 1, "score": 0.3}]
                self._budget_rejects = {"tier1": 2}
                self._budget_degraded = True
                # hypothesis_graph stub
                self.hypothesis_graph = None

        eng = _FakeEngine()
        saved = save_engine_state(eng, "loop_abc123", ws)
        assert saved is not None, "save returned None with flag on"
        assert saved.run_id == "loop_abc123"
        assert saved._iteration == 7
        assert saved._pivot_count == 1
        assert saved._budget_degraded is True
        # 文件落到正确路径
        assert _engine_state_path(ws, "loop_abc123").exists(), "snapshot not written"

        loaded = load_engine_state("loop_abc123", ws)
        assert loaded is not None, "load returned None after save"
        assert loaded._iteration == 7
        assert loaded._consecutive_failures == 3
        assert loaded._refine_count == 2
        assert loaded._pivot_count == 1
        assert loaded._next_phase_hint == "execute"
        assert loaded._refined_hypothesis == "h_new"
        assert loaded._plan_check_history == [{"phase": "plan", "ok": False}]
        assert loaded._plan_check_patterns == [{"pat": "missing_test"}]
        assert loaded._last_persona == "domain_skeptic"
        assert loaded._last_surprise == 0.42
        assert loaded._evals_history == [{"step": 1, "score": 0.3}]
        assert loaded._budget_rejects == {"tier1": 2}
        assert loaded._budget_degraded is True
        assert loaded.run_id == "loop_abc123"
        assert loaded.saved_at == saved.saved_at
        print("1. save+load round-trip OK")

        # ── 场景 2: load missing → None ─────────────────────────────
        missing = load_engine_state("loop_does_not_exist", ws)
        assert missing is None, "load missing file should return None"
        print("2. load missing → None OK")

        # ── 场景 3: atomic write 不留 partial ───────────────────────
        # 模拟 crash: 在 save_engine_state 中间不让 atomic_write_json 失败.
        # atomic_write_json 用 tmp + os.replace, 即便 process 在 rename 前被杀,
        # 也只会留 .tmp 文件, 原文件 (如果存在) 不破. 这里验: 第二次 save 覆盖
        # 第一次, 中间无 partial <run_id>.json 残缺.
        eng._iteration = 99
        save_engine_state(eng, "loop_overwrite", ws)
        path = _engine_state_path(ws, "loop_overwrite")
        # 写完后立刻读, 内容必须完整 (JSON parse 成功)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["_iteration"] == 99
        # 不应该有 .tmp 文件残留 (atomic_write_json 内部清理)
        tmp_files = list(path.parent.glob("*.tmp"))
        assert not tmp_files, f"partial .tmp files leaked: {tmp_files}"
        # 二次覆盖后字段正确
        eng._iteration = 100
        save_engine_state(eng, "loop_overwrite", ws)
        data2 = json.loads(path.read_text(encoding="utf-8"))
        assert data2["_iteration"] == 100
        print("3. atomic write no partial OK")

        # ── 场景 4: flag off 时 save/load 全部 no-op ────────────────
        os.environ[_PERSISTENCE_FLAG] = "0"
        flag_off_saved = save_engine_state(eng, "loop_off", ws)
        assert flag_off_saved is None, "flag off should return None from save"
        flag_off_loaded = load_engine_state("loop_off", ws)
        assert flag_off_loaded is None, "flag off should return None from load"
        assert not _engine_state_path(ws, "loop_off").exists(), \
            "flag off should not write file"
        print("4. flag off no-op OK")

        # ── 场景 5: apply_state_to_engine 写回字段 ───────────────────
        os.environ[_PERSISTENCE_FLAG] = "1"
        eng2 = _FakeEngine()
        eng2._iteration = 0
        eng2._pivot_count = 0
        loaded2 = load_engine_state("loop_abc123", ws)
        apply_state_to_engine(loaded2, eng2)
        assert eng2._iteration == 7
        assert eng2._pivot_count == 1
        assert eng2._last_persona == "domain_skeptic"
        print("5. apply_state_to_engine OK")

        # ── 场景 6: engine_state_digest 稳定 ─────────────────────────
        d1 = engine_state_digest(loaded2)
        d2 = engine_state_digest(loaded2)
        assert d1 == d2, "digest should be stable for same state"
        assert len(d1) == 8
        eng3 = _FakeEngine()
        eng3._iteration = 999
        s3 = _snapshot_engine(eng3, "loop_x")
        d3 = engine_state_digest(s3)
        assert d3 != d1, "digest should differ for different state"
        print("6. engine_state_digest stable + distinct OK")

        print("ALL CHECKS PASSED")
    finally:
        if prev is None:
            os.environ.pop(_PERSISTENCE_FLAG, None)
        else:
            os.environ[_PERSISTENCE_FLAG] = prev
        shutil.rmtree(ws.parent, ignore_errors=True)


if __name__ == "__main__":
    _selfcheck()
