"""Agent checkpoint — snapshot cognitive state for resume-after-crash.

Distinct from huginn.workflows.checkpoint (which snapshots computational
pipeline stage outputs). This one captures the agent's runtime state:
memory cursor, target chain progress, pending prospective intentions, and
the audit chain head at save time so resume can detect tampering.

Layout: <workspace>/.huginn/checkpoints/<task_id>/step_<N>.json
Retention: most recent 3 + one milestone every 10 steps.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from huginn.utils.common import atomic_write_json

from huginn.events.audit_log import verify_audit_chain

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


@dataclass
class Checkpoint:
    task_id: str
    step_id: int
    phase: str  # "execute" / "validate" / "report"
    context_digest: str  # hash of compressed context; full context lives elsewhere
    memory_cursor: str | None  # LongTermMemory entry_id of last appended entry
    target_chain_progress: dict[str, float]  # target_id -> 0.0..1.0
    prospective_queue: list[str]  # pending intention_ids
    audit_hash_head: str  # audit.jsonl chain head at save time
    saved_at: float  # epoch seconds
    # P15: EngineState 的 8 位 hash, resume 时跟重算的 digest 比对, 防 drift.
    # None 表示该 checkpoint 没绑 engine_state (老格式 / 持久化未启用), 跳过校验.
    engine_state_digest: str | None = None


def _checkpoint_dir(workspace: Path, task_id: str) -> Path:
    return Path(workspace).resolve() / ".huginn" / "checkpoints" / task_id


def _checkpoint_path(workspace: Path, task_id: str, step_id: int) -> Path:
    return _checkpoint_dir(workspace, task_id) / f"step_{step_id}.json"


def _audit_jsonl_path(workspace: Path) -> Path:
    # In prod HUGINN_CACHE_DIR redirects audit_log to this same path; we just
    # construct it directly from workspace so the head reader and chain
    # verifier always agree on the file.
    return Path(workspace).resolve() / ".huginn_cache" / "events" / "audit.jsonl"


def _from_dict(data: dict) -> Checkpoint:
    return Checkpoint(**data)


def save_checkpoint(
    task_id: str,
    step_id: int,
    phase: str,
    workspace: Path,
    context_digest: str,
    memory_cursor: str | None,
    target_chain_progress: dict[str, float],
    prospective_queue: list[str],
) -> Checkpoint:
    """Persist a checkpoint for the given step, return the saved object."""
    cp = Checkpoint(
        task_id=task_id,
        step_id=step_id,
        phase=phase,
        context_digest=context_digest,
        memory_cursor=memory_cursor,
        target_chain_progress=dict(target_chain_progress),
        prospective_queue=list(prospective_queue),
        audit_hash_head=_get_audit_hash_head(workspace),
        saved_at=time.time(),
    )
    atomic_write_json(_checkpoint_path(workspace, task_id, step_id), asdict(cp))
    _prune_checkpoints(task_id, workspace)
    return cp


def load_checkpoint(task_id: str, workspace: Path, step_id: int | None = None) -> Checkpoint | None:
    """Load a checkpoint by step, or the latest if step_id is None. None if missing."""
    if step_id is not None:
        path = _checkpoint_path(workspace, task_id, step_id)
        if not path.exists():
            return None
        return _from_dict(json.loads(path.read_text(encoding="utf-8")))
    cps = list_checkpoints(task_id, workspace)
    return cps[-1] if cps else None


def list_checkpoints(task_id: str, workspace: Path) -> list[Checkpoint]:
    """All checkpoints for task, ascending by step_id."""
    d = _checkpoint_dir(workspace, task_id)
    if not d.exists():
        return []
    out: list[Checkpoint] = []
    for p in d.glob("step_*.json"):
        try:
            out.append(_from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    out.sort(key=lambda c: c.step_id)
    return out


def resume_from_checkpoint(checkpoint: Checkpoint, workspace: Path) -> int:
    """Verify audit chain + head match, return next step_id (step_id + 1).

    Raises RuntimeError if the chain is broken or the head has moved since
    the checkpoint was saved. The head check is the tamper signal: a
    checkpoint saved mid-crash has its head match the on-disk head; any
    post-save mutation (legitimate or hostile) breaks the match.
    """
    audit_path = _audit_jsonl_path(workspace)
    if not verify_audit_chain(audit_path):
        raise RuntimeError("audit chain verification failed")
    current = _get_audit_hash_head(workspace)
    if current != checkpoint.audit_hash_head:
        raise RuntimeError(
            f"audit head mismatch: checkpoint={checkpoint.audit_hash_head[:16]} "
            f"current={current[:16]}"
        )
    return checkpoint.step_id + 1


def resume_engine_from_checkpoint(
    checkpoint: Checkpoint,
    workspace: Path,
    run_id: str | None = None,
) -> Any:
    """从 Checkpoint 重建 AutoloopEngine, 恢复 engine_state + hypothesis_graph.

    1. run_id 默认从 checkpoint.task_id 取 (autoloop 把 run_id 当 task_id 用).
    2. load_engine_state 读 <workspace>/.huginn/engine_state/<run_id>.json.
       flag off / 文件缺失 → 返回未 resume 的 engine (不报错).
    3. checkpoint.engine_state_digest 跟重算 digest 比对, 不一致 log warning
       (仍继续 resume, 不阻塞 — drift 信号给调用方, 不强制 fail).
    4. 创建 AutoloopEngine(resume_from_state=run_id), engine.__init__ 内部
       调 apply_state_to_engine + hypothesis_graph.load 完成恢复.

    ponytail: AutoloopEngine import 放函数内, 避免模块级循环 import.
    ceiling: 单进程单 run_id resume; 多任务并发需外部 run_id 隔离 (已满足).
    """
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.runtime.engine_state import (
        engine_state_digest, load_engine_state,
    )

    resolved_run_id = run_id or checkpoint.task_id
    state = load_engine_state(resolved_run_id, workspace)
    if state is None:
        # flag off 或 snapshot 不存在 — 返回普通 engine, 走原 init 路径
        logger.info(
            "resume_engine_from_checkpoint: no engine_state for run_id=%s, "
            "returning fresh engine", resolved_run_id,
        )
        return AutoloopEngine(workspace=workspace)

    # digest drift 检测 — 不阻塞 resume, 只 log
    if checkpoint.engine_state_digest is not None:
        actual = engine_state_digest(state)
        if actual != checkpoint.engine_state_digest:
            logger.warning(
                "engine_state digest drift: checkpoint=%s reloaded=%s "
                "(run_id=%s) — engine_state 可能已被外部修改",
                checkpoint.engine_state_digest, actual, resolved_run_id,
            )

    return AutoloopEngine(
        workspace=workspace, resume_from_state=resolved_run_id,
    )


def _prune_checkpoints(
    task_id: str,
    workspace: Path,
    keep_recent: int = 3,
    milestone_every: int = 10,
) -> None:
    """Keep most recent `keep_recent` + every `milestone_every`-th step. Drop the rest."""
    cps = list_checkpoints(task_id, workspace)
    if len(cps) <= keep_recent:
        return
    keep: set[int] = {cp.step_id for cp in cps[-keep_recent:]}
    for cp in cps:
        if cp.step_id % milestone_every == 0:
            keep.add(cp.step_id)
    for cp in cps:
        if cp.step_id not in keep:
            p = _checkpoint_path(workspace, task_id, cp.step_id)
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def _get_audit_hash_head(workspace: Path) -> str:
    """Read the last _hash from audit.jsonl. Missing file → genesis '0'*64."""
    audit_path = _audit_jsonl_path(workspace)
    if not audit_path.exists():
        return GENESIS_HASH
    last = GENESIS_HASH
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = rec.get("_hash")
            if h:
                last = h
    return last


if __name__ == "__main__":
    import shutil
    import tempfile as _tf

    from huginn.events.audit_log import _compute_hash

    ws = Path(_tf.mkdtemp(prefix="huginn_cp_test_")) / "ws"
    ws.mkdir()
    try:
        # 1. save → load roundtrip
        cp = save_checkpoint(
            task_id="t1", step_id=1, phase="execute", workspace=ws,
            context_digest="abc123", memory_cursor="entry_5",
            target_chain_progress={"t_a": 0.5, "t_b": 0.0},
            prospective_queue=["i1", "i2"],
        )
        loaded = load_checkpoint("t1", ws, step_id=1)
        assert loaded is not None, "load returned None"
        assert loaded.task_id == "t1"
        assert loaded.step_id == 1
        assert loaded.phase == "execute"
        assert loaded.context_digest == "abc123"
        assert loaded.memory_cursor == "entry_5"
        assert loaded.target_chain_progress == {"t_a": 0.5, "t_b": 0.0}
        assert loaded.prospective_queue == ["i1", "i2"]
        assert loaded.audit_hash_head == GENESIS_HASH  # no audit.jsonl → genesis
        assert loaded.saved_at == cp.saved_at
        # input mutation must not leak into the stored checkpoint
        loaded.target_chain_progress["t_a"] = 9.9
        again = load_checkpoint("t1", ws, step_id=1)
        assert again is not None and again.target_chain_progress["t_a"] == 0.5
        print("1. roundtrip OK")

        # load latest (step_id=None) returns highest step
        save_checkpoint(
            task_id="t1", step_id=2, phase="validate", workspace=ws,
            context_digest="def", memory_cursor=None,
            target_chain_progress={"t_a": 0.5}, prospective_queue=[],
        )
        latest = load_checkpoint("t1", ws)
        assert latest is not None and latest.step_id == 2
        print("2. load latest OK")

        # 3. audit hash head is genesis when audit.jsonl absent
        assert _get_audit_hash_head(ws) == GENESIS_HASH
        print("3. genesis head OK")

        # 4. resume returns step_id + 1
        nxt = resume_from_checkpoint(loaded, ws)
        assert nxt == 2, f"expected 2, got {nxt}"
        print("4. resume returns step+1 OK")

        # 5. retention: save steps 3..15 for task t2, expect {10, 13, 14, 15} kept
        for n in range(3, 16):
            save_checkpoint(
                task_id="t2", step_id=n, phase="execute", workspace=ws,
                context_digest="x", memory_cursor=None,
                target_chain_progress={}, prospective_queue=[],
            )
        kept = sorted(c.step_id for c in list_checkpoints("t2", ws))
        # recent 3 = {13,14,15}, milestone every 10 = {10}
        assert kept == [10, 13, 14, 15], f"retention wrong: {kept}"
        print(f"5. retention OK: kept={kept}")

        # 6. tamper detection — write a valid audit record so the chain stays
        # intact but the head advances past genesis; head mismatch must raise.
        audit_path = _audit_jsonl_path(ws)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        prev = GENESIS_HASH
        record = {"type": "test_event", "_prev_hash": prev}
        record["_hash"] = _compute_hash(record, prev)
        audit_path.write_text(
            json.dumps(record, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        # chain is valid (single well-formed record), but head moved
        assert verify_audit_chain(audit_path), "setup: chain should be valid"
        assert _get_audit_hash_head(ws) != GENESIS_HASH, "setup: head should have moved"
        try:
            resume_from_checkpoint(loaded, ws)
            raise AssertionError("expected RuntimeError on head mismatch")
        except RuntimeError:
            pass
        print("6. tamper/head-mismatch detection OK")

        # cleanup audit.jsonl so the workspace is clean for any re-run
        try:
            audit_path.unlink()
        except FileNotFoundError:
            pass

        # 7. P15: engine_state_digest 字段 — 默认 None, 显式赋值后 round-trip 保留
        from huginn.runtime.engine_state import engine_state_digest, EngineState
        cp_digest = save_checkpoint(
            task_id="t3", step_id=1, phase="execute", workspace=ws,
            context_digest="digest_xyz", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
        )
        # 默认 None (向后兼容 — 老 checkpoint 没这个字段)
        assert cp_digest.engine_state_digest is None, \
            "engine_state_digest default should be None"
        loaded_no_digest = load_checkpoint("t3", ws, step_id=1)
        assert loaded_no_digest is not None
        assert loaded_no_digest.engine_state_digest is None
        # 显式赋值后 round-trip 保留
        cp_digest.engine_state_digest = "abcd1234"
        # 重写文件 (模拟 P15 save trigger 把 digest 写进去)
        atomic_write_json(
            _checkpoint_path(ws, "t3", 1),
            asdict(cp_digest),
        )
        loaded_with_digest = load_checkpoint("t3", ws, step_id=1)
        assert loaded_with_digest is not None
        assert loaded_with_digest.engine_state_digest == "abcd1234", \
            f"digest round-trip failed: {loaded_with_digest.engine_state_digest}"
        print("7. engine_state_digest round-trip OK")

        # 8. engine_state_digest 跟 EngineState 计算结果一致
        state = EngineState(_iteration=5, _pivot_count=2, run_id="loop_test")
        d = engine_state_digest(state)
        assert len(d) == 8
        cp2 = save_checkpoint(
            task_id="t4", step_id=1, phase="execute", workspace=ws,
            context_digest="x", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
        )
        cp2.engine_state_digest = d
        atomic_write_json(_checkpoint_path(ws, "t4", 1), asdict(cp2))
        loaded_cp2 = load_checkpoint("t4", ws, step_id=1)
        assert loaded_cp2.engine_state_digest == d
        print("8. engine_state_digest matches EngineState OK")

        # 9. H3: resume_engine_from_checkpoint — no engine_state → fresh engine.
        # Mock AutoloopEngine 避免重 init (model/memory/kg 副作用). 验证:
        # - 无 engine_state 文件时返回 AutoloopEngine(workspace=ws) (无 resume_from_state)
        # - 有 engine_state 文件时返回 AutoloopEngine(workspace=ws, resume_from_state=run_id)
        import huginn.autoloop.engine as _eng_mod
        _orig_AE = _eng_mod.AutoloopEngine
        _calls: list[dict] = []
        class _StubAE:
            def __init__(self, **kw):
                _calls.append(kw)
                self.workspace = kw.get("workspace")
        _eng_mod.AutoloopEngine = _StubAE
        try:
            # 无 engine_state → fresh engine
            cp_no_state = save_checkpoint(
                task_id="t5", step_id=1, phase="execute", workspace=ws,
                context_digest="x", memory_cursor=None,
                target_chain_progress={}, prospective_queue=[],
            )
            eng1 = resume_engine_from_checkpoint(cp_no_state, ws)
            assert isinstance(eng1, _StubAE), "应返回 stub 实例"
            assert len(_calls) == 1, f"应调 1 次 AutoloopEngine, got {len(_calls)}"
            assert _calls[0].get("workspace") == ws, "workspace 应传入"
            assert _calls[0].get("resume_from_state") is None, \
                "无 engine_state 时不应传 resume_from_state"
            print("9a. resume_engine_from_checkpoint (no state → fresh) OK")

            # 有 engine_state → resumed engine (带 resume_from_state)
            # 需 HUGINN_USE_PERSISTENCE=1, 否则 load_engine_state 直接返 None.
            import os as _os9
            _saved_persist = _os9.environ.get("HUGINN_USE_PERSISTENCE")
            _os9.environ["HUGINN_USE_PERSISTENCE"] = "1"
            from huginn.runtime.engine_state import save_engine_state, EngineState
            # save_engine_state 第一个参数是 engine (有 _iteration 等属性), 不是 state.
            # ponytail: 直接用 EngineState 当 engine — _snapshot_engine 走 getattr 路径.
            save_engine_state(state, "t5", ws)
            _calls.clear()
            try:
                eng2 = resume_engine_from_checkpoint(cp_no_state, ws)
                assert isinstance(eng2, _StubAE)
                assert len(_calls) == 1
                assert _calls[0].get("workspace") == ws
                assert _calls[0].get("resume_from_state") == "t5", \
                    f"有 engine_state 时应传 resume_from_state=t5, got {_calls[0]}"
                print("9b. resume_engine_from_checkpoint (with state → resumed) OK")

                # digest drift → log warning 但仍 resume (不阻塞)
                cp_drift = save_checkpoint(
                    task_id="t5", step_id=2, phase="execute", workspace=ws,
                    context_digest="x", memory_cursor=None,
                    target_chain_progress={}, prospective_queue=[],
                )
                cp_drift.engine_state_digest = "deadbeef"  # 故意 wrong
                _calls.clear()
                eng3 = resume_engine_from_checkpoint(cp_drift, ws)
                assert isinstance(eng3, _StubAE), "drift 不应阻塞 resume"
                assert len(_calls) == 1
                print("9c. resume_engine_from_checkpoint (digest drift → warn but resume) OK")
            finally:
                if _saved_persist is None:
                    _os9.environ.pop("HUGINN_USE_PERSISTENCE", None)
                else:
                    _os9.environ["HUGINN_USE_PERSISTENCE"] = _saved_persist
        finally:
            _eng_mod.AutoloopEngine = _orig_AE

        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)
