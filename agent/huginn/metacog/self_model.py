"""SelfModel — agent 对自己能力的 internal model.

v15 Phase 5: 跟 world-model (HypothesisManifold) co-evolve. 不是孤立的能力
清单, 是被主循环每步执行结果持续更新的动态 model. 失败一律降级, 不阻塞.

三档分类 (capable / uncertain / blind):
  - capable:  agent 能做到 (e.g. 解析 CSV, 调 matplotlib)
  - uncertain: agent 不确定能做到 (e.g. 复现复杂 DL 模型)
  - blind:    agent 做不到 (e.g. 安装大型软件包, 长训练, GPU 不可用)

跨 task 累积: task-local self_model.json 跟 cross-task self_model_cross_task.json
合并, cross-task 的 capable/blind 优先 (历史经验优先于本 task 试探).

ponytail: 单文件单类, stdlib + 可选 LLM. 不引新依赖, 不上 embedding. 升级路径:
LLM semantic classify + per-skill 时间衰减 + Bayesian tier posterior.
"""
from __future__ import annotations

# 直接跑脚本时把 agent/ 加到 sys.path (被 import 时不执行, rcb_runner 已设好)
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _agent_root = str(_Path(__file__).resolve().parents[2])
    if _agent_root not in _sys.path:
        _sys.path.insert(0, _agent_root)

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 三档分类
TIERS = ("capable", "uncertain", "blind")

# 连续成功/失败多少次触发升降级
_PROMOTE_THRESHOLD = 3  # uncertain/blind → capable after N successes
_DEMOTE_THRESHOLD = 3   # uncertain → blind after N failures

# keyword 规则匹配 — LLM 失败时降级用. (keywords, skill, tier).
# ponytail: keyword 匹配, 不上 semantic. 同义改写漏检是已知天花板,
# 升级路径: TF-IDF + cosine 跟 hint_coordinator._keyword_overlap 共用基础设施.
_KEYWORD_RULES: list[tuple[tuple[str, ...], str, str]] = [
    # 资源不足 → blind (agent 物理上做不到)
    (("timeout", "timed out", "deadline exceeded"), "exec_timeout", "blind"),
    (("out of memory", "oom", "cuda oom", "memoryerror"), "memory_limit", "blind"),
    (("disk full", "no space left"), "disk_space", "blind"),
    (("permission denied", "access denied"), "permission_denied", "blind"),
    # 包安装 → blind (RCB 沙箱一般不让装)
    (("no module named", "importerror", "modulenotfound"), "pkg_install", "blind"),
    (("cannot install", "pip install failed", "build wheel failed"), "pkg_install", "blind"),
    # 常用 stdlib/工具 → capable
    (("matplotlib", "plt."), "matplotlib", "capable"),
    (("pandas", "dataframe", "pd.read"), "pandas", "capable"),
    (("numpy", "np.array"), "numpy", "capable"),
    (("scipy",), "scipy", "capable"),
    (("sklearn", "scikit"), "sklearn", "capable"),
    # 重型 sim / DL → uncertain (能跑但容易翻车)
    (("rdkit",), "rdkit", "uncertain"),
    (("openmm",), "openmm", "uncertain"),
    (("vasp", "incar"), "vasp", "uncertain"),
    (("gaussian", "g16"), "gaussian", "uncertain"),
    (("lammps",), "lammps", "uncertain"),
    (("qe", "quantum espresso"), "qe", "uncertain"),
    (("pytorch", "torch.cuda"), "pytorch_training", "uncertain"),
    (("tensorflow", "tf.keras"), "tensorflow_training", "uncertain"),
    (("training", "epoch", "convergence"), "dl_training", "uncertain"),
]


@dataclass
class SkillRecord:
    """单条能力记录."""
    skill: str
    tier: str  # capable / uncertain / blind
    reason: str = ""
    last_updated: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SkillRecord:
        return cls(
            skill=str(d.get("skill", "")),
            tier=str(d.get("tier", "uncertain")),
            reason=str(d.get("reason", "")),
            last_updated=float(d.get("last_updated", 0.0) or 0.0),
            success_count=int(d.get("success_count", 0) or 0),
            failure_count=int(d.get("failure_count", 0) or 0),
        )


class SelfModel:
    """agent 对自己能力的 internal model. 三档分类 + 跨 task 累积.

    持久化:
      - task-local: <ws>/.huginn/self_model.json (本 task 累积)
      - cross-task: <cross_task_dir>/self_model_cross_task.json (跨 task 累积,
        只存 capable/blind, 不存 uncertain)
    加载时合并, cross-task 的 capable/blind 优先 (历史经验优先于本 task 试探).

    ponytail: 不上 embedding, 不上 SQLite. JSON 文件 + stdlib. 升级路径:
    CrossTaskStore 共用 SQLite, self_model 表 + JOIN.
    """

    def __init__(
        self,
        task_local_path: Path | None = None,
        cross_task_path: Path | None = None,
        model: Any = None,
    ):
        self.task_local_path = Path(task_local_path) if task_local_path else None
        self.cross_task_path = Path(cross_task_path) if cross_task_path else None
        self.model = model
        # skill -> SkillRecord
        self._skills: dict[str, SkillRecord] = {}
        self._load()

    # ---- public API ----

    def update_from_step(
        self,
        step_result: dict | list[dict] | None,
        task_context: str = "",
    ) -> None:
        """从一步执行结果更新 self-model. 失败降级静默, 不阻塞主循环.

        step_result:
          - dict: 单条工具调用 {tool_name, success, error_message, duration, ...}
          - list[dict]: 多条工具调用
          - None / 空 list: 跳过 (空 step)
        task_context: checklist / attempted text, 给 LLM 分类用.
        """
        if step_result is None:
            return
        if isinstance(step_result, dict):
            records = [step_result]
        elif isinstance(step_result, list):
            records = step_result
        else:
            return
        for rec in records:
            if not isinstance(rec, dict):
                continue
            try:
                self._update_one(rec, task_context)
            except Exception as e:
                logger.debug("self_model update_one failed: %s", e)
        try:
            self._save()
        except Exception as e:
            logger.debug("self_model save failed: %s", e)

    def get_tier(self, skill: str) -> str:
        """查询某 skill 的 tier. 未知返回 'uncertain' (保守)."""
        rec = self._skills.get(skill)
        return rec.tier if rec is not None else "uncertain"

    def get_skill(self, skill: str) -> SkillRecord | None:
        return self._skills.get(skill)

    def list_by_tier(self, tier: str) -> list[SkillRecord]:
        if tier not in TIERS:
            return []
        return [r for r in self._skills.values() if r.tier == tier]

    def feedback_from_imagination(self, skill: str, success: bool) -> None:
        """Task 12.3: imagination 结果反馈到 self-model.

        成功 → uncertain 升级 capable (绕过盲点 = 实际能做)
        失败 → uncertain 降级 blind (确认做不到)
        capable/blind 不动 (已有定论, 一次 imagination 不翻案).
        失败降级静默.
        """
        try:
            rec = self._skills.get(skill)
            if rec is None:
                return
            if rec.tier == "uncertain":
                rec.tier = "capable" if success else "blind"
                rec.last_updated = time.time()
                rec.reason = (
                    "imagination success -> capable"
                    if success else
                    "imagination failure -> blind"
                )
                self._save()
        except Exception as e:
            logger.debug("self_model feedback failed: %s", e)

    def to_dict(self) -> dict:
        return {
            "skills": [r.to_dict() for r in self._skills.values()],
            "last_updated": time.time(),
        }

    # ---- internals ----

    def _update_one(self, rec: dict, task_context: str) -> None:
        tool_name = str(rec.get("tool_name") or rec.get("tool") or "")
        success = bool(rec.get("success", True))
        err = str(rec.get("error_message") or rec.get("error") or "")
        # duration 当前不参与 tier 决策, 仅记录. 升级路径: 长 duration → uncertain.
        # _ = float(rec.get("duration") or 0.0)

        # 调 LLM 分类 (capable/uncertain/blind + skill 名), 失败降级 keyword
        tier, reason, skill = self._classify(tool_name, success, err, task_context)
        if not skill:
            skill = tool_name or "unknown"

        existing = self._skills.get(skill)
        if existing is None:
            self._skills[skill] = SkillRecord(
                skill=skill,
                tier=tier,
                reason=reason,
                last_updated=time.time(),
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
            )
            return

        existing.last_updated = time.time()
        if success:
            existing.success_count += 1
            # 连续成功 → 升级 capable (blind 也升 uncertain, 给一次重新试探机会)
            if existing.tier == "blind" and existing.failure_count == 0:
                existing.tier = "uncertain"
                existing.reason = "blind -> uncertain after success"
            elif existing.tier != "capable" and existing.success_count >= _PROMOTE_THRESHOLD:
                existing.tier = "capable"
                existing.reason = (
                    f"promoted after {existing.success_count} successes"
                )
        else:
            existing.failure_count += 1
            # uncertain 连续失败 → 降级 blind
            if existing.tier == "uncertain" and existing.failure_count >= _DEMOTE_THRESHOLD:
                existing.tier = "blind"
                existing.reason = (
                    f"demoted after {existing.failure_count} failures"
                )
        # LLM/keyword 给的 tier 优先于规则推断 (单次失败信息更准)
        if tier != "uncertain":
            existing.tier = tier
            existing.reason = reason

    def _classify(
        self,
        tool_name: str,
        success: bool,
        err: str,
        task_context: str,
    ) -> tuple[str, str, str]:
        """返回 (tier, reason, skill). LLM 失败降级 keyword 规则."""
        # 成功 + 无错误 → capable
        if success and not err:
            return "capable", "tool call succeeded", tool_name
        # LLM 分类
        if self.model is not None:
            try:
                tier, reason, skill = self._llm_classify(tool_name, err, task_context)
                if tier in TIERS:
                    return tier, reason, skill
            except Exception as e:
                logger.debug("self_model llm_classify fallback: %s", e)
        # keyword 规则降级
        return self._keyword_classify(tool_name, err)

    def _llm_classify(
        self,
        tool_name: str,
        err: str,
        task_context: str,
    ) -> tuple[str, str, str]:
        from huginn.metacog.step_evaluator import _build_messages, _resp_to_text
        sys_text = (
            "You classify an agent's tool-call failure into one of three tiers "
            "of the agent's self-capability:\n"
            "- capable: agent can do this normally (transient error, retry will fix)\n"
            "- uncertain: agent is unsure if it can do this (complex setup, "
            "depends on environment)\n"
            "- blind: agent cannot do this (resource limit, missing install, "
            "long training, hardware unavailable)\n"
            'Return STRICT JSON: {"tier": "capable"|"uncertain"|"blind", '
            '"reason": <short string>, "skill": <short string>}\n'
            "skill is the capability name (e.g. 'matplotlib', 'pytorch_training', "
            "'vasp', 'pkg_install'). Return JSON only, no prose."
        )
        usr = (
            f"tool_name: {tool_name}\n"
            f"error_message: {err[:500]}\n"
            f"task_context: {task_context[:300]}\n"
            f"\nClassify this failure."
        )
        messages = _build_messages(sys_text, usr)
        if hasattr(self.model, "invoke"):
            text = _resp_to_text(self.model.invoke(messages))
        else:
            raise ValueError("model has no sync invoke")
        obj = _parse_first_json(text)
        if obj is None:
            raise ValueError("llm returned non-json")
        return (
            str(obj.get("tier", "uncertain")),
            str(obj.get("reason", "")),
            str(obj.get("skill", "")),
        )

    def _keyword_classify(self, tool_name: str, err: str) -> tuple[str, str, str]:
        """keyword 规则匹配 error_message. LLM 失败时降级用.

        ponytail: 关键词匹配, 不上 semantic. 同义改写漏检是已知天花板.
        """
        err_lower = (err or "").lower()
        text = f"{tool_name} {err_lower}"
        for keywords, skill, tier in _KEYWORD_RULES:
            if isinstance(keywords, str):
                keywords = (keywords,)
            for kw in keywords:
                if kw.lower() in text:
                    return tier, f"keyword match: {kw}", skill
        # 没匹配到 → 默认 uncertain + tool_name 作 skill
        return (
            "uncertain",
            "no keyword match, default uncertain",
            tool_name or "unknown",
        )

    def _save(self) -> None:
        """持久化到 task-local + cross-task. 失败静默."""
        data = self.to_dict()
        if self.task_local_path is not None:
            try:
                self.task_local_path.parent.mkdir(parents=True, exist_ok=True)
                self.task_local_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.debug("self_model task_local save failed: %s", e)
        if self.cross_task_path is not None:
            try:
                self.cross_task_path.parent.mkdir(parents=True, exist_ok=True)
                # cross-task: 只持久化 capable + blind (历史经验)
                cross_data = {
                    "skills": [
                        r.to_dict() for r in self._skills.values()
                        if r.tier in ("capable", "blind")
                    ],
                    "last_updated": time.time(),
                }
                self.cross_task_path.write_text(
                    json.dumps(cross_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.debug("self_model cross_task save failed: %s", e)

    def _load(self) -> None:
        """加载 task-local + cross-task, 合并. cross-task 优先."""
        # 先加载 cross-task (历史经验优先级高)
        if self.cross_task_path is not None and self.cross_task_path.exists():
            try:
                data = json.loads(self.cross_task_path.read_text(encoding="utf-8"))
                for s in data.get("skills", []):
                    rec = SkillRecord.from_dict(s)
                    if rec.tier in ("capable", "blind"):
                        self._skills[rec.skill] = rec
            except Exception as e:
                logger.debug("self_model cross_task load failed: %s", e)
        # 再加载 task-local (uncertain 可被覆盖, capable/blind 不被 cross-task 覆盖)
        if self.task_local_path is not None and self.task_local_path.exists():
            try:
                data = json.loads(self.task_local_path.read_text(encoding="utf-8"))
                for s in data.get("skills", []):
                    rec = SkillRecord.from_dict(s)
                    existing = self._skills.get(rec.skill)
                    if existing is None:
                        self._skills[rec.skill] = rec
                    elif rec.tier == "uncertain":
                        # task-local uncertain 不覆盖 cross-task capable/blind
                        if existing.tier == "uncertain":
                            self._skills[rec.skill] = rec
                    else:
                        # task-local capable/blind 覆盖 cross-task (本 task 实测)
                        self._skills[rec.skill] = rec
            except Exception as e:
                logger.debug("self_model task_local load failed: %s", e)


# ---------- helpers ----------

def _parse_first_json(text: str) -> dict | None:
    """从 text 抓第一个平衡的 {...} JSON. 失败返回 None.

    ponytail: 跟 imagination._parse_first_json 同款. 不 import 是为了
    避免 imagination -> self_model 循环依赖 (imagination 不调 self_model,
    但 step_evaluator 路径上 import 顺序不可控, 独立更稳).
    """
    if not text:
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_step_result_from_audit(
    audit_path: Path | None,
    step_id: int,
) -> list[dict]:
    """从 audit.jsonl 抓指定 step 的工具调用记录.

    返回 list[dict], 每条: {tool_name, success, error_message, duration}.
    audit_log 不存在/空/没工具事件 → 空列表.

    ponytail: 跟 step_evaluator.compute_tool_call_health 同款扫法, 但输出
    per-tool 记录给 SelfModel 用. 升级路径: 增量索引 / SQLite audit.
    schema 没 step_id 时全收 (跟 step_evaluator 行为一致).
    """
    if audit_path is None:
        return []
    path = Path(audit_path)
    if not path.exists():
        return []

    calls: list[dict] = []
    pending: dict[str, dict] = {}  # sig -> call record (等待 result/error)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # schema 没有 step_id 就全收; 有了就按 step_id 过滤
                rec_step = rec.get("step_id") or rec.get("iteration")
                if rec_step is not None:
                    try:
                        if int(rec_step) != step_id:
                            continue
                    except (TypeError, ValueError):
                        pass
                etype = rec.get("type", "")
                if etype not in ("tool.call", "tool.result", "tool.error"):
                    continue
                data = rec.get("data") or {}
                if not isinstance(data, dict):
                    data = {}
                tool_name = (
                    data.get("tool") or rec.get("tool_name")
                    or rec.get("tool") or "?"
                )
                ts = rec.get("timestamp") or rec.get("ts") or 0
                try:
                    ts_f = float(ts) if ts else 0.0
                except (TypeError, ValueError):
                    ts_f = 0.0
                if etype == "tool.call":
                    sig = f"{tool_name}:{str(data.get('input', ''))[:64]}"
                    pending[sig] = {
                        "tool_name": str(tool_name),
                        "success": False,  # 默认失败, 等 result/error 改
                        "error_message": "",
                        "duration": 0.0,
                        "_call_ts": ts_f,
                    }
                elif etype == "tool.result":
                    # 找最近的 pending call 同 tool
                    for sig, call in list(pending.items()):
                        if call["tool_name"] == tool_name and not call["success"]:
                            call["success"] = True
                            if call.get("_call_ts") and ts_f:
                                call["duration"] = ts_f - call["_call_ts"]
                            calls.append(call)
                            del pending[sig]
                            break
                elif etype == "tool.error":
                    err = str(
                        data.get("error") or rec.get("error_type") or ""
                    )
                    for sig, call in list(pending.items()):
                        if call["tool_name"] == tool_name and not call["success"]:
                            call["success"] = False
                            call["error_message"] = err
                            if call.get("_call_ts") and ts_f:
                                call["duration"] = ts_f - call["_call_ts"]
                            calls.append(call)
                            del pending[sig]
                            break
    except Exception as exc:
        logger.debug("extract_step_result_from_audit failed: %s", exc)
        return []
    # 残留 pending (有 call 没 result/error) 算失败
    for call in pending.values():
        call["success"] = False
        call["error_message"] = "no result/error event"
        calls.append(call)
    # 清理内部字段
    for c in calls:
        c.pop("_call_ts", None)
    return calls


# ---------- Self-check ----------

class _SeqMock:
    """Mock LLM: 按顺序返回预设 response. 用完最后一个重复最后一个."""
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.idx = 0
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        r = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return r


class _ErrMock:
    """Mock LLM: 永远抛异常."""
    def __init__(self, error: Exception):
        self.error = error
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        raise self.error


def _selfcheck() -> None:
    """Assert-based demo: 三档分类 + 跨 task 累积 + imagination feedback + 失败降级."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Case 1: 成功调用 → capable
        sm = SelfModel(
            task_local_path=td_path / "ws1" / ".huginn" / "self_model.json",
            cross_task_path=td_path / "cross1" / "self_model_cross_task.json",
            model=None,
        )
        sm.update_from_step({
            "tool_name": "matplotlib", "success": True, "error_message": "",
        })
        assert sm.get_tier("matplotlib") == "capable", \
            f"case1: capable failed: {sm.get_tier('matplotlib')}"
        print("[CHECK] case1 success -> capable (skill=matplotlib)")

        # Case 2: timeout → blind (keyword)
        sm.update_from_step({
            "tool_name": "long_runner", "success": False,
            "error_message": "TimeoutError: tool call timed out after 30s",
        })
        assert sm.get_tier("exec_timeout") == "blind", \
            f"case2: blind failed: {sm.get_tier('exec_timeout')}"
        print("[CHECK] case2 timeout -> blind (skill=exec_timeout)")

        # Case 3: import error → blind (keyword)
        sm.update_from_step({
            "tool_name": "unknown_pkg", "success": False,
            "error_message": "ModuleNotFoundError: No module named 'foo'",
        })
        assert sm.get_tier("pkg_install") == "blind", \
            f"case3: blind failed: {sm.get_tier('pkg_install')}"
        print("[CHECK] case3 import error -> blind (skill=pkg_install)")

        # Case 4: 未知错误 → uncertain (keyword fallback)
        sm.update_from_step({
            "tool_name": "weird_tool", "success": False,
            "error_message": "something went wrong",
        })
        assert sm.get_tier("weird_tool") == "uncertain", \
            f"case4: uncertain failed: {sm.get_tier('weird_tool')}"
        print("[CHECK] case4 unknown error -> uncertain (skill=weird_tool)")

        # Case 5: LLM 分类 — pytorch_training → blind
        sm_llm = SelfModel(
            task_local_path=td_path / "ws2" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross2" / "sm.json",
            model=_SeqMock(
                '{"tier": "blind", "reason": "needs GPU", '
                '"skill": "pytorch_training"}'
            ),
        )
        sm_llm.update_from_step({
            "tool_name": "pytorch", "success": False,
            "error_message": "CUDA error",
        })
        assert sm_llm.get_tier("pytorch_training") == "blind", \
            f"case5: LLM blind failed: {sm_llm.get_tier('pytorch_training')}"
        print("[CHECK] case5 LLM classify -> blind (skill=pytorch_training)")

        # Case 6: LLM 失败降级到 keyword (lammps → uncertain)
        sm_err = SelfModel(
            task_local_path=td_path / "ws3" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross3" / "sm.json",
            model=_ErrMock(RuntimeError("llm down")),
        )
        sm_err.update_from_step({
            "tool_name": "lammps", "success": False,
            "error_message": "lammps exec not found",
        })
        assert sm_err.get_tier("lammps") == "uncertain", \
            f"case6: keyword fallback failed: {sm_err.get_tier('lammps')}"
        print("[CHECK] case6 LLM error -> keyword fallback (skill=lammps)")

        # Case 7: 跨 task 累积 — cross-task 文件持久化 + 加载合并
        # case1-4 的 sm 已 save 到 cross1/, 新 SelfModel 加载应继承 capable/blind
        sm2 = SelfModel(
            task_local_path=td_path / "ws_new" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross1" / "self_model_cross_task.json",
            model=None,
        )
        # matplotlib (capable) 和 exec_timeout (blind) 应从 cross-task 加载
        assert sm2.get_tier("matplotlib") == "capable", \
            "case7: cross-task capable not loaded"
        assert sm2.get_tier("exec_timeout") == "blind", \
            "case7: cross-task blind not loaded"
        # uncertain 不持久化到 cross-task, 应默认 uncertain
        assert sm2.get_tier("weird_tool") == "uncertain", \
            "case7: uncertain should default (not in cross-task)"
        print("[CHECK] case7 cross-task accumulation OK")

        # Case 8: imagination feedback — uncertain → capable (success)
        sm_fb = SelfModel(
            task_local_path=td_path / "ws4" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross4" / "sm.json",
            model=None,
        )
        sm_fb.update_from_step({
            "tool_name": "rdkit", "success": False,
            "error_message": "rdkit parse error",
        })
        assert sm_fb.get_tier("rdkit") == "uncertain"
        sm_fb.feedback_from_imagination("rdkit", success=True)
        assert sm_fb.get_tier("rdkit") == "capable", \
            f"case8: upgrade failed: {sm_fb.get_tier('rdkit')}"
        print("[CHECK] case8 imagination success -> uncertain to capable")

        # Case 9: imagination feedback — uncertain → blind (failure)
        sm_fb2 = SelfModel(
            task_local_path=td_path / "ws5" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross5" / "sm.json",
            model=None,
        )
        sm_fb2.update_from_step({
            "tool_name": "rdkit", "success": False,
            "error_message": "rdkit error",
        })
        sm_fb2.feedback_from_imagination("rdkit", success=False)
        assert sm_fb2.get_tier("rdkit") == "blind", \
            f"case9: downgrade failed: {sm_fb2.get_tier('rdkit')}"
        print("[CHECK] case9 imagination failure -> uncertain to blind")

        # Case 10: 多条 step_result (list) — 一次性更新多个 skill
        sm_list = SelfModel(
            task_local_path=td_path / "ws6" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross6" / "sm.json",
            model=None,
        )
        sm_list.update_from_step([
            {"tool_name": "numpy", "success": True, "error_message": ""},
            {"tool_name": "pandas", "success": True, "error_message": ""},
            {"tool_name": "big_sim", "success": False,
             "error_message": "OOM cuda out of memory"},
        ])
        assert sm_list.get_tier("numpy") == "capable"
        assert sm_list.get_tier("pandas") == "capable"
        assert sm_list.get_tier("memory_limit") == "blind"
        print("[CHECK] case10 list step_result OK")

        # Case 11: extract_step_result_from_audit — mock audit.jsonl
        audit_path = td_path / "audit.jsonl"
        with audit_path.open("w", encoding="utf-8") as f:
            # step 1: matplotlib 成功
            f.write(json.dumps({
                "type": "tool.call", "step_id": 1,
                "data": {"tool": "matplotlib"}, "timestamp": 1000.0,
            }) + "\n")
            f.write(json.dumps({
                "type": "tool.result", "step_id": 1,
                "data": {"tool": "matplotlib"}, "timestamp": 1001.5,
            }) + "\n")
            # step 1: vasp 失败
            f.write(json.dumps({
                "type": "tool.call", "step_id": 1,
                "data": {"tool": "vasp"}, "timestamp": 1002.0,
            }) + "\n")
            f.write(json.dumps({
                "type": "tool.error", "step_id": 1,
                "data": {"tool": "vasp", "error": "OMP Error"},
                "timestamp": 1005.0,
            }) + "\n")
            # step 2: 别的 step (应被过滤)
            f.write(json.dumps({
                "type": "tool.call", "step_id": 2,
                "data": {"tool": "other"}, "timestamp": 2000.0,
            }) + "\n")

        records = extract_step_result_from_audit(audit_path, step_id=1)
        assert len(records) == 2, \
            f"case11: expected 2, got {len(records)}"
        tools = {r["tool_name"] for r in records}
        assert tools == {"matplotlib", "vasp"}, \
            f"case11: tools mismatch: {tools}"
        mpl = [r for r in records if r["tool_name"] == "matplotlib"][0]
        assert mpl["success"] is True, "case11: matplotlib should succeed"
        assert abs(mpl["duration"] - 1.5) < 0.01, \
            f"case11: duration mismatch: {mpl['duration']}"
        vasp = [r for r in records if r["tool_name"] == "vasp"][0]
        assert vasp["success"] is False, "case11: vasp should fail"
        assert "OMP Error" in vasp["error_message"], \
            f"case11: err msg missing: {vasp['error_message']}"
        print("[CHECK] case11 audit extraction OK (2 records, mpl ok, vasp fail)")

        # Case 12: 失败降级 — None / 空 / 非法输入都不阻塞
        sm_safe = SelfModel(
            task_local_path=td_path / "ws7" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross7" / "sm.json",
            model=None,
        )
        sm_safe.update_from_step(None)  # 不抛
        sm_safe.update_from_step([])  # 不抛
        sm_safe.update_from_step("not a dict")  # 不抛
        sm_safe.update_from_step([{"invalid": "no tool_name"}])  # 不抛
        # 跨 task load 文件损坏不抛
        bad_cross = td_path / "bad_cross" / "sm.json"
        bad_cross.parent.mkdir(parents=True, exist_ok=True)
        bad_cross.write_text("not json", encoding="utf-8")
        sm_bad = SelfModel(
            task_local_path=None, cross_task_path=bad_cross, model=None)
        assert sm_bad.get_tier("anything") == "uncertain"
        print("[CHECK] case12 failure degradation OK (None/empty/bad input/bad file)")

        # Case 13: 连续成功 N 次升级 — uncertain → capable
        sm_promote = SelfModel(
            task_local_path=td_path / "ws8" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross8" / "sm.json",
            model=None,
        )
        # 先让 rdkit 进 uncertain
        sm_promote.update_from_step({
            "tool_name": "rdkit", "success": False,
            "error_message": "rdkit parse error",
        })
        assert sm_promote.get_tier("rdkit") == "uncertain"
        # 连续 3 次成功 → capable
        for _ in range(_PROMOTE_THRESHOLD):
            sm_promote.update_from_step({
                "tool_name": "rdkit", "success": True, "error_message": "",
            })
        assert sm_promote.get_tier("rdkit") == "capable", \
            f"case13: promote failed: {sm_promote.get_tier('rdkit')}"
        print(f"[CHECK] case13 promote after {_PROMOTE_THRESHOLD} successes OK")

        # Case 14: 跨 task 优先级 — cross-task blind 不被 task-local uncertain 覆盖
        # 已有 cross1 (matplotlib capable, exec_timeout blind, pkg_install blind)
        sm_priority = SelfModel(
            task_local_path=td_path / "ws_priority" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross1" / "self_model_cross_task.json",
            model=None,
        )
        # 即使本 task 又试 matplotlib 失败 (keyword 未命中 → uncertain), 不覆盖 cross-task capable
        sm_priority.update_from_step({
            "tool_name": "matplotlib", "success": False,
            "error_message": "weird unrelated error",
        })
        # _update_one 走 keyword_classify 没匹配到 → uncertain + skill=matplotlib
        # 但 LLM/keyword 给的 tier=uncertain 不覆盖现有 capable (规则: tier != uncertain 才覆盖)
        # 不过 _update_one 写了 success_count + failure_count, tier 仍是 capable
        assert sm_priority.get_tier("matplotlib") == "capable", \
            f"case14: cross-task capable should not be downgraded: {sm_priority.get_tier('matplotlib')}"
        print("[CHECK] case14 cross-task capable not downgraded by single failure")

    print("OK self_model self-check passed (14 cases)")


if __name__ == "__main__":
    _selfcheck()
