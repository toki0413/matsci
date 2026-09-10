"""可执行物理经验库 (Executable Experience Archive) — 补第②个真缺口.

背景: 现有 `knowledge/distiller` 存的是**文本结论**, `metacog/episodic_replay` 存的是
**方法/验证状态**; 两者都**不可重放、不可复验真值数值**. Code-as-World 的 EWR(scene.json)
给了我们关键提示: 结构化物理经验应当是**可执行、可被回读校验**的, 而不只是可检索的文本。

本模块实现一个**域无关、不绑书生**的可执行经验档案:
  - 每条经验 = (goal, 维度, 配置 cfg, 真实数值轨迹 objectives/summary, 闭式, 真值 ground_truth);
  - 可序列化到工作区 JSON(带版本号), 可收纳;
  - **replay** 时可用同一 run 回调重跑, 并用 `claim_reward`(连续奖励+溯源)校验
    新旧结果是否一致 & 是否命中真值 —— 让"经验库"同时是"可回收的 RL 记忆"。

设计原则(huginn 一贯): 只存真实计算数值; 重放失败/不一致 → 标记 degraded 而非伪造;
零 LLM/零网络; 纯标准库, 可在缺重依赖环境下独立单测。

关联: `huginn.validation.claim_reward`(第③点) 提供连续 grounded accuracy 奖励,
本模块把一次"真实执行"固化为**可重放的经验单元**, 喂给该奖励做 RL 回流。
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

EXPERIENCE_VERSION = 1


# 用与 claim_reward 相同的惰性绑定, 避免包 __init__ 拉起 langchain 重依赖.
def _load_claim_reward():
    import importlib.util as _util
    _root = Path(__file__).resolve().parent
    _spec = _util.spec_from_file_location(
        "claim_reward_embed", _root.parent / "validation" / "claim_reward.py"
    )
    _mod = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


class ExecutableExperience:
    """一条可执行物理经验: 可落盘、可收纳、可重放复验真值."""

    def __init__(
        self,
        goal: str,
        dimension: str,
        config: dict[str, Any],
        objectives: dict[str, float],
        summary: dict[str, Any] | None = None,
        closed_form: str | None = None,
        ground_truth: dict[str, float] | None = None,
        verdict: str = "pass",
        *,
        name: str = "",
        ts: str = "",
        quantities: dict[str, Any] | None = None,
    ) -> None:
        self.name = name or f"exp_{uuid.uuid4().hex[:10]}"
        self.goal = goal
        self.dimension = dimension
        self.config = config
        self.objectives = objectives          # 真实计算数值
        self.summary = summary or {}
        self.closed_form = closed_form
        self.ground_truth = ground_truth or {}  # 主张数值 -> 真值 (供 reward 对账)
        self.verdict = verdict
        # 域科学契约工件(量纲/有效域): 由 compile_domain_guards 透出, replay 时可并入判官.
        self.quantities = dict(quantities or {})
        self.ts = ts or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------- 序列化 ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EXPERIENCE_VERSION,
            "name": self.name,
            "goal": self.goal,
            "dimension": self.dimension,
            "config": self.config,
            "objectives": {str(k): float(v) for k, v in self.objectives.items()},
            "summary": self.summary,
            "closed_form": self.closed_form,
            "ground_truth": {str(k): float(v) for k, v in self.ground_truth.items()},
            "verdict": self.verdict,
            "quantities": dict(self.quantities),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutableExperience":
        return cls(
            goal=d.get("goal", ""),
            dimension=d.get("dimension", ""),
            config=d.get("config", {}),
            objectives={str(k): float(v) for k, v in (d.get("objectives") or {}).items()},
            summary=d.get("summary", {}),
            closed_form=d.get("closed_form"),
            ground_truth={str(k): float(v) for k, v in (d.get("ground_truth") or {}).items()},
            verdict=d.get("verdict", "pass"),
            name=d.get("name", ""),
            ts=d.get("ts", ""),
            quantities=d.get("quantities") or {},
        )

    # ---------- 可重放复验 ----------
    def replay(
        self,
        runner: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        reward_mode: str = "mra",
        quantities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """用同一 run 回调重跑本经验, 并用连续奖励复验真值命中.

        runner(cfg) -> {"summary": {...}, "objectives": {name: float}, "target": Optional}
        ground_truth 需提供 (来自首次真实执行时记录的、或被外部 benchmark 标定的真值);
        取回归 runner 的 target 写进 objectives, 让 claim_reward 能对账. 无真值 → 只给
        溯源接地分(grounded), 不给 reward 数值(诚实, 不伪造).

        ``quantities``(可选): 域科学契约(量纲/有效域), 缺省用本经验自带的. 提供时把判官
        升级为"溯源 AND 契约"——即便数值溯源到轨迹, 越出有效域也会在 reward 里暴露
        contract_verdict=needs_grounding(本轮 A 的接线点).

        Returns:
            {"reproduced": bool, "new_objectives": {...}, "reward": {grounded, accuracy, total} | None,
             "delta": {pairs, max_abs_err}}
        """
        cr = _load_claim_reward()
        res = runner(self.config)
        new_obj = {str(k): float(v) for k, v in (res.get("objectives") or {}).items()}
        eff_q = quantities if quantities is not None else self.quantities
        # 1) 新旧一致性: 仅对两次真实执行都出现的量比较(缺项不判错).
        pairs = {}
        for k, v in new_obj.items():
            if k in self.objectives:
                pairs[k] = (v, float(self.objectives[k]))
        max_abs = max((abs(a - b) for a, b in pairs.values()), default=0.0)
        reproduced = max_abs <= 1e-6  # 同一 cfg 重跑 → 数值应逐位一致(确定性执行)
        # 2) 真值复验: 归并 target 与 ground_truth (均为「目标键名→真值」),
        #    与 new_obj 按键名对齐成 (预测值→真值) 映射, 再喂 claim_reward 连续奖励.
        truth_by_name = dict(self.ground_truth or {})
        tgt = res.get("target")
        if isinstance(tgt, dict):
            for k, v in tgt.items():
                truth_by_name.setdefault(str(k), v)
        pred_to_truth: dict[float, float] = {}
        for k, objv in new_obj.items():
            if k in truth_by_name:
                try:
                    pred_to_truth[float(objv)] = float(truth_by_name[k])
                except (TypeError, ValueError):
                    continue
        if pred_to_truth:
            # 融合判官: 溯源 AND (含 contract 时) 有效域 —— objectives+quantities 一并喂入.
            gd = cr.grounding_source_reward(
                json.dumps(new_obj), [json.dumps(new_obj)],
                objectives=new_obj, quantities=eff_q,
            )
            detail = cr.grounded_accuracy_reward(
                json.dumps(new_obj), [json.dumps(new_obj)],
                values=pred_to_truth, mode=reward_mode,
            )
            reward = {
                "grounded": gd["grounding_score"],
                "accuracy": detail["reward"],
                "total": detail["reward"],  # grounded_accuracy_reward 已含溯源加权
                "contract_verdict": gd.get("contract_verdict"),
                "contract_score": gd.get("contract_score"),
                "domain_violations": gd.get("domain_violations", []),
            }
        else:
            reward = None
        return {
            "reproduced": bool(reproduced),
            "new_objectives": new_obj,
            "delta": {"pairs": list(pairs.keys()), "max_abs_err": round(max_abs, 6)},
            "reward": reward,
        }


class ExperienceArchive:
    """可执行物理经验档案: 目录化 JSON 存储 + 收纳检索."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def save(self, exp: ExecutableExperience) -> str:
        self._path(exp.name).write_text(
            json.dumps(exp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return exp.name

    def load(self, name: str) -> ExecutableExperience | None:
        p = self._path(name)
        if not p.exists():
            return None
        try:
            return ExecutableExperience.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — 损坏条目视为不存在, 不炸
            return None

    def list_all(self) -> list[ExecutableExperience]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            e = self.load(p.stem)
            if e is not None:
                out.append(e)
        return out

    def find(self, *, dimension: str | None = None, goal_contains: str | None = None) -> list[ExecutableExperience]:
        out = []
        for e in self.list_all():
            if dimension and e.dimension != dimension:
                continue
            if goal_contains and goal_contains not in e.goal:
                continue
            out.append(e)
        return out

    def ingest_from_experiment(
        self,
        experiment_result: dict[str, Any],
        *,
        goal: str,
        dimension: str,
        config: dict[str, Any],
        closed_form: str | None = None,
        ground_truth: dict[str, float] | None = None,
    ) -> str:
        """从一次真实实验的结果 dict({summary, objectives}) 固化为经验归档."""
        exp = ExecutableExperience(
            goal=goal,
            dimension=dimension,
            config=config,
            objectives=(experiment_result.get("objectives") or {}),
            summary=(experiment_result.get("summary") or {}),
            closed_form=closed_form,
            ground_truth=ground_truth,
            verdict="pass",
        )
        return self.save(exp)


__all__ = [
    "ExecutableExperience",
    "ExperienceArchive",
    "EXPERIENCE_VERSION",
]