"""BlindSpotMapper — 把 self-model 的盲点映射成 hint + imagination trigger.

v15 Phase 5 Task 11: 盲点不是孤立标签, 是触发 imagination 的种子.
blind 档直接是盲点 (priority=high), uncertain 档是潜在盲点 (priority=medium).
跨 task 累积的 blind 一律 high (历史经验, 不给本 task 试探翻案).

输出两类东西:
  1. hint 文本: 注入到 prompt, 让 agent 看到"自己在哪有盲点 + 怎么绕"
  2. BlindSpot 对象: 喂给 imagine_from_blind_spot, 在 manifold 上生成能绕过
     盲点的 hypothesis

失败一律降级 (返回空 list / 空字符串), 不阻塞主循环.

ponytail: 单文件, stdlib only. 不引新依赖. 升级路径: LLM 推断 workaround
+ dynamic priority (按 failure_count 加权) + 跟 manifold 联动 (低 log_posterior
region 的 h 对应的 skill 升 priority).
"""
from __future__ import annotations

# 直接跑脚本时把 agent/ 加到 sys.path (被 import 时不执行, rcb_runner 已设好)
if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _agent_root = str(_Path(__file__).resolve().parents[2])
    if _agent_root not in _sys.path:
        _sys.path.insert(0, _agent_root)

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BlindSpot:
    """盲点记录.

    skill:         能力名 (跟 SelfModel.SkillRecord.skill 对齐)
    why_blind:     为什么是盲点 (从 SkillRecord.reason 来)
    possible_workaround: 建议的绕过方式
    priority:      high / medium / low (high 触发 imagination)
    """
    skill: str
    why_blind: str
    possible_workaround: str = ""
    priority: str = "medium"  # high / medium / low

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "why_blind": self.why_blind,
            "possible_workaround": self.possible_workaround,
            "priority": self.priority,
        }


# skill → 默认 workaround 映射. ponytail: keyword 匹配, 不上 LLM.
# 升级路径: LLM 推断 workaround (基于 task_ctx + why_blind).
_DEFAULT_WORKAROUNDS: dict[str, str] = {
    # 资源类盲点
    "exec_timeout": "break the task into smaller steps or reduce problem size",
    "memory_limit": "use streaming / chunking, or reduce model size; persist intermediate to disk",
    "disk_space": "clean intermediate files, or use a smaller dataset sample",
    "permission_denied": "check file permissions or write to a different directory",
    "pkg_install": "use a different tool that's already installed, or pure-stdlib implementation",
    # 重型 sim 类盲点
    "pytorch_training": "use a smaller model, fewer epochs, pre-trained weights, or CPU-friendly surrogate",
    "tensorflow_training": "use a smaller model, fewer epochs, pre-trained weights, or CPU-friendly surrogate",
    "dl_training": "use a smaller model, fewer epochs, or a surrogate / classical baseline",
    "vasp": "use a smaller cell, fewer k-points, or a cheaper pseudopotential; or use QE / open-source DFT",
    "gaussian": "use a smaller basis set, or a semi-empirical method (AM1/PM3)",
    "lammps": "use a smaller system, shorter run, or a cheaper force field",
    "qe": "use a smaller cell, fewer k-points, or a cheaper pseudopotential",
    "openmm": "use a smaller system, shorter trajectory, or implicit solvent",
    "rdkit": "check input SMILES / mol block, or use a different cheminformatics approach",
}


def infer_blind_spots(self_model: Any) -> list[BlindSpot]:
    """从 self-model 推断盲点. 失败返回空 list.

    - blind 档 → priority=high (确认盲点, 触发 imagination)
    - uncertain 档 → priority=medium (潜在盲点, 注入 hint 但不触发 imagination)
    - capable 档 → 不算盲点

    ponytail: 不调 LLM, 直接从 self_model.list_by_tier 抓 + 查 workaround 表.
    升级路径: LLM 推断 workaround + 按 failure_count 加权 priority.
    """
    if self_model is None:
        return []
    blind_spots: list[BlindSpot] = []
    try:
        for rec in self_model.list_by_tier("blind"):
            workaround = _DEFAULT_WORKAROUNDS.get(rec.skill, "")
            blind_spots.append(BlindSpot(
                skill=rec.skill,
                why_blind=rec.reason or "blind spot from past failures",
                possible_workaround=workaround or "find an alternative approach",
                priority="high",
            ))
        for rec in self_model.list_by_tier("uncertain"):
            workaround = _DEFAULT_WORKAROUNDS.get(rec.skill, "")
            blind_spots.append(BlindSpot(
                skill=rec.skill,
                why_blind=rec.reason or "uncertain capability",
                possible_workaround=workaround or "test carefully before relying on it",
                priority="medium",
            ))
    except Exception as e:
        logger.debug("infer_blind_spots failed: %s", e)
        return []
    return blind_spots


def map_blind_spot_to_hint(blind_spot: BlindSpot | None, task_ctx: str = "") -> str:
    """把单条盲点转为 hint 注入到 prompt. 失败返回空字符串, 不阻塞.

    hint 格式: "[blind_spot_hint] 你之前在 {skill} 上遇到过 {why_blind}.
                考虑 {possible_workaround}. (高优先级盲点, 优先绕过而非硬刚.)"

    ponytail: 模板字符串拼接, 不调 LLM. task_ctx 当前未用 (升级路径:
    LLM 根据 task_ctx 调整 workaround 措辞).
    """
    try:
        if blind_spot is None or not getattr(blind_spot, "skill", ""):
            return ""
        parts = [
            f"[blind_spot_hint] 你之前在 {blind_spot.skill} 上遇到过 "
            f"{blind_spot.why_blind}."
        ]
        if blind_spot.possible_workaround:
            parts.append(f"考虑 {blind_spot.possible_workaround}.")
        if blind_spot.priority == "high":
            parts.append("这是高优先级盲点, 优先绕过而非硬刚.")
        return " ".join(parts)
    except Exception as e:
        logger.debug("map_blind_spot_to_hint failed: %s", e)
        return ""


def map_blind_spots_to_hint(
    blind_spots: list[BlindSpot] | None,
    task_ctx: str = "",
    max_n: int = 3,
) -> str:
    """多条盲点合并成一条 hint. 失败返回空串.

    ponytail: max_n=3 防 prompt 膨胀. high 优先 (sorted stable).
    """
    if not blind_spots:
        return ""
    try:
        # high 优先, 取前 max_n 条
        sorted_bs = sorted(
            blind_spots,
            key=lambda b: 0 if getattr(b, "priority", "") == "high" else 1,
        )
        hints = []
        for bs in sorted_bs[:max_n]:
            h = map_blind_spot_to_hint(bs, task_ctx)
            if h:
                hints.append(h)
        return "\n".join(hints)
    except Exception as e:
        logger.debug("map_blind_spots_to_hint failed: %s", e)
        return ""


def pick_imagination_seed(
    blind_spots: list[BlindSpot] | None,
) -> BlindSpot | None:
    """从盲点列表挑一个作为 imagination 的种子.

    选 high priority 中 failure_count 最高的 (盲得最厉害的). 没 high 就 None
    (imagination 仍可由 stagnation 触发, 不强求 blind spot 触发).

    ponytail: 单次线性扫描. 升级路径: 跟 manifold 联动 — 选低 log_posterior
    region 对应的 skill.
    """
    if not blind_spots:
        return None
    try:
        high = [b for b in blind_spots if b.priority == "high"]
        if not high:
            return None
        # stable: 第一个 high (按 infer_blind_spots 顺序 = self_model 内 dict 顺序)
        return high[0]
    except Exception as e:
        logger.debug("pick_imagination_seed failed: %s", e)
        return None


# ---------- Self-check ----------

def _selfcheck() -> None:
    """Assert-based demo: infer_blind_spots + map_blind_spot_to_hint + 失败降级."""
    import tempfile
    from pathlib import Path
    from huginn.metacog.self_model import SelfModel

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        sm = SelfModel(
            task_local_path=td_path / "ws" / ".huginn" / "sm.json",
            cross_task_path=td_path / "cross" / "sm.json",
            model=None,
        )
        # 添加 blind + uncertain skills
        sm.update_from_step({
            "tool_name": "pytorch", "success": False,
            "error_message": "CUDA OOM out of memory",
        })  # memory_limit -> blind
        sm.update_from_step({
            "tool_name": "lammps", "success": False,
            "error_message": "lammps exec not found",
        })  # lammps -> uncertain
        sm.update_from_step({
            "tool_name": "rdkit", "success": False,
            "error_message": "rdkit parse error",
        })  # rdkit -> uncertain

        # Case 1: infer_blind_spots 返回非空
        bs_list = infer_blind_spots(sm)
        assert len(bs_list) >= 2, f"case1: expected >=2, got {len(bs_list)}"
        skills = {b.skill for b in bs_list}
        # memory_limit (blind) + lammps (uncertain) + rdkit (uncertain)
        assert "memory_limit" in skills, f"case1: blind missing: {skills}"
        print(f"[CHECK] case1 infer_blind_spots OK ({len(bs_list)} spots: {skills})")

        # Case 2: blind 档 → high priority
        high_bs = [b for b in bs_list if b.priority == "high"]
        assert len(high_bs) >= 1, f"case2: no high priority"
        for b in high_bs:
            assert b.skill in (
                "memory_limit", "pytorch_training", "pkg_install",
                "exec_timeout", "disk_space", "permission_denied",
            ), f"case2: unexpected high skill: {b.skill}"
        print(f"[CHECK] case2 blind -> high priority OK ({len(high_bs)} high)")

        # Case 3: uncertain 档 → medium priority
        medium_bs = [b for b in bs_list if b.priority == "medium"]
        assert len(medium_bs) >= 1, f"case3: no medium priority"
        for b in medium_bs:
            assert b.skill in (
                "rdkit", "lammps", "vasp", "gaussian", "qe", "openmm",
                "pytorch_training", "tensorflow_training", "dl_training",
            ), f"case3: unexpected medium skill: {b.skill}"
        print(f"[CHECK] case3 uncertain -> medium priority OK ({len(medium_bs)} medium)")

        # Case 4: map_blind_spot_to_hint 非空 + 含 marker + skill
        hint = map_blind_spot_to_hint(bs_list[0])
        assert hint, f"case4: hint empty"
        assert "[blind_spot_hint]" in hint, f"case4: missing marker:\n{hint}"
        assert bs_list[0].skill in hint, f"case4: skill missing in hint"
        print(f"[CHECK] case4 hint non-empty: {hint[:80]}...")

        # Case 5: high priority hint 含 "高优先级"
        high_hint = map_blind_spot_to_hint(high_bs[0])
        assert "高优先级" in high_hint, f"case5: missing high marker:\n{high_hint}"
        print(f"[CHECK] case5 high priority hint OK")

        # Case 6: workaround 在 hint 中 (memory_limit 有默认 workaround)
        bs_mem = next((b for b in bs_list if b.skill == "memory_limit"), None)
        assert bs_mem is not None, "case6: memory_limit blind spot missing"
        assert bs_mem.possible_workaround, "case6: workaround empty"
        hint_w = map_blind_spot_to_hint(bs_mem)
        assert bs_mem.possible_workaround in hint_w, \
            f"case6: workaround missing in hint:\n{hint_w}"
        print(f"[CHECK] case6 workaround in hint OK")

        # Case 7: map_blind_spots_to_hint 多条合并
        combined = map_blind_spots_to_hint(bs_list, max_n=5)
        assert combined, f"case7: combined empty"
        assert combined.count("[blind_spot_hint]") >= 2, \
            f"case7: combined should have >=2 hints:\n{combined}"
        # high 应排在前面
        first_high_idx = combined.find("高优先级")
        first_marker_idx = combined.find("[blind_spot_hint]")
        assert first_high_idx > first_marker_idx, \
            f"case7: high should be in first hint block"
        print(f"[CHECK] case7 combined hints OK ({combined.count('[blind_spot_hint]')} spots)")

        # Case 8: max_n 限制
        limited = map_blind_spots_to_hint(bs_list, max_n=1)
        assert limited.count("[blind_spot_hint]") <= 1, \
            f"case8: max_n=1 should limit to 1 hint:\n{limited}"
        print(f"[CHECK] case8 max_n limit OK")

        # Case 9: 失败降级 — self_model=None / blind_spot=None / 空 list
        assert infer_blind_spots(None) == [], "case9: None should return []"
        assert map_blind_spot_to_hint(None) == "", "case9: None blind_spot should return ''"
        assert map_blind_spots_to_hint([]) == "", "case9: empty list should return ''"
        assert map_blind_spots_to_hint(None) == "", "case9: None list should return ''"
        # 损坏的 blind_spot 不抛
        assert map_blind_spot_to_hint(BlindSpot(skill="", why_blind="")) == "", \
            "case9: empty skill should return ''"
        print(f"[CHECK] case9 failure degradation OK (None/empty/bad input)")

        # Case 10: BlindSpot dataclass 序列化
        bs_obj = BlindSpot(
            skill="vasp",
            why_blind="no VASP license",
            possible_workaround="use QE or open-source DFT",
            priority="high",
        )
        d = bs_obj.to_dict()
        assert d["skill"] == "vasp"
        assert d["priority"] == "high"
        assert d["possible_workaround"] == "use QE or open-source DFT"
        print(f"[CHECK] case10 BlindSpot dataclass OK")

        # Case 11: pick_imagination_seed — 选 high priority 第一个
        seed = pick_imagination_seed(bs_list)
        assert seed is not None, "case11: seed should not be None with high bs"
        assert seed.priority == "high", f"case11: seed should be high"
        # 没 high 时返回 None
        only_medium = [b for b in bs_list if b.priority == "medium"]
        seed_none = pick_imagination_seed(only_medium)
        assert seed_none is None, "case11: medium only should return None"
        # 空列表 / None 返回 None
        assert pick_imagination_seed(None) is None
        assert pick_imagination_seed([]) is None
        print(f"[CHECK] case11 pick_imagination_seed OK (seed={seed.skill})")

        # Case 12: 跨 task 累积的 blind 也是 high (priority 来自 tier, 不分本地/跨task)
        # 写一个 cross-task 文件包含 blind skill, 加载后 infer 应仍 priority=high
        cross_path = td_path / "cross_acc" / "sm_cross.json"
        cross_path.parent.mkdir(parents=True, exist_ok=True)
        cross_path.write_text(
            '{"skills": [{"skill": "pkg_install", "tier": "blind", '
            '"reason": "cross-task history: cannot install", '
            '"success_count": 0, "failure_count": 5}], '
            '"last_updated": 0.0}',
            encoding="utf-8",
        )
        sm_cross = SelfModel(
            task_local_path=td_path / "ws_cross" / "sm.json",
            cross_task_path=cross_path,
            model=None,
        )
        bs_cross = infer_blind_spots(sm_cross)
        pkg_bs = next((b for b in bs_cross if b.skill == "pkg_install"), None)
        assert pkg_bs is not None, "case12: pkg_install not loaded from cross-task"
        assert pkg_bs.priority == "high", \
            f"case12: cross-task blind should be high: {pkg_bs.priority}"
        print(f"[CHECK] case12 cross-task blind -> high priority OK")

    print("OK blind_spot_mapper self-check passed (12 cases)")


if __name__ == "__main__":
    _selfcheck()
