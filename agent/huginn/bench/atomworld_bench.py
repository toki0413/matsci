"""AtomWorld AtomMotor-2K 子任务 — 3D 空间推理 verifiable benchmark.

来源: arXiv:2510.04704 (NUS + MasterAI-EAM, ICML 2026).
论文实证 Claude Opus 4.6 rotation 类 < 12%, 是 text-centric VLM 丢 3D 信息的直接证据.

跟其他 bench/*_bench.py 不一样: AtomWorld 是 CIF-in/CIF-out, 不 fit BenchmarkTask
(text-in/text-out) 模式. 所以走独立 Subtask 类, run(agent_inference_fn) 直接返回
汇总 dict, 不进 get_suite_tasks() 列表 (跟 "research" suite 一样返 []).

env flag HUGINN_USE_ATOMWORLD=1 + atomworld 包安装才生效, 默认 off 行为不变.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

# AtomWorld 数据目录. 真包安装时数据通常在 ./atomworld_data 或环境变量指定位置.
# ponytail: 不硬编码, 允许 env 覆盖; 默认 ./atomworld_data 跟 README 一致.
_DEFAULT_DATA_DIR = os.environ.get("HUGINN_ATOMWORLD_DATA_DIR", "./atomworld_data")


class AtomMotor2KSubtask:
    """AtomWorld AtomMotor-2K 子任务 — 3D 空间推理 verifiable benchmark.

    full set = 2500 题 (15 actions × ~167 题/action). 默认 n_per_action=10 跑 subset
    验证用 (150 题), full 跑设 n_per_action=None 或大数.

    用法:
        sub = AtomMotor2KSubtask(n_per_action=10)
        if sub.is_available():
            report = sub.run(lambda cif, prompt: agent.generate(cif, prompt))
    """

    NAME = "atomworld_atommotor_2k"

    def __init__(
        self,
        n_per_action: int | None = 10,
        data_dir: str = _DEFAULT_DATA_DIR,
    ) -> None:
        """n_per_action=None 表示跑全集 (每 action 全部 sample)."""
        self._n_per_action = n_per_action
        self._data_dir = data_dir

    def is_available(self) -> bool:
        """flag + atomworld 包是否就位. 任一缺失返 False, run() 会跳过."""
        from huginn.tools.atomworld_tool import is_available
        return (
            os.environ.get("HUGINN_USE_ATOMWORLD", "0") == "1"
            and is_available()
        )

    def run(
        self,
        agent_inference_fn: Callable[[str, str], str],
    ) -> dict[str, Any]:
        """跑 AtomMotor-2K subset.

        agent_inference_fn(input_cif: str, action_prompt: str) -> str
            接收 input CIF + action prompt, 返回 generated CIF.

        返回:
            {"skipped": True} — flag off 或包未装
            {"per_action": {...}, "overall": {...}} — 跑完的汇总
        """
        if not self.is_available():
            logger.info("AtomWorld skipped (flag off or package not installed)")
            return {"skipped": True}

        from atomworld import load_data
        from huginn.tools.atomworld_tool import evaluate, list_actions

        per_action: dict[str, dict[str, Any]] = {}
        all_results: list[dict[str, Any]] = []

        # list_actions() 已返回全名 (add_atom_action 等), 直接喂给 load_data.
        # 名字跟 atomworld API 不匹配时 load_data raise, try/except 跳过.
        for action_name in list_actions():
            try:
                df = load_data(self._data_dir, action_name=action_name)
            except Exception as e:
                logger.warning("load_data %s failed: %s", action_name, e)
                continue

            if self._n_per_action is not None:
                df = df.head(self._n_per_action)

            results: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                input_cif = row["input_cif"]
                action_prompt = row["action_prompt"]
                target_cif = row["output_cif"]
                try:
                    predicted_cif = agent_inference_fn(input_cif, action_prompt)
                except Exception as e:
                    logger.debug("agent_inference_fn failed: %s", e)
                    results.append({"correct": False, "rmsd": None, "max_dist": None})
                    continue
                try:
                    er = evaluate(target_cif, predicted_cif)
                    results.append({
                        "correct": er.correct,
                        "rmsd": er.rmsd,
                        "max_dist": er.max_dist,
                    })
                except Exception as e:
                    logger.debug("evaluate failed: %s", e)
                    results.append({"correct": False, "rmsd": None, "max_dist": None})

            n = len(results)
            success = sum(1 for r in results if r["correct"])
            rmsds = [r["rmsd"] for r in results if r["rmsd"] is not None]
            max_dists = [r["max_dist"] for r in results if r["max_dist"] is not None]
            per_action[action_name] = {
                "success_rate": success / n if n > 0 else 0.0,
                "mean_rmsd": sum(rmsds) / len(rmsds) if rmsds else None,
                "mean_max_dist": sum(max_dists) / len(max_dists) if max_dists else None,
                "n": n,
            }
            all_results.extend(results)

        n_total = len(all_results)
        success_total = sum(1 for r in all_results if r["correct"])
        rmsds_all = [r["rmsd"] for r in all_results if r["rmsd"] is not None]
        max_dists_all = [r["max_dist"] for r in all_results if r["max_dist"] is not None]
        overall = {
            "success_rate": success_total / n_total if n_total > 0 else 0.0,
            "mean_rmsd": sum(rmsds_all) / len(rmsds_all) if rmsds_all else None,
            "mean_max_dist": sum(max_dists_all) / len(max_dists_all) if max_dists_all else None,
            "n": n_total,
        }

        return {"per_action": per_action, "overall": overall}


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """3 场景: flag off 跳过 / flag on 包未装跳过 / run mock 路径不崩.

    ponytail: 非平凡逻辑留 runnable check. 不依赖真 atomworld 包也不调真 LLM.
    ceiling: 没验 load_data 真 DataFrame 路径 (需真包+数据), 升级路径见 acceptance test.
    """
    import os
    # 备份 env, selfcheck 改完恢复
    orig_flag = os.environ.get("HUGINN_USE_ATOMWORLD", "0")
    try:
        # 1. flag off -> is_available False, run 返回 skipped
        os.environ["HUGINN_USE_ATOMWORLD"] = "0"
        sub = AtomMotor2KSubtask(n_per_action=2)
        assert sub.is_available() is False, "flag off 时 is_available 应 False"
        report = sub.run(lambda cif, p: "")
        assert report == {"skipped": True}, f"flag off 应返 skipped, got {report}"
        print("1. flag off -> skipped OK")

        # 2. flag on 但包未装 (atomworld 可能在测试环境没装) -> is_available False
        os.environ["HUGINN_USE_ATOMWORLD"] = "1"
        from huginn.tools.atomworld_tool import is_available
        if not is_available():
            assert sub.is_available() is False, "包未装时 is_available 应 False"
            report = sub.run(lambda cif, p: "")
            assert report == {"skipped": True}
            print("2. flag on + 包未装 -> skipped OK")
        else:
            # 包真装了, 跳过这条 (CI 环境一般没装, 本地装了也不算错)
            print("2. atomworld 已装, 跳过包未装场景")
    finally:
        os.environ["HUGINN_USE_ATOMWORLD"] = orig_flag

    print("atomworld_bench selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
