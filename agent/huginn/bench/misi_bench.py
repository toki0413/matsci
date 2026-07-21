"""MiSI-Bench 微观空间智能子任务 — 分子结构 3D 空间推理 verifiable benchmark.

来源: arXiv:2512.10867 (人大+清华+北大+中科院+达摩院, 2025-12).
论文实证 SOTA VLM 在 rotation task 落后人类, 7B SFT 在旋转 task 超人类 (~90%),
但氢键识别 task SFT 仍落后人类 — 验证族系 IV (domain knowledge) 单独需要.

跟其他 bench/*_bench.py 不一样: MiSI 是 image-in/text-out, 不 fit BenchmarkTask
(text-in/text-out) 模式. 所以走独立 Subtask 类, run(agent_inference_fn) 直接返回
汇总 dict, 不进 get_suite_tasks() 列表 (跟 "research" / "atommotor" suite 一样返 []).

env flag HUGINN_USE_MISI=1 + datasets + rdkit 包安装才生效, 默认 off 行为不变.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

# MiSI-Bench HuggingFace 数据集名.
# ponytail: 不硬编码本地路径, 用 HF datasets 加载 (跟 atomworld 用 load_data 风格一致).
_MISI_DATASET = "zongzhao/MiSI-bench"

# 9 个 task 的 ID (T1-T9, 跟论文 Fig 2 一致).
# ponytail: 每个 task 独立 evaluate, 跟 atomworld 每 action 独立汇总风格一致.
MISI_TASKS = [
    "translation",           # T1 平移
    "rotation",              # T2 旋转 (对照 AtomWorld rotation <12%)
    "zooming",               # T3 缩放
    "residue_ligand_hbond",  # T4 残基-配体氢键
    "translation_rotation",  # T5 平移-旋转复合
    "rotation_rotation",     # T6 旋转-旋转复合
    "ligand_docking",        # T7 配体对接
    "interaction_location",  # T8 相互作用定位
    "pocket_ligand_hbond",   # T9 口袋-配体氢键
]


class MISISubtask:
    """MiSI-Bench 微观空间智能子任务 — 分子结构 3D 空间推理.

    full set = 163K QA + 588K images + 4000 复合物. 默认 n_per_task=10 跑 subset
    验证用 (90 题), full 跑设 n_per_task=None 或大数.

    用法:
        sub = MISISubtask(n_per_task=10)
        if sub.is_available():
            report = sub.run(lambda images, q: agent.generate(images, q))
    """

    NAME = "misi_microscopic_spatial"

    def __init__(self, n_per_task: int | None = 10) -> None:
        """n_per_task=None 表示跑全集 (每 task 全部 sample)."""
        self._n_per_task = n_per_task

    def is_available(self) -> bool:
        """flag + datasets + rdkit 是否就位. 任一缺失返 False, run() 会跳过."""
        if os.environ.get("HUGINN_USE_MISI", "0") != "1":
            return False
        try:
            import datasets  # noqa: F401
            import rdkit  # noqa: F401
            return True
        except ImportError:
            return False

    def run(
        self,
        agent_inference_fn: Callable[[list, str], str],
    ) -> dict[str, Any]:
        """跑 MiSI-Bench subset.

        agent_inference_fn(images: list, question: str) -> str
            接收多视角图像 list + 问题文本, 返回答案.

        返回:
            {"skipped": True} — flag off 或包未装
            {"per_task": {...}, "overall": {...}} — 跑完的汇总
        """
        if not self.is_available():
            logger.info("MiSI skipped (flag off or packages not installed)")
            return {"skipped": True}

        from datasets import load_dataset

        per_task: dict[str, dict[str, Any]] = {}
        all_results: list[dict[str, Any]] = []

        for task_id in MISI_TASKS:
            try:
                # config_name = task_id. 失败时兜底不带 config (按 task 字段过滤).
                try:
                    ds = load_dataset(_MISI_DATASET, task_id, split="test")
                except Exception:
                    ds = load_dataset(_MISI_DATASET, split="test")
                    if "task" in ds.column_names:
                        ds = ds.filter(lambda r: r["task"] == task_id)
            except Exception as e:
                logger.warning("load_dataset %s failed: %s", task_id, e)
                continue

            if self._n_per_task is not None:
                ds = ds.select(range(min(self._n_per_task, len(ds))))

            results: list[dict[str, Any]] = []
            for row in ds:
                images = row.get("images", [])
                question = row.get("question", "")
                try:
                    predicted = agent_inference_fn(images, question)
                except Exception as e:
                    logger.debug("agent_inference_fn failed: %s", e)
                    results.append({"correct": False})
                    continue

                correct = _evaluate_task(task_id, predicted, row)
                results.append({"correct": correct})

            n = len(results)
            success = sum(1 for r in results if r["correct"])
            per_task[task_id] = {
                "accuracy": success / n if n > 0 else 0.0,
                "n": n,
            }
            all_results.extend(results)

        n_total = len(all_results)
        success_total = sum(1 for r in all_results if r["correct"])
        overall = {
            "mean_accuracy": success_total / n_total if n_total > 0 else 0.0,
            "n_total": n_total,
        }
        return {"per_task": per_task, "overall": overall}


def _evaluate_task(task_id: str, predicted: str, row: dict) -> bool:
    """按 task 类型分流 evaluation.

    ponytail: T1/T2/T3/T5/T6 多选 QA 用字符串匹配 (跟 atomworld evaluate 风格一致,
    不上 NLP 语义匹配). T4/T9 氢键 F1 跟 T7/T8 verifiable metric 是 P0 简化版,
    后续可升级. ceiling: 字符串匹配对 "A" vs "A. ..." 等 format 差异敏感, 多选已用
    首字符匹配缓解; T4/T7/T8/T9 精确匹配对 whitespace/case 敏感, 升级路径接 rdkit.
    """
    gt = str(row.get("answer", "")).strip()
    pred = str(predicted).strip()

    if task_id in ("translation", "rotation", "zooming",
                   "translation_rotation", "rotation_rotation"):
        # 多选 QA: 答案是 A/B/C/D, 首字符匹配
        if not gt or not pred:
            return False
        return pred[0].upper() == gt[0].upper()

    if task_id in ("residue_ligand_hbond", "pocket_ligand_hbond",
                   "ligand_docking", "interaction_location"):
        # 氢键 F1 / RMSD / 质心距离: P0 精确匹配, 后续接 rdkit 几何验证
        return bool(pred) and pred == gt

    # 未知 task, 保守 False
    return False


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """3 场景: flag off 跳过 / flag on 包未装跳过 / mock 路径不崩.

    ponytail: 非平凡逻辑留 runnable check. 不依赖真 datasets/rdkit 包也不调真 LLM.
    ceiling: 没验 HF datasets 真 DataFrame 路径 (需真包+网络), 升级路径见 acceptance test.
    """
    import os
    orig_flag = os.environ.get("HUGINN_USE_MISI", "0")
    try:
        # 1. flag off -> is_available False, run 返回 skipped
        os.environ["HUGINN_USE_MISI"] = "0"
        sub = MISISubtask(n_per_task=2)
        assert sub.is_available() is False, "flag off 时 is_available 应 False"
        report = sub.run(lambda imgs, q: "")
        assert report == {"skipped": True}, f"flag off 应返 skipped, got {report}"
        print("1. flag off -> skipped OK")

        # 2. flag on 但包未装 (datasets/rdkit 在测试环境可能没装) -> is_available False
        os.environ["HUGINN_USE_MISI"] = "1"
        try:
            import datasets  # noqa: F401
            import rdkit  # noqa: F401
            packages_ok = True
        except ImportError:
            packages_ok = False

        if not packages_ok:
            assert sub.is_available() is False, "包未装时 is_available 应 False"
            report = sub.run(lambda imgs, q: "")
            assert report == {"skipped": True}
            print("2. flag on + 包未装 -> skipped OK")
        else:
            # 包装了, 跳过降级场景, 直接验 mock 路径
            print("2. datasets/rdkit 已装, 跳过包未装场景")

        # 3. mock load_dataset 路径: 验 _evaluate_task 逻辑不崩
        # ponytail: 不调真 HF, 只验 evaluate 分流逻辑
        assert _evaluate_task("translation", "A", {"answer": "A"}) is True
        assert _evaluate_task("translation", "B", {"answer": "A"}) is False
        assert _evaluate_task("rotation", "C. ...", {"answer": "C"}) is True
        assert _evaluate_task("residue_ligand_hbond", "exact", {"answer": "exact"}) is True
        assert _evaluate_task("residue_ligand_hbond", "wrong", {"answer": "exact"}) is False
        assert _evaluate_task("unknown_task", "x", {"answer": "x"}) is False
        # 空 edge case
        assert _evaluate_task("translation", "", {"answer": "A"}) is False
        assert _evaluate_task("translation", "A", {}) is False
        print("3. _evaluate_task 分流 OK")

    finally:
        os.environ["HUGINN_USE_MISI"] = orig_flag

    print("misi_bench selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
