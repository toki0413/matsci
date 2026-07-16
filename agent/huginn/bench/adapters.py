"""开源 benchmark 数据集 adapter.

优先 ModelScope (国内直连免翻墙), 降级到 HF mirror.
对标社区: MMMU/GPQA/HLE/MMLU/MMLU-Pro/CMMLU/SciQ/ARC.

题量来源:
  - MMLU (HF): 14042 test (材料/化学/物理等 57 学科)
  - MMLU-Pro (ModelScope): ~12032 test (10 选项, 更难)
  - CMMLU (ModelScope): 中文 67 学科, 含高中/大学物理化学
  - SciQ (HF): 1000 test (自然科学)
  - ARC-Challenge (HF): 1172 test (科学推理)
  - GPQA (ModelScope): 448 PhD 级 (HF 上 gated, ModelScope 免 token)

evaluator 统一字母匹配 (A-J), 不依赖 LLM judge.
"""

from __future__ import annotations

import os
import random
import re
from typing import Any

# 国内镜像加速
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .task import BenchmarkTask


def _eval_letter(answer: str):
    """生成一个 evaluator, 检查 agent 输出是否包含正确字母."""
    answer_upper = answer.upper().strip()

    def evaluate(output: str) -> tuple[bool, str, float]:
        # 提取输出里的字母 (A/B/C/D 或 a/b/c/d)
        # 优先匹配 "answer is X", "选X", "X." 等模式
        m = re.search(
            r"(?:answer\s*[:is]+\s*|选\s*|答案[:\s]*|^)\s*([A-Da-d])\b",
            output,
            re.IGNORECASE,
        )
        if m:
            got = m.group(1).upper()
            if got == answer_upper:
                return True, f"选 {got} (正确 {answer_upper})", 1.0
            return False, f"选 {got}, 正确 {answer_upper}", 0.0
        # 回退: 找输出里第一个独立字母
        m2 = re.search(r"\b([A-Da-d])\b", output)
        if m2:
            got = m2.group(1).upper()
            if got == answer_upper:
                return True, f"选 {got} (正确 {answer_upper})", 1.0
            return False, f"选 {got}, 正确 {answer_upper}", 0.0
        return False, f"未找到选项字母 (正确 {answer_upper})", 0.0

    return evaluate


def _format_mc(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    """格式化多选题 prompt."""
    if labels is None:
        labels = ["A", "B", "C", "D"][: len(choices)]
    lines = [question.strip(), ""]
    for lab, ch in zip(labels, choices):
        lines.append(f"{lab}. {ch}")
    lines.append("")
    lines.append("请只回答选项字母 (A/B/C/D).")
    return "\n".join(lines)


# ── MMLU adapter ─────────────────────────────────────────────────

# 材料科学/物理/化学相关学科 (MMLU subject 名)
MMLU_SCIENCE_SUBJECTS = {
    "college_chemistry", "college_physics", "high_school_chemistry",
    "high_school_physics", "chemistry", "physics",
    "astronomy", "college_biology", "high_school_biology",
    "conceptual_physics", "electrical_engineering",
    "machine_learning", "computer_science",
}


def load_mmlu_tasks(
    subject_filter: set[str] | None = None,
    max_tasks: int = 500,
    seed: int = 42,
) -> list[BenchmarkTask]:
    """从 MMLU test split 加载多选题.

    subject_filter: 限定学科 (默认用 MMLU_SCIENCE_SUBJECTS)
    max_tasks: 最多加载多少题 (从全量里随机抽样)
    """
    from datasets import load_dataset

    subjects = subject_filter or MMLU_SCIENCE_SUBJECTS
    ds = load_dataset("cais/mmlu", "all", split="test")
    # 过滤学科
    filtered = [ex for ex in ds if ex["subject"] in subjects]
    if len(filtered) > max_tasks:
        rng = random.Random(seed)
        filtered = rng.sample(filtered, max_tasks)

    tasks: list[BenchmarkTask] = []
    for i, ex in enumerate(filtered):
        choices = ex["choices"]
        if len(choices) != 4:
            continue
        answer_idx = ex["answer"]
        answer_letter = "ABCD"[answer_idx]
        labels = ["A", "B", "C", "D"]
        prompt = _format_mc(ex["question"], choices, labels)
        tasks.append(BenchmarkTask(
            id=f"mmlu-{ex['subject']}-{i}",
            category="knowledge",
            prompt=prompt,
            evaluator=_eval_letter(answer_letter),
            tags=["mmlu", "knowledge", ex["subject"]],
            requires_api_key=True,
            reference=f"答案: {answer_letter}. {choices[answer_idx]}",
            timeout_seconds=60.0,
        ))
    return tasks


# ── SciQ adapter ─────────────────────────────────────────────────

def load_sciq_tasks(max_tasks: int = 500, seed: int = 42) -> list[BenchmarkTask]:
    """从 SciQ test split 加载, 4 选 1."""
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="test")
    examples = list(ds)
    if len(examples) > max_tasks:
        rng = random.Random(seed)
        examples = rng.sample(examples, max_tasks)

    tasks: list[BenchmarkTask] = []
    for i, ex in enumerate(examples):
        correct = ex["correct_answer"]
        distractors = [ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        # 随机打乱选项位置
        rng = random.Random(seed + i)
        all_opts = [correct] + distractors
        rng.shuffle(all_opts)
        answer_idx = all_opts.index(correct)
        answer_letter = "ABCD"[answer_idx]
        prompt = _format_mc(ex["question"], all_opts)
        tasks.append(BenchmarkTask(
            id=f"sciq-{i}",
            category="knowledge",
            prompt=prompt,
            evaluator=_eval_letter(answer_letter),
            tags=["sciq", "knowledge", "science"],
            requires_api_key=True,
            reference=f"答案: {answer_letter}. {correct}",
            timeout_seconds=60.0,
        ))
    return tasks


# ── ARC adapter ──────────────────────────────────────────────────

def load_arc_tasks(max_tasks: int = 500, seed: int = 42) -> list[BenchmarkTask]:
    """从 ARC-Challenge test split 加载."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = list(ds)
    if len(examples) > max_tasks:
        rng = random.Random(seed)
        examples = rng.sample(examples, max_tasks)

    tasks: list[BenchmarkTask] = []
    for i, ex in enumerate(examples):
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]
        # 找 answer_key 对应的 label index
        if answer_key in labels:
            answer_idx = labels.index(answer_key)
        else:
            continue
        answer_letter = "ABCD"[answer_idx] if answer_idx < 4 else answer_key
        prompt = _format_mc(ex["question"], choices, labels)
        tasks.append(BenchmarkTask(
            id=f"arc-{i}",
            category="knowledge",
            prompt=prompt,
            evaluator=_eval_letter(answer_letter),
            tags=["arc", "knowledge", "science"],
            requires_api_key=True,
            reference=f"答案: {answer_letter}. {choices[answer_idx]}",
            timeout_seconds=60.0,
        ))
    return tasks


# ── GPQA adapter (ModelScope, HF 上是 gated) ─────────────────────

# ponytail: GPQA 在 HF 上 gated 需认证, ModelScope 上公开免 token.
# modelscope SDK 1.38 和 datasets 5.0 不兼容 (verification_mode 被移除),
# 所以绕过 MsDataset.load, 直接 dataset_file_download + pandas 读 CSV.

# GPQA 子领域里和材料/物理/化学相关的关键词 (宽松匹配)
GPQA_SCIENCE_KEYWORDS = (
    "physics", "chemistry", "biology", "quantum", "electromagnet",
    "thermodynamic", "astrophysic", "astronomy", "relativistic",
    "mechanic", "optic", "materials", "genetic", "molecular",
    "biochemistr", "organic", "inorganic", "electrochemistr",
    "high-energy", "particle physics", "condensed matter",
)


def _eval_letter_multi(answer: str, n_options: int):
    """多选项字母匹配 (支持 A-J, 用于 MMLU-Pro 10 选项等)."""
    answer_upper = answer.upper().strip()
    letters = "ABCDEFGHIJ"[:n_options]

    def evaluate(output: str) -> tuple[bool, str, float]:
        m = re.search(
            r"(?:answer\s*[:is]+\s*|选\s*|答案[:\s]*|^)\s*([A-Ja-j])\b",
            output, re.IGNORECASE,
        )
        if m:
            got = m.group(1).upper()
            if got == answer_upper:
                return True, f"选 {got} (正确 {answer_upper})", 1.0
            return False, f"选 {got}, 正确 {answer_upper}", 0.0
        m2 = re.search(r"\b([A-Ja-j])\b", output)
        if m2:
            got = m2.group(1).upper()
            if got == answer_upper:
                return True, f"选 {got} (正确 {answer_upper})", 1.0
            return False, f"选 {got}, 正确 {answer_upper}", 0.0
        return False, f"未找到选项字母 (正确 {answer_upper})", 0.0
    return evaluate


def _is_science_subdomain(sub: str) -> bool:
    s = (sub or "").lower()
    return any(kw in s for kw in GPQA_SCIENCE_KEYWORDS)


def load_gpqa_tasks(
    max_tasks: int = 100,
    seed: int = 42,
    subset: str = "gpqa_main",
    science_only: bool = True,
) -> list[BenchmarkTask]:
    """从 ModelScope 加载 GPQA PhD 级题目.

    HF 上 GPQA 是 gated dataset (需认证), ModelScope 公开免 token.
    science_only: 只保留物理/化学/材料等子领域 (默认 True)
    """
    import pandas as pd
    from modelscope.hub.file_download import dataset_file_download

    csv_path = dataset_file_download("modelscope/gpqa", f"{subset}.csv")
    df = pd.read_csv(csv_path)

    if science_only:
        df = df[df["Subdomain"].apply(_is_science_subdomain)]

    if len(df) > max_tasks:
        df = df.sample(n=max_tasks, random_state=seed)

    tasks: list[BenchmarkTask] = []
    for i, row in df.iterrows():
        question = str(row.get("Question") or "")
        correct = str(row.get("Correct Answer") or "")
        wrongs = [
            str(row.get("Incorrect Answer 1") or ""),
            str(row.get("Incorrect Answer 2") or ""),
            str(row.get("Incorrect Answer 3") or ""),
        ]
        if not question or question == "nan" or not correct or correct == "nan":
            continue
        # 打乱顺序
        rng = random.Random(seed + i)
        all_opts = [correct] + [w for w in wrongs if w and w != "nan"]
        rng.shuffle(all_opts)
        answer_idx = all_opts.index(correct)
        answer_letter = "ABCD"[answer_idx]
        labels = list("ABCD")[: len(all_opts)]
        prompt = _format_mc(question, all_opts, labels)
        sub = str(row.get("Subdomain") or "unknown")
        tasks.append(BenchmarkTask(
            id=f"gpqa-{sub[:20]}-{i}",
            category="knowledge",
            prompt=prompt,
            evaluator=_eval_letter_multi(answer_letter, len(all_opts)),
            tags=["gpqa", "knowledge", "phd", sub.lower().replace(" ", "_")],
            requires_api_key=True,
            reference=f"答案: {answer_letter}. {correct[:80]}",
            # GPQA 难, 深度思考 + 多轮 code_tool/web_search, 给足时间
            timeout_seconds=300.0,
        ))
    return tasks


# ── CMMLU adapter (中文, ModelScope) ─────────────────────────────

CMMLU_SCIENCE_SUBJECTS = {
    "high_school_physics", "high_school_chemistry", "high_school_biology",
    "college_physics", "college_chemistry", "college_biology",
    "conceptual_physics", "astronomy", "electrical_engineering",
    "machine_learning", "genetics", "virology", "nutrition",
    "food_science", "botany", "traditional_chinese_medicine",
}


def load_cmmlu_tasks(
    subject_filter: set[str] | None = None,
    max_tasks: int = 500,
    seed: int = 42,
) -> list[BenchmarkTask]:
    """从 ModelScope 加载 CMMLU 中文科学评测.

    CMMLU 是 zip 包, 下载解压后读 test/*.csv.
    CSV 列: Question, A, B, C, D, Answer (字母)
    """
    import os
    import zipfile

    import pandas as pd
    from modelscope.hub.file_download import dataset_file_download

    # 下载并解压 zip (缓存在 modelscope cache 目录)
    zip_path = dataset_file_download("opencompass/cmmlu", "cmmlu_v1_0_1.zip")
    extract_dir = os.path.join(os.path.dirname(zip_path), "cmmlu_extracted")
    if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    test_dir = os.path.join(extract_dir, "test")

    subjects = subject_filter or CMMLU_SCIENCE_SUBJECTS
    all_rows: list[dict] = []
    for sub in subjects:
        csv_file = os.path.join(test_dir, f"{sub}.csv")
        if not os.path.isfile(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        for _, row in df.iterrows():
            all_rows.append({
                "subject": sub,
                "question": str(row.get("Question") or ""),
                "A": str(row.get("A") or ""),
                "B": str(row.get("B") or ""),
                "C": str(row.get("C") or ""),
                "D": str(row.get("D") or ""),
                "answer": str(row.get("Answer") or ""),
            })

    if len(all_rows) > max_tasks:
        rng = random.Random(seed)
        all_rows = rng.sample(all_rows, max_tasks)

    tasks: list[BenchmarkTask] = []
    for i, ex in enumerate(all_rows):
        question = ex["question"]
        opts = [ex["A"], ex["B"], ex["C"], ex["D"]]
        opts = [o for o in opts if o and o != "nan"]
        if len(opts) != 4 or not question or question == "nan":
            continue
        answer_letter = ex["answer"].upper().strip()
        if answer_letter not in "ABCD":
            continue
        sub = ex["subject"]
        prompt = _format_mc(question, opts, ["A", "B", "C", "D"])
        tasks.append(BenchmarkTask(
            id=f"cmmlu-{sub}-{i}",
            category="knowledge",
            prompt=prompt,
            evaluator=_eval_letter(answer_letter),
            tags=["cmmlu", "knowledge", "chinese", sub],
            requires_api_key=True,
            reference=f"答案: {answer_letter}. {opts['ABCD'.index(answer_letter)]}",
            timeout_seconds=120.0,
        ))
    return tasks


# ── MMLU-Pro adapter (10 选项, 更难, ModelScope) ─────────────────

MMLU_PRO_SCIENCE_CATEGORIES = {
    "physics", "chemistry", "biology", "mathematics",
    "computer science", "engineering", "economics",
}


def load_mmlu_pro_tasks(
    category_filter: set[str] | None = None,
    max_tasks: int = 500,
    seed: int = 42,
) -> list[BenchmarkTask]:
    """从 ModelScope 加载 MMLU-Pro (10 选项, 更难).

    数据是 parquet, 直接 dataset_file_download + pandas 读.
    字段: question, options (list of 10), answer (字母 A-J), category
    """
    import pandas as pd
    from modelscope.hub.file_download import dataset_file_download

    pq_path = dataset_file_download(
        "modelscope/MMLU-Pro", "data/test-00000-of-00001.parquet"
    )
    df = pd.read_parquet(pq_path)

    cats = category_filter or MMLU_PRO_SCIENCE_CATEGORIES
    df = df[df["category"].str.lower().isin(cats)]

    if len(df) > max_tasks:
        df = df.sample(n=max_tasks, random_state=seed)

    tasks: list[BenchmarkTask] = []
    for i, row in df.iterrows():
        question = str(row["question"]) if pd.notna(row["question"]) else ""
        opts_raw = row["options"]
        opts = list(opts_raw) if opts_raw is not None else []
        if len(opts) < 4 or not question or question == "nan":
            continue
        answer_letter = str(row["answer"]).upper().strip() if pd.notna(row["answer"]) else ""
        if answer_letter not in "ABCDEFGHIJ"[: len(opts)]:
            continue
        n = len(opts)
        labels = list("ABCDEFGHIJ")[:n]
        prompt = _format_mc(question, [str(o) for o in opts], labels)
        cat = str(row["category"]) if pd.notna(row["category"]) else "unknown"
        answer_idx = labels.index(answer_letter) if answer_letter in labels else 0
        tasks.append(BenchmarkTask(
            id=f"mmlupro-{cat}-{i}",
            category="knowledge",
            prompt=prompt,
            evaluator=_eval_letter_multi(answer_letter, n),
            tags=["mmlu_pro", "knowledge", "hard", cat.lower().replace(" ", "_")],
            requires_api_key=True,
            reference=f"答案: {answer_letter}. {str(opts[answer_idx])[:80]}",
            timeout_seconds=90.0,
        ))
    return tasks


# ── 统一入口 ─────────────────────────────────────────────────────

def load_all_external(
    max_per_dataset: int = 400,
    seed: int = 42,
    include_modelscope: bool = True,
) -> list[BenchmarkTask]:
    """加载全部外部数据集, 合并成一个大列表.

    默认每个数据集抽样 400 题. include_modelscope=True 时加 GPQA/CMMLU/MMLU-Pro.
    """
    import logging
    log = logging.getLogger(__name__)
    tasks: list[BenchmarkTask] = []

    # HF 系 (国内走 hf-mirror.com)
    for name, fn in [
        ("MMLU", lambda: load_mmlu_tasks(max_tasks=max_per_dataset, seed=seed)),
        ("SciQ", lambda: load_sciq_tasks(max_tasks=max_per_dataset, seed=seed)),
        ("ARC", lambda: load_arc_tasks(max_tasks=max_per_dataset, seed=seed)),
    ]:
        try:
            tasks.extend(fn())
        except Exception as e:
            log.warning("%s 加载失败: %s", name, e)

    if not include_modelscope:
        return tasks

    # ModelScope 系 (国内直连, GPQA 免 token)
    for name, fn in [
        ("GPQA", lambda: load_gpqa_tasks(
            max_tasks=min(100, max_per_dataset), seed=seed)),
        ("CMMLU", lambda: load_cmmlu_tasks(
            max_tasks=max_per_dataset, seed=seed)),
        ("MMLU-Pro", lambda: load_mmlu_pro_tasks(
            max_tasks=max_per_dataset, seed=seed)),
    ]:
        try:
            tasks.extend(fn())
        except Exception as e:
            log.warning("%s (ModelScope) 加载失败: %s", name, e)
    return tasks


if __name__ == "__main__":
    tasks = load_all_external(max_per_dataset=50)
    print(f"Total: {len(tasks)} tasks")
    for t in tasks[:3]:
        print(f"\n--- {t.id} ---")
        print(t.prompt[:200])
        print(f"ref: {t.reference}")
        # 测 evaluator
        r = t.evaluate(t.reference or "A")
        print(f"eval: {r.passed} {r.reason}")
