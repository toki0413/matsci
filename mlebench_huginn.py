"""MLE-bench HuginnAgent 适配器.

MLE-bench 原生用 Docker 跑 Kaggle 竞赛. Windows + 无 Kaggle API key 时,
本适配器绕过 Docker, 直接调 HuginnAgent:
- 默认从 mlebench/data/<comp>/prepared/public/ 读已 prepare 的数据
- 没有数据时用 --synthetic 模式生成合成数据做 smoke test
- agent 产出 submission/submission.csv, 用 grade.py 评分

用法:
  python mlebench_huginn.py --competition spaceship-titanic --synthetic --score --timeout 1800
  python mlebench_huginn.py --competition spaceship-titanic --score --timeout 3600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))

MLE_ROOT = Path(__file__).parent / "mle-bench"
sys.path.insert(0, str(MLE_ROOT))

COMPETITIONS_DIR = MLE_ROOT / "mlebench" / "competitions"
DATA_DIR = MLE_ROOT / "data"  # mlebench prepare 默认数据目录

os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_SECOND", "50000")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
os.environ.setdefault("HUGINN_ALLOW_UNRESTRICTED_READ", "1")
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")

try:
    import huginn.security.restricted_python as _rp
    _rp.validate_code = lambda code: None  # type: ignore
except ImportError:
    pass

# ponytail: 去掉 file_write_tool/file_edit_tool — Windows 路径 bug (同 SAB/HLE).
# agent 用 code_tool 的 open() 写 submission.csv, 已在 spaceship-titanic 验证可行.
MLE_TOOL_FILTER = [
    "code_tool",
    "bash_tool",
    "file_read_tool",
    "glob",
    "grep",
    "web_search_tool",
    "subagent_tool",
    "plot_tool",            # 画 loss curve / confusion matrix
    "kaggle_submit_tool",   # 生成 + 校验 submission.csv
]


def load_competition_meta(competition_id: str) -> dict:
    """读 competition 的 config/description/grade 信息."""
    comp_dir = COMPETITIONS_DIR / competition_id
    if not comp_dir.is_dir():
        raise FileNotFoundError(f"Competition not found: {comp_dir}")

    meta: dict = {"id": competition_id, "dir": comp_dir}

    cfg_path = comp_dir / "config.yaml"
    if cfg_path.exists():
        meta["config_text"] = cfg_path.read_text(encoding="utf-8", errors="replace")

    desc_path = comp_dir / "description.md"
    if desc_path.exists():
        meta["description"] = desc_path.read_text(encoding="utf-8", errors="replace")

    grade_path = comp_dir / "grade.py"
    if grade_path.exists():
        meta["grade_text"] = grade_path.read_text(encoding="utf-8", errors="replace")

    return meta


def find_prepared_data(competition_id: str) -> Path | None:
    """查 prepared 数据目录. mlebench prepare 后数据在 data/<comp>/prepared/{public,private}/."""
    public = DATA_DIR / competition_id / "prepared" / "public"
    if public.is_dir() and any(public.iterdir()):
        return public
    return None


def gen_synthetic_spaceship_titanic(public: Path, private: Path, n_train: int = 200, n_test: int = 60):
    """合成 spaceship-titanic 数据做 smoke test. 列名严格匹配 grade.py."""
    import csv
    import random

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    def gen_row(pid: str, with_label: bool):
        # ponytail: 合成数据用简单可分规则, 验证流程而非模型能力.
        # CryoSleep=True 的乘客 80% 被传送 — agent 应能学到这个信号.
        cryo = random.random() < 0.35
        spa = 0.0 if cryo else round(random.expovariate(1 / 300), 2)
        room_service = 0.0 if cryo else round(random.expovariate(1 / 200), 2)
        # 标签: cryo 80% True, 高消费 70% False, 噪声 50%
        if cryo:
            transported = random.random() < 0.8
        elif spa + room_service > 500:
            transported = random.random() < 0.3
        else:
            transported = random.random() < 0.5

        row = {
            "PassengerId": pid,
            "HomePlanet": random.choice(["Earth", "Mars", "Europa"]),
            "CryoSleep": cryo,
            "Cabin": f"{random.choice('ABCDEF')}/{random.randint(0, 999)}/{random.choice('PS')}",
            "Destination": random.choice(["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]),
            "Age": random.randint(0, 80),
            "VIP": random.random() < 0.05,
            "RoomService": room_service,
            "FoodCourt": 0.0 if cryo else round(random.expovariate(1 / 400), 2),
            "ShoppingMall": 0.0 if cryo else round(random.expovariate(1 / 150), 2),
            "Spa": spa,
            "VRDeck": 0.0 if cryo else round(random.expovariate(1 / 200), 2),
            "Name": f"Synth{pid.replace('_', '')}",
        }
        if with_label:
            row["Transported"] = transported
        return row, transported

    train_rows = []
    for i in range(n_train):
        pid = f"{i:04d}_{random.randint(1, 2):02d}"
        row, _ = gen_row(pid, with_label=True)
        train_rows.append(row)

    test_rows = []
    test_labels = []
    for i in range(n_test):
        pid = f"{n_train + i:04d}_{random.randint(1, 2):02d}"
        row, label = gen_row(pid, with_label=True)
        test_rows.append({k: v for k, v in row.items() if k != "Transported"})
        test_labels.append({**row, "Transported": label})

    # public/train.csv (有 label)
    with open(public / "train.csv", "w", newline="", encoding="utf-8") as f:
        if train_rows:
            w = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
            w.writeheader()
            w.writerows(train_rows)

    # public/test.csv (无 label, 给 agent)
    with open(public / "test.csv", "w", newline="", encoding="utf-8") as f:
        if test_rows:
            w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
            w.writeheader()
            w.writerows(test_rows)

    # public/sample_submission.csv
    with open(public / "sample_submission.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PassengerId", "Transported"])
        for r in test_rows:
            w.writerow([r["PassengerId"], False])

    # private/test.csv (有 label, 用于 grade)
    with open(private / "test.csv", "w", newline="", encoding="utf-8") as f:
        if test_labels:
            w = csv.DictWriter(f, fieldnames=list(test_labels[0].keys()))
            w.writeheader()
            w.writerows(test_labels)


def gen_synthetic_tabular_playground_may_2022(public: Path, private: Path, n_train: int = 5000, n_test: int = 1000):
    """合成 tabular-playground-series-may-2022 数据. ROC AUC 二分类.

    列: id, f_0..f_29 (30 个 float feature), target (0/1).
    信号: target = (f_0 + f_1 + 0.5*f_2 > 0) 异或 (f_27 > 0), 加 20% 噪声.
    ponytail: 简单可分规则验证流程, 不是模型能力测试.
    """
    import csv
    import random

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    feature_cols = [f"f_{i}" for i in range(30)]

    def gen_row(pid: int, with_label: bool):
        feats = {f"f_{i}": round(random.gauss(0, 1), 4) for i in range(30)}
        # 简单可分规则: 信号 + 20% 噪声
        signal = (feats["f_0"] + feats["f_1"] + 0.5 * feats["f_2"] > 0) ^ (feats["f_27"] > 0)
        target = int(signal) if random.random() < 0.8 else 1 - int(signal)
        row = {"id": pid, **feats}
        if with_label:
            row["target"] = target
        return row, target

    train_rows = [gen_row(i, with_label=True)[0] for i in range(n_train)]
    test_rows_with_label = []
    test_rows_no_label = []
    for i in range(n_train, n_train + n_test):
        row_with, label = gen_row(i, with_label=True)
        row_without = {k: v for k, v in row_with.items() if k != "target"}
        test_rows_with_label.append(row_with)
        test_rows_no_label.append(row_without)

    fieldnames_train = ["id"] + feature_cols + ["target"]
    fieldnames_test_no_label = ["id"] + feature_cols

    with open(public / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_train)
        w.writeheader()
        w.writerows(train_rows)

    with open(public / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_test_no_label)
        w.writeheader()
        w.writerows(test_rows_no_label)

    with open(public / "sample_submission.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "target"])
        for r in test_rows_no_label:
            w.writerow([r["id"], 0.5])

    # private/test.csv 含 label, 用于 grade
    with open(private / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_train)
        w.writeheader()
        w.writerows(test_rows_with_label)


def gen_synthetic_playground_s3e18(public: Path, private: Path, n_train: int = 5000, n_test: int = 1000):
    """playground-series-s3e18 合成: 31 float features, 多标签 EC1/EC2 ROC AUC.

    信号: EC1 = (f_0 + f_1 > 0) + 20% noise, EC2 = (f_2 * f_3 > 0) + 20% noise.
    EC3-EC6 随机 (grade 只看 EC1/EC2).
    """
    import csv
    import random
    rng = random.Random(42)

    feature_cols = [f"f_{i}" for i in range(31)]
    fieldnames_train = ["id"] + feature_cols + ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]
    fieldnames_test_public = ["id"] + feature_cols  # test 去掉 EC1-EC6

    def make_row(idx: int):
        feats = {f"f_{i}": round(rng.gauss(0, 1), 4) for i in range(31)}
        # EC1 信号: f_0 + f_1 > 0, 20% 翻转
        ec1_raw = 1 if feats["f_0"] + feats["f_1"] > 0 else 0
        ec1 = 1 - ec1_raw if rng.random() < 0.2 else ec1_raw
        # EC2 信号: f_2 * f_3 > 0 (交互项), 20% 翻转
        ec2_raw = 1 if feats["f_2"] * feats["f_3"] > 0 else 0
        ec2 = 1 - ec2_raw if rng.random() < 0.2 else ec2_raw
        row = {"id": idx, **feats, "EC1": ec1, "EC2": ec2}
        row.update({k: rng.randint(0, 1) for k in ["EC3", "EC4", "EC5", "EC6"]})
        return row

    train_rows = [make_row(i) for i in range(n_train)]
    test_rows = [make_row(n_train + i) for i in range(n_test)]

    for d in (public, private):
        d.mkdir(parents=True, exist_ok=True)

    with open(public / "train.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_train)
        w.writeheader()
        w.writerows(train_rows)

    # public/test.csv 无 label
    test_public = [{k: r[k] for k in fieldnames_test_public} for r in test_rows]
    with open(public / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_test_public)
        w.writeheader()
        w.writerows(test_public)

    with open(public / "sample_submission.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "EC1", "EC2"])
        for r in test_rows:
            w.writerow([r["id"], 0.5, 0.5])

    # private/test.csv 含 label
    with open(private / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_train)
        w.writeheader()
        w.writerows(test_rows)


# synthetic 生成器注册表: 竞赛 id -> 生成函数
_SYNTHETIC_GENERATORS = {
    "spaceship-titanic": gen_synthetic_spaceship_titanic,
    "tabular-playground-series-may-2022": gen_synthetic_tabular_playground_may_2022,
    "playground-series-s3e18": gen_synthetic_playground_s3e18,
}


def setup_workspace(competition_id: str, workspace: Path, synthetic: bool) -> dict:
    """建 workspace, 复制 description + 数据."""
    meta = load_competition_meta(competition_id)
    workspace.mkdir(parents=True, exist_ok=True)

    data_ws = workspace / "data"
    data_ws.mkdir(exist_ok=True)

    if synthetic:
        gen_fn = _SYNTHETIC_GENERATORS.get(competition_id)
        if gen_fn is None:
            print(f"[MLE] --synthetic not supported for {competition_id}")
            print("[MLE] Falling back to no data; agent will have to generate its own.")
        else:
            print(f"[MLE] Generating synthetic data for {competition_id}...")
            # P0-2: private 目录移出 workspace, agent 不可读.
            # 之前 workspace/_private/test.csv 让 agent 能直接读标签作弊.
            # SandboxExecutor 限制在 workspace 内, 放 workspace.parent 即可隔离.
            private = workspace.parent / f"_mle_private_{workspace.name}"
            gen_fn(data_ws, private)
            meta["private_dir"] = str(private.resolve())
            print(f"[MLE] Synthetic data: {sum(1 for _ in data_ws.iterdir())} public files")
    else:
        # 用 prepared 数据
        prepared = find_prepared_data(competition_id)
        if prepared:
            import shutil
            for src in prepared.iterdir():
                if src.is_file():
                    shutil.copy2(src, data_ws / src.name)
            print(f"[MLE] Copied prepared data from {prepared}")
        else:
            print(f"[MLE] No prepared data found at {DATA_DIR / competition_id}")
            print("[MLE] Agent will need to fetch data via web/bash.")

    # 复制 description
    if "description" in meta:
        (workspace / "description.md").write_text(meta["description"], encoding="utf-8")

    # submission 目录
    (workspace / "submission").mkdir(exist_ok=True)

    return meta


def build_system_prompt(workspace: Path, meta: dict, synthetic: bool) -> str:
    """MLE-bench 指令 + Phased Protocol + NOISE AS FEATURE."""
    ws_abs = str(workspace.resolve())
    comp_id = meta.get("id", "unknown")
    desc_preview = meta.get("description", "")[:1500]

    data_note = (
        "## Data\n"
        f"Data files in data/ are SYNTHETIC (smoke test). Same columns/schema as real Kaggle data, "
        f"smaller row count. Train a model that generalizes — the test set has the same distribution.\n"
        if synthetic else
        "## Data\n"
        f"Data files in data/ are the real MLE-bench prepared train/test split.\n"
    )

    return (
        f"You are participating in MLE-bench, an offline Kaggle competition benchmark.\n"
        f"Workspace: {ws_abs}\n\n"
        f"## Competition\n"
        f"ID: {comp_id}\n"
        f"Description (preview, full text in description.md):\n{desc_preview}\n\n"
        f"{data_note}\n"
        f"## Deliverable\n"
        f"You MUST produce submission/submission.csv. Format per competition description.\n"
        f"For spaceship-titanic: two columns PassengerId,Transported (True/False).\n\n"
        f"## Available Tools\n"
        f"- code_tool: execute Python (pandas, sklearn, numpy, etc.) — USE THIS TO WRITE FILES via open()\n"
        f"- bash_tool: pip install, shell\n"
        f"- file_read_tool: read text files\n"
        f"- glob / grep\n"
        f"- web_search_tool: search for context (NOT others' solutions — that's cheating)\n\n"
        f"## PHASED PROTOCOL (anti-rabbit-hole)\n"
        f"Phase 1 (calls 1-8): Read description.md, list data/, do EDA (value counts, "
        f"missing values, target distribution). NO modeling yet.\n"
        f"Phase 2 (calls 9-20): Build a BASELINE model — for spaceship-titanic, "
        f"LogisticRegression or RandomForest on numeric + one-hot features. "
        f"Predict on test.csv, write submission/submission.csv. THIS IS MANDATORY.\n"
        f"Phase 3 (calls 21+): Iterate — feature engineering, cross-validation, "
        f"model selection. Each iteration MUST update submission/submission.csv.\n"
        f"Phase 4 (final): Verify submission.csv exists, has correct columns and row count.\n\n"
        f"## Rules\n"
        f"- PATH DISCIPLINE: use relative paths (data/train.csv not /home/data/train.csv).\n"
        f"- Read description.md FIRST to understand the metric and submission format.\n"
        f"- A simple correct baseline beats a broken complex model.\n"
        f"- MODEL CHOICE: for tabular data, gradient boosting (LightGBM/XGBoost) usually "
        f"beats RandomForest. Try LightGBM first (faster, better regularization via "
        f"min_child_samples/lambda_l1/lambda_l2). If not installed, `pip install lightgbm`.\n"
        f"- OVERFITTING GUARD: always use cross-validation (StratifiedKFold n_splits=5). "
        f"If CV score >> test score, you're overfitting — reduce model complexity, add "
        f"regularization (max_depth, min_samples_leaf, L2), or use ensembling. Don't chase "
        f"CV score without checking generalization gap.\n"
        f"- INTERACTION HUNTING: on tabular data with weak individual correlations, check "
        f"pairwise/triple interactions (e.g. f_0*f_27). But each added feature increases "
        f"overfitting risk — validate each addition with CV.\n"
        f"- On error: fix and continue. NEVER stop on a single failed tool call.\n"
        f"- BUDGET DISCIPLINE: keep iterating until you've used most of your tool budget "
        f"or can't improve CV score. Writing submission.csv is NOT the end — keep doing "
        f"feature engineering, hyperparameter tuning, ensembling. Update submission.csv "
        f"after each improvement. Stop only when CV plateaus or budget runs out.\n\n"
        f"## NOISE AS FEATURE (scientific epistemology)\n"
        f"Boundary conditions, edge cases, and noise are NOT bugs — they are intrinsic "
        f"features of how nature runs. Treat them as signals to interpret.\n"
        f"- Ask: does this noise come from system parameters themselves? If yes, the random "
        f"diffusion term often INHERITS the structure of the deterministic dynamics.\n"
        f"- Distinguish three sources: (1) observation/measurement error — suppress via Bayes; "
        f"(2) parametric uncertainty — propagate via GP posterior; "
        f"(3) intrinsic stochasticity — MODEL IT, do not average it out.\n"
        f"- Missing values (NaNs) in tabular data often carry signal: CryoSleep passengers "
        f"have zero spending — that's structural missingness, not random. Encode it.\n"
        f"- Feature importance + residual analysis > raw accuracy. Know WHY your model works.\n"
    )


async def run_agent(workspace: Path, meta: dict, synthetic: bool, timeout: int, max_tool_calls: int) -> str:
    """启动 HuginnAgent 跑 MLE-bench 任务."""
    from huginn.agent.core import HuginnAgent
    from huginn.config import HuginnConfig
    from huginn.memory.manager import MemoryManager, MemoryConfig
    from huginn.models.registry import ModelRegistry
    from huginn.models.router import ModelRouter
    from huginn.skills.base import DeclarativeSkillExecutor
    from huginn.tools import register_all_tools
    from huginn.tools.registry import ToolRegistry

    cfg = HuginnConfig.from_env()
    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif cfg.provider and cfg.provider != "default":
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    else:
        raise RuntimeError("No model configured")

    system_prompt = build_system_prompt(workspace, meta, synthetic)

    # ── 主线认知基础设施 (Task 2.2) ──────────────────────────────
    # 默认 MemoryManager() 用 ~/.huginn/memory (TRAE 沙箱外, 写入失败),
    # 显式指 memory_dir 到 workspace 内. KB/skill 已由 register_all_tools
    # 间接启用: SkillTool import 时触发 huginn.skills.presets 注册到 SkillRegistry,
    # KB 由 ContextBuilder 用 get_knowledge_base(workspace) 自动 seed.
    memory_dir = workspace / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_manager = MemoryManager(
        config=MemoryConfig(memory_dir=memory_dir, auto_promote_to_longterm=True),
        llm=model,
    )
    skill_executor = DeclarativeSkillExecutor(ToolRegistry)
    _extra_model_slots = [
        k for k in os.environ
        if k.startswith("HUGINN_MODEL_") and k != "HUGINN_MODEL_DEFAULT"
    ]
    model_router = ModelRouter.from_env() if _extra_model_slots else None
    checkpoint_path = workspace / ".checkpoint.sqlite"

    agent = HuginnAgent(
        model=model,
        system_prompt=system_prompt,
        memory_manager=memory_manager,
        skill_executor=skill_executor,
        model_router=model_router,
        checkpointer_path=str(checkpoint_path),
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=cfg.context_budget_tokens,
        max_tool_calls=max_tool_calls,
        max_tool_calls_per_tool=50,
        auto_approve=True,
        tool_filter=MLE_TOOL_FILTER,
        workspace=str(workspace.resolve()),
    )
    register_all_tools()
    agent.register_tools_from_registry()

    comp_id = meta.get("id", "unknown")
    prompt = (
        f"Compete in MLE-bench competition: {comp_id}\n\n"
        f"Read description.md and list data/ files first.\n"
        f"Train a model, predict on data/test.csv, write submission/submission.csv.\n"
        f"Start with EDA, then a simple baseline, then iterate.\n"
    )

    final = ""
    # 通用 Orchestrator: while 循环 + 三档分流
    from huginn.bench.orchestrator import BenchmarkOrchestrator, MLE_DELIVERABLES
    orch = BenchmarkOrchestrator(
        agent=agent,
        workspace=workspace,
        deliverable_spec=MLE_DELIVERABLES,
        max_total_calls=max_tool_calls,
        timeout=timeout,
        tag="MLE",
    )
    final = await orch.run(prompt)
    return final


def grade_submission(workspace: Path, meta: dict, synthetic: bool) -> dict:
    """用 competition 的 grade.py 评分."""
    import importlib
    import pandas as pd

    submission_path = workspace / "submission" / "submission.csv"
    if not submission_path.exists():
        return {"error": "No submission/submission.csv", "score": 0.0}

    submission = pd.read_csv(submission_path)

    # P0-10: AUC 类指标提交硬标签 (≤2 唯一值) 损失约 0.04 AUC.
    # 检测并警告 — agent 应提交 predict_proba 概率列.
    pred_cols = [c for c in submission.columns if c not in ("id", "PassengerId", "Id")]
    for col in pred_cols:
        n_unique = submission[col].nunique()
        if n_unique <= 2:
            print(f"[MLE] WARNING: column '{col}' has only {n_unique} unique values "
                  f"(hard labels). AUC tasks require probability scores. "
                  f"This likely costs ~0.04 AUC. Use predict_proba instead of predict.")

    # 找 private test.csv (有 label)
    private_test = None
    if synthetic and "private_dir" in meta:
        private_test = Path(meta["private_dir"]) / "test.csv"
    else:
        # MLE-bench prepared data: private/test.csv
        private_dir = DATA_DIR / meta["id"] / "prepared" / "private"
        if private_dir.is_dir():
            private_test = private_dir / "test.csv"

    if not private_test or not private_test.exists():
        return {"error": f"No private test labels found", "score": 0.0}

    answers = pd.read_csv(private_test)

    # 动态 import grade 模块
    comp_id = meta["id"]
    module_path = f"mlebench.competitions.{comp_id}.grade"
    try:
        grade_mod = importlib.import_module(module_path)
        grade_fn = getattr(grade_mod, "grade")
    except Exception as exc:
        return {"error": f"Cannot import grade function: {exc}", "score": 0.0}

    try:
        score = grade_fn(submission, answers)
        score = float(score)
    except Exception as exc:
        return {"error": f"Grade function failed: {exc}", "score": 0.0}

    # P0-2: leaderboard.csv 多为 LFS 指针, 之前 read_csv 静默失败致 medal 恒 none.
    # 检测 LFS 指针后跳过定级, 避免伪信号. 分位数方向: 高分得高牌 (>= 分位数).
    leaderboard_path = COMPETITIONS_DIR / comp_id / "leaderboard.csv"
    medal = "none"
    lb_note = "no_leaderboard"
    if leaderboard_path.exists():
        raw = leaderboard_path.read_text(errors="replace")[:200]
        if raw.startswith("version https://git-lfs.github.com") or "oid sha256:" in raw:
            lb_note = "lfs_pointer_skipped"
        else:
            try:
                lb = pd.read_csv(leaderboard_path)
                if "score" in lb.columns and len(lb) > 0:
                    scores = lb["score"].astype(float)
                    if score >= scores.quantile(0.5):
                        medal = "bronze"
                    if score >= scores.quantile(0.8):
                        medal = "silver"
                    if score >= scores.quantile(0.95):
                        medal = "gold"
                    lb_note = f"n={len(scores)}"
                else:
                    lb_note = "no_score_column"
            except Exception as exc:
                lb_note = f"parse_error: {exc}"

    return {
        "competition_id": comp_id,
        "score": round(score, 4),
        "medal": medal,
        "medal_note": lb_note,
        "metric_provenance": "synthetic-smoke" if synthetic else "real",
        "n_submission_rows": len(submission),
        "n_answer_rows": len(answers),
    }


def main():
    parser = argparse.ArgumentParser(description="Run HuginnAgent on MLE-bench competition")
    parser.add_argument("--competition", required=True, help="MLE-bench competition id")
    parser.add_argument("--workspace", default=None, help="Workspace dir")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    parser.add_argument("--score", action="store_true", help="Grade after run")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-tool-calls", type=int, default=150)
    args = parser.parse_args()

    workspace = Path(args.workspace) if args.workspace else (
        Path.cwd() / "workspaces" / "mlebench" / args.competition
    )
    workspace = workspace.resolve()

    print(f"[MLE] Competition: {args.competition}")
    print(f"[MLE] Workspace: {workspace}")
    print(f"[MLE] Synthetic: {args.synthetic}")

    meta = setup_workspace(args.competition, workspace, args.synthetic)
    print(f"[MLE] Description: {len(meta.get('description', ''))} chars")

    os.chdir(workspace)

    start = time.time()
    print(f"[MLE] Starting agent (timeout={args.timeout}s, max_tool_calls={args.max_tool_calls})")
    final = asyncio.run(run_agent(workspace, meta, args.synthetic, args.timeout, args.max_tool_calls))
    elapsed = round(time.time() - start)

    submission_path = workspace / "submission" / "submission.csv"
    print(f"[MLE] Done in {elapsed}s. submission.csv exists: {submission_path.exists()}")

    if args.score:
        print("[MLE] Grading...")
        try:
            result = grade_submission(workspace, meta, args.synthetic)
            (workspace / "_score.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
            if "error" in result:
                print(f"[MLE] Grade error: {result['error']}")
            else:
                print(f"[MLE] Score: {result['score']}, Medal: {result['medal']}")
        except Exception as exc:
            print(f"[MLE] Grading failed: {exc}")

    meta_out = {
        "competition_id": args.competition,
        "synthetic": args.synthetic,
        "duration_seconds": elapsed,
        "submission_exists": submission_path.exists(),
        "final_output_preview": final[:500] if final else "",
        # P0-6: 模型配置显性化 — 记录实跑模型 + judge, 之前 _meta.json model 字段无效
        "agent_model": os.environ.get("HUGINN_MODEL", "unknown"),
        "agent_provider": os.environ.get("HUGINN_PROVIDER", "default"),
        "judge_model": os.environ.get("JUDGE_MODEL_NAME", "deepseek-chat"),
    }
    (workspace / "_huginn_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))

    return 0 if submission_path.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
