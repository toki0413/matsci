"""RCB 启动前环境冒烟测试.

从 rcb_runner.py 剥离 — 纯环境探测, 不依赖 agent/LLM 状态.
检查 RDKit / sklearn GPR / torch / web_search 四项, 失败即 fail-fast.
"""
from __future__ import annotations

import sys
import time


def rcb_smoke_test() -> None:
    """P1-A7: 启动前环境冒烟 — 跑 RDKit + sklearn GP + torch 微型测试.

    RCB 三个出分任务 100% 命中工具链摩擦 (RDKit/sklearn/torch).
    不先冒烟, agent 跑半小时后才在 bash_tool 里撞 ImportError, 预算烧光.
    失败即 fail-fast 打印修复清单, 不进 async run.
    """
    t0 = time.time()
    failures: list[str] = []

    # 1. RDKit — Material 类任务核心依赖
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None, "MolFromSmiles returned None"
        _ = Descriptors.MolWt(mol)
    except Exception as e:
        failures.append(f"RDKit: {e}. Fix: pip install rdkit")

    # 2. sklearn GPR — Materials/Physics 建模主力
    try:
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        gpr = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=0)
        gpr.fit(X, y)
        _ = gpr.predict(np.random.randn(5, 3))
    except Exception as e:
        failures.append(f"sklearn GPR: {e}. Fix: pip install scikit-learn")

    # 3. torch — GNN/VAE 类任务核心依赖
    try:
        import torch
        t = torch.randn(3, 3)
        _ = t @ t.T
    except Exception as e:
        failures.append(
            f"torch: {e}. Fix: pip install torch "
            f"(CPU version: pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        )

    # 4. C4: web 检索健康检查 — arxiv API 探测, 失败只 warn 不 fail
    # bench 不强依赖网络 (可走 materials_database_tool/rag_tool 降级), 但检索链
    # 全灭时 agent 会裸答猜值 (roadmap 判据 8). 探测结果打印让用户知情.
    try:
        from huginn.tools.web_search_tool import web_search_health_check
        _ws_ok, _ws_msg = web_search_health_check(timeout=8.0)
        if _ws_ok:
            print(f"[SMOKE] web_search: {_ws_msg}")
        else:
            print(f"[SMOKE] web_search WARNING: {_ws_msg} (bench 可继续, 检索走降级)")
    except Exception as _e:
        print(f"[SMOKE] web_search WARNING: health check failed {_e} (bench 可继续)")

    if failures:
        print("=" * 60)
        print("SMOKE TEST FAILED — 环境冒烟未通过, 不启动 RCB run")
        print("=" * 60)
        for f in failures:
            print(f"  FAIL: {f}")
        print("\n修复后重试, 或用 HUGINN_SKIP_SMOKE=1 跳过 (风险自负)")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"[SMOKE] OK ({elapsed:.1f}s) — RDKit/sklearn/torch 就绪")
