"""C2ST evaluator tool — Classifier Two-Sample Test.

治 ζ_c2st: agent 反复写 classifier 评估 posterior 质量.
统一工具: 训 RandomForest 区分两组样本, 返回 {c2st, roc_auc, accuracy}.
0.5 = 同分布, 1.0 = 完全可分. 保存到 outputs/c2st_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class C2STInput(BaseModel):
    x_samples: str = Field(
        description="JSON-encoded 2D array [[...], ...] — sample set X (e.g. posterior samples)"
    )
    y_samples: str = Field(
        description="JSON-encoded 2D array [[...], ...] — sample set Y (e.g. ground truth samples)"
    )
    output_path: str = Field(
        default="outputs/c2st_results.json",
        description="Output JSON path"
    )
    test_size: float = Field(default=0.3, gt=0, lt=1)
    random_state: int = Field(default=42)
    n_estimators: int = Field(default=100, gt=0)


class C2STEvaluatorTool(HuginnTool):
    """Classifier Two-Sample Test using sklearn RandomForest."""

    name = "c2st_evaluator_tool"
    category = "analysis"
    description = (
        "Classifier Two-Sample Test (C2ST): train a RandomForest to distinguish "
        "two sample sets. Returns c2st_score (0.5=same, 1.0=perfectly separable), "
        "roc_auc, accuracy. Use to evaluate posterior quality vs ground truth."
    )
    destructive = False
    input_schema = C2STInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = C2STInput(**args)

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, accuracy_score
        except ImportError as e:
            return ToolResult(
                data=None, success=False,
                error=f"scikit-learn required: {e}. Run: pip install scikit-learn",
            )

        def _is_numeric(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        def _extract_list(obj):
            # 递归从嵌套 dict/list 中扒出纯数值 2D list. agent 传各种格式:
            #   {"samples": [...]}                  (M5 一层 dict)
            #   {"samples": {"theta": [...]}}       (M6 嵌套 dict)
            #   [{"key": val}, ...]                 (M7 list 含 dict 元素, max-by-len 误选)
            # 一层 .get("samples") 不够, max(seen, key=len) 也不够 — 必须过滤掉含
            # dict 元素的 list, 否则 np.array(..., dtype=float) 在 dict 上崩.
            seen = []
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, list):
                    # 只收纯数值 list 或纯数值 2D list (元素全为数值 或 元素全为数值 list)
                    if cur and all(_is_numeric(e) for e in cur):
                        seen.append(cur)
                    elif cur and all(
                        isinstance(e, list) and e and all(_is_numeric(v) for v in e)
                        for e in cur
                    ):
                        seen.append(cur)
                    else:
                        # 含 dict/str 等, 继续往里挖
                        stack.extend(cur)
                elif isinstance(cur, dict):
                    for v in cur.values():
                        stack.append(v)
            if not seen:
                raise ValueError(f"no numeric list found in {type(obj).__name__}")
            # 取最长的 (posterior samples 通常比 metadata 长)
            return max(seen, key=len)

        try:
            X = np.array(_extract_list(json.loads(input_data.x_samples)), dtype=float)
            Y = np.array(_extract_list(json.loads(input_data.y_samples)), dtype=float)
        except (json.JSONDecodeError, ValueError) as e:
            return ToolResult(data=None, success=False, error=f"Invalid sample JSON: {e}")

        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
            return ToolResult(
                data=None, success=False,
                error=f"Samples must be 2D arrays with same dim. Got X{X.shape}, Y{Y.shape}",
            )

        # 平衡样本数 (取较小值)
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]

        # 构造二分类: X=label 0, Y=label 1
        data = np.vstack([X, Y])
        labels = np.concatenate([np.zeros(n), np.ones(n)])

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=input_data.test_size,
            random_state=input_data.random_state, stratify=labels,
        )

        clf = RandomForestClassifier(
            n_estimators=input_data.n_estimators,
            random_state=input_data.random_state,
        )
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        c2st = float(roc_auc_score(y_test, y_proba))
        acc = float(accuracy_score(y_test, y_pred))

        result = {
            "c2st_score": round(c2st, 4),
            "roc_auc": round(c2st, 4),  # 同义, C2ST 就是 AUC
            "accuracy": round(acc, 4),
            "n_samples_per_set": n,
            "sample_dim": int(X.shape[1]),
            "interpretation": (
                "same distribution" if abs(c2st - 0.5) < 0.05
                else "different distributions" if c2st > 0.7
                else "weakly distinguishable"
            ),
        }

        out_path = Path(input_data.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return ToolResult(
            data=result,
            success=True,
            side_effects=[str(out_path)],
        )


if __name__ == "__main__":
    import asyncio

    async def _test():
        tool = C2STEvaluatorTool()
        rng = np.random.default_rng(0)
        # 同分布 → c2st ≈ 0.5
        same_x = json.dumps(rng.standard_normal((200, 4)).tolist())
        same_y = json.dumps(rng.standard_normal((200, 4)).tolist())
        r1 = await tool.call({"x_samples": same_x, "y_samples": same_y, "output_path": "_test_c2st_same.json"})
        print(f"same dist: c2st={r1.data['c2st_score']} ({r1.data['interpretation']})")
        assert abs(r1.data["c2st_score"] - 0.5) < 0.15, f"expected ~0.5, got {r1.data['c2st_score']}"

        # 不同分布 → c2st > 0.8
        diff_x = json.dumps(rng.standard_normal((200, 4)).tolist())
        diff_y = json.dumps((rng.standard_normal((200, 4)) + 3).tolist())  # 偏移均值
        r2 = await tool.call({"x_samples": diff_x, "y_samples": diff_y, "output_path": "_test_c2st_diff.json"})
        print(f"diff dist: c2st={r2.data['c2st_score']} ({r2.data['interpretation']})")
        assert r2.data["c2st_score"] > 0.8, f"expected >0.8, got {r2.data['c2st_score']}"

        Path("_test_c2st_same.json").unlink(missing_ok=True)
        Path("_test_c2st_diff.json").unlink(missing_ok=True)

        # M7 bug: agent 传 [{"meta": ...}, ...] 含 dict 元素的 list,
        # 旧 max(seen, key=len) 误选这个 list, np.array(dtype=float) 在 dict 上崩.
        # 修复后应跳过含 dict 的 list, 找到同 dict 内嵌的纯数值 list.
        buggy_x = json.dumps([
            {"task": "linear_gaussian", "samples": rng.standard_normal((200, 4)).tolist()}
        ])
        buggy_y = json.dumps([
            {"task": "linear_gaussian", "samples": (rng.standard_normal((200, 4)) + 3).tolist()}
        ])
        r3 = await tool.call({"x_samples": buggy_x, "y_samples": buggy_y, "output_path": "_test_c2st_buggy.json"})
        print(f"nested-dict-with-list-of-dict: c2st={r3.data['c2st_score']}")
        assert r3.success and r3.data["c2st_score"] > 0.8, f"M7 bug not fixed: {r3.error}"
        Path("_test_c2st_buggy.json").unlink(missing_ok=True)

        print("[c2st_tool] self-check OK")

    asyncio.run(_test())
