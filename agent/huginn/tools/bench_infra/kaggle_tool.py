"""Kaggle submit tool — 生成 + 校验 submission.csv.

治 ζ_kaggle: mlebench agent 反复写 CSV 格式化代码, 列名/行数错失分.
统一工具: 接 predictions + id_column + class_columns, 生成规范 CSV.
校验: 行数 / 列名 / 数据类型 / 缺失值.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class KaggleSubmitInput(BaseModel):
    predictions: str = Field(
        description="JSON-encoded predictions. Format: "
        "regression -> {id: value, ...} or [[id, value], ...]; "
        "classification -> {id: {class: prob, ...}, ...} or [{id, class, prob}, ...]"
    )
    id_column: str = Field(default="id", description="Column name for IDs")
    class_columns: list[str] | None = Field(
        default=None,
        description="Classification: explicit class column names. If None, infer from data."
    )
    output_path: str = Field(default="submission/submission.csv")
    expected_n_rows: int | None = Field(
        default=None, ge=0,
        description="If set, validate that submission has exactly this many rows"
    )
    task_type: str = Field(
        default="auto",
        description="auto | regression | classification. auto infers from predictions shape.",
    )


class KaggleSubmitTool(HuginnTool):
    """Generate and validate a Kaggle-format submission.csv."""

    name = "kaggle_submit_tool"
    category = "analysis"
    description = (
        "Generate a Kaggle-format submission.csv from predictions. "
        "Validates row count, column names, missing values. "
        "Supports regression (id, target) and classification (id, class_1, class_2, ...)."
    )
    destructive = False
    input_schema = KaggleSubmitInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = KaggleSubmitInput(**args)

        try:
            preds = json.loads(input_data.predictions)
        except json.JSONDecodeError as e:
            return ToolResult(data=None, success=False, error=f"Invalid predictions JSON: {e}")

        # 统一成 list of (id, value_or_dict)
        normalized = self._normalize(preds)
        if not normalized:
            return ToolResult(data=None, success=False, error="Empty predictions")

        # 推断 task_type
        first_val = normalized[0][1]
        is_classification = isinstance(first_val, dict)
        if input_data.task_type == "auto":
            task_type = "classification" if is_classification else "regression"
        else:
            task_type = input_data.task_type

        # 收集 class columns
        if task_type == "classification":
            if input_data.class_columns:
                class_cols = input_data.class_columns
            else:
                # 从数据推断
                class_set = set()
                for _, v in normalized:
                    if isinstance(v, dict):
                        class_set.update(v.keys())
                class_cols = sorted(class_set)
                if not class_cols:
                    return ToolResult(data=None, success=False, error="Classification but no class labels found")
        else:
            class_cols = ["target"]

        # 写 CSV
        out_path = Path(input_data.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = [input_data.id_column] + class_cols
        n_rows = 0
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for id_, val in normalized:
                if task_type == "classification":
                    if isinstance(val, dict):
                        row = [id_] + [val.get(c, 0.0) for c in class_cols]
                    else:
                        # 单值 → one-hot 推断
                        row = [id_] + [1.0 if c == str(val) else 0.0 for c in class_cols]
                else:
                    row = [id_, val]
                writer.writerow(row)
                n_rows += 1

        # 校验
        issues = []
        if input_data.expected_n_rows is not None and n_rows != input_data.expected_n_rows:
            issues.append(f"row count {n_rows} != expected {input_data.expected_n_rows}")

        # 回读校验缺失值
        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            read_header = next(reader)
            if read_header != header:
                issues.append(f"header mismatch: {read_header} != {header}")
            empty_cells = 0
            for row in reader:
                for cell in row:
                    if cell == "" or cell is None:
                        empty_cells += 1
            if empty_cells:
                issues.append(f"{empty_cells} empty cells")

        result = {
            "output_path": str(out_path),
            "n_rows": n_rows,
            "n_columns": len(header),
            "columns": header,
            "task_type": task_type,
            "file_size_bytes": out_path.stat().st_size,
            "validation": {
                "passed": len(issues) == 0,
                "issues": issues,
            },
        }

        return ToolResult(
            data=result,
            success=len(issues) == 0,
            side_effects=[str(out_path)],
            error="; ".join(issues) if issues else None,
        )

    def _normalize(self, preds) -> list[tuple[Any, Any]]:
        """把各种 prediction 格式统一成 [(id, value_or_dict), ...]."""
        if isinstance(preds, dict):
            return list(preds.items())
        if isinstance(preds, list):
            out = []
            for item in preds:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    out.append((item[0], item[1]))
                elif isinstance(item, dict):
                    # 从 dict 里取 id 字段
                    id_ = item.get("id", item.get("Id", item.get("ID")))
                    if id_ is None:
                        continue
                    # 剩下的当 class probs 或 target
                    rest = {k: v for k, v in item.items() if k.lower() not in ("id", "id_", "id ")}
                    if len(rest) == 1 and "target" in rest:
                        out.append((id_, rest["target"]))
                    elif len(rest) == 1:
                        out.append((id_, list(rest.values())[0]))
                    else:
                        out.append((id_, rest))
            return out
        return []


if __name__ == "__main__":
    import asyncio

    async def _test():
        tool = KaggleSubmitTool()

        # regression
        preds_reg = json.dumps({"a": 1.5, "b": 2.3, "c": 0.1})
        r1 = await tool.call({
            "predictions": preds_reg,
            "output_path": "_test_sub_reg.csv",
            "expected_n_rows": 3,
        })
        print(f"regression: {r1.data['n_rows']} rows, valid={r1.data['validation']['passed']}")
        assert r1.data["validation"]["passed"]
        assert r1.data["task_type"] == "regression"

        # classification
        preds_clf = json.dumps({
            "img1": {"cat": 0.9, "dog": 0.1},
            "img2": {"cat": 0.3, "dog": 0.7},
        })
        r2 = await tool.call({
            "predictions": preds_clf,
            "output_path": "_test_sub_clf.csv",
            "expected_n_rows": 2,
        })
        print(f"classification: {r2.data['n_rows']} rows, cols={r2.data['columns']}")
        assert r2.data["validation"]["passed"]
        assert r2.data["task_type"] == "classification"

        # 故意错行数
        r3 = await tool.call({
            "predictions": preds_reg,
            "output_path": "_test_sub_bad.csv",
            "expected_n_rows": 10,
        })
        print(f"bad rows: valid={r3.data['validation']['passed']}, issues={r3.data['validation']['issues']}")
        assert not r3.data["validation"]["passed"]

        for p in ["_test_sub_reg.csv", "_test_sub_clf.csv", "_test_sub_bad.csv"]:
            Path(p).unlink(missing_ok=True)
        print("[kaggle_tool] self-check OK")

    asyncio.run(_test())
