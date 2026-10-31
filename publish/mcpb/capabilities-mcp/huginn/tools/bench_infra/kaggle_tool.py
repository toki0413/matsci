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

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool


class KaggleSubmitInput(BaseModel):
    predictions: str = Field(
        description="JSON-encoded predictions. Format: "
        "regression -> {id: value, ...} or [[id, value], ...]; "
        "classification -> {id: {class: prob, ...}, ...} or [{id, class, prob}, ...]"
    )
    id_column: str = Field(default="id", description="Column name for IDs")
    class_columns: list[str] | None = Field(
        default=None,
        description="Classification: explicit class column names. If None, infer from data.",
    )
    output_path: str = Field(default="submission/submission.csv")
    expected_n_rows: int | None = Field(
        default=None,
        ge=0,
        description="If set, validate that submission has exactly this many rows",
    )
    task_type: str = Field(
        default="auto",
        description="auto | regression | classification. auto infers from predictions shape.",
    )


def _infer_task_type(input_data, normalized) -> str:
    first_val = normalized[0][1]
    is_classification = isinstance(first_val, dict)
    if input_data.task_type == "auto":
        return "classification" if is_classification else "regression"
    return input_data.task_type


def _infer_class_cols(input_data, normalized) -> list[str]:
    if input_data.class_columns:
        return input_data.class_columns
    # 从数据推断
    class_set = set()
    for _, v in normalized:
        if isinstance(v, dict):
            class_set.update(v.keys())
    return sorted(class_set)


def _write_rows(writer, normalized, task_type: str, class_cols: list[str]) -> int:
    n_rows = 0
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
    return n_rows


def _write_csv(
    out_path: Path,
    header: list[str],
    normalized,
    task_type: str,
    class_cols: list[str],
) -> int:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        return _write_rows(writer, normalized, task_type, class_cols)


def _validate_csv(
    out_path: Path, header: list[str], input_data, n_rows: int
) -> list[str]:
    issues = []
    if input_data.expected_n_rows is not None and n_rows != input_data.expected_n_rows:
        issues.append(f"row count {n_rows} != expected {input_data.expected_n_rows}")

    # 回读校验缺失值
    with open(out_path, encoding="utf-8") as f:
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
    return issues


def _build_result(
    out_path: Path, n_rows: int, header: list[str], task_type: str, issues: list[str]
) -> dict[str, Any]:
    return {
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
            return ToolResult(
                data=None, success=False, error=f"Invalid predictions JSON: {e}"
            )

        # 统一成 list of (id, value_or_dict)
        normalized = self._normalize(preds)
        if not normalized:
            return ToolResult(data=None, success=False, error="Empty predictions")

        # 推断 task_type
        task_type = _infer_task_type(input_data, normalized)

        # 收集 class columns
        if task_type == "classification":
            class_cols = _infer_class_cols(input_data, normalized)
            if not class_cols:
                return ToolResult(
                    data=None,
                    success=False,
                    error="Classification but no class labels found",
                )
        else:
            class_cols = ["target"]

        # 写 CSV
        out_path = Path(input_data.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = [input_data.id_column] + class_cols
        n_rows = _write_csv(out_path, header, normalized, task_type, class_cols)

        # 校验
        issues = _validate_csv(out_path, header, input_data, n_rows)
        result = _build_result(out_path, n_rows, header, task_type, issues)

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
                if isinstance(item, list | tuple) and len(item) == 2:
                    out.append((item[0], item[1]))
                elif isinstance(item, dict):
                    # 从 dict 里取 id 字段
                    id_ = item.get("id", item.get("Id", item.get("ID")))
                    if id_ is None:
                        continue
                    # 剩下的当 class probs 或 target
                    rest = {
                        k: v
                        for k, v in item.items()
                        if k.lower() not in ("id", "id_", "id ")
                    }
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

    async def _test_regression(tool):
        # regression
        preds_reg = json.dumps({"a": 1.5, "b": 2.3, "c": 0.1})
        r1 = await tool.call(
            {
                "predictions": preds_reg,
                "output_path": "_test_sub_reg.csv",
                "expected_n_rows": 3,
            }
        )
        print(
            f"regression: {r1.data['n_rows']} rows, valid={r1.data['validation']['passed']}"
        )
        assert r1.data["validation"]["passed"]
        assert r1.data["task_type"] == "regression"
        return preds_reg

    async def _test_classification(tool):
        # classification
        preds_clf = json.dumps(
            {
                "img1": {"cat": 0.9, "dog": 0.1},
                "img2": {"cat": 0.3, "dog": 0.7},
            }
        )
        r2 = await tool.call(
            {
                "predictions": preds_clf,
                "output_path": "_test_sub_clf.csv",
                "expected_n_rows": 2,
            }
        )
        print(f"classification: {r2.data['n_rows']} rows, cols={r2.data['columns']}")
        assert r2.data["validation"]["passed"]
        assert r2.data["task_type"] == "classification"

    async def _test_bad_rows(tool, preds_reg):
        # 故意错行数
        r3 = await tool.call(
            {
                "predictions": preds_reg,
                "output_path": "_test_sub_bad.csv",
                "expected_n_rows": 10,
            }
        )
        print(
            f"bad rows: valid={r3.data['validation']['passed']}, issues={r3.data['validation']['issues']}"
        )
        assert not r3.data["validation"]["passed"]

    async def _test():
        tool = KaggleSubmitTool()
        preds_reg = await _test_regression(tool)
        await _test_classification(tool)
        await _test_bad_rows(tool, preds_reg)

        for p in ["_test_sub_reg.csv", "_test_sub_clf.csv", "_test_sub_bad.csv"]:
            Path(p).unlink(missing_ok=True)
        print("[kaggle_tool] self-check OK")

    asyncio.run(_test())
