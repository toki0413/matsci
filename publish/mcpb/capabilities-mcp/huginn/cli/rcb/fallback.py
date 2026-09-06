"""RCB report 兜底 + fallback figure 生成.

从 rcb_runner.py 剥离 — 纯文件/数据操作, 不依赖 agent/LLM 状态.
agent 没写 report.md 时强制写, 仍不写就自动生成;
images/ 没图时从 outputs/ 数据生成 fallback figures.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_retry_budget(extra_budget: int | None) -> Any:
    """A4: 构造 Step-3 retry 专用预算. 提到模块级便于 self-check.

    ponytail: recursion_limit ≈ max_calls * 5 (streaming.py:1293 公式).
    返回 None 表示不覆盖 (走 agent 全局 max_tool_calls).
    """
    if not extra_budget or extra_budget <= 0:
        return None
    from huginn.phases import BudgetSpec as _BS  # noqa: N814
    return _BS(
        max_calls=extra_budget,
        recursion_limit=max(250, extra_budget * 5),
    )


async def step2_5_report_fallback(
    ws: Path,
    stream_chat_fn,
) -> None:
    """Step 2.5: report.md 兜底 — agent 没写就强制写, 仍不写就自动生成.

    σ₆ 修复: 减 CSM (σ₃) 后失去 completion guidance, 加 lightweight gate.
    agent 可能在 Step 2 提前终止 (text-only response), 没写 report.md.

    方案 B: 没 figure 时用 outputs/ 数据生成 fallback figures, image
    criterion 至少有图可评 (0→25 分). ponytail: matplotlib Agg backend,
    不依赖 display. 升级: 让 agent 自己生成, 这是兜底.
    """
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        print("\n=== Step 2.5: report.md Emergency Write ===\n", flush=True)
        # ponytail: 独立 thread 隔离 Step 2 TFM fork 可能留下的 dangling
        # tool_calls. 同 Step 3 思路 (commit a30f922). 不隔离 → fork 失败时
        # 主 thread 历史带 dangling → Step 2.5 也 400 → 落回 fallback.
        _step25_tid = f"rcb_{ws.name}_step25"
        await stream_chat_fn(
            "CRITICAL: report/report.md does NOT exist. Session scores ZERO without it.\n"
            "Write report/report.md NOW using file_write_tool. Base it on:\n"
            "- Your Step 1 methodology checklist\n"
            "- Your code in code/ and results in outputs/\n"
            "Minimum: # Title, ## Methodology, ## Results (images/*.png), ## Discussion.\n"
            "Be HONEST. A short honest report beats no report. Write it NOW.",
            "step2.5",
            tid=_step25_tid
        )
    if not report_path.exists():
        print("[fallback: auto-generating minimal report.md]", flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # 方案 B: images/ 没图就用 outputs/ 数据生成 fallback figures.
        _imgs_dir = ws / "report" / "images"
        _imgs_dir.mkdir(parents=True, exist_ok=True)
        _existing_imgs = list(_imgs_dir.glob("*.png")) if _imgs_dir.exists() else []
        if not _existing_imgs:
            _n_gen = generate_fallback_figures(ws, _imgs_dir)
            if _n_gen > 0:
                print(f"[fallback: generated {_n_gen} figures from outputs/]", flush=True)
        _metrics_parts = []
        for _p in (ws / "outputs").glob("*.json"):
            with contextlib.suppress(Exception):
                _metrics_parts.append(f"### {_p.name}\n```json\n{_p.read_text(encoding='utf-8')}\n```")
        _metrics = "\n".join(_metrics_parts) or "None"
        _imgs = "\n".join(f"![{p.name}](images/{p.name})" for p in _imgs_dir.glob("*.png")) or "None"
        _code_dir = ws / "code"
        _code = "\n".join(f"- `{p.name}`" for p in _code_dir.glob("*.py")) or "None" if _code_dir.exists() else "None"
        report_path.write_text(
            f"# Research Report (Auto-generated Fallback)\n\n"
            f"## Methodology\nAgent did not write report.md; auto-generated from artifacts.\n\n"
            f"### Code\n{_code}\n\n### Metrics\n{_metrics}\n\n## Results\n{_imgs}\n",
            encoding="utf-8"
        )


def generate_fallback_figures(ws: Path, imgs_dir: Path) -> int:
    """从 outputs/ 数据文件生成 fallback figures. 返回生成数量.

    ponytail: matplotlib Agg backend 不依赖 display. 每种文件类型生成一张图:
    - .json (metrics dict) → bar chart
    - .npy 1D → histogram; 2D → scatter (前两列)
    - .csv → line plot (前两列)
    - .npy scalar / empty → skip
    失败静默, 不阻塞 report 生成. 升级: 让 agent 自己生成, 这是兜底.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.debug("matplotlib not available, fallback figures skipped", exc_info=True)
        return 0
    # 字体统一 Arial 20pt+ 加粗 (用户规则)
    try:
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.weight"] = "bold"
    except Exception:
        logger.debug("matplotlib rcParams setup skipped", exc_info=True)

    n_gen = 0
    outputs_dir = ws / "outputs"
    if not outputs_dir.exists():
        return 0

    # .json metrics → bar chart
    for jp in outputs_dir.glob("*.json"):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
            if not isinstance(d, dict) or not d:
                continue
            # 只取数值字段, 跳过非数值
            numeric = {k: float(v) for k, v in d.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if not numeric:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(list(numeric.keys())[:15], list(numeric.values())[:15])
            ax.set_title(f"Metrics: {jp.stem}", fontsize=22, fontweight="bold")
            ax.set_ylabel("Value", fontsize=20, fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fp = imgs_dir / f"fallback_{jp.stem}_metrics.png"
            plt.savefig(fp, dpi=120, bbox_inches="tight")
            plt.close(fig)
            n_gen += 1
            if n_gen >= 4:
                return n_gen
        except Exception:
            logger.debug("fallback json figure skipped", exc_info=True)
            continue

    # .npy → histogram (1D) / scatter (2D) / bar chart (0-dim dict)
    try:
        import numpy as np
        for np_p in outputs_dir.glob("*.npy"):
            try:
                arr = np.load(np_p, allow_pickle=True)
                if arr.size == 0:
                    continue
                # 0-dim object array 通常是 dict (np.save(scalar_dict)) —
                # 当 metrics bar chart 处理. ponytail: 不区分 dict / scalar,
                # 是 dict 就画, 不是就 skip.
                if arr.ndim == 0 and arr.dtype == object:
                    obj = arr.item()
                    if not isinstance(obj, dict):
                        continue
                    numeric = {k: float(v) for k, v in obj.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    if not numeric:
                        continue
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(list(numeric.keys())[:15], list(numeric.values())[:15])
                    ax.set_title(f"Metrics: {np_p.stem}", fontsize=22, fontweight="bold")
                    ax.set_ylabel("Value", fontsize=20, fontweight="bold")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    fp = imgs_dir / f"fallback_{np_p.stem}_metrics.png"
                    plt.savefig(fp, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    n_gen += 1
                elif arr.dtype == object:
                    continue
                elif arr.ndim == 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(arr, bins=30)
                    ax.set_title(f"Distribution: {np_p.stem}", fontsize=22, fontweight="bold")
                    ax.set_xlabel("Value", fontsize=20, fontweight="bold")
                    ax.set_ylabel("Count", fontsize=20, fontweight="bold")
                    plt.tight_layout()
                    fp = imgs_dir / f"fallback_{np_p.stem}_hist.png"
                    plt.savefig(fp, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    n_gen += 1
                elif arr.ndim == 2 and arr.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(arr[:, 0], arr[:, 1], s=20, alpha=0.6)
                    ax.set_title(f"Scatter: {np_p.stem}", fontsize=22, fontweight="bold")
                    ax.set_xlabel("col 0", fontsize=20, fontweight="bold")
                    ax.set_ylabel("col 1", fontsize=20, fontweight="bold")
                    plt.tight_layout()
                    fp = imgs_dir / f"fallback_{np_p.stem}_scatter.png"
                    plt.savefig(fp, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    n_gen += 1
                if n_gen >= 4:
                    return n_gen
            except Exception:
                logger.debug("fallback npy figure skipped", exc_info=True)
                continue
    except ImportError:
        logger.debug("numpy not available, npy figures skipped", exc_info=True)

    # .csv → line plot (找第一对都是数值的列, 跳过 SMILES 等文本列)
    import csv
    for cp in outputs_dir.glob("*.csv"):
        try:
            with cp.open(encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if len(rows) < 2:
                continue
            header = rows[0]
            # 找第一对都是数值的列对 (col_x, col_y). ponytail: O(cols^2 * rows)
            # 但 cols/rows 都小, 零开销. 升级: pandas 自动 dtype 推断.
            _x_idx = _y_idx = -1
            for _xi in range(min(len(header), 8)):
                for _yi in range(_xi + 1, min(len(header), 8)):
                    _ok = 0
                    _total_check = 0
                    for r in rows[1:20]:
                        if len(r) <= _yi:
                            continue
                        _total_check += 1
                        try:
                            float(r[_xi])
                            float(r[_yi])
                            _ok += 1
                        except (ValueError, TypeError):
                            pass
                    if _total_check > 0 and _ok == _total_check:
                        _x_idx, _y_idx = _xi, _yi
                        break
                if _x_idx >= 0:
                    break
            if _x_idx < 0:
                continue
            xs, ys = [], []
            for r in rows[1:]:
                if len(r) <= _y_idx:
                    continue
                try:
                    xs.append(float(r[_x_idx]))
                    ys.append(float(r[_y_idx]))
                except (ValueError, TypeError):
                    continue
            if len(xs) < 2:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(xs, ys, s=15, alpha=0.5)
            ax.set_title(f"Scatter: {cp.stem}", fontsize=22, fontweight="bold")
            ax.set_xlabel(header[_x_idx] if _x_idx < len(header) else f"col {_x_idx}", fontsize=20, fontweight="bold")
            ax.set_ylabel(header[_y_idx] if _y_idx < len(header) else f"col {_y_idx}", fontsize=20, fontweight="bold")
            plt.tight_layout()
            fp = imgs_dir / f"fallback_{cp.stem}_scatter.png"
            plt.savefig(fp, dpi=120, bbox_inches="tight")
            plt.close(fig)
            n_gen += 1
            if n_gen >= 4:
                return n_gen
        except Exception:
            logger.debug("fallback csv figure skipped", exc_info=True)
            continue

    return n_gen
