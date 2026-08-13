"""RCB 纯工具函数 — 不依赖其他 rcb_* 模块."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _detect_gpu_safe() -> bool:
    """检测 GPU 是否可用且 cudnn 不崩溃.

    ponytail: 触发一次小 cudnn op 验证 DLL 完整性. 损坏的 cudnn DLL
    在 Windows 上会导致栈缓冲区溢出 (0xC0000409) 进程崩溃, 比 try→fail→log
    更严重. 升级路径: 按 torch 版本 + cuda 版本 + cudnn 版本做兼容性矩阵.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.randn(8, 8, device="cuda")
        y = x @ x.T
        z = torch.nn.functional.conv2d(
            torch.randn(1, 1, 8, 8, device="cuda"),
            torch.randn(1, 1, 3, 3, device="cuda"),
        )
        del x, y, z
        return True
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return False


# v14 Task 1: Meta-Trace simplicial complex schema helpers
# 把 RCBench workspace 目录名 (带时间戳后缀) 剥成短 task_id, 再推断 domain.
_TASK_ID_TS_RE = re.compile(r"^(.+?)_\d{8}_\d{6}$")
_DOMAIN_KNOWN = {"astronomy", "material", "math"}


def _infer_task_id_from_workspace(ws_name: str) -> str:
    """Astronomy_000_20260720_034353 → Astronomy_000. 老目录名无时间戳则原样返回."""
    m = _TASK_ID_TS_RE.match(ws_name)
    return m.group(1) if m else ws_name


def _infer_domain(task_id: str) -> str:
    """Astronomy_000 → astronomy. 不在白名单返回 unknown, 不抛错."""
    if not task_id:
        return "unknown"
    head = task_id.split("_", 1)[0].lower()
    return head if head in _DOMAIN_KNOWN else "unknown"


def _make_simplex_id(task_id: str, iteration: int, role: str) -> str:
    """trace:{task_id}:iter_{N}:{role} — 同 task 内同 role 同 iter 唯一."""
    return f"trace:{task_id}:iter_{iteration}:{role}"


# 文件重写 stagnation 检测: 扫 code/ 下 *_vN.{py,ipynb,sh} 同 base 名多版本.
_REWRITE_VERSION_RE = re.compile(
    r"^(?P<base>.+?)_v(?P<n>\d+)\.(py|ipynb|sh|r|R)$"
)


def _detect_file_rewrite_stagnation(code_dir: Path) -> tuple[bool, str]:
    """扫 code/ 目录, 若同一 base 名出现 >=3 个版本号, 返回 (True, 提示).

    ponytail: 纯文件名扫描, O(n) 一次 glob. 不读文件内容, 不追 mtime —
    iter 边界调一次, 开销可忽略. 真实 stagnation 还需配合 darwin_score 无提升,
    但 advisory only 先发提示, 不强阻断.
    """
    try:
        if not code_dir.exists():
            return False, ""
        counts: dict[str, list[int]] = {}
        for p in code_dir.glob("*_v*.*"):
            m = _REWRITE_VERSION_RE.match(p.name)
            if not m:
                continue
            base = m.group("base")
            n = int(m.group("n"))
            counts.setdefault(base, []).append(n)
        for base, versions in counts.items():
            if len(versions) >= 3:
                vs = sorted(versions)
                return True, (
                    f"[file_rewrite_stagnation] {base}_v{vs[0]}.py → "
                    f"{base}_v{vs[-1]}.py ({len(vs)} versions). "
                    f"You've rewritten this file {len(vs)} times. "
                    f"STOP refining it. Pivot to a DIFFERENT checklist item "
                    f"(traceback algorithm, symbolic engine, data analysis, "
                    f"ablation — anything not requiring the infeasible component). "
                    f"Persistent rewriting without progress = stagnation."
                )
        return False, ""
    except Exception:
        return False, ""


# metric 白名单 — 抓数值时只保留这些, 避免误抓年份/版本号
_METRIC_WHITELIST = frozenset({
    "mae", "rmse", "mse", "r2", "r²", "r3", "accuracy",
    "precision", "recall", "f1", "auc", "pearson", "spearman",
    "loss", "error", "score", "bias",
})
# regex: metric = value / metric: value / metric of value / metric ≈ value
_NUMERIC_PAIR_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_²³]{1,15})\s*(?:=|:|of|≈|~|is)\s*"
    r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
)


def _extract_numeric_targets(text: str) -> dict[str, float]:
    """从 text 抓 'metric = value' 模式, 返回 {metric: value}.

    metric 名白名单过滤, 避免误抓年份/版本号. 不抓单位.
    """
    if not text:
        return {}
    targets: dict[str, float] = {}
    for m in _NUMERIC_PAIR_RE.finditer(text):
        name = m.group(1).lower()
        try:
            val = float(m.group(2))
        except ValueError:
            logger.debug("best-effort op failed", exc_info=True)
            continue
        if name not in _METRIC_WHITELIST:
            continue
        if abs(val) > 1e6:
            continue
        targets[name] = val
    return targets


def _save_manifold(manifold, path: Path) -> None:
    """manifold 状态持久化到 jsonl. 一行一个 hypothesis. 覆盖写.

    失败静默 — 持久化是 best-effort, 不阻塞主循环.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for _h_id, h in manifold._hyp.items():
                f.write(json.dumps({
                    "type": "hypothesis",
                    "h_id": h.h_id,
                    "description": h.description,
                    "predictions": h.predictions,
                    "n_params": h.n_params,
                }, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("manifold state save skipped", exc_info=True)


def _load_manifold(path: Path):
    """从 jsonl 加载 manifold. 文件不存在或损坏返回 None."""
    from huginn.metacog.hypothesis_manifold import Hypothesis, HypothesisManifold
    if not path.exists():
        return None
    manifold = HypothesisManifold()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") != "hypothesis":
                    continue
                h = Hypothesis(
                    h_id=obj["h_id"],
                    description=obj.get("description", ""),
                    predictions=obj.get("predictions", {}),
                    n_params=int(obj.get("n_params", 0)),
                )
                with contextlib.suppress(ValueError):
                    manifold.add(h)  # duplicate h_id, 跳过
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return None
    return manifold if manifold._hyp else None


# v14 Task 19: model_version 跟踪 — env 没设则 unknown. 进程启动时读一次够.
_MODEL_VERSION = (
    os.environ.get("DEEPSEEK_MODEL_NAME")
    or os.environ.get("OPENAI_MODEL_NAME")
    or "unknown"
)


# v14 Task 15: 跨 task darwin prior — 同 domain 历史 high darwin entry 影响 hint 优先级.
# 模块级 lazy init, 跟 _compute_darwin_score 同样 try/except 防御. 失败留 None,
# 调用方降级空 list. ponytail: SQLite 单文件, 不需要 server. 跨 domain 隔离在
# query_high_darwin(domain=...) 层实现, 这里只持有连接.
_cross_task_store = None
try:
    from huginn.metacog.cross_task_store import CrossTaskStore
    _cross_task_store = CrossTaskStore()
except Exception as _e:
    logger.debug("cross_task_store init skipped: %s", _e)