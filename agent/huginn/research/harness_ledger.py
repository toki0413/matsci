"""Self-Harness 组织层治理账本 — M3: 跨 run 按 task_episode 聚合.

输入: 已有 run 产出的 harness dict 或 HarnessReport (M1/M2 已落地). 本模块**不重复实现**
评分逻辑, 只消费已产出的 harness dict——把多条 run 按 task_episode 聚合成一条「治理账本行」,
供组织层查询「同一需求的多次实录」.

红线(spec 对齐): 「配置存在 ≠ 能力可用」. 这里聚合绝不补分/不平滑, 如实呈现每条实录的分值;
同时把「靠没跑的门禁凑分」的维度诚实暴露出来——summary 对每个维度标注 evidence 三态
(observed/unobserved/missing) 的占比, observed_ratio 低 = 该维大部分 run 是从「配置存在但
没跑到」凑出来的.

纯 stdlib: json / pathlib / tempfile / os / hashlib / collections / dataclasses / statistics.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

# 五维固定顺序(与 harness.py 的 dimensions 顺序一致), 保证聚合输出可解释
_DIMENSIONS = (
    "task_understanding",
    "controlled_execution",
    "change_validation",
    "reliable_delivery",
    "learning_capture",
)
_EVIDENCE_BITS = ("observed", "unobserved", "missing")


@dataclass
class HarnessLedger:
    """按 task_episode 索引的治理账本.

    entries 保留 append 的追加顺序(即 run 的先后序), 供「同一需求的多次实录」按序号追溯;
    skipped 记录因字段缺失/格式异常而被跳过(不抛)的条数(fail-open).
    """

    entries: list[dict] = field(default_factory=list)
    skipped: int = 0

    # ── 追加 ───────────────────────────────────────────────────────────────
    def append(self, harness: dict | Any) -> HarnessLedger:
        """把一条 out.harness(或 HarnessReport) 追加进账本(按 task_episode 索引).

        - 传入 ``HarnessReport`` 时自动取其 ``to_dict()``;
        - 传入 dict 时原样消费(已是 to_dict() 产物);
        - fail-open: 非 dict / 缺 task_episode / dimensions 非合法的 list → 跳过并计数 skipped,
          绝不抛异常打断读书流程。
        """
        raw = harness
        if not isinstance(raw, dict):
            to_dict = getattr(raw, "to_dict", None)
            if callable(to_dict):
                try:
                    raw = to_dict()
                except Exception:  # noqa: BLE001 — 报告序列化失败, 跳过此条不抛
                    self.skipped += 1
                    return self
        if not isinstance(raw, dict):
            self.skipped += 1
            return self
        ep = raw.get("task_episode")
        dims = raw.get("dimensions")
        if not isinstance(ep, str) or not ep:
            self.skipped += 1
            return self
        if not isinstance(dims, list):
            self.skipped += 1
            return self
        self.entries.append(raw)
        return self

    # ── 持久化(原子写) ─────────────────────────────────────────────────────
    def persist(self, path: str | Path) -> Path:
        """把当前账本写成 JSONL(逐行一条 harness), 走临时文件 + 替换保证原子性.

        entries 里任一条无法 JSON 序列化时按 fail-open 跳过并计数 skipped(不抛)。
        典型用法: load → append → persist, 文件始终是「已消费 harness 的完整账本」。
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for e in self.entries:
            try:
                lines.append(json.dumps(e, ensure_ascii=False))
            except (TypeError, ValueError):  # 无法序列化 → 跳过此条, 不破坏整本
                self.skipped += 1
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            os.replace(tmp, p)
        except Exception:  # noqa: BLE001 — 写盘失败: 清理临时文件后重抛(镜像原文件完好)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return p

    @classmethod
    def load(cls, path: str | Path) -> HarnessLedger:
        """从 JSONL 读回一个账本. 文件不存在 → 空账本; 损坏行按 fail-open 跳过计数."""
        ledger = cls()
        p = Path(path)
        if not p.exists():
            return ledger
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ledger.append(json.loads(line))
                except Exception:  # noqa: BLE001 — 单行 JSON 损坏, 跳过不抛
                    ledger.skipped += 1
        return ledger

    # ── 统计辅助 ───────────────────────────────────────────────────────────
    @staticmethod
    def _metric(e: dict, key: str) -> Any:
        """取一条 harness 的「按 key 聚合的数值」: key=overall 取顶层; 否则视为维度名."""
        if key == "overall":
            return e.get("overall")
        for d in e.get("dimensions", []):
            if isinstance(d, dict) and d.get("name") == key:
                return d.get("score")
        return None

    @staticmethod
    def _stats(values: list[float]) -> dict[str, Any]:
        """对一组数值求 mean/min/max/count; 空列表四项全为 None/0(count)."""
        count = len(values)
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": count,
            "mean": round(fmean(values), 3),
            "min": min(values),
            "max": max(values),
        }

    # ── 聚合查询 ───────────────────────────────────────────────────────────
    def summary(self, task_episode: str | None = None, *, by: str = "overall") -> dict[str, Any]:
        """跨 run 聚合: 同一 task_episode 的多条实录合成一条治理账本行.

        - ``task_episode=None`` → 返回 {ep: 该 episode 的聚合} (列出所有 episode);
        - 给定 episode → 该 episode 的聚合:
            * ``<by>``(默认 overall) 的 mean/min/max/count;
            * 每维平均分 score_mean;
            * 每维 evidence 三态(observed/unobserved/missing) 的分布与占比。

        红线: 聚合只如实呈现, 不补分/不平滑; observed_ratio 低的维度即「靠没跑的门禁凑分」,
        在此被诚实暴露(heads-up), 不替 run 清洗高分。
        """
        if task_episode is None:
            seen: list[str] = []
            for e in self.entries:
                ep = e.get("task_episode")
                if isinstance(ep, str) and ep and ep not in seen:
                    seen.append(ep)
            return {ep: self.summary(ep, by=by) for ep in seen}

        rows = [e for e in self.entries if e.get("task_episode") == task_episode]
        # 只对数值字段聚合(fail-open: 缺这个字段/非数值的 run 不计入, 也让 count 暴露了覆盖率)
        metric = self._stats([v for e in rows
                              if isinstance(v := self._metric(e, by), (int, float))])

        dims: dict[str, dict[str, Any]] = {}
        for dim in _DIMENSIONS:
            scores: list[float] = []
            evidence: Counter = Counter()
            for e in rows:
                for d in e.get("dimensions", []):
                    if not isinstance(d, dict) or d.get("name") != dim:
                        continue
                    s = d.get("score")
                    if isinstance(s, (int, float)):
                        scores.append(s)
                    ev = d.get("evidence")
                    if ev in _EVIDENCE_BITS:
                        evidence[ev] += 1
            total = sum(evidence.values())
            dims[dim] = {
                "score_mean": round(fmean(scores), 3) if scores else None,
                "evidence": {
                    "observed": evidence.get("observed", 0),
                    "unobserved": evidence.get("unobserved", 0),
                    "missing": evidence.get("missing", 0),
                    **{
                        f"{bit}_ratio": (round(evidence.get(bit, 0) / total, 3) if total else None)
                        for bit in _EVIDENCE_BITS
                    },
                },
            }

        return {
            "task_episode": task_episode,
            "runs": len(rows),
            by: metric,
            "dimensions": dims,
        }

    def cross_run(self, task_episode: str) -> list[dict[str, Any]]:
        """列出该 episode 下所有 run(按 append 序/序号), 供 reading 追溯「多次实录」."""
        return [
            {**e, "run": i}
            for i, e in enumerate(self.entries)
            if e.get("task_episode") == task_episode
        ]
