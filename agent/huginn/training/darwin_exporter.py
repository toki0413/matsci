"""v14 Task 17: darwin_score 反向传播到 model fine-tune 的训练数据导出.

env HUGINN_COEVOLUTION=1 时 RCBench task 完成后自动调用:
  - darwin_score >= 0.8 的 entry -> SFT 样本 (prompt=attempted, completion=found)
  - 同 attempted 差距 >= 0.5 的 pair -> DPO pair (chosen=高 darwin, rejected=低 darwin)

ponytail: 导出格式简单 JSON Lines, 不写 arrow/parquet. 升级路径: HF datasets.
天花板: 同秒内重复 export 会覆盖前一份 (filename ts 秒级精度); 跨秒调用会生成
新版本文件, 不视为重复写. 升级路径: 用 simplex_id 做内容哈希去重.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class DarwinRewardExporter:
    """把 trace entry 按 darwin_score 导出为 SFT/DPO 训练数据.

    env HUGINN_COEVOLUTION=1 时 RCBench task 完成后自动调用.

    ponytail: 导出格式简单 JSON, 不写arrow/parquet. 升级路径: HF datasets.
    """

    SFT_THRESHOLD = 0.8
    DPO_GAP_THRESHOLD = 0.5

    def __init__(self, output_dir: Path | None = None):
        if output_dir is None:
            cache_dir = Path(os.environ.get("HUGINN_CACHE_DIR") or (Path.home() / ".huginn"))
            output_dir = cache_dir / "training_pool"
        self.output_dir = Path(output_dir)
        self.sft_dir = self.output_dir / "sft"
        self.dpo_dir = self.output_dir / "dpo"
        self.sft_dir.mkdir(parents=True, exist_ok=True)
        self.dpo_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_task_id(task_id: str | None) -> str:
        tid = (task_id or "task").replace("/", "_").replace("\\", "_")
        return tid or "task"

    def export_sft(self, trace_entries: list, task_id: str | None = None) -> int:
        """导出 darwin_score >= 0.8 的 entry 为 SFT 样本. 返回导出数.

        每个样本一行 JSON: {prompt, completion, darwin_score, domain, task_id, simplex_id}.
        写入 sft_dir/sft_{task_id}_{ts}.jsonl.
        """
        samples: list[dict] = []
        for entry in trace_entries:
            try:
                darwin = float(entry.get("darwin_score") or 0.0)
            except (TypeError, ValueError):
                continue
            if darwin < self.SFT_THRESHOLD:
                continue
            samples.append({
                "prompt": entry.get("attempted") or "",
                "completion": entry.get("found") or "",
                "darwin_score": darwin,
                "domain": entry.get("domain") or "unknown",
                "task_id": entry.get("task_id") or task_id or "unknown",
                "simplex_id": entry.get("simplex_id"),
            })
        if not samples:
            return 0
        ts = int(time.time())
        out_path = self.sft_dir / f"sft_{self._safe_task_id(task_id)}_{ts}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return len(samples)

    def export_dpo(self, trace_entries: list, task_id: str | None = None) -> int:
        """导出 darwin 差距 >= 0.5 的 entry pair 为 DPO pair. 返回导出对数.

        对每个 (task_id, attempted) 分组, 取组内 darwin 最高 vs 最低,
        差距 >= 0.5 时导出为 chosen/rejected pair.
        写入 dpo_dir/dpo_{task_id}_{ts}.jsonl.
        """
        # 按 (task_id, attempted) 分组 — 同 attempted 才能做 chosen/rejected 对照
        by_key: dict[tuple[str, str], list[dict]] = {}
        for entry in trace_entries:
            attempted = entry.get("attempted") or ""
            if not attempted:
                continue
            e_task = entry.get("task_id") or task_id or "unknown"
            by_key.setdefault((e_task, attempted), []).append(entry)

        pairs: list[dict] = []
        for (e_task, attempted), entries in by_key.items():
            if len(entries) < 2:
                continue
            try:
                sorted_entries = sorted(
                    entries,
                    key=lambda e: float(e.get("darwin_score") or 0.0),
                    reverse=True,
                )
            except (TypeError, ValueError):
                continue
            high = sorted_entries[0]
            low = sorted_entries[-1]
            try:
                d_high = float(high.get("darwin_score") or 0.0)
                d_low = float(low.get("darwin_score") or 0.0)
            except (TypeError, ValueError):
                continue
            if d_high - d_low < self.DPO_GAP_THRESHOLD:
                continue
            # 同一条 entry 不当 pair (高低调换后 darwin 相同时跳过)
            if high is low:
                continue
            pairs.append({
                "prompt": attempted,
                "chosen": high.get("found") or "",
                "rejected": low.get("found") or "",
                "chosen_darwin": d_high,
                "rejected_darwin": d_low,
                "domain": high.get("domain") or "unknown",
                "task_id": e_task,
            })
        if not pairs:
            return 0
        ts = int(time.time())
        out_path = self.dpo_dir / f"dpo_{self._safe_task_id(task_id)}_{ts}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        return len(pairs)

    def export_failure_trace(self, trace_entries: list, task_id: str, task_score: float) -> int:
        """导出失败 task 的所有 trace entry 为 negative sample. 返回导出数.

        spec §"失败 trace 进训练池": task score < 20 时所有 trace entry 标 failure_trace=true.
        这些 entry 在 SFT 时作为 negative sample (completion 是错误答案).

        ponytail: 复用 sft_dir, 文件名加 failure_ 前缀. 升级路径: 独立 failure_pool/.
        """
        # spec 硬阈值: score < 20 才导出
        if task_score >= 20:
            return 0
        out_path = self.sft_dir / f"failure_{self._safe_task_id(task_id)}_{int(time.time())}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for entry in trace_entries:
                sample = {
                    "prompt": entry.get("attempted") or "",
                    "completion": entry.get("found") or "",
                    "darwin_score": entry.get("darwin_score") or 0.0,
                    "domain": entry.get("domain") or "unknown",
                    "task_id": task_id,
                    "simplex_id": entry.get("simplex_id") or "",
                    "failure_trace": True,
                    "negative": True,
                    "task_score": task_score,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
        return count


if __name__ == "__main__":
    # v14 Task 17 self-check — 不引框架, 全用 assert.
    import shutil
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="darwin_exporter_selfcheck_"))
    try:
        exporter = DarwinRewardExporter(output_dir=tmp_root)

        # 构造 10 个 entry:
        #  - 3 个 darwin_score >= 0.8 (0.9, 0.85, 0.8) — 应进 SFT
        #  - 3 个 darwin_score 0.4-0.6 (0.5, 0.45, 0.4) — 不进 SFT, 不进 DPO
        #  - 4 个 darwin_score < 0.3 (0.2, 0.1, 0.15, 0.05) — 不进 SFT
        #  - attempted="compute orbital elements" 的 2 个: darwin=0.9 / 0.3, 差距 0.6 >= 0.5
        #    (0.3 那条单独加, 不属于上面的 0.4-0.6 桶)
        entries = [
            {"attempted": "compute orbital elements", "found": "keplerian elements derived",
             "darwin_score": 0.9, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:0"},
            {"attempted": "estimate stellar mass", "found": "M = 1.2 solar mass",
             "darwin_score": 0.85, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:1"},
            {"attempted": "compute galaxy redshift", "found": "z = 0.023",
             "darwin_score": 0.8, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:2"},
            {"attempted": "fit light curve", "found": "phase folded",
             "darwin_score": 0.5, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:3"},
            {"attempted": "estimate period", "found": "P ~ 2.3 d",
             "darwin_score": 0.45, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:4"},
            {"attempted": "check transit signature", "found": "noise dominated",
             "darwin_score": 0.4, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:5"},
            {"attempted": "spectral classification", "found": "G2V",
             "darwin_score": 0.2, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:6"},
            {"attempted": "estimate age", "found": "~4.3 Gyr",
             "darwin_score": 0.1, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:7"},
            {"attempted": "compute distance modulus", "found": "m-M = 5.2",
             "darwin_score": 0.15, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:8"},
            {"attempted": "compute orbital elements", "found": "wrong elements",
             "darwin_score": 0.3, "domain": "astronomy", "task_id": "Astronomy_000",
             "simplex_id": "trace:9"},
        ]
        assert len(entries) == 10, f"fixture count mismatch: {len(entries)}"

        # case 6: 空 trace_entries 返回 0
        assert exporter.export_sft([], task_id="empty") == 0, "empty sft should be 0"
        assert exporter.export_dpo([], task_id="empty") == 0, "empty dpo should be 0"

        # case 1: export_sft 应返回 3 (0.9, 0.85, 0.8)
        n_sft = exporter.export_sft(entries, task_id="Astronomy_000")
        assert n_sft == 3, f"sft count: expected 3, got {n_sft}"

        # case 2: sft_dir 下有 jsonl, 每行一个 JSON, 含 prompt/completion/darwin_score
        sft_files = list(exporter.sft_dir.glob("sft_Astronomy_000_*.jsonl"))
        assert sft_files, "sft file not created"
        sft_samples = []
        with sft_files[0].open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    assert "prompt" in s and "completion" in s and "darwin_score" in s, \
                        f"sft sample missing fields: {s}"
                    assert s["darwin_score"] >= 0.8, f"sft sample below threshold: {s}"
                    sft_samples.append(s)
        assert len(sft_samples) == 3, f"sft samples in file: expected 3, got {len(sft_samples)}"

        # case 3: export_dpo 应返回 >=1 (0.9 vs 0.3 差距 0.6 >= 0.5)
        n_dpo = exporter.export_dpo(entries, task_id="Astronomy_000")
        assert n_dpo >= 1, f"dpo pair count: expected >=1, got {n_dpo}"

        # case 4: dpo_dir 下有 jsonl, 含 prompt/chosen/rejected
        dpo_files = list(exporter.dpo_dir.glob("dpo_Astronomy_000_*.jsonl"))
        assert dpo_files, "dpo file not created"
        dpo_pairs = []
        with dpo_files[0].open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    assert "prompt" in p and "chosen" in p and "rejected" in p, \
                        f"dpo pair missing fields: {p}"
                    assert p["chosen"] == "keplerian elements derived", \
                        f"chosen should be high-darwin found, got {p['chosen']}"
                    assert p["rejected"] == "wrong elements", \
                        f"rejected should be low-darwin found, got {p['rejected']}"
                    dpo_pairs.append(p)
        assert len(dpo_pairs) >= 1, f"dpo pairs in file: expected >=1, got {len(dpo_pairs)}"

        # case 5: 重复 export 不重复写 — 同 ts (秒级) 覆盖, 文件数不增
        files_before = list(exporter.sft_dir.glob("sft_Astronomy_000_*.jsonl"))
        # 立即再调一次, 同秒覆盖
        n_sft_again = exporter.export_sft(entries, task_id="Astronomy_000")
        assert n_sft_again == 3, f"repeat sft count: expected 3, got {n_sft_again}"
        files_after = list(exporter.sft_dir.glob("sft_Astronomy_000_*.jsonl"))
        # 容差 1: 允许跨秒新生成一份, 但不应无限增长
        assert len(files_after) - len(files_before) <= 1, \
            f"repeat export created extra files: before={len(files_before)} after={len(files_after)}"

        # case 7: 失败 task (score=15) — 5 个 entry 全部导出为 negative sample
        _fail_entries = entries[:5]
        _fail_files_before = list(exporter.sft_dir.glob("failure_test_task_*.jsonl"))
        n_failure = exporter.export_failure_trace(_fail_entries, "test_task", 15)
        assert n_failure == 5, f"failure count: expected 5, got {n_failure}"
        _fail_files_after = list(exporter.sft_dir.glob("failure_test_task_*.jsonl"))
        assert len(_fail_files_after) - len(_fail_files_before) >= 1, "failure file not created"
        with _fail_files_after[-1].open(encoding="utf-8") as _f:
            _fail_samples = [json.loads(_l) for _l in _f if _l.strip()]
        assert len(_fail_samples) == 5, \
            f"failure samples in file: expected 5, got {len(_fail_samples)}"
        for _s in _fail_samples:
            assert _s.get("failure_trace") is True, f"failure_trace flag missing: {_s}"
            assert _s.get("negative") is True, f"negative flag missing: {_s}"
            assert _s.get("task_score") == 15, f"task_score mismatch: {_s}"

        # case 8: 成功 task (score=25) — 不应导出
        _fail_files_before_c8 = list(exporter.sft_dir.glob("failure_test_task_*.jsonl"))
        n_failure_high = exporter.export_failure_trace(_fail_entries, "test_task", 25)
        assert n_failure_high == 0, f"high score should not export: got {n_failure_high}"
        _fail_files_after_c8 = list(exporter.sft_dir.glob("failure_test_task_*.jsonl"))
        assert len(_fail_files_after_c8) == len(_fail_files_before_c8), \
            "high score should not create failure file"

        print("v14 Task 17 self-check PASSED")
        print("v14 Task 18 self-check PASSED")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
