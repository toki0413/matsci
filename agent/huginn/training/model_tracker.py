"""v14 Task 19: model_version 跟踪 + coevolution_ready 触发提示.

跟 darwin_exporter 配套:
  - trace entry 加 model_version 字段 (从 DEEPSEEK_MODEL_NAME / OPENAI_MODEL_NAME env 读)
  - 每周 (按调用时机) 生成 per-model-version 统计报告
  - 训练池累计 ≥1000 SFT OR ≥500 DPO 时日志输出 coevolution_ready

ponytail: 简单统计 + 文件计数, 不上 ML. 升级路径: 真正的 model selection /
cross-validation. 不自动触发 fine-tune, 只做数据池累积 + 触发建议.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class ModelVersionTracker:
    """跟踪 model version 跟 darwin_score 的相关性.

    每次调用生成 per-model-version 统计报告.
    训练池累计 ≥1000 SFT OR ≥500 DPO 时 check_coevolution_ready 返回 True.

    ponytail: 简单统计, 不上 ML. 升级路径: 真正的 model selection.
    """

    SFT_READY_THRESHOLD = 1000
    DPO_READY_THRESHOLD = 500

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = Path(os.environ.get("HUGINN_CACHE_DIR") or (Path.home() / ".huginn"))
        self.cache_dir = Path(cache_dir)
        self.training_pool_dir = self.cache_dir / "training_pool"

    def generate_report(self, trace_entries: list) -> dict:
        """生成 per-model-version 统计报告.

        按 model_version 分组, 算每组 avg_darwin_score / avg_supported_ratio
        / beta_0_distribution / beta_1_distribution.
        """
        by_version: dict[str, list] = defaultdict(list)
        for entry in trace_entries:
            version = entry.get("model_version", "unknown") or "unknown"
            by_version[version].append(entry)

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(trace_entries),
            "per_version": {},
        }
        for version, entries in by_version.items():
            darwin_scores = [
                e.get("darwin_score", 0.0) for e in entries
                if isinstance(e.get("darwin_score"), (int, float))
            ]
            supported_ratios = [
                e.get("supported_ratio", 0.0) for e in entries
                if isinstance(e.get("supported_ratio"), (int, float))
            ]
            betti_0_list = [
                e.get("beta_0", 1) for e in entries
                if isinstance(e.get("beta_0"), (int, float))
            ]
            betti_1_list = [
                e.get("beta_1", 0) for e in entries
                if isinstance(e.get("beta_1"), (int, float))
            ]

            report["per_version"][version] = {
                "n_entries": len(entries),
                "avg_darwin_score": sum(darwin_scores) / max(len(darwin_scores), 1),
                "avg_supported_ratio": sum(supported_ratios) / max(len(supported_ratios), 1),
                "beta_0_distribution": {
                    "min": min(betti_0_list) if betti_0_list else 0,
                    "max": max(betti_0_list) if betti_0_list else 0,
                    "avg": sum(betti_0_list) / max(len(betti_0_list), 1),
                },
                "beta_1_distribution": {
                    "min": min(betti_1_list) if betti_1_list else 0,
                    "max": max(betti_1_list) if betti_1_list else 0,
                    "avg": sum(betti_1_list) / max(len(betti_1_list), 1),
                },
            }
        return report

    def save_report(self, report: dict) -> Path:
        """保存报告到 cache_dir/coevolution_report_<date>.json. 返回路径."""
        date_str = datetime.now().strftime("%Y%m%d")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.cache_dir / f"coevolution_report_{date_str}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report_path

    def check_coevolution_ready(self) -> tuple[bool, int, int]:
        """检查训练池是否达 1000 SFT 或 500 DPO. 返回 (ready, n_sft, n_dpo).

        ponytail: 按文件名 sft/failure/dpo 分类计数, 不读文件内容.
        天花板: 文件名不规范的样本会被漏掉. 升级路径: 维护 manifest.json 索引.
        """
        n_sft = 0
        n_dpo = 0
        if self.training_pool_dir.exists():
            for f in self.training_pool_dir.rglob("*.jsonl"):
                name = f.name.lower()
                if "sft" in name or "failure" in name:
                    with f.open(encoding="utf-8") as fp:
                        n_sft += sum(1 for _ in fp)
                elif "dpo" in name:
                    with f.open(encoding="utf-8") as fp:
                        n_dpo += sum(1 for _ in fp)
        ready = n_sft >= self.SFT_READY_THRESHOLD or n_dpo >= self.DPO_READY_THRESHOLD
        return ready, n_sft, n_dpo


if __name__ == "__main__":
    # v14 Task 19 self-check — 不引框架, 全用 assert.
    import shutil
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="model_tracker_selfcheck_"))
    try:
        tracker = ModelVersionTracker(cache_dir=tmp_root)

        # case 1: 3 model_version × 10 entry → generate_report 返回 3 version
        entries = []
        for v_idx, version in enumerate(["deepseek-chat", "gpt-4o", "unknown"]):
            for i in range(10):
                entries.append({
                    "model_version": version,
                    "darwin_score": 0.5 + 0.05 * (v_idx + i % 3),
                    "supported_ratio": 0.3 + 0.03 * (i % 5),
                    "beta_0": 1 + (i % 3),
                    "beta_1": i % 4,
                })
        report = tracker.generate_report(entries)
        assert report["total_entries"] == 30, f"total_entries: {report['total_entries']}"
        assert len(report["per_version"]) == 3, \
            f"per_version count: {len(report['per_version'])}"
        for version in ["deepseek-chat", "gpt-4o", "unknown"]:
            assert version in report["per_version"], f"missing version: {version}"
            stats = report["per_version"][version]
            assert stats["n_entries"] == 10, f"{version} n_entries: {stats['n_entries']}"
            for key in ("avg_darwin_score", "avg_supported_ratio",
                        "beta_0_distribution", "beta_1_distribution"):
                assert key in stats, f"{version} missing {key}"
            for dist_key in ("beta_0_distribution", "beta_1_distribution"):
                for sub in ("min", "max", "avg"):
                    assert sub in stats[dist_key], f"{version} {dist_key} missing {sub}"

        # case 2: save_report 在 cache_dir 下生成 coevolution_report_<date>.json
        report_path = tracker.save_report(report)
        assert report_path.exists(), f"report file not created: {report_path}"
        assert report_path.name.startswith("coevolution_report_"), \
            f"report filename: {report_path.name}"
        assert report_path.name.endswith(".json"), f"report ext: {report_path.name}"
        with report_path.open(encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["total_entries"] == 30, "reloaded report mismatch"
        assert len(parsed["per_version"]) == 3, "reloaded per_version mismatch"

        # case 3: 1001 SFT 样本 → check_coevolution_ready 返回 (True, 1001, 0)
        sft_dir = tracker.training_pool_dir / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        sft_file = sft_dir / "sft_test_001.jsonl"
        with sft_file.open("w", encoding="utf-8") as f:
            for i in range(1001):
                f.write(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) + "\n")
        ready, n_sft, n_dpo = tracker.check_coevolution_ready()
        assert ready is True, f"1001 sft should be ready: ready={ready}"
        assert n_sft == 1001, f"n_sft: expected 1001, got {n_sft}"
        assert n_dpo == 0, f"n_dpo: expected 0, got {n_dpo}"

        # case 4: 500 DPO 样本 → check_coevolution_ready 返回 (True, *, 500)
        # 先删 sft 文件, 只剩 dpo 池
        sft_file.unlink()
        dpo_dir = tracker.training_pool_dir / "dpo"
        dpo_dir.mkdir(parents=True, exist_ok=True)
        dpo_file = dpo_dir / "dpo_test_001.jsonl"
        with dpo_file.open("w", encoding="utf-8") as f:
            for i in range(500):
                f.write(json.dumps({"prompt": f"q{i}", "chosen": "a", "rejected": "b"}) + "\n")
        ready, n_sft, n_dpo = tracker.check_coevolution_ready()
        assert ready is True, f"500 dpo should be ready: ready={ready}"
        assert n_dpo == 500, f"n_dpo: expected 500, got {n_dpo}"
        assert n_sft == 0, f"n_sft: expected 0, got {n_sft}"

        # case 5: 空训练池 (删 dpo 文件)
        dpo_file.unlink()
        ready, n_sft, n_dpo = tracker.check_coevolution_ready()
        assert ready is False, f"empty pool should not be ready: ready={ready}"
        assert n_sft == 0 and n_dpo == 0, \
            f"empty pool counts: n_sft={n_sft}, n_dpo={n_dpo}"

        # case 6 (附加): failure_trace 文件名也算 SFT
        fail_file = tracker.training_pool_dir / "failure_Astronomy_000.jsonl"
        with fail_file.open("w", encoding="utf-8") as f:
            for i in range(1001):
                f.write(json.dumps({"prompt": f"q{i}", "completion": f"bad{i}"}) + "\n")
        ready, n_sft, n_dpo = tracker.check_coevolution_ready()
        assert ready is True, f"1001 failure samples should be ready: ready={ready}"
        assert n_sft == 1001, f"failure n_sft: expected 1001, got {n_sft}"

        print("v14 Task 19 self-check PASSED")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
