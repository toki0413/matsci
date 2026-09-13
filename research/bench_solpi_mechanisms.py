"""bench_solpi_mechanisms —— 四件套 token 节省的可复现 benchmark。

统计口径: 用 ``rough_token_count_for_text``(≈chars/4) 估算 token。它衡量的是
"少放多少 token 进上下文", 即 SoL-Pi 的受约束效率主张。诚实边界:
  - 这是**机制级**节流测算, 不是整轮 agent 会话的端到端成本(那样需要真 LLM +
    定价表, 且依赖具体长轨迹)。
  - ① Action Fusion 节省的是"少一条验证工具的往返 prompt", 用一个可配置的
    ``avg_validation_turn_tokens`` 估算, 标注为估算值。
  - ④ 报告"可折叠步数/保留步数"的计数, 不虚报 token(每步 size 不确定)。

用法:
  PYTHONPATH=/workspace/agent python3 research/bench_solpi_mechanisms.py
  PYTHONPATH=/workspace/agent python3 research/bench_solpi_mechanisms.py --log-theme lammps --mutations 4 --validation-turn-tokens 700
  PYTHONPATH=/workspace/agent python3 research/bench_solpi_mechanisms.py --json
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("HUGINN_CACHE_DIR", tempfile.mkdtemp())

from huginn.utils.tokens import rough_token_count_for_text  # noqa: E402
from huginn.tools.observation_pack import ObservationArchive  # noqa: E402
from huginn.tools.evidence_reducer import reduce_log_to_receipt  # noqa: E402
from huginn.autoloop.online_compact import (  # noqa: E402
    gate_compaction,
    select_compact_candidates,
    _fake_step,
)


def _tok(s: str) -> int:
    return rough_token_count_for_text(s)


def _vasp_like_log() -> str:
    """造一个 >20k token 的超长 VASP 风格日志, 触发 offload/收据化."""
    lines = []
    lines.extend(f"INFO   FFT grid : {i} nx ny nz" for i in range(4000))
    lines.append("ERROR  ZBRENT: did not find energy minimum")
    lines.extend(f"DATA   energy(no entropy) = {i:.6f}   total force = {i * 0.01:.4f}"
                 for i in range(8000))
    return "\n".join(lines)


def _lammps_like_log() -> str:
    """造一个超长 LAMMPS 风格日志(热力学步), 触发 offload/收据化."""
    lines = [f"Step {i} Temp {300.0 + i * 0.1:.4f} Press {1.0 if i % 2 == 0 else 1.2:.4f}"
             for i in range(12000)]
    lines.append("ERROR: Bond atoms 12 13 missing at step 4999 (../bond.cpp:523)")
    return "\n".join(lines)


def _log_theme(name: str) -> str:
    return _lammps_like_log() if name == "lammps" else _vasp_like_log()


def run_bench(
    log: str, *, mutations: int = 8, validation_turn_tokens: int = 900
) -> dict:
    """机制级 token 节省测算, 返回可复用 dict(供 --json / 测试复用)."""
    log_tok = _tok(log)

    # ② ObservationPack
    archive = ObservationArchive("bench")
    handle, _ = archive.archive_body(log)
    first_page = archive.recall(handle, page=0, page_size=4000)
    preview_tok = _tok(first_page["text"])
    save2 = log_tok - preview_tok

    # ③ Evidence-Preserving Reducer
    rec = reduce_log_to_receipt(ObservationArchive("bench3"), log)
    receipt_tok = _tok("\n".join([*rec.head, *rec.tail, *rec.quotes]))
    save3 = log_tok - receipt_tok

    # ④ Online Compact (计数)
    steps = [_fake_step(f"s{i}", status="completed", has_evidence=True) for i in range(12)] + \
            [_fake_step(f"active{i}", status="in_progress", has_evidence=True) for i in range(3)]
    cand = select_compact_candidates(steps)
    g = gate_compaction(steps, window_pct=70, cache_write_read_ratio=20)

    # ① Action Fusion (估算折算)
    save1 = validation_turn_tokens * mutations

    return {
        "log_theme_tok": log_tok,
        "obs_raw_tok": log_tok,
        "obs_first_page_tok": preview_tok,
        "obs_save_tok": save2,
        "obs_handle": handle[:12] + "...",
        "receipt_tok": receipt_tok,
        "receipt_save_tok": save3,
        "receipt_sealed": rec.sealed,
        "receipt_omitted_lines": rec.omitted,
        "compact_total_steps": len(steps),
        "compact_foldable": len(cand),
        "compact_preserve": len(g["preserve_evidence"]),
        "compact_should": g["should_compact"],
        "fusion_save_tok": save1,
        "fusion_mutations": mutations,
        "total_save_tok_est": save2 + save3 + save1,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="SoL-Pi 四件套 token 节省 benchmark")
    ap.add_argument("--log-theme", choices=["vasp", "lammps"], default="vasp")
    ap.add_argument("--mutations", type=int, default=8)
    ap.add_argument("--validation-turn-tokens", type=int, default=900)
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = ap.parse_args(argv)

    log = _log_theme(args.log_theme)
    r = run_bench(log, mutations=args.mutations,
                  validation_turn_tokens=args.validation_turn_tokens)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    print("=" * 74)
    print(f"SoL-Pi 四件套 token 节省 benchmark (机制级, {args.log_theme} 日志)")
    print("=" * 74)
    print(f"\n[基准] 日志 {r['log_theme_tok']:,} tok\n")

    print("机制② ObservationPack:")
    print(f"  原始全量 {r['obs_raw_tok']:,} tok | 上下文只放句柄+首屏 {r['obs_first_page_tok']:,} tok")
    print(f"  单次省 {r['obs_save_tok']:,} tok (handle={r['obs_handle']} 可分页召回无丢字)\n")

    print(f"机制③ Evidence-Preserving Reducer: 收据 {r['receipt_tok']:,} tok "
          f"(省略 {r['receipt_omitted_lines']} 行, sealed={r['receipt_sealed']}) 单次省 {r['receipt_save_tok']:,} tok\n")

    print(f"机制④ Online Context Compact: {r['compact_total_steps']} 步 → 可折叠 "
          f"{r['compact_foldable']}(已验证+密封) / 必须保留证据 {r['compact_preserve']}, should_compact={r['compact_should']}\n")

    print(f"机制① Action Fusion: {r['fusion_mutations']} 次变更 ×1 条验证往返 "
          f"(={args.validation_turn_tokens} tok/轮, 估算) 省 {r['fusion_save_tok']:,} tok\n")

    print("=" * 74)
    print(f"四件套合计(代表一轮, 估算): 省 ≈{r['total_save_tok_est']:,} tok")
    print("诚实边界: 二/三是机制级实测量, 一是估算折算(需真 LLM 定价复核), 四是计数。")
    print("=" * 74)


if __name__ == "__main__":
    main()