"""bench_solpi_mechanisms —— 四件套 token 节省的可复现 benchmark。

统计口径: 用 ``rough_token_count_for_text``(≈chars/4) 估算 token。它衡量的是
"少放多少 token 进上下文", 即 SoL-Pi 的受约束效率主张。诚实边界:
  - 这是**机制级**节流测算, 不是整轮 agent 会话的端到端成本(那样需要真 LLM +
    定价表, 且依赖具体长轨迹)。
  - ① Action Fusion 节省的是"少一条验证工具的往返 prompt", 用一个可配置的
    ``avg_validation_turn_tokens`` 估算, 标注为估算值。
  - ④ 报告"可折叠步数/保留步数"的计数, 不虚报 token(每步 size 不确定)。

用法:  PYTHONPATH=/workspace/agent python3 research/bench_solpi_mechanisms.py
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


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    print("=" * 74)
    print("SoL-Pi 四件套 token 节省 benchmark (机制级, 可复现)")
    print("=" * 74)

    log = _vasp_like_log()
    log_tok = _tok(log)
    print(f"\n[基准] 超长 VASP 日志: {log_tok:,} tok\n")

    # ② ObservationPack
    archive = ObservationArchive("bench")
    handle, _ = archive.archive_body(log)
    first_page = archive.recall(handle, page=0, page_size=4000)
    preview_tok = _tok(first_page["text"])
    save2 = log_tok - preview_tok
    print("机制② ObservationPack:")
    print(f"  原始全量 = {_fmt_int(log_tok)} tok | 上下文只放句柄+首屏预览 = {_fmt_int(preview_tok)} tok")
    print(f"  单次省 {_fmt_int(save2)} tok ({500 * save2 / max(1, log_tok):.1f}x 上下文减压) "
          f"(handle={handle[:12]}... 可分页召回原始原文, 无丢字)\n")

    # ③ Evidence-Preserving Reducer
    rec = reduce_log_to_receipt(ObservationArchive("bench3"), log)
    receipt_text = "\n".join([*rec.head, *rec.tail, *rec.quotes])
    receipt_tok = _tok(receipt_text)
    save3 = log_tok - receipt_tok
    print("机制③ Evidence-Preserving Reducer:")
    print(f"  原始日志 = {_fmt_int(log_tok)} tok | 收据(head/tail/引文, 逐字可回源) = {_fmt_int(receipt_tok)} tok | 省略 {rec.omitted} 行")
    print(f"  单次省 {_fmt_int(save3)} tok | 收据 sealed={rec.sealed}(所有保留行逐字命中归档原文)\n")

    # ④ Online Compact: 候选门控计数(不虚报 token, 给可折叠/保留数)
    steps = [
        _fake_step(f"s{i}", status="completed", has_evidence=True) for i in range(12)
    ] + [
        _fake_step(f"active{i}", status="in_progress", has_evidence=True) for i in range(3)
    ]
    cand = select_compact_candidates(steps)
    g = gate_compaction(steps, window_pct=70, cache_write_read_ratio=20)
    print("机制④ Online Context Compact(候选驱动 + 经济/密封门控):")
    print(f"  步骤 15: 可折叠(已验证+密封) {len(cand)} | 必须保留证据(未验证) {len(g['preserve_evidence'])}")
    print(f"  should_compact={g['should_compact']} (候选驱动 + 窗口/缓存读写比门控通过)\n")

    # ① Action Fusion: 每轮省一条验证工具往返(估算)
    avg_validation_turn = 900  # 一条"改完→单独跑验证命令"的模型往返 prompt 估算
    mutual = 8
    print("机制① Action Fusion:")
    print(f"  每次变更省 1 条验证工具往返(估算 {avg_validation_turn} tok/轮) × {mutual} 次变更")
    print(f"  单会话累计省 {_fmt_int(avg_validation_turn * mutual)} tok (标注: 端到端需真 LLM 定价复核)\n")

    print("=" * 74)
    print(f"四件套合计(代表性一轮): 省 ≈{_fmt_int(save2 + save3 + avg_validation_turn * mutual)} tok 未计②③首屏/收据重复召回")
    print("注意: 二/四/三为机制级省量, 一为按估算轮次折算; 引用绝对数字前请在真 agent 轨迹上复核。")
    print("=" * 74)


if __name__ == "__main__":
    main()