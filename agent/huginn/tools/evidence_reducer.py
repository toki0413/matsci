"""evidence_reducer —— Huginn 版 SoL-Pi「Evidence-Preserving Reducer」（落地③号）。

缺口: Huginn 把长诊断日志用 ``compress.smart_compress_text`` 做**有损**LLM摘要,
被丢弃的引文无法回源验证——摘要可悄悄编造。SoL-Pi 的不变量: 任何进收据的保留
引用必须**逐字**匹配归档原文, 且封进上下文前验证通过。

本模块落地该机制, 核心纪律是**只允许保留源码原文行, 绝不允许生成式散文/编造引文**:
  - ``reduce_log_to_receipt(archive, log_text)``: 归档全文→句柄, 收据里 head/tail/
    引文抽样全是**逐字取自源码的行**, 不做任何 LLM 改写; 统计省略行数。
  - ``seal_receipt(receipt)``: 只有当所有保留行都能在归档原文里逐字命中才 ``sealed``。
  - ``verify_receipt(receipt, archive)``: 独立回源校验——任何保留行在原文缺 ⇒ False。

与 ObservationArchive 的关系: 复用其 ``@obs:<id>`` 句柄作为"归档原文"锚点, 收据
只带句柄+逐字行, 需要全文时可召回。

设计约束: 纯本地、无 LLM; 收据生成/验证都是纯函数, 可独立单测。默认 local-only:
本模块不外发日志(reducer 远程化留给将来可被显式配置的接口)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from huginn.tools.observation_pack import ObservationArchive


@dataclass
class Receipt:
    """一份"只含逐字原文行"的压缩收据."""

    handle: str  # 归档原文的句柄 (@obs:<id>)
    head: list[str] = field(default_factory=list)      # 保留的源码首部行
    tail: list[str] = field(default_factory=list)      # 保留的源码尾部行
    quotes: list[str] = field(default_factory=list)    # 抽样保留的诊断引文行
    omitted: int = 0                                   # 被省略的行数
    sealed: bool = False                               # 全部保留行已在原文验证通过

    def retained_lines(self) -> list[str]:
        return [*self.head, *self.tail, *self.quotes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "head": self.head,
            "tail": self.tail,
            "quotes": self.quotes,
            "omitted": self.omitted,
            "sealed": self.sealed,
        }


def _pick_quotes(lines: list[str], *, max_quotes: int = 4) -> list[str]:
    """从源码行里采样保留引文: 尽量取非空、含异常的诊断行, 只许逐字照抄."""
    out: list[str] = []
    prefer = [ln for ln in lines if ln.strip() and any(
        k in ln.lower() for k in ("error", "warn", "fail", "exception", "traceback")
    )]
    pool = prefer or [ln for ln in lines if ln.strip()]
    for ln in pool:
        if len(out) >= max_quotes:
            break
        out.append(ln)
    if not out and lines:
        # 没诊断字样时取首行作引文, 仍是逐字照抄
        out.append(lines[0])
    return out


def reduce_log_to_receipt(
    archive: ObservationArchive,
    log_text: str,
    *,
    head_lines: int = 8,
    tail_lines: int = 6,
    max_quotes: int = 4,
) -> Receipt:
    """归档全文并产出**只含逐字原文行**的收据.

    不做任何 LLM 生成; 省略中间行并记 omitted。head/tail/quotes 全部取自
    ``log_text.splitlines()`` 的原子行, 因此天然是原文子串。
    """
    lines = log_text.splitlines()
    # 保留行先选出, 用于后续 seal 时回源校验
    head = [ln for ln in lines[:head_lines] if ln.strip()]
    tail = [ln for ln in lines[-tail_lines:] if ln.strip()]
    quotes = _pick_quotes(lines, max_quotes=max_quotes)
    handle, _path = archive.archive_body(log_text)
    omitted = max(0, len(lines) - len(head) - len(tail) - len(quotes) - int(bool(
        set(head) & set(tail)
    )))
    receipt = Receipt(
        handle=handle, head=head, tail=tail, quotes=quotes, omitted=omitted
    )
    receipt.sealed = _seal_receipt(receipt, archive)
    return receipt


def _seal_receipt(receipt: Receipt, archive: ObservationArchive) -> bool:
    """seal: 只有当所有保留行都逐字出现在归档原文里才为真. 否则不再改 head/tail."""
    body = archive.recall(receipt.handle)
    if not body["found"]:
        return False
    source = body["text"]
    for line in receipt.retained_lines():
        if line and line not in source:
            return False
    return True


def verify_receipt(receipt: Receipt, archive: ObservationArchive) -> bool:
    """独立回源校验: 任一保留行在归档原文缺失 ⇒ False (收据不可信)."""
    return _seal_receipt(receipt, archive)


def _selfcheck() -> None:
    import os
    import tempfile

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    try:
        archive = ObservationArchive("selftest")
        # 1. 正常日志 → 收据 sealed, 引文全是逐字行
        log = "\n".join(
            [f"INFO step {i}" for i in range(50)]
            + ["ERROR vasp relax did not converge"]
            + [f"DATA energy={i}" for i in range(20)]
        )
        rec = reduce_log_to_receipt(archive, log)
        assert rec.sealed, rec.to_dict()
        assert rec.handle.startswith("@obs:")
        assert all(ln in log for ln in rec.retained_lines()), "quotes must be verbatim from source"
        assert any("ERROR" in q for q in rec.quotes) or not rec.quotes, rec.quotes
        assert rec.omitted >= 1, rec.omitted
        # 2. verify_receipt 独立通过
        assert verify_receipt(rec, archive) is True

        # 3. 篡改引文 → verify 失败 (收据不可信)
        rec2 = Receipt(handle=rec.handle, quotes=["ERROR made-up line not in source"])
        assert verify_receipt(rec2, archive) is False

        # 4. 空日志 → 收据空, omitted=0
        rec3 = reduce_log_to_receipt(archive, "")
        assert rec3.sealed and rec3.omitted == 0, rec3.to_dict()
        print("OK evidence_reducer self-check passed (verbatim-anchored receipt + seal)")
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("HUGINN_CACHE_DIR", None)


if __name__ == "__main__":
    _selfcheck()