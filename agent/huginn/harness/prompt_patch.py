"""H1: Prompt Template Patch 闭环.

agent 跑完一轮 autoloop, _learn 把 r_phys 喂给 patch store, LLM 看上轮
directive + r_phys 生成新 patch (block_name + new_text). 下一轮
_build_hypothesis_prompt / _build_plan_prompt 拼完 blocks 后调
apply_patches 替换/前置/后置对应 block.

store 用懒加载单例 + JSON 文件, 不走 EvolutionEngine (避免 P5 两套系统耦合).
toggle: cfg.feature_flags.harness_prompt_patch (默认 off, mtime auto reload).

数学: harness 变体 v 的 prompt_patches 是 {b: new_text}, 应用时 block b
用 new_text 替换默认值. 双层优化: 内层 = 跑任务拿 r_phys, 外层 = LLM
看 r_phys 生成新 patch. Beta(α, β) 信念独立维护 (跟 ToolBelief 数学同,
key schema 不同), 不复用 ToolBelief 避免污染 skill 进化数据.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# LRU 上限, 跟 spec 一致. ponytail: 不做 per-phase quota, 全局 LRU 够用.
_PATCH_STORE_MAX = 20


def _harness_enabled(key: str, default: bool = False) -> bool:
    """读 cfg.feature_flags.<key>, mtime 自动 reload. 默认 off."""
    try:
        from huginn.config import get_config
        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        return bool(ff.get(key, default))
    except Exception:
        return default


@dataclass
class PromptPatch:
    """patch store 里的一条记录.

    alpha/beta: Beta(α, β) 信念, success → α+=1, fail → β+=1.
    Ponytail: 暂不做 ANCCR 时间加权 (ToolBelief 那套), 直接 alpha/beta 够用.
    升级路径: 加 weighted_alpha/beta 跟 ToolBelief 数学统一.

    directive_in: 生成此 patch 时用的 RSI directive (修 P4: directive
    从 memory 软检索改成 patch store 定向读, find_by_directive 反查).
    """
    id: str
    phase: str
    block_name: str
    new_text: str
    op: str = "replace"  # replace | prepend | append
    alpha: int = 1
    beta: int = 1
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    directive_in: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "block_name": self.block_name,
            "new_text": self.new_text,
            "op": self.op,
            "alpha": self.alpha,
            "beta": self.beta,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "directive_in": self.directive_in,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptPatch":
        return cls(
            id=d["id"],
            phase=d["phase"],
            block_name=d["block_name"],
            new_text=d["new_text"],
            op=d.get("op", "replace"),
            alpha=d.get("alpha", 1),
            beta=d.get("beta", 1),
            created_at=d.get("created_at", time.time()),
            last_used=d.get("last_used", time.time()),
            directive_in=d.get("directive_in", ""),
        )


class PromptPatchStore:
    """单例 patch store. 存 .huginn/prompt_patches/<id>.json, LRU 上限 20.

    跨 iter 状态持久: 进程内单例 + 磁盘文件. 失败静默, 不阻塞主循环.
    跟 _get_evolution (engine.py:460) 同模式懒加载.
    """
    _instance: "PromptPatchStore | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = Path(
            os.environ.get("HUGINN_CACHE_DIR", Path.home() / ".huginn")
        )
        self._store_dir = cache_dir / "prompt_patches"
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("prompt_patches dir create failed", exc_info=True)
        self._patches: dict[str, PromptPatch] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> "PromptPatchStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_all(self) -> None:
        try:
            for f in self._store_dir.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    p = PromptPatch.from_dict(d)
                    self._patches[p.id] = p
                except Exception:
                    logger.debug("patch load fail: %s", f, exc_info=True)
        except Exception:
            logger.debug("patch dir scan fail", exc_info=True)

    def list_patches(self, phase: str | None = None) -> list[PromptPatch]:
        """列出 patch. phase 过滤可选. 按 Beta mean 降序."""
        with self._lock:
            ps = list(self._patches.values())
        if phase:
            ps = [p for p in ps if p.phase == phase]
        ps.sort(
            key=lambda p: p.alpha / max(1, p.alpha + p.beta),
            reverse=True,
        )
        return ps

    def add_patch(self, patch: PromptPatch) -> bool:
        """加 patch. 超 LRU 上限踢掉 Beta mean 最低 + last_used 最老的.

        body 块 replace 时检查 {context} 占位符 (失败模式 9):
        缺占位符降级 prepend, 避免丢上下文.
        """
        if patch.block_name == "body" and patch.op == "replace":
            if "{context}" not in patch.new_text and "{hypothesis}" not in patch.new_text:
                logger.info(
                    "patch %s body replace → prepend (lost context placeholder)",
                    patch.id,
                )
                patch.op = "prepend"
        with self._lock:
            self._patches[patch.id] = patch
            if len(self._patches) > _PATCH_STORE_MAX:
                worst = min(
                    self._patches.values(),
                    key=lambda p: (
                        p.alpha / max(1, p.alpha + p.beta),
                        p.last_used,
                    ),
                )
                self._patches.pop(worst.id, None)
                try:
                    (self._store_dir / f"{worst.id}.json").unlink(missing_ok=True)
                except Exception:
                    pass
        self._save_patch(patch)
        return True

    def _save_patch(self, patch: PromptPatch) -> None:
        try:
            f = self._store_dir / f"{patch.id}.json"
            f.write_text(
                json.dumps(patch.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("patch save fail: %s", patch.id, exc_info=True)

    def update_alpha_beta(self, patch_id: str, success: bool) -> None:
        """Beta 更新. success → α+=1, fail → β+=1. 持久化."""
        with self._lock:
            p = self._patches.get(patch_id)
            if p is None:
                return
            if success:
                p.alpha += 1
            else:
                p.beta += 1
            p.last_used = time.time()
        self._save_patch(p)

    def find_by_directive(self, directive: str) -> PromptPatch | None:
        """根据 RSI directive 反查 patch.

        修 P4: directive 从 memory 软检索改成 patch store 定向读.
        """
        with self._lock:
            for p in self._patches.values():
                if p.directive_in and p.directive_in == directive:
                    return p
        return None


def apply_patches(
    blocks: list[tuple[str, str]],
    phase: str,
) -> list[tuple[str, str]]:
    """在 _build_*_prompt 拼完 blocks 后调, 按 phase 查 patch store 应用.

    必须在 _trim_to_budget 调用前注入 (engine.py:7278/7518), 因为
    _scan_block_conflicts 已在 _trim_to_budget 内自动跑 (engine.py:842),
    patch 后的 blocks 自动过 conflict 检查, 不需要额外接入.

    toggle off 时直接返回原 blocks (零开销).
    只应用 Beta mean > 0.5 的 patch (低信念 patch 等积累数据).
    同名 block 取最高 Beta mean 的 patch.
    """
    if not _harness_enabled("harness_prompt_patch"):
        return blocks
    # H3 接入: toggle on 时按 UCB 选 block 子集 (核心 block 必保留)
    try:
        from huginn.harness.joint_optimizer import select_block_subset_for_phase
        blocks = select_block_subset_for_phase(phase, blocks)
    except Exception:
        logger.debug("H3 block subset in apply_patches failed", exc_info=True)
    try:
        store = PromptPatchStore.get_instance()
        patches = store.list_patches(phase=phase)
    except Exception:
        logger.debug("apply_patches: store fail", exc_info=True)
        return blocks
    if not patches:
        return blocks
    good = [
        p for p in patches
        if p.alpha / max(1, p.alpha + p.beta) > 0.5
    ]
    if not good:
        return blocks
    by_block: dict[str, PromptPatch] = {}
    for p in good:
        cur = by_block.get(p.block_name)
        if cur is None or (
            p.alpha / max(1, p.alpha + p.beta)
            > cur.alpha / max(1, cur.alpha + cur.beta)
        ):
            by_block[p.block_name] = p
    out: list[tuple[str, str]] = []
    for name, text in blocks:
        p = by_block.get(name)
        if p is None:
            out.append((name, text))
            continue
        if p.op == "replace":
            new_text = p.new_text
        elif p.op == "prepend":
            new_text = p.new_text + "\n" + text
        else:  # append
            new_text = text + "\n" + p.new_text
        out.append((name, new_text))
    return out


async def generate_patch(
    phase: str,
    blocks: list[tuple[str, str]],
    r_phys: float | None,
    directive: str,
    llm_chat_fn: Any,
) -> PromptPatch | None:
    """LLM 看 r_phys + directive + 当前 blocks, 生成新 patch.

    llm_chat_fn: async callable(prompt, task=...) -> str (engine._llm_chat 同签名).
    失败 (JSON parse / block 不存在) 静默返回 None, 不阻塞主循环.

    只在 r_phys <= 0.7 时生成 (高 r_phys 不需要改).
    ponytail: 不做 patch diff / version control, JSON 文件够用.
    """
    if not _harness_enabled("harness_prompt_patch"):
        return None
    if r_phys is None or r_phys > 0.7:
        return None
    block_names = [name for name, _ in blocks]
    prompt = (
        "You are optimizing a research agent's prompt template. Based on the "
        "last iteration's physical validation score and self-directive, "
        "propose ONE block-level patch.\n\n"
        f"Phase: {phase}\n"
        f"Available blocks: {block_names}\n"
        f"R_phys (last iter): {r_phys}\n"
        f"Self-directive: {directive}\n\n"
        "Output JSON only:\n"
        '{"block_name": "<one of available>", '
        '"op": "replace|prepend|append", '
        '"new_text": "<new block content>"}\n'
        "Rules:\n"
        "- replace 'body' block: must preserve {context} or {hypothesis} placeholder\n"
        "- new_text max 500 chars\n"
        "- op=prepend/append preserves original block text"
    )
    try:
        response = await llm_chat_fn(prompt, task="summarize")
    except Exception:
        logger.debug("generate_patch LLM fail", exc_info=True)
        return None
    if not (response and response.strip()):
        return None
    txt = response.strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(txt)
    except Exception:
        logger.debug("generate_patch JSON parse fail: %s", txt[:200])
        return None
    block_name = d.get("block_name", "")
    if block_name not in block_names:
        logger.debug(
            "generate_patch: block_name %s not in %s",
            block_name, block_names,
        )
        return None
    op = d.get("op", "replace")
    if op not in ("replace", "prepend", "append"):
        op = "replace"
    new_text = str(d.get("new_text", ""))[:500]
    if not new_text.strip():
        return None
    patch = PromptPatch(
        id=f"pp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
        phase=phase,
        block_name=block_name,
        new_text=new_text,
        op=op,
        directive_in=directive[:300],
    )
    PromptPatchStore.get_instance().add_patch(patch)
    return patch


def _selfcheck() -> None:
    """H1 selfcheck: apply_patches + Beta 更新 + body 占位符降级."""
    import shutil
    import tempfile

    import huginn.harness.prompt_patch as pp

    # 1. toggle off: passthrough
    # 注意: 全部走 pp. 前缀, 避免 python -m 时 __main__ 跟 sys.modules 里的
    # huginn.harness.prompt_patch 是两份模块对象, monkey-patch 失效.
    blocks = [("body", "original {context}"), ("mem", "mem text")]
    out = pp.apply_patches(blocks, "hypothesize")
    assert out is blocks, "toggle off should return same list"
    print("1. apply_patches toggle off → passthrough OK")

    # 2. toggle on: 模拟 patch 应用
    orig = pp._harness_enabled
    pp._harness_enabled = lambda key, default=False: (
        True if key == "harness_prompt_patch" else default
    )
    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    pp.PromptPatchStore._instance = None

    store = pp.PromptPatchStore.get_instance()
    p = pp.PromptPatch(
        id="test1",
        phase="hypothesize",
        block_name="mem",
        new_text="PATCHED PREFIX",
        op="prepend",
    )
    store.add_patch(p)
    # alpha=3, beta=1 → 0.75 > 0.5 才会被 apply
    store.update_alpha_beta("test1", success=True)
    store.update_alpha_beta("test1", success=True)

    out = pp.apply_patches(blocks, "hypothesize")
    assert out[0] == ("body", "original {context}"), f"body changed: {out[0]}"
    assert out[1][0] == "mem"
    assert "PATCHED PREFIX" in out[1][1], f"mem not patched: {out[1][1]}"
    assert "mem text" in out[1][1], f"original mem lost: {out[1][1]}"
    print("2. apply_patches toggle on + prepend patch OK")

    # 3. body replace 缺 {context} → 降级 prepend
    p2 = pp.PromptPatch(
        id="test2",
        phase="hypothesize",
        block_name="body",
        new_text="body without placeholder",
        op="replace",
    )
    store.add_patch(p2)
    assert p2.op == "prepend", f"body replace should degrade: {p2.op}"
    print("3. body replace missing {context} → prepend degrade OK")

    # 4. update_alpha_beta 持久化
    store.update_alpha_beta("test2", success=True)
    p2_reload = store._patches["test2"]
    assert p2_reload.alpha == 2, f"alpha should be 2: {p2_reload.alpha}"
    print("4. update_alpha_beta persistence OK")

    # 5. find_by_directive 反查 (P4 修复点)
    p3 = pp.PromptPatch(
        id="test3",
        phase="hypothesize",
        block_name="mem",
        new_text="directive test",
        op="prepend",
        directive_in="avoid VASP when kpoints < 4",
    )
    store.add_patch(p3)
    found = store.find_by_directive("avoid VASP when kpoints < 4")
    assert found is not None and found.id == "test3"
    print("5. find_by_directive (P4 fix) OK")

    # 清理
    shutil.rmtree(tmp, ignore_errors=True)
    del os.environ["HUGINN_CACHE_DIR"]
    pp.PromptPatchStore._instance = None
    pp._harness_enabled = orig
    print("H1 prompt_patch selfcheck OK (5/5)")


if __name__ == "__main__":
    _selfcheck()
