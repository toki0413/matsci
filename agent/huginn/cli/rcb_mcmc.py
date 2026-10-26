"""RCB MCMC 模式 — 依赖 rcb_cognition / rcb_utils."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

from huginn.cli.rcb_cognition import _init_hypothesis_manifold
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


def _register_se3_maps(ws: Path, task_id: str, mode: str, manifold: Any) -> None:
    """SE(3) 层: 从 engine_state 读 cognitive_maps, 按 index 注册到 hypothesis."""
    try:
        from huginn.metacog.structure_cognitive_map import (
            StructureCognitiveMap as SCM,  # noqa: N817
        )
        from huginn.runtime.engine_state import load_engine_state as les

        est = les(task_id, ws)
        cmaps = getattr(est, "cognitive_maps", {}) if est else {}
        if not cmaps:
            print(
                f"[mcmc-{mode}] SE(3) enabled but no cognitive_maps found, "
                f"degrading to fisher",
                flush=True,
            )
            return
        # Map each cognitive_map to a hypothesis by matching h_id.
        # ponytail: simple 1:1 mapping by index; upgrade: explicit
        #   structure_id assignment via UI or hypothesis metadata.
        h_ids = list(manifold._hyp)
        for i, (mid, mdict) in enumerate(cmaps.items()):
            if i >= len(h_ids):
                break
            try:
                manifold.register_structure(
                    h_ids[i], mid, SCM.from_engine_state_dict(mdict)
                )
            except Exception:
                logger.debug("cognitive_map structure register skipped", exc_info=True)
        print(
            f"[mcmc-{mode}] SE(3) loaded {len(manifold._structure_maps)} "
            f"cognitive_map(s) for structure-guided proposal",
            flush=True,
        )
    except Exception as e:
        print(
            f"[mcmc-{mode}] SE(3) cognitive_map load failed: {e}, degrading to fisher",
            flush=True,
        )


def _register_haptic_layers(ws: Path, mode: str, manifold: Any) -> int:
    """触觉层: 从 haptic_layers.json 读力学属性并注册, 返回注册数量."""
    # ponytail: 文件不存在或空 → 不注册, haptic_enabled=True 时
    #   _haptic_proposal 对所有 h_id 返回 None, 安全退化到 fisher.
    #   升级路径: VASP static / ML potential / Materials MP 查询结果写进这个文件.
    hap_path = ws / HUGINN_DIR_NAME / "haptic_layers.json"
    n_hap = 0
    if hap_path.exists():
        try:
            from huginn.metacog.haptic_property_layer import (
                HapticPropertyLayer as HPL,  # noqa: N817
            )

            h_ids = list(manifold._hyp)
            raw = json.loads(hap_path.read_text(encoding="utf-8"))
            # key 可以是 h_id 或结构 id, 优先 h_id 匹配, 否则按 index 回退
            for i, h_id in enumerate(h_ids):
                d = raw.get(h_id)
                if d is None and i < len(raw):
                    d = list(raw.values())[i]
                if d is None:
                    continue
                try:
                    manifold.register_haptic(h_id, HPL.from_dict(d))
                    n_hap += 1
                except Exception:
                    logger.debug("mcmc haptic layer register skipped", exc_info=True)
        except Exception as e:
            print(
                f"[mcmc-{mode}] haptic load failed: {e}, degrading to fisher",
                flush=True,
            )
    if n_hap > 0:
        print(
            f"[mcmc-{mode}] haptic loaded {n_hap} layer(s) for haptic-guided proposal",
            flush=True,
        )
    else:
        print(
            f"[mcmc-{mode}] haptic enabled but no layers in "
            f"{hap_path}, degrading to fisher",
            flush=True,
        )
    return n_hap


def _fit_alignment(mode: str, manifold: Any, dataset: Any) -> None:
    """数据量 >= 10 时 fit AlignmentFunction 并注入 manifold."""
    try:
        from huginn.metacog.alignment import AlignmentFunction
        from huginn.metacog.haptic_descriptor import HapticDescriptor
        from huginn.metacog.structure_descriptor import StructureDescriptor

        n_pairs = dataset.count("structure", "haptic")
        if n_pairs < 10:
            print(
                f"[mcmc-{mode}] alignment dataset has only {n_pairs} pairs "
                f"(need >=10), degrading to fisher",
                flush=True,
            )
            return
        af = AlignmentFunction(
            StructureDescriptor(), HapticDescriptor(), min_samples=10
        )
        af.fit(dataset)
        if af.ready:
            manifold.set_alignment_function(af)
            print(
                f"[mcmc-{mode}] alignment fitted on {n_pairs} pairs, "
                f"alignment-guided proposal enabled",
                flush=True,
            )
        else:
            print(
                f"[mcmc-{mode}] alignment fit returned not-ready "
                f"({n_pairs} pairs), degrading to fisher",
                flush=True,
            )
    except Exception as e:
        print(
            f"[mcmc-{mode}] alignment load/fit failed: {e}, degrading to fisher",
            flush=True,
        )


def _setup_alignment(ws: Path, mode: str, manifold: Any) -> Any:
    """对齐层: 从 alignment_dataset.json 加载数据集, 足够时 fit 并注入."""
    # ponytail: 文件不存在 / 数据不足 / fit 失败 → 返回 None 或未注入,
    #   alignment_enabled=True 时 _alignment_proposal 全返 None, 安全退化 fisher.
    align_path = ws / HUGINN_DIR_NAME / "alignment_dataset.json"
    if not align_path.exists():
        print(
            f"[mcmc-{mode}] alignment enabled but no dataset at "
            f"{align_path}, degrading to fisher",
            flush=True,
        )
        return None
    try:
        from huginn.metacog.alignment_dataset import AlignmentDataset

        dataset = AlignmentDataset.load(align_path)
    except Exception as e:
        print(
            f"[mcmc-{mode}] alignment load/fit failed: {e}, degrading to fisher",
            flush=True,
        )
        return None
    _fit_alignment(mode, manifold, dataset)
    return dataset


def _add_surprise_pair(
    manifold: Any, dataset: Any, sd: Any, hd: Any, h_id: str, score: float
) -> None:
    """把一个 surprise 命中的 (structure, haptic) 对写回数据集."""
    h = manifold._hyp.get(h_id)
    if h is None or h.structure_id is None:
        return
    cmap = manifold._structure_maps.get(h.structure_id)
    layer = manifold._haptic_layers.get(h_id)
    if cmap is None or layer is None:
        return
    with contextlib.suppress(Exception):
        dataset.add(
            sd.encode(cmap),
            hd.encode(layer),
            "structure",
            "haptic",
            metadata={"h_id": h_id, "surprise": score},
        )


def _apply_surprise_feedback(ws: Path, mode: str, manifold: Any, dataset: Any) -> None:
    """查 surprise, score > 2.0 的 hypothesis 触发数据回流.

    ponytail: advisory only, 失败只 warn 不阻塞. model=None 时 trigger 走空.
    """
    try:
        findings: list[tuple[str, float]] = []
        for h_id in manifold._hyp:
            sc = manifold.check_surprise(h_id)
            if sc is not None and sc > 2.0:
                findings.append((h_id, sc))
        if not findings:
            return
        print(
            f"[mcmc-{mode}] surprise detected on {len(findings)} hypothesis(es)",
            flush=True,
        )
        # 数据回流: 把当前 (structure, haptic) 对存入 dataset
        from huginn.metacog.haptic_descriptor import (
            HapticDescriptor as HD,  # noqa: N817
        )
        from huginn.metacog.structure_descriptor import (
            StructureDescriptor as SD,  # noqa: N817
        )

        sd, hd = SD(), HD()
        for h_id, score in findings:
            _add_surprise_pair(manifold, dataset, sd, hd, h_id, score)
        try:
            dataset.save(ws / HUGINN_DIR_NAME / "alignment_dataset.json")
        except Exception as e:
            print(f"[mcmc-{mode}] dataset save failed: {e}", flush=True)
    except Exception as e:
        print(f"[mcmc-{mode}] surprise check failed: {e}", flush=True)


def _load_observations(ws: Path) -> list:
    """读 observations.jsonl — 主循环 _iter_observations 跨轮累积的观测量."""
    from huginn.metacog.hypothesis_manifold import Observation

    obs_path = ws / HUGINN_DIR_NAME / "observations.jsonl"
    obs_list: list = []
    if obs_path.exists():
        for line in obs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                obs_list.append(
                    Observation(
                        name=o["name"],
                        value=o["value"],
                        sigma=o.get("sigma", 1.0),
                    )
                )
            except Exception:
                logger.debug("observation line parse skipped", exc_info=True)
    return obs_list


def _resume_single_state(ws: Path, task_id: str, engine: Any) -> None:
    """single 模式 resume: 有 checkpoint 就从断点继续."""
    from huginn.runtime.engine_state import load_engine_state

    resume = load_engine_state(task_id, ws)
    if resume is None or resume._mcmc_step_count <= 0:
        return
    engine._mcmc_current = resume._mcmc_current
    engine._mcmc_accept_count = resume._mcmc_accept_count
    engine._mcmc_step_count = resume._mcmc_step_count
    if resume._mcmc_rng_state is not None:
        engine._mcmc_rng.setstate(resume._mcmc_rng_state)
    print(
        f"[mcmc-single] resume from step={engine._mcmc_step_count} "
        f"current={engine._mcmc_current} "
        f"accept={engine._mcmc_accept_count}",
        flush=True,
    )


def _initial_single_current(manifold: Any, obs_list: list, engine: Any) -> Any:
    """single 模式初始 current: resume 优先, 否则 abductive_inference, 最后随机."""
    current = engine._mcmc_current
    if current is not None:
        return current
    try:
        abd = manifold.abductive_inference(obs_list)
        current = abd.h_id if abd else None
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        current = None
    if current is None:
        current = random.Random(42).choice(list(manifold._hyp))
    return current


def _save_single_checkpoint(
    engine: Any, step: int, current: Any, task_id: str, ws: Path
) -> None:
    """落盘 single 模式 checkpoint, 失败只打印不抛."""
    from huginn.runtime.engine_state import save_engine_state

    try:
        engine._mcmc_rng_state = engine._mcmc_rng.getstate()
        save_engine_state(engine, task_id, ws)
        print(
            f"[mcmc-single] ckpt step={step} current={current} "
            f"accept={engine._mcmc_accept_count}",
            flush=True,
        )
    except Exception as e:
        print(f"[mcmc-single] ckpt failed: {e}", flush=True)


def _single_mcmc_step(
    manifold: Any,
    engine: Any,
    obs_list: list,
    step: int,
    n_steps: int,
    anneal: bool,
    t_high: float,
    gpp: float,
    flags: dict[str, Any],
    current: Any,
    cached_log_p: float | None,
) -> tuple[Any, float | None]:
    """单步: 温度退火 + 全局 proposal + Gramian, 更新 engine 计数."""
    prev = current
    temperature = t_high * (1.0 / t_high) ** (step / n_steps) if anneal else 1.0
    current, cached_log_p = manifold.mcmc_step(
        obs_list,
        current,
        rng=engine._mcmc_rng,
        cached_log_p_current=cached_log_p,
        temperature=temperature,
        global_proposal_prob=gpp,
        # Gramian 谱预条件 (沿高可控/高信息主轴提议); 数据不足自动退化.
        gramian_enabled=(os.environ.get("HUGINN_MCMC_GRAMIAN", "1") == "1"),
        gramian_k=int(os.environ.get("HUGINN_MCMC_GRAMIAN_K", "1")),
        **flags,
    )
    if current != prev:
        engine._mcmc_accept_count += 1
    engine._mcmc_step_count += 1
    engine._mcmc_current = current
    engine._iteration = step
    return current, cached_log_p


def _run_single_mode(
    manifold: Any,
    engine: Any,
    obs_list: list,
    task_id: str,
    ws: Path,
    n_steps: int,
    checkpoint_interval: int,
    flags: dict[str, Any],
) -> None:
    """单链 MCMC: 温度退火 + 全局 proposal 混合, 定期落盘."""
    _start_single_sandbox(n_steps, checkpoint_interval)

    _resume_single_state(ws, task_id, engine)
    current = _initial_single_current(manifold, obs_list, engine)
    cached_log_p: float | None = None
    start_step = engine._mcmc_step_count + 1
    # P2-7: 也走温度退火 + 全局 proposal 混合, 与 mcmc-multi 对齐.
    # 默认 t_high=10, 几何退火到 temperature(=1.0). HUGINN_MCMC_NO_ANNEAL=1 关闭.
    anneal = os.environ.get("HUGINN_MCMC_NO_ANNEAL", "0") != "1"
    t_high = float(os.environ.get("HUGINN_MCMC_T_HIGH", "10"))
    gpp = float(os.environ.get("HUGINN_MCMC_GLOBAL_PROPOSAL", "0.3"))
    for step in range(start_step, n_steps + 1):
        current, cached_log_p = _single_mcmc_step(
            manifold,
            engine,
            obs_list,
            step,
            n_steps,
            anneal,
            t_high,
            gpp,
            flags,
            current,
            cached_log_p,
        )

        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            _save_single_checkpoint(engine, step, current, task_id, ws)

    _print_single_summary(engine)


def _start_single_sandbox(n_steps: int, checkpoint_interval: int) -> None:
    from huginn.security.sandbox import create_sandbox

    # 单链: 一个 standard 沙箱 (8GB/4cpu/6h), MCMC 在主进程跑
    create_sandbox(profile="standard")
    print(
        f"[mcmc-single] sandbox profile=standard, "
        f"steps={n_steps}, ckpt_interval={checkpoint_interval}",
        flush=True,
    )


def _print_single_summary(engine: Any) -> None:
    rate = (
        engine._mcmc_accept_count / engine._mcmc_step_count
        if engine._mcmc_step_count > 0
        else 0.0
    )
    print(
        f"[mcmc-single] done: total_steps={engine._mcmc_step_count} "
        f"accept_rate={rate:.3f}",
        flush=True,
    )


def _save_multi_chain_checkpoint(
    engine: Any, chain_id: int, state: dict, task_id: str, ws: Path
) -> None:
    """多链 checkpoint: 每链 state 存 _mcmc_chains, 整体落盘一次."""
    from huginn.runtime.engine_state import save_engine_state

    engine._mcmc_chains[chain_id] = state
    engine._mcmc_current = state.get("current")
    engine._mcmc_accept_count = state.get("accept_count", 0)
    engine._mcmc_step_count = state.get("step", 0)
    engine._mcmc_rng_state = state.get("rng_state")
    try:
        save_engine_state(engine, task_id, ws)
        print(
            f"[mcmc-multi] chain {chain_id} ckpt at step {state.get('step')}",
            flush=True,
        )
    except Exception as e:
        print(f"[mcmc-multi] chain {chain_id} ckpt failed: {e}", flush=True)


async def _run_multi_mode(
    manifold: Any,
    engine: Any,
    obs_list: list,
    task_id: str,
    ws: Path,
    n_steps: int,
    n_chains: int,
    checkpoint_interval: int,
    flags: dict[str, Any],
) -> int:
    """多链 MCMC: K 链并行, R̂ 诊断."""
    n_per_chain = max(1, n_steps // n_chains)
    _start_multi_sandboxes(n_chains, n_per_chain)
    # ponytail: mcmc_multi_chain 内部 asyncio.gather 已并行, semaphore 没法
    #   注入 (要改 manifold 签名). n_chains > cpu_count 时是协程级切换不是
    #   真并行. 升级路径: ProcessPool + 每链独立进程, semaphore 限并发进程数.
    #   见 hypothesis_manifold.py L444 注释.

    def _on_chain_checkpoint(chain_id: int, state: dict) -> None:
        _save_multi_chain_checkpoint(engine, chain_id, state, task_id, ws)

    result = await manifold.mcmc_multi_chain(
        obs_list,
        n_chains=n_chains,
        n_steps_per_chain=n_per_chain,
        checkpoint_interval=checkpoint_interval,
        on_chain_checkpoint=_on_chain_checkpoint,
        **flags,
    )

    r_hat = result.get("r_hat", float("nan"))
    print(
        f"[mcmc-multi] done: r_hat={r_hat:.4f} "
        f"converged={result.get('converged')} "
        f"accept_rates={result.get('accept_rates')}",
        flush=True,
    )
    return 0


def _start_multi_sandboxes(n_chains: int, n_per_chain: int) -> None:
    from huginn.security.sandbox import create_sandbox

    cpu = os.cpu_count() or 1
    max_concurrent = min(n_chains, cpu)
    # 每链一个 standard 容器; n_chains > cpu_count 时实际是协程级切换
    for _ in range(n_chains):
        create_sandbox(profile="standard")
    print(
        f"[mcmc-multi] {n_chains} sandboxes (profile=standard), "
        f"cpu_count={cpu}, max_concurrent={max_concurrent}, "
        f"steps_per_chain={n_per_chain}",
        flush=True,
    )


def _proposal_flags(
    se3_enabled: bool,
    se3_angle_sigma: float,
    haptic_enabled: bool,
    haptic_temperature: float,
    alignment_enabled: bool,
    alignment_temperature: float,
) -> dict[str, Any]:
    """mcmc_step / mcmc_multi_chain 共用的 proposal 层开关."""
    return {
        "se3_enabled": se3_enabled,
        "se3_angle_sigma": se3_angle_sigma,
        "haptic_enabled": haptic_enabled,
        "haptic_temperature": haptic_temperature,
        "alignment_enabled": alignment_enabled,
        "alignment_temperature": alignment_temperature,
    }


def _init_mcmc_manifold(ws: Path, task_id: str) -> Any:
    """复用 RCB 路径的 manifold 初始化 — 优先从盘上加载, 没有就建 generic."""
    instr_path = ws / "INSTRUCTIONS.md"
    return _init_hypothesis_manifold(
        ws=ws,
        task_id=task_id,
        checklist="",
        instructions=instr_path if instr_path.exists() else "",
        scan_text="",
        model=None,
        task_ctx="",
    )


def _apply_mcmc_proposal_setup(
    ws: Path,
    task_id: str,
    mode: str,
    manifold: Any,
    toggles: tuple,
) -> None:
    """按开关注册 SE(3) / 触觉 / 对齐 proposal 层."""
    se3_enabled, haptic_enabled, alignment_enabled = toggles
    if se3_enabled:
        _register_se3_maps(ws, task_id, mode, manifold)
    if haptic_enabled:
        _register_haptic_layers(ws, mode, manifold)
    if alignment_enabled:
        dataset = _setup_alignment(ws, mode, manifold)
        if dataset is not None:
            _apply_surprise_feedback(ws, mode, manifold, dataset)


def _load_mcmc_observations(ws: Path, mode: str) -> list:
    """读 observations — 主循环 _iter_observations 跨轮累积, 这里从盘上恢复."""
    obs_list = _load_observations(ws)
    if not obs_list:
        obs_path = ws / HUGINN_DIR_NAME / "observations.jsonl"
        print(
            f"[mcmc-{mode}] no observations in {obs_path}, cannot run MCMC",
            file=sys.stderr,
        )
    return obs_list


def _make_mcmc_engine(ws: Path) -> Any:
    """engine holder — 跟主循环 L1054 一致, 让 save_engine_state 能拉到 _mcmc_* 字段."""
    import types as _mcmc_types

    return _mcmc_types.SimpleNamespace(
        _mcmc_current=None,
        _mcmc_rng=random.Random(int(os.environ.get("HUGINN_MCMC_SEED", "42"))),
        _mcmc_rng_state=None,
        _mcmc_accept_count=0,
        _mcmc_step_count=0,
        _mcmc_chains={},
        _iteration=0,
        workspace=ws,
        hypothesis_graph=None,
    )


async def _dispatch_mcmc_mode(
    manifold: Any,
    engine: Any,
    obs_list: list,
    task_id: str,
    ws: Path,
    mode: str,
    dims: tuple,
    flags: dict[str, Any],
) -> int:
    """single 走单链并返回 0, 其余走多链."""
    n_steps, n_chains, checkpoint_interval = dims
    if mode == "single":
        _run_single_mode(
            manifold,
            engine,
            obs_list,
            task_id,
            ws,
            n_steps,
            checkpoint_interval,
            flags,
        )
        return 0

    return await _run_multi_mode(
        manifold,
        engine,
        obs_list,
        task_id,
        ws,
        n_steps,
        n_chains,
        checkpoint_interval,
        flags,
    )


async def _run_mcmc_mode(
    ws: Path,
    task_id: str,
    mode: str,
    n_steps: int,
    n_chains: int,
    checkpoint_interval: int,
    *,
    se3_enabled: bool = False,
    se3_angle_sigma: float = 30.0,
    haptic_enabled: bool = False,
    haptic_temperature: float = 1.0,
    alignment_enabled: bool = False,
    alignment_temperature: float = 1.0,
) -> int:
    """Task 4.1+4.2: 纯 MCMC 模式入口 — 不跑 RCB agent 主循环.

    single: 单链 N 步, standard 沙箱, 每 checkpoint_interval 步落盘
    multi:  K 链并行 (asyncio.gather 在 manifold 内部), 每链 N//K 步, R̂ 诊断
    """
    manifold = _init_mcmc_manifold(ws, task_id)
    if manifold is None or len(manifold._hyp) < 2:
        print(f"[mcmc-{mode}] manifold init failed or <2 hypotheses", file=sys.stderr)
        return 1

    _apply_mcmc_proposal_setup(
        ws, task_id, mode, manifold, (se3_enabled, haptic_enabled, alignment_enabled)
    )

    obs_list = _load_mcmc_observations(ws, mode)
    if not obs_list:
        return 1

    engine = _make_mcmc_engine(ws)
    flags = _proposal_flags(
        se3_enabled,
        se3_angle_sigma,
        haptic_enabled,
        haptic_temperature,
        alignment_enabled,
        alignment_temperature,
    )

    dims = (n_steps, n_chains, checkpoint_interval)
    return await _dispatch_mcmc_mode(
        manifold, engine, obs_list, task_id, ws, mode, dims, flags
    )
