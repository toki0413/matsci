"""RCB MCMC 模式 — 依赖 rcb_cognition / rcb_utils."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

from huginn.cli.rcb_cognition import _init_hypothesis_manifold
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


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
    import random as _mcmc_random
    import types as _mcmc_types

    from huginn.metacog.hypothesis_manifold import Observation
    from huginn.runtime.engine_state import save_engine_state
    from huginn.security.sandbox import create_sandbox

    # 复用 RCB 路径的 manifold 初始化 — 优先从盘上加载, 没有就建 generic
    _instr_path = ws / "INSTRUCTIONS.md"
    _hypo_manifold = _init_hypothesis_manifold(
        ws=ws, task_id=task_id, checklist="",
        instructions=_instr_path if _instr_path.exists() else "",
        scan_text="", model=None, task_ctx="",
    )
    if _hypo_manifold is None or len(_hypo_manifold._hyp) < 2:
        print(f"[mcmc-{mode}] manifold init failed or <2 hypotheses",
              file=sys.stderr)
        return 1

    # SE(3): load cognitive_maps from engine_state, register with hypotheses.
    # ponytail: no cognitive_map -> se3_enabled=True safely degrades to fisher
    #   (because _has_structure returns False for all hypotheses).
    if se3_enabled:
        try:
            from huginn.metacog.structure_cognitive_map import (
                StructureCognitiveMap as _SCM,  # noqa: N814
            )
            from huginn.runtime.engine_state import load_engine_state as _les
            _est = _les(task_id, ws)
            _cmaps = getattr(_est, "cognitive_maps", {}) if _est else {}
            if _cmaps:
                # Map each cognitive_map to a hypothesis by matching h_id.
                # ponytail: simple 1:1 mapping by index; upgrade: explicit
                #   structure_id assignment via UI or hypothesis metadata.
                _h_ids = list(_hypo_manifold._hyp)
                for _i, (_mid, _mdict) in enumerate(_cmaps.items()):
                    if _i >= len(_h_ids):
                        break
                    try:
                        _cm = _SCM.from_engine_state_dict(_mdict)
                        _hypo_manifold.register_structure(
                            _h_ids[_i], _mid, _cm)
                    except Exception:
                        logger.debug("cognitive_map structure register skipped", exc_info=True)
                print(f"[mcmc-{mode}] SE(3) loaded {len(_hypo_manifold._structure_maps)} "
                      f"cognitive_map(s) for structure-guided proposal", flush=True)
            else:
                print(f"[mcmc-{mode}] SE(3) enabled but no cognitive_maps found, "
                      f"degrading to fisher", flush=True)
        except Exception as _e:
            print(f"[mcmc-{mode}] SE(3) cognitive_map load failed: {_e}, "
                  f"degrading to fisher", flush=True)

    # 触觉层: 从 .huginn/haptic_layers.json 加载力学属性, register 到 hypothesis.
    # ponytail: 文件不存在或空 → _haptic_layers 保持空, haptic_enabled=True 时
    #   _haptic_proposal 对所有 h_id 返回 None, 安全退化到 fisher.
    #   升级路径: VASP static / ML potential / Materials MP 查询结果写进这个文件.
    if haptic_enabled:
        _hap_path = ws / HUGINN_DIR_NAME / "haptic_layers.json"
        _n_hap = 0
        if _hap_path.exists():
            try:
                from huginn.metacog.haptic_property_layer import (
                    HapticPropertyLayer as _HPL,  # noqa: N814
                )
                _h_ids = list(_hypo_manifold._hyp)
                _raw = json.loads(_hap_path.read_text(encoding="utf-8"))
                # key 可以是 h_id 或结构 id, 优先 h_id 匹配, 否则按 index 回退
                for _i, _h_id in enumerate(_h_ids):
                    _d = _raw.get(_h_id)
                    if _d is None and _i < len(_raw):
                        _d = list(_raw.values())[_i]
                    if _d is None:
                        continue
                    try:
                        _layer = _HPL.from_dict(_d)
                        _hypo_manifold.register_haptic(_h_id, _layer)
                        _n_hap += 1
                    except Exception:
                        logger.debug("mcmc haptic layer register skipped", exc_info=True)
            except Exception as _e:
                print(f"[mcmc-{mode}] haptic load failed: {_e}, "
                      f"degrading to fisher", flush=True)
        if _n_hap > 0:
            print(f"[mcmc-{mode}] haptic loaded {_n_hap} layer(s) for "
                  f"haptic-guided proposal", flush=True)
        else:
            print(f"[mcmc-{mode}] haptic enabled but no layers in "
                  f"{_hap_path}, degrading to fisher", flush=True)

    # 对齐层: 从 .huginn/alignment_dataset.json 加载 (structure, haptic) 对,
    # 数据量 >= 10 时自动 fit AlignmentFunction, 注入 manifold 引导 proposal.
    # ponytail: 文件不存在 / 数据不足 / fit 失败 → _alignment_fn 保持 None,
    #   alignment_enabled=True 时 _alignment_proposal 全返 None, 安全退化 fisher.
    _alignment_dataset = None
    if alignment_enabled:
        _align_path = ws / HUGINN_DIR_NAME / "alignment_dataset.json"
        if _align_path.exists():
            try:
                from huginn.metacog.alignment import AlignmentFunction
                from huginn.metacog.alignment_dataset import AlignmentDataset
                from huginn.metacog.haptic_descriptor import HapticDescriptor
                from huginn.metacog.structure_descriptor import StructureDescriptor

                _alignment_dataset = AlignmentDataset.load(_align_path)
                _n_pairs = _alignment_dataset.count("structure", "haptic")
                if _n_pairs >= 10:
                    _af = AlignmentFunction(
                        StructureDescriptor(), HapticDescriptor(), min_samples=10)
                    _af.fit(_alignment_dataset)
                    if _af.ready:
                        _hypo_manifold.set_alignment_function(_af)
                        print(f"[mcmc-{mode}] alignment fitted on {_n_pairs} pairs, "
                              f"alignment-guided proposal enabled", flush=True)
                    else:
                        print(f"[mcmc-{mode}] alignment fit returned not-ready "
                              f"({_n_pairs} pairs), degrading to fisher", flush=True)
                else:
                    print(f"[mcmc-{mode}] alignment dataset has only {_n_pairs} pairs "
                          f"(need >=10), degrading to fisher", flush=True)
            except Exception as _e:
                print(f"[mcmc-{mode}] alignment load/fit failed: {_e}, "
                      f"degrading to fisher", flush=True)
        else:
            print(f"[mcmc-{mode}] alignment enabled but no dataset at "
                  f"{_align_path}, degrading to fisher", flush=True)

    # Surprise 检查: haptic + alignment 都 ready 时, 对每个有 structure+haptic
    # 的 hypothesis 查 surprise. score > 2.0 触发新 hypothesis 生成 + 数据回流.
    # ponytail: advisory only, 失败只 warn 不阻塞. model=None 时 trigger 走空.
    if alignment_enabled and _alignment_dataset is not None:
        try:
            _surprise_findings: list[tuple[str, float]] = []
            for _h_id in _hypo_manifold._hyp:
                _sc = _hypo_manifold.check_surprise(_h_id)
                if _sc is not None and _sc > 2.0:
                    _surprise_findings.append((_h_id, _sc))
            if _surprise_findings:
                print(f"[mcmc-{mode}] surprise detected on "
                      f"{len(_surprise_findings)} hypothesis(es)", flush=True)
                # 数据回流: 把当前 (structure, haptic) 对存入 dataset
                from huginn.metacog.haptic_descriptor import (
                    HapticDescriptor as _HD,  # noqa: N814
                )
                from huginn.metacog.structure_descriptor import (
                    StructureDescriptor as _SD,  # noqa: N814
                )
                _sd, _hd = _SD(), _HD()
                for _h_id, _score in _surprise_findings:
                    _h = _hypo_manifold._hyp.get(_h_id)
                    if _h is None or _h.structure_id is None:
                        continue
                    _cmap = _hypo_manifold._structure_maps.get(_h.structure_id)
                    _layer = _hypo_manifold._haptic_layers.get(_h_id)
                    if _cmap is None or _layer is None:
                        continue
                    with contextlib.suppress(Exception):
                        _alignment_dataset.add(
                            _sd.encode(_cmap), _hd.encode(_layer),
                            "structure", "haptic",
                            metadata={"h_id": _h_id, "surprise": _score})
                try:
                    _alignment_dataset.save(_align_path)
                except Exception as _e:
                    print(f"[mcmc-{mode}] dataset save failed: {_e}", flush=True)
        except Exception as _e:
            print(f"[mcmc-{mode}] surprise check failed: {_e}", flush=True)

    # 读 observations — 主循环 _iter_observations 跨轮累积, 这里从盘上恢复
    obs_path = ws / HUGINN_DIR_NAME / "observations.jsonl"
    obs_list: list = []
    if obs_path.exists():
        for _line in obs_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line:
                continue
            try:
                _o = json.loads(_line)
                obs_list.append(Observation(
                    name=_o["name"], value=_o["value"],
                    sigma=_o.get("sigma", 1.0),
                ))
            except Exception:
                logger.debug("observation line parse skipped", exc_info=True)
    if not obs_list:
        print(f"[mcmc-{mode}] no observations in {obs_path}, cannot run MCMC",
              file=sys.stderr)
        return 1

    # engine holder — 跟主循环 L1054 一致, 让 save_engine_state 能拉到 _mcmc_* 字段
    _engine = _mcmc_types.SimpleNamespace(
        _mcmc_current=None,
        _mcmc_rng=_mcmc_random.Random(
            int(os.environ.get("HUGINN_MCMC_SEED", "42"))),
        _mcmc_rng_state=None,
        _mcmc_accept_count=0,
        _mcmc_step_count=0,
        _mcmc_chains={},
        _iteration=0,
        workspace=ws,
        hypothesis_graph=None,
    )

    if mode == "single":
        # 单链: 一个 standard 沙箱 (8GB/4cpu/6h), MCMC 在主进程跑
        _sandbox = create_sandbox(profile="standard")
        print(f"[mcmc-single] sandbox profile=standard, "
              f"steps={n_steps}, ckpt_interval={checkpoint_interval}",
              flush=True)

        # resume: 尝试加载已有 checkpoint, 有就从断点继续
        from huginn.runtime.engine_state import load_engine_state
        _resume_state = load_engine_state(task_id, ws)
        if _resume_state is not None and _resume_state._mcmc_step_count > 0:
            _engine._mcmc_current = _resume_state._mcmc_current
            _engine._mcmc_accept_count = _resume_state._mcmc_accept_count
            _engine._mcmc_step_count = _resume_state._mcmc_step_count
            if _resume_state._mcmc_rng_state is not None:
                _engine._mcmc_rng.setstate(_resume_state._mcmc_rng_state)
            print(f"[mcmc-single] resume from step={_engine._mcmc_step_count} "
                  f"current={_engine._mcmc_current} "
                  f"accept={_engine._mcmc_accept_count}", flush=True)

        # 初始 current: resume 优先, 否则 abductive_inference, 最后随机
        current = _engine._mcmc_current
        if current is None:
            try:
                _abd = _hypo_manifold.abductive_inference(obs_list)
                current = _abd.h_id if _abd else None
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                current = None
            if current is None:
                current = _mcmc_random.Random(42).choice(list(_hypo_manifold._hyp))

        cached_log_p: float | None = None
        _start_step = _engine._mcmc_step_count + 1
        # P2-7: mcmc-single 也走温度退火 + 全局 proposal 混合, 与 mcmc-multi 对齐.
        # 默认 t_high=10, 几何退火到 temperature(=1.0). HUGINN_MCMC_NO_ANNEAL=1 关闭.
        _anneal = os.environ.get("HUGINN_MCMC_NO_ANNEAL", "0") != "1"
        _t_high = float(os.environ.get("HUGINN_MCMC_T_HIGH", "10"))
        _gpp = float(os.environ.get("HUGINN_MCMC_GLOBAL_PROPOSAL", "0.3"))
        for step in range(_start_step, n_steps + 1):
            prev = current
            T = 1.0
            if _anneal:
                T = _t_high * (1.0 / _t_high) ** (step / n_steps)
            current, cached_log_p = _hypo_manifold.mcmc_step(
                obs_list, current,
                rng=_engine._mcmc_rng,
                cached_log_p_current=cached_log_p,
                temperature=T,
                global_proposal_prob=_gpp,
                se3_enabled=se3_enabled,
                se3_angle_sigma=se3_angle_sigma,
                haptic_enabled=haptic_enabled,
                haptic_temperature=haptic_temperature,
                alignment_enabled=alignment_enabled,
                alignment_temperature=alignment_temperature,
            )
            if current != prev:
                _engine._mcmc_accept_count += 1
            _engine._mcmc_step_count += 1
            _engine._mcmc_current = current
            _engine._iteration = step

            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                try:
                    _engine._mcmc_rng_state = _engine._mcmc_rng.getstate()
                    save_engine_state(_engine, task_id, ws)
                    print(f"[mcmc-single] ckpt step={step} current={current} "
                          f"accept={_engine._mcmc_accept_count}", flush=True)
                except Exception as _e:
                    print(f"[mcmc-single] ckpt failed: {_e}", flush=True)

        _rate = _engine._mcmc_accept_count / _engine._mcmc_step_count if _engine._mcmc_step_count > 0 else 0.0
        print(f"[mcmc-single] done: total_steps={_engine._mcmc_step_count} "
              f"accept_rate={_rate:.3f}", flush=True)
        return 0

    # multi 模式
    n_per_chain = max(1, n_steps // n_chains)
    _cpu = os.cpu_count() or 1
    _max_concurrent = min(n_chains, _cpu)
    # 每链一个 standard 容器; n_chains > cpu_count 时实际是协程级切换
    _sandboxes = [create_sandbox(profile="standard") for _ in range(n_chains)]
    print(f"[mcmc-multi] {n_chains} sandboxes (profile=standard), "
          f"cpu_count={_cpu}, max_concurrent={_max_concurrent}, "
          f"steps_per_chain={n_per_chain}", flush=True)
    # ponytail: mcmc_multi_chain 内部 asyncio.gather 已并行, semaphore 没法
    #   注入 (要改 manifold 签名). n_chains > cpu_count 时是协程级切换不是
    #   真并行. 升级路径: ProcessPool + 每链独立进程, semaphore 限并发进程数.
    #   见 hypothesis_manifold.py L444 注释.

    def _on_chain_checkpoint(chain_id: int, state: dict) -> None:
        # 多链 checkpoint: 每链 state 存 _mcmc_chains, 整体落盘一次
        _engine._mcmc_chains[chain_id] = state
        _engine._mcmc_current = state.get("current")
        _engine._mcmc_accept_count = state.get("accept_count", 0)
        _engine._mcmc_step_count = state.get("step", 0)
        _engine._mcmc_rng_state = state.get("rng_state")
        try:
            save_engine_state(_engine, task_id, ws)
            print(f"[mcmc-multi] chain {chain_id} ckpt at step "
                  f"{state.get('step')}", flush=True)
        except Exception as _e:
            print(f"[mcmc-multi] chain {chain_id} ckpt failed: {_e}",
                  flush=True)

    result = await _hypo_manifold.mcmc_multi_chain(
        obs_list,
        n_chains=n_chains,
        n_steps_per_chain=n_per_chain,
        checkpoint_interval=checkpoint_interval,
        se3_enabled=se3_enabled,
        se3_angle_sigma=se3_angle_sigma,
        haptic_enabled=haptic_enabled,
        haptic_temperature=haptic_temperature,
        alignment_enabled=alignment_enabled,
        alignment_temperature=alignment_temperature,
        on_chain_checkpoint=_on_chain_checkpoint,
    )

    _r_hat = result.get("r_hat", float("nan"))
    print(f"[mcmc-multi] done: r_hat={_r_hat:.4f} "
          f"converged={result.get('converged')} "
          f"accept_rates={result.get('accept_rates')}", flush=True)
    return 0
