"""协作对象的属性转发基类 (去 mixin 的唯一实现点).

AutoloopEngine 把方法族拆到 EngineObserve / EngineAct / EngineReflect 等协作对象,
但这些对象的方法体仍大量读写引擎字段 —— 之前每个文件各自复制一份
``__getattr__``/``__setattr__`` 与手工维护的 ``_OWN_ATTRS`` 名单, 结果是:

* 同一段转发逻辑在 10 个文件里各写一遍;
* ``_OWN_ATTRS`` 靠人肉同步, 实测 EngineControl 声明的 36 个名字里 36 个都不属于
  自己 (全是 EngineObserve 的方法), EnginePerceive / VisualInspect / RecallModes
  声明的 37 个名字全部属于 PlanCheck —— 名单一旦撞上真实引擎字段, 赋值就会静默
  落在协作对象实例上而不是引擎上, 产生"看起来改了、其实改在别处"的状态漂移。

本基类把这件事收敛成一处决策:

* 读: 未定义属性一律转发到 ``self.engine`` (用 ``object.__getattribute__`` 直达,
  避免 engine 就是 self 的测试替身场景递归);
* 写: 名字属于**本类定义面** (方法 / 类属性, 由 ``dir(type(self))`` 自动推导) 时落
  本对象实例 dict (这才是"测试覆写自身方法"的语义); 其余名字转发回 engine;
* 写自身 ``engine`` 或 engine is self 时直写实例 dict, 不递归。

自动推导取代手工名单是这里的关键: 名单不再需要人维护, 也就不存在"漏登记"
导致的静默漂移。
"""

from __future__ import annotations

from typing import Any, ClassVar


class EngineOwnershipViolation(RuntimeError):
    """协作对象写了不属于自己的单写者引擎字段."""


def _is_real_engine(obj: Any) -> bool:
    """engine 是否为真实 AutoloopEngine (而非测试替身).

    单写者约束只对真实引擎生效: 单元测试常把协作对象直接当引擎替身用
    (``EngineReflect(fake)`` 然后往它身上写状态), 这时按类名判定不误伤.
    用类名而不是 import, 避免 ``_facade`` 与 ``engine`` 的循环导入.
    """
    return any(cls.__name__ == "AutoloopEngine" for cls in type(obj).__mro__)


#: 单写者字段: 引擎字段 → 唯一允许经协作对象写它的类名.
#:
#: 由静态分析得出 (每个字段只被一个协作模块 ``self.x = ...`` 写入). 这些字段的写入
#: 归一个对象所有, 别的协作对象再写就会在第一次写时炸出来, 而不是留下"两处都能改"
#: 的隐形漂移. 未列在此的引擎字段 = 多写者字段 (见 ``SHARED_FIELDS``), 它们的共享
#: 是既定事实, 不加限制.
SINGLE_WRITER: dict[str, str] = {
    "_branch_incubator": "HypothesisLoop",
    "_budget": "CognitiveRunner",
    "_clarification_mgr": "EngineControl",
    "_consecutive_failures": "CognitiveRunner",
    "_consecutive_failures_by_type": "CognitiveRunner",
    "_current_hyp_id_for_plan": "CognitiveRunner",
    "_current_phase": "CognitiveRunner",
    "_current_plan_target_files": "EngineAct",
    "_current_run_phases": "CognitiveRunner",
    "_darwin_best_score": "CognitiveRunner",
    "_darwin_last_score": "CognitiveRunner",
    "_darwin_stagnation": "CognitiveRunner",
    "_deliverable_manifest": "CognitiveRunner",
    "_drift_info": "CognitiveRunner",
    "_episodic_replay_obj": "EngineObserve",
    "_evals_history": "CognitiveRunner",
    "_event_bus": "EngineControl",
    "_exec_converged": "EngineReflect",
    "_exec_cycling": "EngineReflect",
    "_exec_fp_history": "EngineReflect",
    "_execution_ledger": "CognitiveRunner",
    "_experiment_count_since_self_model_update": "EngineReflect",
    "_fp_result_ref": "EngineReflect",
    "_hypo_manifold": "EngineObserve",
    "_iteration": "CognitiveRunner",
    "_iteration_history": "CognitiveRunner",
    "_jepa_corpus_seen": "EngineReflect",
    "_jepa_predictor_cache": "EngineReflect",
    "_jepa_span_cache": "EngineReflect",
    "_jepa_st": "EngineReflect",
    "_kb": "EnginePerceive",
    "_last_completion_iter": "CognitiveRunner",
    "_last_context": "HypothesisLoop",
    "_last_execution_files": "EngineAct",
    "_last_execution_result": "EngineAct",
    "_last_failure_mode": "EngineReflect",
    "_last_good_hyp_id": "CognitiveRunner",
    "_last_good_hypothesis": "CognitiveRunner",
    "_last_good_plan": "CognitiveRunner",
    "_last_had_activity": "CognitiveRunner",
    "_last_hypothesis_blocks": "EngineObserve",
    "_last_hypothesis_confidence": "CognitiveRunner",
    "_last_hypothesis_evidence_strength": "CognitiveRunner",
    "_last_persona": "HypothesisLoop",
    "_last_plan_blocks": "EngineObserve",
    "_last_rule_hit_id": "EngineAct",
    "_last_run_failure_pattern": "CognitiveRunner",
    "_last_surprise_rel": "EngineReflect",
    "_last_validation": "EngineReflect",
    "_last_visual_base64": "VisualInspect",
    "_light_off_track_streak": "CognitiveRunner",
    "_max_iterations": "CognitiveRunner",
    "_max_refines": "CognitiveRunner",
    "_mcmc_accept_count": "EngineObserve",
    "_mcmc_cached_log_p": "EngineObserve",
    "_mcmc_current": "EngineObserve",
    "_mcmc_step_count": "EngineObserve",
    "_metacog_auditor": "EngineObserve",
    "_metacog_block_registry": "EngineObserve",
    "_metacog_completion_auditor": "EngineObserve",
    "_metacog_convergence_detector": "EngineObserve",
    "_metacog_last_audit": "HypothesisLoop",
    "_metacog_last_topology": "HypothesisLoop",
    "_metacog_method_registry": "EngineObserve",
    "_monitor_state": "CognitiveRunner",
    "_next_phase_hint": "CognitiveRunner",
    "_novelty_prev_nodes": "CognitiveRunner",
    "_novelty_prev_rejected": "CognitiveRunner",
    "_novelty_starved": "CognitiveRunner",
    "_objective": "CognitiveRunner",
    "_parent_run_id": "CognitiveRunner",
    "_perception": "EnginePerceive",
    "_persona_manager": "EnginePerceive",
    "_plan_check_history": "CognitiveRunner",
    "_plan_check_warnings": "CognitiveRunner",
    "_plan_store": "EngineControl",
    "_pmk_side_effect_iter": "EngineObserve",
    "_pre_execute_provenance_version": "CognitiveRunner",
    "_prev_exec_fingerprint": "EngineReflect",
    "_prev_exec_fp_src": "EngineReflect",
    "_prev_run_context": "CognitiveRunner",
    "_progress_task_id": "CognitiveRunner",
    "_provenance_logger": "CognitiveRunner",
    "_provenance_record": "CognitiveRunner",
    "_refine_count": "CognitiveRunner",
    "_refined_hypothesis": "CognitiveRunner",
    "_related_chain": "EngineObserve",
    "_repeat_exec_streak": "EngineReflect",
    "_run_goal_id": "CognitiveRunner",
    "_run_id": "CognitiveRunner",
    "_run_start_iso": "CognitiveRunner",
    "_run_start_time": "CognitiveRunner",
    "_scene_tag_extra_keywords": "CognitiveRunner",
    "_surprise_buckets": "EngineReflect",
    "_target_chains": "EnginePerceive",
    "_target_chains_built": "EnginePerceive",
    "_task_metrics": "CognitiveRunner",
    "_task_state_for_metrics": "CognitiveRunner",
    "_traj_run_ids": "EngineReflect",
    "_validate_window": "CognitiveRunner",
    "_wake_scheduler": "CognitiveRunner",
}

#: 既定多写者字段 (按阶段交接状态): 这些字段的共享是设计决定, 不是遗漏.
#: 列在这里是为了让"哪些状态被跨阶段共享"成为一处可读的决定, 而不是散落在代码里.
SHARED_FIELDS: frozenset[str] = frozenset({
    "_budget_degraded",
    "_current_prediction",
    "_episodic_writer",
    "_force_imaginate",
    "_grill_active",
    "_grill_turns",
    "_jepa_plan_inputs",
    "_last_author_prompt_len",
    "_last_blind_confidence",
    "_last_hypothesis",
    "_last_phase_unavailable",
    "_last_plan_failure",
    "_last_raw_hypothesis",
    "_last_reconstruct_disagree",
    "_last_selected_prediction",
    "_last_surprise",
    "_last_task_perf",
    "_last_traj_match_doc_id",
    "_last_traj_match_run_id",
    "_last_visual_context",
    "_plan_check_last_result",
    "_plan_check_patterns",
    "_rename_debt",
    "_rename_streak",
    "_should_stop",
    "_speculator_hint",
    "_traj_history",
})


class EngineFacade:
    """未定义属性读写转发到 ``self.engine`` 的协作对象基类."""

    #: 每个子类的"自有名字"缓存 —— dir() 结果按类缓存, 避免每次赋值都重新计算.
    _own_names_cache: ClassVar[dict[type, frozenset[str]]] = {}

    def __init__(self, engine: Any) -> None:
        object.__setattr__(self, "engine", engine)

    @classmethod
    def _own_names(cls) -> frozenset[str]:
        """本类**结构上拥有**的属性名 (方法 / 类属性 / 描述符)."""
        cached = EngineFacade._own_names_cache.get(cls)
        if cached is None:
            cached = frozenset(name for name in dir(cls) if name != "engine")
            EngineFacade._own_names_cache[cls] = cached
        return cached

    def __getattr__(self, name: str) -> Any:
        # object.__getattribute__ 直达 engine 实例属性, 避免 engine == self 时递归.
        return object.__getattribute__(self.engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "engine":
            object.__setattr__(self, name, value)
            return
        # 本类定义面: 赋值意图是覆写自身方法 (测试 mock), 落本对象实例 dict.
        if name in self._own_names():
            object.__setattr__(self, name, value)
            return
        # engine == self (测试替身) 时直写实例 dict 避免转发自递归.
        if self.engine is self:
            object.__setattr__(self, name, value)
            return
        owner = SINGLE_WRITER.get(name)
        if owner is not None and owner != type(self).__name__ and _is_real_engine(self.engine):
            raise EngineOwnershipViolation(
                f"{type(self).__name__} 不能写 {name!r}: 该字段的单写者是 {owner}. "
                "跨阶段共享状态请显式登记到 _facade.SHARED_FIELDS, 或由 owner 提供写接口."
            )
        setattr(self.engine, name, value)
