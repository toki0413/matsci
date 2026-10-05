"""EngineAct — AutoloopEngine 的 plan / execute 阶段方法族 协作对象.

从 engine.py 拆出 (P3 slim-down 续). 包含:
- _plan (假设 → 步骤, 含 PlanStore 落盘 + cost 确认门)
- _execute (按 mode 分派到 coder/workflow/explore/code_lab/skill/visual_inspect)
- _execute_coder / _execute_workflow / _execute_explore / _execute_skill
- _execute_dynamic_workflow (A5 并行 subtask) + _execute_dynamic_workflow_bandit (H2)
- _record_provenance, _try_evolved_fix
- _llm_chat (LLM 调用入口, 含 streaming + persona + thinking effort + GRILL 注入)

去 mixin 阶段4: 原 EngineActMixin(926 行/15 方法) 改为普通类 EngineAct。
引擎经组合持有 self._engine_actor = EngineAct(self), 保留同名薄委托方法
(_plan / _execute / _execute_coder / _llm_chat 等) → 既有 self.method() 调用点
(多个 mixin 共用 _llm_chat) 零改动。不再靠多继承把认知中枢堆成 god-class。

设计关键 (ponytail):
- 方法体大量读写引擎状态(字段+方法): _grill_active/_current_prediction/
  model/model_router → 用「全属性转发」: __getattr__ 把未定义属性读转发到 engine,
  __setattr__ 转发写。字段/方法与 M1-M3 一样留在引擎, 协作对象不复制状态。
- 防递归: __getattr__ 用 object.__getattribute__ 直达 engine 实例属性
  (engine==self 的 mock/selfcheck 场景不递归); __setattr__ 在 engine is self 时
  直写实例 dict 避免转发自递归.
- 对 engine.py 模块级符号 (helper 函数 / 常量) 用方法内 lazy import, 避免 circular
- 协作对象不额外持有业务状态 (除 engine 引用), 全部经转发走 engine
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

# 提示面升级阈值 (与 engine_reflect 同一旋钮): 由本对象在组装作者提示时**消费**,
# 故常量必须同源. engine_reflect 不反向 import 本模块, 无循环风险.
# with_retry: autoloop 的 LLM 调用此前**绕开**了统一重试层 (agent/streaming.py 有,
# 这条路径没有), 一次限流/网络抖动就被当成"阶段无产出" → redirect→pivot→停机
# (run74). 这里接上.
from huginn.autoloop.engine_reflect import _REPEAT_HARD_STREAK
from huginn.llm_retry import with_retry

logger = logging.getLogger(__name__)

# 报告 citation 台账容量: 每条一次真实 execute, 只留紧凑"数值面".
_EXEC_LEDGER_MAX = 24
_EXEC_LEDGER_ENTRY_CHARS = 600


class EngineAct:
    """plan / execute 阶段方法族协作对象.

    未定义的属性读写经 __getattr__/__setattr__ 转发到 self.engine —
    引擎字段/方法(如 _grill_active / _current_prediction / model / _llm_chat 等)
    不会被复制两份, 方法体零改动、行为完全等价.
    """

    def __init__(self, engine: Any) -> None:
        object.__setattr__(self, "engine", engine)

    def __getattr__(self, name: str) -> Any:
        # object.__getattribute__ 直达 engine 实例属性, 避免 engine==self 时递归
        return object.__getattribute__(self.engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "engine":
            object.__setattr__(self, name, value)
            return
        # engine==self (mock/selfcheck) 直写实例 dict 避免转发自递归; 否则转发回引擎
        if self.engine is self:
            object.__setattr__(self, name, value)
            return
        setattr(self.engine, name, value)

    async def _plan(
        self, hypothesis: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate a plan from hypothesis and persist it to PlanStore.

        以前只返回一个临时 dict, turn 结束就丢了. 现在往 PlanStore 落一份,
        跨会话可恢复, 用户也能 confirm/reject. PlanStore 不可用时退回老行为.
        """
        prompt = self._build_plan_prompt(hypothesis, context)
        try:
            # OAK 启发: 三阶段角色分工 — hypothesize 用 reasoning (强模型发散),
            # plan 用 planning (中档模型收敛), execute 不调 LLM 直接跑工具
            response = await self._llm_chat(
                prompt, persona_name="default", task="planning"
            )
            plan = self._parse_plan(response)
            # LLM 可达 (即便解析为空) → 不是"不可用": 清标记, 让空产出正常驱动
            # reflect 的重定向逻辑, 不被上一阶段的瞬时故障标记误伤.
            self._last_phase_unavailable = False
        except Exception as _exc:  # 防御: 尽力操作失败返回空
            from huginn.llm_retry import is_transient_error
            if is_transient_error(_exc):
                self._last_phase_unavailable = True
            logger.debug("best-effort op failed", exc_info=True)
            return None

        # H4: GRILL 退出检查 — LLM 输出 plan 时若含 "shared understanding" 确认标记
        # 说明用户已确认, 退出 grill 模式. 之前只进入不退出, LLM 永远带 grill 约束.
        # ponytail: 简单字符串匹配, 不上 LLM judge. ceiling: LLM 不说这个词就永远不退.
        if self._grill_active and response:
            _resp_lower = response.lower()
            if (
                "shared understanding" in _resp_lower
                or "shared understanding reached" in _resp_lower
                or "confirmed decisions" in _resp_lower
            ):
                self._grill_active = False
                self._grill_turns = 0
                logger.info("GRILL mode exited: shared understanding confirmed")

        if not plan:
            return None

        # B: 硬路由 — 根据上下文信号覆盖 LLM 的 mode 选择
        plan = self._override_plan_mode(plan)

        # KRCL 启发: 反向校验 plan 能否达成 hypothesis, 失败反馈 LLM 重生成
        # 单 LLM 反向校验, 最多 1 次重试, 失败不阻塞 (标 warning 继续)
        plan = await self._plan_check_and_refine(plan, hypothesis, context)

        # JEPA 阶段2-0: 把 plan 的 expected_prediction 同步为 JEPA prediction buffer,
        # 使 AutoloopEngine 的 validate 阶段也能采集 plan→actual 配对 (与 cognitive_loop
        # L2323 行为对齐). 真实模型常把数值预测写进 description 而非结构化
        # expected_prediction 键, 故缺失时兜底 description, 保证采集对真实输出触发;
        # 两者皆空则保持空, validate 不采, 与原行为一致.
        self._current_prediction = (
            (plan.get("expected_prediction") or plan.get("description") or "").strip()
        )

        # JEPA 方案① 结构化计划槽: 把 plan 时已知的输入/公式以 PLAN_SLOTS 块追加到
        # prediction 文本(现行 span predictor 按行切 span, 零改动消费), 并暂存槽
        # 供 _record_jepa_pair 落库 + 泄漏检测。仅方法级目标会有 prediction_inputs。
        try:
            from huginn.jepa_slots import append_slots
            _inputs = plan.get("prediction_inputs") or []
            _formula = plan.get("plan_formula", "")
            if _inputs:
                self._current_prediction = append_slots(
                    self._current_prediction, _inputs, _formula
                )
                self._jepa_plan_inputs = {"inputs": _inputs, "formula": _formula}
        except Exception:  # noqa: BLE001 — 槽是增量, 失败不阻塞
            logger.debug("[jepa-slots] engine_act slot append failed", exc_info=True)

        # 意图口径 (S2) 输入: 缓存本轮 plan 声明的目标文件集, 供 _learn 的
        # _apply_strict_scope 判"改动是否偏离本轮意图". 声明缺失 (LLM 未输出
        # FILES:) → 空表 → 意图口径 no-op (零回归). 与合规口径 (S1) 独立开关.
        self._current_plan_target_files = list(plan.get("target_files") or [])

        # 落 PlanStore: 创建 plan → cost 确认门 → confirm/reject
        plan_store = self._get_plan_store()
        if plan_store is None:
            return plan

        try:
            from huginn.autoloop.plan_store import PlanStep

            steps = [
                PlanStep(
                    id="step_0",
                    description=plan.get("description", ""),
                    tool=plan.get("mode", ""),
                    target_files=list(plan.get("target_files") or []),
                )
            ]
            persisted = plan_store.create_plan(
                objective=hypothesis,
                steps=steps,
                auto_confirm=False,
                metadata={"mode": plan.get("mode", ""), "source": "autoloop"},
            )

            # cost 确认门: None = 不用问 (非高成本 mode / manager 不可用),
            # 直接放行; 显式拒绝才拦. bool 和字符串都兼容 (测试 mock 常传 bool)
            # H4: reject_tokens 从 PhaseRegistry extra 取 (toggle off 回退 hardcode tuple)
            from huginn.harness.phase_spec import get_phase_extra
            _reject_tokens = get_phase_extra("_plan", "reject_tokens", [
                "no", "n", "cancel", "reject", "decline", "stop", "abort",
            ])
            answer = await self._maybe_clarify("plan", plan)
            if answer is None:
                should_confirm = True
            elif isinstance(answer, bool):
                should_confirm = answer
            elif isinstance(answer, str):
                should_confirm = answer.lower().strip() not in tuple(_reject_tokens)
            else:
                should_confirm = bool(answer)

            if should_confirm:
                plan_store.confirm_plan(persisted.id)
                plan_store.mark_executing(persisted.id)
            else:
                plan_store.reject_plan(persisted.id, reason="user declined")
                return None

            plan["plan_id"] = persisted.id
        except Exception as e:
            logger.warning("plan store persistence failed: %s", e)
            # PlanStore 挂了不阻塞执行, 退回老的纯 dict 路径

        return plan

    #: 计算/预测指令词 (英文 + 中文). autoloop 的真实 objective 多为中文
    #: ("写并运行…统计均方违规, 求 N_c(w)…拟合 beta"), 只认英文动词会让内建
    #: 执行路径永不触发 → execute 落到 explore 空转, report 再拿现编数字当证据.
    _NUMERIC_VERBS = (
        "compute", "calculate", "predict", "evaluate",
        "计算", "求解", "算出", "核算", "拟合", "统计", "测量", "预测", "评估",
    )
    #: 公式/数值迹象 (中文侧). 中文目标常无 ASCII 运算符, 但这些词已足以
    #: 说明是定量任务 (与英文侧 `[/^*+=()]` 等价).
    _FORMULA_HINTS = ("拟合", "均方", "误差", "斜率", "阈值", "方差", "相关系数")

    def _is_deterministic_numeric(self, description: str) -> bool:
        """启发式: 该项是否为"确定性数值计算"目标(供内建 probe 执行兜底).

        平衡点落地: execute 只对这类明确可算的目标主动生成 probe, 不打扰开放探索任务。
        保守: 需同时含 数字 + 计算/预测指令词 + 公式迹象, 缺一不触发。
        中英双语: 中文动词/中文数学词同样算命中, 否则中文 objective 永远走不到内建执行。
        """
        import re as _re

        if not isinstance(description, str) or not description.strip():
            return False
        d = description.strip()
        if not _re.search(r"\d", d):
            return False
        low = d.lower()
        if not any(v in low for v in self._NUMERIC_VERBS):
            return False
        return bool(_re.search("[/^*+=()]", d)) or any(h in d for h in self._FORMULA_HINTS)

    async def _request_numeric_probe(self, description: str) -> str:
        """平衡点·内建执行: 让 LLM 只产出能算出数值的纯 python, harness 负责运行取数.

        不信任该代码的科学可信度(那是 validate/裁决层的事), 只确保"真跑通、打印了数字"。
        返回可运行脚本; 失败/含危险调用 → 空串, 触发方回落原分派(不回归)。
        """
        import re as _re

        prompt = (
            "Given the following quantitative task, output ONLY a short, self-contained "
            "python snippet (no explanation, no surrounding text) that computes the requested "
            "numeric quantity and prints the final numeric value on stdout. Express the stated "
            "physical/math relation exactly; do not fabricate values.\n\nTASK: " + description
        )
        try:
            raw = await self._llm_chat(prompt, model=self.verification_model)
        except Exception:  # — LLM 不可用/超时则拿不到探针代码, 回落空探针, 不阻塞
            return ""
        m = _re.search(r"```python\s*(.*?)```", raw, _re.S) or _re.search(
            r"```\s*(.*?)```", raw, _re.S
        )
        code = (m.group(1) if m else raw).strip()
        if not code:
            return ""
        # 安全边界: 拒绝对外部 shell/系统有副作用的危险调用
        if _re.search(
            r"\b(import\s+(os|sys|subprocess|builtins)|from\s+(os|sys|subprocess)|__import__|\beval\s*\(|\bexec\s*\()",
            code,
        ):
            return ""
        return code

    # 计算实验意图词 (命题无关): 目标是"要真跑一段数值实验/仿真/训练/扫描"的,
    # 就让书生在 Code Lab 亲手写并真跑, 而不是只产出一段抄来的闭式或空转.
    _CODE_EXPERIMENT_HINTS = (
        "实验", "仿真", "模拟", "训练", "扫描", "网格", "留出", "泛化",
        "探针", "基准", "收敛", "灵敏度",
        "experiment", "simulat", "train", "scan", "sweep", "generaliz",
        "held-out", "holdout", "probe", "benchmark", "convergence",
    )

    def _is_code_experiment(self, text: str) -> bool:
        """启发式: 该项是否为"需书生亲手写代码真跑的计算实验"目标 (命题无关).

        与 ``_is_deterministic_numeric``(平凡闭式) 互补: 这里指开放的计算实验 ——
        多步数值流程(训练/扫描/网格/统计), 该进 Code Lab 沙箱真跑, 而不是抄一段
        闭式公式糊过去. 只认"实验意图"词, 不绑定任何具体命题; 中英双语.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        low = text.lower()
        return any(h in low for h in self._CODE_EXPERIMENT_HINTS)

    async def _request_code_lab_experiment(
        self, goal: str, guards: dict[str, Any] | None = None,
        repair_hint: str = "", prev_code: str = "", focus: str = "",
    ) -> str:
        """平衡点·内建执行: 让书生亲手写一段 Code Lab ``run(cfg)`` 实验代码.

        命题无关: 提示词来自 ``code_lab.build_author_prompt`` 的单一契约. 这里只取
        代码, 不评科学可信度(那是 validate/裁决层的事). 失败返回空串 → 回落原分派.
        ``repair_hint`` 非空时把沙箱真实报错回灌, 并附上 ``prev_code``(上一版失败
        代码), 让书生在此基础上做**最小改动**修 bug, 而不是凭空重写再犯同一个错.
        ``focus`` 是本轮可变的聚焦文本(见 ``_build_codelab_focus``), 防止作者提示
        因只喂恒定 goal 而逐字节冻结.
        """
        from huginn.research.code_lab import (
            build_author_prompt,
            extract_code,
            load_scaffold,
        )

        guard_block = "\n".join(
            f"- {g}" for g in ((guards or {}).get("prompt_guards") or [])[:8]
        )
        # 任务脚手架由环境变量声明(命题资产, 非平台内核): 不给就是命题无关的默认骨架.
        import os as _osc
        scaffold = load_scaffold(_osc.environ.get("HUGINN_CODELAB_SCAFFOLD", ""))
        prompt = build_author_prompt(
            goal, scaffold=scaffold, guard_block=guard_block,
            repair_hint=repair_hint, prev_code=prev_code, focus=focus,
        )
        try:
            raw = await self._llm_chat(prompt, model=self.verification_model)
        except Exception as e:  # — LLM 不可用/超时 → 拿不到实验代码, 回落, 不阻塞
            import os as _ose
            if _ose.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
                logger.warning("[code-lab-author] LLM 调用失败: %r", e)
            return ""
        code = extract_code(raw or "")
        import os as _ose
        if _ose.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
            logger.warning(
                "[code-lab-author] raw_len=%d extracted_len=%d prompt_len=%d",
                len(raw or ""), len(code), len(prompt),
            )
        return code

    def _run_code_lab(
        self, code: str, guards: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """在 Code Lab 安全沙箱真跑书生的实验代码; 无证据返回 (None, 原因).
        返回原因让上层能把真实报错回灌书生重写 (自修复), 不伪造.

        ``timeout_s`` 非空时覆盖 env 默认超时 —— D6① 用它把单片沙箱超时封顶到
        剩余挂钟预算, 使单次尝试不可能烧穿预算 (None = 用 `HUGINN_CODELAB_TIMEOUT_S`).
        """
        import os as _os

        from huginn.research.code_lab import load_scaffold, sandbox_run

        timeout = (
            float(timeout_s) if timeout_s is not None
            else float(_os.environ.get("HUGINN_CODELAB_TIMEOUT_S", "600"))
        )
        extra = tuple((guards or {}).get("imports_whitelist_extra") or ())
        aliases = (guards or {}).get("cfg_aliases") or None
        scaffold = load_scaffold(_os.environ.get("HUGINN_CODELAB_SCAFFOLD", ""))
        res, reason = sandbox_run(
            code, {"seed": 0}, timeout=timeout,
            imports_whitelist_extra=extra, cfg_aliases=aliases,
            scaffold=scaffold,
        )
        import os as _osd
        if _osd.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
            _ok = None if res is None else res.get("success")
            _nobj = 0 if res is None else len(res.get("objectives", {}) or {})
            logger.warning("[code-lab-run] res_none=%s success=%s nobj=%d reason=%r",
                           res is None, _ok, _nobj, (reason or "")[:200])
        if res is None:
            logger.info("code_lab 执行未产出证据(不伪造, 回落原分派): %s", reason)
            return None, reason or "执行未产出证据"
        return res, ""

    def _build_codelab_focus(self, description: str) -> str:
        """v11 反冻结: 给作者 prompt 注入**本轮可变**的聚焦文本.

        作者提示若只喂恒定 ``_objective``(研究总目标), 每轮逐字节相同 → 同一问题
        被反复问, 执行输出恒同(run47 实测 prompt_len 恒 5379, nobj 恒 36), 执行层
        零新信息. 这里把随迭代演进的当前假设(``_last_hypothesis``)与本轮实验步骤
        (plan ``description``)并进来, prompt 遂不再冻结. 无可用文本时返空串, 行为
        退回旧路径(向后兼容).
        """
        parts: list[str] = []
        # v12 提示面升级 (B1): engine_reflect 检测到"本轮执行指纹与上一轮完全一致"
        # (无效重跑) 并累计 _repeat_exec_streak. 该软提示原本只进假设生成提示, 到不了
        # **真正写实验**的作者提示 → 指纹照旧、循环撞收敛提前离场 (run50 实测 streak
        # 1-4 指纹恒同). 这里**按 streak 现算**强制变异令并注入作者提示, 附上一轮真实
        # 结果(_prev_exec_fp_src), 逼书生改实验族/参数而非原样重跑. 无独立标志状态机.
        _streak = int(getattr(self, "_repeat_exec_streak", 0) or 0)
        _cycling = bool(getattr(self, "_exec_cycling", False))
        if _streak >= _REPEAT_HARD_STREAK or _cycling:
            _prev = str(getattr(self, "_prev_exec_fp_src", "") or "")
            # 两种病态共用一条硬约束, 但病因不同, 措辞要分开 —— 否则给作者看错方向.
            _why = (
                "【强制变异·硬约束】最近数轮实验在**两种等价结果间来回换**"
                "(A,B,A,B…), 无新信息 (=未推进研究). "
                if _cycling and _streak < _REPEAT_HARD_STREAK
                else "【强制变异·硬约束】上一轮真实执行结果与更早一轮指纹完全一致"
                     "(=无效重跑, 未推进研究). "
            )
            parts.append(
                _why
                + "本轮实验**必须**至少改动一项结构: "
                "更换 family/约束族、改扫描范围或步长、改网络架构/宽度集合、"
                "改优化器或 seeds. 禁止原样重跑上一配置."
                + (("上一轮真实结果(你必须产出与它不同的数值):" + _prev[:500])
                   if _prev else "")
            )
        _hyp = str(getattr(self, "_last_hypothesis", "") or "").strip()
        if _hyp:
            parts.append(f"当前待检验假设: {_hyp[:600]}")
        _step = (description or "").strip()
        if _step and _step not in _hyp:
            parts.append(f"本轮实验步骤: {_step[:600]}")
        return "\n".join(parts)

    async def _execute_code_lab(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """mode=code_lab: 书生亲写一段计算实验 → Code Lab 沙箱真跑 → 证据.

        与闭式 probe 分层: 闭式只认平凡数值片段; 这里认"实验意图"并允许更长时/
        更多依赖(仍受沙箱白名单约束). 失败返回 success=False, 由 validate 层裁决.
        沙箱真报错时把报错回灌书生重写 (命题无关·通用修 bug), 最多 N 轮自修复.
        """
        # 优先用研究目标(全文本, 命题无关的真问题), plan 步骤只作兜底 —— plan 描述
        # 常带 "FILES:/SKILL:" 之类执行噪声, 不适合喂给"实验作者"提示.
        goal = str(getattr(self, "_objective", "") or "") or description
        # v11 反冻结: goal 恒定时作者提示逐字节冻结, 用本轮可变 focus 解冻.
        focus = self._build_codelab_focus(description)
        code = await self._request_code_lab_experiment(goal, focus=focus)
        if not code:
            import os as _ose
            if _ose.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
                logger.warning("[code-lab] 未产出可解析实验代码 → 回落; goal[:80]=%r",
                               goal[:80])
            return {"mode": "code_lab", "status": "failed", "success": False,
                    "error": "书生未产出可解析的实验代码"}
        import os as _os
        max_repairs = int(_os.environ.get("HUGINN_CODELAB_REPAIR_ATTEMPTS", "3"))
        last_err = ""

        # D1: 统一 deadline 原语 — 每次尝试前查一次, 已耗尽就不再起新尝试
        # (最多再多跑一个已在飞的尝试). 修复循环每次 ``_run_code_lab`` 最长可烧满
        # ``HUGINN_CODELAB_TIMEOUT_S`` (默认 900s), 最多 ``max_repairs+1`` 次 → 单轮
        # 可越过挂钟上限近一小时 (run56 实测). 纯时间边界, 不做科学判断; fail-open.
        # 旧私有 ``_wall_clock_expired`` 已删除: 时间口径下沉到 ``_budget_exhausted``
        # (且按本 run goal id 判定, 旧实现用全局 get_active() 会跨 run 串台).
        #
        # D6 (D-slice 同款, execute 侧): 把"修复循环"切成细粒度 slice, 不设硬时长.
        #   核心 slice (attempt 0) 必跑, 但其沙箱超时**封顶到剩余预算** → 单片越不了界;
        #   可选 slice (修复重写 attempt>=1) 启动前查可负担性, 门槛自校准 = max(下限,
        #   上一片实测挂钟). 非长程/关开关时两函数 fail-open → 逐字节等价旧行为.
        _prev_slice_cost = 0.0  # 上一片 (上一次尝试: 沙箱 + 其后作者 LLM) 的实测挂钟
        # D6 钩子: 长程宿主 (Engine/engine_control) 提供预算切片三函数; 短程宿主 /
        # 离线 stub 无这三个属性 → None, 门与封顶整体 fail-open, 行为 100% 不变
        # (与 branch_incubator 的 fail-open 一致). getattr 兜底避免 stub 上 AttributeError.
        _attempt_timeout = getattr(self, "_codelab_attempt_timeout", None)
        _repair_affordable = getattr(self, "_codelab_repair_affordable", None)
        _slice_budget = getattr(self, "_codelab_slice_budget", None)
        for attempt in range(max_repairs + 1):
            if self._budget_exhausted():
                last_err = "挂钟预算耗尽, 中止 code_lab 修复循环"
                logger.warning(
                    "code_lab 修复循环: 挂钟预算已耗尽, 停止第 %d 次尝试", attempt + 1
                )
                break
            # D6② 可选 slice 门: 修复重写启动前查可负担性 (自校准门槛).
            if (
                attempt > 0
                and callable(_repair_affordable)
                and not _repair_affordable(_prev_slice_cost)
            ):
                _rem = _slice_budget() if callable(_slice_budget) else None
                self._emit_control_trace(
                    "code_lab_slice_skip",
                    f"slice=repair#{attempt}: budget<max(min,{_prev_slice_cost:.0f}s) "
                    f"remaining={-1 if _rem is None else round(_rem)}s",
                    action="skip",
                )
                logger.warning(
                    "code_lab 修复循环: 剩余预算不足以再跑一轮修复(≈%.0fs), 收尾",
                    _prev_slice_cost,
                )
                break
            _t_slice = time.monotonic()
            res, reason = self._run_code_lab(
                code,
                timeout_s=_attempt_timeout() if callable(_attempt_timeout) else None,
            )
            _prev_slice_cost = time.monotonic() - _t_slice  # D6 自校准: 实测本片
            if res is not None:
                return {
                    "mode": "code_lab",
                    "status": "completed",
                    "success": bool(res.get("success", True)),
                    "result": res.get("summary", {}),
                    "objectives": res.get("objectives", {}),
                    "script": code,
                    "reproducible": True,
                }
            last_err = reason
            # 观测: 超时是"算力预算"问题而非代码 bug, 单独记 trace, 便于统计
            # "被超时饿死的迭代占比" (run59/60 实测三组均被 900s 超时饿死).
            if ("超时" in (reason or "")) or ("timeout" in (reason or "").lower()):
                self._emit_control_trace(
                    "code_lab_timeout",
                    f"attempt={attempt + 1}/{max_repairs + 1} {str(reason)[:120]}",
                    action="advisory_hint",
                )
            if attempt < max_repairs:
                # D6 自校准: 修复重写的作者 LLM 调用也是下一个 slice 的成本, 计入
                # 实测 —— 否则低估下一片 (run79 超支主要来自作者调用, 非沙箱).
                _t_auth = time.monotonic()
                repaired = await self._request_code_lab_experiment(
                    goal, repair_hint=last_err, prev_code=code, focus=focus
                )
                _prev_slice_cost += time.monotonic() - _t_auth
                if not repaired:
                    break
                code = repaired
        logger.warning(
            "code_lab 沙箱执行经 %d 轮修复仍未产出证据(不伪造): %s",
            max_repairs + 1, last_err,
        )
        # 诊断: 把最后一版(仍失败的)书生代码落日志, 便于对症改 harness/提示, 不伪造证据.
        # 用 warning 级: 默认 root logger 无 handler, info 级看不到失败代码.
        logger.warning("code_lab 最后一版书生代码:\n%s", code[:2000])
        return {"mode": "code_lab", "status": "failed", "success": False,
                "error": f"Code Lab 执行未通过沙箱校验: {last_err}", "script": code}

    async def _execute(self, plan: dict[str, Any], context: dict[str, Any]) -> Any:
        """Execute the plan using the appropriate sub-engine."""
        mode = plan.get("mode", "coder")
        description = plan.get("description", "")
        import os as _osd
        if _osd.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
            try:
                _obj = str(getattr(self, "_objective", "") or "")
                logger.warning(
                    "[exec-route] mode=%r desc[:80]=%r obj_len=%d "
                    "is_exp_desc=%s is_exp_obj=%s is_det_desc=%s",
                    mode, (description or "")[:80], len(_obj),
                    self._is_code_experiment(description),
                    self._is_code_experiment(_obj),
                    self._is_deterministic_numeric(description),
                )
            except Exception:  # — 原因: 路由诊断日志失败不影响执行, 忽略
                pass

        # 方案1·攻 execute (2026-09-11 A线根因后半段 + 平衡点落地):
        # ① plan 已带"可运行数值脚本片段" → 直接真实执行成证据;
        # ② plan 无片段但目标是"确定性数值计算" → harness 主动请求一段最小 probe 并运行,
        #    让"产出执行证据"从模型的 privilege 变成 execute 的内建动作 (执行 on-code,
        #    裁决仍由 validate 的诚实层负责)。两步都 parse 不到可执行片段 → 回落原分派, 不回归。
        try:
            _snip = self._extract_run_snippet(plan)
            if _snip:
                _cout = self._run_snippet_to_output(_snip)
                if _cout:
                    result = {
                        "mode": "closed_form_exec",
                        "status": "completed",
                        "success": True,
                        "result": _cout.strip(),
                        "script": _snip,
                        "reproducible": True,
                    }
                    self._record_provenance("closed_form_exec", plan, result)
                    self._last_execution_result = {
                        "_tool_name": "closed_form_exec",
                        "_tool_input": plan,
                        "result": result,
                    }
                    logger.info(
                        "execute closed-form fast-path: plan 内含可运行数值片段, 已真实执行 → evidence"
                    )
                    return result
            # 方案2·接 Code Lab (2026-09-27 割裂感根因): 目标是"计算实验类"时, 让书生
            # **亲手写**一段 run(cfg) 实验代码, 在 Code Lab 安全沙箱真跑 → 结构化
            # objectives 作证据. 与闭式 probe 分层: 闭式只认平凡数值片段, 这里认开放
            # 实验意图(训练/扫描/探针...), 允更长时/更多依赖. 命题无关: 只按目标的实验
            # 意图词触发, 不绑定任何具体命题. 失败即回落原分派, 不回归.
            # 顺序: 实验意图优先于平凡 probe —— 否则描述里的 "predict"/"(1)" 会让
            # _is_deterministic_numeric 先命中, 把真实验目标塞进"抄一段闭式"的窄路,
            # probe 跑不出数就再落到 coder 空转 (2026-09-27 run7 实测: 0 tool_calls).
            # 二者级联(if 而非 elif): Code Lab 失败仍可退到 probe, 不互相遮蔽.
            if self._is_code_experiment(description) or self._is_code_experiment(
                str(getattr(self, "_objective", "") or "")
            ):
                _lab = await self._execute_code_lab(description, context)
                if _lab.get("success"):
                    self._record_provenance("code_lab", plan, _lab)
                    self._last_execution_result = {
                        "_tool_name": "code_lab",
                        "_tool_input": plan,
                        "result": _lab,
                    }
                    logger.info(
                        "execute code_lab fast-path: 书生亲写实验 → Code Lab 沙箱真跑 → evidence"
                    )
                    return _lab
            # 独立判定: plan 片段若没跑出数, 仍尝试 objective 探针(平凡确定性数值).
            if self._is_deterministic_numeric(description) or self._is_deterministic_numeric(
                str(getattr(self, "_objective", "") or "")
            ):
                # 以可靠文本源为准: plan.description 在 run_cognitive 路径可能为空,
                # 用 objective(确定性全文本)兜底判定并生成 probe。
                _src = description if self._is_deterministic_numeric(description) else str(
                    getattr(self, "_objective", "") or ""
                )
                _probe = await self._request_numeric_probe(_src)
                if _probe:
                    _cout = self._run_snippet_to_output(_probe)
                    if _cout:
                        result = {
                            "mode": "probe_exec",
                            "status": "completed",
                            "success": True,
                            "result": _cout.strip(),
                            "script": _probe,
                            "reproducible": True,
                        }
                        self._record_provenance("probe_exec", plan, result)
                        self._last_execution_result = {
                            "_tool_name": "probe_exec",
                            "_tool_input": plan,
                            "result": result,
                        }
                        logger.info(
                            "execute builtin probe: 确定性数值目标 → harness 内建生成并执行 probe → evidence"
                        )
                        return result
        except Exception:  # 防御: 探针快路失败转入通用执行
            logger.debug("execute builtin-probe fast-path failed (fall through)", exc_info=True)

        # 非阻塞诊断(2026-09-11): real-loop 里 probe 屡不触发, 这里永久留一口子
        # 记录判定输入, 便于定位为何没走内建执行(不等重跑才猜)。
        try:
            _det_desc = self._is_deterministic_numeric(description)
            _det_obj = self._is_deterministic_numeric(str(getattr(self, "_objective", "") or ""))
            if _det_desc or _det_obj:
                logger.info(
                    "[execute-probe] 判定到确定性目标但未产出探针证据: "
                    "det_desc=%s det_obj=%s, desc[:40]=%r, obj[:40]=%r, _tool=%r",
                    _det_desc, _det_obj,
                    (description or "")[:40], (str(getattr(self, "_objective", "") or ""))[:40],
                    plan.get("mode"),
                )
        except Exception:  # — execute 分流探针失败则继续走通用执行, 不阻断
            pass

        # H4: toggle on 时从 PhaseRegistry 取 dispatch_table 替代 hardcode if/elif
        # ponytail: dispatch_table 存 [method_name, arg_mode], arg_mode 决定传
        # plan 还是 description (executor 签名不统一, 不改签名最小 diff).
        # 升级路径: 统一所有 executor 签名为 (plan, context), 去掉 arg_mode.
        from huginn.harness.phase_spec import get_phase_dispatch_table
        dispatch = get_phase_dispatch_table()
        if dispatch is not None:
            entry = dispatch.get(mode)
            if entry is None:
                raise ValueError(f"Unknown plan mode: {mode}")
            method_name, arg_mode = entry[0], entry[1]
            executor = getattr(self, method_name)
            arg = plan if arg_mode == "plan" else description
            result = await executor(arg, context)
            # workflow 失败走 evolved_fix (原 hardcode 逻辑保留)
            if mode == "workflow" and isinstance(result, dict) and not result.get("success", True):
                result = (
                    await self._try_evolved_fix(mode, description, result) or result
                )
        elif mode == "coder":
            # Use CoderRunner to modify code
            result = await self._execute_coder(description, context)
        elif mode == "workflow":
            # Use WorkflowEngine to run computational pipeline
            result = await self._execute_workflow(description, context)
            # On failure, try applying a learned heuristic fix before giving up
            if isinstance(result, dict) and not result.get("success", True):
                result = (
                    await self._try_evolved_fix(mode, description, result) or result
                )
        elif mode == "dynamic_workflow":
            # A5: agent 写的并行 subtask 脚本, orchestrator 并发跑
            result = await self._execute_dynamic_workflow(plan, context)
        elif mode == "explore":
            # Use ExplorationOrchestrator to search design space
            result = await self._execute_explore(description, context)
            # 空转兜底: 未注册设计空间时 explore 只回一个 0-branch 空 pod
            # (0.0s, n_explored=0, 无 result) — 无证据却有"成功"外形, report 遂把
            # LLM 现编的数字当 Results 写出去. 目标是"确定性数值计算"(中英动词都认)
            # 时回落 coder 真写码真执行 (Write+Bash), 不空转.
            if (
                isinstance(result, dict)
                and (
                    result.get("success") is False
                    or (
                        not result.get("result")
                        and int(result.get("n_explored", 0) or 0) == 0
                    )
                )
                and (
                    self._is_deterministic_numeric(description)
                    or self._is_deterministic_numeric(
                        str(getattr(self, "_objective", "") or "")
                    )
                )
            ):
                logger.info("explore 空转 (无证据) → 回落 coder 真实执行")
                result = await self._execute_coder(description, context)
                mode = "coder"
        elif mode == "code_lab":
            # 书生亲写一段计算实验 → Code Lab 安全沙箱真跑 → 结构化证据.
            result = await self._execute_code_lab(description, context)
        elif mode == "skill":
            # Run a pre-built composite skill pipeline
            result = await self._execute_skill(plan, context)
        elif mode == "visual_inspect":
            # Path C: interactive visual inspection using existing visual tools
            # A3 + 批次 D: consistency_check 默认开启 (闭环), 除非显式关闭.
            # 精修闭环要求每次 zoom 都做 re-ask 稳定性自检, 不再依赖 agent 记得声明.
            consistency = not (
                context.get("consistency_check") is False
                or "no consistency" in plan.lower()
                or "consistency_check=false" in plan.lower()
            )
            result = await self._execute_visual_inspect(
                description, context, consistency_check=consistency
            )
        else:
            raise ValueError(f"Unknown plan mode: {mode}")

        # provenance: 记一次 tool call, mode 当工具名, plan 当输入参数
        self._record_provenance(mode, plan, result)
        # Step 8: 力学结果自动收集到 AlignmentDataset (失败不阻塞主循环)
        self._collect_alignment_pair(result, tool_name=mode)
        # 缓存给 _build_plan_prompt 的 pipeline suggest_next 用
        self._last_execution_result = {
            "_tool_name": mode,
            "_tool_input": plan,
            "result": (
                result if isinstance(result, dict) else {"value": str(result)[:500]}
            ),
        }
        # P21/dsh: 记录本轮触碰的文件面, 供 _is_related_chain 判断相关任务链
        # (高重叠 → 下轮降级引导). context 无 changed_files 时记空, 不触发门控.
        self._last_execution_files = context.get("changed_files", [])
        return result

    def _record_provenance(
        self, tool_name: str, input_params: dict[str, Any], output: Any
    ) -> None:
        """往当前 run 的 provenance record 追加一次 tool-call 快照.

        run() 启动时建好 self._provenance_record; 没建 (比如单测里直接调
        _execute) 就跳过, 不强求调用方先 setup. provenance 是 best-effort,
        快照挂了不能把 execute 带挂.
        """
        # 报告 citation 台账: 每次真实 execute 都留一条, 与 provenance 是否
        # 启用无关 (单测路径无 _provenance_record 也要能攒证据).
        self._append_execution_ledger(tool_name, output)
        record = getattr(self, "_provenance_record", None)
        if record is None:
            return
        try:
            from huginn.provenance import capture

            record.add_snapshot(capture(tool_name, input_params, output=output))
        except Exception:  # 防御: 快照捕获失败不阻断记录
            logger.warning(
                "error in _record_provenance: capture snapshot failed", exc_info=True
            )

    def _append_execution_ledger(self, tool_name: str, output: Any) -> None:
        """把一次真实执行的紧凑快照追加进报告 citation 台账 (best-effort).

        报告面此前只拿 `_last_execution_result` (**末轮**), 中间轮的真实数值全丢
        ——run56 末轮 execute 全是超时/零证据, 报告仍写出干净的 N_c(w) 表 (编造).
        台账把本轮**每次** execute 的数值面留下来, 供报告生成阶段做"数值必须溯源
        到真实 execution_result"的 citation 门 (控制面审计 C 族诚实门).

        只留数值来源字段 (objectives/summary/result...), 丢掉脚本体与标准输出;
        容量与单条长度都封顶, 台账本身不成为新的状态负担. 纯 best-effort.
        """
        try:
            ledger = getattr(self, "_execution_ledger", None)
            if ledger is None:  # 引擎未初始化该字段 (老替身) → 跳过
                return
            import json

            payload = output if isinstance(output, dict) else {"value": str(output)}
            # 丢掉脚本体/标准输出等长文本, 保留 objectives/summary 等数值来源.
            slim = {
                k: v
                for k, v in payload.items()
                if k not in ("script", "code", "stdout", "stderr")
            }
            text = json.dumps(slim, ensure_ascii=False, default=str)
            if len(text) > _EXEC_LEDGER_ENTRY_CHARS:
                text = text[:_EXEC_LEDGER_ENTRY_CHARS]
            ledger.append(
                {"idx": len(ledger) + 1, "tool": tool_name, "result": text}
            )
            if len(ledger) > _EXEC_LEDGER_MAX:
                del ledger[: len(ledger) - _EXEC_LEDGER_MAX]
        except Exception:  # 防御: 台账 best-effort, 挂了不能带挂 execute
            logger.debug("append execution ledger failed", exc_info=True)

    async def _try_evolved_fix(
        self, tool_name: str, tool_input: dict[str, Any], error_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check if the evolution engine has a learned fix for this error.

        This is the other half of the Learn→Execute loop: when a tool fails,
        we ask evolution if it's seen this error before and has a fix.
        Returns a patched result dict on hit, None on miss.
        """
        # H2: variant 失败不走 evolved_fix (P6 guard) — 直接回 bandit loop 记录
        if isinstance(error_result, dict) and error_result.get("_variant_id"):
            return None
        # 桥 E: 默认未命中, 命中时覆盖. 放这儿让 variant/异常/miss 都清空.
        self._last_rule_hit_id = ""
        try:
            evolution = self._get_evolution()
            error_str = str(error_result.get("error", ""))
            fix = evolution.apply_heuristic_fix(tool_name, tool_input, error_str)
            if fix:
                self._last_rule_hit_id = fix.get("rule_id", "")
                patched_desc = fix.get("description")
                if not patched_desc:
                    logger.warning("evolved fix hit but no description, skipping")
                    return None
                return await self._execute_workflow(patched_desc, {"_evolved_fix": True})
        except Exception:  # 防御: 启发式修复失败回退
            logger.warning(
                "error in _try_evolved_fix: apply_heuristic_fix failed", exc_info=True
            )
        return None

    async def _execute_dynamic_workflow(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """A5: 跑 agent 提交的并行工作流脚本.

        plan 里带 "script" 字段 (WorkflowScript.from_dict 的输入), 直接走
        WorkflowOrchestrator.run() 同步等完. 失败的 subtask 不炸整体,
        返回聚合结果让 validate/learn 阶段看.
        """
        # D1: 预算耗尽不再起新的并行工作流 — 它会一次拉起多个并发子任务,
        # 单轮就能烧穿剩余预算. 起动作前查一次, 已耗尽直接空手收口.
        if self._budget_exhausted():
            logger.warning("dynamic_workflow skipped: wall-clock budget exhausted")
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "budget_exhausted": True,
                "error": "挂钟预算耗尽, 跳过动态工作流",
            }
        # H2: bandit loop — plan 带 n_variants 且 toggle on 时走 variant 演化
        from huginn.autoloop.engine import _harness_workflow_evolution_enabled
        if plan.get("n_variants") and _harness_workflow_evolution_enabled():
            return await self._execute_dynamic_workflow_bandit(plan, context)

        from huginn.autoloop.dynamic_workflow import (
            WorkflowOrchestrator,
            WorkflowScript,
        )
        from huginn.core_types import ToolContext

        raw_script = plan.get("script") or {}
        if isinstance(raw_script, str):
            # agent 可能传 JSON 字符串
            import json

            try:
                raw_script = json.loads(raw_script)
            except json.JSONDecodeError:
                raw_script = {}
        script = WorkflowScript.from_dict(raw_script)
        if not script.subtasks:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        orch = WorkflowOrchestrator(
            max_concurrent=script.max_concurrent,
        )
        ctx = ToolContext(
            session_id=f"dynwf_{script.id}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        result = await orch.run(script, ctx, phase="_execute")
        return {
            "mode": "dynamic_workflow",
            "success": result.success,
            "workflow_id": result.id,
            "n_total": result.n_total,
            "n_completed": result.n_completed,
            "n_failed": result.n_failed,
            "summary": result.summary(),
        }

    async def _execute_dynamic_workflow_bandit(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """H2: 对同一 objective 生成 N 个 variant, bandit 选一个跑.

        generate_variants → bandit.select_variant → orch.run →
        返回 dict 带 _variant_id / _objective_hash / _novelty / _efficiency,
        供 _validate 调 bandit.record_variant_outcome + archive.add_variant.
        失败静默回退到原 _execute_dynamic_workflow 逻辑.
        """
        import json as _json

        from huginn.autoloop.bandit import (
            VariantArchive,
            WorkflowBandit,
            _objective_hash,
            compute_novelty,
        )
        from huginn.autoloop.dynamic_workflow import (
            WorkflowOrchestrator,
            WorkflowScript,
        )
        from huginn.autoloop.variant_gen import generate_variants
        from huginn.core_types import ToolContext

        raw_script = plan.get("script") or {}
        if isinstance(raw_script, str):
            try:
                raw_script = _json.loads(raw_script)
            except _json.JSONDecodeError:
                raw_script = {}
        try:
            base_script = WorkflowScript.from_dict(raw_script)
        except Exception:  # 防御: 动态工作流失败返回失败态
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "base script parse fail",
            }
        if not base_script.subtasks:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        objective = base_script.objective or str(plan.get("objective", ""))
        obj_hash = _objective_hash(objective)
        n_variants = int(plan.get("n_variants", 3))

        # 生成 variants (参数扰动优先, base_script 非空走扰动)
        try:
            variants = await generate_variants(
                objective,
                n=n_variants,
                base_script=base_script,
                llm_chat_fn=getattr(self, "_llm_chat", None),
            )
        except Exception:  # 防御: H2 变体生成失败用空集
            logger.debug("H2 generate_variants failed", exc_info=True)
            variants = []
        if not variants:
            variants = [base_script]

        # bandit select (variant_id 加时间戳前缀避免跨轮冲突)
        _run_prefix = f"r{int(time.time() * 1000) % 100000}"
        bandit = WorkflowBandit.get_instance()
        candidate_ids = [f"{_run_prefix}_var_{i}" for i in range(len(variants))]
        chosen_id = bandit.select_variant(candidate_ids, obj_hash)
        if chosen_id is None:
            chosen_id = candidate_ids[0]
        chosen_idx = candidate_ids.index(chosen_id)
        chosen = variants[chosen_idx]
        variant_id = chosen_id

        # 跑选中 variant
        orch = WorkflowOrchestrator(max_concurrent=chosen.max_concurrent)
        ctx = ToolContext(
            session_id=f"dynwf_{chosen.id}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        try:
            result = await orch.run(chosen, ctx, phase="_execute")
        except Exception as exc:
            logger.debug("H2 bandit variant run failed: %s", exc, exc_info=True)
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": f"variant run fail: {exc}",
                "_variant_id": variant_id,
                "_objective_hash": obj_hash,
                "_objective": objective,
                "_script_dict": chosen.to_dict(),
                "_novelty": 0.0,
                "_efficiency": 0.0,
            }

        # novelty vs archive
        try:
            archive = VariantArchive.get_instance()
            existing = archive.list_variants(obj_hash)
            novelty = compute_novelty(chosen.to_dict(), existing)
        except Exception:  # 防御: 新颖度计算失败取零
            logger.debug("best-effort op failed", exc_info=True)
            novelty = 0.0

        efficiency = result.n_completed / max(1, result.n_total)
        return {
            "mode": "dynamic_workflow",
            "success": result.success,
            "workflow_id": result.id,
            "n_total": result.n_total,
            "n_completed": result.n_completed,
            "n_failed": result.n_failed,
            "summary": result.summary(),
            "_variant_id": variant_id,
            "_objective_hash": obj_hash,
            "_objective": objective,
            "_script_dict": chosen.to_dict(),
            "_novelty": float(novelty),
            "_efficiency": float(efficiency),
        }

    # ──────────────────────────────────────────────────────────────
    # Execution helpers
    # ──────────────────────────────────────────────────────────────

    async def _execute_coder(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the coder loop on the description, reusing self.coder."""
        task = f"""Task: {description}

Context:
- Changed files: {context.get('changed_files', [])}
- Git diff: {context.get('git_diff', '')[:500]}

Please modify the code to address this task."""
        try:
            # CoderRunner.run 是同步的, 丢线程里避免阻塞事件循环
            result = await asyncio.to_thread(self.coder.run, task)
            messages = result.get("messages", [])
            tool_calls = sum(1 for m in messages if getattr(m, "tool_calls", None))
            return {
                "mode": "coder",
                "status": "completed",
                "success": True,
                "final_answer": result.get("final_answer", ""),
                "tool_calls": tool_calls,
            }
        except Exception as e:
            logger.exception("coder execution failed")
            return {
                "mode": "coder",
                "status": "failed",
                "success": False,
                "error": str(e),
            }

    # domain → 默认模板名; get_template 拿不到就 fallback standard_dft
    # ponytail: 硬编码映射表, 新模板加一行即可; 想自动发现就扫 WORKFLOW_TEMPLATES
    _DOMAIN_TEMPLATE_NAMES = {
        "cfd": "turbulent_flow",
        "fea": "structural_analysis",
        "qc": "wavefunction_analysis",
        "symbolic": "constitutive_derivation",
        "dft": "standard_dft",
    }

    def _classify_workflow_domain(self, description: str) -> str:
        """廉价关键词分类, 决定走哪个 workflow 模板."""
        text = description.lower()
        if any(k in text for k in ("cfd", "fluid", "fluent", "openfoam")):
            return "cfd"
        if any(k in text for k in ("fea", "stress", "mechanical", "abaqus", "ansys")):
            return "fea"
        if any(k in text for k in ("quantum", "qc", "chemistry", "gaussian", "orca")):
            return "qc"
        if any(k in text for k in ("symbolic", "regression", "拟合")):
            return "symbolic"
        return "dft"

    async def _execute_workflow(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a workflow task, picking template by domain when possible."""
        try:
            from huginn.core_types import ToolContext
            from huginn.workflows.templates import (
                get_template,
                standard_dft_workflow,
            )

            domain = self._classify_workflow_domain(description)
            template_name = self._DOMAIN_TEMPLATE_NAMES.get(domain, "standard_dft")
            template_fn = get_template(template_name) or standard_dft_workflow

            # 找工作区里的输入文件; 只对 DFT/QC 用 structure_path
            structure_files = (
                list(self.workspace.rglob("*.cif"))
                + list(self.workspace.rglob("*.poscar"))
                + list(self.workspace.rglob("*.vasp"))
            )
            geometry_files = (
                list(self.workspace.rglob("*.stp"))
                + list(self.workspace.rglob("*.stl"))
                + list(self.workspace.rglob("*.msh"))
                + list(self.workspace.rglob("*.inp"))
            )
            xyz_files = list(self.workspace.rglob("*.xyz")) + list(
                self.workspace.rglob("*.pdb")
            )
            structure_path = (
                str(structure_files[0]) if structure_files else "structure.cif"
            )

            # 不同域模板参数不一样, 廉价 try 一组; 失败就 fallback DFT
            try:
                if domain == "cfd":
                    geo = str(geometry_files[0]) if geometry_files else "geometry.stp"
                    stages = template_fn(geometry_file=geo)
                elif domain == "fea":
                    geo = str(geometry_files[0]) if geometry_files else "geometry.inp"
                    stages = template_fn(geometry_file=geo)
                elif domain == "qc":
                    struct = str(xyz_files[0]) if xyz_files else structure_path
                    stages = template_fn(structure_file=struct)
                elif domain == "symbolic":
                    # symbolic 模板要 free_energy_expr, 没法从工作区推断, 拿 description 顶
                    stages = template_fn(free_energy_expr=description)
                else:
                    stages = template_fn(structure_path=structure_path, engine="vasp")
            except Exception as tmpl_err:
                logger.warning(
                    "workflow template %s (%s) failed: %s, fallback to standard_dft",
                    template_name,
                    domain,
                    tmpl_err,
                )
                stages = standard_dft_workflow(structure_path, engine="vasp")

            tool_context = ToolContext(
                session_id=f"workflow_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            result = await self.workflow_engine.execute(stages, tool_context)
            return {
                "mode": "workflow",
                "success": result.success,
                "stages": len(stages),
                "domain": domain,
                "outputs": result.outputs,
                "error": result.error,
                "stage_results": [
                    {
                        "name": s.stage_name,
                        "success": s.success,
                        "output": s.output_data,
                    }
                    for s in result.stages
                ],
            }
        except Exception as e:
            return {"mode": "workflow", "success": False, "error": str(e)}

    async def _execute_explore(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute an exploration task."""
        try:
            result = await self.explorer.explore(
                objective=description,
                initial_branches=[
                    {"name": "baseline", "hypothesis": f"Baseline for: {description}"}
                ],
                max_iterations=5,
            )
            return {
                "mode": "explore",
                "n_explored": result.n_branches_explored,
                "n_pruned": result.n_branches_pruned,
                "convergence": result.convergence_reason,
            }
        except Exception as e:
            return {"mode": "explore", "success": False, "error": str(e)}

    async def _execute_skill(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a pre-built composite skill pipeline."""
        try:
            from huginn.skills.base import DeclarativeSkillExecutor
            from huginn.skills.composite import _ensure_registered
            from huginn.skills.registry import SkillRegistry

            _ensure_registered()

            skill_name = plan.get("skill", "")
            skill = SkillRegistry.get(skill_name)
            if not skill:
                # Fuzzy match if exact name missing
                matches = SkillRegistry.search(
                    skill_name or plan.get("description", "")
                )
                skill = matches[0] if matches else None
            if not skill:
                return {
                    "mode": "skill",
                    "success": False,
                    "error": f"no matching skill for '{skill_name}'",
                }

            # Reuse the same tool registry as the rest of the engine
            from huginn.tools.registry import ToolRegistry

            executor = DeclarativeSkillExecutor(ToolRegistry)
            result = await executor.execute(skill, {}, context)
            return {"mode": "skill", "skill": skill.name, **result}
        except Exception as e:
            return {"mode": "skill", "success": False, "error": str(e)}

    # P2 slim-down: visual_inspect 方法族已下沉到 VisualInspect 协作对象
    # (visual_inspect.py). 引擎组合持有 self._visual_inspector, 经薄委托调用.

    async def _llm_chat(
        self,
        prompt: str,
        persona_name: str | None = None,
        model: Any = None,
        task: str | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response.

        persona_name 不为空时, 把对应 persona 的 system prompt 作为
        SystemMessage 插在最前, 实现"每阶段开始注入 persona system prompt".
        persona 找不到就退化为不注入, 行为跟改动前一致.

        model 不为空时用传入的模型 (用于三槽 verification), 否则用默认 self.model.

        task 不为空时, 优先从 model_router 路由 (team 模式):
        - "reasoning"/"science" → 强模型 (云端, 发散性假设)
        - "planning" → 中档模型 (收敛, 把假设变步骤)  [OAK 三阶段角色]
        - "summarize"/"format" → 便宜模型 (本地/小模型)
        - "verification" → 独立验证模型
        model 参数优先于 task — 显式指定的模型不被路由覆盖.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Team 模式: task 路由优先, 但显式 model 不被覆盖
        if model is None and task is not None:
            router = getattr(self, "model_router", None)
            if router is not None:
                try:
                    routed = router.select(
                        task,
                        prefer_cheap=(
                            task in ("summarize", "format", "archival", "planning")
                        ),
                    )
                    if routed is not None:
                        model = routed
                except Exception:  # 防御: 路由选择失败用回退模型
                    logger.debug(
                        "model router select failed — using fallback model",
                        exc_info=True,
                    )

        llm = model or self.model
        messages: list[Any] = []
        if persona_name:
            sys_prompt = self._persona_system_prompt(persona_name)
            # H4: GRILL 模式 active 时把 GRILL_SYSTEM_PROMPT_CN 追加到 system prompt.
            # 之前 should_pause_for_decision 触发 GRILL 后只 auto-resume, LLM 看不到
            # grill 约束, "一次一问" 形同虚设. 现在注入, LLM 自己负责流程.
            if self._grill_active:
                try:
                    from huginn.runtime.pre_plan_grill import GRILL_SYSTEM_PROMPT_CN
                    sys_prompt = (sys_prompt or "") + "\n\n" + GRILL_SYSTEM_PROMPT_CN
                    self._grill_turns += 1
                    # 退出检查: LLM 输出含 "shared understanding" 确认标记 → 退出.
                    # 依赖上层 (run_cognitive) 把 LLM 输出回传后再判断, 这里只计数.
                    # ceiling: 简单字符串匹配, 升级用 LLM judge.
                    if self._grill_turns > 20:
                        logger.warning(
                            "GRILL 超过 20 轮, 强制退出 (可能 LLM 没理解确认标记)"
                        )
                        self._grill_active = False
                except ImportError:
                    logger.debug("pre_plan_grill import failed, GRILL prompt 跳过")
            # 插件 prompt 段 (如 asd_ste100 的受控语言规则): autoloop 直接拼 persona
            # system prompt, **绕过** agent.build_prompt, 故注册进 prompt_segments 的
            # 插件段在这条路径上永不生效 (没通电). 这里显式消费只取插件段 (排除框架
            # 骨架段, 避免与 persona/phase 重复注入). 组装失败不影响主 LLM 调用.
            try:
                from huginn.plugins.prompt_segments import (
                    assemble_plugin_prompt_segments,
                )

                _plugin_seg = assemble_plugin_prompt_segments(
                    mode="default",
                    phase=self._current_phase or "",
                    metacog_state="",
                    system_prompt=sys_prompt,
                )
                if _plugin_seg:
                    sys_prompt = f"{sys_prompt}\n\n{_plugin_seg}" if sys_prompt else _plugin_seg
            except Exception:  # 防御: 段组装失败退回 persona-only, 不阻塞主路径
                logger.debug("plugin prompt segments failed", exc_info=True)
            if sys_prompt:
                sys_msg = SystemMessage(content=sys_prompt)
                # 静态 system prompt 跨调用不变, 给 Anthropic 打 cache 标记.
                # Kimi/Moonshot 走 OpenAI 协议, cache_control 无效, 不打标记.
                _ident = f"{type(llm).__name__}{getattr(llm, 'model', '')}".lower()
                if any(k in _ident for k in ("anthropic", "claude")):
                    sys_msg.additional_kwargs["cache_control"] = {"type": "ephemeral"}
                messages.append(sys_msg)
        # Controllable thinking effort: 按 current phase 注入思考深度指令.
        # Inkling 启发 — 连续旋钮, prompt 层实现, 对所有 provider 统一.
        # 无 _current_phase (非 phase 上下文调用, 如 _feynman_learn) 时不注入.
        from huginn.autoloop.engine import (
            _PHASE_THINKING_EFFORT,
            _autoloop_streaming_enabled,
            _effort_to_prompt,
        )
        effort_directive = ""
        if self._current_phase:
            effort = _PHASE_THINKING_EFFORT.get(self._current_phase, 0.5)
            effort_directive = _effort_to_prompt(effort)
        if effort_directive:
            prompt = f"[Thinking effort: {effort_directive}]\n\n{prompt}"
        messages.append(HumanMessage(content=prompt))
        # P0-1: 流式化 — astream 替代 ainvoke, 增量 chunk 通过 progress_cb 推 WS.
        # 700 万步场景: decider 思考过程实时可见, 不再黑盒. fail 回退 ainvoke.
        # ponytail: 只在 progress_cb 存在时流式, 否则 ainvoke (兼容无 WS 场景).
        from huginn.core_types import progress_cb as _progress_cb

        _cb = _progress_cb.get(None)
        if _cb is None or not hasattr(llm, "astream") or not _autoloop_streaming_enabled():
            # with_retry: 429/限流(含 400 包装)/529/网络抖动 分级退避重试, 不可重试
            # 错误原样抛出 (语义与改动前一致, 只是多了瞬时错误的韧性).
            response = await with_retry(
                lambda: llm.ainvoke(messages), source="autoloop"
            )
            await self._track_llm_usage(getattr(response, "usage_metadata", None))
            return str(response.content)
        # 流式: 累积 content, 同时推 thinking chunk 到 WS
        parts: list[str] = []
        _usage_meta = None
        try:
            async for chunk in llm.astream(messages):
                _delta = ""
                # P2-7: 末 chunk 常带 usage_metadata, 累加到 token budget.
                _um = getattr(chunk, "usage_metadata", None)
                if _um:
                    _usage_meta = _um
                # langchain BaseMessageChunk: chunk.content 是 str 或 list
                if hasattr(chunk, "content"):
                    if isinstance(chunk.content, str):
                        _delta = chunk.content
                    elif isinstance(chunk.content, list):
                        # 多模态 chunk, 只取 text block
                        _delta = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in chunk.content
                        )
                if _delta:
                    parts.append(_delta)
                    with contextlib.suppress(Exception):
                        # cb 失败不阻塞 LLM
                        await _cb({
                            "type": "autoloop_thinking",
                            "phase": self._current_phase or "decider",
                            "delta": _delta[:200],  # 截断防超大 delta
                        })
            await self._track_llm_usage(_usage_meta)
            return "".join(parts)
        except Exception as e:
            # 流式失败回退 ainvoke (某些 provider astream 实现有 bug).
            # 回退同样走 with_retry — astream 抛限流时, 这一步是唯一的韧性出口.
            logger.debug("astream failed, fallback to ainvoke: %s", e)
            response = await with_retry(
                lambda: llm.ainvoke(messages), source="autoloop"
            )
            await self._track_llm_usage(getattr(response, "usage_metadata", None))
            return str(response.content)
