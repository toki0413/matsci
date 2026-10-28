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

from huginn.autoloop._facade import EngineFacade

# 提示面升级阈值 (与 engine_reflect 同一旋钮): 由本对象在组装作者提示时**消费**,
# 故常量必须同源. engine_reflect 不反向 import 本模块, 无循环风险.
# with_retry: autoloop 的 LLM 调用此前**绕开**了统一重试层 (agent/streaming.py 有,
# 这条路径没有), 一次限流/网络抖动就被当成"阶段无产出" → redirect→pivot→停机
# (run74). 这里接上.
from huginn.autoloop.engine_reflect import (
    _CITATION_MIN_MAGNITUDE,
    _REPEAT_HARD_STREAK,
    _numeric_tokens,
)
from huginn.autoloop.exec_observation import classify_exit_class
from huginn.llm_retry import with_retry

logger = logging.getLogger(__name__)

# 报告 citation 台账容量: 每条一次真实 execute, 只留紧凑"数值面".
_EXEC_LEDGER_MAX = 24
_EXEC_LEDGER_ENTRY_CHARS = 600

# code_lab 超时后的最小重试预算 (秒): 低于此值的沙箱跑不出任何有意义的实验.
_CODELAB_MIN_TIMEOUT_S = 60.0
# 未拿到显式超时时的兜底 (与 HUGINN_CODELAB_TIMEOUT_S 的默认档一致).
_CODELAB_FALLBACK_TIMEOUT_S = 600.0

def _ledger_text(output: Any) -> str:
    """执行回执 → 紧凑 JSON 文本; 超长时在词边界截断并补窗口外数值面."""
    import json

    payload = output if isinstance(output, dict) else {"value": str(output)}
    # 丢掉脚本体/标准输出等长文本, 保留 objectives/summary 等数值来源.
    slim = {
        k: v
        for k, v in payload.items()
        if k not in ("script", "code", "stdout", "stderr")
    }
    text = json.dumps(slim, ensure_ascii=False, default=str)
    if len(text) <= _EXEC_LEDGER_ENTRY_CHARS:
        return text
    # 硬截断会把"真实存在但落在窗口外"的数值从证据面抹掉: ws9 的
    # ev5 在 600 字符后才写 max_l2_err / score, 报告如实抄了这些数,
    # 审计门却在台账里查不到 → 判成编造. 截在词边界, 再把窗口外的
    # **数值面**补成紧凑后缀 —— 报告面照样短, 证据面不失真.
    cut = text.rfind(" ", 0, _EXEC_LEDGER_ENTRY_CHARS)
    if cut < _EXEC_LEDGER_ENTRY_CHARS // 2:
        cut = _EXEC_LEDGER_ENTRY_CHARS
    tail_nums = sorted(_numeric_tokens(text[cut:], _CITATION_MIN_MAGNITUDE))
    return text[:cut] + (" nums_after_cut=" + " ".join(tail_nums) if tail_nums else "")

def _next_timeout_cap(
    current_cap: float | None,
    effective_timeout: float | None,
    timed_out: bool,
) -> tuple[float | None, bool]:
    """超时后的下一轮超时预算: ``(新上限, 是否还允许再试)``.

    超时是**算力预算**问题而非代码 bug: 代码不变时按原超时重跑只会再烧一遍
    (ws9 iteration 3 实测 600s x 4). 首次超时把上限减半再给一次机会; 连续
    第二次超时就停 —— 继续重试的期望收益为零, 挂钟代价是线性的.
    """
    if not timed_out:
        return current_cap, True
    if current_cap is not None:
        return current_cap, False
    base = (
        effective_timeout
        if effective_timeout and effective_timeout > 0
        else (_CODELAB_FALLBACK_TIMEOUT_S)
    )
    return max(_CODELAB_MIN_TIMEOUT_S, base / 2.0), True

class EngineAct(EngineFacade):
    """plan / execute 阶段方法族协作对象.

    未定义的属性读写经 __getattr__/__setattr__ 转发到 self.engine —
    引擎字段/方法(如 _grill_active / _current_prediction / model / _llm_chat 等)
    不会被复制两份, 方法体零改动、行为完全等价.
    """

    #: plan / validate 阶段允许挂载的**只读**工具白名单.
    #: 只收 read_only=True 且无副作用的工具 —— 这两个阶段此前完全没有工具面,
    #: 书生只能凭记忆"编"计划与批评, 无法查文献/翻记忆/看自己的执行 trace.
    _READONLY_PHASE_TOOLS: tuple[str, ...] = (
        "literature_tool",
        "recall",
        "self_observe",
        "recall_observation",
        "web_search_tool",
        "agentic_search_tool",
        "paper_tool",
    )

    def _readonly_tool_usable(self, tool: Any, name: str) -> bool:
        """三重过滤: ``read_only`` → 前置状态 → active → ``is_available()``."""
        if tool is None or not getattr(tool, "read_only", False):
            return False
        if name == "recall_observation" and not self._has_archived_observations():
            # 前置状态工具: 没有 offload 句柄时它 100% 失败 (ws7 实测书生
            # 只能瞎传 '@obs:' → malformed handle). 按需裁剪, 不给"必然失败"
            # 的选项, 免得浪费一轮工具调用并污染成功率.
            return False
        if not getattr(tool, "active", True):
            return False
        try:
            return bool(tool.is_available())
        except Exception:  # 防御: 可用性探测异常按不可用处理
            return False

    def _readonly_perm(self) -> Any:
        """最小权限授权: 白名单 7 个只读工具 AUTO, 其余仍走 plan_mode 的 ASK."""
        from huginn.core_types import PermissionMode
        from huginn.permissions import PermissionConfig

        # 显式授权: 默认权限表把多数工具判成 ASK, 而阶段内没有审批人 → 每次调用
        # 都恒拒 (实测回执 "recall 工具需要授权"). 这里按**最小权限**只给白名单
        # 内这 7 个只读工具 AUTO, 其余仍走 plan_mode 的 ASK/写工具降级.
        perm = PermissionConfig(plan_mode=True)
        for name in self._READONLY_PHASE_TOOLS:
            perm.set_mode(name, PermissionMode.AUTO)
        return perm

    @staticmethod
    def _ensure_tools_registered(register_all_tools: Any, registry_cls: Any) -> None:
        """注册时机修正: 按需补一次 register_all_tools (自带幂等门).

        run_cognitive_driver 只构造 AutoloopEngine, 从不调用 register_all_tools();
        工具要到 execute/coder 内部才被懒注册. 于是 plan (execute 之前) 看到的
        ToolRegistry 是空的 → 白名单过滤后恒为 []. 这里按需补一次, 代价只在
        首次 (~秒级), 之后是常数返回.
        """
        if registry_cls.list_tools():
            return
        try:
            register_all_tools()
        except Exception:  # 防御: 注册失败按无工具处理, 不阻塞阶段推进
            logger.debug("register_all_tools failed", exc_info=True)

    def _adapt_readonly_tool(
        self, adapter: Any, tool: Any, name: str, perm: Any
    ) -> Any | None:
        """单个只读工具适配 (失败返回 None, 不影响其余工具)."""
        try:
            # memory_manager 必须透传: engine 持有 MemoryManager, 但 adapter
            # 默认丢弃 → recall/remember 在 driver 上下文恒返回 "Memory manager
            # not available" (ws7 实测 2/2 失败). 工具面给了却不给运行期依赖,
            # 等于又一层"注册≠可用".
            return adapter.adapt(
                tool,
                permission_config=perm,
                memory_manager=getattr(self, "memory", None),
            )
        except Exception:  # 防御: 单个工具适配失败不影响其余工具
            logger.debug("adapt readonly tool %s failed", name, exc_info=True)
            return None

    def _resolve_readonly_tools(self) -> list[Any]:
        """按白名单取只读工具并适配为 LangChain 工具 (best-effort).

        三重过滤: 已注册 → ``read_only`` → 运行时 ``is_available()``. 任一不满足
        即跳过; 适配失败也跳过. 全部失效时返回 [], 调用方退化为无工具调用.
        """
        try:
            from huginn.tools import register_all_tools
            from huginn.tools.adapter import ToolAdapter
            from huginn.tools.registry import ToolRegistry
        except Exception:  # 防御: 工具子系统不可用时不阻塞 plan/validate
            logger.debug("readonly tool import failed", exc_info=True)
            return []

        self._ensure_tools_registered(register_all_tools, ToolRegistry)
        perm = self._readonly_perm()
        adapter = ToolAdapter()
        out: list[Any] = []
        for name in self._READONLY_PHASE_TOOLS:
            tool = ToolRegistry.get(name)
            if not self._readonly_tool_usable(tool, name):
                continue
            adapted = self._adapt_readonly_tool(adapter, tool, name, perm)
            if adapted is not None:
                out.append(adapted)
        if out:
            # info 级: 触发率评估需要"这一阶段到底给了模型哪些工具"的可见证据.
            logger.info(
                "readonly phase tools available: %s",
                [getattr(t, "name", "?") for t in out],
            )
        return out

    def _has_archived_observations(self) -> bool:
        """本 session 是否已有 offload 句柄 (决定 recall_observation 是否挂出)."""
        try:
            from huginn.tools.observation_pack import ObservationArchive

            # adapter 构造 ToolContext 时 session_id 固定为 "default", 这里必须
            # 用同一口径, 否则裁剪判断与实际召回目录对不上.
            return ObservationArchive("default").has_entries()
        except Exception:  # 防御: 探测失败按"无句柄"处理, 只是少给一个工具
            return False

    def _plan_grill_exit(self, response: str | None) -> None:
        """H4: GRILL 退出检查 — 输出含确认标记则退出 grill 模式."""
        # ponytail: 简单字符串匹配, 不上 LLM judge. ceiling: LLM 不说这个词就永远不退.
        if not (self._grill_active and response):
            return
        _resp_lower = response.lower()
        if (
            "shared understanding" in _resp_lower
            or "shared understanding reached" in _resp_lower
            or "confirmed decisions" in _resp_lower
        ):
            self._grill_active = False
            self._grill_turns = 0
            logger.info("GRILL mode exited: shared understanding confirmed")

    def _plan_jepa_slots(self, plan: dict[str, Any]) -> None:
        """JEPA 方案① 结构化计划槽: 把已知输入/公式追加到 prediction 文本."""
        # 并暂存槽供 _record_jepa_pair 落库 + 泄漏检测。仅方法级目标会有 prediction_inputs。
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

    async def _plan_cost_gate(self, plan: dict[str, Any]) -> bool:
        """cost 确认门: True = 放行; False = 用户显式拒绝."""
        # None = 不用问 (非高成本 mode / manager 不可用), 直接放行; 显式拒绝才拦.
        # bool 和字符串都兼容 (测试 mock 常传 bool)
        # H4: reject_tokens 从 PhaseRegistry extra 取 (toggle off 回退 hardcode tuple)
        from huginn.harness.phase_spec import get_phase_extra

        _reject_tokens = get_phase_extra(
            "_plan",
            "reject_tokens",
            ["no", "n", "cancel", "reject", "decline", "stop", "abort"],
        )
        answer = await self._maybe_clarify("plan", plan)
        if answer is None:
            return True
        if isinstance(answer, bool):
            return answer
        if isinstance(answer, str):
            return answer.lower().strip() not in tuple(_reject_tokens)
        return bool(answer)

    async def _plan_persist(
        self, hypothesis: str, plan: dict[str, Any]
    ) -> dict[str, Any] | None:
        """落 PlanStore: 创建 plan → cost 确认门 → confirm/reject."""
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
            if not await self._plan_cost_gate(plan):
                plan_store.reject_plan(persisted.id, reason="user declined")
                return None
            plan_store.confirm_plan(persisted.id)
            plan_store.mark_executing(persisted.id)
            plan["plan_id"] = persisted.id
        except Exception as e:
            logger.warning("plan store persistence failed: %s", e)
            # PlanStore 挂了不阻塞执行, 退回老的纯 dict 路径
        return plan

    async def _plan_llm(self, prompt: str) -> tuple[str | None, dict | None]:
        """调 LLM 产出 plan; 瞬时故障标记阶段不可用, 失败返回 (None, None).

        OAK 启发: 三阶段角色分工 — hypothesize 用 reasoning (强模型发散),
        plan 用 planning (中档模型收敛), execute 不调 LLM 直接跑工具.
        """
        try:
            response = await self._llm_chat(
                prompt,
                persona_name="default",
                task="planning",
                tools=self._resolve_readonly_tools(),
            )
            plan = self._parse_plan(response)
            # LLM 可达 (即便解析为空) → 不是"不可用": 清标记, 让空产出正常驱动
            # reflect 的重定向逻辑, 不被上一阶段的瞬时故障标记误伤.
            self._last_phase_unavailable = False
            return response, plan
        except Exception as _exc:  # 防御: 尽力操作失败返回空
            from huginn.llm_retry import is_transient_error

            if is_transient_error(_exc):
                self._last_phase_unavailable = True
            logger.debug("best-effort op failed", exc_info=True)
            return None, None

    async def _plan_postprocess(
        self, plan: dict[str, Any], hypothesis: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """B 硬路由 + KRCL 反向校验 + JEPA 槽位 + 意图口径缓存."""
        # B: 硬路由 — 根据上下文信号覆盖 LLM 的 mode 选择
        plan = self._override_plan_mode(plan)
        # KRCL 启发: 反向校验 plan 能否达成 hypothesis, 失败反馈 LLM 重生成
        # 单 LLM 反向校验, 最多 1 次重试, 失败不阻塞 (标 warning 继续)
        plan = await self._plan_check_and_refine(plan, hypothesis, context)
        # JEPA 阶段2-0: 把 plan 的 expected_prediction 同步为 JEPA prediction buffer,
        # 使 AutoloopEngine 的 validate 阶段也能采集 plan→actual 配对 (与
        # cognitive_loop L2323 行为对齐). 真实模型常把数值预测写进 description
        # 而非结构化 expected_prediction 键, 故缺失时兜底 description, 保证采集对
        # 真实输出触发; 两者皆空则保持空, validate 不采, 与原行为一致.
        self._current_prediction = (
            plan.get("expected_prediction") or plan.get("description") or ""
        ).strip()
        self._plan_jepa_slots(plan)
        # 意图口径 (S2) 输入: 缓存本轮 plan 声明的目标文件集, 供 _learn 的
        # _apply_strict_scope 判"改动是否偏离本轮意图". 声明缺失 (LLM 未输出
        # FILES:) → 空表 → 意图口径 no-op (零回归). 与合规口径 (S1) 独立开关.
        self._current_plan_target_files = list(plan.get("target_files") or [])
        return plan

    #: plan 空 DESCRIPTION 的强制重写指令 (只重试一次).
    _PLAN_DESC_RETRY_SUFFIX = (
        "\n\nCRITICAL: DESCRIPTION is mandatory and must be non-empty. "
        "It is the exact task handed to the executor. Output it on one line."
    )

    @staticmethod
    def _plan_description_missing(plan: dict[str, Any] | None) -> bool:
        return not str((plan or {}).get("description") or "").strip()

    async def _plan(
        self, hypothesis: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate a plan from hypothesis and persist it to PlanStore.

        以前只返回一个临时 dict, turn 结束就丢了. 现在往 PlanStore 落一份,
        跨会话可恢复, 用户也能 confirm/reject. PlanStore 不可用时退回老行为.

        结构性锚点: DESCRIPTION 是下发给执行器的**执行对象**, 不能为空.
        空描述先强制重写一次; 仍空则 fail-fast 出诊断, 不再用"当前假设"兜底
        (兜底会让模型自造执行对象 → 假设漂移).
        """
        self._last_plan_failure = None
        prompt = self._build_plan_prompt(hypothesis, context)
        response, plan = await self._plan_llm(prompt)
        if response is None:
            return None
        self._plan_grill_exit(response)
        if not plan:
            self._last_plan_failure = {
                "kind": "empty_description",
                "attempts": 1,
                "diagnostic": "planner 未产出可解析的 MODE/DESCRIPTION",
            }
            return None
        if self._plan_description_missing(plan):
            logger.warning("plan DESCRIPTION 为空 → 强制重写一次")
            response, _retry = await self._plan_llm(
                prompt + self._PLAN_DESC_RETRY_SUFFIX
            )
            if response is None:
                return None
            self._plan_grill_exit(response)
            if _retry:
                plan = _retry
        if self._plan_description_missing(plan):
            self._last_plan_failure = {
                "kind": "empty_description",
                "attempts": 2,
                "diagnostic": "planner 连续两次产出空 DESCRIPTION, 拒绝下发执行器",
            }
            logger.error(
                "plan 连续两次 DESCRIPTION 为空 → fail-fast (不再用当前假设兜底)"
            )
            return None
        plan = await self._plan_postprocess(plan, hypothesis, context)
        return await self._plan_persist(hypothesis, plan)

    #: 计算/预测指令词 (英文 + 中文). autoloop 的真实 objective 多为中文
    #: ("写并运行…统计均方违规, 求 N_c(w)…拟合 beta"), 只认英文动词会让内建
    #: 执行路径永不触发 → execute 落到 explore 空转, report 再拿现编数字当证据.
    _NUMERIC_VERBS = (
        "compute",
        "calculate",
        "predict",
        "evaluate",
        "计算",
        "求解",
        "算出",
        "核算",
        "拟合",
        "统计",
        "测量",
        "预测",
        "评估",
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
        return bool(_re.search("[/^*+=()]", d)) or any(
            h in d for h in self._FORMULA_HINTS
        )

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
            "physical/math relation exactly; do not fabricate values.\n\nTASK: "
            + description
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
        "实验",
        "仿真",
        "模拟",
        "训练",
        "扫描",
        "网格",
        "留出",
        "泛化",
        "探针",
        "基准",
        "收敛",
        "灵敏度",
        "experiment",
        "simulat",
        "train",
        "scan",
        "sweep",
        "generaliz",
        "held-out",
        "holdout",
        "probe",
        "benchmark",
        "convergence",
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

    # 「必须外部取证」信号: Code Lab 沙箱移除 ``open()``、import 白名单不含
    # urllib/requests/socket、``bash_tool``/``code_tool`` 在 _BLOCKED_TOOLS ——
    # 它**物理上**无法下载也无法落盘. 目标一旦要求真实数据获取, 让 code_lab
    # 接管 execute 就只能交出一段"合成 stand-in"(ws13 实测: 书生自述
    # "real data fetch is handled by the outer orchestrator"), 而报告仍把它
    # 当执行证据抄出去. 这类目标改道 coder: 唯一带 bash_tool/写文件能力的
    # 真实执行路径.
    _IO_REQUIRED_HINTS = (
        "下载",
        "download",
        "抓取",
        "fetch",
        "csv",
        "数据集",
        "dataset",
        "sha256",
        "source_url",
        "http://",
        "https://",
        "落盘",
    )

    def _needs_external_data_io(self, text: Any) -> bool:
        """启发式: 该项是否要求"从外部取数/落盘"(Code Lab 结构性做不到)."""
        if not isinstance(text, str) or not text.strip():
            return False
        low = text.lower()
        return any(h in low for h in self._IO_REQUIRED_HINTS)

    @staticmethod
    def _codelab_author_prompt(
        goal: str,
        guards: dict[str, Any] | None,
        repair_hint: str,
        prev_code: str,
        focus: str,
    ) -> str:
        """组装作者提示: 脚手架 + 守卫块 + 修 bug 回灌 (命题无关)."""
        from huginn.research.code_lab import build_author_prompt, load_scaffold

        guard_block = "\n".join(
            f"- {g}" for g in ((guards or {}).get("prompt_guards") or [])[:8]
        )
        # 任务脚手架由环境变量声明(命题资产, 非平台内核): 不给就是命题无关的默认骨架.
        import os as _osc

        scaffold = load_scaffold(_osc.environ.get("HUGINN_CODELAB_SCAFFOLD", ""))
        return build_author_prompt(
            goal,
            scaffold=scaffold,
            guard_block=guard_block,
            repair_hint=repair_hint,
            prev_code=prev_code,
            focus=focus,
        )

    @staticmethod
    def _codelab_author_debug(raw: str | None, code: str, prompt_len: int) -> None:
        """调试开关下的长度快照 (默认关闭, 不产生日志)."""
        import os as _ose

        if _ose.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
            logger.warning(
                "[code-lab-author] raw_len=%d extracted_len=%d prompt_len=%d",
                len(raw or ""),
                len(code),
                prompt_len,
            )

    async def _request_code_lab_experiment(
        self,
        goal: str,
        guards: dict[str, Any] | None = None,
        repair_hint: str = "",
        prev_code: str = "",
        focus: str = "",
    ) -> str:
        """平衡点·内建执行: 让书生亲手写一段 Code Lab ``run(cfg)`` 实验代码.

        命题无关: 提示词来自 ``code_lab.build_author_prompt`` 的单一契约. 这里只取
        代码, 不评科学可信度(那是 validate/裁决层的事). 失败返回空串 → 回落原分派.
        ``repair_hint`` 非空时把沙箱真实报错回灌, 并附上 ``prev_code``(上一版失败
        代码), 让书生在此基础上做**最小改动**修 bug, 而不是凭空重写再犯同一个错.
        ``focus`` 是本轮可变的聚焦文本(见 ``_build_codelab_focus``), 防止作者提示
        因只喂恒定 goal 而逐字节冻结.
        """
        from huginn.research.code_lab import extract_code

        prompt = self._codelab_author_prompt(
            goal, guards, repair_hint, prev_code, focus
        )
        # 落一份不自带调试开关的长度快照: replay_audit 的"输入冻结"判定改由
        # episodic 的 prompt_len 取(旧码只读 HUGINN_EXEC_ROUTE_DEBUG 下的 run.log 行).
        self._last_author_prompt_len = len(prompt)
        try:
            raw = await self._llm_chat(prompt, model=self.verification_model)
        except Exception as e:  # — LLM 不可用/超时 → 拿不到实验代码, 回落, 不阻塞
            import os as _ose

            if _ose.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
                logger.warning("[code-lab-author] LLM 调用失败: %r", e)
            return ""
        code = extract_code(raw or "")
        self._codelab_author_debug(raw, code, len(prompt))
        return code

    def _run_code_lab(
        self,
        code: str,
        guards: dict[str, Any] | None = None,
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
            float(timeout_s)
            if timeout_s is not None
            else float(_os.environ.get("HUGINN_CODELAB_TIMEOUT_S", "600"))
        )
        extra = tuple((guards or {}).get("imports_whitelist_extra") or ())
        aliases = (guards or {}).get("cfg_aliases") or None
        scaffold = load_scaffold(_os.environ.get("HUGINN_CODELAB_SCAFFOLD", ""))
        res, reason = sandbox_run(
            code,
            {"seed": 0},
            timeout=timeout,
            imports_whitelist_extra=extra,
            cfg_aliases=aliases,
            scaffold=scaffold,
        )
        import os as _osd

        if _osd.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
            _ok = None if res is None else res.get("success")
            _nobj = 0 if res is None else len(res.get("objectives", {}) or {})
            logger.warning(
                "[code-lab-run] res_none=%s success=%s nobj=%d reason=%r",
                res is None,
                _ok,
                _nobj,
                (reason or "")[:200],
            )
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
                _why + "本轮实验**必须**至少改动一项结构: "
                "更换 family/约束族、改扫描范围或步长、改网络架构/宽度集合、"
                "改优化器或 seeds. 禁止原样重跑上一配置."
                + (
                    ("上一轮真实结果(你必须产出与它不同的数值):" + _prev[:500])
                    if _prev
                    else ""
                )
            )
        _hyp = str(getattr(self, "_last_hypothesis", "") or "").strip()
        if _hyp:
            parts.append(f"当前待检验假设: {_hyp[:600]}")
        _step = (description or "").strip()
        if _step and _step not in _hyp:
            parts.append(f"本轮实验步骤: {_step[:600]}")
        return "\n".join(parts)

    async def _codelab_start(self, description: str) -> tuple[str, str, str]:
        """确定 goal / focus 并让书生产出首版实验代码; code 为空表示失败."""
        # 优先用研究目标(全文本, 命题无关的真问题), plan 步骤只作兜底 —— plan 描述
        # 常带 "FILES:/SKILL:" 之类执行噪声, 不适合喂给"实验作者"提示.
        goal = str(getattr(self, "_objective", "") or "") or description
        # v11 反冻结: goal 恒定时作者提示逐字节冻结, 用本轮可变 focus 解冻.
        focus = self._build_codelab_focus(description)
        code = await self._request_code_lab_experiment(goal, focus=focus)
        if not code:
            import os as _ose

            if _ose.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
                logger.warning(
                    "[code-lab] 未产出可解析实验代码 → 回落; goal[:80]=%r", goal[:80]
                )
        return code or "", goal, focus

    def _codelab_gate(
        self,
        attempt: int,
        repair_affordable: Any,
        prev_cost: float,
        slice_budget: Any,
    ) -> tuple[bool, str | None]:
        """起新尝试前的门: 返回 (是否中止, 需要覆盖的 last_err)."""
        # D1: 统一 deadline 原语 — 每次尝试前查一次, 已耗尽就不再起新尝试
        # (最多再多跑一个已在飞的尝试). 修复循环每次 ``_run_code_lab`` 最长可烧满
        # ``HUGINN_CODELAB_TIMEOUT_S`` (默认 900s), 最多 ``max_repairs+1`` 次 → 单轮
        # 可越过挂钟上限近一小时 (run56 实测). 纯时间边界, 不做科学判断; fail-open.
        # 旧私有 ``_wall_clock_expired`` 已删除: 时间口径下沉到 ``_budget_exhausted``
        # (且按本 run goal id 判定, 旧实现用全局 get_active() 会跨 run 串台).
        if self._budget_exhausted():
            logger.warning(
                "code_lab 修复循环: 挂钟预算已耗尽, 停止第 %d 次尝试", attempt + 1
            )
            return True, "挂钟预算耗尽, 中止 code_lab 修复循环"
        # D6② 可选 slice 门: 修复重写启动前查可负担性 (自校准门槛).
        if (
            attempt > 0
            and callable(repair_affordable)
            and not repair_affordable(prev_cost)
        ):
            _rem = slice_budget() if callable(slice_budget) else None
            self._emit_control_trace(
                "code_lab_slice_skip",
                f"slice=repair#{attempt}: budget<max(min,{prev_cost:.0f}s) "
                f"remaining={-1 if _rem is None else round(_rem)}s",
                action="skip",
            )
            logger.warning(
                "code_lab 修复循环: 剩余预算不足以再跑一轮修复(≈%.0fs), 收尾",
                prev_cost,
            )
            return True, None
        return False, None

    @staticmethod
    def _codelab_timeout(attempt_timeout: Any, timeout_cap: float | None) -> Any:
        """本片沙箱超时: 宿主给的预算再按超时上限封顶 (fail-open)."""
        eff = attempt_timeout() if callable(attempt_timeout) else None
        if timeout_cap is not None:
            eff = min(eff, timeout_cap) if eff else timeout_cap
        return eff

    @staticmethod
    def _codelab_success(code: str, res: dict[str, Any]) -> dict[str, Any]:
        """沙箱跑通 → 组装证据 dict."""
        return {
            "mode": "code_lab",
            "status": "completed",
            "success": bool(res.get("success", True)),
            "result": res.get("summary", {}),
            "objectives": res.get("objectives", {}),
            "script": code,
            "reproducible": True,
        }

    def _codelab_timeout_trace(
        self,
        reason: Any,
        attempt: int,
        max_repairs: int,
        timeout_cap: float | None,
        effective_timeout: Any,
    ) -> tuple[float | None, bool]:
        """超时 → 记 trace + 收紧预算; 返回 (新 cap, 是否允许继续重试)."""
        # 观测: 超时是"算力预算"问题而非代码 bug, 单独记 trace, 便于统计
        # "被超时饿死的迭代占比" (run59/60 实测三组均被 900s 超时饿死).
        if not (("超时" in (reason or "")) or ("timeout" in (reason or "").lower())):
            return timeout_cap, True
        self._emit_control_trace(
            "code_lab_timeout",
            f"attempt={attempt + 1}/{max_repairs + 1} {str(reason)[:120]}",
            action="advisory_hint",
        )
        cap, allow_more = _next_timeout_cap(timeout_cap, effective_timeout, True)
        if not allow_more:
            logger.warning(
                "code_lab 连续超时, 停止重试 (超时属算力预算, "
                "按原超时重跑只会空转): 本轮已用 %d 次尝试",
                attempt + 1,
            )
        return cap, allow_more

    async def _codelab_repair(
        self,
        goal: str,
        code: str,
        focus: str,
        last_err: str,
        prev_cost: float,
    ) -> tuple[str | None, float]:
        """把沙箱报错回灌书生重写; 返回 (新代码, 更新后的片成本)."""
        # D6 自校准: 修复重写的作者 LLM 调用也是下一个 slice 的成本, 计入
        # 实测 —— 否则低估下一片 (run79 超支主要来自作者调用, 非沙箱).
        _t_auth = time.monotonic()
        repaired = await self._request_code_lab_experiment(
            goal, repair_hint=last_err, prev_code=code, focus=focus
        )
        return repaired, prev_cost + time.monotonic() - _t_auth

    async def _codelab_loop(
        self, goal: str, code: str, focus: str, max_repairs: int
    ) -> tuple[dict[str, Any] | None, str, str]:
        """修复循环: 返回 ``(成功回执|None, 最终代码, 最后一次失败原因)``.

        D6 (D-slice 同款, execute 侧): 把"修复循环"切成细粒度 slice, 不设硬时长.
          核心 slice (attempt 0) 必跑, 但其沙箱超时**封顶到剩余预算** → 单片越不了界;
          可选 slice (修复重写 attempt>=1) 启动前查可负担性, 门槛自校准 = max(下限,
          上一片实测挂钟). 非长程/关开关时两函数 fail-open → 逐字节等价旧行为.
        """
        last_err = ""
        _prev_slice_cost = 0.0  # 上一片 (上一次尝试: 沙箱 + 其后作者 LLM) 的实测挂钟
        # 超时是算力问题: 首次超时减半预算, 连续超时直接停.
        _timeout_cap: float | None = None
        # D6 钩子: 长程宿主 (Engine/engine_control) 提供预算切片三函数; 短程宿主 /
        # 离线 stub 无这三个属性 → None, 门与封顶整体 fail-open, 行为 100% 不变
        # (与 branch_incubator 的 fail-open 一致). getattr 兜底避免 stub 上
        # AttributeError.
        _attempt_timeout = getattr(self, "_codelab_attempt_timeout", None)
        _repair_affordable = getattr(self, "_codelab_repair_affordable", None)
        _slice_budget = getattr(self, "_codelab_slice_budget", None)
        for attempt in range(max_repairs + 1):
            stop, override = self._codelab_gate(
                attempt, _repair_affordable, _prev_slice_cost, _slice_budget
            )
            if stop:
                if override is not None:
                    last_err = override
                break
            _t_slice = time.monotonic()
            _effective_timeout = self._codelab_timeout(_attempt_timeout, _timeout_cap)
            res, reason = self._run_code_lab(code, timeout_s=_effective_timeout)
            _prev_slice_cost = time.monotonic() - _t_slice  # D6 自校准: 实测本片
            if res is not None:
                return self._codelab_success(code, res), code, last_err
            last_err = reason
            _timeout_cap, _allow_more = self._codelab_timeout_trace(
                reason, attempt, max_repairs, _timeout_cap, _effective_timeout
            )
            if not _allow_more:
                break
            if attempt < max_repairs:
                repaired, _prev_slice_cost = await self._codelab_repair(
                    goal, code, focus, last_err, _prev_slice_cost
                )
                if not repaired:
                    break
                code = repaired
        return None, code, last_err

    async def _execute_code_lab(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """mode=code_lab: 书生亲写一段计算实验 → Code Lab 沙箱真跑 → 证据.

        与闭式 probe 分层: 闭式只认平凡数值片段; 这里认"实验意图"并允许更长时/
        更多依赖(仍受沙箱白名单约束). 失败返回 success=False, 由 validate 层裁决.
        沙箱真报错时把报错回灌书生重写 (命题无关·通用修 bug), 最多 N 轮自修复.
        """
        code, goal, focus = await self._codelab_start(description)
        if not code:
            return {
                "mode": "code_lab",
                "status": "failed",
                "success": False,
                "error": "书生未产出可解析的实验代码",
            }
        import os as _os

        max_repairs = int(_os.environ.get("HUGINN_CODELAB_REPAIR_ATTEMPTS", "3"))
        result, code, last_err = await self._codelab_loop(
            goal, code, focus, max_repairs
        )
        if result is not None:
            return result
        logger.warning(
            "code_lab 沙箱执行经 %d 轮修复仍未产出证据(不伪造): %s",
            max_repairs + 1,
            last_err,
        )
        # 诊断: 把最后一版(仍失败的)书生代码落日志, 便于对症改 harness/提示, 不伪造证据.
        # 用 warning 级: 默认 root logger 无 handler, info 级看不到失败代码.
        logger.warning("code_lab 最后一版书生代码:\n%s", code[:2000])
        return {
            "mode": "code_lab",
            "status": "failed",
            "success": False,
            "error": f"Code Lab 执行未通过沙箱校验: {last_err}",
            "script": code,
        }

    async def _execute(self, plan: dict[str, Any], context: dict[str, Any]) -> Any:
        """Execute the plan using the appropriate sub-engine."""
        # 本轮 execute 起点清空: 供 episodic 快照读"本轮作者提示长度"(不依赖调试开关).
        # 只有本轮真调过 build_author_prompt 才有值; 未调 → None, 不入冻结判定.
        self._last_author_prompt_len = None
        mode = plan.get("mode", "coder")
        description = plan.get("description", "")
        self._act_log_route(mode, description)

        try:
            result = await self._act_fast_paths(plan, description, context)
        except Exception:  # 防御: 探针快路失败转入通用执行
            logger.debug(
                "execute builtin-probe fast-path failed (fall through)", exc_info=True
            )
            result = None
        if result is not None:
            return result

        # 非阻塞诊断: 记录判定输入, 便于定位为何没走内建执行.
        self._act_log_probe_diagnostics(plan, description)

        result, mode = await self._act_dispatch(mode, plan, description, context)
        result, mode = await self._act_coder_fallback(
            mode, description, context, result
        )
        return self._act_finalize(mode, plan, context, result)

    def _act_log_route(self, mode: Any, description: Any) -> None:
        """HUGINN_EXEC_ROUTE_DEBUG 开启时打印路由判定输入."""
        import os as _osd

        if not _osd.environ.get("HUGINN_EXEC_ROUTE_DEBUG"):
            return
        try:
            _obj = str(getattr(self, "_objective", "") or "")
            logger.warning(
                "[exec-route] mode=%r desc[:80]=%r obj_len=%d "
                "is_exp_desc=%s is_exp_obj=%s is_det_desc=%s",
                mode,
                (description or "")[:80],
                len(_obj),
                self._is_code_experiment(description),
                self._is_code_experiment(_obj),
                self._is_deterministic_numeric(description),
            )
        except Exception:  # — 原因: 路由诊断日志失败不影响执行, 忽略
            pass

    async def _act_fast_paths(
        self, plan: dict[str, Any], description: Any, context: dict[str, Any]
    ) -> Any | None:
        """内建执行快路: 闭式片段 → Code Lab → 数值 probe; 命中即返回."""
        result = self._act_closed_form_fast_path(plan)
        if result is not None:
            return result
        result = await self._act_code_lab_fast_path(plan, description, context)
        if result is not None:
            return result
        return await self._act_numeric_probe_fast_path(plan, description)

    def _act_closed_form_fast_path(self, plan: dict[str, Any]) -> Any | None:
        """plan 已带可运行数值脚本片段 → 直接真实执行成证据."""
        _snip = self._extract_run_snippet(plan)
        if not _snip:
            return None
        _cout = self._run_snippet_to_output(_snip)
        if not _cout:
            return None
        result = {
            "mode": "closed_form_exec",
            "status": "completed",
            "success": True,
            "result": _cout.strip(),
            "script": _snip,
            "reproducible": True,
        }
        self._record_provenance("closed_form_exec", plan, result)
        self._record_internal_tool_call("closed_form_exec", plan, result)
        self._last_execution_result = {
            "_tool_name": "closed_form_exec",
            "_tool_input": plan,
            "result": result,
        }
        logger.info(
            "execute closed-form fast-path: plan 内含可运行数值片段, 已真实执行 → evidence"
        )
        return result

    async def _act_code_lab_fast_path(
        self, plan: dict[str, Any], description: Any, context: dict[str, Any]
    ) -> Any | None:
        """目标是计算实验类 → 书生亲写 run(cfg), Code Lab 沙箱真跑."""
        # 顺序: 实验意图优先于平凡 probe, 二者级联(if 而非 elif): 失败仍可退到 probe.
        if not (
            self._is_code_experiment(description)
            or self._is_code_experiment(str(getattr(self, "_objective", "") or ""))
        ):
            return None
        # 目标要求外部取数/落盘时, code_lab 没有 open()/网络 import → 必然只能
        # 拿合成 stand-in 交差. 这类目标不走快路, 交给 dispatch 里的 coder 真跑.
        if self._needs_external_data_io(description) or self._needs_external_data_io(
            str(getattr(self, "_objective", "") or "")
        ):
            logger.info(
                "code_lab fast-path skipped: 目标要求真实数据获取/落盘 (code_lab 无 open/无网络)"
            )
            return None
        _lab = await self._execute_code_lab(description, context)
        # 成败都记: 失败的内建实验也占用了真实算力, 不能从遥测里消失.
        self._record_internal_tool_call("code_lab", plan, _lab)
        if not _lab.get("success"):
            return None
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

    async def _act_numeric_probe_fast_path(
        self, plan: dict[str, Any], description: Any
    ) -> Any | None:
        """确定性数值目标 → harness 内建生成并执行最小 probe."""
        if not (
            self._is_deterministic_numeric(description)
            or self._is_deterministic_numeric(
                str(getattr(self, "_objective", "") or "")
            )
        ):
            return None
        # 以可靠文本源为准: plan.description 可能为空, 用 objective 兜底.
        _src = (
            description
            if self._is_deterministic_numeric(description)
            else str(getattr(self, "_objective", "") or "")
        )
        # 需要外部取数的目标: probe 只会在内存里现编一段算式的数值, 拿不到真实
        # 数据也落不了盘 —— 用它顶替 execute 就是"合成代替真实"的老毛病.
        # 交给 coder 真跑(下载+落盘+分析).
        if self._needs_external_data_io(
            description
        ) or self._needs_external_data_io(str(getattr(self, "_objective", "") or "")):
            logger.info(
                "numeric-probe fast-path skipped: 目标要求真实数据获取/落盘 (probe 无 I/O)"
            )
            return None
        _probe = await self._request_numeric_probe(_src)
        if not _probe:
            return None
        _cout = self._run_snippet_to_output(_probe)
        if not _cout:
            return None
        result = {
            "mode": "probe_exec",
            "status": "completed",
            "success": True,
            "result": _cout.strip(),
            "script": _probe,
            "reproducible": True,
        }
        self._record_provenance("probe_exec", plan, result)
        self._record_internal_tool_call("probe_exec", plan, result)
        self._last_execution_result = {
            "_tool_name": "probe_exec",
            "_tool_input": plan,
            "result": result,
        }
        logger.info(
            "execute builtin probe: 确定性数值目标 → harness 内建生成并执行 probe → evidence"
        )
        return result

    def _act_log_probe_diagnostics(
        self, plan: dict[str, Any], description: Any
    ) -> None:
        """非阻塞诊断: 判定到确定性目标却没产出 probe 证据时留痕."""
        try:
            _det_desc = self._is_deterministic_numeric(description)
            _det_obj = self._is_deterministic_numeric(
                str(getattr(self, "_objective", "") or "")
            )
            if not (_det_desc or _det_obj):
                return
            logger.info(
                "[execute-probe] 判定到确定性目标但未产出探针证据: "
                "det_desc=%s det_obj=%s, desc[:40]=%r, obj[:40]=%r, _tool=%r",
                _det_desc,
                _det_obj,
                (description or "")[:40],
                (str(getattr(self, "_objective", "") or ""))[:40],
                plan.get("mode"),
            )
        except Exception:  # — execute 分流探针失败则继续走通用执行, 不阻断
            pass

    async def _act_dispatch(
        self,
        mode: str,
        plan: dict[str, Any],
        description: Any,
        context: dict[str, Any],
    ) -> tuple[Any, str]:
        """按 mode 分派到对应执行器, 返回 (result, 生效 mode)."""
        # H4: toggle on 时从 PhaseRegistry 取 dispatch_table 替代 hardcode if/elif.
        # ponytail: dispatch_table 存 [method_name, arg_mode], arg_mode 决定传
        # plan 还是 description (executor 签名不统一, 不改签名最小 diff).
        from huginn.harness.phase_spec import get_phase_dispatch_table

        dispatch = get_phase_dispatch_table()
        if dispatch is not None:
            return await self._act_dispatch_from_table(
                dispatch, mode, plan, description, context
            )
        # 显式等值判断 (而非元组成员) 保留静态可见性: contract_audit 的"工作流面"
        # 按 mode == "<mode>" 正则抽取执行面分支, 元组成员会让 code_lab 消失.
        is_primary = (
            mode == "coder"
            or mode == "workflow"
            or mode == "dynamic_workflow"
            or mode == "explore"
            or mode == "code_lab"
        )
        if is_primary:
            return await self._act_dispatch_primary(mode, plan, description, context)
        return await self._act_dispatch_inspect(mode, plan, description, context)

    async def _act_dispatch_from_table(
        self,
        dispatch: Any,
        mode: str,
        plan: dict[str, Any],
        description: Any,
        context: dict[str, Any],
    ) -> tuple[Any, str]:
        """dispatch_table 路径: entry = [method_name, arg_mode]."""
        entry = dispatch.get(mode)
        if entry is None:
            raise ValueError(f"Unknown plan mode: {mode}")
        method_name, arg_mode = entry[0], entry[1]
        executor = getattr(self, method_name)
        arg = plan if arg_mode == "plan" else description
        result = await executor(arg, context)
        # workflow 失败走 evolved_fix (原 hardcode 逻辑保留)
        if (
            mode == "workflow"
            and isinstance(result, dict)
            and not result.get("success", True)
        ):
            result = await self._try_evolved_fix(mode, description, result) or result
        return result, mode

    async def _act_dispatch_primary(
        self,
        mode: str,
        plan: dict[str, Any],
        description: Any,
        context: dict[str, Any],
    ) -> tuple[Any, str]:
        """coder / workflow / dynamic_workflow / explore / code_lab."""
        if mode == "coder":
            # Use CoderRunner to modify code
            return await self._execute_coder(description, context), mode
        # 目标要求真实数据获取/落盘时, 除 coder 外的模式都缺"手": workflow 只跑
        # 模板管线, dynamic_workflow 只并发预置 subtask, explore 只搜设计空间,
        # skill 是预置流程, code_lab 无 open()/无网络 import. ws16 实测 plan 把
        # "下载 UCI CSV 再分析"写成 MODE: workflow → 下载根本不会发生.
        # 一律改道 coder: 唯一带 bash_tool + file_write 的真实执行路径.
        if self._needs_external_data_io(description) or self._needs_external_data_io(
            str(getattr(self, "_objective", "") or "")
        ):
            logger.info(
                "dispatch %s→coder: 目标要求真实数据获取/落盘 (该模式无 I/O 能力)",
                mode,
            )
            return await self._execute_coder(description, context), "coder"
        if mode == "workflow":
            # Use WorkflowEngine to run computational pipeline
            result = await self._execute_workflow(description, context)
            # On failure, try applying a learned heuristic fix before giving up
            if isinstance(result, dict) and not result.get("success", True):
                result = (
                    await self._try_evolved_fix(mode, description, result) or result
                )
            return result, mode
        if mode == "dynamic_workflow":
            # A5: agent 写的并行 subtask 脚本, orchestrator 并发跑
            return await self._execute_dynamic_workflow(plan, context), mode
        if mode == "explore":
            # Use ExplorationOrchestrator to search design space
            result = await self._execute_explore(description, context)
            # 空转兜底: 无证据却有"成功"外形 → 确定性数值目标回落 coder 真写码.
            if self._act_explore_is_idle(result, description):
                logger.info("explore 空转 (无证据) → 回落 coder 真实执行")
                result = await self._execute_coder(description, context)
                mode = "coder"
            return result, mode
        # 书生亲写一段计算实验 → Code Lab 安全沙箱真跑 → 结构化证据.
        return await self._execute_code_lab(description, context), mode

    def _act_explore_is_idle(self, result: Any, description: Any) -> bool:
        """explore 是否无证据空转, 且目标是确定性数值计算."""
        if not isinstance(result, dict):
            return False
        empty = result.get("success") is False or (
            not result.get("result") and int(result.get("n_explored", 0) or 0) == 0
        )
        deterministic = self._is_deterministic_numeric(
            description
        ) or self._is_deterministic_numeric(str(getattr(self, "_objective", "") or ""))
        return empty and deterministic

    async def _act_dispatch_inspect(
        self,
        mode: str,
        plan: dict[str, Any],
        description: Any,
        context: dict[str, Any],
    ) -> tuple[Any, str]:
        """skill / visual_inspect / trace_inspect / frame_inspect."""
        if mode == "skill":
            # Run a pre-built composite skill pipeline
            return await self._execute_skill(plan, context), mode
        if mode == "visual_inspect":
            # A3 + 批次 D: consistency_check 默认开启 (闭环), 除非显式关闭.
            consistency = not (
                context.get("consistency_check") is False
                or "no consistency" in plan.lower()
                or "consistency_check=false" in plan.lower()
            )
            result = await self._execute_visual_inspect(
                description, context, consistency_check=consistency
            )
            return result, mode
        if mode == "trace_inspect":
            # VISTA 借鉴: 模型主动 recall 自己的过程级执行 trace (read-only).
            return await self._execute_trace_inspect(description, context), mode
        if mode == "frame_inspect":
            # VISTA 借鉴: 无损帧回看 / 区域裁剪 / read_pixels (flag 关时返回 disabled).
            return await self._execute_frame_inspect(description, context), mode
        raise ValueError(f"Unknown plan mode: {mode}")

    async def _act_coder_fallback(
        self, mode: str, description: Any, context: dict[str, Any], result: Any
    ) -> tuple[Any, str]:
        """coder 未产出可用结果 → 按 code_lab → probe_exec 顺序自动回落."""
        # P1: coder 失败不再直接落盘成终态, 都不行才保留原始失败回执.
        if (
            mode != "coder"
            or not isinstance(result, dict)
            or result.get("success", True)
        ):
            return result, mode
        fallback = await self._fallback_from_coder(description, context)
        if fallback is not None:
            result = fallback
            mode = str(fallback.get("mode", mode))
        return result, mode

    def _act_finalize(
        self, mode: str, plan: dict[str, Any], context: dict[str, Any], result: Any
    ) -> Any:
        """provenance / 内建 tool_call / 视觉帧入库 / alignment / 结果缓存."""
        # provenance: 记一次 tool call, mode 当工具名, plan 当输入参数
        self._record_provenance(mode, plan, result)
        # dispatch / 回落阶段的内建执行成功路径: 同样补 tool_call span,
        # 与快路的记法一致 (快路已提前 return, 不会重复).
        if mode in self._INTERNAL_EXEC_MODES:
            self._record_internal_tool_call(mode, plan, result)
        # VISTA 借鉴: 本轮若产出视觉帧, 原样落盘进无损观测记忆.
        # frame_inspect 本身是"回看已有帧", 不重复入库.
        if mode != "frame_inspect":
            self._recall_modes.capture_if_enabled(mode, result)
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

    #: 引擎内建执行快路 (不经 ToolAdapter 的 LLM function call).
    #: 这些路径真实跑代码/数值, 必须进 trajectory.tool_calls, 否则利用率被低估.
    _INTERNAL_EXEC_MODES: frozenset[str] = frozenset(
        {"code_lab", "probe_exec", "closed_form_exec"}
    )

    def _record_internal_tool_call(
        self, tool_name: str, input_params: Any, output: Any
    ) -> None:
        """把内建执行快路记成一条与 adapter 同形的 tool_call span.

        code_lab / probe_exec / closed_form_exec 不经过 ToolAdapter, 以前只在
        provenance.tool_chain 留痕 → trajectory.tool_calls 恒为 0: 利用率被系统性
        低估, G31 审计还会把"真跑了实验"误判成空转. success 口径与
        `_derive_coder_outcome` 一致: 显式 False 或带 error 即失败.
        """
        try:
            from huginn.telemetry import get_telemetry_collector
            from huginn.tools.adapter import _truncate_for_trajectory

            ok = not (
                isinstance(output, dict)
                and (output.get("success") is False or output.get("error"))
            )
            with get_telemetry_collector().span("tool_call", tool=tool_name) as span:
                span.metadata["success"] = ok
                span.metadata["stage"] = "internal"
                span.metadata["args"] = _truncate_for_trajectory(input_params)
                span.metadata["result"] = _truncate_for_trajectory(output)
                if not ok and isinstance(output, dict):
                    span.metadata["error"] = str(
                        output.get("error") or "internal execution failed"
                    )[:2000]
        except Exception:  # 遥测 fail-open, 绝不阻断执行
            logger.debug("internal tool_call span failed (fail-open)", exc_info=True)

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

        ARTEX 借鉴: 额外带 ``intent`` (本轮假设 id, 即"方向") 与 ``ts`` —— 让台账
        从"报告 citation 的数值面"升级为**可按方向检索的过程级 trace**
        (ARTEX 的 ``get_worker_trace(intent_id, step_ids)`` 同构).
        """
        try:
            ledger = getattr(self, "_execution_ledger", None)
            if ledger is None:  # 引擎未初始化该字段 (老替身) → 跳过
                return
            ledger.append(
                {
                    "idx": len(ledger) + 1,
                    "tool": tool_name,
                    "intent": str(getattr(self, "_current_hyp_id_for_plan", "") or ""),
                    "ts": time.time(),
                    # 执行健康标签 (只读): 把"成功/空跑/工具报错/超时/环境错误"分开,
                    # 避免把工具故障误判成假设被证伪. 不改控制流, 仅作台账元信息.
                    "exit_class": classify_exit_class(output),
                    "result": _ledger_text(output),
                }
            )
            if len(ledger) > _EXEC_LEDGER_MAX:
                del ledger[: len(ledger) - _EXEC_LEDGER_MAX]
        except Exception:  # 防御: 台账 best-effort, 挂了不能带挂 execute
            logger.debug("append execution ledger failed", exc_info=True)

    # ── ARTEX 借鉴: 过程级 trace 检索 ────────────────────────────────────────
    # ARTEX 的 worker 能检索同僚的**过程级**执行日志 (search_all_worker_traces /
    # list_worker_traces / get_worker_trace), 把"特殊报错/隐藏参数"这类未进官方
    # fact 的线索在 worker 间复用. 本仓库对应物是 execution_ledger (每次 execute
    # 一条紧凑快照). 这里补上按 关键词 / tool / intent 的检索口 + exclude_self
    # (不检索自己刚写的那条, 避免自证). 纯读, fail-open.

    def list_execution_traces(self) -> list[dict[str, Any]]:
        """按追加顺序列出全部过程级 trace (深拷贝浅层字段, 防外部改写台账)."""
        ledger = getattr(self, "_execution_ledger", None) or []
        return [dict(e) for e in ledger]

    def get_execution_trace(self, idx: int) -> dict[str, Any] | None:
        """按稳定 idx 取一条 trace; 不存在 → None."""
        for e in getattr(self, "_execution_ledger", None) or []:
            if e.get("idx") == idx:
                return dict(e)
        return None

    def search_execution_traces(
        self,
        query: str = "",
        *,
        tool: str = "",
        intent: str = "",
        exclude_self: bool = True,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """检索过程级 trace (ARTEX ``search_all_worker_traces`` 同构).

        - ``query``: 关键词 (大小写不敏感), 在 result/tool/intent 文本里匹配.
        - ``tool`` / ``intent``: 精确过滤.
        - ``exclude_self=True``: 剔除 intent == 当前方向的条目 (不检索自己).
        - ``limit``: 最多返回条数 (默认 8), 最近的优先.
        纯读 fail-open: 任何异常返回 [].
        """
        try:
            ledger = getattr(self, "_execution_ledger", None) or []
            self_intent = str(getattr(self, "_current_hyp_id_for_plan", "") or "")
            q = (query or "").strip().lower()
            out: list[dict[str, Any]] = []
            for e in ledger:
                if tool and e.get("tool") != tool:
                    continue
                if intent and e.get("intent") != intent:
                    continue
                if exclude_self and self_intent and e.get("intent") == self_intent:
                    continue
                if q:
                    hay = (
                        f"{e.get('tool', '')} {e.get('intent', '')} "
                        f"{e.get('result', '')}"
                    ).lower()
                    if q not in hay:
                        continue
                out.append(dict(e))
            # 最近优先; 返回时按时间正序 (调用方按 idx 递增阅读更自然).
            out = out[-limit:]
            return out
        except Exception:  # 防御: 检索 best-effort, 挂了返回空
            logger.debug("search execution traces failed", exc_info=True)
            return []

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
                return await self._execute_workflow(
                    patched_desc, {"_evolved_fix": True}
                )
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

        script = self._dynwf_script(plan)
        if not script.subtasks:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        return await self._dynwf_run(script)

    @staticmethod
    def _script_dict(plan: dict[str, Any]) -> dict[str, Any]:
        """plan 的 script 字段 → dict (agent 可能传 JSON 字符串)."""
        raw = plan.get("script") or {}
        if isinstance(raw, str):
            import json

            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        return raw

    def _dynwf_script(self, plan: dict[str, Any]) -> Any:
        """plan → WorkflowScript."""
        from huginn.autoloop.dynamic_workflow import WorkflowScript

        return WorkflowScript.from_dict(self._script_dict(plan))

    async def _dynwf_run(self, script: Any) -> dict[str, Any]:
        """建 orchestrator + ToolContext 跑脚本, 返回聚合结果."""
        from huginn.autoloop.dynamic_workflow import WorkflowOrchestrator
        from huginn.core_types import ToolContext

        orch = WorkflowOrchestrator(max_concurrent=script.max_concurrent)
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

    def _bandit_base(
        self, plan: dict[str, Any]
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """plan → base WorkflowScript; 解析失败/无 subtask 时返回失败回执."""
        from huginn.autoloop.dynamic_workflow import WorkflowScript

        try:
            base_script = WorkflowScript.from_dict(self._script_dict(plan))
        except Exception:  # 防御: 动态工作流失败返回失败态
            return None, {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "base script parse fail",
            }
        if not base_script.subtasks:
            return None, {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        return base_script, None

    def _bandit_fail_payload(
        self,
        exc: Exception,
        chosen: Any,
        variant_id: str,
        obj_hash: str,
        objective: str,
    ) -> dict[str, Any]:
        """variant 跑挂时的失败回执 (保留溯源字段供 _validate 记账)."""
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

    async def _execute_dynamic_workflow_bandit(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """H2: 对同一 objective 生成 N 个 variant, bandit 选一个跑.

        generate_variants → bandit.select_variant → orch.run →
        返回 dict 带 _variant_id / _objective_hash / _novelty / _efficiency,
        供 _validate 调 bandit.record_variant_outcome + archive.add_variant.
        失败静默回退到原 _execute_dynamic_workflow 逻辑.
        """
        from huginn.autoloop.bandit import _objective_hash

        base_script, fail = self._bandit_base(plan)
        if base_script is None:
            return fail
        objective = base_script.objective or str(plan.get("objective", ""))
        obj_hash = _objective_hash(objective)
        variants = await self._bandit_variants(
            objective, int(plan.get("n_variants", 3)), base_script
        )
        chosen, variant_id = self._bandit_select(variants, obj_hash)
        try:
            result = await self._bandit_run(chosen)
        except Exception as exc:
            return self._bandit_fail_payload(
                exc, chosen, variant_id, obj_hash, objective
            )
        return self._bandit_payload(
            result,
            chosen=chosen,
            variant_id=variant_id,
            obj_hash=obj_hash,
            objective=objective,
            novelty=self._bandit_novelty(chosen, obj_hash),
        )

    async def _bandit_variants(
        self, objective: str, n: int, base_script: Any
    ) -> list[Any]:
        """生成 variants (参数扰动优先, base_script 非空走扰动); 失败回退 base."""
        from huginn.autoloop.variant_gen import generate_variants

        try:
            variants = await generate_variants(
                objective,
                n=n,
                base_script=base_script,
                llm_chat_fn=getattr(self, "_llm_chat", None),
            )
        except Exception:  # 防御: H2 变体生成失败用空集
            logger.debug("H2 generate_variants failed", exc_info=True)
            variants = []
        return variants or [base_script]

    @staticmethod
    def _bandit_select(variants: list[Any], obj_hash: str) -> tuple[Any, str]:
        """bandit 选 variant; variant_id 加时间戳前缀避免跨轮冲突."""
        from huginn.autoloop.bandit import WorkflowBandit

        _run_prefix = f"r{int(time.time() * 1000) % 100000}"
        bandit = WorkflowBandit.get_instance()
        candidate_ids = [f"{_run_prefix}_var_{i}" for i in range(len(variants))]
        chosen_id = bandit.select_variant(candidate_ids, obj_hash)
        if chosen_id is None:
            chosen_id = candidate_ids[0]
        return variants[candidate_ids.index(chosen_id)], chosen_id

    async def _bandit_run(self, chosen: Any) -> Any:
        """跑选中 variant."""
        from huginn.autoloop.dynamic_workflow import WorkflowOrchestrator
        from huginn.core_types import ToolContext

        orch = WorkflowOrchestrator(max_concurrent=chosen.max_concurrent)
        ctx = ToolContext(
            session_id=f"dynwf_{chosen.id}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        return await orch.run(chosen, ctx, phase="_execute")

    @staticmethod
    def _bandit_novelty(chosen: Any, obj_hash: str) -> float:
        """novelty vs archive; 计算失败取零."""
        from huginn.autoloop.bandit import VariantArchive, compute_novelty

        try:
            existing = VariantArchive.get_instance().list_variants(obj_hash)
            return float(compute_novelty(chosen.to_dict(), existing))
        except Exception:  # 防御: 新颖度计算失败取零
            logger.debug("best-effort op failed", exc_info=True)
            return 0.0

    @staticmethod
    def _bandit_payload(
        result: Any,
        *,
        chosen: Any,
        variant_id: str,
        obj_hash: str,
        objective: str,
        novelty: float,
    ) -> dict[str, Any]:
        """组装 bandit 回执 (带 variant 元信息供 validate 记录)."""
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
            "_efficiency": float(result.n_completed / max(1, result.n_total)),
        }

    # ──────────────────────────────────────────────────────────────
    # Execution helpers
    # ──────────────────────────────────────────────────────────────

    #: CoderRunner 命中迭代上限时 final_answer 的固定前缀 (见 coder/loop.py run()).
    _CODER_LIMIT_PREFIX = "Coder reached the maximum iteration limit."

    def _derive_coder_outcome(
        self, final_answer: Any, messages: list[Any]
    ) -> tuple[bool, dict[str, Any]]:
        """由真实回执/产物推导 coder 是否成功, 不再硬编码 ``success=True``.

        CoderRunner.run() 只回 ``{final_answer, messages}``, 本身不含 success 字段,
        因此此前的实现无论 coder 是否空转都写死成功 (run 实测 100 次工具调用
        全部报错, 结果仍被当成 completed). 这里按三个可观测信号裁决:

        1. 至少发生过一次真实工具调用 (``AIMessage.tool_calls`` 非空);
        2. 没有命中迭代上限 (``final_answer`` 不以固定前缀开头);
        3. 至少有一次工具回执**不是错误** (``ToolMessage.content`` 为
           ``json.dumps({"error": ...})`` 的记为失败).

        允许部分工具失败 (agent 试错后修正是正常行为), 只要求有可用回执.
        返回 ``(success, diagnostics)``, 诊断字段可直接写进遥测/报告.
        """
        final = (
            final_answer if isinstance(final_answer, str) else str(final_answer or "")
        )
        n_tool_calls, n_tool_results, tool_errors = EngineAct._coder_tool_signals(
            messages
        )
        usable = n_tool_results - len(tool_errors)
        hit_limit = final.startswith(self._CODER_LIMIT_PREFIX)
        # hit_limit 只作诊断, 不再作失败判据: 数据/科研任务里 coder 常在产出可用
        # 结果后继续探索到上限 (ws22 实测: UCI 数据已下载且 sha256 与源站逐位一致,
        # 仍被判 failed → validate 反复重规划, 动作全耗在同一件事上). 判据改为
        # "发生过真实工具调用且有可用回执", 上限命中单独记录供审计区分"做完未收尾".
        success = bool(n_tool_calls) and usable > 0 and bool(final.strip())

        diag: dict[str, Any] = {
            "tool_calls": n_tool_calls,
            "tool_results": n_tool_results,
            "usable_tool_results": usable,
            "tool_error_count": len(tool_errors),
            "tool_errors": tool_errors[:5],
            "hit_iteration_limit": hit_limit,
        }
        if not success:
            diag["failure_reason"] = EngineAct._coder_failure_reason(
                hit_limit=hit_limit, n_tool_calls=n_tool_calls, usable=usable
            )
        return success, diag

    @staticmethod
    def _coder_tool_signals(messages: list[Any]) -> tuple[int, int, list[str]]:
        """扫消息: (工具调用数, 工具回执数, 错误回执列表)."""
        import json

        n_tool_calls = 0
        n_tool_results = 0
        tool_errors: list[str] = []
        for msg in messages or []:
            if getattr(msg, "tool_calls", None):
                n_tool_calls += len(msg.tool_calls)
            if getattr(msg, "type", "") != "tool":
                continue
            n_tool_results += 1
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("error"):
                name = getattr(msg, "name", "") or "?"
                tool_errors.append(f"{name}: {str(payload['error'])[:200]}")
        return n_tool_calls, n_tool_results, tool_errors

    @staticmethod
    def _coder_failure_reason(
        *, hit_limit: bool, n_tool_calls: int, usable: int
    ) -> str:
        """失败归因 (按优先级: 迭代上限 → 无调用 → 全失败 → 空答案)."""
        if hit_limit:
            return "iteration_limit"
        if not n_tool_calls:
            return "no_tool_calls"
        if usable <= 0:
            return "all_tool_calls_failed"
        return "empty_answer"

    def _coder_task_tail(self, task_desc: str) -> str:
        """coder 任务收尾指令: 数据/研究类任务明确"在工作区跑脚本", 别改源码.

        默认收尾是 "modify the code" —— 科研任务会被引去读仓库/跑 pytest
        (ws18 实测: 下载已完成后转去查 research_records、生成 .pytest_cache).
        取数类任务改成"在工作区写脚本并真跑", 产物落当前工作区.
        """
        if self._needs_external_data_io(task_desc) or self._needs_external_data_io(
            str(getattr(self, "_objective", "") or "")
        ):
            return (
                "Do this inside the current workspace: write the needed scripts and run "
                "them with bash_tool. Do NOT modify the huginn source tree or run repo "
                "test suites; all artifacts (data/, scripts, results, figures) belong in "
                "the current working directory."
            )
        return "Please modify the code to address this task."

    async def _execute_coder(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the coder loop on the description, reusing self.coder."""
        # 结构性锚点: 没有执行对象就不执行. 旧代码在 description 为空时用
        # "当前假设"兜底下发 coder —— 那等于让模型自造执行对象, 是 ws28 的
        # 假设漂移通道. 现在直接 fail-fast 出诊断, 由上层重写 plan.
        task_desc = str(description or "").strip()
        if not task_desc:
            logger.error("coder 收到空 description → fail-fast (拒绝执行)")
            return {
                "mode": "coder",
                "status": "failed",
                "success": False,
                "error": "empty_description",
                "diagnostic": "plan DESCRIPTION 为空, 执行对象不存在; 需重写 plan",
            }
        task = f"""Task: {task_desc}

Context:
- Changed files: {context.get("changed_files", [])}
- Git diff: {context.get("git_diff", "")[:500]}

{self._coder_task_tail(task_desc)}"""
        try:
            # CoderRunner.run 是同步的, 丢线程里避免阻塞事件循环
            result = await asyncio.to_thread(self.coder.run, task)
            messages = result.get("messages", [])
            final_answer = result.get("final_answer", "")
            success, outcome = self._derive_coder_outcome(final_answer, messages)
            payload: dict[str, Any] = {
                "mode": "coder",
                "status": "completed" if success else "failed",
                "success": success,
                "final_answer": final_answer,
                "tool_calls": outcome["tool_calls"],
                "coder_outcome": outcome,
            }
            if not success:
                payload["error"] = (
                    f"coder 未产出可用结果 ({outcome.get('failure_reason', 'unknown')})"
                )
            return payload
        except Exception as e:
            logger.exception("coder execution failed")
            return {
                "mode": "coder",
                "status": "failed",
                "success": False,
                "error": str(e),
            }

    async def _fallback_from_coder(
        self, description: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """coder 未产出可用结果时的降级执行链: code_lab → probe_exec.

        只在 coder 已判定失败时调用 (见 `_execute` 末尾), 因此不改变任何正常
        路径; 每条回落都必须**真实执行并产出证据**才算数, 否则继续下一条.
        全部落空返回 None, 调用方保留原始失败回执 (如实失败优于伪造成功).

        顺序与 `_execute` 快路一致: 实验意图优先 (code_lab 可写完整实验),
        平凡确定性数值再退 probe_exec.
        """
        reasons: list[str] = []
        objective = str(getattr(self, "_objective", "") or "")
        # 目标要真实数据获取/落盘时, 回落链 (code_lab → probe_exec) 结构性做不到:
        # code_lab 无 open()/无网络 import, probe 只在内存里现编算式. 让它们"成功"
        # 就是把合成数当真实证据 —— 宁可如实失败, 也不产假证据 (ws13 教训).
        if self._needs_external_data_io(description) or self._needs_external_data_io(
            objective
        ):
            logger.warning(
                "coder 失败且目标要求真实数据获取/落盘: 跳过 code_lab/probe 回落"
                "(二者无 I/O, 只会产出合成证据)"
            )
            return None
        lab = await EngineAct._fb_code_lab(self, description, context, reasons)
        if lab is not None:
            return lab
        probe = await EngineAct._fb_probe(self, description, objective, reasons)
        if probe is not None:
            return probe
        # warning 级: 回落全失败是"本可修但没修成"的信号, 必须在默认日志级别可见,
        # 否则又回到"失败静默"的老问题.
        logger.warning("coder 回落链已耗尽: %s", "; ".join(reasons))
        return None

    async def _fb_code_lab(
        self, description: str, context: dict[str, Any], reasons: list[str]
    ) -> dict[str, Any] | None:
        """coder 失败 → code_lab 真实执行; 不成功则记 reasons 返回 None."""
        try:
            lab = await self._execute_code_lab(description, context)
            if isinstance(lab, dict) and lab.get("success"):
                lab["fallback_from"] = "coder"
                logger.info("coder 未产出可用结果 → 回落 code_lab 真实执行成功")
                return lab
            # 失败尝试也要可见 (成功路径由 _execute 末尾统一补 span).
            self._record_internal_tool_call("code_lab", {"fallback_from": "coder"}, lab)
            reasons.append("code_lab: " + str((lab or {}).get("error", "no evidence")))
        except Exception as exc:  # 防御: 回落本身失败不能带挂 execute
            logger.warning("coder→code_lab fallback failed", exc_info=True)
            reasons.append(f"code_lab: {exc}")
        return None

    async def _fb_probe(
        self, description: str, objective: str, reasons: list[str]
    ) -> dict[str, Any] | None:
        """coder 失败 → 平凡数值 probe_exec; 非确定性目标直接记 skip."""
        if not (
            self._is_deterministic_numeric(description)
            or self._is_deterministic_numeric(objective)
        ):
            reasons.append("probe_exec: 目标非确定性数值, 跳过")
            return None
        src = description if self._is_deterministic_numeric(description) else objective
        try:
            probe = await self._request_numeric_probe(src)
            if probe:
                cout = self._run_snippet_to_output(probe)
                if cout:
                    logger.info("coder 未产出可用结果 → 回落 probe_exec 真实执行成功")
                    return {
                        "mode": "probe_exec",
                        "status": "completed",
                        "success": True,
                        "result": cout.strip(),
                        "script": probe,
                        "reproducible": True,
                        "fallback_from": "coder",
                    }
            reasons.append("probe_exec: 未产出可执行片段或执行无输出")
        except Exception as exc:  # 防御: 同上, 纯回落路径 fail-open
            logger.warning("coder→probe_exec fallback failed", exc_info=True)
            reasons.append(f"probe_exec: {exc}")
        return None

    # domain → 默认模板名; get_template 拿不到就 fallback standard_dft
    # ponytail: 硬编码映射表, 新模板加一行即可; 想自动发现就扫 WORKFLOW_TEMPLATES
    _DOMAIN_TEMPLATE_NAMES = {
        "cfd": "turbulent_flow",
        "fea": "structural_analysis",
        "qc": "wavefunction_analysis",
        "symbolic": "constitutive_derivation",
        "dft": "standard_dft",
    }

    #: DFT 域的**显式**关键词. 只有描述里真出现这些词才认 DFT —— 绝不因"没匹配上
    #: 别的域"就默认 DFT: 旧实现 ``return "dft"`` 会让纯 ML/数学命题(如"解空间刚性")
    #: 被静默塞进 standard_dft 模板, 报告随之为 DFT workflow 写 Methods/Results
    #: (run88 实测域漂移: 目标是小前馈网络, 报告却在讲 scf/band/dos).
    _DFT_DOMAIN_KEYWORDS = (
        "dft",
        "density functional",
        "vasp",
        "quantum espresso",
        "ab initio",
        "第一性原理",
        "scf",
        "band structure",
        "band gap",
        "density of states",
        "k-point",
        "kpoint",
        "pseudopotential",
        "electronic structure",
        "能带",
        "态密度",
        "电子结构",
    )

    def _classify_workflow_domain(self, description: str) -> str:
        """廉价关键词分类, 决定走哪个 workflow 模板; **认不出返回 ""**(不默认 DFT).

        返回 "" 表示"这不是已知的 cfd/fea/qc/symbolic/dft 计算域", 调用方
        ``_execute_workflow`` 据此拒绝伪造 domain 流程, 交上层回落 —— 宁可如实
        失败, 也不把无关命题套进某个物理模板 (报告漂移的根因).
        """
        text = (description or "").lower()
        if any(k in text for k in ("cfd", "fluid", "fluent", "openfoam")):
            return "cfd"
        if any(k in text for k in ("fea", "stress", "mechanical", "abaqus", "ansys")):
            return "fea"
        if any(k in text for k in ("quantum", "qc", "chemistry", "gaussian", "orca")):
            return "qc"
        if any(k in text for k in ("symbolic", "regression", "拟合")):
            return "symbolic"
        if any(k in text for k in self._DFT_DOMAIN_KEYWORDS):
            return "dft"
        return ""

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

            # 认域时同时看 plan 描述与全局 objective —— plan 可能只写动作短语,
            # 真正的领域约束常待在 objective 里.
            domain = self._classify_workflow_domain(
                description or str(getattr(self, "_objective", "") or "")
            )
            if not domain:
                # 认不出计算域 → **拒绝默认套 DFT 模板**. 旧实现 fallback 到
                # standard_dft, 于是纯 ML/数学命题的报告被写成 DFT workflow 的
                # Methods/Results (run88 实测域漂移). 如实失败, 交上层回落, 不伪造.
                logger.warning(
                    "workflow 无法归入已知计算域, 拒绝 DFT 兜底 (避免报告域漂移): "
                    "desc[:80]=%r",
                    (description or "")[:80],
                )
                return {
                    "mode": "workflow",
                    "success": False,
                    "domain": None,
                    "error": (
                        "目标无法归入 cfd/fea/qc/symbolic/dft 任一计算域, "
                        "拒绝默认 DFT 模板 (防止报告域漂移)"
                    ),
                }
            template_name = self._DOMAIN_TEMPLATE_NAMES[domain]
            template_fn = get_template(template_name) or standard_dft_workflow
            stages = self._wf_stages(domain, template_fn, template_name, description)
            tool_context = ToolContext(
                session_id=f"workflow_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            result = await self.workflow_engine.execute(stages, tool_context)
            return self._wf_payload(result, domain, stages)
        except Exception as e:
            return {"mode": "workflow", "success": False, "error": str(e)}

    def _wf_inputs(self) -> dict[str, Any]:
        """扫工作区里的结构/几何/坐标输入文件."""
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
        return {
            # 只对 DFT/QC 用 structure_path
            "structure": str(structure_files[0])
            if structure_files
            else "structure.cif",
            "geometry": geometry_files,
            "xyz": xyz_files,
        }

    @staticmethod
    def _wf_call_template(
        domain: str, template_fn: Any, description: str, inputs: dict[str, Any]
    ) -> Any:
        """不同域模板参数不一样, 按域挑一组调用."""
        if domain == "cfd":
            geo = str(inputs["geometry"][0]) if inputs["geometry"] else "geometry.stp"
            return template_fn(geometry_file=geo)
        if domain == "fea":
            geo = str(inputs["geometry"][0]) if inputs["geometry"] else "geometry.inp"
            return template_fn(geometry_file=geo)
        if domain == "qc":
            xyz = inputs["xyz"]
            return template_fn(
                structure_file=str(xyz[0]) if xyz else inputs["structure"]
            )
        if domain == "symbolic":
            # symbolic 模板要 free_energy_expr, 没法从工作区推断, 拿 description 顶
            return template_fn(free_energy_expr=description)
        return template_fn(structure_path=inputs["structure"], engine="vasp")

    def _wf_stages(
        self,
        domain: str,
        template_fn: Any,
        template_name: str,
        description: str,
    ) -> Any:
        """取模板 stages; 模板调用失败 fallback 到 standard_dft."""
        from huginn.workflows.templates import standard_dft_workflow

        inputs = self._wf_inputs()
        try:
            return self._wf_call_template(domain, template_fn, description, inputs)
        except Exception as tmpl_err:
            logger.warning(
                "workflow template %s (%s) failed: %s, fallback to standard_dft",
                template_name,
                domain,
                tmpl_err,
            )
            return standard_dft_workflow(inputs["structure"], engine="vasp")

    @staticmethod
    def _wf_payload(result: Any, domain: str, stages: Any) -> dict[str, Any]:
        """组装 workflow 回执 (stage_results 按真实字段名)."""
        # ``WorkflowResult.stages`` 是 ``dict[id, ComputationalStage]``, 直接
        # 迭代拿到的是 **id 字符串**, 不是 stage 对象; ``ComputationalStage``
        # 也只有 ``name`` / ``result``(ToolResult) 字段, 无 ``stage_name`` /
        # ``output_data``。run84 里这一处 AttributeError 被下方 except 兜成
        # ``{"success": false, "error": "'str' object has no attribute
        # 'stage_name'"}``, 使 execute 的**整条 workflow 证据归零**。取
        # ``.values()`` 并按真实字段名组装。
        return {
            "mode": "workflow",
            "success": result.success,
            "stages": len(stages),
            "domain": domain,
            "outputs": result.outputs,
            "error": result.error,
            "stage_results": [
                {
                    "name": s.name,
                    "success": (
                        bool(s.result.success)
                        if s.result is not None
                        else s.status == "completed"
                    ),
                    "output": s.result.data if s.result is not None else None,
                }
                for s in result.stages.values()
            ],
        }

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

    def _llm_chat_route(self, model: Any, task: str | None) -> Any:
        """Team 模式 task→model 路由; 显式 model 不被覆盖.

        - "reasoning"/"science" → 强模型 (云端, 发散性假设)
        - "planning" → 中档模型 (收敛, 把假设变步骤)
        - "summarize"/"format" → 便宜模型 (本地/小模型)
        - "verification" → 独立验证模型
        """
        if model is not None or task is None:
            return model
        router = getattr(self, "model_router", None)
        if router is None:
            return model
        try:
            routed = router.select(
                task,
                prefer_cheap=(task in ("summarize", "format", "archival", "planning")),
            )
            if routed is not None:
                return routed
        except Exception:  # 防御: 路由选择失败用回退模型
            logger.debug(
                "model router select failed — using fallback model",
                exc_info=True,
            )
        return model

    def _llm_chat_grill_prompt(self, sys_prompt: str | None) -> str | None:
        """GRILL active 时把 GRILL_SYSTEM_PROMPT_CN 追加到 system prompt."""
        if not self._grill_active:
            return sys_prompt
        try:
            from huginn.runtime.pre_plan_grill import GRILL_SYSTEM_PROMPT_CN

            sys_prompt = (sys_prompt or "") + "\n\n" + GRILL_SYSTEM_PROMPT_CN
            self._grill_turns += 1
            # 退出检查: 输出含确认标记由上层判断; ceiling: 简单计数.
            if self._grill_turns > 20:
                logger.warning("GRILL 超过 20 轮, 强制退出 (可能 LLM 没理解确认标记)")
                self._grill_active = False
        except ImportError:
            logger.debug("pre_plan_grill import failed, GRILL prompt 跳过")
        return sys_prompt

    def _llm_chat_plugin_segment(self, sys_prompt: str | None) -> str | None:
        """拼插件 prompt 段 (排除框架骨架段, 避免与 persona/phase 重复)."""
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
                return f"{sys_prompt}\n\n{_plugin_seg}" if sys_prompt else _plugin_seg
        except Exception:  # 防御: 段组装失败退回 persona-only, 不阻塞主路径
            logger.debug("plugin prompt segments failed", exc_info=True)
        return sys_prompt

    def _llm_chat_system_message(self, llm: Any, persona_name: str | None) -> Any:
        """persona system prompt + GRILL + 插件段 → SystemMessage (可 None).

        persona 找不到就退化不注入; 组装失败不影响主 LLM 调用.
        """
        if not persona_name:
            return None
        from langchain_core.messages import SystemMessage

        sys_prompt = self._persona_system_prompt(persona_name)
        sys_prompt = self._llm_chat_grill_prompt(sys_prompt)
        sys_prompt = self._llm_chat_plugin_segment(sys_prompt)
        if not sys_prompt:
            return None
        sys_msg = SystemMessage(content=sys_prompt)
        # 静态 system prompt 跨调用不变, 给 Anthropic 打 cache 标记.
        # Kimi/Moonshot 走 OpenAI 协议, cache_control 无效, 不打标记.
        _ident = f"{type(llm).__name__}{getattr(llm, 'model', '')}".lower()
        if any(k in _ident for k in ("anthropic", "claude")):
            sys_msg.additional_kwargs["cache_control"] = {"type": "ephemeral"}
        return sys_msg

    def _llm_chat_apply_effort(self, prompt: str) -> str:
        """按 current phase 注入思考深度指令 (Inkling 启发, prompt 层)."""
        from huginn.autoloop.engine import _PHASE_THINKING_EFFORT, _effort_to_prompt

        if not self._current_phase:
            return prompt
        effort = _PHASE_THINKING_EFFORT.get(self._current_phase, 0.5)
        directive = _effort_to_prompt(effort)
        if not directive:
            return prompt
        return f"[Thinking effort: {directive}]\n\n{prompt}"

    @staticmethod
    def _llm_chat_should_stream(llm: Any, cb: Any) -> bool:
        """有 progress 回调 + provider 支持 astream + 开关开启才流式."""
        from huginn.autoloop.engine import _autoloop_streaming_enabled

        if cb is None or not hasattr(llm, "astream"):
            return False
        return _autoloop_streaming_enabled()

    async def _llm_chat_invoke(self, llm: Any, messages: list[Any]) -> str:
        """非流式 ainvoke + usage 记账.

        with_retry: 429/限流(含 400 包装)/529/网络抖动 分级退避重试, 不可重试
        错误原样抛出 (语义与改动前一致, 只是多了瞬时错误的韧性).
        """
        response = await with_retry(lambda: llm.ainvoke(messages), source="autoloop")
        await self._track_llm_usage(getattr(response, "usage_metadata", None))
        return str(response.content)

    async def _llm_chat_stream(self, llm: Any, messages: list[Any], cb: Any) -> str:
        """流式累积 content, 同时推 thinking chunk 到 WS; 失败回退 ainvoke."""
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
                        await cb(
                            {
                                "type": "autoloop_thinking",
                                "phase": self._current_phase or "decider",
                                "delta": _delta[:200],  # 截断防超大 delta
                            }
                        )
            await self._track_llm_usage(_usage_meta)
            return "".join(parts)
        except Exception as e:
            # 流式失败回退 ainvoke (某些 provider astream 实现有 bug).
            logger.debug("astream failed, fallback to ainvoke: %s", e)
            return await self._llm_chat_invoke(llm, messages)

    async def _llm_chat(
        self,
        prompt: str,
        persona_name: str | None = None,
        model: Any = None,
        task: str | None = None,
        tools: list[Any] | None = None,
        max_tool_rounds: int = 2,
    ) -> str:
        """Send a prompt to the LLM and return the response.

        persona_name 非空时注入该 persona 的 system prompt; model 显式优先,
        否则按 task 走 model_router; 有 tools 走有界工具循环, 否则流式/ainvoke.
        """
        from langchain_core.messages import HumanMessage

        model = self._llm_chat_route(model, task)
        llm = model or self.model
        messages: list[Any] = []
        sys_msg = self._llm_chat_system_message(llm, persona_name)
        if sys_msg is not None:
            messages.append(sys_msg)
        # Controllable thinking effort: 按 current phase 注入思考深度指令.
        # 无 _current_phase (非 phase 上下文调用, 如 _feynman_learn) 时不注入.
        messages.append(HumanMessage(content=self._llm_chat_apply_effort(prompt)))
        # P2-1: plan/validate 等"推理型"阶段可挂只读工具. 有工具时走有界工具循环
        # (disable 流式: tool_calls 需完整 AIMessage, 增量 chunk 无法可靠拼装).
        # 无工具时逐字节等价原路径.
        if tools:
            return await self._llm_chat_with_tools(
                llm, messages, tools, max_tool_rounds=max_tool_rounds
            )
        # 700 万步场景: decider 思考过程实时可见, 不再黑盒. fail 回退 ainvoke.
        # ponytail: 只在 progress_cb 存在时流式, 否则 ainvoke (兼容无 WS 场景).
        from huginn.core_types import progress_cb as _progress_cb

        _cb = _progress_cb.get(None)
        if not self._llm_chat_should_stream(llm, _cb):
            return await self._llm_chat_invoke(llm, messages)
        return await self._llm_chat_stream(llm, messages, _cb)

    async def _llm_chat_with_tools(
        self,
        llm: Any,
        messages: list[Any],
        tools: list[Any],
        max_tool_rounds: int = 2,
    ) -> str:
        """有界工具循环: 让推理型阶段 (plan/validate) 能真正调用只读工具.

        provider 不支持 ``bind_tools`` 时 fail-open 退化为无工具单次调用 ——
        绝不让"多给一个能力"变成"原本能跑的阶段跑不动". 每轮最多执行模型返回
        的全部 tool_calls, 轮数封顶 ``max_tool_rounds``, 之后强制取最后一条
        文本, 保证一定终止且 token 有界.
        """
        try:
            bound = llm.bind_tools(tools)
        except Exception as exc:  # 防御: 无 function-calling 的 provider 退回原行为
            logger.info("bind_tools unsupported (%s) — 退化为无工具调用", exc)
            response = await with_retry(
                lambda: llm.ainvoke(messages), source="autoloop"
            )
            await self._track_llm_usage(getattr(response, "usage_metadata", None))
            return str(response.content)

        history = list(messages)
        last_text = ""
        for _ in range(max(0, max_tool_rounds) + 1):
            response = await with_retry(
                lambda: bound.ainvoke(history), source="autoloop"
            )
            await self._track_llm_usage(getattr(response, "usage_metadata", None))
            content = getattr(response, "content", "")
            last_text = content if isinstance(content, str) else str(content)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                return last_text
            history.append(response)
            for call in calls:
                history.append(await self._run_phase_tool_call(tools, call))
            logger.info(
                "[%s] 阶段工具调用 %d 次: %s",
                self._current_phase or "phase",
                len(calls),
                ", ".join(str(c.get("name", "?")) for c in calls),
            )
        # 轮数耗尽: 返回最后一轮文本 (可能为空), 不无限续跑
        return last_text

    async def _run_phase_tool_call(self, tools: list[Any], call: dict[str, Any]) -> Any:
        """执行单个阶段工具调用, 失败也回一条 ToolMessage (错误回执可自修)."""
        import json

        from langchain_core.messages import ToolMessage

        name = str(call.get("name", "") or "")
        args = call.get("args") or {}
        call_id = str(call.get("id", "") or name or "phase_tool")
        tool = next((t for t in tools if getattr(t, "name", "") == name), None)
        if tool is None:
            payload: Any = {"error": f"unknown or unavailable tool: {name}"}
        else:
            try:
                payload = await tool.ainvoke(args)
            except Exception as exc:  # 防御: 工具异常转结构化回执, 不中断循环
                logger.debug("phase tool %s failed", name, exc_info=True)
                payload = {"error": f"{type(exc).__name__}: {exc}"}
        content = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, default=str, ensure_ascii=False)
        )
        return ToolMessage(content=content, tool_call_id=call_id, name=name)
