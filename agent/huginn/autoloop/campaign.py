"""Campaign manager — 多实验编排 + 假设闭环驱动.

R4 (W4): 把一组相关 Experiment 串成 Campaign, 每个 Experiment 绑定一个假设.
CampaignManager 按串行 / 并行 / 依赖图编排实验, 跑完后:
- 假设支持 → graph.support()
- 假设反驳 → graph.refute() + refine_failed() 产出新假设 + 自动入队新 Experiment
- 实验崩溃 → 标 failed, 不动假设图

跟 HypothesisGraph (R5) 协同形成"假设-实验-修正"闭环.
runner 是注入的 async callable, 测试用 mock, 生产用 AutoloopEngine.run.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from huginn.autoloop.hypothesis_loop import HypothesisGraph


ExperimentStatus = Literal["pending", "running", "completed", "failed"]
CampaignStatus = Literal["active", "completed", "aborted"]
RunMode = Literal["serial", "parallel", "dag"]


# ── data structures ──────────────────────────────────────────────────────────


@dataclass
class Experiment:
    """单个实验: 绑定一个假设, 跑一次 autoloop."""

    id: str
    hypothesis_id: str
    objective: str
    status: ExperimentStatus = "pending"
    # 依赖的实验 id (dag 模式用); 串行/并行忽略
    depends_on: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "objective": self.objective,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "result": dict(self.result),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Campaign:
    """一组相关实验, 共享假设图."""

    id: str
    name: str
    objective: str
    graph: HypothesisGraph
    experiments: dict[str, Experiment] = field(default_factory=dict)
    status: CampaignStatus = "active"
    created_at: str = ""
    # refine 出来的新实验自动追加到这里, run_campaign 循环消费
    _queue: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "graph": self.graph.to_dict(),
            "experiments": {k: v.to_dict() for k, v in self.experiments.items()},
            "status": self.status,
            "created_at": self.created_at,
        }


# runner 协议: 接收 Experiment, 返回 result dict.
# result 必须含 "success" (bool); 可选 "evidence" / "outputs" / "hypothesis_supported".
RunnerFn = Callable[[Experiment], Awaitable[dict[str, Any]]]


class CampaignError(Exception):
    """campaign 操作错误."""


# ── manager ──────────────────────────────────────────────────────────────────


class CampaignManager:
    """创建 / 编排 / 聚合 campaign.

    典型流程:
        mgr = CampaignManager()
        cid = mgr.create_campaign("LJ 优化", "找最低能态")
        h1 = mgr.campaign(cid).graph.add_hypothesis("如果...", prediction="...")
        mgr.add_experiment(cid, hypothesis_id=h1, objective="跑 MD")
        results = await mgr.run_campaign(cid, runner=my_runner)
        summary = mgr.aggregate(cid)
    """

    def __init__(self, model: Any | None = None) -> None:
        self._campaigns: dict[str, Campaign] = {}
        self._model = model  # 传给 graph.refine_failed 做 LLM 增强

    def campaign(self, campaign_id: str) -> Campaign:
        if campaign_id not in self._campaigns:
            raise CampaignError(f"campaign {campaign_id} 不存在")
        return self._campaigns[campaign_id]

    def create_campaign(self, name: str, objective: str) -> str:
        cid = f"c_{uuid.uuid4().hex[:8]}"
        campaign = Campaign(
            id=cid,
            name=name,
            objective=objective,
            graph=HypothesisGraph(),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self._campaigns[cid] = campaign
        return cid

    def add_experiment(
        self,
        campaign_id: str,
        hypothesis_id: str,
        objective: str,
        depends_on: list[str] | None = None,
    ) -> str:
        """加一个实验到 campaign. 返回 experiment id."""
        c = self.campaign(campaign_id)
        # 确认假设存在
        c.graph.get(hypothesis_id)
        eid = f"e_{uuid.uuid4().hex[:8]}"
        exp = Experiment(
            id=eid,
            hypothesis_id=hypothesis_id,
            objective=objective,
            depends_on=list(depends_on or []),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        c.experiments[eid] = exp
        c._queue.append(eid)
        return eid

    # ── 编排 ─────────────────────────────────────────────────────────

    async def run_campaign(
        self,
        campaign_id: str,
        runner: RunnerFn,
        mode: RunMode = "serial",
        max_parallel: int = 4,
        max_refines: int = 8,
    ) -> dict[str, Any]:
        """跑完 campaign 里所有实验 (含 refine 出来的新实验).

        mode:
        - serial: 一个一个跑, 失败的也继续下一个
        - parallel: asyncio.gather 并发 (忽略 depends_on)
        - dag: 按 depends_on 拓扑序, 同层并发

        每个实验跑完调 _record_result 更新假设图, 反驳时触发 refine_failed
        自动加新实验到队列. serial/dag 模式下队列动态增长, max_refines 限制
        refine 轮数避免无限循环 (超过的 refined 实验留 pending 不跑).
        """
        c = self.campaign(campaign_id)
        if mode == "parallel":
            await self._run_parallel(c, runner, max_parallel)
        elif mode == "dag":
            await self._run_dag(c, runner, max_parallel, max_refines)
        else:
            await self._run_serial(c, runner, max_refines)
        c.status = "completed"
        return self.aggregate(campaign_id)

    async def _run_serial(self, c: Campaign, runner: RunnerFn, max_refines: int) -> None:
        """串行跑, 队列动态增长 (refine 产新实验会入队). max_refines 限制 refine 轮数."""
        initial = len(c.experiments)
        ran = 0
        while c._queue:
            if ran >= initial + max_refines:
                break
            eid = c._queue.pop(0)
            exp = c.experiments[eid]
            await self._run_one(c, exp, runner)
            ran += 1

    async def _run_parallel(self, c: Campaign, runner: RunnerFn, max_parallel: int) -> None:
        """全并发跑当前队列里的实验. refine 出的新实验不再跑 (一轮制)."""
        batch = list(c._queue)
        c._queue.clear()
        sem = asyncio.Semaphore(max_parallel)

        async def guarded(exp: Experiment) -> None:
            async with sem:
                await self._run_one(c, exp, runner)

        await asyncio.gather(*(guarded(c.experiments[eid]) for eid in batch),
                             return_exceptions=True)

    async def _run_dag(self, c: Campaign, runner: RunnerFn, max_parallel: int, max_refines: int) -> None:
        """按 depends_on 拓扑序, 同层并发. refine 新实验按 serial 补跑, 受 max_refines 限制."""
        initial = len(c.experiments)
        ran = 0
        # 先按拓扑序跑已有的; 清空队列因为 topo 从 order 消费, 不从 _queue
        order = self._topo_sort(c)
        c._queue.clear()
        sem = asyncio.Semaphore(max_parallel)
        for layer in order:
            if not layer:
                continue
            tasks = []
            for eid in layer:
                tasks.append(self._guarded_run(c, c.experiments[eid], runner, sem))
            await asyncio.gather(*tasks, return_exceptions=True)
            ran += len(layer)
        # refine 出的新实验串行补跑, 总实验数不超过 initial + max_refines
        while c._queue and ran < initial + max_refines:
            eid = c._queue.pop(0)
            await self._run_one(c, c.experiments[eid], runner)
            ran += 1

    async def _guarded_run(
        self, c: Campaign, exp: Experiment, runner: RunnerFn, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            await self._run_one(c, exp, runner)

    async def _run_one(self, c: Campaign, exp: Experiment, runner: RunnerFn) -> None:
        """跑单个实验 + 记结果 + 触发假设图更新."""
        exp.status = "running"
        exp.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            result = await runner(exp)
            exp.result = result
            exp.status = "completed"
            self._record_result(c, exp, result)
        except Exception as exc:
            exp.status = "failed"
            exp.error = str(exc)
        exp.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _record_result(self, c: Campaign, exp: Experiment, result: dict[str, Any]) -> None:
        """根据实验结果更新假设图. 反驳时触发 refine_failed."""
        success = result.get("success", False)
        if not success:
            # 实验本身失败 (崩溃/超时), 不动假设图
            return

        hypothesis_supported = result.get("hypothesis_supported")
        evidence = result.get("evidence", {})

        if hypothesis_supported is True:
            c.graph.support(exp.hypothesis_id, evidence)
        elif hypothesis_supported is False:
            c.graph.refute(exp.hypothesis_id, evidence)
            # 触发修正: 生成新假设 + 自动加实验
            try:
                new_hyp_id = c.graph.refine_failed(
                    exp.hypothesis_id, evidence, model=self._model
                )
                new_eid = self.add_experiment(
                    c.id,
                    hypothesis_id=new_hyp_id,
                    objective=f"验证修正假设 (源自 {exp.id})",
                )
                # serial/dag 模式下 _queue 已有新实验, 会被消费
            except Exception as exc:
                # refine 失败不阻塞 campaign, 假设就停在 refuted.
                # 记到 experiment.error 方便后续排查
                exp.error = f"refine_failed 失败: {exc}"

    # ── 聚合 ─────────────────────────────────────────────────────────

    def aggregate(self, campaign_id: str) -> dict[str, Any]:
        """汇总 campaign 结果: 实验统计 + 假设图状态."""
        c = self.campaign(campaign_id)
        exps = list(c.experiments.values())
        return {
            "campaign_id": c.id,
            "name": c.name,
            "objective": c.objective,
            "status": c.status,
            "n_experiments": len(exps),
            "n_completed": sum(1 for e in exps if e.status == "completed"),
            "n_failed": sum(1 for e in exps if e.status == "failed"),
            "n_pending": sum(1 for e in exps if e.status == "pending"),
            "hypotheses": {
                "supported": len(c.graph.supported()),
                "refuted": len(c.graph.refuted()),
                "untested": len(c.graph.frontier()),
                "total": len(c.graph.all_nodes()),
            },
            "experiments": [e.to_dict() for e in exps],
        }

    # ── 内部 ─────────────────────────────────────────────────────────

    @staticmethod
    def _topo_sort(c: Campaign) -> list[list[str]]:
        """把实验按 depends_on 分层 (BFS), 同层可并发. 有环抛 CampaignError."""
        exps = c.experiments
        in_deg: dict[str, int] = {eid: 0 for eid in exps}
        for exp in exps.values():
            for dep in exp.depends_on:
                if dep in exps:
                    in_deg[exp.id] += 1

        layers: list[list[str]] = []
        current = [eid for eid, d in in_deg.items() if d == 0]
        done: set[str] = set()
        while current:
            layers.append(current)
            done.update(current)
            nxt: list[str] = []
            for eid in exps:
                if eid in done:
                    continue
                # 检查所有依赖是否已完成
                deps = exps[eid].depends_on
                if all(d in done for d in deps if d in exps):
                    nxt.append(eid)
            current = nxt

        # 有环的实验不会被排进任何 layer, 检测并报错
        if len(done) != len(exps):
            cyclic = [eid for eid in exps if eid not in done]
            raise CampaignError(
                f"实验依赖图存在环, 涉及: {cyclic}"
            )
        return layers
