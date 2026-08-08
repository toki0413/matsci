"""Multi-agent swarm orchestration.

HuginnSwarm coordinates specialized workers:

- Planner: breaks a user task into a JSON plan
- Scientist: chooses physical models and equations
- Coder: writes code / tool calls
- Critic: reviews outputs for correctness
- Executor: runs external tools (often the main HuginnAgent)

The supervisor executes plan steps respecting dependencies; independent
steps run in parallel.
"""

from __future__ import annotations

import abc
import asyncio
import concurrent.futures
import enum
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class AgentRole(enum.StrEnum):
    PLANNER = "planner"
    SCIENTIST = "scientist"
    CODER = "coder"
    CRITIC = "critic"
    EXECUTOR = "executor"


@dataclass
class SwarmAgent:
    """A worker agent in the swarm."""

    name: str
    role: AgentRole
    agent: Any
    instructions: str = ""


@dataclass
class SwarmPlanStep:
    """One step in a swarm execution plan."""

    id: str
    role: AgentRole
    task: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SwarmStep:
    """One step in a swarm execution trace."""

    role: AgentRole
    agent_name: str
    input_task: str
    output: str
    duration_ms: float = 0.0


class HuginnSwarm:
    """Supervisor-based multi-agent orchestrator."""

    # Default plan prompt if the user-supplied planner is unavailable.
    _PLANNER_PROMPT = (
        "You are a task planner. Break the user task into steps. "
        "Respond with a JSON array only. Each item must have:\n"
        '{"id": "step1", "role": "scientist", "task": "...", "depends_on": []}\n'
        "Available roles: scientist, coder, executor, critic. "
        "Use depends_on to declare steps that must finish before this one."
    )

    def __init__(
        self,
        workers: list[SwarmAgent],
        backend: "DistributedSwarmBackend | None" = None,
    ) -> None:
        self.workers = {w.role: w for w in workers}
        self.trace: list[SwarmStep] = []
        # backend=None 时默认 InProcessBackend, 等价于现状 (async 直接跑).
        # 传 RedisBackend / PostgresBackend 才会走跨进程队列.
        self.backend: DistributedSwarmBackend = backend or InProcessBackend(self)

    def add_worker(self, worker: SwarmAgent) -> HuginnSwarm:
        self.workers[worker.role] = worker
        return self

    async def run(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the task through the swarm and return the final result."""
        self.trace.clear()
        ctx = dict(context or {})
        ctx["original_task"] = task

        # 1. Planning
        plan_text = await self._delegate(AgentRole.PLANNER, task, ctx)
        ctx["planner_output"] = plan_text
        steps = self._parse_plan(plan_text)
        if not steps:
            steps = self._default_plan(task)
        ctx["plan"] = self._plan_to_text(steps)

        # 2. Execute planned steps respecting dependencies.
        step_outputs = await self._execute_plan(steps, ctx)

        # Map outputs to legacy context keys for convenience.
        role_outputs: dict[AgentRole, str] = {}
        for step, output in zip(steps, step_outputs):
            role_outputs[step.role] = output
        ctx["scientific_reasoning"] = role_outputs.get(AgentRole.SCIENTIST, "")
        ctx["code"] = role_outputs.get(AgentRole.CODER, "")
        ctx["execution_result"] = role_outputs.get(
            AgentRole.EXECUTOR, "No executor step completed."
        )

        # 3. Critic review (only if the plan did not already include one).
        critic_output = role_outputs.get(AgentRole.CRITIC, "")
        if not critic_output and AgentRole.CRITIC in self.workers:
            critic_input = (
                f"Task: {task}\n"
                f"Plan: {ctx['plan']}\n"
                f"Execution result: {ctx['execution_result']}"
            )
            critic_output = await self._delegate(AgentRole.CRITIC, critic_input, ctx)
        ctx["review"] = critic_output

        return {
            "task": task,
            "context": ctx,
            "trace": [self._step_to_dict(s) for s in self.trace],
            "final_output": ctx["execution_result"],
        }

    def _parse_plan(self, text: str) -> list[SwarmPlanStep]:
        """Parse planner output into steps."""
        if not text:
            return []
        # Strip markdown fences.
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip("json").strip()
        try:
            data = json.loads(text)
        except Exception:
            return []
        if not isinstance(data, list):
            return []

        steps: list[SwarmPlanStep] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(
                    SwarmPlanStep(
                        id=str(item.get("id", f"step{len(steps) + 1}")),
                        role=AgentRole(item.get("role", "executor")),
                        task=str(item.get("task", "")),
                        depends_on=[str(d) for d in item.get("depends_on", []) if d],
                    )
                )
            except Exception:
                continue
        return steps

    def _default_plan(self, task: str) -> list[SwarmPlanStep]:
        """Fallback plan when planner output is unusable."""
        steps: list[SwarmPlanStep] = []
        order = [
            AgentRole.SCIENTIST,
            AgentRole.CODER,
            AgentRole.EXECUTOR,
            AgentRole.CRITIC,
        ]
        prev_id: str | None = None
        for role in order:
            if role not in self.workers:
                continue
            step_id = f"{role.value}_step"
            depends = [prev_id] if prev_id else []
            steps.append(
                SwarmPlanStep(
                    id=step_id,
                    role=role,
                    task=f"{role.value.replace('_', ' ').title()} for: {task}",
                    depends_on=depends,
                )
            )
            prev_id = step_id
        return steps

    @staticmethod
    def _plan_to_text(steps: list[SwarmPlanStep]) -> str:
        lines = []
        for s in steps:
            deps = f" (after {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"{s.id}: [{s.role.value}] {s.task}{deps}")
        return "\n".join(lines)

    async def _execute_plan(
        self,
        steps: list[SwarmPlanStep],
        ctx: dict[str, Any],
    ) -> list[str]:
        """Execute steps respecting dependencies; independent steps run in parallel."""
        results: dict[str, str] = {}
        pending = {s.id: s for s in steps}

        while pending:
            ready = [
                s
                for s in pending.values()
                if all(dep in results for dep in s.depends_on)
            ]
            if not ready:
                # Cyclic dependency fallback: run remaining sequentially.
                ready = list(pending.values())

            async def run_one(step: SwarmPlanStep) -> tuple[str, str]:
                worker = self.workers.get(step.role)
                if not worker:
                    return step.id, ""
                # Build input using outputs from dependencies.
                try:
                    dep_text = "\n".join(
                        f"{dep}: {results[dep]}"
                        for dep in step.depends_on
                        if dep in results
                    )
                    task = step.task
                    if dep_text:
                        task = f"{task}\n\nContext from previous steps:\n{dep_text}"
                    output = await self._run_agent(worker, task, ctx)
                    return step.id, output
                except Exception as exc:
                    # 失败隔离: 单个 sub-agent 异常不应团灭整个并行 batch.
                    # 与 parallel_executor._run_one 的 try/except 对齐. 之前
                    # asyncio.gather 会 propagate 异常杀死整批, 导致一个
                    # RateLimitError 让 plan 后续步骤全死.
                    logger.warning(
                        "swarm run_one step=%s failed (isolated): %s",
                        step.id, exc, exc_info=True,
                    )
                    return step.id, f"[ERROR] step {step.id} failed: {exc}"

            batch_results = await asyncio.gather(*(run_one(s) for s in ready))
            for step_id, output in batch_results:
                results[step_id] = output
                pending.pop(step_id)

        return [results[s.id] for s in steps]

    async def _delegate(self, role: AgentRole, task: str, ctx: dict[str, Any]) -> str:
        worker = self.workers.get(role)
        if not worker:
            return ""
        return await self._run_agent(worker, task, ctx)

    async def _run_agent(
        self,
        worker: SwarmAgent,
        task: str,
        ctx: dict[str, Any],
    ) -> str:
        start = time.time()
        full_prompt = (
            f"{worker.instructions}\n\n{task}" if worker.instructions else task
        )
        final_output = ""
        async for state in worker.agent.chat(
            full_prompt, thread_id=ctx.get("thread_id", "swarm")
        ):
            messages = state.get("messages", [])
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    final_output = str(content)
        duration_ms = round((time.time() - start) * 1000, 2)
        step = SwarmStep(
            role=worker.role,
            agent_name=worker.name,
            input_task=task,
            output=final_output,
            duration_ms=duration_ms,
        )
        self.trace.append(step)
        return final_output

    @staticmethod
    def _step_to_dict(step: SwarmStep) -> dict[str, Any]:
        return {
            "role": step.role.value,
            "agent_name": step.agent_name,
            "input_task": step.input_task,
            "output": step.output,
            "duration_ms": step.duration_ms,
        }


# ─────────────────────────────────────────────────────────────────────────
# 跨进程扩展 backend (Task 26.10 方向 4)
#
# 默认 InProcessBackend 就是现状的 async 跑法, 单进程内并发.
# 想跨进程 / 跨机器时, HUGINN_SWARM_DISTRIBUTED=1 切到 RedisBackend,
# HUGINN_SWARM_DISTRIBUTED=postgres 切到 PostgresBackend. 队列只承担
# 任务分发, 依赖解析仍走 TaskDAG (task_dag.py) + provenance, 跟
# parallel_executor._depends_on 的单进程启发式是两套不重叠的兜底.
# ─────────────────────────────────────────────────────────────────────────


class DistributedSwarmBackend(abc.ABC):
    """swarm 任务分发后端抽象.

    调用方 submit_task 拿到 task_id, 之后 get_result 阻塞等结果.
    具体实现可以是同进程的线程池 (InProcessBackend)、Redis 队列
    (RedisBackend)、Postgres LISTEN/NOTIFY (PostgresBackend).
    """

    @abc.abstractmethod
    def submit_task(self, task: dict[str, Any]) -> str:
        """提交任务, 返回 task_id."""

    @abc.abstractmethod
    def get_result(self, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
        """阻塞等结果, 超时返回 {"error": ...}."""

    @abc.abstractmethod
    def close(self) -> None:
        """释放连接 / 线程池."""


class InProcessBackend(DistributedSwarmBackend):
    """默认 backend — 在当前进程的线程池里 async 跑 swarm.run.

    单进程场景下跟原来一样, 只是 submit_task/get_result 提供了同步接口
    方便外部调度器统一调用. 跨进程时换 RedisBackend / PostgresBackend.
    """

    def __init__(
        self,
        swarm: HuginnSwarm,
        max_workers: int = 4,
    ) -> None:
        self._swarm = swarm
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # ponytail: 完成的 future 不主动清理, 长 run 会缓慢长大.
        # 升级: 加 LRU / TTL 回收, 或者切到 asyncio Future + done callback.
        self._pending: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}

    def submit_task(self, task: dict[str, Any]) -> str:
        task_id = uuid.uuid4().hex
        fut = self._pool.submit(self._run_async, task)
        self._pending[task_id] = fut
        return task_id

    def get_result(self, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
        fut = self._pending.get(task_id)
        if fut is None:
            return {"error": f"unknown task_id: {task_id}"}
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {"error": f"timeout after {timeout}s", "task_id": task_id}
        except Exception as exc:  # 线程里抛出来的, 包成 dict 不外泄
            return {"error": str(exc), "task_id": task_id}

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def _run_async(self, task: dict[str, Any]) -> dict[str, Any]:
        """在线程里跑 swarm.run, asyncio.run 自带 loop."""
        return asyncio.run(
            self._swarm.run(task.get("task", ""), task.get("context"))
        )


class RedisBackend(DistributedSwarmBackend):
    """用 Redis list 当任务队列, 结果写 result:<task_id> key.

    生产者: LPUSH swarm:queue <json>
    消费者: BRPOP swarm:queue 拉任务, 跑完 SETEX result:<id> <ttl> <json>
    取结果: 轮询 GET result:<id>, 超时放弃.

    没装 redis-py 时 __init__ 直接 raise ImportError, 不会拖垮整个模块.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        queue_key: str = "huginn:swarm:queue",
        result_prefix: str = "huginn:swarm:result:",
        result_ttl: int = 3600,
    ) -> None:
        try:
            import redis  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - 跟环境绑死
            raise ImportError(
                "RedisBackend 需要 redis-py: pip install redis"
            ) from exc
        self._redis = redis
        self._redis_url = redis_url
        self._queue_key = queue_key
        self._result_prefix = result_prefix
        self._result_ttl = result_ttl
        # 延迟到 submit_task 才真连, 避免实例化就拖死测试.
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._redis.from_url(self._redis_url)
        return self._client

    def submit_task(self, task: dict[str, Any]) -> str:
        client = self._ensure_client()
        task_id = task.get("task_id") or uuid.uuid4().hex
        payload = json.dumps({"task_id": task_id, **task})
        client.lpush(self._queue_key, payload)
        return task_id

    def get_result(self, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
        client = self._ensure_client()
        key = f"{self._result_prefix}{task_id}"
        deadline = time.time() + timeout
        # 轮询, 间隔 100ms. BRPOP 不能直接等 result key (那是 SETEX 写的),
        # 只能轮询. ponytail: 短任务够用; 升级: 用 Pub/Sub 通知避免空轮询.
        while time.time() < deadline:
            raw = client.get(key)
            if raw is not None:
                try:
                    return json.loads(raw)
                except Exception:
                    return {"error": "result decode failed", "raw": str(raw)}
            time.sleep(0.1)
        return {"error": f"timeout after {timeout}s", "task_id": task_id}

    def write_result(self, task_id: str, result: dict[str, Any]) -> None:
        """worker 侧调用: 把结果写回 Redis."""
        client = self._ensure_client()
        key = f"{self._result_prefix}{task_id}"
        client.setex(key, self._result_ttl, json.dumps(result))

    def pop_task(self, timeout: int = 0) -> tuple[str, dict[str, Any]] | None:
        """worker 侧调用: BRPOP 队列拉一个任务.

        timeout=0 表示无限阻塞. 返回 (task_id, task_dict), 队列空且
        非阻塞时返回 None.
        """
        client = self._ensure_client()
        item = client.brpop(self._queue_key, timeout=timeout)
        if item is None:
            return None
        _key, payload = item
        data = json.loads(payload)
        task_id = data.pop("task_id")
        return task_id, data

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


class PostgresBackend(DistributedSwarmBackend):
    """用 Postgres LISTEN/NOTIFY 当任务队列 (无 Redis 依赖时的备选).

    生产者: INSERT 一行到 swarm_tasks, NOTIFY channel 带 task_id
    消费者: LISTEN channel, 收到 NOTIFY 后 SELECT payload, 跑完 UPDATE result
    取结果: 轮询 SELECT result FROM swarm_tasks WHERE task_id=...

    需要预先建表:
        CREATE TABLE swarm_tasks (
            task_id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            result JSONB,
            created_at TIMESTAMPTZ DEFAULT now(),
            finished_at TIMESTAMPTZ
        );
    """

    def __init__(
        self,
        dsn: str = "dbname=huginn user=huginn",
        channel: str = "huginn_swarm",
        poll_interval: float = 0.1,
    ) -> None:
        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - 跟环境绑死
            raise ImportError(
                "PostgresBackend 需要 psycopg2: pip install psycopg2-binary"
            ) from exc
        self._psycopg2 = psycopg2
        self._dsn = dsn
        self._channel = channel
        self._poll_interval = poll_interval
        self._conn: Any = None

    def _ensure_conn(self) -> Any:
        if self._conn is None:
            self._conn = self._psycopg2.connect(self._dsn)
        return self._conn

    def submit_task(self, task: dict[str, Any]) -> str:
        conn = self._ensure_conn()
        task_id = task.get("task_id") or uuid.uuid4().hex
        payload = json.dumps({**task})
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO swarm_tasks (task_id, payload) VALUES (%s, %s)",
                (task_id, payload),
            )
            cur.execute(f"NOTIFY {self._channel}, %s", (task_id,))
        conn.commit()
        return task_id

    def get_result(self, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
        conn = self._ensure_conn()
        deadline = time.time() + timeout
        # ponytail: 轮询 SELECT, 短任务够用; 升级: 单独 LISTEN 连接等结果通知.
        while time.time() < deadline:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT result FROM swarm_tasks WHERE task_id = %s",
                    (task_id,),
                )
                row = cur.fetchone()
                if row is not None and row[0] is not None:
                    try:
                        return json.loads(row[0])
                    except Exception:
                        return {"error": "result decode failed", "raw": str(row[0])}
            time.sleep(self._poll_interval)
        return {"error": f"timeout after {timeout}s", "task_id": task_id}

    def write_result(self, task_id: str, result: dict[str, Any]) -> None:
        """worker 侧调用: 把结果写回 swarm_tasks."""
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE swarm_tasks SET result = %s, finished_at = now() "
                "WHERE task_id = %s",
                (json.dumps(result), task_id),
            )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def make_swarm_backend(swarm: HuginnSwarm | None = None) -> DistributedSwarmBackend:
    """工厂: 从 HUGINN_SWARM_DISTRIBUTED env var 选 backend.

    - 未设置 / 空 / "0"      -> InProcessBackend(swarm)
    - "1" / "redis"          -> RedisBackend()
    - "postgres" / "pg"      -> PostgresBackend()

    Redis/Postgres 模式下 swarm 可以为 None (只当队列分发, 执行端在别的进程).
    """
    mode = (os.environ.get("HUGINN_SWARM_DISTRIBUTED") or "").strip().lower()
    if mode in ("1", "redis"):
        return RedisBackend()
    if mode in ("postgres", "pg"):
        return PostgresBackend()
    if swarm is None:
        raise ValueError("InProcessBackend 需要传入 swarm 实例")
    return InProcessBackend(swarm)


# ── selfcheck ──────────────────────────────────────────────


class _FakeChatAgent:
    """self-check 用的 stub agent, 模拟 langgraph agent 的 async chat."""

    def __init__(self, marker: str) -> None:
        self._marker = marker

    async def chat(self, prompt: str, thread_id: str = "swarm"):
        class _Msg:
            content = f"[{self._marker}] {prompt[:40]!r}"

        yield {"messages": [_Msg()]}


def _build_demo_swarm() -> HuginnSwarm:
    """造一个 4-worker swarm: scientist / coder / executor / critic.

    没 planner, swarm.run 会走 _default_plan 的串行链.
    """
    roles = (
        AgentRole.SCIENTIST,
        AgentRole.CODER,
        AgentRole.EXECUTOR,
        AgentRole.CRITIC,
    )
    workers = [
        SwarmAgent(name=role.value, role=role, agent=_FakeChatAgent(role.value))
        for role in roles
    ]
    return HuginnSwarm(workers)


if __name__ == "__main__":
    # ── 1. InProcessBackend demo: 4 个独立任务并发 ──────────────────
    # backend 内部 ThreadPoolExecutor(max_workers=4), 4 个 swarm.run 同时跑.
    # 注: swarm.trace 是 instance 级, 多线程并发 run 会互相 clear/append.
    #     跨进程场景下每个 worker 进程有自己的 swarm 实例, trace 不共享, 所以
    #     这个 race 只在 InProcess 多 worker 时出现. demo 只验 final_output
    #     (来自 ctx, 是 local 的, 不会 race).
    swarm = _build_demo_swarm()
    backend = swarm.backend  # 默认 InProcessBackend(swarm, max_workers=4)

    tasks = [
        {
            "task": f"独立任务 #{i}: 算 material {i}",
            "context": {"thread_id": f"t{i}"},
        }
        for i in range(4)
    ]
    task_ids = [backend.submit_task(t) for t in tasks]
    results = [backend.get_result(tid, timeout=30.0) for tid in task_ids]

    assert len(results) == 4, f"应返回 4 个结果, got {len(results)}"
    for tid, r in zip(task_ids, results):
        assert "error" not in r, f"task {tid} 失败: {r}"
        final = r.get("final_output", "")
        assert final, f"task {tid} final_output 空"
        print(f"[ok] InProcess task {tid[:8]} final={final[:60]!r}")
    backend.close()
    print("[ok] InProcessBackend: 4 个独立任务并发完成")

    # ── 2. RedisBackend import 可用性 ────────────────────────────────
    # 真连 Redis 需要 redis-server 跑着, selfcheck 只验 import + 实例化.
    try:
        rb = RedisBackend()
        print("[ok] RedisBackend: redis-py 可用 (实例化成功, 不真连 Redis)")
        rb.close()
    except ImportError as exc:
        print(f"[skip] RedisBackend: {exc}")

    # ── 3. PostgresBackend import 可用性 ─────────────────────────────
    try:
        pb = PostgresBackend()
        print("[ok] PostgresBackend: psycopg2 可用 (实例化成功, 不真连 PG)")
        pb.close()
    except ImportError as exc:
        print(f"[skip] PostgresBackend: {exc}")

    # ── 4. 工厂函数默认走 InProcess ──────────────────────────────────
    saved = os.environ.pop("HUGINN_SWARM_DISTRIBUTED", None)
    try:
        b = make_swarm_backend(swarm)
        assert isinstance(b, InProcessBackend), (
            f"默认应 InProcess, got {type(b).__name__}"
        )
        print(f"[ok] make_swarm_backend() 默认 -> {type(b).__name__}")
    finally:
        if saved is not None:
            os.environ["HUGINN_SWARM_DISTRIBUTED"] = saved

    print("[swarm] self-check OK")
