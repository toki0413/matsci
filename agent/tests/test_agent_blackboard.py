"""P2 智能体黑板契约门 (红蓝双轨).

蓝队: 黑板条目带 kind/provenance/source, digest 可拼进下游 prompt;
       orchestrator/team/subagent 的依赖传递走结构化投影.
红队: 未给 source 的 measured 必须被拒 (非实测不得冒充实测);
       子 agent 产出只能标 reported; 黑板写入失败不得拖垮派发.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from huginn.agents.blackboard import Blackboard
from huginn.agents.orchestrator import Orchestrator, SubTask, TaskPlan
from huginn.agents.subagent import SubagentDispatch
from huginn.agents.team import ModelTeam, TeamMember, TeamRole, TeamStep
from huginn.models.registry import ModelCaps


# ── 纯黑板语义 ──────────────────────────────────────────────


class TestBlackboardSemantics:
    def test_red_measured_without_source_rejected(self):
        board = Blackboard("t")
        with pytest.raises(ValueError, match="source"):
            board.post("x", "1.23", provenance="measured")

    def test_blue_measured_with_source_is_kept(self):
        board = Blackboard("t")
        art = board.post(
            "x", "1.23", kind="evidence", provenance="measured",
            meta={"source": "vasp_tool"},
        )
        assert art.provenance == "measured"
        assert "source=vasp_tool" in board.digest()

    def test_blue_digest_labels_kind_and_provenance(self):
        board = Blackboard("t")
        board.post("x", "agent says 1.2", kind="finding", provenance="reported")
        text = board.digest()
        assert "[finding/reported] x" in text
        assert "agent says 1.2" in text

    def test_red_only_measured_filters_reported(self):
        board = Blackboard("t")
        board.post("guess", "0.9", provenance="predicted")
        board.post("relay", "0.8", provenance="reported")
        board.post(
            "obs", "0.7", provenance="measured", meta={"source": "run.log"}
        )
        text = board.digest(only_measured=True)
        assert "obs" in text
        assert "guess" not in text and "relay" not in text

    def test_red_unknown_kind_and_bad_provenance_rejected(self):
        board = Blackboard("t")
        with pytest.raises(ValueError, match="kind"):
            board.post("x", "v", kind="rumor")
        with pytest.raises(ValueError, match="provenance"):
            board.post("x", "v", provenance="guess")

    def test_blue_empty_digest_returns_empty_string(self):
        assert Blackboard("t").digest(keys=["missing"]) == ""


# ── Orchestrator 接线 ──────────────────────────────────────


class _FakeAgent:
    def __init__(self, output: str, prompts: list[str], *, raises: bool = False):
        self._output = output
        self._prompts = prompts
        self._raises = raises

    def invoke(self, prompt: str):
        self._prompts.append(prompt)
        if self._raises:
            raise RuntimeError("boom")
        return {"messages": [SimpleNamespace(content=self._output)]}


class _FakeFactory:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.outputs = {"lead": "alpha", "critic": "gamma"}
        self.fail_agents: set[str] = set()

    def create(self, agent_id: str, **_kwargs):
        return _FakeAgent(
            self.outputs.get(agent_id, "noop"),
            self.prompts,
            raises=agent_id in self.fail_agents,
        )


class TestOrchestratorBlackboard:
    def _orch(self) -> tuple[Orchestrator, _FakeFactory]:
        factory = _FakeFactory()
        orch = Orchestrator(factory)
        orch._make_sub_memory = lambda: None
        return orch, factory

    def test_blue_dependent_step_reads_digest_not_raw_string(self):
        orch, factory = self._orch()
        plan = TaskPlan(
            objective="obj",
            tasks=[
                SubTask(task_id="t1", agent_id="lead", prompt="first"),
                SubTask(task_id="t2", agent_id="critic", prompt="second",
                        depends_on=["t1"]),
            ],
        )
        result = asyncio.run(orch._execute_taskplan(plan))

        assert result.success
        assert result.blackboard is not None
        assert result.blackboard.read("t1").value == "alpha"
        # 第二条 prompt 拿到带来源标签的黑板投影, 而不是 "Output from t1:".
        assert "[finding/reported] t1" in factory.prompts[1]
        assert "Output from t1:" not in factory.prompts[1]
        # outputs 仍保留, 向后兼容.
        assert result.outputs["t1"] == "alpha"

    def test_red_failed_step_lands_as_error_entry(self):
        orch, factory = self._orch()
        factory.fail_agents.add("lead")
        plan = TaskPlan(
            objective="obj",
            tasks=[SubTask(task_id="t1", agent_id="lead", prompt="first")],
        )
        result = asyncio.run(orch._execute_taskplan(plan))

        assert not result.success
        art = result.blackboard.read("t1")
        assert art is not None and art.kind == "error"
        assert "boom" in art.value

    def test_blue_objective_is_posted_as_plan(self):
        orch, _ = self._orch()
        plan = TaskPlan(
            objective="solve it",
            tasks=[SubTask(task_id="t1", agent_id="lead", prompt="x")],
        )
        result = asyncio.run(orch._execute_taskplan(plan))
        art = result.blackboard.read("objective")
        assert art.kind == "plan" and art.value == "solve it"


# ── ModelTeam 接线 ─────────────────────────────────────────


class _RecordingChat:
    def __init__(self, seen: list[str], output: str):
        self._seen = seen
        self._output = output

    async def __call__(self, task: str, **_kwargs):
        self._seen.append(task)
        yield {"messages": [SimpleNamespace(content=self._output, tool_calls=None)]}


class _FakeTeamAgent:
    def __init__(self, seen: list[str], output: str):
        self.chat = _RecordingChat(seen, output)


def _team_member(role: TeamRole, seen: list[str], output: str) -> TeamMember:
    member = TeamMember(
        name=f"{role.value}-test", profile_id="test", role=role,
        model_name="fake", caps=ModelCaps(),
    )
    member._agent = _FakeTeamAgent(seen, output)
    return member


class TestTeamBlackboard:
    def test_blue_dependent_step_uses_blackboard_digest(self):
        seen: list[str] = []
        member = _team_member(TeamRole.SCIENTIST, seen, "alpha")
        team = ModelTeam([member])
        board = Blackboard("team:run")

        steps = [
            TeamStep(id="s1", role=TeamRole.SCIENTIST, task="first"),
            TeamStep(id="s2", role=TeamRole.SCIENTIST, task="second",
                     depends_on=["s1"]),
        ]
        outputs = asyncio.run(
            team._execute_plan(steps, {}, [], "run", board=board)
        )

        assert outputs == ["alpha", "alpha"]
        assert board.read("s1").value == "alpha"
        assert "[finding/reported] s1" in seen[1]
        assert "Context from previous steps" in seen[1]

    def test_red_step_failure_lands_as_error_kind(self):
        seen: list[str] = []
        member = _team_member(TeamRole.SCIENTIST, seen, "x")
        team = ModelTeam([member])
        board = Blackboard("team:run")
        # 让 _run_member 抛错 → _execute_plan 捕获并返回 [ERROR] 文本.
        member._agent = SimpleNamespace(chat=self._raising_chat)

        steps = [TeamStep(id="s1", role=TeamRole.SCIENTIST, task="first")]
        outputs = asyncio.run(
            team._execute_plan(steps, {}, [], "run", board=board)
        )

        assert outputs[0].startswith("[ERROR]")
        assert board.read("s1").kind == "error"

    @staticmethod
    def _raising_chat(*_args, **_kwargs):
        async def _gen():
            raise RuntimeError("team boom")
            yield  # pragma: no cover

        return _gen()


# ── Subagent 接线 ──────────────────────────────────────────


class _FakeSubAgent:
    def __init__(self, seen: list[str]):
        self._seen = seen
        self.tool_filter: set[str] = set()
        self.langchain_tools: list = []
        self._max_tool_calls = 0

    def register_tools_from_registry(self) -> None:
        return None

    async def chat(self, task: str, _thread_id: str | None = None):
        self._seen.append(task)
        yield {"messages": [SimpleNamespace(content="sub-result", tool_calls=None)]}


class _FakeSubFactory:
    def __init__(self, seen: list[str]):
        self._seen = seen

    def get_profile(self, profile_id: str):
        return object()

    def list_profiles(self):
        return []

    def create(self, **_kwargs):
        return _FakeSubAgent(self._seen)


class TestSubagentBlackboard:
    def test_blue_upstream_digest_injected_into_isolated_task(self):
        seen: list[str] = []
        board = Blackboard("task")
        board.post("upstream", "measured alpha", provenance="reported")
        dispatch = SubagentDispatch()

        result = asyncio.run(
            dispatch.dispatch(
                "explore", "do work",
                context={
                    "agent_factory": _FakeSubFactory(seen),
                    "blackboard": board,
                    "blackboard_keys": ["upstream"],
                    "blackboard_key": "downstream",
                },
            )
        )

        assert result.success
        assert "## Shared blackboard" in seen[0]
        assert "[finding/reported] upstream" in seen[0]
        # 产出回写黑板, 且只能标 reported.
        art = board.read("downstream")
        assert art is not None
        assert art.provenance == "reported"
        assert art.value == "sub-result"

    def test_red_board_write_failure_does_not_break_dispatch(self):
        class _ExplodingBoard:
            def digest(self, **_kwargs):
                raise RuntimeError("digest boom")

            def post(self, *_args, **_kwargs):
                raise RuntimeError("post boom")

        seen: list[str] = []
        result = asyncio.run(
            SubagentDispatch().dispatch(
                "explore", "do work",
                context={
                    "agent_factory": _FakeSubFactory(seen),
                    "blackboard": _ExplodingBoard(),
                    "blackboard_keys": ["upstream"],
                },
            )
        )

        assert result.success and result.summary == "sub-result"

    def test_red_measured_never_auto_claimed_by_helper(self):
        board = Blackboard("task")
        SubagentDispatch._share_to_blackboard(
            board, {}, "explore", "tid", "text"
        )
        art = board.read("subagent:explore:tid")
        assert art.provenance == "reported"
