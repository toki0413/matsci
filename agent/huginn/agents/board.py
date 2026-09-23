"""任务看板 / 专家团画布 —— 从会话原始事件重建的只读投影.

这是**读 side**: 不改事件、不写文件. 输入是 ``SessionEventLog`` 里的原始事件
(``tool_call`` / ``tool_result``), 输出是前端画布可直接渲染的图:

  nodes    每个子任务 (subagent dispatch) 一个节点, 带专家/状态/层号
  edges    子任务依赖 (来自 dispatch_parallel 的 ``dependencies``)
  layers   拓扑分层 (``TaskDAG.parallel_layers``) — 同层节点可并行
  experts  专家泳道: 每个 spec 一行, 含已派发/完成/失败/在跑计数
  stats    总览计数

数据全部来自**已有事件**, 不新增写路径:
  - ``tool_call``   ``name=subagent_tool``, ``args.action ∈ {dispatch, dispatch_parallel}``
  - ``tool_result`` 按 ``tool_call_id`` 关联, 解析 JSON 拿 ``success`` / ``summary``

状态机: 有 tool_call 无 tool_result → ``running``; 有 result 且 success → ``done``,
否则 ``failed``. 这样前端轮询就能看到"谁在跑、谁跑完、谁挂了".

``build_board`` 是 fail-open 的: 日志读不出来 / DAG 有环 / JSON 解析失败都不抛,
最坏返回一个空看板 —— 看板挂了不能把主循环带崩.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from huginn.events.session_log import (
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    SessionEventLog,
)

logger = logging.getLogger(__name__)

# 子任务派发工具的注册名 (见 tools/subagent_tool.py)。
_DISPATCH_TOOL = "subagent_tool"
# 会产生子任务节点的 action; list_types 不产节点。
_DISPATCH_ACTIONS = frozenset({"dispatch", "dispatch_parallel"})

# 任务 id 的截断长度 — 跟 subagent_tool._dispatch_parallel 里
# ``f"{spec_name}:{task[:20]}"`` 的约定一致, 两边必须对齐才能匹配依赖。
_TASK_ID_LEN = 20
# 节点摘要预览上限, 防止超长 summary 撑爆响应。
_SUMMARY_LEN = 400


@dataclass
class BoardNode:
    """画布上的一个子任务节点."""

    id: str
    expert: str
    task: str
    status: str                 # pending | running | done | failed
    layer: int = 0
    deps: list[str] = field(default_factory=list)
    success: bool | None = None
    summary: str = ""
    seq: int = 0                # 产生它的 tool_call 事件 seq (前端增量游标)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expert": self.expert,
            "task": self.task,
            "status": self.status,
            "layer": self.layer,
            "deps": list(self.deps),
            "success": self.success,
            "summary": self.summary,
            "seq": self.seq,
        }


def _parse_content(content: Any) -> dict[str, Any] | None:
    """tool_result 的 content 可能是 JSON 字符串, 尝试解析成 dict."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text or text[0] not in "{[":
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # 结果不是 JSON (子 agent 摘要被压成散文) → 交给调用方走兜底
        return None
    return parsed if isinstance(parsed, dict) else None


def _node_id(expert: str, task: str, used: set[str]) -> str:
    """生成稳定且唯一的节点 id.

    基准 id 与 ``subagent_tool`` 的 DAG 约定一致 (``expert:task[:20]``), 这样
    LLM 填的 ``dependencies`` 能直接对上. 同一会话里重复派发同款任务时加
    ``#n`` 后缀避免撞 id.
    """
    base = f"{expert}:{task[:_TASK_ID_LEN]}"
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}#{n}" in used:
        n += 1
    candidate = f"{base}#{n}"
    used.add(candidate)
    return candidate


def _collect_calls(
    log: SessionEventLog,
) -> list[tuple[dict[str, Any], str, int]]:
    """扫出所有子任务派发的 tool_call.

    返回 ``[(call_payload, tool_call_id, seq), ...]``, 按 seq 升序.
    """
    calls: list[tuple[dict[str, Any], str, int]] = []
    for ev in log:
        if ev.kind != EVENT_TOOL_CALL:
            continue
        payload = ev.payload or {}
        if payload.get("name") != _DISPATCH_TOOL:
            continue
        args = payload.get("args")
        if not isinstance(args, dict):
            continue
        if args.get("action") not in _DISPATCH_ACTIONS:
            continue
        calls.append((payload, str(payload.get("tool_call_id") or ""), ev.seq))
    return calls


def _collect_results(log: SessionEventLog) -> dict[str, Any]:
    """tool_call_id -> tool_result 的 content."""
    results: dict[str, Any] = {}
    for ev in log:
        if ev.kind != EVENT_TOOL_RESULT:
            continue
        payload = ev.payload or {}
        tid = payload.get("tool_call_id")
        if tid:
            results[str(tid)] = payload.get("content")
    return results


def _specs_from_call(
    args: dict[str, Any], seq: int, used: set[str]
) -> list[BoardNode]:
    """把一条派发 tool_call 的 args 展开成节点 (不含状态, 待 result 回填)."""
    nodes: list[BoardNode] = []
    action = args.get("action")

    if action == "dispatch":
        expert = str(args.get("spec_name") or "unknown")
        task = str(args.get("task") or "")
        nodes.append(
            BoardNode(
                id=_node_id(expert, task, used),
                expert=expert,
                task=task,
                status="running",
                seq=seq,
            )
        )
        return nodes

    # dispatch_parallel: tasks 是 [{spec_name, task}, ...]
    raw_tasks = args.get("tasks")
    if not isinstance(raw_tasks, list):
        return nodes
    for t in raw_tasks:
        if not isinstance(t, dict):
            continue
        expert = str(t.get("spec_name") or "unknown")
        task = str(t.get("task") or "")
        nodes.append(
            BoardNode(
                id=_node_id(expert, task, used),
                expert=expert,
                task=task,
                status="running",
                seq=seq,
            )
        )
    # 依赖: [[u, v], ...] — JSON 里 tuple 会变 list, 两种都认
    deps_raw = args.get("dependencies")
    if isinstance(deps_raw, list):
        by_index: dict[str, list[str]] = {}
        for pair in deps_raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                u, v = str(pair[0]), str(pair[1])
                by_index.setdefault(v, []).append(u)
        for node in nodes:
            node.deps = list(by_index.get(node.id, []))
    return nodes


def _apply_result(node: BoardNode, parsed: dict[str, Any] | None, raw: Any) -> None:
    """把 tool_result 的解析结果回填到节点状态上."""
    if parsed is None:
        # 非 JSON 结果 → 工具跑过了就算完成, 保留原文前 400 字做摘要
        node.status = "done"
        node.success = True
        node.summary = str(raw)[:_SUMMARY_LEN]
        return
    success = parsed.get("success")
    node.success = bool(success) if success is not None else None
    node.status = "done" if node.success is not False else "failed"
    summary = parsed.get("summary") or parsed.get("error") or ""
    node.summary = str(summary)[:_SUMMARY_LEN]


def _match_parallel_results(
    nodes: list[BoardNode], parsed: dict[str, Any] | None, raw: Any
) -> None:
    """dispatch_parallel 的单条 result 对应多个节点: 按 ``results`` 顺序回填."""
    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list) or len(results) != len(nodes):
        # 数量对不上 (解析失败/结构变了) → 整体按 success 兜底, 不猜
        for node in nodes:
            _apply_result(node, parsed, raw)
        return
    for node, item in zip(nodes, results):
        if isinstance(item, dict):
            _apply_result(node, item, item)
        else:
            _apply_result(node, None, item)


def _assign_layers(
    nodes: list[BoardNode],
) -> tuple[list[list[str]], list[str], int]:
    """用 TaskDAG 算分层 / 关键路径 / 并行度.

    三项各自降级: 依赖非法或成环 → 全铺一层; networkx 缺失时关键路径取节点
    顺序、并行度退化为最大层宽. 看板不能因为一个算法不可用就整块空掉.
    返回 ``(layers, critical_path, width)``.
    """
    ids = [n.id for n in nodes]
    id_set = set(ids)
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for n in nodes:
        for u in n.deps:
            if u in id_set and u != n.id and (u, n.id) not in seen:
                seen.add((u, n.id))
                edges.append((u, n.id))

    try:
        from huginn.agents.task_dag import TaskDAG

        dag = TaskDAG(tasks=ids, dependencies=edges)
    except Exception:  # noqa: BLE001 — 成环/依赖非法 → 全铺一层, 不抛给前端
        logger.debug("board: DAG 构建失败, 退化为单层", exc_info=True)
        return [ids], list(ids), len(ids)

    try:
        layers = dag.parallel_layers()
    except Exception:  # noqa: BLE001 — 分层算法异常 → 全铺一层兜底
        logger.debug("board: parallel_layers 失败, 退化为单层", exc_info=True)
        layers = [ids]

    try:
        critical_path = dag.critical_path()
    except Exception:  # noqa: BLE001 — networkx 缺失等 → 关键路径取节点顺序
        logger.debug("board: critical_path 不可用, 退化为节点顺序", exc_info=True)
        critical_path = list(ids)

    try:
        width = dag.antichain_width()
    except Exception:  # noqa: BLE001 — networkx 缺失等 → 并行度退化为最大层宽
        logger.debug("board: antichain_width 不可用, 退化为最大层宽", exc_info=True)
        width = max((len(layer) for layer in layers), default=len(ids))

    return layers, critical_path, width


def _build_experts(
    nodes: list[BoardNode], roster: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """专家泳道: 已派发的 spec + roster 里还没动过的 spec."""
    lanes: dict[str, dict[str, Any]] = {}
    for entry in roster or []:
        name = str(entry.get("name") or "")
        if not name:
            continue
        lanes[name] = {
            "name": name,
            "description": str(entry.get("description") or ""),
            "total": 0,
            "done": 0,
            "failed": 0,
            "running": 0,
            "node_ids": [],
        }
    for node in nodes:
        lane = lanes.get(node.expert)
        if lane is None:
            lane = {
                "name": node.expert,
                "description": "",
                "total": 0,
                "done": 0,
                "failed": 0,
                "running": 0,
                "node_ids": [],
            }
            lanes[node.expert] = lane
        lane["total"] += 1
        lane["node_ids"].append(node.id)
        if node.status == "done":
            lane["done"] += 1
        elif node.status == "failed":
            lane["failed"] += 1
        else:
            lane["running"] += 1
    return [lanes[k] for k in sorted(lanes)]


def build_board(
    thread_id: str,
    *,
    log: SessionEventLog | None = None,
    expert_roster: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """从会话原始事件重建任务看板 / 专家团画布. 只读, fail-open.

    Args:
        thread_id: 会话 id.
        log: 可注入的事件日志 (测试用); 缺省按 thread_id 打开.
        expert_roster: 可选, 可用专家清单 (``SubagentDispatch.list_specs()`` 形状),
            用于把"还没派发"的专家也画进泳道.

    Returns:
        ``{thread_id, exists, nodes, edges, layers, critical_path, width,
        experts, stats, next_seq}``.
    """
    try:
        if log is None:
            log = SessionEventLog.open(thread_id)
    except Exception:  # noqa: BLE001 — 日志不可用 → 空看板, 不抛
        logger.debug("board: 打开会话日志 %s 失败", thread_id, exc_info=True)
        return _empty_board(thread_id)

    try:
        used: set[str] = set()
        nodes: list[BoardNode] = []
        results = _collect_results(log)

        for payload, tid, seq in _collect_calls(log):
            args = payload.get("args") or {}
            batch = _specs_from_call(args, seq, used)
            if not batch:
                continue
            if args.get("action") == "dispatch_parallel":
                raw = results.get(tid)
                if tid in results:
                    _match_parallel_results(batch, _parse_content(raw), raw)
                nodes.extend(batch)
            else:
                if tid in results:
                    _apply_result(batch[0], _parse_content(results[tid]), results[tid])
                nodes.extend(batch)

        layers, critical_path, width = _assign_layers(nodes)
        layer_of: dict[str, int] = {}
        for i, layer in enumerate(layers):
            for nid in layer:
                layer_of[nid] = i
        for node in nodes:
            node.layer = layer_of.get(node.id, 0)

        node_ids = {n.id for n in nodes}
        edges = [
            {"from": u, "to": n.id}
            for n in nodes
            for u in n.deps
            if u in node_ids
        ]
        stats = {
            "total": len(nodes),
            "done": sum(1 for n in nodes if n.status == "done"),
            "failed": sum(1 for n in nodes if n.status == "failed"),
            "running": sum(1 for n in nodes if n.status == "running"),
        }
        return {
            "thread_id": thread_id,
            "exists": True,
            "nodes": [n.to_dict() for n in nodes],
            "edges": edges,
            "layers": layers,
            "critical_path": critical_path,
            "width": width,
            "experts": _build_experts(nodes, expert_roster),
            "stats": stats,
            "next_seq": log.seq,
        }
    except Exception:  # noqa: BLE001 — 投影失败 → 空看板, 看板不能带崩主循环
        logger.warning("board: 构建 %s 看板失败", thread_id, exc_info=True)
        return _empty_board(thread_id)


def _empty_board(thread_id: str) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "exists": False,
        "nodes": [],
        "edges": [],
        "layers": [],
        "critical_path": [],
        "width": 0,
        "experts": [],
        "stats": {"total": 0, "done": 0, "failed": 0, "running": 0},
        "next_seq": 0,
    }
