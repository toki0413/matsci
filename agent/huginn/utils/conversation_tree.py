"""Conversation branch tree — enables forking and backtracking in research dialogs.

Each node stores a message (user or assistant) and its children. The agent
can fork from any node to explore an alternative hypothesis without losing
the original conversation. The active path is the sequence of nodes from
the root to the current leaf.

Usage:
    tree = ConversationTree()
    root = tree.add_message("user", "What is the band gap of Si?")
    reply = tree.add_message("assistant", "1.12 eV", parent=root)
    # Fork from root to explore a different question
    alt_reply = tree.add_message("assistant", "Let me calculate it...", parent=root)
    tree.set_active_leaf(alt_reply)
    messages = tree.active_path_messages()
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# P1-2 CRDT 升级 ConversationTree 分支合并: research mode 进入 hypothesis/planning
# 时 fork_from_active 创建兄弟分支, 但没有合并语义 — 多分支回主干时 findings/evidence
# 应用 G-Set union, best_value 用 LWW. 当前是"最后写的赢", 跨分支信息丢失.
# ponytail: 复用 subagent_tool._crdt_merge, 不重复实现半格.
# ceiling: 只合 metadata 字段, content (消息文本) 仍走 LLM 摘要 (v2).
def _crdt_branch_merge_enabled() -> bool:
    """toggle: env HUGINN_CRDT_BRANCH_MERGE (默认 on). off 时不合并分支."""
    return os.environ.get("HUGINN_CRDT_BRANCH_MERGE", "1") != "0"


@dataclass
class ConversationNode:
    """A single message node in the conversation tree."""

    id: str
    role: str  # "user", "assistant", "system", "tool"
    content: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationNode:
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


class ConversationTree:
    """A tree of conversation messages supporting branching and backtracking.

    Nodes are stored in a flat dict keyed by id. The ``_active_leaf_id``
    tracks which leaf is currently the "tip" of the conversation — new
    messages are appended as children of the active leaf.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ConversationNode] = {}
        self._root_id: str | None = None
        self._active_leaf_id: str | None = None

    @property
    def root_id(self) -> str | None:
        return self._root_id

    @property
    def active_leaf_id(self) -> str | None:
        return self._active_leaf_id

    def add_message(
        self,
        role: str,
        content: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationNode:
        """Add a message node. If parent_id is None, appends to the active leaf."""
        if parent_id is None:
            parent_id = self._active_leaf_id

        node_id = uuid.uuid4().hex[:12]
        node = ConversationNode(
            id=node_id,
            role=role,
            content=content,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node

        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(node_id)

        if self._root_id is None:
            self._root_id = node_id

        self._active_leaf_id = node_id
        return node

    def fork(self, from_node_id: str) -> ConversationNode | None:
        """Create a new branch starting from an existing node.

        Sets the active leaf to the forked node's parent so the next
        add_message creates a sibling of the original node.
        """
        node = self._nodes.get(from_node_id)
        if node is None:
            return None
        # Set active leaf to the parent so the next message forks from here
        self._active_leaf_id = node.parent_id
        return node

    def fork_from_active(self) -> ConversationNode | None:
        """OAK 启发: 从当前 active leaf fork, 下条消息成为兄弟节点.

        用于 phase 转移时标记新实验分支 — 不丢弃旧分支, 保留完整探索树.
        """
        if self._active_leaf_id is None:
            return None
        return self.fork(self._active_leaf_id)

    def merge_branch_into_active(
        self, branch_leaf_id: str,
    ) -> ConversationNode | None:
        """P1-2 CRDT 合并分支 metadata 回 active leaf.

        收集 branch_leaf_id 到 fork 点 (跟 active path 的最近共同祖先) 路径上
        所有节点的 metadata, 用 _crdt_merge 半格合并:
        - findings/evidence: G-Set union (dedupe by content hash)
        - best_value/best_encut 等 LWW 字段: created_at 新者胜
        - success: OR-join
        合并结果写到 active leaf 的 metadata, 跨分支信息不丢.

        ponytail: 只合 metadata, content (消息文本) 不合 (留给 LLM 摘要 v2).
        ceiling: 共同祖先用 path 交集算, O(depth²), 树深小时够用.
        """
        if not _crdt_branch_merge_enabled():
            return None
        if branch_leaf_id not in self._nodes or self._active_leaf_id is None:
            return None
        if branch_leaf_id == self._active_leaf_id:
            return None  # 自己合自己无意义

        # 找共同祖先: branch path 跟 active path 的最后一个公共节点
        branch_path = self._path_to_root(branch_leaf_id)
        active_path = self._path_to_root(self._active_leaf_id)
        active_set = set(active_path)
        lca = None
        for nid in branch_path:
            if nid in active_set:
                lca = nid
                break
        if lca is None:
            return None  # 无共同祖先, 不合

        # 收集 branch 上 lca 之后 (不含 lca) 到 branch_leaf 的所有 assistant 节点 metadata
        branch_nodes: list[dict] = []
        for nid in branch_path:
            if nid == lca:
                break
            node = self._nodes[nid]
            if node.role == "assistant" and node.metadata:
                # 给 _crdt_merge 加 ts (用 created_at 转 epoch)
                md = dict(node.metadata)
                try:
                    md["ts"] = datetime.fromisoformat(
                        node.created_at.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    md["ts"] = 0.0
                branch_nodes.append(md)

        if not branch_nodes:
            return None  # 分支无 assistant metadata, 无东西可合

        # 收集 active leaf 的 metadata 也加入 (作为 "已合并" 基线)
        active_node = self._nodes[self._active_leaf_id]
        if active_node.metadata:
            md_active = dict(active_node.metadata)
            try:
                md_active["ts"] = datetime.fromisoformat(
                    active_node.created_at.replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                md_active["ts"] = 0.0
            branch_nodes.append(md_active)

        # 调 _crdt_merge (复用 P1-2 半格)
        try:
            from huginn.tools.subagent_tool import _crdt_merge
            merged = _crdt_merge(branch_nodes)
        except Exception:
            return None

        # 把合并字段写回 active leaf metadata (不覆盖原始 content)
        # 只写 G-Set / LWW / belief 字段, 跳过 _crdt_merge 的元信息字段
        skip_keys = {"merged", "n_sources", "sources", "ts", "success", "summary", "errors"}
        for k, v in merged.items():
            if k in skip_keys or k.endswith("_ts"):
                continue
            active_node.metadata[k] = v

        return active_node

    def _path_to_root(self, node_id: str) -> list[str]:
        """从 node_id 往上到 root 的路径 (含自身)."""
        path: list[str] = []
        current: str | None = node_id
        while current is not None and current in self._nodes:
            path.append(current)
            current = self._nodes[current].parent_id
        return path

    def set_active_leaf(self, node_id: str) -> bool:
        """Switch the active conversation path to end at ``node_id``."""
        if node_id not in self._nodes:
            return False
        self._active_leaf_id = node_id
        return True

    def active_path(self) -> list[str]:
        """Return the list of node ids from root to the active leaf."""
        if self._active_leaf_id is None:
            return []
        path: list[str] = []
        current: str | None = self._active_leaf_id
        while current is not None:
            path.append(current)
            node = self._nodes.get(current)
            if node is None:
                break
            current = node.parent_id
        path.reverse()
        return path

    def active_path_messages(self) -> list[dict[str, str]]:
        """Return messages on the active path as dicts with role/content."""
        result = []
        for node_id in self.active_path():
            node = self._nodes[node_id]
            result.append({"role": node.role, "content": node.content})
        return result

    def get_branches(self, node_id: str | None = None) -> list[list[str]]:
        """Return all branches (paths from root to leaf) under ``node_id``.

        If node_id is None, returns all branches in the tree.
        """
        if node_id is None:
            node_id = self._root_id
        if node_id is None or node_id not in self._nodes:
            return []

        branches: list[list[str]] = []
        self._collect_branches(node_id, [], branches)
        return branches

    def _collect_branches(
        self, node_id: str, path: list[str], branches: list[list[str]]
    ) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        current_path = path + [node_id]
        if not node.children_ids:
            branches.append(current_path)
            return
        for child_id in node.children_ids:
            self._collect_branches(child_id, current_path, branches)

    def get_node(self, node_id: str) -> ConversationNode | None:
        return self._nodes.get(node_id)

    def list_nodes(self) -> list[ConversationNode]:
        return list(self._nodes.values())

    def depth(self, node_id: str) -> int:
        """Return the depth of a node (root = 0)."""
        depth = 0
        current = self._nodes.get(node_id)
        while current is not None and current.parent_id is not None:
            depth += 1
            current = self._nodes.get(current.parent_id)
        return depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "root_id": self._root_id,
            "active_leaf_id": self._active_leaf_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTree:
        tree = cls()
        for nid, ndata in data.get("nodes", {}).items():
            tree._nodes[nid] = ConversationNode.from_dict(ndata)
        tree._root_id = data.get("root_id")
        tree._active_leaf_id = data.get("active_leaf_id")
        return tree

    def summary(self) -> dict[str, Any]:
        """Return a summary of the tree structure."""
        branches = self.get_branches()
        return {
            "total_nodes": len(self._nodes),
            "total_branches": len(branches),
            "active_depth": len(self.active_path()),
            "root_id": self._root_id,
            "active_leaf_id": self._active_leaf_id,
        }


def _selfcheck() -> int:
    """assert-based demo for P1-2 CRDT branch merge."""
    import os as _os

    _saved = _os.environ.get("HUGINN_CRDT_BRANCH_MERGE")
    _os.environ["HUGINN_CRDT_BRANCH_MERGE"] = "1"

    # 场景 1: fork + 合并分支 findings (G-Set union)
    t = ConversationTree()
    t.add_message("user", "GaN calc")
    # 主干 assistant 消息带 findings
    a1 = t.add_message("assistant", "main-1", metadata={"findings": ["encut=520 works"]})
    # fork 出分支探索 alternative
    t.fork_from_active()
    b1 = t.add_message("assistant", "branch-1", metadata={"findings": ["kpoints=4x4 better"]})
    # 回主干, 继续走主干
    t.set_active_leaf(a1.id)
    a2 = t.add_message("assistant", "main-2", metadata={"findings": ["converged"]})

    # 合并 branch b1 回 active leaf a2
    merged = t.merge_branch_into_active(b1.id)
    assert merged is not None, "应返回合并后的 active node"
    findings = merged.metadata.get("findings", [])
    assert "encut=520 works" in findings or "kpoints=4x4 better" in findings, \
        f"findings 应 union, got {findings}"
    # branch 的 findings 应进入 active
    assert "kpoints=4x4 better" in findings, \
        f"分支 findings 应合并回主干, got {findings}"

    # 场景 2: LWW 字段 — best_encut 取 created_at 新者
    t2 = ConversationTree()
    t2.add_message("user", "q")
    m1 = t2.add_message("assistant", "main", metadata={"best_encut": 520})
    t2.fork_from_active()
    b2 = t2.add_message("assistant", "branch", metadata={"best_encut": 540})
    t2.set_active_leaf(m1.id)
    # active 新消息 (时间更晚)
    m2 = t2.add_message("assistant", "main2", metadata={"best_encut": 500})
    t2.merge_branch_into_active(b2.id)
    # m2 时间更晚, best_encut 应保留 500 (LWW ts 大者胜)
    assert t2.get_node(m2.id).metadata.get("best_encut") == 500, \
        f"LWW 应取 active (ts 新), got {t2.get_node(m2.id).metadata.get('best_encut')}"

    # 场景 3: toggle off → 不合并, 返回 None
    _os.environ["HUGINN_CRDT_BRANCH_MERGE"] = "0"
    t3 = ConversationTree()
    t3.add_message("user", "q")
    m3 = t3.add_message("assistant", "m", metadata={"findings": ["x"]})
    t3.fork_from_active()
    b3 = t3.add_message("assistant", "b", metadata={"findings": ["y"]})
    t3.set_active_leaf(m3.id)
    out3 = t3.merge_branch_into_active(b3.id)
    assert out3 is None, "toggle off 应返回 None"

    # 场景 4: branch_leaf == active_leaf → 自己合自己无意义
    _os.environ["HUGINN_CRDT_BRANCH_MERGE"] = "1"
    t4 = ConversationTree()
    t4.add_message("user", "q")
    n4 = t4.add_message("assistant", "m")
    out4 = t4.merge_branch_into_active(n4.id)
    assert out4 is None, "自合应返回 None"

    # 场景 5: 幂等 — 两次合并结果一致 (半格公理)
    t5 = ConversationTree()
    t5.add_message("user", "q")
    m5 = t5.add_message("assistant", "main", metadata={"findings": ["a"]})
    t5.fork_from_active()
    b5 = t5.add_message("assistant", "branch", metadata={"findings": ["b"]})
    t5.set_active_leaf(m5.id)
    t5.merge_branch_into_active(b5.id)
    findings_1 = sorted(t5.get_node(t5.active_leaf_id).metadata.get("findings", []))
    t5.merge_branch_into_active(b5.id)
    findings_2 = sorted(t5.get_node(t5.active_leaf_id).metadata.get("findings", []))
    assert findings_1 == findings_2, \
        f"幂等公理: 两次合并应一致, got {findings_1} vs {findings_2}"

    if _saved is None:
        _os.environ.pop("HUGINN_CRDT_BRANCH_MERGE", None)
    else:
        _os.environ["HUGINN_CRDT_BRANCH_MERGE"] = _saved

    print("conversation_tree selfcheck OK (P1-2 CRDT branch merge: union/LWW/toggle/idempotent)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selfcheck())
