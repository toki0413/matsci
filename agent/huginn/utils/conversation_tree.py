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

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
