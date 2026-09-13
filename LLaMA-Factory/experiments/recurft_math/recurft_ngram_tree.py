"""Pure helpers for feature-flagged multi-candidate n-gram verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DraftTreeNode:
    token_id: int
    parent: int
    depth: int


@dataclass(frozen=True)
class DraftTree:
    """A prefix-sharing tree whose node zero is the already chosen root token."""

    nodes: tuple[DraftTreeNode, ...]

    def edge_index(self) -> dict[tuple[int, int], int]:
        return {
            (node.parent, node.token_id): index
            for index, node in enumerate(self.nodes)
            if node.parent >= 0
        }


def build_draft_tree(
    root_token_id: int,
    branches: Iterable[Sequence[int]],
    max_nodes: int | None = None,
) -> DraftTree:
    """Merge candidate continuations into a deterministic prefix trie."""

    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must include at least the root node")

    nodes = [DraftTreeNode(token_id=int(root_token_id), parent=-1, depth=0)]
    edges: dict[tuple[int, int], int] = {}
    for branch in branches:
        parent = 0
        for raw_token_id in branch:
            token_id = int(raw_token_id)
            edge = (parent, token_id)
            child = edges.get(edge)
            if child is None:
                if max_nodes is not None and len(nodes) >= max_nodes:
                    break
                child = len(nodes)
                edges[edge] = child
                nodes.append(
                    DraftTreeNode(
                        token_id=token_id,
                        parent=parent,
                        depth=nodes[parent].depth + 1,
                    )
                )
            parent = child
    return DraftTree(nodes=tuple(nodes))


def tree_position_ids(prefix_length: int, tree: DraftTree) -> tuple[int, ...]:
    """Return RoPE positions; siblings at the same depth share a position."""

    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    return tuple(prefix_length + node.depth for node in tree.nodes)


def tree_attention_mask(prefix_length: int, tree: DraftTree) -> tuple[tuple[bool, ...], ...]:
    """Build a query-by-key mask with full prefix and ancestor-only tree access."""

    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")

    width = prefix_length + len(tree.nodes)
    rows: list[tuple[bool, ...]] = []
    for node_index in range(len(tree.nodes)):
        row = [True] * prefix_length + [False] * len(tree.nodes)
        ancestor = node_index
        while ancestor >= 0:
            row[prefix_length + ancestor] = True
            ancestor = tree.nodes[ancestor].parent
        if len(row) != width:
            raise AssertionError("tree attention row has an invalid width")
        rows.append(tuple(row))
    return tuple(rows)


def matching_path_indices(tree: DraftTree, continuation: Sequence[int]) -> tuple[int, ...]:
    """Follow target tokens through the trie and return the longest matching path."""

    edges = tree.edge_index()
    path = [0]
    parent = 0
    for raw_token_id in continuation:
        child = edges.get((parent, int(raw_token_id)))
        if child is None:
            break
        path.append(child)
        parent = child
    return tuple(path)


def target_guided_leaf_path(
    tree: DraftTree,
    predicted_tokens: Sequence[int],
) -> tuple[tuple[int, ...], int]:
    """Choose the target-matching path, then finish one branch after a mismatch."""

    if len(predicted_tokens) != len(tree.nodes):
        raise ValueError("predicted_tokens must provide one target token per tree node")

    edges = tree.edge_index()
    children: dict[int, list[int]] = {}
    for index, node in enumerate(tree.nodes):
        if node.parent >= 0:
            children.setdefault(node.parent, []).append(index)

    path = [0]
    current = 0
    matched_drafts = 0
    while True:
        child = edges.get((current, int(predicted_tokens[current])))
        if child is not None:
            path.append(child)
            current = child
            matched_drafts += 1
            continue

        remaining_children = children.get(current, [])
        if not remaining_children:
            break
        current = remaining_children[0]
        path.append(current)
        while children.get(current):
            current = children[current][0]
            path.append(current)
        break
    return tuple(path), matched_drafts


def tree_branch_budget(
    generated_position: int,
    medium_position: int,
    long_position: int,
    early_branches: int,
    mid_branches: int,
    late_branches: int,
) -> int:
    if generated_position < 0:
        raise ValueError("generated_position must be non-negative")
    if not 0 <= medium_position <= long_position:
        raise ValueError("tree branch positions must satisfy 0 <= medium <= long")
    if min(early_branches, mid_branches, late_branches) < 1:
        raise ValueError("tree branch budgets must be positive")
    if generated_position < medium_position:
        return early_branches
    if generated_position < long_position:
        return mid_branches
    return late_branches


def selected_tree_cache_positions(prefix_length: int, path_indices: Sequence[int]) -> tuple[int, ...]:
    """Map flattened tree indices to temporary KV positions for branch adoption."""

    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    if any(index < 0 for index in path_indices):
        raise ValueError("path indices must be non-negative")
    return tuple(prefix_length + index for index in path_indices)
