import unittest

from experiments.recurft_math.recurft_ngram_tree import (
    build_draft_tree,
    matching_path_indices,
    selected_tree_cache_positions,
    target_guided_leaf_path,
    tree_attention_mask,
    tree_branch_budget,
    tree_position_ids,
)


class DraftTreeTest(unittest.TestCase):
    def test_tree_merges_shared_prefixes_deterministically(self) -> None:
        tree = build_draft_tree(10, [[11, 12, 13], [11, 12, 14], [11, 15], [16]])

        self.assertEqual(
            [(node.token_id, node.parent, node.depth) for node in tree.nodes],
            [
                (10, -1, 0),
                (11, 0, 1),
                (12, 1, 2),
                (13, 2, 3),
                (14, 2, 3),
                (15, 1, 2),
                (16, 0, 1),
            ],
        )

    def test_tree_mask_allows_prefix_and_ancestors_only(self) -> None:
        tree = build_draft_tree(10, [[11, 12, 13], [11, 12, 14], [11, 15], [16]])
        mask = tree_attention_mask(2, tree)

        self.assertEqual(mask[4], (True, True, True, True, True, False, True, False, False))
        self.assertEqual(mask[6], (True, True, True, False, False, False, False, False, True))
        self.assertEqual(tree_position_ids(2, tree), (2, 3, 4, 5, 5, 4, 3))

    def test_matching_path_and_cache_positions_stop_at_first_missing_edge(self) -> None:
        tree = build_draft_tree(10, [[11, 12, 13], [11, 12, 14], [16]])

        path = matching_path_indices(tree, [11, 12, 14, 99])
        self.assertEqual(path, (0, 1, 2, 4))
        self.assertEqual(selected_tree_cache_positions(20, path), (20, 21, 22, 24))

    def test_tree_budget_keeps_existing_prefixes_but_stops_new_nodes(self) -> None:
        tree = build_draft_tree(10, [[11, 12, 13], [11, 12, 14], [11]], max_nodes=4)
        self.assertEqual([node.token_id for node in tree.nodes], [10, 11, 12, 13])

    def test_target_guided_path_follows_matches_then_finishes_a_branch(self) -> None:
        tree = build_draft_tree(10, [[11, 12, 13], [11, 12, 14], [16, 17]])
        predictions = [11, 12, 99, 0, 0, 0, 0]

        path, matched = target_guided_leaf_path(tree, predictions)
        self.assertEqual(path, (0, 1, 2, 3))
        self.assertEqual(matched, 2)

    def test_target_guided_path_can_accept_a_complete_leaf(self) -> None:
        tree = build_draft_tree(10, [[11, 12], [16, 17]])
        path, matched = target_guided_leaf_path(tree, [11, 12, 0, 0, 0])
        self.assertEqual(path, (0, 1, 2))
        self.assertEqual(matched, 2)

    def test_tree_branch_budget_uses_position_regimes(self) -> None:
        values = [
            tree_branch_budget(position, 64, 192, 1, 2, 4)
            for position in (0, 63, 64, 191, 192, 512)
        ]
        self.assertEqual(values, [1, 1, 2, 2, 4, 4])

    def test_tree_helpers_reject_invalid_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "root"):
            build_draft_tree(10, [], max_nodes=0)
        tree = build_draft_tree(10, [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            tree_attention_mask(-1, tree)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            selected_tree_cache_positions(0, [-1])


if __name__ == "__main__":
    unittest.main()
