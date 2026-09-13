import random

import pytest
import torch

from experiments.recurft_math.recurft_speculative_generate import (
    SuffixNgramIndex,
    allowed_category_prefix_length,
    auto_route_monitor_decision,
    count_accepted_draft_margin_records,
    draft_margin_threshold_for_position,
    find_suffix_ngram_draft,
    latent_margin_gate_route_update,
    latent_target_margin_allows,
    limit_target_match_expand_records,
    limit_ngram_draft_tokens,
    ngram_predecode_cost_route,
    ngram_predecode_occurrence_route,
    ngram_width_for_position,
    ngram_wide_route_decision,
    ngram_unchecked_gate_allows,
    parse_position_width_schedule,
    precommit_unchecked_prefix_length,
    target_match_accepts_candidate,
    top_token_and_logit_margin,
    unchecked_budget_exhausted_for_position,
    verification_risk_budget_decision,
    verifier_previous_score_rejects,
)


def test_allowed_category_prefix_length_stops_at_first_reject() -> None:
    allowed = {"word", "space"}
    assert allowed_category_prefix_length(["word", "space", "word"], allowed) == 3
    assert allowed_category_prefix_length(["word", "number", "word"], allowed) == 1
    assert allowed_category_prefix_length(["symbol", "word"], allowed) == 0


def test_verifier_previous_score_requires_an_existing_score() -> None:
    assert not verifier_previous_score_rejects([], [], 0.5)
    assert not verifier_previous_score_rejects([None], [0.5], 0.5)
    assert verifier_previous_score_rejects([0.4], [0.5], 0.1)
    assert not verifier_previous_score_rejects([0.6], [0.5], 0.9)
    assert verifier_previous_score_rejects([0.4], [None], 0.5)


def test_verification_risk_budget_stops_before_exceeding_budget() -> None:
    stop, survival, risk = verification_risk_budget_decision(0.9, 0.8, 0.25, 4, 2)
    assert stop
    assert survival == pytest.approx(0.72)
    assert risk == pytest.approx(0.28)


def test_verification_risk_budget_honors_minimum_block() -> None:
    stop, survival, risk = verification_risk_budget_decision(1.0, 0.5, 0.1, 1, 2)
    assert not stop
    assert survival == pytest.approx(0.5)
    assert risk == pytest.approx(0.5)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_verification_risk_budget_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        verification_risk_budget_decision(1.0, probability, 0.5, 2, 2)


def test_indexed_ngram_lookup_matches_full_scan() -> None:
    rng = random.Random(20260715)
    for _ in range(200):
        prompt = [rng.randrange(12) for _ in range(rng.randrange(2, 96))]
        generated = [rng.randrange(12) for _ in range(rng.randrange(0, 48))]
        first_token = rng.randrange(12)
        min_size = rng.randrange(1, 4)
        max_size = rng.randrange(min_size, 9)
        max_draft = rng.randrange(1, 33)
        index = SuffixNgramIndex(prompt, min_size, max_size)
        index.append(generated)
        expected = find_suffix_ngram_draft(
            prompt + generated + [first_token],
            max_draft,
            min_size,
            max_size,
        )
        assert index.find(first_token, max_draft) == expected


def test_indexed_ngram_lookup_tracks_incremental_commits() -> None:
    rng = random.Random(17)
    prompt = [rng.randrange(8) for _ in range(64)]
    generated: list[int] = []
    index = SuffixNgramIndex(prompt, 2, 8)
    for _ in range(80):
        first_token = rng.randrange(8)
        assert index.find(first_token, 31) == find_suffix_ngram_draft(
            prompt + generated + [first_token], 31, 2, 8
        )
        committed = [first_token] + [rng.randrange(8) for _ in range(rng.randrange(0, 4))]
        generated.extend(committed)
        index.append(committed)


def test_prompt_latest_ngram_lookup_ignores_generated_occurrences() -> None:
    prompt = [1, 2, 3, 9, 1, 2, 3, 8]
    generated = [1, 2, 3, 7, 1, 2]
    index = SuffixNgramIndex(prompt, 2, 3)
    index.append(generated)

    assert index.find(3, 1, "latest") == ([7], 3)
    assert index.find(3, 1, "prompt_latest") == ([8], 3)


def test_consensus_ngram_lookup_uses_majority_with_recency_tie_break() -> None:
    prompt = [1, 2, 5, 1, 2, 5, 1, 2, 6, 1]
    index = SuffixNgramIndex(prompt, 2, 2)

    assert index.find(2, 1, "latest") == ([6], 2)
    assert index.find(2, 1, "consensus") == ([5], 2)

    tied = SuffixNgramIndex([1, 2, 5, 1, 2, 6, 1], 2, 2)
    assert tied.find(2, 1, "consensus") == ([6], 2)


def test_weighted_ngram_lookup_can_prefer_recent_minority() -> None:
    prompt = [1, 2, 5, 1, 2, 5, 1, 2, 6, 1]
    index = SuffixNgramIndex(prompt, 2, 2)

    assert index.find(2, 1, "consensus") == ([5], 2)
    assert index.find(2, 1, "weighted", 0.5) == ([6], 2)


def test_indexed_ngram_lookup_returns_distinct_recent_branches() -> None:
    index = SuffixNgramIndex([1, 2, 5, 6, 1, 2, 7, 8, 1], 2, 2)

    branches, size = index.find_branches(2, 2, 4)
    assert size == 2
    assert branches == [[7, 8], [5, 6]]


def test_prompt_consensus_does_not_cross_prompt_boundary() -> None:
    prompt = [1, 2, 5, 1, 2]
    index = SuffixNgramIndex(prompt, 2, 2)
    index.append([6, 1])

    continuation, size = index.find(2, 4, "prompt_consensus")
    assert size == 2
    assert continuation == [5, 1, 2]


def test_predecode_occurrence_route_preserves_schedules_and_routes_request_regimes() -> None:
    common = {
        "enabled": True,
        "medium_prompt_tokens": 512,
        "long_prompt_tokens": 2048,
        "short_output_tokens": 128,
    }
    assert ngram_predecode_occurrence_route(
        **common,
        prompt_tokens=100,
        requested_output_tokens=1024,
        has_position_schedule=True,
    ) == ("consensus", "explicit_position_schedule_long_output")
    assert ngram_predecode_occurrence_route(
        **common,
        prompt_tokens=718,
        requested_output_tokens=128,
        has_position_schedule=False,
    ) == ("consensus", "medium_prompt_short_output")
    assert ngram_predecode_occurrence_route(
        **common,
        prompt_tokens=744,
        requested_output_tokens=256,
        has_position_schedule=False,
    ) == ("prompt_consensus", "medium_prompt_long_output")
    assert ngram_predecode_occurrence_route(
        **common,
        prompt_tokens=3047,
        requested_output_tokens=512,
        has_position_schedule=False,
    ) == ("consensus", "long_prompt_long_output")
    assert ngram_predecode_occurrence_route(
        **common,
        prompt_tokens=4096,
        requested_output_tokens=64,
        has_position_schedule=False,
    ) == ("latest", "short_or_long_prompt_short_output")


def test_predecode_cost_route_preserves_schedules_and_routes_request_regimes() -> None:
    common = {
        "enabled": True,
        "medium_prompt_tokens": 512,
        "long_prompt_tokens": 2048,
        "short_output_tokens": 128,
        "narrow_width": 4,
        "medium_width": 16,
        "wide_width": 64,
    }
    assert ngram_predecode_cost_route(
        **common,
        prompt_tokens=4096,
        requested_output_tokens=64,
        has_position_schedule=True,
    ) == (None, "explicit_position_schedule")
    assert ngram_predecode_cost_route(
        **common,
        prompt_tokens=4096,
        requested_output_tokens=64,
        has_position_schedule=False,
    ) == (64, "long_prompt_short_output")
    assert ngram_predecode_cost_route(
        **common,
        prompt_tokens=1024,
        requested_output_tokens=512,
        has_position_schedule=False,
    ) == (16, "medium_prompt_long_output")
    assert ngram_predecode_cost_route(
        **common,
        prompt_tokens=4096,
        requested_output_tokens=512,
        has_position_schedule=False,
    ) == (4, "short_or_long_prompt_long_output")


def test_draft_margin_acceptance_respects_sequence_budget() -> None:
    records = [
        {"margin": 4.0, "draft_category_allowed": True},
        {"margin": 3.5, "draft_category_allowed": True},
        {"margin": 3.0, "draft_category_allowed": True},
    ]
    assert count_accepted_draft_margin_records(records, 2.75, None, 0) == 3
    assert count_accepted_draft_margin_records(records, 2.75, 4, 2) == 2
    assert count_accepted_draft_margin_records(records, 2.75, 2, 2) == 0


def test_target_match_expand_stops_at_mismatch_budget() -> None:
    records = [
        {"match": True},
        {"match": False},
        {"match": True},
        {"match": False},
        {"match": True},
    ]
    assert limit_target_match_expand_records(records, 5, 0, None) == 5
    assert limit_target_match_expand_records(records, 5, 0, 1) == 3
    assert limit_target_match_expand_records(records, 5, 1, 1) == 1


def test_draft_margin_schedule_uses_position_threshold() -> None:
    threshold = lambda position: draft_margin_threshold_for_position(
        2.75,
        position,
        256,
        512,
        early_threshold=4.0,
        mid_threshold=3.0,
        late_threshold=2.0,
    )
    assert threshold(255) == 4.0
    assert threshold(256) == 3.0
    assert threshold(511) == 3.0
    assert threshold(512) == 2.0

    records = [
        {"margin": 3.5, "generated_position": 255},
        {"margin": 3.5, "generated_position": 256},
    ]
    assert (
        count_accepted_draft_margin_records(
            records,
            2.75,
            None,
            0,
            margin_threshold_for_position=threshold,
        )
        == 0
    )
    assert (
        count_accepted_draft_margin_records(
            records[1:],
            2.75,
            None,
            0,
            margin_threshold_for_position=threshold,
        )
        == 1
    )


def test_draft_margin_acceptance_stops_at_first_rejection() -> None:
    records = [
        {"margin": 4.0, "draft_category_allowed": True},
        {"margin": 2.5, "draft_category_allowed": True},
        {"margin": 5.0, "draft_category_allowed": True},
    ]
    assert count_accepted_draft_margin_records(records, 2.75, 8, 0) == 1

    records[1] = {"margin": 4.0, "draft_category_allowed": False}
    assert count_accepted_draft_margin_records(records, 2.75, 8, 0) == 1


def test_draft_margin_acceptance_respects_window_budget() -> None:
    records = [
        {"margin": 4.0, "draft_category_allowed": True, "generated_position": 70},
        {"margin": 3.5, "draft_category_allowed": True, "generated_position": 71},
    ]
    assert count_accepted_draft_margin_records(
        records,
        2.75,
        None,
        0,
        max_unchecked_per_window=1,
        unchecked_budget_window_tokens=64,
        accepted_unchecked_positions=[],
    ) == 1
    assert count_accepted_draft_margin_records(
        records,
        2.75,
        None,
        0,
        max_unchecked_per_window=1,
        unchecked_budget_window_tokens=64,
        accepted_unchecked_positions=[65],
    ) == 0


def test_draft_margin_window_budget_resets_at_boundary() -> None:
    records = [
        {"margin": 4.0, "draft_category_allowed": True, "generated_position": 128},
    ]
    assert count_accepted_draft_margin_records(
        records,
        2.75,
        None,
        0,
        max_unchecked_per_window=1,
        unchecked_budget_window_tokens=64,
        accepted_unchecked_positions=[127],
    ) == 1


def test_ngram_width_adapts_to_match_size() -> None:
    tokens = list(range(7))
    widths = {2: 3, 3: 4, 8: 7}
    assert limit_ngram_draft_tokens(tokens, 2, widths) == [0, 1, 2]
    assert limit_ngram_draft_tokens(tokens, 3, widths) == [0, 1, 2, 3]
    assert limit_ngram_draft_tokens(tokens, 8, widths) == tokens
    assert limit_ngram_draft_tokens(tokens, 5, widths) == tokens


def test_ngram_width_schedule_uses_latest_position_boundary() -> None:
    schedule = parse_position_width_schedule(
        "0:7,512:31,768:63",
        flag_name="--ngram-max-draft-tokens-by-position",
    )
    assert ngram_width_for_position(5, 0, schedule) == 7
    assert ngram_width_for_position(5, 511, schedule) == 7
    assert ngram_width_for_position(5, 512, schedule) == 31
    assert ngram_width_for_position(5, 767, schedule) == 31
    assert ngram_width_for_position(5, 768, schedule) == 63


def test_ngram_width_schedule_preserves_default_before_first_boundary() -> None:
    schedule = parse_position_width_schedule(
        "512:31",
        flag_name="--ngram-max-draft-tokens-by-position",
    )
    assert ngram_width_for_position(7, 511, schedule) == 7
    assert ngram_width_for_position(7, 512, schedule) == 31


def test_ngram_wide_route_waits_for_enough_wide_hits() -> None:
    assert ngram_wide_route_decision(7, 0, 8, 0.2) == (False, None)
    assert ngram_wide_route_decision(8, 1, 8, 0.2) == (True, 0.125)
    assert ngram_wide_route_decision(8, 2, 8, 0.2) == (False, 0.25)


@pytest.mark.parametrize(
    ("threshold", "margin", "expected"),
    [
        (0.0, None, True),
        (3.0, 2.99, False),
        (3.0, 3.0, True),
        (3.0, 5.0, True),
        (3.0, None, False),
    ],
)
def test_latent_target_margin_gate(
    threshold: float,
    margin: float | None,
    expected: bool,
) -> None:
    assert latent_target_margin_allows(threshold, margin) is expected


def test_latent_margin_gate_route_closes_on_configured_streak() -> None:
    streak = 0
    for expected_streak in (1, 2, 3):
        streak, close = latent_margin_gate_route_update(True, streak, 4)
        assert streak == expected_streak
        assert close is False
    assert latent_margin_gate_route_update(True, streak, 4) == (4, True)


def test_latent_margin_gate_route_pass_resets_streak_and_zero_disables_close() -> None:
    assert latent_margin_gate_route_update(False, 7, 4) == (0, False)
    assert latent_margin_gate_route_update(True, 7, 0) == (8, False)


def test_auto_route_monitor_preserves_one_shot_and_supports_periodic_checks() -> None:
    assert auto_route_monitor_decision(31, 4, 32, 0.10, 16) == (
        False,
        False,
        32,
        None,
    )
    assert auto_route_monitor_decision(32, 4, 32, 0.10, 0) == (
        True,
        False,
        0,
        0.125,
    )
    assert auto_route_monitor_decision(32, 4, 32, 0.10, 16) == (
        True,
        False,
        48,
        0.125,
    )
    assert auto_route_monitor_decision(48, 4, 48, 0.10, 16) == (
        True,
        True,
        0,
        4 / 48,
    )


@pytest.mark.parametrize(
    ("match_size", "position", "min_match", "min_position", "expected"),
    [
        (8, 512, 8, 512, True),
        (7, 512, 8, 512, False),
        (8, 511, 8, 512, False),
        (None, 800, 8, 512, False),
        (8, 800, 0, 512, False),
    ],
)
def test_ngram_unchecked_gate(
    match_size: int | None,
    position: int,
    min_match: int,
    min_position: int,
    expected: bool,
) -> None:
    assert ngram_unchecked_gate_allows(match_size, position, min_match, min_position) is expected


def test_ngram_precommit_prefix_respects_sequence_budget() -> None:
    assert precommit_unchecked_prefix_length("ngram_prefix", [None] * 7, 512, 0.0, 2, 0) == 3
    assert precommit_unchecked_prefix_length("ngram_prefix", [None] * 7, 512, 0.0, 2, 1) == 2
    assert precommit_unchecked_prefix_length("ngram_prefix", [None] * 7, 512, 0.0, 2, 2) == 1


def test_ngram_precommit_prefix_respects_window_budget() -> None:
    assert precommit_unchecked_prefix_length(
        "ngram_prefix",
        [None] * 7,
        127,
        0.0,
        None,
        0,
        max_unchecked_per_window=1,
        unchecked_budget_window_tokens=128,
        accepted_unchecked_positions=[127],
    ) == 2


def test_ngram_precommit_prefix_stops_before_sensitive_token_category() -> None:
    assert precommit_unchecked_prefix_length(
        "ngram_prefix",
        [None] * 7,
        512,
        0.0,
        2,
        0,
        ngram_draft_allowed=[True, False, True, True, True, True, True],
    ) == 2
    assert precommit_unchecked_prefix_length(
        "ngram_prefix",
        [None] * 7,
        512,
        0.0,
        2,
        0,
        ngram_draft_allowed=[False, True, True, True, True, True, True],
    ) == 1


def test_precommit_whole_block_keeps_all_drafts() -> None:
    assert precommit_unchecked_prefix_length("whole_block", [None, None, None], 10, 0.0, None, 0) == 4


def test_precommit_draft_margin_trims_before_target_forward() -> None:
    assert (
        precommit_unchecked_prefix_length(
            "draft_margin",
            [4.0, 2.5, 5.0],
            128,
            2.75,
            8,
            0,
        )
        == 2
    )
    assert (
        precommit_unchecked_prefix_length(
            "draft_margin",
            [4.0, 3.5],
            128,
            2.75,
            1,
            0,
        )
        == 2
    )


def test_precommit_draft_margin_uses_position_schedule() -> None:
    threshold = lambda position: 4.0 if position < 256 else 2.5
    assert (
        precommit_unchecked_prefix_length(
            "draft_margin",
            [3.5, 3.5],
            254,
            2.75,
            8,
            0,
            margin_threshold_for_position=threshold,
        )
        == 1
    )
    assert (
        precommit_unchecked_prefix_length(
            "draft_margin",
            [3.5, 3.5],
            255,
            2.75,
            8,
            0,
            margin_threshold_for_position=threshold,
        )
        == 3
    )


def test_unchecked_budget_exhaustion_short_circuits_global_budget() -> None:
    assert unchecked_budget_exhausted_for_position(129, 2, 2)
    assert not unchecked_budget_exhausted_for_position(129, 2, 1)
    assert not unchecked_budget_exhausted_for_position(129, None, 8)


def test_unchecked_budget_exhaustion_is_window_local() -> None:
    assert unchecked_budget_exhausted_for_position(
        70,
        None,
        1,
        max_unchecked_per_window=1,
        unchecked_budget_window_tokens=64,
        accepted_unchecked_positions=[65],
    )
    assert not unchecked_budget_exhausted_for_position(
        128,
        None,
        1,
        max_unchecked_per_window=1,
        unchecked_budget_window_tokens=64,
        accepted_unchecked_positions=[65],
    )


def test_target_match_lambda_one_is_exact() -> None:
    assert target_match_accepts_candidate(True, 0.0, 1.0)
    assert not target_match_accepts_candidate(False, 1.0, 1.0)
    assert target_match_accepts_candidate(True, None, 1.0)
    assert not target_match_accepts_candidate(False, None, 1.0)


def test_target_match_schedule_endpoints_and_relative_support() -> None:
    assert target_match_accepts_candidate(False, 0.0, 0.0)
    assert target_match_accepts_candidate(False, None, 0.0)
    assert target_match_accepts_candidate(False, 0.6, 0.5)
    assert not target_match_accepts_candidate(False, 0.4, 0.5)
    with pytest.raises(ValueError, match="relative support"):
        target_match_accepts_candidate(False, None, 0.5)


def test_top_token_and_logit_margin() -> None:
    token, margin = top_token_and_logit_margin(torch.tensor([[1.0, 4.5, 2.0, -3.0]]))
    assert token == 1
    assert margin == 2.5
