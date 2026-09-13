from experiments.recurft_math.recurft_verification_policy import agreement_gate_decision


def decide(streak, matches, draft_margins, target_margins, *, warmup=3):
    return agreement_gate_decision(
        streak,
        matches,
        draft_margins,
        target_margins,
        min_consecutive_matches=warmup,
        min_draft_margin=2.0,
        min_target_margin=3.0,
    )


def test_gate_opens_only_after_consecutive_qualified_agreements():
    first = decide(0, [True], [2.5], [3.5])
    assert first.qualified
    assert first.next_streak == 1
    assert not first.gate_open

    second = decide(first.next_streak, [True, True], [2.1, 4.0], [3.1, 5.0])
    assert second.qualified
    assert second.next_streak == 3
    assert second.gate_open


def test_mismatch_resets_an_open_gate():
    decision = decide(8, [True, False], [5.0, 5.0], [5.0, 5.0])
    assert not decision.qualified
    assert decision.next_streak == 0
    assert not decision.gate_open
    assert decision.reason == "target_mismatch"


def test_low_or_missing_margin_is_uncertainty_and_resets():
    low_draft = decide(2, [True], [1.9], [4.0])
    missing_target = decide(2, [True], [4.0], [None])

    assert low_draft.reason == "low_draft_margin"
    assert low_draft.next_streak == 0
    assert missing_target.reason == "missing_target_margin"
    assert missing_target.next_streak == 0


def test_no_draft_observation_cannot_open_gate():
    decision = decide(20, [], [], [])
    assert decision.reason == "no_observation"
    assert decision.next_streak == 0
    assert not decision.gate_open


def test_observation_lengths_must_match():
    try:
        decide(0, [True], [4.0], [])
    except ValueError as error:
        assert "equal lengths" in str(error)
    else:
        raise AssertionError("expected unequal observations to fail")
