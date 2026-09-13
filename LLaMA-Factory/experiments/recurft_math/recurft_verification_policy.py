"""Lightweight verification-frequency controllers for RecurFT decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AgreementGateDecision:
    """Result of one exact-agreement observation."""

    qualified: bool
    next_streak: int
    gate_open: bool
    reason: str


def agreement_gate_decision(
    current_streak: int,
    draft_matches: Sequence[bool],
    draft_margins: Sequence[float | None],
    target_margins: Sequence[float | None],
    *,
    min_consecutive_matches: int,
    min_draft_margin: float,
    min_target_margin: float,
) -> AgreementGateDecision:
    """Open the gate only after a fully observed, high-margin exact-match streak.

    Missing observations are treated as uncertainty and reset the streak. A caller can
    use this for both single-token shadow probes and active multi-token verification
    blocks. The gate never relies on verifier probabilities alone.
    """
    if current_streak < 0:
        raise ValueError("current_streak must be non-negative.")
    if min_consecutive_matches <= 0:
        raise ValueError("min_consecutive_matches must be positive.")
    if min_draft_margin < 0.0 or min_target_margin < 0.0:
        raise ValueError("agreement margins must be non-negative.")
    if not (
        len(draft_matches) == len(draft_margins) == len(target_margins)
    ):
        raise ValueError("agreement observations must have equal lengths.")
    if not draft_matches:
        return AgreementGateDecision(False, 0, False, "no_observation")
    if not all(draft_matches):
        return AgreementGateDecision(False, 0, False, "target_mismatch")
    if any(margin is None for margin in draft_margins):
        return AgreementGateDecision(False, 0, False, "missing_draft_margin")
    if any(float(margin) < min_draft_margin for margin in draft_margins):
        return AgreementGateDecision(False, 0, False, "low_draft_margin")
    if any(margin is None for margin in target_margins):
        return AgreementGateDecision(False, 0, False, "missing_target_margin")
    if any(float(margin) < min_target_margin for margin in target_margins):
        return AgreementGateDecision(False, 0, False, "low_target_margin")

    next_streak = current_streak + len(draft_matches)
    return AgreementGateDecision(
        True,
        next_streak,
        next_streak >= min_consecutive_matches,
        "qualified",
    )
