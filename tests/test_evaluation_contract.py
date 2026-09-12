from __future__ import annotations

import pandas as pd
import pytest

from src.archive.evaluation_contract import assess_candidate_evidence


def _row(axis: str, role: str, gain: float = 2.0) -> dict[str, object]:
    return {
        "axis": axis,
        "role": role,
        "parent_parity": True,
        "recipe_frozen_before_axis": True,
        "gain_vs_incumbent": gain,
        "positive_month_fraction": 0.8,
        "worst_month_gain": -1.0,
        "minimum_domain_gain": 0.1,
        "pitcher_bootstrap_p05": 0.2,
        "crossed_bootstrap_p05": 0.1,
        "block_bootstrap_p05": 0.3,
    }


def test_contract_requires_two_noncontaminated_primary_axes() -> None:
    evidence = pd.DataFrame(
        [
            _row("outer_2022", "nested_outer"),
            _row("outer_2023", "locked_shadow"),
            _row("full_2024", "development_contaminated", gain=100.0),
        ]
    )
    result = assess_candidate_evidence(
        evidence,
        family_trial_count=12,
        family_trials_complete=True,
        reality_check_p_value=0.04,
    )
    assert result["eligible_for_public_probe"] is True


def test_contaminated_gain_cannot_rescue_failed_primary_axis() -> None:
    evidence = pd.DataFrame(
        [
            _row("outer_2022", "nested_outer", gain=-0.1),
            _row("outer_2023", "nested_outer"),
            _row("full_2024", "development_contaminated", gain=1000.0),
        ]
    )
    result = assess_candidate_evidence(
        evidence,
        family_trial_count=3,
        family_trials_complete=True,
        reality_check_p_value=0.01,
    )
    assert result["eligible_for_public_probe"] is False
    assert result["checks"]["all_primary_rows_pass"] is False


def test_incomplete_family_fails_reality_scope() -> None:
    evidence = pd.DataFrame(
        [_row("outer_2022", "nested_outer"), _row("outer_2023", "nested_outer")]
    )
    result = assess_candidate_evidence(
        evidence,
        family_trial_count=2,
        family_trials_complete=False,
        reality_check_p_value=0.02,
    )
    assert result["eligible_for_public_probe"] is False
    assert result["checks"]["family_trials_complete"] is False


def test_duplicate_axis_cannot_fake_two_primary_axes() -> None:
    evidence = pd.DataFrame(
        [_row("outer_2022", "nested_outer"), _row("outer_2022", "nested_outer")]
    )
    with pytest.raises(ValueError, match="exactly once"):
        assess_candidate_evidence(
            evidence,
            family_trial_count=2,
            family_trials_complete=True,
            reality_check_p_value=0.02,
        )


def test_truthy_string_cannot_pass_boolean_contract() -> None:
    rows = [_row("outer_2022", "nested_outer"), _row("outer_2023", "nested_outer")]
    rows[0]["parent_parity"] = "False"
    with pytest.raises(ValueError, match="must contain booleans"):
        assess_candidate_evidence(
            pd.DataFrame(rows),
            family_trial_count=2,
            family_trials_complete=True,
            reality_check_p_value=0.02,
        )


def test_truthy_string_cannot_mark_family_ledger_complete() -> None:
    evidence = pd.DataFrame(
        [_row("outer_2022", "nested_outer"), _row("outer_2023", "nested_outer")]
    )
    with pytest.raises(ValueError, match="must be boolean"):
        assess_candidate_evidence(
            evidence,
            family_trial_count=2,
            family_trials_complete="False",  # type: ignore[arg-type]
            reality_check_p_value=0.02,
        )
