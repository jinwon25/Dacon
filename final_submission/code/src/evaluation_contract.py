"""Machine-checkable promotion contract for post-1158 candidates.

The contract separates nested outer evidence from development-contaminated
confirmation axes.  A positive point estimate on a reused axis can never, by
itself, authorize a Public probe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "axis",
    "role",
    "parent_parity",
    "recipe_frozen_before_axis",
    "gain_vs_incumbent",
    "positive_month_fraction",
    "worst_month_gain",
    "minimum_domain_gain",
    "pitcher_bootstrap_p05",
    "crossed_bootstrap_p05",
    "block_bootstrap_p05",
}
ALLOWED_ROLES = {"nested_outer", "locked_shadow", "development_contaminated"}


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    """Accept actual booleans only, avoiding truthy strings such as ``"False"``."""

    valid = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain booleans")
    return series.astype(bool)


def assess_candidate_evidence(
    evidence: pd.DataFrame,
    *,
    family_trial_count: int,
    family_trials_complete: bool,
    reality_check_p_value: float,
    minimum_primary_axes: int = 2,
    minimum_month_fraction: float = 0.75,
    minimum_worst_month_gain: float = -5.0,
    reality_alpha: float = 0.10,
) -> dict[str, Any]:
    """Assess evidence without treating reused development axes as independent."""

    missing = REQUIRED_COLUMNS.difference(evidence.columns)
    if missing:
        raise ValueError(f"evidence missing columns: {sorted(missing)}")
    if evidence.empty:
        raise ValueError("evidence is empty")
    if family_trial_count < 1:
        raise ValueError("family_trial_count must be positive")
    if not isinstance(family_trials_complete, (bool, np.bool_)):
        raise ValueError("family_trials_complete must be boolean")
    if minimum_primary_axes < 1:
        raise ValueError("minimum_primary_axes must be positive")
    if not 0.0 <= float(minimum_month_fraction) <= 1.0:
        raise ValueError("minimum_month_fraction must be in [0, 1]")
    if not np.isfinite(float(reality_check_p_value)):
        raise ValueError("reality_check_p_value must be finite")
    if not 0.0 <= float(reality_check_p_value) <= 1.0:
        raise ValueError("reality_check_p_value must be in [0, 1]")

    local = evidence.copy()
    if local[["axis", "role"]].isna().any().any():
        raise ValueError("axis and role must be non-null")
    local["axis"] = local["axis"].astype(str).str.strip()
    if local["axis"].eq("").any():
        raise ValueError("axis must be non-empty")
    if local["axis"].duplicated().any():
        raise ValueError("each evidence axis must appear exactly once")
    unknown_roles = set(local["role"].astype(str)).difference(ALLOWED_ROLES)
    if unknown_roles:
        raise ValueError(f"unknown evidence roles: {sorted(unknown_roles)}")
    numeric = [
        "gain_vs_incumbent",
        "positive_month_fraction",
        "worst_month_gain",
        "minimum_domain_gain",
        "pitcher_bootstrap_p05",
        "crossed_bootstrap_p05",
        "block_bootstrap_p05",
    ]
    for column in numeric:
        local[column] = pd.to_numeric(local[column], errors="raise")
    if not np.isfinite(local[numeric].to_numpy(np.float64)).all():
        raise ValueError("evidence contains non-finite metrics")
    if not local["positive_month_fraction"].between(0.0, 1.0).all():
        raise ValueError("positive_month_fraction must be in [0, 1]")
    local["parent_parity"] = _strict_bool(local["parent_parity"], "parent_parity")
    local["recipe_frozen_before_axis"] = _strict_bool(
        local["recipe_frozen_before_axis"], "recipe_frozen_before_axis"
    )

    primary = local.loc[local["role"].isin(["nested_outer", "locked_shadow"])]
    primary_rows_pass = (
        primary["parent_parity"]
        & primary["recipe_frozen_before_axis"]
        & primary["gain_vs_incumbent"].gt(0.0)
        & primary["positive_month_fraction"].ge(float(minimum_month_fraction))
        & primary["worst_month_gain"].gt(float(minimum_worst_month_gain))
        & primary["minimum_domain_gain"].ge(0.0)
        & primary["pitcher_bootstrap_p05"].gt(0.0)
        & primary["crossed_bootstrap_p05"].gt(0.0)
        & primary["block_bootstrap_p05"].gt(0.0)
    )
    checks = {
        "minimum_primary_axes": bool(len(primary) >= int(minimum_primary_axes)),
        "all_primary_rows_pass": bool(len(primary) > 0 and primary_rows_pass.all()),
        "family_trials_complete": bool(family_trials_complete),
        "family_trial_count_documented": bool(family_trial_count >= 1),
        "reality_check_pass": bool(float(reality_check_p_value) <= float(reality_alpha)),
    }
    eligible = bool(all(checks.values()))
    return {
        "eligible_for_public_probe": eligible,
        "checks": checks,
        "primary_axes": primary["axis"].astype(str).tolist(),
        "development_contaminated_axes": local.loc[
            local["role"].eq("development_contaminated"), "axis"
        ].astype(str).tolist(),
        "family_trial_count": int(family_trial_count),
        "reality_check_p_value": float(reality_check_p_value),
        "interpretation": (
            "development_contaminated axes are diagnostics only; they cannot satisfy "
            "the minimum-primary-axis requirement"
        ),
    }
