"""Audit deployable historical TrackMan command proxies above the 1158 OOF.

The official TrackMan file has release and flight measurements but omits both
plate location and intended target.  Consequently, pitch-level physical values
cannot directly identify command.  This audit focuses on a defensible proxy:
historical within-pitch-family repeatability of release and flight metrics.

Profiles are built from TrackMan seasons strictly before each origin and mapped
through >=99% pure alignment-derived identities.  Association is measured at
the pitcher level against both observed control and residual control above the
exact public-1158 OOF parent.  The hidden test file is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.axes import _cached_v25_axes
from src.archive.v65_trackman_data_census import PHYSICAL_COLUMNS, derive_temporal_entity_map
from src.archive.v66_direct_trackman_gate import _load_pairs


PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR", "F")
MINIMUM_SUPPORT = 20
MINIMUM_PURITY = 0.99
RESIDUAL_PRIOR = 200.0


def weighted_correlation(
    left: pd.Series, right: pd.Series, weight: pd.Series
) -> float:
    x = pd.to_numeric(left, errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(right, errors="coerce").to_numpy(np.float64)
    w = pd.to_numeric(weight, errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if valid.sum() < 3:
        return float("nan")
    x, y, w = x[valid], y[valid], w[valid]
    w = w / w.sum()
    x_centered = x - np.sum(w * x)
    y_centered = y - np.sum(w * y)
    denominator = np.sqrt(
        np.sum(w * np.square(x_centered))
        * np.sum(w * np.square(y_centered))
    )
    if denominator <= 0:
        return float("nan")
    return float(np.sum(w * x_centered * y_centered) / denominator)


def build_trackman_physical_profile(history: pd.DataFrame) -> pd.DataFrame:
    """Build means and repeatability features without outcome information."""

    grouped = history.groupby("pitcher_trackman_id", observed=True)
    profile = grouped.size().rename("tm_n").to_frame()
    overall = grouped[list(PHYSICAL_COLUMNS)].agg(["mean", "std"])
    overall.columns = [f"tm_{feature}_{stat}" for feature, stat in overall.columns]
    profile = profile.join(overall)

    pitch_counts = (
        history.groupby(["pitcher_trackman_id", "pitch_type_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    total = pitch_counts.sum(axis=1).replace(0, np.nan)
    for pitch_group in PITCH_GROUPS:
        values = pitch_counts[pitch_group] if pitch_group in pitch_counts else 0
        profile[f"tm_{pitch_group}_rate"] = values / total
        subset = history.loc[history["pitch_type_group"].astype(str).eq(pitch_group)]
        statistics = subset.groupby("pitcher_trackman_id", observed=True)[
            list(PHYSICAL_COLUMNS)
        ].agg(["mean", "std"])
        statistics.columns = [
            f"tm_{pitch_group}_{feature}_{stat}"
            for feature, stat in statistics.columns
        ]
        profile = profile.join(statistics)

    # A pooled within-family SD is not inflated by different pitch-family means.
    within = (
        history.groupby(
            ["pitcher_trackman_id", "pitch_type_group"], observed=True
        )[list(PHYSICAL_COLUMNS)]
        .agg(["count", "std"])
    )
    for feature in PHYSICAL_COLUMNS:
        local = within[feature].reset_index()
        local = local.loc[local["count"].ge(2) & local["std"].notna()].copy()
        local["weighted_std"] = local["count"] * local["std"]
        pooled = local.groupby("pitcher_trackman_id", observed=True).agg(
            numerator=("weighted_std", "sum"), denominator=("count", "sum")
        )
        profile[f"tm_within_pitch_{feature}_std"] = (
            pooled["numerator"] / pooled["denominator"]
        )

    latest_season = int(history["season"].max())
    latest = history.loc[history["season"].eq(latest_season)]
    latest_grouped = latest.groupby("pitcher_trackman_id", observed=True)
    profile["tm_latest_n"] = latest_grouped.size()
    latest_stats = latest_grouped[list(PHYSICAL_COLUMNS)].agg(["mean", "std"])
    latest_stats.columns = [
        f"tm_latest_{feature}_{stat}" for feature, stat in latest_stats.columns
    ]
    profile = profile.join(latest_stats)
    for feature in PHYSICAL_COLUMNS:
        profile[f"tm_latest_{feature}_mean_delta"] = (
            profile[f"tm_latest_{feature}_mean"] - profile[f"tm_{feature}_mean"]
        )
        profile[f"tm_latest_{feature}_std_ratio"] = (
            profile[f"tm_latest_{feature}_std"]
            / profile[f"tm_{feature}_std"].replace(0, np.nan)
        )
    profile["tm_latest_fraction"] = profile["tm_latest_n"] / profile["tm_n"]
    return profile.reset_index()


def build_origin_profiles(
    pairs: pd.DataFrame,
    trackman: pd.DataFrame,
    origins: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for origin in origins:
        mapping, audit = derive_temporal_entity_map(
            pairs,
            origin,
            minimum_support=MINIMUM_SUPPORT,
            minimum_purity=MINIMUM_PURITY,
        )
        history = trackman.loc[trackman["season"].lt(origin)]
        physical = build_trackman_physical_profile(history)
        profile = mapping[
            ["pitcher_id", "pitcher_trackman_id", "support", "purity"]
        ].merge(
            physical,
            on="pitcher_trackman_id",
            how="left",
            validate="many_to_one",
        )
        profile.insert(0, "origin", origin)
        pieces.append(profile)
        audits.append(
            {
                **audit,
                "profile_rows": int(len(profile)),
                "profile_columns": int(len(profile.columns)),
                "median_trackman_rows": float(profile["tm_n"].median()),
            }
        )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(audits)


def _domain_mask(frame: pd.DataFrame, domain: str) -> np.ndarray:
    if domain == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["domain3"].astype(str).eq(domain).to_numpy()


def pitcher_residual_table(
    frame: pd.DataFrame,
    parent: np.ndarray,
    domain: str,
    *,
    residual_prior: float = RESIDUAL_PRIOR,
) -> pd.DataFrame:
    mask = _domain_mask(frame, domain)
    local = frame.loc[mask, ["pitcher_id", "target"]].copy()
    local["parent"] = np.asarray(parent, dtype=np.float64)[mask]
    local["residual"] = local["target"] - local["parent"]
    output = local.groupby("pitcher_id", observed=True).agg(
        rows=("target", "size"),
        target_rate=("target", "mean"),
        parent_rate=("parent", "mean"),
        residual_mean=("residual", "mean"),
        residual_sum=("residual", "sum"),
    )
    output["residual_eb"] = output["residual_sum"] / (
        output["rows"] + float(residual_prior)
    )
    output["reliability"] = output["rows"] / (
        output["rows"] + float(residual_prior)
    )
    return output.reset_index()


def _signal_rows(
    profile: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    axis: str,
    domain: str,
) -> list[dict[str, object]]:
    joined = residual.merge(profile, on="pitcher_id", how="inner", validate="one_to_one")
    excluded = {
        "origin",
        "pitcher_id",
        "pitcher_trackman_id",
        "support",
        "purity",
        "rows",
        "target_rate",
        "parent_rate",
        "residual_mean",
        "residual_sum",
        "residual_eb",
        "reliability",
    }
    features = [
        column
        for column in profile.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(profile[column])
    ]
    rows: list[dict[str, object]] = []
    for feature in features:
        valid = joined[[feature, "residual_eb", "target_rate", "reliability"]].dropna()
        if len(valid) < 30:
            continue
        rows.append(
            {
                "axis": axis,
                "domain": domain,
                "feature": feature,
                "pitchers": int(len(valid)),
                "weighted_residual_correlation": weighted_correlation(
                    valid[feature], valid["residual_eb"], valid["reliability"]
                ),
                "spearman_residual_correlation": weighted_correlation(
                    valid[feature].rank(method="average"),
                    valid["residual_eb"].rank(method="average"),
                    valid["reliability"],
                ),
                "weighted_target_rate_correlation": weighted_correlation(
                    valid[feature], valid["target_rate"], valid["reliability"]
                ),
            }
        )
    return rows


def _robust_summary(signal: pd.DataFrame) -> pd.DataFrame:
    key_axes = ("late_2023", "early_2024_full_parent", "late_2024_replication_parent")
    local = signal.loc[signal["axis"].isin(key_axes)].copy()
    rows: list[dict[str, object]] = []
    for (domain, feature), group in local.groupby(["domain", "feature"], observed=True):
        values = group.set_index("axis")["weighted_residual_correlation"].reindex(key_axes)
        finite = values.dropna()
        if len(finite) != len(key_axes):
            continue
        signs = np.sign(finite.to_numpy(np.float64))
        rows.append(
            {
                "domain": domain,
                "feature": feature,
                "late_2023_correlation": values.loc["late_2023"],
                "early_2024_correlation": values.loc["early_2024_full_parent"],
                "late_2024_correlation": values.loc[
                    "late_2024_replication_parent"
                ],
                "sign_consistency": float(abs(signs.sum()) / len(signs)),
                "minimum_abs_correlation": float(np.abs(finite).min()),
                "median_abs_correlation": float(np.abs(finite).median()),
                "mean_correlation": float(finite.mean()),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values(
        ["sign_consistency", "minimum_abs_correlation", "median_abs_correlation"],
        ascending=False,
    )


def run(
    project: Path,
    alignment_dir: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    pairs, _ = _load_pairs(project, alignment_dir)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=["season", "pitcher_trackman_id", "pitch_type_group", *PHYSICAL_COLUMNS],
        low_memory=False,
    )
    profiles, mapping_audit = build_origin_profiles(
        pairs, trackman, origins=(2023, 2024, 2025)
    )
    axes = _cached_v25_axes(project, raw)
    late23 = axes["selection_late_2023"]
    full24 = axes["outer_full_2024"]
    replication24 = axes["replication_late_2024"]
    cached23 = np.load(current_oof_dir / "selection_late_2023.npz")
    cached24 = np.load(current_oof_dir / "outer_full_2024.npz")
    cached_replication = np.load(current_oof_dir / "replication_late_2024.npz")
    if not np.array_equal(cached23["target"], late23["target"].to_numpy(np.float64)):
        raise ValueError("late-2023 OOF parity failure")
    if not np.array_equal(cached24["target"], full24["target"].to_numpy(np.float64)):
        raise ValueError("full-2024 OOF parity failure")
    if not np.array_equal(
        cached_replication["target"], replication24["target"].to_numpy(np.float64)
    ):
        raise ValueError("late-2024 replication OOF parity failure")

    early_mask = full24["game_month"].le(7).to_numpy()
    late_mask = full24["game_month"].ge(8).to_numpy()
    audit_axes = {
        "late_2023": (
            late23,
            cached23["final_gate_parent"].astype(np.float64),
            2023,
        ),
        "early_2024_full_parent": (
            full24.loc[early_mask].reset_index(drop=True),
            cached24["final_gate_parent"].astype(np.float64)[early_mask],
            2024,
        ),
        "late_2024_full_parent": (
            full24.loc[late_mask].reset_index(drop=True),
            cached24["final_gate_parent"].astype(np.float64)[late_mask],
            2024,
        ),
        "late_2024_replication_parent": (
            replication24,
            cached_replication["final_gate_parent"].astype(np.float64),
            2024,
        ),
    }
    signal_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for axis_name, (frame, parent, origin) in audit_axes.items():
        profile = profiles.loc[profiles["origin"].eq(origin)].drop(columns="origin")
        covered = frame["pitcher_id"].isin(profile["pitcher_id"])
        coverage_rows.append(
            {
                "axis": axis_name,
                "origin": origin,
                "rows": int(len(frame)),
                "profile_row_coverage": float(covered.mean()),
                "profile_pitcher_coverage": float(
                    frame.loc[covered, "pitcher_id"].nunique()
                    / frame["pitcher_id"].nunique()
                ),
            }
        )
        for domain in DOMAINS:
            residual = pitcher_residual_table(frame, parent, domain)
            signal_rows.extend(
                _signal_rows(
                    profile,
                    residual,
                    axis=axis_name,
                    domain=domain,
                )
            )

    signal = pd.DataFrame(signal_rows)
    robust = _robust_summary(signal)
    coverage = pd.DataFrame(coverage_rows)
    profiles.to_csv(output_dir / "origin_physical_profiles.csv", index=False)
    mapping_audit.to_csv(output_dir / "mapping_audit.csv", index=False)
    coverage.to_csv(output_dir / "profile_coverage.csv", index=False)
    signal.to_csv(output_dir / "pitcher_level_signal.csv", index=False)
    robust.to_csv(output_dir / "robust_signal_summary.csv", index=False)

    consistent = robust.loc[
        robust["sign_consistency"].eq(1.0)
        & robust["minimum_abs_correlation"].ge(0.05)
    ]
    result = {
        "protocol": "V67_TRACKMAN_COMMAND_PROXY_CENSUS_ABOVE_1158_V1",
        "test_csv_read": False,
        "profile_contract": "TrackMan season < origin; direct map support>=20 and purity>=0.99",
        "profile_feature_count": int(len(profiles.columns) - 5),
        "coverage": coverage.to_dict(orient="records"),
        "consistent_signal_count": int(len(consistent)),
        "top_consistent_signals": consistent.head(30).to_dict(orient="records"),
        "interpretation_guardrails": [
            "Association is pitcher-level and does not recover intended target.",
            "Raw TrackMan value and repeatability are distinct; pitch-family SD avoids arsenal-mix inflation.",
            "2024 axes are descriptive audits and must not be used for post-hoc recipe tuning.",
        ],
        "row_local_deployment_possible": True,
        "other_test_rows_required": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.current_oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
