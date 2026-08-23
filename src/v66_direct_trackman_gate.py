"""Audit alignment-derived TrackMan identities in the exact 1158 final gate.

The public 1158 release uses a low-amplitude, row-local R_ANCHOR shrink whose
only TrackMan inputs are whether a pitcher was linked and the amount of prior
TrackMan history.  Its historical linkage is a pitch-mix Hungarian assignment.

This experiment freezes the parent prediction and the complete gate formula.
It changes only the target-free pitcher identity source:

* ``direct_only``: a >=99% pure majority identity from pre-origin aligned rows;
* ``direct_first``: direct identity when available, legacy identity otherwise;
* ``agreement_only``: only identities on which direct and legacy agree.

No current/audit-season TrackMan row, audit target, test row aggregate, or test
distribution is used to construct a profile.  The primary support threshold
is declared as 20 aligned rows; 5 and 100 are reported as sensitivity checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v30_diverse_covariance_screen import _cached_v25_axes, diagnostics
from src.v61_final_gate_oof import (
    _source_global_rate,
    apply_final_gate,
)
from src.v65_trackman_data_census import derive_temporal_entity_map


PRIMARY_SUPPORT = 20
SUPPORT_SENSITIVITY = (5, 20, 100)
MINIMUM_PURITY = 0.99
ROUTES = ("direct_only", "direct_first", "agreement_only")


def build_direct_profiles(
    pairs: pd.DataFrame,
    trackman: pd.DataFrame,
    origins: tuple[int, ...],
    *,
    minimum_support: int = PRIMARY_SUPPORT,
    minimum_purity: float = MINIMUM_PURITY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly pre-origin row-local gate profiles and map diagnostics."""

    profiles: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for origin in origins:
        mapping, audit = derive_temporal_entity_map(
            pairs,
            int(origin),
            minimum_support=int(minimum_support),
            minimum_purity=float(minimum_purity),
        )
        history = trackman.loc[trackman["season"].lt(int(origin))]
        counts = (
            history.groupby("pitcher_trackman_id", observed=True)
            .size()
            .rename("tm_pitcher_n")
            .reset_index()
        )
        if mapping.empty:
            profile = pd.DataFrame(
                columns=["season", "pitcher_id", "pitcher_trackman_id", "tm_linked", "tm_pitcher_n"]
            )
        else:
            profile = mapping[
                ["pitcher_id", "pitcher_trackman_id", "support", "total_support", "purity"]
            ].merge(
                counts,
                on="pitcher_trackman_id",
                how="left",
                validate="many_to_one",
            )
            profile.insert(0, "season", int(origin))
            profile["tm_linked"] = profile["tm_pitcher_n"].notna().astype(np.int8)
        if profile.duplicated(["season", "pitcher_id"]).any():
            raise ValueError(f"duplicate direct profile keys for origin={origin}")
        audits.append(
            {
                **audit,
                "profile_entities": int(len(profile)),
                "profile_trackman_rows": int(profile["tm_pitcher_n"].fillna(0).sum()),
            }
        )
        profiles.append(profile)
    return pd.concat(profiles, ignore_index=True), pd.DataFrame(audits)


def combine_profiles(
    direct: pd.DataFrame,
    legacy: pd.DataFrame,
    route: str,
) -> pd.DataFrame:
    """Combine identity sources without using prediction rows or labels."""

    if route not in ROUTES:
        raise ValueError(f"unknown route: {route}")
    keys = ["season", "pitcher_id"]
    direct_columns = [*keys, "pitcher_trackman_id", "tm_linked", "tm_pitcher_n"]
    legacy_columns = [*keys, "pitcher_trackman_id", "tm_linked", "tm_pitcher_n"]
    joined = direct[direct_columns].merge(
        legacy[legacy_columns],
        on=keys,
        how="outer",
        suffixes=("_direct", "_legacy"),
        validate="one_to_one",
    )
    direct_valid = joined["tm_linked_direct"].fillna(0).eq(1)
    legacy_valid = joined["tm_linked_legacy"].fillna(0).eq(1)
    agreement = (
        direct_valid
        & legacy_valid
        & joined["pitcher_trackman_id_direct"].eq(
            joined["pitcher_trackman_id_legacy"]
        )
    )
    if route == "direct_only":
        keep = direct_valid
        use_direct = direct_valid
    elif route == "direct_first":
        keep = direct_valid | legacy_valid
        use_direct = direct_valid
    else:
        keep = agreement
        use_direct = agreement
    output = joined.loc[keep, keys].copy()
    source = joined.loc[keep]
    output["tm_linked"] = 1
    output["tm_pitcher_n"] = np.where(
        use_direct.loc[keep],
        source["tm_pitcher_n_direct"],
        source["tm_pitcher_n_legacy"],
    )
    output["pitcher_trackman_id"] = np.where(
        use_direct.loc[keep],
        source["pitcher_trackman_id_direct"],
        source["pitcher_trackman_id_legacy"],
    )
    return output.reset_index(drop=True)


def _load_pairs(
    project: Path, alignment_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(
        project / "data" / "train.csv",
        usecols=["season", "pitcher_id"],
        low_memory=False,
    )
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=["season", "pitcher_trackman_id"],
        low_memory=False,
    )
    with np.load(alignment_dir / "pitch_alignment.npz", allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
        season = saved["season"].astype(np.int16)
    if len(main_index) != len(trackman_index):
        raise ValueError("alignment index length mismatch")
    if len(np.unique(main_index)) != len(main_index):
        raise ValueError("main alignment indices are not one-to-one")
    if len(np.unique(trackman_index)) != len(trackman_index):
        raise ValueError("TrackMan alignment indices are not one-to-one")
    pairs = pd.DataFrame(
        {
            "season": season,
            "pitcher_id": main.iloc[main_index]["pitcher_id"].to_numpy(),
            "pitcher_trackman_id": trackman.iloc[trackman_index][
                "pitcher_trackman_id"
            ].to_numpy(),
        }
    )
    if not np.array_equal(
        pairs["season"].to_numpy(np.int16),
        main.iloc[main_index]["season"].to_numpy(np.int16),
    ):
        raise ValueError("main/alignment season mismatch")
    if not np.array_equal(
        pairs["season"].to_numpy(np.int16),
        trackman.iloc[trackman_index]["season"].to_numpy(np.int16),
    ):
        raise ValueError("TrackMan/alignment season mismatch")
    return pairs, trackman


def _profile_route_audit(
    direct: pd.DataFrame, legacy: pd.DataFrame, route_profile: pd.DataFrame
) -> dict[str, object]:
    keys = ["season", "pitcher_id"]
    overlap = direct.merge(legacy, on=keys, suffixes=("_direct", "_legacy"))
    overlap = overlap.loc[
        overlap["tm_linked_direct"].eq(1) & overlap["tm_linked_legacy"].eq(1)
    ]
    agreement = overlap["pitcher_trackman_id_direct"].eq(
        overlap["pitcher_trackman_id_legacy"]
    )
    return {
        "profile_entities": int(len(route_profile)),
        "direct_entities": int(direct["tm_linked"].eq(1).sum()),
        "legacy_entities": int(legacy["tm_linked"].eq(1).sum()),
        "identity_overlap": int(len(overlap)),
        "identity_agreement": float(agreement.mean()) if len(overlap) else None,
    }


def attach_legacy_id(
    legacy_profile: pd.DataFrame, legacy_linkage: pd.DataFrame
) -> pd.DataFrame:
    """Restore the identity column omitted from the compact gate profile."""

    keys = ["season", "pitcher_id"]
    required_profile = {*keys, "tm_linked", "tm_pitcher_n"}
    required_linkage = {*keys, "pitcher_trackman_id"}
    if missing := required_profile - set(legacy_profile.columns):
        raise ValueError(f"legacy profile columns missing: {sorted(missing)}")
    if missing := required_linkage - set(legacy_linkage.columns):
        raise ValueError(f"legacy linkage columns missing: {sorted(missing)}")
    if legacy_linkage.duplicated(keys).any():
        raise ValueError("duplicate legacy linkage keys")
    output = legacy_profile.merge(
        legacy_linkage[[*keys, "pitcher_trackman_id"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    linked_without_id = output["tm_linked"].eq(1) & output[
        "pitcher_trackman_id"
    ].isna()
    if linked_without_id.any():
        raise ValueError("linked legacy profile is missing its TrackMan identity")
    return output


def run(
    project: Path,
    alignment_dir: Path,
    legacy_profiles_path: Path,
    current_oof_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    legacy_profiles_path = legacy_profiles_path.resolve()
    current_oof_dir = current_oof_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    pairs, trackman = _load_pairs(project, alignment_dir)
    legacy = pd.read_csv(legacy_profiles_path, low_memory=False)
    if "pitcher_trackman_id" not in legacy.columns:
        linkage_path = legacy_profiles_path.with_name("trackman_linkage_2023_2025.csv")
        if not linkage_path.exists():
            raise FileNotFoundError(
                "compact legacy profile needs sibling trackman_linkage_2023_2025.csv"
            )
        legacy = attach_legacy_id(
            legacy, pd.read_csv(linkage_path, low_memory=False)
        )
    origins = (2023, 2024, 2025)
    axes = _cached_v25_axes(project, raw)

    metric_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    primary_profiles: dict[str, pd.DataFrame] = {}
    for support in SUPPORT_SENSITIVITY:
        direct, map_audit = build_direct_profiles(
            pairs,
            trackman,
            origins,
            minimum_support=support,
            minimum_purity=MINIMUM_PURITY,
        )
        for route in ROUTES:
            routed = combine_profiles(direct, legacy, route)
            if support == PRIMARY_SUPPORT:
                primary_profiles[route] = routed
            for origin in origins:
                route_origin = routed.loc[routed["season"].eq(origin)]
                direct_origin = direct.loc[direct["season"].eq(origin)]
                legacy_origin = legacy.loc[legacy["season"].eq(origin)]
                mapping_rows.append(
                    {
                        "minimum_support": support,
                        "origin": origin,
                        "route": route,
                        **_profile_route_audit(
                            direct_origin, legacy_origin, route_origin
                        ),
                    }
                )
            for axis_name, frame in axes.items():
                season = int(frame["season"].iloc[0])
                cached = np.load(current_oof_dir / f"{axis_name}.npz")
                target = cached["target"].astype(np.float64)
                eta_parent = cached["eta15_parent"].astype(np.float64)
                champion = cached["final_gate_parent"].astype(np.float64)
                if not np.array_equal(target, frame["target"].to_numpy(np.float64)):
                    raise ValueError(f"target parity failure: {axis_name}")
                candidate, gate = apply_final_gate(
                    frame,
                    eta_parent,
                    routed.loc[routed["season"].eq(season)],
                    _source_global_rate(raw, axis_name),
                )
                result = diagnostics(frame, champion, candidate, gate > 0)
                base_result = diagnostics(frame, eta_parent, candidate, gate > 0)
                metric_rows.append(
                    {
                        "minimum_support": support,
                        "route": route,
                        "axis": axis_name,
                        "gain_vs_champion": result["gain"],
                        "gain_vs_eta15": base_result["gain"],
                        "positive_month_fraction_vs_champion": result[
                            "positive_month_fraction"
                        ],
                        "worst_month_gain_vs_champion": result["worst_month_gain"],
                        "minimum_domain_gain_vs_champion": result[
                            "minimum_domain_gain"
                        ],
                        "active_rows": int(np.count_nonzero(gate > 0)),
                        "active_fraction": float(np.mean(gate > 0)),
                        "mean_active_gate": float(gate[gate > 0].mean())
                        if np.any(gate > 0)
                        else 0.0,
                    }
                )
        map_audit.assign(minimum_support=support).to_csv(
            output_dir / f"direct_mapping_support{support}.csv", index=False
        )

    metrics = pd.DataFrame(metric_rows)
    mappings = pd.DataFrame(mapping_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    mappings.to_csv(output_dir / "profile_routes.csv", index=False)
    for route, profile in primary_profiles.items():
        profile.to_csv(output_dir / f"profile_{route}_support20.csv", index=False)

    wide = metrics.pivot_table(
        index=["minimum_support", "route"],
        columns="axis",
        values="gain_vs_champion",
    ).reset_index()
    audit_axes = ["outer_full_2024", "replication_late_2024"]
    wide["minimum_2024_gain"] = wide[audit_axes].min(axis=1)
    wide["mean_2024_gain"] = wide[audit_axes].mean(axis=1)
    wide = wide.sort_values(
        ["minimum_2024_gain", "mean_2024_gain"], ascending=False
    )
    wide.to_csv(output_dir / "robust_summary.csv", index=False)
    primary = metrics.loc[
        metrics["minimum_support"].eq(PRIMARY_SUPPORT)
        & metrics["route"].eq("direct_only")
    ]
    primary_by_axis = {
        row["axis"]: {
            key: row[key]
            for key in (
                "gain_vs_champion",
                "gain_vs_eta15",
                "positive_month_fraction_vs_champion",
                "worst_month_gain_vs_champion",
                "minimum_domain_gain_vs_champion",
                "active_rows",
                "active_fraction",
                "mean_active_gate",
            )
        }
        for _, row in primary.iterrows()
    }
    eligible = bool(
        len(primary) == len(axes)
        and (primary["gain_vs_champion"] > 0).all()
        and (
            primary["positive_month_fraction_vs_champion"] >= 0.5
        ).all()
        and (primary["worst_month_gain_vs_champion"] > -0.5).all()
    )
    result = {
        "protocol": "V66_ALIGNMENT_DERIVED_TRACKMAN_GATE_ABOVE_1158_V1",
        "frozen_gate_formula": True,
        "primary_recipe": {
            "route": "direct_only",
            "minimum_support": PRIMARY_SUPPORT,
            "minimum_purity": MINIMUM_PURITY,
        },
        "primary_audits": primary_by_axis,
        "eligible_for_packaging": eligible,
        "decision": "promote" if eligible else "reject",
        "row_local_inference": True,
        "other_test_rows_used": False,
        "test_distribution_used": False,
        "audit_targets_used_to_construct_profiles": False,
        "sensitivity": wide.to_dict(orient="records"),
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
    parser.add_argument("--legacy-profiles", type=Path, required=True)
    parser.add_argument("--current-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.project,
        args.alignment_dir,
        args.legacy_profiles,
        args.current_oof_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
