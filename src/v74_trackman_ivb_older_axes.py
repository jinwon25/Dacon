"""Independent older-axis falsification of the frozen v70 IVB recipe.

The v70 calendar rule was formulated after inspecting 2024, so 2024 cannot
authorize packaging.  This audit freezes every material choice from v70:

* one feature: historical pitcher mean induced vertical break;
* all domains;
* April through September only;
* Ridge alpha and correction eta selected from source OOF only.

The primary axes predate the reused 2024 audits.  Their parent probabilities
are the legal season-forward incumbents saved by the selected-state pipeline.
This does not claim exact historical reconstruction of public 1158; it asks
the narrower causal question whether the IVB residual direction transfers
above a strong contemporaneous OOF parent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v66_direct_trackman_gate import _load_pairs
from src.v67_trackman_command_proxy_census import (
    PHYSICAL_COLUMNS,
    build_origin_profiles,
)
from src.v70_trackman_ivb_calendar_gate import calendar_transition


PRIMARY_AXES = ("early22_to_late22", "full22_to_full23")


def _state_frame(raw: pd.DataFrame, state_dir: Path, year: int) -> tuple[pd.DataFrame, np.ndarray]:
    year_frame = raw.loc[raw["season"].eq(int(year))].reset_index(drop=True)
    with np.load(state_dir / f"selected_state_o{year}.npz", allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        parent = saved["incumbent"].astype(np.float64)
        domain = saved["domain3"].astype(str)
        month = saved["game_month"].astype(np.int16)
        pitcher = saved["pitcher_id"].astype(np.int64)
    expected_target = year_frame["control_success"].to_numpy(np.float64)
    expected_month = year_frame["game_month"].to_numpy(np.int16)
    expected_pitcher = year_frame["pitcher_id"].to_numpy(np.int64)
    if not np.array_equal(target, expected_target):
        raise ValueError(f"target parity failure for {year}")
    if not np.array_equal(month, expected_month):
        raise ValueError(f"month parity failure for {year}")
    if not np.array_equal(pitcher, expected_pitcher):
        raise ValueError(f"pitcher parity failure for {year}")
    frame = year_frame[["season", "game_month", "pitcher_id"]].copy()
    frame["target"] = target
    frame["domain3"] = domain
    return frame, parent


def independent_gate(audits: dict[str, dict[str, object]]) -> dict[str, bool]:
    primary = [audits[name] for name in PRIMARY_AXES]
    return {
        "both_older_gains_positive": bool(all(float(item["gain"]) > 0.0 for item in primary)),
        "all_active_months_positive": bool(
            all(float(item["positive_month_fraction"]) == 1.0 for item in primary)
        ),
        "all_domains_nonnegative": bool(
            all(float(item["minimum_domain_gain"]) >= 0.0 for item in primary)
        ),
        "source_oof_selected_nonzero_eta": bool(
            all(float(item["source_oof_eta"]) > 0.0 for item in primary)
        ),
    }


def run(
    project: Path,
    alignment_dir: Path,
    state_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    state_dir = state_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    frames: dict[int, pd.DataFrame] = {}
    parents: dict[int, np.ndarray] = {}
    for year in (2022, 2023, 2024):
        frames[year], parents[year] = _state_frame(raw, state_dir, year)

    pairs, _ = _load_pairs(project, alignment_dir)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=["season", "pitcher_trackman_id", "pitch_type_group", *PHYSICAL_COLUMNS],
        low_memory=False,
    )
    profiles, mapping_audit = build_origin_profiles(
        pairs, trackman, origins=(2022, 2023, 2024)
    )
    profile = {
        year: profiles.loc[profiles["origin"].eq(year)].drop(columns="origin")
        for year in (2022, 2023, 2024)
    }

    early22_mask = frames[2022]["game_month"].le(7).to_numpy()
    late22_mask = frames[2022]["game_month"].ge(8).to_numpy()
    axes = {
        "early22_to_late22": (
            frames[2022].loc[early22_mask].reset_index(drop=True),
            parents[2022][early22_mask],
            profile[2022],
            frames[2022].loc[late22_mask].reset_index(drop=True),
            parents[2022][late22_mask],
            profile[2022],
        ),
        "full22_to_full23": (
            frames[2022],
            parents[2022],
            profile[2022],
            frames[2023],
            parents[2023],
            profile[2023],
        ),
        "full23_to_full24_confirmation": (
            frames[2023],
            parents[2023],
            profile[2023],
            frames[2024],
            parents[2024],
            profile[2024],
        ),
    }
    audits: dict[str, dict[str, object]] = {}
    prediction_payload: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for name, values in axes.items():
        result, candidate = calendar_transition(*values)
        audits[name] = result
        prediction_payload[name] = candidate
        rows.append(
            {
                "axis": name,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"months", "domain_gains"}
                },
            }
        )

    gates = independent_gate(audits)
    passes = bool(all(gates.values()))
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    mapping_audit.to_csv(output_dir / "mapping_audit.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", **prediction_payload)
    result = {
        "protocol": "V74_FROZEN_V70_IVB_OLDER_AXES_V1",
        "recipe_frozen_before_primary_audits": True,
        "recipe": {
            "feature": "tm_induced_vert_break_mean",
            "domain": "ALL",
            "active_months": [4, 5, 6, 7, 8, 9],
            "ridge_and_eta": "source-only OOF",
        },
        "parent_contract": (
            "contemporaneous legal season-forward incumbent OOF; mechanism audit, "
            "not exact public-1158 historical reconstruction"
        ),
        "primary_axes": list(PRIMARY_AXES),
        "audits": audits,
        "gates": gates,
        "passes_independent_older_axes": passes,
        "eligible_for_packaging": False,
        "decision": "advance_to_exact-parent_recheck" if passes else "reject_v70",
        "reason_not_packaged": (
            "older axes test transfer of the direction above historical parents; "
            "a standalone candidate still requires exact current-parent reconstruction"
        ),
        "test_csv_read": False,
        "row_local_inference": True,
        "other_test_rows_used": False,
        "test_distribution_used": False,
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
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.state_dir, args.output_dir)


if __name__ == "__main__":
    main()
