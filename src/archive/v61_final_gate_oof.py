"""Reconstruct temporal OOF predictions for the final 0819 TrackMan gate.

The public 1158.0746 release applies a tiny R_ANCHOR shrink after the eta=0.15
v25 direct model.  Earlier follow-up screens only reconstructed the eta=0.15
parent because year-specific TrackMan profiles were not saved beside the final
package.  This module rebuilds those profiles from official train and official
TrackMan history with the already-tested, strictly pre-origin Hungarian
linkage.  It also proves that the 2025 reconstruction matches the standalone
payload before historical OOF is trusted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import read_main, read_trackman
from src.features import hierarchical_prior
from src.trackman_linkage import build_pitcher_profile_table
from src.core.diagnostics import diagnostics
from src.core.axes import _cached_v25_axes
from src.archive.v58_eta15_rebase_audit import CURRENT_ETA, eta_parent


GATE_ETA = 0.03
GATE_CAP = 0.03
PROFILE_KEYS = ["season", "pitcher_id"]
PROFILE_GATE_COLUMNS = ["season", "pitcher_id", "tm_linked", "tm_pitcher_n"]
AXIS_SOURCE_MASKS = {
    "selection_late_2023": lambda frame: frame["season"].lt(2023)
    | (frame["season"].eq(2023) & frame["game_month"].le(7)),
    "outer_full_2024": lambda frame: frame["season"].le(2023),
    "replication_late_2024": lambda frame: frame["season"].lt(2024)
    | (frame["season"].eq(2024) & frame["game_month"].le(7)),
}


def gate_weights(
    frame: pd.DataFrame,
    profile: pd.DataFrame,
) -> np.ndarray:
    """Return the exact row-local 0819 gate weight for each input row."""

    missing = set(PROFILE_GATE_COLUMNS) - set(profile.columns)
    if missing:
        raise ValueError(f"TrackMan profile columns missing: {sorted(missing)}")
    source = frame[["season", "pitcher_id"]].copy()
    source["__row_order"] = np.arange(len(source), dtype=np.int64)
    linked = (
        source.merge(
            profile[PROFILE_GATE_COLUMNS],
            on=PROFILE_KEYS,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        .sort_values("__row_order", kind="stable")
        .reset_index(drop=True)
    )
    tm_linked = (
        pd.to_numeric(linked["tm_linked"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    tm_n = (
        pd.to_numeric(linked["tm_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    tm_conf = np.where(tm_linked > 0.0, tm_n / (tm_n + 500.0), 0.0)
    if "domain3" in frame:
        anchor = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    else:
        regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R")
        anchor = (
            regular
            & (frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13))
        ).to_numpy()
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).to_numpy()
    strikes = (
        pd.to_numeric(frame["strikes_before"], errors="coerce")
        .fillna(-1)
        .to_numpy()
    )
    pressure = np.where((balls == 3) | (strikes == 2), 1.25, 0.75)
    return np.clip(
        np.where(anchor, GATE_ETA * tm_conf * pressure, 0.0),
        0.0,
        GATE_CAP,
    )


def apply_final_gate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    profile: pd.DataFrame,
    global_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact final gate and return prediction plus gate weights."""

    parent = np.asarray(parent, dtype=np.float64)
    if parent.shape != (len(frame),):
        raise ValueError("parent probability length mismatch")
    gate = gate_weights(frame, profile)
    prior = hierarchical_prior(frame, float(global_rate), 200.0, 0.25)
    candidate = parent * (1.0 - gate) + prior * gate
    return np.clip(candidate, 1e-6, 1.0 - 1e-6), gate


def compare_profiles(
    reconstructed: pd.DataFrame,
    frozen: pd.DataFrame,
) -> dict[str, object]:
    """Compare the reconstructed 2025 gate fields with the frozen payload."""

    columns = PROFILE_GATE_COLUMNS
    left = reconstructed[columns].sort_values(PROFILE_KEYS).reset_index(drop=True)
    right = frozen[columns].sort_values(PROFILE_KEYS).reset_index(drop=True)
    keys_equal = left[PROFILE_KEYS].equals(right[PROFILE_KEYS])
    linked_equal = left["tm_linked"].fillna(-1).equals(right["tm_linked"].fillna(-1))
    left_n = pd.to_numeric(left["tm_pitcher_n"], errors="coerce").to_numpy(np.float64)
    right_n = pd.to_numeric(right["tm_pitcher_n"], errors="coerce").to_numpy(np.float64)
    if len(left_n) != len(right_n):
        max_n_diff = float("inf")
    else:
        max_n_diff = float(np.nanmax(np.abs(left_n - right_n)))
    return {
        "reconstructed_rows": int(len(left)),
        "frozen_rows": int(len(right)),
        "keys_equal": bool(keys_equal),
        "tm_linked_equal": bool(linked_equal),
        "tm_pitcher_n_max_abs_diff": max_n_diff,
        "status": "pass"
        if len(left) == len(right)
        and keys_equal
        and linked_equal
        and max_n_diff == 0.0
        else "fail",
    }


def _source_global_rate(raw: pd.DataFrame, axis_name: str) -> float:
    mask = AXIS_SOURCE_MASKS[axis_name](raw)
    source = raw.loc[mask, "control_success"]
    if source.empty:
        raise ValueError(f"empty source for global rate: {axis_name}")
    return float(source.mean())


def run(
    project: Path,
    standalone_payload: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    standalone_payload = standalone_payload.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_main(project / "data" / "train.csv")
    trackman = read_trackman(project / "data" / "trackman_history.csv")
    config = json.loads(
        (project / "configs" / "trackman_linkage.json").read_text(encoding="utf-8")
    )
    profiles, linkage = build_pitcher_profile_table(
        raw,
        trackman,
        origins=[2023, 2024, 2025],
        spec=config["linkage"],
    )
    profiles.to_csv(output_dir / "trackman_profiles_2023_2025.csv", index=False)
    linkage.to_csv(output_dir / "trackman_linkage_2023_2025.csv", index=False)

    frozen_profile = pd.read_csv(
        standalone_payload / "model" / "trackman_pitcher_profiles.csv"
    )
    reconstruction = compare_profiles(
        profiles.loc[profiles["season"].eq(2025)], frozen_profile
    )
    if reconstruction["status"] != "pass":
        raise ValueError(f"2025 TrackMan profile reconstruction failed: {reconstruction}")

    axes = _cached_v25_axes(project, raw)
    audits: dict[str, object] = {}
    for axis_name, frame in axes.items():
        season = int(frame["season"].iloc[0])
        profile = profiles.loc[profiles["season"].eq(season)]
        parent = eta_parent(frame, CURRENT_ETA)
        global_rate = _source_global_rate(raw, axis_name)
        candidate, gate = apply_final_gate(
            frame, parent, profile, global_rate
        )
        active = gate > 0.0
        result = diagnostics(frame, parent, candidate, active)
        result.update(
            {
                "source_global_rate": global_rate,
                "profile_origin": season,
                "gate_active_rows": int(active.sum()),
                "gate_active_fraction": float(active.mean()),
                "gate_mean_active": float(gate[active].mean()) if active.any() else 0.0,
                "gate_max": float(gate.max()) if len(gate) else 0.0,
            }
        )
        audits[axis_name] = result
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            eta15_parent=parent,
            final_gate_parent=candidate,
            gate=gate,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    summary = {
        "protocol": "V61_FINAL_0819_TRACKMAN_GATE_TEMPORAL_OOF_V1",
        "public_release": 1158.0745556751,
        "parent": "eta=0.15 R_ANCHOR parent",
        "profile_method": "strictly pre-origin official TrackMan Hungarian linkage",
        "profile_reconstruction_2025": reconstruction,
        "audits": audits,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "test_distribution_used": False,
        "audit_labels_used_for_profile_or_gate": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--standalone-payload", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v61_final_gate_oof_20260822_01"),
    )
    args = parser.parse_args()
    output = args.output_dir
    if not output.is_absolute():
        output = Path.cwd() / output
    run(args.project, args.standalone_payload, output)


if __name__ == "__main__":
    main()
