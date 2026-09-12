"""Nested v22 audit of hierarchical failure-mode profile predictions.

Candidate family and blend strength are selected only on late 2023. The
selected recipe is then evaluated once on full 2024 and late 2024. All
features behind the raw profiles are frozen before the audit season and every
prediction remains row-local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss
from src.champion.v24_semantic_signal_screen import _v22


DOMAINS = ("R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)
DIRECTIONS = ("blend_to_raw", "add_v21_residual")


def compose(
    v21: np.ndarray,
    v22: np.ndarray,
    raw: np.ndarray,
    weight: float,
    direction: str,
) -> np.ndarray:
    if direction == "blend_to_raw":
        delta = raw - v22
    elif direction == "add_v21_residual":
        delta = raw - v21
    else:
        raise ValueError(f"unknown direction: {direction}")
    return np.clip(v22 + float(weight) * delta, 0.001, 0.999)


def diagnostics(
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    month: np.ndarray,
    domain: np.ndarray,
) -> dict[str, object]:
    month_gains = {}
    for value in sorted(np.unique(month)):
        mask = month == value
        month_gains[str(int(value))] = float(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )
    domain_gains = {}
    for value in DOMAINS:
        mask = domain == value
        domain_gains[value] = float(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )
    gain = float(_bss(target, candidate) - _bss(target, parent))
    return {
        "gain": gain,
        "positive_month_fraction": float(
            np.mean(np.asarray(list(month_gains.values())) > 0.0)
        ),
        "worst_month_gain": float(min(month_gains.values())),
        "minimum_domain_gain": float(min(domain_gains.values())),
        "month_gains": month_gains,
        "domain_gains": domain_gains,
        "mean_shift": float(np.mean(candidate - parent)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
    }


def _load_profiles(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        return (
            saved["target"].astype(np.float64),
            saved["game_month"].astype(np.int16),
            [str(value) for value in saved["names"].tolist()],
            saved["raw"].astype(np.float64),
        )


def _champion_axis(project: Path, axis: str) -> dict[str, np.ndarray]:
    with np.load(
        project / "artifacts/champion_oof_20260817_01" / f"{axis}.npz",
        allow_pickle=True,
    ) as saved:
        return {key: saved[key] for key in saved.files}


def run(project: Path, profile_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    profile_dir = (project / profile_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project / "data/train.csv", low_memory=False)

    target23, month23, names23, raw23 = _load_profiles(
        profile_dir / "failure_mode_profiles_o2023.npz"
    )
    target24, month24, names24, raw24 = _load_profiles(
        profile_dir / "failure_mode_profiles_o2024.npz"
    )
    if names23 != names24:
        raise ValueError("profile names differ between audit years")

    selection_axis = _champion_axis(project, "y2023_early_to_late")
    selection_mask = month23 >= 8
    selection_rows = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    selection_target = target23[selection_mask]
    selection_v21 = selection_axis["v21"].astype(np.float64)
    selection_domain = selection_axis["domain3"].astype(str)
    selection_month = month23[selection_mask]
    selection_v22 = _v22(selection_rows, selection_v21, selection_domain)
    if not np.array_equal(selection_target, selection_axis["target"].astype(np.float64)):
        raise ValueError("2023 target alignment failed")

    rows: list[dict[str, object]] = []
    for index, name in enumerate(names23):
        raw = raw23[selection_mask, index]
        for direction in DIRECTIONS:
            for weight in WEIGHTS:
                candidate = compose(
                    selection_v21, selection_v22, raw, weight, direction
                )
                result = diagnostics(
                    selection_target,
                    selection_v22,
                    candidate,
                    selection_month,
                    selection_domain,
                )
                rows.append(
                    {
                        "candidate": name,
                        "direction": direction,
                        "weight": weight,
                        "gain": result["gain"],
                        "positive_month_fraction": result["positive_month_fraction"],
                        "worst_month_gain": result["worst_month_gain"],
                        "minimum_domain_gain": result["minimum_domain_gain"],
                        "selection_score": min(
                            result["gain"],
                            result["worst_month_gain"],
                            result["minimum_domain_gain"],
                        ),
                    }
                )
    table = pd.DataFrame(rows).sort_values(
        ["selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    table.to_csv(output_dir / "selection.csv", index=False)
    selected = table.iloc[0].to_dict()
    selection_gate = bool(
        selected["gain"] > 0.0
        and selected["positive_month_fraction"] >= 1.0
        and selected["worst_month_gain"] > 0.0
        and selected["minimum_domain_gain"] > 0.0
    )

    summary: dict[str, object] = {
        "protocol": "V25_FAILURE_PROFILE_NESTED_V22_V1",
        "selection_axis": "late 2023 only",
        "candidate_count": int(len(table)),
        "selected": selected,
        "selection_gate_passed": selection_gate,
        "outer_audit_run": False,
        "eligible_for_packaging": False,
    }
    if selection_gate:
        outer_axis = _champion_axis(project, "y2023_to_y2024")
        outer_rows = train.loc[train["season"].eq(2024)].reset_index(drop=True)
        outer_v21 = outer_axis["v21"].astype(np.float64)
        outer_domain = outer_axis["domain3"].astype(str)
        outer_v22 = _v22(outer_rows, outer_v21, outer_domain)
        if not np.array_equal(target24, outer_axis["target"].astype(np.float64)):
            raise ValueError("2024 target alignment failed")
        raw_index = names24.index(str(selected["candidate"]))
        outer_candidate = compose(
            outer_v21,
            outer_v22,
            raw24[:, raw_index],
            float(selected["weight"]),
            str(selected["direction"]),
        )
        outer = diagnostics(
            target24, outer_v22, outer_candidate, month24, outer_domain
        )
        late_mask = month24 >= 8
        late = diagnostics(
            target24[late_mask],
            outer_v22[late_mask],
            outer_candidate[late_mask],
            month24[late_mask],
            outer_domain[late_mask],
        )
        eligible = bool(
            outer["gain"] >= 5.0
            and outer["positive_month_fraction"] >= 0.75
            and outer["minimum_domain_gain"] > 0.0
            and outer["worst_month_gain"] > -10.0
            and late["gain"] > 0.0
            and late["minimum_domain_gain"] > 0.0
        )
        summary.update(
            {
                "outer_audit_run": True,
                "outer_2024": outer,
                "late_2024": late,
                "eligible_for_packaging": eligible,
            }
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=target24,
            v21=outer_v21,
            v22=outer_v22,
            raw=raw24[:, raw_index],
            candidate=outer_candidate,
            game_month=month24,
            domain3=outer_domain,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("artifacts/v22_failure_mode_profiles_20260817_02"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v25_failure_profile_nested_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.profile_dir, args.output_dir)


if __name__ == "__main__":
    main()
