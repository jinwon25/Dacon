"""Re-screen legal season-forward OOF prediction banks above v21.

Earlier experiments compared most of these models with v17 or v19.  This
module answers the incremental question that matters now: does a small move
from the reconstructed v21 champion toward any already-computed legal OOF
prediction improve all three forward audit axes?

Oracle columns and current-pitch labels are excluded explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss


AXES = (
    "y2023_to_y2024",
    "y2023_early_to_late",
    "y2024_early_to_late",
)
WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)
FORBIDDEN_TOKENS = ("oracle", "type_label", "mode_label", "raw_by_type", "raw_by_mode")


def _year_and_mask(axis: str, game_month: np.ndarray) -> tuple[int, np.ndarray]:
    if axis == "y2023_to_y2024":
        return 2024, np.ones(len(game_month), dtype=bool)
    if axis == "y2023_early_to_late":
        return 2023, game_month >= 8
    if axis == "y2024_early_to_late":
        return 2024, game_month >= 8
    raise ValueError(axis)


def _safe_name(prefix: str, name: str) -> str:
    return f"{prefix}::{name}"


def _prediction_bank(project: Path, year: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}

    mode_path = (
        project
        / "artifacts"
        / "latent_failure_mode_state_20260816_01"
        / f"latent_failure_mode_o{year}.npz"
    )
    with np.load(mode_path, allow_pickle=True) as saved:
        names = [str(value) for value in saved["names"].tolist()]
        for index, name in enumerate(names):
            if not any(token in name.lower() for token in FORBIDDEN_TOKENS):
                output[_safe_name("failure_mode", name)] = saved["raw"][
                    :, index
                ].astype(np.float64)

    pitch_path = (
        project
        / "artifacts"
        / "latent_pitch_type_state_20260816_01"
        / f"latent_pitch_type_o{year}.npz"
    )
    with np.load(pitch_path, allow_pickle=True) as saved:
        output["latent_pitch_type::student_raw"] = saved["student_raw"].astype(
            np.float64
        )

    variants_path = (
        project
        / "artifacts"
        / "multi_year_state_variants_20260816_01"
        / f"state_variants_o{year}.npz"
    )
    with np.load(variants_path, allow_pickle=True) as saved:
        for name in saved.files:
            if name.startswith(("global_", "domain_")):
                output[_safe_name("state_variant", name)] = saved[name].astype(
                    np.float64
                )

    selected_path = (
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / f"selected_state_o{year}.npz"
    )
    with np.load(selected_path, allow_pickle=True) as saved:
        output["state_selected::raw"] = saved["raw"].astype(np.float64)

    profile_path = (
        project
        / "artifacts"
        / "pitch_type_profile_20260816_01"
        / f"pitch_type_profile_o{year}.npz"
    )
    with np.load(profile_path, allow_pickle=True) as saved:
        names = [str(value) for value in saved["names"].tolist()]
        for index, name in enumerate(names):
            if not any(token in name.lower() for token in FORBIDDEN_TOKENS):
                output[_safe_name("pitch_profile", name)] = saved["raw"][
                    :, index
                ].astype(np.float64)

    exact_path = (
        project
        / "artifacts"
        / "v14_exact_model_screen_20260815_01"
        / f"exact_model_screen_o{year}.npz"
    )
    with np.load(exact_path, allow_pickle=True) as saved:
        for name in (
            "exact_lgb",
            "exact_ridge",
            "trend_lgb",
            "binary_seed_ensemble",
            "l2_leaves7",
            "l2_leaves15",
            "residual_l2_leaves7",
            "residual_l2_leaves15",
        ):
            output[_safe_name("exact", name)] = saved[name].astype(np.float64)

    recent_path = (
        project
        / "artifacts"
        / "recent_shared_exact_asof_20260815_02"
        / f"recent_shared_o{year}.npz"
    )
    with np.load(recent_path, allow_pickle=True) as saved:
        for name in ("prediction", "exact_lgb", "exact_ridge", "trend_lgb"):
            output[_safe_name("recent_exact", name)] = saved[name].astype(
                np.float64
            )
    return output


def _delta_bank(project: Path, year: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    pfd_path = (
        project
        / "artifacts"
        / "trackman_distillation_20260816_02"
        / f"distillation_o{year}.npz"
    )
    with np.load(pfd_path, allow_pickle=True) as saved:
        for name in (
            "student_with_ids",
            "student_without_ids",
            "pfd_softlabel_l025",
            "pfd_softlabel_l050",
            "pfd_softlabel_l075",
        ):
            output[_safe_name("pfd_delta", name)] = saved[name].astype(np.float64)
    return output


def _model_residual_bank(
    project: Path, axis: str, audit_rows: int
) -> dict[str, np.ndarray]:
    path = project / "artifacts" / "v20_model_residual_20260816_01" / "predictions.npz"
    with np.load(path, allow_pickle=True) as saved:
        output = {}
        for name in ("context_l3", "context_l7", "state_l3", "state_l7", "full_l3", "full_l7"):
            value = saved[f"{axis}__{name}"].astype(np.float64)
            if len(value) != audit_rows:
                raise ValueError(f"model residual row mismatch: {axis} {name}")
            output[_safe_name("residual_delta", name)] = value
        return output


def run(project: Path, champion_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    champion_dir = (project / champion_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_by_year = {
        year: _prediction_bank(project, year) for year in (2023, 2024)
    }
    delta_by_year = {year: _delta_bank(project, year) for year in (2023, 2024)}
    rows = []
    for axis in AXES:
        with np.load(champion_dir / f"{axis}.npz", allow_pickle=True) as saved:
            target = saved["target"].astype(np.float64)
            v21 = saved["v21"].astype(np.float64)
            audit_month = saved["game_month"].astype(np.int16)
        year, full_mask = _year_and_mask(axis, np.asarray(
            np.load(
                project
                / "artifacts"
                / "state_mode_joint_20260816_02"
                / f"joint_candidate_o{2024 if axis != 'y2023_early_to_late' else 2023}.npz",
                allow_pickle=True,
            )["game_month"]
        ))
        if not np.array_equal(audit_month, np.asarray(
            np.load(
                project
                / "artifacts"
                / "state_mode_joint_20260816_02"
                / f"joint_candidate_o{year}.npz",
                allow_pickle=True,
            )["game_month"]
        )[full_mask]):
            raise ValueError(f"audit mask mismatch for {axis}")

        banks: list[tuple[str, dict[str, np.ndarray]]] = [
            ("prediction", prediction_by_year[year]),
            ("delta", delta_by_year[year]),
            ("delta", _model_residual_bank(project, axis, len(target))),
        ]
        for kind, bank in banks:
            for name, full_value in bank.items():
                value = full_value if len(full_value) == len(target) else full_value[full_mask]
                if len(value) != len(target):
                    raise ValueError(f"bank row mismatch for {axis}: {name}")
                for weight in WEIGHTS:
                    if kind == "prediction":
                        candidate = v21 + weight * (value - v21)
                    else:
                        candidate = v21 + weight * value
                    candidate = np.clip(candidate, 0.001, 0.999)
                    rows.append(
                        {
                            "axis": axis,
                            "signal": name,
                            "kind": kind,
                            "weight": weight,
                            "gain": _bss(target, candidate) - _bss(target, v21),
                            "mean_shift": float(np.mean(candidate - v21)),
                            "mean_abs_shift": float(np.mean(np.abs(candidate - v21))),
                        }
                    )
    metrics = pd.DataFrame(rows)
    robust = (
        metrics.groupby(["signal", "kind", "weight"], as_index=False)
        .agg(
            min_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            max_gain=("gain", "max"),
        )
        .sort_values(["min_gain", "mean_gain"], ascending=False)
        .reset_index(drop=True)
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "V22_LEGAL_OOF_BANK_ABOVE_V21_V1",
        "axes": list(AXES),
        "signals": int(metrics["signal"].nunique()),
        "candidate_count": int(len(robust)),
        "strictly_positive": int((robust["min_gain"] > 0.0).sum()),
        "best": robust.head(40).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--champion-dir",
        type=Path,
        default=Path("artifacts/champion_oof_20260817_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v22_oof_bank_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.champion_dir, args.output_dir)


if __name__ == "__main__":
    main()
