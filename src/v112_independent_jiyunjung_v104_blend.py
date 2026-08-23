"""Audit a strict independent model as a frozen probability blend over v104.

The challenger OOF is trained separately for each target season T using only
seasons before T.  The single blend dose is fitted on full-2022 and late-2023;
full/late-2024 are locked transfer audits.  Public scores and test rows are not
read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics, _robust_axis
from src.v104_source_stability_mask import _point_pass
from src.v110_v104_cross_architecture_rebase import apply_correction, fit_alpha
from src.v77_team_oof_constrained_blend import single_candidate_headroom
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V112_INDEPENDENT_JIYUNJUNG_V104_BLEND_V1"
AXES = ("full_2022", "late_2023", "full_2024", "late_2024")
SOURCE_AXES = ("full_2022", "late_2023")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_strict_year(
    strict_oof_dir: Path, raw: pd.DataFrame, year: int
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction_path = strict_oof_dir / f"predictions_{year}.npy"
    target_path = strict_oof_dir / f"targets_{year}.npy"
    metadata_path = strict_oof_dir / f"metadata_{year}.json"
    for path in (prediction_path, target_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
    target = np.load(target_path, allow_pickle=False).astype(np.float64)
    expected = raw.loc[raw["season"].eq(year), "control_success"].to_numpy(np.float64)
    if not np.array_equal(target, expected):
        raise ValueError(f"strict target/order mismatch for {year}")
    if len(prediction) != len(expected) or not np.isfinite(prediction).all():
        raise ValueError(f"invalid strict prediction for {year}")
    if float(prediction.min()) < 0.0 or float(prediction.max()) > 1.0:
        raise ValueError(f"strict prediction outside [0, 1] for {year}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if bool(metadata.get("outer_aggregate_used_for_calibration", True)):
        raise ValueError(f"outer aggregate was used for {year}")
    if str(metadata.get("prediction_sha256", "")).upper() != _sha256(prediction_path):
        raise ValueError(f"strict prediction hash mismatch for {year}")
    if str(metadata.get("target_sha256", "")).upper() != _sha256(target_path):
        raise ValueError(f"strict target hash mismatch for {year}")
    return prediction, metadata


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    strict_oof_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(train_csv, low_memory=False)
    exact = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    late24_mask = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = _slice_axis(exact["full_2024"], late24_mask)
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    frames["late_2024"] = frames["full_2024"].loc[late24_mask].reset_index(drop=True)

    with np.load(v104_dir / "selected_axes.npz") as saved:
        v104 = {name: saved[name].astype(np.float64) for name in AXES}
    strict_year: dict[int, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for year in (2022, 2023, 2024):
        strict_year[year], provenance[str(year)] = _load_strict_year(
            strict_oof_dir, raw, year
        )
    late23_mask = raw.loc[raw["season"].eq(2023), "game_month"].ge(8).to_numpy()
    strict = {
        "full_2022": strict_year[2022],
        "late_2023": strict_year[2023][late23_mask],
        "full_2024": strict_year[2024],
        "late_2024": strict_year[2024][late24_mask],
    }

    for name in AXES:
        if not (
            len(exact[name]["target"]) == len(v104[name]) == len(strict[name])
        ):
            raise ValueError(f"axis length mismatch: {name}")
        expected_target = frames[name]["control_success"].to_numpy(np.float64)
        if not np.array_equal(exact[name]["target"], expected_target):
            raise ValueError(f"contract/train target mismatch: {name}")

    rebased = {name: {**exact[name], "parent": v104[name]} for name in AXES}
    route_domains = [str(value) for value in config["route_domains"]]
    route_masks = {
        name: np.isin(exact[name]["domain3"].astype(str), route_domains)
        for name in AXES
    }
    corrections = {
        name: np.where(route_masks[name], strict[name] - v104[name], 0.0)
        for name in AXES
    }
    alpha = fit_alpha(
        [exact[name]["target"][exact[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
        [v104[name][exact[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
        [corrections[name][exact[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
        float(config["alpha_cap"]),
    )
    candidate = {
        name: apply_correction(v104[name], corrections[name], alpha) for name in AXES
    }
    metrics = {name: _axis_metrics(rebased[name], candidate[name]) for name in AXES}
    headroom = {
        name: single_candidate_headroom(
            exact[name]["target"][exact[name]["exact_mask"].astype(bool)],
            v104[name][exact[name]["exact_mask"].astype(bool)],
            apply_correction(v104[name], corrections[name], 1.0)[
                exact[name]["exact_mask"].astype(bool)
            ],
        )
        for name in AXES
    }

    gate = config["selection_gate"]
    point_pass = {name: _point_pass(value, gate) for name, value in metrics.items()}
    family_alpha = sorted(set(float(value) for value in config["family_alpha"]) | {alpha})
    family = {
        name: [
            apply_correction(v104[name], corrections[name], value)
            for value in family_alpha
        ]
        for name in AXES
    }
    robust = {
        name: _robust_axis(
            frames[name], rebased[name], candidate[name], family[name], config,
            100 + 10 * index,
        )
        for index, name in enumerate(AXES)
    }
    robust_pass = {
        name: bool(
            all(
                value[key]["p05"] > 0.0
                for key in ("pitcher", "crossed_pitcher_batter", "chronological_block")
            )
            and value["reality_check"]["p_value"] <= float(gate["reality_check_alpha"])
        )
        for name, value in robust.items()
    }
    eligible = bool(all(point_pass.values()) and all(robust_pass.values()))

    np.savez_compressed(output_dir / "strict_axes.npz", **strict)
    np.savez_compressed(output_dir / "selected_axes.npz", **candidate)
    result = {
        "protocol": PROTOCOL,
        "status": "promote" if eligible else "reject",
        "selected_alpha": alpha,
        "route_domains": route_domains,
        "route_fraction": {
            name: float(route_masks[name].mean()) for name in AXES
        },
        "family_alpha": family_alpha,
        "headroom": headroom,
        "metrics": metrics,
        "point_gate_pass": point_pass,
        "robust": robust,
        "robust_gate_pass": robust_pass,
        "eligible_for_packaging": eligible,
        "strict_oof_provenance": provenance,
        **config["restrictions"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--strict-oof-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v104_dir, args.strict_oof_dir,
        args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
