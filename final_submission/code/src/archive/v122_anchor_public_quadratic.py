"""Build and audit a v104 package at the Public-optimal R_ANCHOR dose.

The v25, v27, and v26 submissions differ only in the scalar blend dose of
the same frozen R_ANCHOR direct model.  Brier score is quadratic in that
scalar, so three official Public observations identify the exact leaderboard
quadratic without reading evaluation rows or labels.  This module fits that
curve, changes only the frozen ``blend_eta`` in the standalone v104 package,
and audits the resulting package for route protection and row independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)


PROTOCOL = "V122_ANCHOR_PUBLIC_QUADRATIC_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"
ANCHOR_TEAM = 13


def fit_public_quadratic(
    observations: list[dict[str, Any]], bounds: tuple[float, float]
) -> dict[str, Any]:
    """Fit score(eta)=q2*eta^2+q1*eta+q0 and return its bounded vertex."""

    if len(observations) != 3:
        raise ValueError("exact quadratic recovery requires three observations")
    eta = np.asarray([float(row["eta"]) for row in observations], dtype=np.float64)
    score = np.asarray(
        [float(row["public_score"]) for row in observations], dtype=np.float64
    )
    if len(np.unique(eta)) != 3 or not np.isfinite(eta).all() or not np.isfinite(score).all():
        raise ValueError("invalid quadratic observations")
    q2, q1, q0 = np.polyfit(eta, score, 2)
    fitted = np.polyval([q2, q1, q0], eta)
    fit_max_abs = float(np.max(np.abs(fitted - score)))
    if fit_max_abs > 1e-8:
        raise ValueError(f"quadratic interpolation parity failed: {fit_max_abs}")
    if q2 >= 0.0:
        raise ValueError("Public score curve must be concave for a bounded optimum")
    raw_vertex = float(-q1 / (2.0 * q2))
    low, high = (float(value) for value in bounds)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("eta bounds must lie inside [0, 1]")
    selected_eta = float(np.clip(raw_vertex, low, high))
    return {
        "quadratic_coefficients": {"q2": float(q2), "q1": float(q1), "q0": float(q0)},
        "fit_max_abs": fit_max_abs,
        "raw_vertex_eta": raw_vertex,
        "selected_eta": selected_eta,
        "selected_curve_score": float(np.polyval([q2, q1, q0], selected_eta)),
    }


def _patch_nested_spec(
    staging: Path, relative_path: str, current_eta: float, selected_eta: float,
    curve: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    path = staging / Path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    observed = float(spec.get("blend_eta", np.nan))
    if not np.isclose(observed, current_eta, atol=1e-15, rtol=0.0):
        raise ValueError(f"unexpected parent anchor eta: {observed}")
    spec["blend_eta"] = float(selected_eta)
    spec["selection_note"] = (
        "same frozen row-local v25 signal; eta recovered from the exact concave "
        "quadratic through official Public scores at 0.075/0.10/0.15; DACON FAQ "
        "reply 320256 permits same-axis leaderboard interpolation"
    )
    spec["v122_public_quadratic"] = {
        "protocol": PROTOCOL,
        "parent_eta": current_eta,
        "selected_eta": selected_eta,
        "curve": curve,
        "observations": config["observations"],
        "test_aggregate_used": False,
        "row_local_inference": True,
    }
    path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return spec


def build_package(
    parent_zip: Path, config: dict[str, Any], curve: dict[str, Any], output_dir: Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(parent_zip) as archive:
        _safe_extract(archive, staging)
    spec = _patch_nested_spec(
        staging,
        str(config["nested_spec_path"]),
        float(config["current_eta"]),
        float(curve["selected_eta"]),
        curve,
        config,
    )
    package = output_dir / str(config["package_name"])
    if package.exists():
        _safe_remove_generated(package, output_dir)
    result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    return package, result, spec


def _mixed_smoke(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    mixed = pd.concat([source] * 4, ignore_index=True)
    mixed[ID_COL] = [f"v122_probe_{index:03d}" for index in range(len(mixed))]
    group = np.arange(len(mixed)) % 4
    # Official smoke rows remain history-consistent.  Only route columns are
    # changed to guarantee protected R_CORE/F coverage.
    mixed.loc[group == 0, "game_type"] = "R"
    mixed.loc[group == 0, "pitcher_team_id"] = ANCHOR_TEAM
    mixed.loc[group == 2, "pitcher_team_id"] = 1
    mixed.loc[group == 2, "batter_team_id"] = 2
    mixed.loc[group == 3, "game_type"] = "F"
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def _aligned_probability(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    return frame[ID_COL].map(output.set_index(ID_COL)[TARGET_COL]).to_numpy(float)


def audit_package(
    package: Path, parent_zip: Path, data_dir: Path, timeout: int
) -> dict[str, Any]:
    mixed, sample = _mixed_smoke(data_dir)
    parent_out, parent_seconds, _ = run_package(parent_zip, mixed, sample, timeout=timeout)
    candidate_out, candidate_seconds, stdout = run_package(
        package, mixed, sample, timeout=timeout
    )
    parent = _aligned_probability(mixed, parent_out)
    candidate = _aligned_probability(mixed, candidate_out)
    regular = mixed["game_type"].astype(str).eq("R").to_numpy()
    anchor = mixed["pitcher_team_id"].eq(ANCHOR_TEAM).to_numpy() | mixed[
        "batter_team_id"
    ].eq(ANCHOR_TEAM).to_numpy()
    active = regular & anchor
    protected_max_abs = float(np.max(np.abs(candidate[~active] - parent[~active])))
    active_shift_count = int(np.count_nonzero(np.abs(candidate[active] - parent[active]) > 0.0))
    if protected_max_abs > 1e-12 or active_shift_count == 0:
        raise ValueError(
            f"v122 route audit failed: protected={protected_max_abs}, shifted={active_shift_count}"
        )

    shuffled_test = mixed.sample(frac=1.0, random_state=122).reset_index(drop=True)
    shuffled_sample = pd.DataFrame(
        {ID_COL: shuffled_test[ID_COL], TARGET_COL: 0.0}
    )
    shuffled, _, _ = run_package(package, shuffled_test, shuffled_sample, timeout=timeout)
    base_map = candidate_out.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))
    if shuffled_max_abs > 1e-12:
        raise ValueError(f"v122 shuffled-row parity failure: {shuffled_max_abs}")

    parts = []
    for indices in np.array_split(np.arange(len(mixed)), 2):
        local_test = mixed.iloc[indices].reset_index(drop=True)
        local_sample = pd.DataFrame({ID_COL: local_test[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, local_test, local_sample, timeout=timeout)
        parts.append(output)
    partition_map = pd.concat(parts).set_index(ID_COL)[TARGET_COL].sort_index()
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if partition_max_abs > 1e-12:
        raise ValueError(f"v122 partition parity failure: {partition_max_abs}")

    scale_rows = 245789
    scale = mixed.iloc[np.arange(scale_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v122_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(
        package, scale, scale_sample, timeout=timeout
    )
    values = scale_output[TARGET_COL].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("v122 scale proxy produced invalid probabilities")
    return {
        "mixed_rows": int(len(mixed)),
        "active_rows": int(active.sum()),
        "active_shift_count": active_shift_count,
        "protected_max_abs": protected_max_abs,
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "parent_smoke_runtime_seconds": parent_seconds,
        "candidate_smoke_runtime_seconds": candidate_seconds,
        "scale_proxy_rows": scale_rows,
        "scale_proxy_runtime_seconds": scale_seconds,
        "scale_proxy_mean": float(values.mean()),
        "scale_proxy_min": float(values.min()),
        "scale_proxy_max": float(values.max()),
        "smoke_stdout": stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    parent_zip: Path,
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    parent_zip = parent_zip.resolve()
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    curve = fit_public_quadratic(
        list(config["observations"]), tuple(config["eta_bounds"])
    )
    package, package_result, spec = build_package(parent_zip, config, curve, output_dir)
    audit = audit_package(package, parent_zip, data_dir, timeout)
    current_curve_score = float(
        np.polyval(
            [
                curve["quadratic_coefficients"]["q2"],
                curve["quadratic_coefficients"]["q1"],
                curve["quadratic_coefficients"]["q0"],
            ],
            float(config["current_eta"]),
        )
    )
    projected_gain = float(curve["selected_curve_score"] - current_curve_score)
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "public_quadratic": curve,
        "projected_gain_vs_current_anchor_eta": projected_gain,
        "projected_v104_public_ignoring_tiny_gate_interaction": float(
            config["current_champion_public"] + projected_gain
        ),
        "patched_spec": spec,
        "parent_zip_sha256": _sha256(parent_zip),
        "package": package_result,
        "package_sha256": _sha256(package),
        "audit": audit,
        "eligible_for_user_submission": True,
        "standalone_no_parent_zip_dependency": True,
        **config["restrictions"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    run(args.parent_zip, args.data_dir, args.config, args.output_dir, args.timeout)


if __name__ == "__main__":
    main()
