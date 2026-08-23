"""Build and audit the conservative v124 multi-axis Public-quadratic stack.

The selected scalar doses are fixed before this builder reads any evaluation
rows.  They combine official sequential Public deltas with strictly-forward
2023-to-2024 OOF Brier curvature.  The official test file is used only for
package execution, row-order/partition parity, and runtime checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.package import verify_package
from src.core.packaging import (
    _safe_extract,
    _safe_remove_generated,
    _sha256,
    _zip_directory,
    run_package,
)


PROTOCOL = "V124_PUBLIC_QUADRATIC_STACK_V1"
ID_COL = "row_id"
TARGET_COL = "control_success"
ANCHOR_TEAM = 13
EXPECTED_PARENT_VALUES = {
    "v14_anchor_weight": 0.2,
    "v14_f_trend_alpha": 0.15,
    "v16_correction_weight": 1.5,
    "v20_extra_mode_weight": 0.045,
    "v20_eb_weights": [0.07, 0.16, 0.18],
    "v20_pfd_overlay_weight": 0.26,
    "v21_recipe_weights": [0.275, 0.075, 0.05, 0.1, 0.2, 0.05],
    "v22_domain_weights": {"R_CORE": 0.035, "R_ANCHOR": 0.02, "F": 0.02},
    "v22_asof_weight": 0.05,
    "v25_blend_eta": 0.15,
}


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(float(observed), float(expected), atol=1e-15, rtol=0.0):
        raise ValueError(f"unexpected parent {label}: {observed} != {expected}")


def _provenance(axis: str, scale: float) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "axis": axis,
        "selected_scale_or_eta": float(scale),
        "selection_source": (
            "official sequential Public deltas plus strict-forward 2023-to-2024 "
            "OOF Brier curvature; no evaluation-row aggregate"
        ),
        "row_local_inference": True,
        "test_aggregate_used": False,
    }


def patch_specifications(
    specs: dict[str, dict[str, Any]], selected: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Apply the preregistered scalar changes to decoded nested JSON specs."""

    if not np.isclose(float(selected["v19_scale"]), 1.0, atol=0.0, rtol=0.0):
        raise ValueError("v124 must preserve the v19 state model exactly")

    v17_scale = float(selected["v17_scale"])
    v14 = specs["v14"]
    _assert_close(v14["anchor_weight"], EXPECTED_PARENT_VALUES["v14_anchor_weight"], "v14 anchor_weight")
    _assert_close(v14["f_trend_alpha"], EXPECTED_PARENT_VALUES["v14_f_trend_alpha"], "v14 f_trend_alpha")
    v14["anchor_weight"] = float(v14["anchor_weight"]) * v17_scale
    v14["f_trend_alpha"] = float(v14["f_trend_alpha"]) * v17_scale
    v14["v124_public_quadratic_stack"] = _provenance("v17_v14_domain_components", v17_scale)

    v16 = specs["v16"]
    _assert_close(v16["correction_weight"], EXPECTED_PARENT_VALUES["v16_correction_weight"], "v16 correction_weight")
    v16["correction_weight"] = float(v16["correction_weight"]) * v17_scale
    v16["effects"] = {
        key: float(value) * v17_scale for key, value in v16["effects"].items()
    }
    for key in ("minimum", "maximum", "mean", "mean_absolute", "weighted_mean_absolute"):
        if key in v16.get("effect_summary", {}):
            v16["effect_summary"][key] = float(v16["effect_summary"][key]) * v17_scale
    v16["v124_public_quadratic_stack"] = _provenance("v17_v16_rcore_effects", v17_scale)

    v20_scale = float(selected["v20_scale"])
    v20 = specs["v20"]
    _assert_close(v20["extra_mode"]["weight"], EXPECTED_PARENT_VALUES["v20_extra_mode_weight"], "v20 extra_mode weight")
    observed_v20_eb = [float(recipe["weight"]) for recipe in v20["eb_recipes"]]
    if observed_v20_eb != EXPECTED_PARENT_VALUES["v20_eb_weights"]:
        raise ValueError(f"unexpected parent v20 EB weights: {observed_v20_eb}")
    _assert_close(v20["pfd"]["overlay_weight"], EXPECTED_PARENT_VALUES["v20_pfd_overlay_weight"], "v20 PFD overlay_weight")
    v20["extra_mode"]["weight"] = float(v20["extra_mode"]["weight"]) * v20_scale
    for recipe in v20["eb_recipes"]:
        recipe["weight"] = float(recipe["weight"]) * v20_scale
    v20["pfd"]["overlay_weight"] = float(v20["pfd"]["overlay_weight"]) * v20_scale
    v20["v124_public_quadratic_stack"] = _provenance("v20_all_components", v20_scale)

    v21_scale = float(selected["v21_scale"])
    v21 = specs["v21"]
    observed_v21 = [float(recipe["weight"]) for recipe in v21["recipes"]]
    if observed_v21 != EXPECTED_PARENT_VALUES["v21_recipe_weights"]:
        raise ValueError(f"unexpected parent v21 recipe weights: {observed_v21}")
    for recipe in v21["recipes"]:
        recipe["weight"] = float(recipe["weight"]) * v21_scale
    v21["v124_public_quadratic_stack"] = _provenance("v21_all_recipes", v21_scale)

    v22_scale = float(selected["v22_scale"])
    v22 = specs["v22"]
    for domain, expected in EXPECTED_PARENT_VALUES["v22_domain_weights"].items():
        _assert_close(v22["domain_calibration"][domain]["weight"], expected, f"v22 {domain} weight")
        v22["domain_calibration"][domain]["weight"] = (
            float(v22["domain_calibration"][domain]["weight"]) * v22_scale
        )
    _assert_close(v22["asof_prior"]["weight"], EXPECTED_PARENT_VALUES["v22_asof_weight"], "v22 asof weight")
    v22["asof_prior"]["weight"] = float(v22["asof_prior"]["weight"]) * v22_scale
    v22["v124_public_quadratic_stack"] = _provenance("v22_calibration_and_asof", v22_scale)

    v25 = specs["v25"]
    _assert_close(v25["blend_eta"], EXPECTED_PARENT_VALUES["v25_blend_eta"], "v25 blend_eta")
    v25["blend_eta"] = float(selected["anchor_eta"])
    v25["selection_note"] = (
        "same frozen row-local R_ANCHOR model at eta=0.36; exact official Public "
        "quadratic from eta 0.075/0.10/0.15; DACON FAQ reply 320256"
    )
    v25["v124_public_quadratic_stack"] = _provenance("v25_anchor_eta", float(selected["anchor_eta"]))
    return specs


def _member_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _zip_member_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {name: _member_sha256(archive.read(name)) for name in archive.namelist()}


def _audit_archive_diff(
    parent_zip: Path, package: Path, expected_changed: set[str]
) -> dict[str, Any]:
    parent = _zip_member_hashes(parent_zip)
    candidate = _zip_member_hashes(package)
    if set(parent) != set(candidate):
        raise ValueError("v124 changed the archive member set")
    changed = {name for name in parent if parent[name] != candidate[name]}
    if changed != expected_changed:
        raise ValueError(
            f"v124 archive diff mismatch: changed={sorted(changed)}, expected={sorted(expected_changed)}"
        )
    return {
        "member_count": len(parent),
        "changed_members": sorted(changed),
        "unchanged_members": len(parent) - len(changed),
    }


def _projection(
    research: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    names = ("v17_scale", "v19_scale", "v20_scale", "v21_scale", "v22_scale", "anchor_eta")
    current = np.asarray([config["current_point"][name] for name in names], dtype=np.float64)
    selected = np.asarray([config["selected_point"][name] for name in names], dtype=np.float64)
    scores: dict[str, float] = {}
    for scenario, payload in research["scenarios"].items():
        linear = np.asarray(payload["public_linear"], dtype=np.float64)
        gram = np.asarray(payload["public_gram"], dtype=np.float64)
        current_value = float(current @ linear - current @ gram @ current)
        selected_value = float(selected @ linear - selected @ gram @ selected)
        scores[scenario] = float(config["current_champion_public"] + selected_value - current_value)
        expected = float(config["expected_projection"][scenario])
        _assert_close(scores[scenario], expected, f"projection {scenario}")
    minimum = min(scores.values())
    _assert_close(minimum, config["expected_projection"]["minimum"], "minimum projection")
    return {
        "scenario_public_scores": scores,
        "minimum_projected_public": minimum,
        "target_public": float(config["target_public"]),
        "clears_target_in_all_scenarios": bool(minimum >= float(config["target_public"])),
    }


def build_package(
    parent_zip: Path, config: dict[str, Any], output_dir: Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    staging = output_dir / "_staging"
    _safe_remove_generated(staging, output_dir)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(parent_zip) as archive:
        _safe_extract(archive, staging)

    paths = {key: staging / Path(value) for key, value in config["nested_paths"].items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    specs = {
        key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()
    }
    patch_specifications(specs, config["selected_point"])
    for key, path in paths.items():
        path.write_text(
            json.dumps(specs[key], ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8",
        )

    package = output_dir / str(config["package_name"])
    if package.exists():
        _safe_remove_generated(package, output_dir)
    package_result = _zip_directory(staging, package)
    _safe_remove_generated(staging, output_dir)
    verify_package(package)
    archive_diff = _audit_archive_diff(
        parent_zip, package, {str(value) for value in config["nested_paths"].values()}
    )
    return package, package_result, archive_diff


def _audit_rows(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    if source.empty:
        raise ValueError("official package smoke source is empty")
    mixed = pd.concat([source] * 6, ignore_index=True)
    mixed[ID_COL] = [f"v124_audit_{index:04d}" for index in range(len(mixed))]
    route = np.arange(len(mixed)) % 3
    mixed.loc[route == 0, "game_type"] = "R"
    mixed.loc[route == 0, "pitcher_team_id"] = 1
    mixed.loc[route == 0, "batter_team_id"] = 2
    mixed.loc[route == 1, "game_type"] = "R"
    mixed.loc[route == 1, "pitcher_team_id"] = ANCHOR_TEAM
    mixed.loc[route == 2, "game_type"] = "F"
    sample = pd.DataFrame({ID_COL: mixed[ID_COL], TARGET_COL: 0.0})
    return mixed, sample


def _aligned(frame: pd.DataFrame, output: pd.DataFrame) -> np.ndarray:
    return frame[ID_COL].map(output.set_index(ID_COL)[TARGET_COL]).to_numpy(np.float64)


def audit_package(
    package: Path, parent_zip: Path, data_dir: Path, timeout: int, runtime_limit: float
) -> dict[str, Any]:
    mixed, sample = _audit_rows(data_dir)
    parent_output, parent_seconds, _ = run_package(parent_zip, mixed, sample, timeout=timeout)
    candidate_output, candidate_seconds, smoke_stdout = run_package(package, mixed, sample, timeout=timeout)
    parent = _aligned(mixed, parent_output)
    candidate = _aligned(mixed, candidate_output)
    if not np.isfinite(candidate).all() or not ((candidate >= 0.0) & (candidate <= 1.0)).all():
        raise ValueError("v124 smoke produced invalid probabilities")
    shift = candidate - parent
    if not np.any(np.abs(shift) > 0.0):
        raise ValueError("v124 is prediction-identical to its parent")

    shuffled_test = mixed.sample(frac=1.0, random_state=124).reset_index(drop=True)
    shuffled_sample = pd.DataFrame({ID_COL: shuffled_test[ID_COL], TARGET_COL: 0.0})
    shuffled, _, _ = run_package(package, shuffled_test, shuffled_sample, timeout=timeout)
    base_map = candidate_output.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_map = shuffled.set_index(ID_COL)[TARGET_COL].sort_index()
    shuffled_max_abs = float(np.max(np.abs(base_map - shuffled_map)))
    if shuffled_max_abs > 1e-12:
        raise ValueError(f"v124 shuffled-row parity failure: {shuffled_max_abs}")

    parts = []
    for indices in np.array_split(np.arange(len(mixed)), 2):
        local_test = mixed.iloc[indices].reset_index(drop=True)
        local_sample = pd.DataFrame({ID_COL: local_test[ID_COL], TARGET_COL: 0.0})
        output, _, _ = run_package(package, local_test, local_sample, timeout=timeout)
        parts.append(output)
    partition_map = pd.concat(parts).set_index(ID_COL)[TARGET_COL].sort_index()
    partition_max_abs = float(np.max(np.abs(base_map - partition_map)))
    if partition_max_abs > 1e-12:
        raise ValueError(f"v124 partition parity failure: {partition_max_abs}")

    scale_rows = 245789
    scale = mixed.iloc[np.arange(scale_rows) % len(mixed)].reset_index(drop=True)
    scale[ID_COL] = [f"v124_scale_{index:06d}" for index in range(scale_rows)]
    scale_sample = pd.DataFrame({ID_COL: scale[ID_COL], TARGET_COL: 0.0})
    scale_output, scale_seconds, scale_stdout = run_package(
        package, scale, scale_sample, timeout=timeout
    )
    values = scale_output[TARGET_COL].to_numpy(np.float64)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("v124 scale proxy produced invalid probabilities")
    if scale_seconds > runtime_limit:
        raise ValueError(f"v124 runtime exceeded limit: {scale_seconds:.3f} > {runtime_limit:.3f}")

    regular = mixed["game_type"].astype(str).eq("R").to_numpy()
    anchor = mixed["pitcher_team_id"].eq(ANCHOR_TEAM).to_numpy() | mixed["batter_team_id"].eq(ANCHOR_TEAM).to_numpy()
    domains = {
        "R_CORE": regular & ~anchor,
        "R_ANCHOR": regular & anchor,
        "F": ~regular,
    }
    domain_shift = {}
    for name, mask in domains.items():
        local = shift[mask]
        domain_shift[name] = {
            "rows": int(mask.sum()),
            "changed_rows": int(np.count_nonzero(np.abs(local) > 0.0)),
            "mean": float(local.mean()),
            "mean_absolute": float(np.mean(np.abs(local))),
            "maximum_absolute": float(np.max(np.abs(local))),
        }
        if domain_shift[name]["changed_rows"] == 0:
            raise ValueError(f"v124 did not change audit predictions in {name}")
    return {
        "smoke_rows": int(len(mixed)),
        "parent_smoke_runtime_seconds": parent_seconds,
        "candidate_smoke_runtime_seconds": candidate_seconds,
        "domain_shift_vs_v104": domain_shift,
        "shuffled_max_abs": shuffled_max_abs,
        "partition_max_abs": partition_max_abs,
        "scale_proxy_rows": scale_rows,
        "scale_proxy_runtime_seconds": scale_seconds,
        "runtime_limit_seconds": runtime_limit,
        "scale_proxy_mean": float(values.mean()),
        "scale_proxy_min": float(values.min()),
        "scale_proxy_max": float(values.max()),
        "smoke_stdout": smoke_stdout,
        "scale_stdout": scale_stdout,
    }


def run(
    project: Path,
    parent_zip: Path,
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    project = project.resolve()
    parent_zip = parent_zip.resolve()
    data_dir = data_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    observed_parent_sha = _sha256(parent_zip)
    if observed_parent_sha != str(config["parent_zip_sha256"]).upper():
        raise ValueError(f"parent SHA mismatch: {observed_parent_sha}")

    research_path = project / str(config["research_manifest"])
    research = json.loads(research_path.read_text(encoding="utf-8"))
    projection = _projection(research, config)
    if not projection["clears_target_in_all_scenarios"]:
        raise ValueError("v124 does not clear the preregistered target in all scenarios")

    package, package_result, archive_diff = build_package(parent_zip, config, output_dir)
    audit = audit_package(
        package,
        parent_zip,
        data_dir,
        timeout,
        float(config["runtime_limit_seconds"]),
    )
    result = {
        "protocol": PROTOCOL,
        "config": config,
        "research_protocol": research["protocol"],
        "projection": projection,
        "parent_zip_sha256": observed_parent_sha,
        "package": package_result,
        "package_sha256": _sha256(package),
        "archive_diff": archive_diff,
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
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    run(args.project, args.parent_zip, args.data_dir, args.config, args.output_dir, args.timeout)


if __name__ == "__main__":
    main()
