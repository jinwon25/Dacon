"""Validate independent team OOF bundles and fit a robust incumbent-anchored blend.

The optimizer never sees Public scores or test-row aggregates.  It fits only on
explicit non-contaminated source axes and applies the frozen weights to audit
axes.  A zero candidate weight is always feasible, so group constraints cannot
force a harmful blend merely to produce a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.metrics import brier_score, brier_skill_score_unclipped, validate_probabilities


PROTOCOL = "TEAM_OOF_BUNDLE_V1"
ALLOWED_ROLES = {"nested_outer", "locked_shadow", "development_contaminated"}
PRIMARY_ROLES = {"nested_outer", "locked_shadow"}
CSV_COLUMNS = (
    "row_id",
    "axis",
    "evaluation_season",
    "target",
    "prediction",
    "domain3",
    "month",
    "pitcher_id",
    "batter_id",
)
MANIFEST_FIELDS = {
    "protocol",
    "model_name",
    "model_sha256",
    "config_sha256",
    "training_data_sha256",
    "comparison_parent_sha256",
    "row_local",
    "test_aggregate_used",
    "public_score_used_for_weight",
    "axes",
}
AXIS_FIELDS = {
    "axis",
    "role",
    "evaluation_season",
    "training_max_season",
    "recipe_frozen_before_axis",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class OOFBundle:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    csv_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate provenance and return unique axis metadata by axis name."""

    missing = MANIFEST_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest["protocol"] != PROTOCOL:
        raise ValueError(f"manifest protocol must be {PROTOCOL}")
    for field in (
        "model_sha256",
        "config_sha256",
        "training_data_sha256",
        "comparison_parent_sha256",
    ):
        if not SHA256_RE.fullmatch(str(manifest[field])):
            raise ValueError(f"{field} must be a SHA-256 hex digest")
    if not str(manifest["model_name"]).strip():
        raise ValueError("model_name must be non-empty")
    if not _strict_bool(manifest["row_local"], "row_local"):
        raise ValueError("row_local must be true")
    if _strict_bool(manifest["test_aggregate_used"], "test_aggregate_used"):
        raise ValueError("test_aggregate_used must be false")
    if _strict_bool(
        manifest["public_score_used_for_weight"], "public_score_used_for_weight"
    ):
        raise ValueError("public_score_used_for_weight must be false")

    axes_raw = manifest["axes"]
    if not isinstance(axes_raw, list) or not axes_raw:
        raise ValueError("axes must be a non-empty list")
    axes: dict[str, dict[str, Any]] = {}
    for item in axes_raw:
        if not isinstance(item, dict):
            raise ValueError("every axis manifest entry must be an object")
        missing_axis = AXIS_FIELDS.difference(item)
        if missing_axis:
            raise ValueError(f"axis manifest missing fields: {sorted(missing_axis)}")
        name = str(item["axis"]).strip()
        if not name or name in axes:
            raise ValueError("axis names must be non-empty and unique")
        role = str(item["role"])
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unknown axis role: {role}")
        evaluation = int(item["evaluation_season"])
        training_max = int(item["training_max_season"])
        if training_max >= evaluation:
            raise ValueError(f"axis {name} is not strict-forward")
        frozen = _strict_bool(
            item["recipe_frozen_before_axis"],
            f"axes[{name}].recipe_frozen_before_axis",
        )
        axes[name] = {
            **item,
            "axis": name,
            "role": role,
            "evaluation_season": evaluation,
            "training_max_season": training_max,
            "recipe_frozen_before_axis": frozen,
        }
    return axes


def load_bundle(csv_path: Path, manifest_path: Path) -> OOFBundle:
    """Load one bundle without accepting implicit row order or missing provenance."""

    csv_path = csv_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    axes = validate_manifest(manifest)
    raw = pd.read_csv(csv_path, low_memory=False)
    missing = set(CSV_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"OOF CSV missing columns: {sorted(missing)}")
    frame = raw.loc[:, list(CSV_COLUMNS)].copy()
    if frame[["row_id", "axis", "domain3", "pitcher_id", "batter_id"]].isna().any().any():
        raise ValueError("OOF identity columns contain missing values")
    frame["row_id"] = frame["row_id"].astype(str)
    frame["axis"] = frame["axis"].astype(str)
    if frame.duplicated(["axis", "row_id"]).any():
        raise ValueError("OOF (axis, row_id) keys must be unique")
    frame["evaluation_season"] = pd.to_numeric(
        frame["evaluation_season"], errors="raise"
    ).astype(int)
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(float)
    if not np.isin(frame["target"].to_numpy(), (0.0, 1.0)).all():
        raise ValueError("target must be binary")
    frame["prediction"] = validate_probabilities(
        pd.to_numeric(frame["prediction"], errors="raise").to_numpy()
    )
    frame["month"] = pd.to_numeric(frame["month"], errors="raise").astype(int)
    if not frame["month"].between(1, 12).all():
        raise ValueError("month must be in [1, 12]")
    csv_axes = set(frame["axis"])
    if csv_axes != set(axes):
        raise ValueError("CSV and manifest axes differ")
    for name, item in axes.items():
        seasons = set(frame.loc[frame["axis"].eq(name), "evaluation_season"])
        if seasons != {int(item["evaluation_season"])}:
            raise ValueError(f"evaluation season mismatch for axis {name}")
    return OOFBundle(frame=frame, manifest=manifest, csv_sha256=_sha256(csv_path))


def align_bundles(
    incumbent: OOFBundle, candidates: list[OOFBundle]
) -> tuple[pd.DataFrame, list[str]]:
    """Strictly align candidate predictions to incumbent keys and metadata."""

    if not candidates:
        raise ValueError("at least one candidate bundle is required")
    base = incumbent.frame.sort_values(["axis", "row_id"], kind="stable").reset_index(
        drop=True
    )
    base = base.rename(columns={"prediction": "incumbent_probability"})
    parent_sha = str(incumbent.manifest["model_sha256"]).upper()
    training_sha = str(incumbent.manifest["training_data_sha256"]).upper()
    incumbent_axes = validate_manifest(incumbent.manifest)
    names: list[str] = []
    model_hashes: set[str] = set()
    for candidate in candidates:
        name = str(candidate.manifest["model_name"]).strip()
        if name in names:
            raise ValueError(f"duplicate candidate model_name: {name}")
        model_hash = str(candidate.manifest["model_sha256"]).upper()
        if model_hash in model_hashes:
            raise ValueError(f"duplicate candidate model_sha256: {model_hash}")
        model_hashes.add(model_hash)
        if str(candidate.manifest["comparison_parent_sha256"]).upper() != parent_sha:
            raise ValueError(f"comparison parent mismatch for {name}")
        if str(candidate.manifest["training_data_sha256"]).upper() != training_sha:
            raise ValueError(f"training data mismatch for {name}")
        candidate_axes = validate_manifest(candidate.manifest)
        if set(candidate_axes) != set(incumbent_axes):
            raise ValueError(f"manifest axis mismatch for {name}")
        for axis, incumbent_axis in incumbent_axes.items():
            candidate_axis = candidate_axes[axis]
            for field in ("role", "evaluation_season", "recipe_frozen_before_axis"):
                if candidate_axis[field] != incumbent_axis[field]:
                    raise ValueError(f"axis metadata mismatch for {name}: {axis}.{field}")
        local = candidate.frame.sort_values(
            ["axis", "row_id"], kind="stable"
        ).reset_index(drop=True)
        for column in (
            "row_id",
            "axis",
            "evaluation_season",
            "target",
            "domain3",
            "month",
            "pitcher_id",
            "batter_id",
        ):
            if not np.array_equal(
                base[column].astype(str).to_numpy(), local[column].astype(str).to_numpy()
            ):
                raise ValueError(f"bundle alignment mismatch for {name}: {column}")
        base[f"candidate__{name}"] = local["prediction"].to_numpy(np.float64)
        names.append(name)
    return base, names


def _group_indices(frame: pd.DataFrame, min_group_rows: int) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for axis in sorted(frame["axis"].unique()):
        axis_mask = frame["axis"].eq(axis).to_numpy()
        groups[f"axis={axis}"] = np.flatnonzero(axis_mask)
        for column in ("domain3", "month"):
            values = frame.loc[axis_mask, column].unique()
            for value in sorted(values, key=str):
                mask = axis_mask & frame[column].eq(value).to_numpy()
                index = np.flatnonzero(mask)
                if len(index) >= int(min_group_rows):
                    groups[f"axis={axis}|{column}={value}"] = index
    return groups


def fit_robust_blend(
    frame: pd.DataFrame,
    candidate_names: list[str],
    source_axes: list[str],
    *,
    max_total_candidate_weight: float = 1.0,
    ridge: float = 1e-8,
    max_group_brier_increase: float = 0.0,
    min_group_rows: int = 500,
) -> dict[str, Any]:
    """Fit nonnegative candidate weights under source month/domain safety constraints."""

    if not candidate_names:
        raise ValueError("candidate_names is empty")
    if not source_axes or len(set(source_axes)) != len(source_axes):
        raise ValueError("source_axes must be non-empty and unique")
    source = frame.loc[frame["axis"].isin(source_axes)].reset_index(drop=True)
    if set(source["axis"].unique()) != set(source_axes):
        raise ValueError("one or more source axes are missing")
    if not 0.0 < float(max_total_candidate_weight) <= 1.0:
        raise ValueError("max_total_candidate_weight must be in (0, 1]")
    if ridge < 0.0 or min_group_rows < 1:
        raise ValueError("ridge must be nonnegative and min_group_rows positive")

    target = source["target"].to_numpy(np.float64)
    incumbent = source["incumbent_probability"].to_numpy(np.float64)
    candidate = np.column_stack(
        [source[f"candidate__{name}"].to_numpy(np.float64) for name in candidate_names]
    )
    direction = candidate - incumbent[:, None]
    groups = _group_indices(source, min_group_rows)
    incumbent_loss = np.square(incumbent - target)

    def predict(weight: np.ndarray) -> np.ndarray:
        return incumbent + direction @ weight

    def objective(weight: np.ndarray) -> float:
        residual = predict(weight) - target
        return float(np.mean(np.square(residual)) + float(ridge) * weight @ weight)

    constraints: list[dict[str, Any]] = [
        {
            "type": "ineq",
            "fun": lambda weight: float(max_total_candidate_weight - weight.sum()),
        }
    ]
    for index in groups.values():
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weight, idx=index: float(
                    max_group_brier_increase
                    - np.mean(np.square(predict(weight)[idx] - target[idx]) - incumbent_loss[idx])
                ),
            }
        )
    result = minimize(
        objective,
        x0=np.zeros(len(candidate_names), dtype=np.float64),
        method="SLSQP",
        bounds=[(0.0, max_total_candidate_weight)] * len(candidate_names),
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"robust blend optimization failed: {result.message}")
    weight = np.clip(result.x, 0.0, max_total_candidate_weight)
    prediction = predict(weight)
    group_delta = {
        name: float(
            np.mean(np.square(prediction[index] - target[index]) - incumbent_loss[index])
        )
        for name, index in groups.items()
    }
    return {
        "weights": {name: float(value) for name, value in zip(candidate_names, weight)},
        "incumbent_weight": float(1.0 - weight.sum()),
        "source_rows": int(len(source)),
        "source_axes": list(source_axes),
        "source_brier_incumbent": brier_score(target, incumbent),
        "source_brier_blend": brier_score(target, prediction),
        "source_unclipped_bss_gain": float(
            brier_skill_score_unclipped(target, prediction)
            - brier_skill_score_unclipped(target, incumbent)
        ),
        "maximum_group_brier_increase": float(max(group_delta.values())),
        "group_brier_deltas": group_delta,
        "optimizer_message": str(result.message),
    }


def single_candidate_headroom(
    target: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Return the exact unconstrained quadratic headroom on one labelled axis."""

    y = np.asarray(target, dtype=np.float64)
    p0 = validate_probabilities(incumbent)
    p1 = validate_probabilities(candidate)
    if not (len(y) == len(p0) == len(p1)) or len(y) == 0:
        raise ValueError("headroom arrays must be aligned and non-empty")
    direction = p1 - p0
    curvature = float(np.mean(np.square(direction)))
    linear = float(np.mean((p0 - y) * direction))
    raw = 0.0 if curvature == 0.0 else -linear / curvature
    weight = float(np.clip(raw, 0.0, 1.0))
    blend = p0 + weight * direction
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    brier_reduction = brier_score(y, p0) - brier_score(y, blend)
    if float(np.std(p0)) == 0.0 or float(np.std(p1)) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(p0, p1)[0, 1])
    return {
        "optimal_candidate_weight": weight,
        "brier_reduction": float(brier_reduction),
        "unclipped_bss_gain": float(100000.0 * brier_reduction / reference),
        "prediction_correlation": correlation,
        "shift_rms": float(np.sqrt(curvature)),
    }


def evaluate_axes(
    frame: pd.DataFrame, candidate_names: list[str], weights: dict[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis in sorted(frame["axis"].unique()):
        local = frame.loc[frame["axis"].eq(axis)].reset_index(drop=True)
        target = local["target"].to_numpy(np.float64)
        parent = local["incumbent_probability"].to_numpy(np.float64)
        blend = parent.copy()
        for name in candidate_names:
            blend += float(weights[name]) * (
                local[f"candidate__{name}"].to_numpy(np.float64) - parent
            )
        month_gains = []
        for month in sorted(local["month"].unique()):
            mask = local["month"].eq(month).to_numpy()
            month_gains.append(
                brier_skill_score_unclipped(target[mask], blend[mask])
                - brier_skill_score_unclipped(target[mask], parent[mask])
            )
        domain_gains = []
        for domain in sorted(local["domain3"].unique()):
            mask = local["domain3"].eq(domain).to_numpy()
            domain_gains.append(
                brier_skill_score_unclipped(target[mask], blend[mask])
                - brier_skill_score_unclipped(target[mask], parent[mask])
            )
        rows.append(
            {
                "axis": axis,
                "rows": int(len(local)),
                "gain": float(
                    brier_skill_score_unclipped(target, blend)
                    - brier_skill_score_unclipped(target, parent)
                ),
                "positive_month_fraction": float(np.mean(np.asarray(month_gains) > 0.0)),
                "worst_month_gain": float(min(month_gains)),
                "minimum_domain_gain": float(min(domain_gains)),
                "mean_abs_shift": float(np.mean(np.abs(blend - parent))),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-csv", type=Path, required=True)
    parser.add_argument("--incumbent-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("CSV", "MANIFEST"),
        required=True,
    )
    parser.add_argument("--source-axis", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    incumbent = load_bundle(args.incumbent_csv, args.incumbent_manifest)
    candidates = [load_bundle(Path(csv), Path(manifest)) for csv, manifest in args.candidate]
    aligned, names = align_bundles(incumbent, candidates)
    incumbent_axes = validate_manifest(incumbent.manifest)
    for axis in args.source_axis:
        if axis not in incumbent_axes or incumbent_axes[axis]["role"] not in PRIMARY_ROLES:
            raise ValueError(f"source axis is absent or contaminated: {axis}")
        if not bool(incumbent_axes[axis]["recipe_frozen_before_axis"]):
            raise ValueError(f"source recipe was not frozen before axis: {axis}")
    fitted = fit_robust_blend(aligned, names, args.source_axis)
    audits = evaluate_axes(aligned, names, fitted["weights"])
    result = {
        "protocol": "V77_TEAM_OOF_CONSTRAINED_BLEND_V1",
        "incumbent": incumbent.manifest["model_name"],
        "incumbent_csv_sha256": incumbent.csv_sha256,
        "candidate_csv_sha256": {
            bundle.manifest["model_name"]: bundle.csv_sha256 for bundle in candidates
        },
        "fit": fitted,
        "audits": audits,
        "public_score_used": False,
        "test_rows_read": False,
        "eligible_for_public_probe": False,
        "eligibility_note": "run dependence-aware bootstrap and family Reality Check first",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(audits).to_csv(args.output_dir / "axis_metrics.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
