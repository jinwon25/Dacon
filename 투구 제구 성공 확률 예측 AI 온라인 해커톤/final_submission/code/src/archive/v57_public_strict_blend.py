"""Audit a fully independent chronological model as a small v27 blend.

The challenger is the fixed EXP-021 ``strict`` recipe, reimplemented here.
Its OOF predictions are generated solely from
seasons earlier than each validation season.  This module never reads public
leaderboard feedback and does not use 2024 labels to choose a blend mode,
route, or weight.

Exact-recipe consensus on 2022 and late-2023 freezes one of two ordinary
convex blends (probability or logit space), four deployment routes, and a
small weight.  Full-2024 and late-2024 are opened only after that choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import brier_skill_score_unclipped
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.champion.v50_low_rank_pitcher_context import _frame, select_consensus


TARGET = "control_success"
MODES = ("probability", "logit")
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
STRICT_CANDIDATE = "lowrank_s300_r6"
STRICT_RELATIVE_ROOT = Path("artifacts/EXP-020/low_rank_pitcher_context_eb")


def _route_mask(frame: pd.DataFrame, route: str) -> np.ndarray:
    if route == "ALL":
        return np.ones(len(frame), dtype=bool)
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route}")
    return frame["domain3"].astype(str).eq(route).to_numpy()


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 0.001, 0.999)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def blend_candidate(
    frame: pd.DataFrame,
    parent: np.ndarray,
    challenger: np.ndarray,
    *,
    mode: str,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend only current-row predictions; no validation-row aggregation."""

    parent = np.asarray(parent, dtype=np.float64)
    challenger = np.asarray(challenger, dtype=np.float64)
    if not (len(frame) == len(parent) == len(challenger)):
        raise ValueError("blend row count mismatch")
    if not (np.isfinite(parent).all() and np.isfinite(challenger).all()):
        raise ValueError("blend inputs must be finite")
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("blend weight must be in [0, 1]")
    active = _route_mask(frame, route)
    output = parent.copy()
    if mode == "probability":
        blended = parent + float(weight) * (challenger - parent)
    elif mode == "logit":
        blended = _sigmoid(
            _logit(parent) + float(weight) * (_logit(challenger) - _logit(parent))
        )
    else:
        raise ValueError(f"unknown blend mode: {mode}")
    output[active] = np.clip(blended[active], 0.001, 0.999)
    return output, active


def _screen(
    frame: pd.DataFrame, parent: np.ndarray, challenger: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in MODES:
        signal = f"strict_{mode}"
        for route in ROUTES:
            for weight in WEIGHTS:
                candidate, active = blend_candidate(
                    frame,
                    parent,
                    challenger,
                    mode=mode,
                    route=route,
                    weight=weight,
                )
                result = diagnostics(frame, parent, candidate, active)
                applied_domain = (
                    min(result["domain_gains"].values())
                    if route == "ALL"
                    else result["domain_gains"][route]
                )
                rows.append(
                    {
                        "signal": signal,
                        "domain": route,
                        "weight": float(weight),
                        "gain": float(result["gain"]),
                        "positive_month_fraction": float(
                            result["positive_month_fraction"]
                        ),
                        "worst_month_gain": float(result["worst_month_gain"]),
                        "minimum_domain_gain": float(result["minimum_domain_gain"]),
                        "applied_domain_gain": float(applied_domain),
                        "selection_score": float(
                            min(
                                result["gain"],
                                result["worst_month_gain"],
                                applied_domain,
                            )
                        ),
                        "mean_abs_shift": float(result["mean_abs_shift"]),
                    }
                )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _strict_paths(component_root: Path, year: int) -> tuple[Path, Path]:
    root = component_root / STRICT_RELATIVE_ROOT
    return (
        root / f"predictions_{STRICT_CANDIDATE}_{year}.npy",
        root / f"targets_{year}.npy",
    )


def _load_strict(
    component_root: Path, raw: pd.DataFrame
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    predictions: dict[int, np.ndarray] = {}
    provenance: dict[str, object] = {"files": {}}
    for year in (2022, 2023, 2024):
        prediction_path, target_path = _strict_paths(component_root, year)
        prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
        saved_target = np.load(target_path, allow_pickle=False).astype(np.float64)
        expected = raw.loc[raw["season"].eq(year), TARGET].to_numpy(np.float64)
        if not np.array_equal(saved_target, expected):
            raise ValueError(f"strict OOF target/order mismatch for {year}")
        if len(prediction) != len(expected):
            raise ValueError(f"strict OOF prediction length mismatch for {year}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"strict OOF contains non-finite values for {year}")
        if float(prediction.min()) < 0.0 or float(prediction.max()) > 1.0:
            raise ValueError(f"strict OOF is outside probability bounds for {year}")
        predictions[year] = prediction
        provenance["files"][str(year)] = {
            "prediction_sha256": _sha256(prediction_path),
            "target_sha256": _sha256(target_path),
            "rows": int(len(prediction)),
        }
    return predictions, provenance


def _axis_diversity(
    frame: pd.DataFrame, parent: np.ndarray, challenger: np.ndarray
) -> dict[str, float]:
    target = frame["target"].to_numpy(np.float64)
    parent_error = target - np.asarray(parent, dtype=np.float64)
    challenger_error = target - np.asarray(challenger, dtype=np.float64)
    direction = np.asarray(challenger, dtype=np.float64) - parent
    denominator = float(np.mean(np.square(direction)))
    unconstrained_weight = (
        float(np.mean(parent_error * direction)) / denominator
        if denominator > 0.0
        else 0.0
    )
    correlation = float(np.corrcoef(parent_error, challenger_error)[0, 1])
    return {
        "parent_skill": float(brier_skill_score_unclipped(target, parent)),
        "challenger_skill": float(
            brier_skill_score_unclipped(target, challenger)
        ),
        "error_correlation": correlation,
        "mean_abs_prediction_difference": float(np.mean(np.abs(direction))),
        "same_axis_unconstrained_probability_weight": unconstrained_weight,
        "same_axis_convex_probability_weight": float(
            np.clip(unconstrained_weight, 0.0, 1.0)
        ),
    }


def _mode_from_signal(signal: str) -> str:
    prefix = "strict_"
    if not str(signal).startswith(prefix):
        raise ValueError(f"unexpected strict signal: {signal}")
    mode = str(signal)[len(prefix) :]
    if mode not in MODES:
        raise ValueError(f"unexpected strict mode: {mode}")
    return mode


def run(project: Path, component_root: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    component_root = (project / component_root).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    strict, provenance = _load_strict(component_root, raw)
    axes = _cached_v25_axes(project, raw)

    meta22 = _metadata(project, 2022)
    expected22 = raw.loc[raw["season"].eq(2022), TARGET].to_numpy(np.float64)
    if not np.array_equal(meta22["target"], expected22):
        raise ValueError("2022 parent target/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])

    rows23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    late23_mask = rows23["game_month"].ge(8).to_numpy()
    frame23 = axes["selection_late_2023"]
    if not np.array_equal(
        frame23["target"].to_numpy(np.float64),
        rows23.loc[late23_mask, TARGET].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 parent target/order mismatch")
    strict23 = strict[2023][late23_mask]
    parent23 = v27_parent(frame23)

    stage1 = _screen(frame22, meta22["parent"], strict[2022])
    stage2 = _screen(frame23, parent23, strict23)
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "mode": _mode_from_signal(str(chosen["signal"])),
        "route": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    frame24 = axes["outer_full_2024"]
    late24 = axes["replication_late_2024"]
    rows24 = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    if not np.array_equal(
        frame24["target"].to_numpy(np.float64),
        rows24[TARGET].to_numpy(np.float64),
    ):
        raise ValueError("full-2024 parent target/order mismatch")
    late24_mask = rows24["game_month"].ge(8).to_numpy()
    if not np.array_equal(
        late24["target"].to_numpy(np.float64),
        rows24.loc[late24_mask, TARGET].to_numpy(np.float64),
    ):
        raise ValueError("late-2024 parent target/order mismatch")

    audit_inputs = {
        "outer_full_2024": (frame24, strict[2024]),
        "replication_late_2024": (late24, strict[2024][late24_mask]),
    }
    audits: dict[str, dict[str, object]] = {}
    diversity = {
        "selection_2022": _axis_diversity(frame22, meta22["parent"], strict[2022]),
        "selection_late_2023": _axis_diversity(frame23, parent23, strict23),
    }
    for axis_name, (frame, challenger) in audit_inputs.items():
        parent = v27_parent(frame)
        candidate, active = blend_candidate(
            frame, parent, challenger, **recipe
        )
        audits[axis_name] = diagnostics(frame, parent, candidate, active)
        diversity[axis_name] = _axis_diversity(frame, parent, challenger)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=parent,
            strict=challenger,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": bool(audits["outer_full_2024"]["gain"] >= 5.0),
        "outer_month_fraction_at_least_075": bool(
            audits["outer_full_2024"]["positive_month_fraction"] >= 0.75
        ),
        "outer_worst_month_above_minus_10": bool(
            audits["outer_full_2024"]["worst_month_gain"] > -10.0
        ),
        "outer_minimum_domain_nonnegative": bool(
            audits["outer_full_2024"]["minimum_domain_gain"] >= 0.0
        ),
        "replication_gain_positive": bool(
            audits["replication_late_2024"]["gain"] > 0.0
        ),
        "replication_month_fraction_at_least_two_thirds": bool(
            audits["replication_late_2024"]["positive_month_fraction"] >= 2.0 / 3.0
        ),
    }
    summary = {
        "protocol": "V57_INDEPENDENT_EXP021_STRICT_BLEND_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "challenger": {
            "recipe_origin": "team EXP-021 strict, independently reimplemented",
            "recipe": "EXP-021 strict",
            "oof_model": (
                "50/50 fixed R-LightGBM and HistGradientBoosting residual; "
                "past-OOF team EB all-prior smoothing 1000; pitcher x 24 "
                "context low-rank EB smoothing 300 rank 6"
            ),
            "provenance": provenance,
        },
        "selection": "exact mode/route/weight consensus on 2022 and late-2023 only",
        "grid": {
            "modes": list(MODES),
            "routes": list(ROUTES),
            "weights": list(WEIGHTS),
        },
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(chosen["worst_month_gain_2023"]),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "diversity": diversity,
        "audits": audits,
        "gates": gates,
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "public_score_used_for_recipe_selection": False,
        "audit_labels_used_for_selection": False,
    }
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
        "--component-root",
        type=Path,
        default=Path("artifacts/external_mk_lg9"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v57_public_strict_blend_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.component_root, args.output_dir)


if __name__ == "__main__":
    main()
