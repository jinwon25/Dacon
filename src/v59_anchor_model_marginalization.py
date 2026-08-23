"""Post-break R_ANCHOR model marginalization above the eta=0.15 parent.

The deployed direct model is one spline-logistic fit with C=0.1.  This screen
fits the already-preregistered C=0.01/0.1/1.0 variants on the same source rows
and averages or takes their median.  It also tests a low-degree row-local gate:
use a larger direct-model eta when the three members agree and a smaller eta
when they disagree.  The disagreement cut is frozen from labelled train-era
selection rows and never estimated from test.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.v23_multiyear_direct_screen import _joint_domain
from src.v23_postbreak_gam_screen import SPECS, _fit_predict
from src.v23_structural_residual_screen import _derived, _load_axis
from src.v25_postbreak_anchor_audit import _early_to_late_2024
from src.v30_diverse_covariance_screen import diagnostics
from src.v58_eta15_rebase_audit import CURRENT_ETA, eta_parent


MEMBER_NAMES = ("logistic_c001", "logistic_c01", "logistic_c1")
DISAGREEMENT_QUANTILES = (0.50, 0.75)
LOW_DISAGREEMENT_ETAS = (0.175, 0.20)
HIGH_DISAGREEMENT_ETAS = (0.10, 0.125, 0.15)


def marginal_prediction(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("member matrix must contain at least two models")
    if mode == "mean":
        return matrix.mean(axis=1)
    if mode == "median":
        return np.median(matrix, axis=1)
    if mode == "logit_mean":
        clipped = np.clip(matrix, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).mean(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    raise ValueError(f"unknown marginalization mode: {mode}")


def apply_anchor_direct(
    frame: pd.DataFrame,
    direct: np.ndarray,
    eta: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    v22 = frame["v22"].to_numpy(np.float64)
    direct = np.asarray(direct, dtype=np.float64)
    eta_array = np.broadcast_to(np.asarray(eta, dtype=np.float64), (len(frame),))
    if len(direct) != len(frame):
        raise ValueError("direct prediction length mismatch")
    active = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    candidate = eta_parent(frame, CURRENT_ETA)
    candidate[active] = np.clip(
        v22[active] + eta_array[active] * (direct[active] - v22[active]),
        0.001,
        0.999,
    )
    return candidate, active


def disagreement_eta(
    disagreement: np.ndarray,
    threshold: float,
    low_disagreement_eta: float,
    high_disagreement_eta: float,
) -> np.ndarray:
    disagreement = np.asarray(disagreement, dtype=np.float64)
    if threshold < 0.0:
        raise ValueError("disagreement threshold must be non-negative")
    return np.where(
        disagreement <= float(threshold),
        float(low_disagreement_eta),
        float(high_disagreement_eta),
    )


def _axis_members(
    cache_dir: Path,
    axis_name: str,
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    frozen_c01: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {}
    for member_name in MEMBER_NAMES:
        cache = cache_dir / f"{axis_name}_{member_name}.npy"
        if cache.exists():
            prediction = np.load(cache, allow_pickle=False).astype(np.float64)
            if len(prediction) != len(audit):
                raise ValueError(f"cached member length mismatch: {cache}")
            print(f"[v59 cache] {axis_name} {member_name}", flush=True)
        elif member_name == "logistic_c01":
            prediction = np.asarray(frozen_c01, dtype=np.float64)
            if len(prediction) != len(audit):
                raise ValueError(f"frozen c01 length mismatch: {axis_name}")
            np.save(cache, prediction)
            print(f"[v59 frozen] {axis_name} {member_name}", flush=True)
        else:
            spec = next(item for item in SPECS if item.name == member_name)
            print(f"[v59 fit] {axis_name} {member_name}", flush=True)
            prediction = _fit_predict(fit, audit, spec)
            np.save(cache, prediction)
        output[member_name] = prediction
    return output


def _matrix(members: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([members[name] for name in MEMBER_NAMES])


def _selection_candidates(
    frame: pd.DataFrame,
    member_matrix: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    parent = eta_parent(frame, CURRENT_ETA)
    active = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    disagreement = member_matrix.std(axis=1)
    thresholds = {
        f"q{int(quantile * 100)}": float(
            np.quantile(disagreement[active], quantile)
        )
        for quantile in DISAGREEMENT_QUANTILES
    }
    rows = []
    for mode in ("mean", "median", "logit_mean"):
        direct = marginal_prediction(member_matrix, mode)
        candidate, mask = apply_anchor_direct(frame, direct, CURRENT_ETA)
        result = diagnostics(frame, parent, candidate, mask)
        rows.append(
            {
                "recipe": f"fixed_{mode}",
                "mode": mode,
                "threshold_name": "none",
                "low_disagreement_eta": CURRENT_ETA,
                "high_disagreement_eta": CURRENT_ETA,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"months", "domain_gains"}
                },
                "applied_domain_gain": result["domain_gains"]["R_ANCHOR"],
            }
        )
        for threshold_name, threshold in thresholds.items():
            for low_eta in LOW_DISAGREEMENT_ETAS:
                for high_eta in HIGH_DISAGREEMENT_ETAS:
                    eta = disagreement_eta(
                        disagreement, threshold, low_eta, high_eta
                    )
                    candidate, mask = apply_anchor_direct(frame, direct, eta)
                    result = diagnostics(frame, parent, candidate, mask)
                    rows.append(
                        {
                            "recipe": (
                                f"gated_{mode}_{threshold_name}_"
                                f"lo{low_eta:g}_hi{high_eta:g}"
                            ),
                            "mode": mode,
                            "threshold_name": threshold_name,
                            "low_disagreement_eta": low_eta,
                            "high_disagreement_eta": high_eta,
                            **{
                                key: value
                                for key, value in result.items()
                                if key not in {"months", "domain_gains"}
                            },
                            "applied_domain_gain": result["domain_gains"][
                                "R_ANCHOR"
                            ],
                        }
                    )
    metrics = pd.DataFrame(rows)
    metrics["selection_score"] = metrics[
        ["gain", "worst_month_gain", "applied_domain_gain"]
    ].min(axis=1)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["positive_month_fraction"].eq(1.0)
        & metrics["worst_month_gain"].gt(0.0)
        & metrics["applied_domain_gain"].gt(0.0)
    )
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    return metrics, thresholds


def _apply_recipe(
    frame: pd.DataFrame,
    member_matrix: np.ndarray,
    recipe: dict[str, object],
    thresholds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct = marginal_prediction(member_matrix, str(recipe["mode"]))
    threshold_name = str(recipe["threshold_name"])
    disagreement = member_matrix.std(axis=1)
    if threshold_name == "none":
        eta = np.full(len(frame), CURRENT_ETA, dtype=np.float64)
    else:
        eta = disagreement_eta(
            disagreement,
            thresholds[threshold_name],
            float(recipe["low_disagreement_eta"]),
            float(recipe["high_disagreement_eta"]),
        )
    candidate, active = apply_anchor_direct(frame, direct, eta)
    return candidate, active, disagreement


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)

    rows23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    derived23 = _derived(rows23, _joint_domain(rows23))
    selection_fit = derived23.loc[derived23["game_month"].le(7)].reset_index(
        drop=True
    )
    selection = _load_axis(project, "y2023_early_to_late", raw)

    rows24 = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    derived24 = _derived(rows24, _joint_domain(rows24))
    replication_fit = derived24.loc[derived24["game_month"].le(7)].reset_index(
        drop=True
    )
    _, replication = _early_to_late_2024(project, raw)
    outer = _load_axis(project, "y2023_to_y2024", raw)
    del raw

    # Attach the exact v25 c=0.1 OOF column used to recover eta=0.15 parents.
    old_cache = project / "artifacts" / "v29_anchor_route_20260817_01"
    axes = {
        "selection_late_2023": selection,
        "outer_full_2024": outer,
        "replication_late_2024": replication,
    }
    frozen_c01_by_axis = {}
    for axis_name, frame in axes.items():
        c01 = np.load(old_cache / f"{axis_name}_direct.npy").astype(np.float64)
        frozen_c01_by_axis[axis_name] = c01
        v22 = frame["v22"].to_numpy(np.float64)
        active = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
        v25 = v22.copy()
        v25[active] = np.clip(
            v22[active] + 0.075 * (c01[active] - v22[active]), 0.001, 0.999
        )
        frame["v25"] = v25

    fits = {
        "selection_late_2023": selection_fit,
        "outer_full_2024": derived23,
        "replication_late_2024": replication_fit,
    }
    member_banks = {
        axis_name: _axis_members(
            output_dir,
            axis_name,
            fits[axis_name],
            frame,
            frozen_c01_by_axis[axis_name],
        )
        for axis_name, frame in axes.items()
    }
    selection_metrics, thresholds = _selection_candidates(
        selection, _matrix(member_banks["selection_late_2023"])
    )
    selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = selection_metrics.loc[selection_metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else selection_metrics.iloc[0]
    recipe = {
        key: selected[key]
        for key in (
            "recipe",
            "mode",
            "threshold_name",
            "low_disagreement_eta",
            "high_disagreement_eta",
        )
    }

    audits = {}
    for axis_name in ("outer_full_2024", "replication_late_2024"):
        frame = axes[axis_name]
        matrix = _matrix(member_banks[axis_name])
        candidate, active, disagreement = _apply_recipe(
            frame, matrix, recipe, thresholds
        )
        parent = eta_parent(frame, CURRENT_ETA)
        result = diagnostics(frame, parent, candidate, active)
        audits[axis_name] = result
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            eta15=parent,
            members=matrix,
            disagreement=disagreement,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "selection_gate": bool(selected["passes_selection_gate"]),
        "outer_gain_at_least_3": audits["outer_full_2024"]["gain"] >= 3.0,
        "outer_month_fraction_at_least_075": audits["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_5": audits["outer_full_2024"][
            "worst_month_gain"
        ]
        > -5.0,
        "outer_anchor_gain_positive": audits["outer_full_2024"][
            "domain_gains"
        ]["R_ANCHOR"]
        > 0.0,
        "replication_gain_positive": audits["replication_late_2024"]["gain"]
        > 0.0,
        "replication_all_months_positive": audits["replication_late_2024"][
            "positive_month_fraction"
        ]
        == 1.0,
    }
    summary = {
        "protocol": "V59_ETA15_ANCHOR_MODEL_MARGINALIZATION_V1",
        "parent": "eta=0.15 R_ANCHOR parent; final 0819 gate excluded locally",
        "members": list(MEMBER_NAMES),
        "selection_axis": "2023 March-July fit, August-October choose recipe",
        "audit_axes": "full 2024 and independent early-to-late 2024 refit",
        "disagreement_thresholds_frozen_on_selection_train": thresholds,
        "candidate_count": int(len(selection_metrics)),
        "selection_gate_count": int(
            selection_metrics["passes_selection_gate"].sum()
        ),
        "chosen": {key: (float(value) if isinstance(value, np.floating) else value) for key, value in recipe.items()},
        "selection": {
            key: float(selected[key])
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "applied_domain_gain",
            )
        },
        "audits": audits,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "test_distribution_used": False,
        "audit_labels_used_for_selection": False,
        "family_reuse_warning": "2024 has been viewed for earlier R_ANCHOR studies",
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/v59_anchor_marginalization_20260822_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
