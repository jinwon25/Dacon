"""Prior-OOF centred auxiliary failure-logit correction above frozen v27.

Two binary learners predict mutually exclusive failure components recovered
from labelled training ASOF transitions:

* ``mr``: middle or reverse-location failure;
* ``wayoff``: an unsuccessful pitch that is not ``mr``.

For each audit origin, their centred logits are combined with coefficients
fitted on the immediately preceding season's OOF predictions.  The success
model's scale and intercept remain fixed: only the two auxiliary directions
are added.  The same route and damping must pass 2022 and late-2023 before the
full-2024 and late-2024 audits are opened.

No coefficient, centre, or feature is computed from evaluation/test rows.
Current-fold labels train only the *next* origin's frozen auxiliary model and
offset, exactly as in a rolling-origin pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.failure_mode_privileged_distillation import reconstruct_failure_mode
from src.multi_year_state_model import _add_categories, _state_features
from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.v50_low_rank_pitcher_context import select_consensus


TARGET = "control_success"
HALF_LIFE = 0.5
OFFSET_L2 = 0.001
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.10, 0.25, 0.50, 1.00)
EPS = 1e-5


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(value / (1.0 - value))


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def failure_targets(
    control_success: np.ndarray, mode: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid mask plus the disjoint MR and way-off binary labels."""

    success = np.asarray(control_success, dtype=np.int8)
    mode = np.asarray(mode, dtype=np.int8)
    if len(success) != len(mode):
        raise ValueError("success/mode length mismatch")
    valid = mode >= 0
    mr = (mode == 0).astype(np.int8)
    wayoff = ((success == 0) & (mode != 0) & valid).astype(np.int8)
    if np.any((mr == 1) & (wayoff == 1)):
        raise AssertionError("failure targets are not disjoint")
    if np.any(valid & (success == 0) & ((mr + wayoff) != 1)):
        raise AssertionError("failure partition does not cover every failure")
    return valid, mr, wayoff


def _binary_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=6,
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=700,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=2.0,
        reg_lambda=18.0,
        max_bin=127,
        random_state=seed,
    )


def fit_auxiliary_oof(
    train: pd.DataFrame,
    features: pd.DataFrame,
    categorical: list[str],
    valid_label: np.ndarray,
    mr_label: np.ndarray,
    wayoff_label: np.ndarray,
    audit_year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit both auxiliary learners on strictly earlier seasons."""

    fit = train["season"].lt(audit_year).to_numpy() & valid_label
    audit = train["season"].eq(audit_year).to_numpy()
    if not fit.any() or not audit.any():
        raise ValueError(f"empty auxiliary fold for {audit_year}")
    fit_year = train.loc[fit, "season"].to_numpy(np.float64)
    sample_weight = np.exp2(
        -(float(audit_year - 1) - fit_year) / HALF_LIFE
    )
    sample_weight /= sample_weight.mean()
    predictions: dict[str, np.ndarray] = {}
    rates: dict[str, float] = {}
    for index, (name, label) in enumerate(
        (("mr", mr_label), ("wayoff", wayoff_label))
    ):
        model = _binary_model(seed=52000 + 10 * audit_year + index)
        model.fit(
            features.loc[fit],
            label[fit],
            sample_weight=sample_weight,
            categorical_feature=categorical,
        )
        predictions[name] = model.predict_proba(features.loc[audit])[:, 1]
        rates[name] = float(label[fit].mean())
        del model
        gc.collect()
    return predictions["mr"], predictions["wayoff"], {
        "audit_year": int(audit_year),
        "fit_years": sorted(
            int(value) for value in train.loc[fit, "season"].unique()
        ),
        "fit_rows": int(fit.sum()),
        "audit_rows": int(audit.sum()),
        "half_life": HALF_LIFE,
        "fit_label_rates": rates,
        "prediction_means": {
            name: float(value.mean()) for name, value in predictions.items()
        },
    }


@dataclass(frozen=True)
class OffsetState:
    coefficient_mr: float
    coefficient_wayoff: float
    mean_logit_mr: float
    mean_logit_wayoff: float
    source_rows: int
    source_objective: float


def fit_offset(
    target: np.ndarray,
    base: np.ndarray,
    probability_mr: np.ndarray,
    probability_wayoff: np.ndarray,
    *,
    l2: float = OFFSET_L2,
) -> OffsetState:
    """Fit two centred-logit coefficients with fixed success scale/intercept."""

    target = np.asarray(target, dtype=np.float64)
    base_logit = _logit(base)
    mr_logit = _logit(probability_mr)
    wayoff_logit = _logit(probability_wayoff)
    if not (
        len(target)
        == len(base_logit)
        == len(mr_logit)
        == len(wayoff_logit)
    ):
        raise ValueError("offset source length mismatch")
    mean_mr = float(mr_logit.mean())
    mean_wayoff = float(wayoff_logit.mean())
    design = np.column_stack(
        [mr_logit - mean_mr, wayoff_logit - mean_wayoff]
    )

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        prediction = np.clip(
            _expit(base_logit + design @ coefficient), EPS, 1.0 - EPS
        )
        nll = -float(
            np.mean(
                target * np.log(prediction)
                + (1.0 - target) * np.log1p(-prediction)
            )
        )
        penalty = 0.5 * float(l2) * float(np.square(coefficient).sum())
        gradient = design.T @ (prediction - target) / len(target)
        gradient += float(l2) * coefficient
        return nll + penalty, gradient

    result = minimize(
        objective,
        x0=np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
    )
    if not result.success:
        raise RuntimeError(f"auxiliary offset fit failed: {result.message}")
    return OffsetState(
        coefficient_mr=float(result.x[0]),
        coefficient_wayoff=float(result.x[1]),
        mean_logit_mr=mean_mr,
        mean_logit_wayoff=mean_wayoff,
        source_rows=int(len(target)),
        source_objective=float(result.fun),
    )


def offset_direction(
    probability_mr: np.ndarray,
    probability_wayoff: np.ndarray,
    state: OffsetState,
) -> np.ndarray:
    """Map fixed source centres/coefficients independently to each row."""

    return (
        state.coefficient_mr
        * (_logit(probability_mr) - state.mean_logit_mr)
        + state.coefficient_wayoff
        * (_logit(probability_wayoff) - state.mean_logit_wayoff)
    )


def apply_offset(
    frame: pd.DataFrame,
    parent: np.ndarray,
    direction: np.ndarray,
    route: str,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    domain = frame["domain3"].astype(str).to_numpy()
    active = np.ones(len(frame), dtype=bool) if route == "ALL" else domain == route
    parent = np.asarray(parent, dtype=np.float64)
    output = parent.copy()
    output[active] = np.clip(
        _expit(_logit(parent[active]) + float(weight) * direction[active]),
        0.001,
        0.999,
    )
    return output, active


def _screen(
    frame: pd.DataFrame,
    parent: np.ndarray,
    probability_mr: np.ndarray,
    probability_wayoff: np.ndarray,
    state: OffsetState,
) -> pd.DataFrame:
    direction = offset_direction(probability_mr, probability_wayoff, state)
    rows: list[dict[str, object]] = []
    for route in ROUTES:
        for weight in WEIGHTS:
            candidate, active = apply_offset(
                frame, parent, direction, route, weight
            )
            result = diagnostics(frame, parent, candidate, active)
            applied_domain = (
                min(result["domain_gains"].values())
                if route == "ALL"
                else result["domain_gains"][route]
            )
            rows.append(
                {
                    "signal": "mr_wayoff_auxlogit",
                    "domain": route,
                    "weight": float(weight),
                    "gain": float(result["gain"]),
                    "positive_month_fraction": float(
                        result["positive_month_fraction"]
                    ),
                    "worst_month_gain": float(result["worst_month_gain"]),
                    "minimum_domain_gain": float(
                        result["minimum_domain_gain"]
                    ),
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


def _wave0(project: Path, year: int) -> tuple[np.ndarray, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        return (
            saved["target"].astype(np.float64),
            saved["incumbent"].astype(np.float64),
        )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _add_domain_and_pressure(
        pd.read_csv(project / "data" / "train.csv", low_memory=False)
    )
    mode = reconstruct_failure_mode(train)
    valid, mr_label, wayoff_label = failure_targets(
        train[TARGET].to_numpy(np.int8), mode
    )
    numeric_state, _ = _state_features(train)
    features = _add_categories(train, numeric_state)
    categorical = [name for name in features if name.startswith("cat__")]

    auxiliary: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    auxiliary_audits: dict[str, object] = {}
    for year in (2021, 2022, 2023, 2024):
        print(f"[v52] auxiliary audit_year={year}", flush=True)
        mr, wayoff, audit = fit_auxiliary_oof(
            train,
            features,
            categorical,
            valid,
            mr_label,
            wayoff_label,
            year,
        )
        auxiliary[year] = (mr, wayoff)
        auxiliary_audits[str(year)] = audit
        np.savez_compressed(
            output_dir / f"auxiliary_o{year}.npz",
            target=train.loc[train["season"].eq(year), TARGET].to_numpy(np.int8),
            probability_mr=mr,
            probability_wayoff=wayoff,
        )
    del features, numeric_state
    gc.collect()

    raw = train
    year_rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2021, 2022, 2023, 2024)
    }
    axes = _cached_v25_axes(project, raw)

    target21, parent21 = _wave0(project, 2021)
    if not np.array_equal(
        target21, year_rows[2021][TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2021 wave0 target/order mismatch")
    source_state21 = fit_offset(
        target21, parent21, *auxiliary[2021]
    )

    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"], year_rows[2022][TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2022 target/order mismatch")
    source_state22 = fit_offset(
        meta22["target"], meta22["parent"], *auxiliary[2022]
    )

    late23 = year_rows[2023]["game_month"].ge(8).to_numpy()
    frame23 = axes["selection_late_2023"]
    if not np.array_equal(
        frame23["target"].to_numpy(np.float64),
        year_rows[2023].loc[late23, TARGET].to_numpy(np.float64),
    ):
        raise ValueError("late-2023 target/order mismatch")
    source_state23 = fit_offset(
        frame23["target"].to_numpy(np.float64),
        v27_parent(frame23),
        auxiliary[2023][0][late23],
        auxiliary[2023][1][late23],
    )

    frame22 = pd.DataFrame(
        {
            "target": meta22["target"],
            "game_month": meta22["month"],
            "domain3": meta22["domain"],
        }
    )
    stage1 = _screen(
        frame22,
        meta22["parent"],
        auxiliary[2022][0],
        auxiliary[2022][1],
        source_state21,
    )
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)
    stage2 = _screen(
        frame23,
        v27_parent(frame23),
        auxiliary[2023][0][late23],
        auxiliary[2023][1][late23],
        source_state22,
    )
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "domain": str(chosen["domain"]),
        "weight": float(chosen["weight"]),
    }

    correction24 = offset_direction(
        auxiliary[2024][0], auxiliary[2024][1], source_state23
    )
    late24 = year_rows[2024]["game_month"].ge(8).to_numpy()
    audits: dict[str, dict[str, object]] = {}
    for axis_name, direction in (
        ("outer_full_2024", correction24),
        ("replication_late_2024", correction24[late24]),
    ):
        frame = axes[axis_name]
        parent = v27_parent(frame)
        candidate, active = apply_offset(
            frame,
            parent,
            direction,
            recipe["domain"],
            recipe["weight"],
        )
        audits[axis_name] = diagnostics(frame, parent, candidate, active)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v27=parent,
            direction=direction,
            candidate=candidate,
            active=active,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "two_origin_consensus": bool(chosen["passes_consensus_gate"]),
        "outer_gain_at_least_5": audits["outer_full_2024"]["gain"] >= 5.0,
        "outer_month_fraction_at_least_075": audits["outer_full_2024"][
            "positive_month_fraction"
        ]
        >= 0.75,
        "outer_worst_month_above_minus_10": audits["outer_full_2024"][
            "worst_month_gain"
        ]
        > -10.0,
        "outer_minimum_domain_nonnegative": audits["outer_full_2024"][
            "minimum_domain_gain"
        ]
        >= 0.0,
        "replication_gain_positive": audits["replication_late_2024"]["gain"]
        > 0.0,
        "replication_month_fraction_at_least_two_thirds": audits[
            "replication_late_2024"
        ]["positive_month_fraction"]
        >= 2.0 / 3.0,
    }
    summary = {
        "protocol": "V52_TWO_ORIGIN_PRIOR_OOF_FAILURE_AUX_LOGIT_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "candidate_origin": (
            "independent public-repository MR/way-off auxiliary-logit pattern; "
            "all models, centres, and coefficients refit on local prior OOF"
        ),
        "configuration": {
            "auxiliary_half_life": HALF_LIFE,
            "offset_l2": OFFSET_L2,
            "coefficient_bounds": [-2.0, 2.0],
            "routes": list(ROUTES),
            "weights": list(WEIGHTS),
            "success_scale_fixed": True,
            "success_intercept_fixed": True,
        },
        "failure_label_coverage": float(valid.mean()),
        "auxiliary_audits": auxiliary_audits,
        "source_offset_states": {
            "2021_for_2022": asdict(source_state21),
            "2022_for_2023": asdict(source_state22),
            "late_2023_for_2024": asdict(source_state23),
        },
        "selection": "exact route/damping consensus on 2022 and late-2023 only",
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(
                chosen["worst_month_gain_2023"]
            ),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "audits": audits,
        "gates": {name: bool(value) for name, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
        "row_local_inference": True,
        "test_aggregate_used": False,
        "external_rate_or_coefficient_used": False,
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/v52_failure_aux_logit_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
