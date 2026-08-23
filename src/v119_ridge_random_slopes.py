"""Strongly pooled crossed random slopes above exact v104 R_CORE.

This is an independent implementation of a general hierarchical-model idea,
not a copy of any external repository.  A sparse ridge model has no free
global intercept.  It contains pitcher-specific slopes and crossed entity/
context intercepts; unseen entities contribute exactly zero.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from src.v30_diverse_covariance_screen import diagnostics
from src.v118_abs_policy_did import add_context


PROTOCOL = "V119_RIDGE_RANDOM_SLOPES_V1"
TARGET = "control_success"
SLOPE_NAMES = (
    "intercept",
    "balls",
    "strikes",
    "batter_hand_sign",
    "outs",
    "runners",
    "log_li",
)
GROUP_NAMES = ("batter_id", "pitcher_team_id", "batter_team_id", "context")


def _context_key(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["count_state"].astype(str)
        + "|"
        + frame["pitcher_hand"].astype("string").fillna("__MISSING__").astype(str)
        + "|"
        + frame["batter_hand"].astype("string").fillna("__MISSING__").astype(str)
    ).to_numpy(str)


def _vocabulary(values: np.ndarray) -> dict[Any, int]:
    return {value: index for index, value in enumerate(pd.unique(values))}


def _codes(values: np.ndarray, vocabulary: dict[Any, int]) -> np.ndarray:
    return np.asarray([vocabulary.get(value, -1) for value in values], dtype=np.int64)


@dataclass(frozen=True)
class DesignState:
    centres: np.ndarray
    scales: np.ndarray
    pitcher: dict[Any, int]
    batter: dict[Any, int]
    pitcher_team: dict[Any, int]
    batter_team: dict[Any, int]
    context: dict[Any, int]

    @property
    def widths(self) -> tuple[int, ...]:
        return (
            len(self.pitcher) * len(SLOPE_NAMES),
            len(self.batter),
            len(self.pitcher_team),
            len(self.batter_team),
            len(self.context),
        )


def _raw_slopes(frame: pd.DataFrame) -> np.ndarray:
    batter_hand = pd.to_numeric(frame["batter_hand"], errors="coerce").fillna(0).to_numpy()
    li = np.maximum(pd.to_numeric(frame["li"], errors="coerce").fillna(0).to_numpy(np.float64), 0.0)
    return np.column_stack(
        [
            np.ones(len(frame), dtype=np.float64),
            pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(np.float64),
            pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(np.float64),
            np.where(batter_hand == 1, -1.0, np.where(batter_hand == 2, 1.0, 0.0)),
            pd.to_numeric(frame["outs_before"], errors="coerce").fillna(0).to_numpy(np.float64),
            pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy(np.float64),
            np.log1p(li),
        ]
    )


def fit_state(frame: pd.DataFrame) -> DesignState:
    slopes = _raw_slopes(frame)
    centres = slopes.mean(axis=0)
    scales = slopes.std(axis=0)
    centres[0] = 0.0
    scales[0] = 1.0
    scales = np.where(scales > 1e-8, scales, 1.0)
    return DesignState(
        centres=centres,
        scales=scales,
        pitcher=_vocabulary(frame["pitcher_id"].to_numpy()),
        batter=_vocabulary(frame["batter_id"].to_numpy()),
        pitcher_team=_vocabulary(frame["pitcher_team_id"].to_numpy()),
        batter_team=_vocabulary(frame["batter_team_id"].to_numpy()),
        context=_vocabulary(_context_key(frame)),
    )


def design_matrix(frame: pd.DataFrame, state: DesignState) -> sparse.csr_matrix:
    n_rows = len(frame)
    slopes = (_raw_slopes(frame) - state.centres) / state.scales
    pitcher = _codes(frame["pitcher_id"].to_numpy(), state.pitcher)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    valid_pitcher = pitcher >= 0
    row_index = np.flatnonzero(valid_pitcher)
    for slope_index in range(len(SLOPE_NAMES)):
        row_parts.append(row_index)
        col_parts.append(pitcher[valid_pitcher] * len(SLOPE_NAMES) + slope_index)
        data_parts.append(slopes[valid_pitcher, slope_index])
    offset = state.widths[0]
    groups = (
        (_codes(frame["batter_id"].to_numpy(), state.batter), state.widths[1]),
        (_codes(frame["pitcher_team_id"].to_numpy(), state.pitcher_team), state.widths[2]),
        (_codes(frame["batter_team_id"].to_numpy(), state.batter_team), state.widths[3]),
        (_codes(_context_key(frame), state.context), state.widths[4]),
    )
    for codes, width in groups:
        valid = codes >= 0
        row_parts.append(np.flatnonzero(valid))
        col_parts.append(offset + codes[valid])
        data_parts.append(np.ones(int(valid.sum()), dtype=np.float64))
        offset += width
    rows = np.concatenate(row_parts)
    columns = np.concatenate(col_parts)
    data = np.concatenate(data_parts)
    return sparse.csr_matrix((data, (rows, columns)), shape=(n_rows, offset))


@dataclass(frozen=True)
class RandomSlopeModel:
    state: DesignState
    coefficient: np.ndarray
    correction_cap: float


def fit_model(
    frame: pd.DataFrame,
    parent: np.ndarray,
    alpha: float,
    correction_cap: float,
) -> RandomSlopeModel:
    active = frame["domain3"].eq("R_CORE").to_numpy()
    source = frame.loc[active].reset_index(drop=True)
    state = fit_state(source)
    matrix = design_matrix(source, state)
    residual = source[TARGET].to_numpy(np.float64) - np.asarray(parent, dtype=np.float64)[active]
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr", tol=1e-7)
    model.fit(matrix, residual)
    return RandomSlopeModel(state, np.asarray(model.coef_, dtype=np.float64), correction_cap)


def predict_correction(model: RandomSlopeModel, frame: pd.DataFrame) -> np.ndarray:
    prediction = design_matrix(frame, model.state) @ model.coefficient
    return np.clip(np.asarray(prediction).reshape(-1), -model.correction_cap, model.correction_cap)


def apply(frame: pd.DataFrame, parent: np.ndarray, correction: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    active = frame["domain3"].eq("R_CORE").to_numpy()
    output = np.asarray(parent, dtype=np.float64).copy()
    output[active] = np.clip(output[active] + float(eta) * correction[active], 0.001, 0.999)
    return output, active


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": frame[TARGET].to_numpy(np.float64),
            "game_month": frame["game_month"].to_numpy(np.int16),
            "domain3": frame["domain3"].astype(str).to_numpy(),
        }
    )


def compact(result: dict[str, Any]) -> dict[str, float]:
    return {
        "gain": float(result["gain"]),
        "positive_month_fraction": float(result["positive_month_fraction"]),
        "worst_month_gain": float(result["worst_month_gain"]),
        "active_domain_gain": float(result["domain_gains"].get("R_CORE", 0.0)),
        "mean_abs_shift": float(result["mean_abs_shift"]),
    }


def _load_axis(
    contract_dir: Path,
    raw: pd.DataFrame,
    name: str,
    parent: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    with np.load(contract_dir / f"{name}.npz", allow_pickle=False) as saved:
        index = saved["raw_index"].astype(np.int64)
        target = saved["target"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    frame = add_context(raw.iloc[index].reset_index(drop=True))
    frame["domain3"] = domain
    if not np.array_equal(frame[TARGET].to_numpy(np.float64), target):
        raise ValueError(f"target/order mismatch: {name}")
    parent = np.asarray(parent, dtype=np.float64)
    if parent.shape != (len(frame),):
        raise ValueError(f"parent shape mismatch: {name}")
    return frame, parent


def _source_pass(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        row["gain"] > 0.0
        and row["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and row["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and row["active_domain_gain"] > float(gate["active_domain_gain_min"])
    )


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_axes: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv.resolve(), low_memory=False)
    with np.load(v104_axes.resolve(), allow_pickle=False) as saved:
        axes = {name: saved[name].astype(np.float64) for name in saved.files}
    full22, parent22 = _load_axis(contract_dir, raw, "v84_full_2022", axes["full_2022"])
    late23, parent23 = _load_axis(contract_dir, raw, "v84_late_2023", axes["late_2023"])
    full24, parent24 = _load_axis(contract_dir, raw, "v84_full_2024", axes["full_2024"])
    early22_mask = full22["game_month"].le(7).to_numpy()
    late22_mask = full22["game_month"].ge(8).to_numpy()
    transitions = (
        (
            "early22_to_late22",
            full22.loc[early22_mask].reset_index(drop=True),
            parent22[early22_mask],
            full22.loc[late22_mask].reset_index(drop=True),
            parent22[late22_mask],
        ),
        ("full22_to_late23", full22, parent22, late23, parent23),
    )
    rows: list[dict[str, Any]] = []
    fit_audits: dict[str, Any] = {}
    for axis_name, source, source_parent, audit, audit_parent in transitions:
        for alpha in config["ridge_alpha_grid"]:
            model = fit_model(source, source_parent, float(alpha), float(config["correction_cap"]))
            correction = predict_correction(model, audit)
            fit_audits[f"{axis_name}|a{float(alpha):g}"] = {
                "features": int(len(model.coefficient)),
                "nonzero_coefficients": int(np.sum(np.abs(model.coefficient) > 1e-12)),
                "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
            }
            for eta in config["eta_grid"]:
                candidate, active = apply(audit, audit_parent, correction, float(eta))
                result = diagnostics(_metric_frame(audit), audit_parent, candidate, active)
                rows.append(
                    {
                        "axis": axis_name,
                        "alpha": float(alpha),
                        "eta": float(eta),
                        **compact(result),
                    }
                )
    source = pd.DataFrame(rows)
    gate = config["source_gate"]
    ranking_rows = []
    for values, local in source.groupby(["alpha", "eta"], observed=True):
        records = local.to_dict("records")
        ranking_rows.append(
            {
                "alpha": float(values[0]),
                "eta": float(values[1]),
                "source_gate_passed": len(records) == 2 and all(_source_pass(row, gate) for row in records),
                "minimum_gain": float(local["gain"].min()),
                "minimum_month_fraction": float(local["positive_month_fraction"].min()),
                "worst_month_gain": float(local["worst_month_gain"].min()),
                "minimum_active_domain_gain": float(local["active_domain_gain"].min()),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"], ascending=False
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    final_model = fit_model(
        late23,
        parent23,
        float(selected["alpha"]),
        float(config["correction_cap"]),
    )
    correction24 = predict_correction(final_model, full24)
    candidate24, active24 = apply(full24, parent24, correction24, float(selected["eta"]))
    locked_result = diagnostics(_metric_frame(full24), parent24, candidate24, active24)
    locked = compact(locked_result)
    locked_gate = config["locked_gate"]
    locked_pass = bool(
        locked["gain"] >= float(locked_gate["gain_min"])
        and locked["positive_month_fraction"] >= float(locked_gate["positive_month_fraction_min"])
        and locked["worst_month_gain"] > float(locked_gate["worst_month_gain_min_exclusive"])
        and locked["active_domain_gain"] > float(locked_gate["active_domain_gain_min"])
    )
    source.to_csv(output_dir / "source_metrics.csv", index=False)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    np.savez_compressed(
        output_dir / "locked_full_2024.npz",
        target=full24[TARGET].to_numpy(np.float64),
        parent=parent24,
        candidate=candidate24,
        correction=correction24,
        active=active24,
        pitcher_id=full24["pitcher_id"].to_numpy(),
        batter_id=full24["batter_id"].to_numpy(),
        game_month=full24["game_month"].to_numpy(np.int16),
        domain3=full24["domain3"].astype(str).to_numpy(),
    )
    summary = {
        "protocol": PROTOCOL,
        "status": "robust_audit_required" if locked_pass else "reject",
        "selected_recipe": selected,
        "selected_source_metrics": source.loc[
            source["alpha"].eq(selected["alpha"]) & source["eta"].eq(selected["eta"])
        ].to_dict("records"),
        "source_gate_passed": bool(selected["source_gate_passed"]),
        "locked_full_2024": locked,
        "locked_point_gate_passed": locked_pass,
        "fit_audits": fit_audits,
        "final_feature_count": int(len(final_model.coefficient)),
        **config["restrictions"],
        "eligible_for_packaging": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-axes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v104_axes, args.config, args.output_dir)


if __name__ == "__main__":
    main()
