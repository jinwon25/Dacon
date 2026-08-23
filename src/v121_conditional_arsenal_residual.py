"""Target-free conditional arsenal residual above exact v104 OOF.

Historical aligned TrackMan pitch labels are used only to estimate a pitcher's
count- and batter-hand-conditioned pitch mix before each forecast origin.  The
current audit pitch type and all current-pitch physics are unavailable and are
never read.  A no-intercept ridge maps mix deviations to v104 residuals on two
source transfers; 2024 is opened only after the recipe is frozen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v103_fixed_union_robust import _axis_metrics
from src.v113_fine_pitch_failure_prior_v104 import PITCH_TYPE_NORMALISATION
from src.v97_conditional_direct_forward import _load_contract_axis


PROTOCOL = "V121_CONDITIONAL_ARSENAL_RESIDUAL_V1"
ROUTE = "R_CORE"


def count_state(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["balls_before"].astype("Int64").astype("string")
        + "-"
        + frame["strikes_before"].astype("Int64").astype("string")
    ).astype(str).to_numpy()


def _wide_counts(
    history: pd.DataFrame, keys: list[str], pitch_types: list[str]
) -> pd.DataFrame:
    table = (
        history.groupby([*keys, "pitch_type_fine"], observed=True, sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pitch_types, fill_value=0)
    )
    table.columns = [f"n__{name}" for name in pitch_types]
    return table.reset_index()


def build_arsenal_bank(
    history: pd.DataFrame, pitch_types: list[str], shrink: float
) -> dict[str, Any]:
    if history.empty or float(shrink) <= 0.0:
        raise ValueError("history and positive shrink are required")
    global_mix_series = history["pitch_type_fine"].value_counts(normalize=True)
    global_mix = np.asarray(
        [float(global_mix_series.get(name, 0.0)) for name in pitch_types],
        dtype=np.float64,
    )
    if global_mix.sum() <= 0.0:
        raise ValueError("historical pitch mix is empty")
    global_mix /= global_mix.sum()
    count_columns = [f"n__{name}" for name in pitch_types]
    overall = _wide_counts(history, ["pitcher_id"], pitch_types)
    total = overall[count_columns].sum(axis=1).to_numpy(float)
    for index, name in enumerate(pitch_types):
        overall[f"mix__{name}"] = (
            overall[f"n__{name}"].to_numpy(float) + float(shrink) * global_mix[index]
        ) / (total + float(shrink))
    overall["overall_n"] = total

    def conditional(keys: list[str], label: str) -> pd.DataFrame:
        table = _wide_counts(history, keys, pitch_types)
        local_total = table[count_columns].sum(axis=1).to_numpy(float)
        table = table.merge(
            overall[["pitcher_id", *[f"mix__{name}" for name in pitch_types]]],
            on="pitcher_id", how="left", sort=False, validate="many_to_one",
        )
        for index, name in enumerate(pitch_types):
            backing = table[f"mix__{name}"].fillna(global_mix[index]).to_numpy(float)
            table[f"{label}__mix__{name}"] = (
                table[f"n__{name}"].to_numpy(float) + float(shrink) * backing
            ) / (local_total + float(shrink))
        table[f"{label}__n"] = local_total
        return table[
            [*keys, f"{label}__n", *[f"{label}__mix__{name}" for name in pitch_types]]
        ]

    return {
        "pitch_types": list(pitch_types),
        "global_mix": global_mix,
        "shrink": float(shrink),
        "overall": overall[
            ["pitcher_id", "overall_n", *[f"mix__{name}" for name in pitch_types]]
        ],
        "count": conditional(["pitcher_id", "count_state"], "count"),
        "hand": conditional(["pitcher_id", "batter_hand"], "hand"),
    }


def arsenal_features(rows: pd.DataFrame, bank: dict[str, Any]) -> pd.DataFrame:
    pitch_types = [str(value) for value in bank["pitch_types"]]
    global_mix = np.asarray(bank["global_mix"], dtype=np.float64)
    query = rows[["pitcher_id", "batter_hand"]].reset_index(drop=True).copy()
    query["count_state"] = count_state(rows)
    query["_order"] = np.arange(len(query))
    query = query.merge(
        bank["overall"], on="pitcher_id", how="left", sort=False, validate="many_to_one"
    )
    query = query.merge(
        bank["count"], on=["pitcher_id", "count_state"], how="left", sort=False,
        validate="many_to_one",
    )
    query = query.merge(
        bank["hand"], on=["pitcher_id", "batter_hand"], how="left", sort=False,
        validate="many_to_one",
    )
    output: dict[str, np.ndarray] = {}
    overall_n = query["overall_n"].fillna(0.0).to_numpy(float)
    count_n = query["count__n"].fillna(0.0).to_numpy(float)
    hand_n = query["hand__n"].fillna(0.0).to_numpy(float)
    shrink = float(bank["shrink"])
    output["overall_reliability"] = overall_n / (overall_n + shrink)
    output["count_reliability"] = count_n / (count_n + shrink)
    output["hand_reliability"] = hand_n / (hand_n + shrink)
    for index, name in enumerate(pitch_types):
        overall = query[f"mix__{name}"].fillna(global_mix[index]).to_numpy(float)
        count_raw = query[f"count__mix__{name}"].to_numpy(float)
        hand_raw = query[f"hand__mix__{name}"].to_numpy(float)
        count_mix = np.where(np.isfinite(count_raw), count_raw, overall)
        hand_mix = np.where(np.isfinite(hand_raw), hand_raw, overall)
        output[f"count_delta__{name}"] = count_mix - overall
        output[f"hand_delta__{name}"] = hand_mix - overall
    order = np.argsort(query["_order"].to_numpy())
    frame = pd.DataFrame(output).iloc[order].reset_index(drop=True)
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("conditional arsenal features are non-finite")
    return frame


@dataclass(frozen=True)
class RidgeSpec:
    scale: np.ndarray
    coefficient: np.ndarray


def fit_ridge(
    features: np.ndarray, residual: np.ndarray, alpha: float
) -> RidgeSpec:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(residual, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("ridge arrays are not aligned")
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = x / scale
    coefficient = np.linalg.solve(
        z.T @ z + float(alpha) * np.eye(z.shape[1]), z.T @ y
    )
    return RidgeSpec(scale=scale, coefficient=coefficient)


def predict_ridge(spec: RidgeSpec, features: np.ndarray, cap: float) -> np.ndarray:
    value = (np.asarray(features, dtype=np.float64) / spec.scale) @ spec.coefficient
    return np.clip(value, -float(cap), float(cap))


def apply_correction(
    parent: np.ndarray, domain: np.ndarray, correction: np.ndarray, eta: float
) -> np.ndarray:
    output = np.asarray(parent, dtype=np.float64).copy()
    active = np.asarray(domain).astype(str) == ROUTE
    output[active] = np.clip(
        output[active] + float(eta) * np.asarray(correction, dtype=np.float64)[active],
        0.001, 0.999,
    )
    return output


def _point_pass(metrics: dict[str, Any], gate: dict[str, Any], *, locked: bool) -> bool:
    minimum_gain = float(gate.get("gain_min", 0.0)) if locked else 0.0
    return bool(
        metrics["gain"] >= minimum_gain
        and metrics["positive_month_fraction"] >= float(gate["positive_month_fraction_min"])
        and metrics["worst_month_gain"] > float(gate["worst_month_gain_min_exclusive"])
        and metrics["minimum_domain_gain"] >= float(gate["active_domain_gain_min"])
    )


def _slice_axis(axis: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in axis.items()}


def run(
    train_csv: Path,
    trackman_csv: Path,
    alignment_npz: Path,
    contract_dir: Path,
    v104_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    trackman = pd.read_csv(
        trackman_csv, usecols=["season", "tagged_pitch_type"], low_memory=False
    )
    with np.load(alignment_npz, allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
        aligned_season = saved["season"].astype(np.int16)
    if not np.array_equal(raw.iloc[main_index]["season"].to_numpy(np.int16), aligned_season):
        raise ValueError("main alignment season mismatch")
    if not np.array_equal(trackman.iloc[trackman_index]["season"].to_numpy(np.int16), aligned_season):
        raise ValueError("TrackMan alignment season mismatch")
    pitch_types = [str(value) for value in config["fine_pitch_types"]]
    fine = (
        trackman.iloc[trackman_index]["tagged_pitch_type"].astype(str)
        .replace(PITCH_TYPE_NORMALISATION)
    )
    fine = fine.where(fine.isin(pitch_types[:-1]), pitch_types[-1]).to_numpy(str)
    history = raw.iloc[main_index][
        ["season", "pitcher_id", "batter_hand", "balls_before", "strikes_before"]
    ].reset_index(drop=True)
    history["count_state"] = count_state(history)
    history["pitch_type_fine"] = fine

    exact = {
        name: _load_contract_axis(contract_dir / f"v84_{name}.npz")
        for name in ("full_2022", "late_2023", "full_2024")
    }
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        parent = {name: saved[name].astype(np.float64) for name in saved.files}
    frames = {
        "full_2022": raw.loc[raw["season"].eq(2022)].reset_index(drop=True),
        "late_2023": raw.loc[
            raw["season"].eq(2023) & raw["game_month"].ge(8)
        ].reset_index(drop=True),
        "full_2024": raw.loc[raw["season"].eq(2024)].reset_index(drop=True),
    }
    late24 = exact["full_2024"]["game_month"].astype(np.int16) >= 8
    exact["late_2024"] = _slice_axis(exact["full_2024"], late24)
    frames["late_2024"] = frames["full_2024"].loc[late24].reset_index(drop=True)

    banks = {
        year: build_arsenal_bank(
            history.loc[history["season"].lt(year)].reset_index(drop=True),
            pitch_types,
            float(config["selection_shrink"]),
        )
        for year in (2022, 2023, 2024)
    }
    features = {
        "full_2022": arsenal_features(frames["full_2022"], banks[2022]),
        "late_2023": arsenal_features(frames["late_2023"], banks[2023]),
        "full_2024": arsenal_features(frames["full_2024"], banks[2024]),
    }
    features["late_2024"] = features["full_2024"].loc[late24].reset_index(drop=True)
    early22 = frames["full_2022"]["game_month"].le(7).to_numpy()
    late22 = ~early22
    axis_late22 = _slice_axis(exact["full_2022"], late22)

    trials: list[dict[str, Any]] = []
    trial_candidates: dict[str, dict[str, np.ndarray]] = {}
    trial_metrics: dict[str, dict[str, Any]] = {}
    for alpha in config["ridge_alpha_grid"]:
        active_early = early22 & exact["full_2022"]["exact_mask"].astype(bool) & (
            exact["full_2022"]["domain3"].astype(str) == ROUTE
        )
        spec_early = fit_ridge(
            features["full_2022"].to_numpy(float)[active_early],
            exact["full_2022"]["target"][active_early] - parent["full_2022"][active_early],
            float(alpha),
        )
        correction22 = predict_ridge(
            spec_early, features["full_2022"].to_numpy(float),
            float(config["correction_cap"]),
        )[late22]

        active22 = exact["full_2022"]["exact_mask"].astype(bool) & (
            exact["full_2022"]["domain3"].astype(str) == ROUTE
        )
        spec22 = fit_ridge(
            features["full_2022"].to_numpy(float)[active22],
            exact["full_2022"]["target"][active22] - parent["full_2022"][active22],
            float(alpha),
        )
        correction23 = predict_ridge(
            spec22, features["late_2023"].to_numpy(float),
            float(config["correction_cap"]),
        )
        for eta in config["eta_grid"]:
            key = f"a{float(alpha):g}_e{float(eta):g}"
            candidate_late22 = apply_correction(
                parent["full_2022"][late22], axis_late22["domain3"], correction22,
                float(eta),
            )
            candidate23 = apply_correction(
                parent["late_2023"], exact["late_2023"]["domain3"], correction23,
                float(eta),
            )
            metrics = {
                "early22_to_late22": _axis_metrics(axis_late22, candidate_late22),
                "full22_to_late23": _axis_metrics(exact["late_2023"], candidate23),
            }
            passed = all(
                _point_pass(item, config["source_gate"], locked=False)
                for item in metrics.values()
            )
            trials.append({
                "key": key, "ridge_alpha": float(alpha), "eta": float(eta),
                "source_gate_passed": passed,
                "minimum_gain": min(item["gain"] for item in metrics.values()),
                "minimum_month_fraction": min(
                    item["positive_month_fraction"] for item in metrics.values()
                ),
                "worst_month_gain": min(item["worst_month_gain"] for item in metrics.values()),
            })
            trial_metrics[key] = metrics
            trial_candidates[key] = {
                "late_2022": candidate_late22,
                "late_2023": candidate23,
            }

    ranking = pd.DataFrame(trials).sort_values(
        ["source_gate_passed", "minimum_gain", "worst_month_gain"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    alpha = float(selected["ridge_alpha"])
    eta = float(selected["eta"])
    active23 = exact["late_2023"]["exact_mask"].astype(bool) & (
        exact["late_2023"]["domain3"].astype(str) == ROUTE
    )
    spec23 = fit_ridge(
        features["late_2023"].to_numpy(float)[active23],
        exact["late_2023"]["target"][active23] - parent["late_2023"][active23],
        alpha,
    )
    correction24 = predict_ridge(
        spec23, features["full_2024"].to_numpy(float),
        float(config["correction_cap"]),
    )
    candidate24 = apply_correction(
        parent["full_2024"], exact["full_2024"]["domain3"], correction24, eta
    )
    candidate_late24 = candidate24[late24]
    locked_metrics = {
        "full_2024": _axis_metrics(exact["full_2024"], candidate24),
        "late_2024": _axis_metrics(exact["late_2024"], candidate_late24),
    }
    locked_pass = {
        name: _point_pass(item, config["locked_gate"], locked=True)
        for name, item in locked_metrics.items()
    }
    source_pass = bool(selected["source_gate_passed"])
    eligible = bool(source_pass and all(locked_pass.values()))
    np.savez_compressed(
        output_dir / "selected_axes.npz",
        late_2022=trial_candidates[str(selected["key"])]["late_2022"],
        late_2023=trial_candidates[str(selected["key"])]["late_2023"],
        full_2024=candidate24,
        late_2024=candidate_late24,
    )
    result = {
        "protocol": PROTOCOL,
        "status": "promote_to_robust_audit" if eligible else "reject",
        "aligned_history_rows": int(len(history)),
        "selected": selected,
        "source_ranking": ranking.to_dict(orient="records"),
        "selected_source_metrics": trial_metrics[str(selected["key"])],
        "locked_metrics": locked_metrics,
        "locked_point_gate_pass": locked_pass,
        "eligible_for_robust_audit": eligible,
        "eligible_for_packaging": False,
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
    parser.add_argument("--trackman-csv", type=Path, required=True)
    parser.add_argument("--alignment-npz", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.trackman_csv, args.alignment_npz, args.contract_dir,
        args.v104_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
