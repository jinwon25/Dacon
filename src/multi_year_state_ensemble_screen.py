"""Screen stable ensembles and gates for cached multi-year state predictions.

The screen keeps three notions of evidence separate:

* select on 2022, then report 2023/2024 without retuning;
* select on 2022+2023, then report the 2024 forward holdout;
* an explicitly exploratory 2023/2024 robust upper bound.

All gains are computed analytically from the Brier loss.  Contributions from
the three disjoint competition domains therefore add exactly to the full-fold
Brier-skill gain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
ABS_RANGES = (
    (0.0, np.inf, "allmag"),
    (0.0, 0.005, "lt005"),
    (0.005, 0.020, "005to020"),
    (0.020, 0.040, "020to040"),
    (0.040, np.inf, "ge040"),
    (0.0, 0.020, "lt020"),
    (0.0, 0.040, "lt040"),
)
INCUMBENT_BANDS = (
    (0.0, 1.0, "p_all"),
    (0.2, 0.8, "p_20_80"),
    (0.3, 0.7, "p_30_70"),
    (0.4, 0.6, "p_40_60"),
)
SIGNS = ("all", "positive", "negative")
AGREEMENTS = (0.0, 2.0 / 3.0, 5.0 / 6.0, 1.0)


def _bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(target.mean())
    return float(
        100000.0
        * (1.0 - np.mean(np.square(target - prediction)) / (rate * (1.0 - rate)))
    )


def _load_fold(
    project: Path, variants_dir: Path, selected_dir: Path, year: int
) -> dict[str, object]:
    with np.load(selected_dir / f"selected_state_o{year}.npz", allow_pickle=True) as z:
        selected = {key: z[key] for key in z.files}
    with np.load(variants_dir / f"state_variants_o{year}.npz", allow_pickle=True) as z:
        variants = {key: z[key] for key in z.files}
    for key in ("target", "incumbent", "domain3", "game_month", "pitcher_id"):
        if not np.array_equal(selected[key], variants[key]):
            raise ValueError(f"fold {year}: selected/variant mismatch for {key}")

    individual = {"selected_h05_l15_b075": selected["raw"].astype(np.float64)}
    for key, value in variants.items():
        if key not in {"target", "incumbent", "domain3", "game_month", "pitcher_id"}:
            individual[key] = value.astype(np.float64)
    matrix = np.column_stack(list(individual.values()))
    candidate = dict(individual)
    candidate["mean_all"] = matrix.mean(axis=1)
    candidate["median_all"] = np.median(matrix, axis=1)

    short_names = [name for name in individual if "h05" in name]
    short_matrix = np.column_stack([individual[name] for name in short_names])
    candidate["mean_short"] = short_matrix.mean(axis=1)
    candidate["median_short"] = np.median(short_matrix, axis=1)
    for name in individual:
        if name != "selected_h05_l15_b075":
            candidate[f"pair_selected__{name}"] = 0.5 * (
                individual["selected_h05_l15_b075"] + individual[name]
            )

    incumbent = selected["incumbent"].astype(np.float64)
    individual_correction = matrix - incumbent[:, None]
    return {
        "target": selected["target"].astype(np.float64),
        "incumbent": incumbent,
        "domain3": selected["domain3"].astype(str),
        "game_month": selected["game_month"].astype(np.int16),
        "pitcher_id": selected["pitcher_id"],
        "batter_id": selected["batter_id"],
        "candidate": candidate,
        "individual_correction": individual_correction,
    }


def _agreement(individual_correction: np.ndarray, correction: np.ndarray) -> np.ndarray:
    positive = correction >= 0.0
    same = np.where(
        positive[:, None], individual_correction >= 0.0, individual_correction < 0.0
    )
    return same.mean(axis=1)


def _config_key(row: pd.Series | dict[str, object]) -> str:
    return "|".join(
        str(row[column])
        for column in (
            "raw_name",
            "sign",
            "abs_name",
            "band_name",
            "agreement",
            "weight",
        )
    )


def _screen_fold(year: int, fold: dict[str, object]) -> pd.DataFrame:
    target = fold["target"]
    incumbent = fold["incumbent"]
    domain3 = fold["domain3"]
    error = incumbent - target
    reference = float(target.mean() * (1.0 - target.mean()))
    n_rows = len(target)
    rows: list[dict[str, object]] = []
    for raw_name, raw in fold["candidate"].items():
        correction = raw - incumbent
        magnitude = np.abs(correction)
        agreement = _agreement(fold["individual_correction"], correction)
        for domain in DOMAINS:
            domain_mask = domain3 == domain
            for sign in SIGNS:
                sign_mask = np.ones(n_rows, dtype=bool)
                if sign == "positive":
                    sign_mask = correction > 0.0
                elif sign == "negative":
                    sign_mask = correction < 0.0
                for low_abs, high_abs, abs_name in ABS_RANGES:
                    magnitude_mask = (magnitude >= low_abs) & (magnitude < high_abs)
                    for low_p, high_p, band_name in INCUMBENT_BANDS:
                        probability_mask = (incumbent >= low_p) & (incumbent <= high_p)
                        fixed_mask = domain_mask & sign_mask & magnitude_mask & probability_mask
                        for agreement_min in AGREEMENTS:
                            mask = fixed_mask & (agreement >= agreement_min - 1e-12)
                            count = int(mask.sum())
                            if count == 0:
                                linear = quadratic = 0.0
                            else:
                                selected_correction = correction[mask]
                                linear = float(
                                    np.sum(2.0 * error[mask] * selected_correction) / n_rows
                                )
                                quadratic = float(
                                    np.sum(np.square(selected_correction)) / n_rows
                                )
                            for weight in WEIGHTS:
                                delta_mse = weight * linear + weight * weight * quadratic
                                gain = -100000.0 * delta_mse / reference
                                row = {
                                    "audit_year": year,
                                    "domain": domain,
                                    "raw_name": raw_name,
                                    "sign": sign,
                                    "abs_name": abs_name,
                                    "band_name": band_name,
                                    "agreement": agreement_min,
                                    "weight": weight,
                                    "n_applied": count,
                                    "gain": gain,
                                }
                                row["config"] = _config_key(row)
                                rows.append(row)
    return pd.DataFrame(rows)


def _selection_table(metrics: pd.DataFrame, selection_years: tuple[int, ...]) -> pd.DataFrame:
    source = metrics.loc[metrics["audit_year"].isin(selection_years)]
    summary = (
        source.groupby(["domain", "config"], observed=True)["gain"]
        .agg(selection_mean="mean", selection_min="min")
        .reset_index()
    )
    summary = summary.sort_values(
        ["domain", "selection_mean", "selection_min"],
        ascending=[True, False, False],
        kind="stable",
    )
    chosen = summary.groupby("domain", observed=True).head(1)
    details = metrics.merge(chosen[["domain", "config"]], on=["domain", "config"])
    details["selection_years"] = "+".join(map(str, selection_years))
    return details.sort_values(["audit_year", "domain"], kind="stable")


def _robust_exploratory(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics.loc[metrics["audit_year"].isin((2023, 2024))]
    robust = (
        source.groupby(["domain", "config"], observed=True)["gain"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(
            ["domain", "min_gain", "mean_gain"],
            ascending=[True, False, False],
            kind="stable",
        )
    )
    return robust.groupby("domain", observed=True).head(25)


def run(project: Path, variants_dir: Path, selected_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    variants_dir = (project / variants_dir).resolve()
    selected_dir = (project / selected_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = {
        year: _load_fold(project, variants_dir, selected_dir, year)
        for year in (2022, 2023, 2024)
    }
    metric_parts = []
    for year, fold in folds.items():
        print(f"[ensemble-screen] audit_year={year}", flush=True)
        metric_parts.append(_screen_fold(year, fold))
    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_parquet(output_dir / "metrics.parquet", index=False)

    selected_2022 = _selection_table(metrics, (2022,))
    selected_2022_2023 = _selection_table(metrics, (2022, 2023))
    selected_2022.to_csv(output_dir / "selected_on_2022.csv", index=False)
    selected_2022_2023.to_csv(output_dir / "selected_on_2022_2023.csv", index=False)
    exploratory = _robust_exploratory(metrics)
    exploratory.to_csv(output_dir / "exploratory_robust_2023_2024.csv", index=False)

    def totals(frame: pd.DataFrame) -> list[dict[str, object]]:
        return (
            frame.groupby("audit_year", observed=True)["gain"]
            .sum()
            .rename("routed_gain")
            .reset_index()
            .to_dict(orient="records")
        )

    exploratory_choice = exploratory.groupby("domain", observed=True).head(1)
    exploratory_details = metrics.merge(
        exploratory_choice[["domain", "config"]], on=["domain", "config"]
    )
    summary = {
        "protocol": "MULTI_YEAR_STATE_ENSEMBLE_GATE_SCREEN_V1",
        "candidate_count": int(metrics["config"].nunique()),
        "selection_2022_routed": totals(selected_2022),
        "selection_2022_2023_routed": totals(selected_2022_2023),
        "exploratory_robust_routed": totals(exploratory_details),
        "selected_on_2022": selected_2022.to_dict(orient="records"),
        "selected_on_2022_2023": selected_2022_2023.to_dict(orient="records"),
        "exploratory_choice": exploratory_choice.to_dict(orient="records"),
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
        "--variants-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_variants_20260816_01"),
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_selected_20260816_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_ensemble_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.variants_dir, args.selected_dir, args.output_dir)


if __name__ == "__main__":
    main()
