"""Forward-only empirical-Bayes residual overlays for the v19 champion.

The screen deliberately uses only temporal transfers:

* 2023 full season -> 2024 full season
* 2023 March-July -> 2023 August-October
* 2024 March-July -> 2024 August-October

Each lookup is learned from ``target - v19_oof`` in the source period and is
then frozen before it is joined to the destination rows.  No destination
target, future row, public score, or TrackMan field enters a prediction.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.core.overlay import WEIGHTS, _bss, _feature_frame


ALPHAS = (50.0, 200.0, 800.0, 3200.0, 12800.0)
DOMAINS = ("ALL", "R_CORE", "R_ANCHOR", "F")


@dataclass(frozen=True)
class GroupSpec:
    name: str
    columns: tuple[str, ...]


GROUPS = (
    GroupSpec("count", ("balls_before", "strikes_before")),
    GroupSpec(
        "count_hands",
        ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
    ),
    GroupSpec("hands", ("pitcher_hand", "batter_hand")),
    GroupSpec("base_count", ("base_state", "balls_before", "strikes_before")),
    GroupSpec(
        "inning_side_outs", ("inning_band", "top_bottom", "outs_before")
    ),
    GroupSpec("li_count", ("li_band", "balls_before", "strikes_before")),
    GroupSpec("pitcher_id", ("pitcher_id",)),
    GroupSpec("pitcher_batter_hand", ("pitcher_id", "batter_hand")),
    GroupSpec("pitcher_pressure", ("pitcher_id", "pressure")),
    GroupSpec(
        "pitcher_hand_pressure", ("pitcher_id", "batter_hand", "pressure")
    ),
    GroupSpec("pitcher_count", ("pitcher_id", "balls_before", "strikes_before")),
    GroupSpec("batter_id", ("batter_id",)),
    GroupSpec("batter_pitcher_hand", ("batter_id", "pitcher_hand")),
    GroupSpec("batter_count", ("batter_id", "balls_before", "strikes_before")),
    GroupSpec("pitcher_team", ("pitcher_team_id",)),
    GroupSpec("batter_team", ("batter_team_id",)),
    GroupSpec("team_matchup", ("pitcher_team_id", "batter_team_id")),
    GroupSpec("pitcher_hand_team", ("pitcher_team_id", "batter_hand")),
    GroupSpec("batter_hand_team", ("batter_team_id", "pitcher_hand")),
)


def _load_folds(project: Path) -> dict[int, pd.DataFrame]:
    columns = sorted(
        {
            "season",
            "game_month",
            "inning",
            "top_bottom",
            "outs_before",
            "balls_before",
            "strikes_before",
            "base_state",
            "li",
            "pitcher_id",
            "batter_id",
            "pitcher_hand",
            "batter_hand",
            "pitcher_team_id",
            "batter_team_id",
            "control_success",
        }
    )
    train = pd.read_csv(project / "data" / "train.csv", usecols=columns)
    artifact = project / "artifacts" / "state_mode_joint_20260816_02"
    folds: dict[int, pd.DataFrame] = {}
    for year in (2023, 2024):
        frame = _feature_frame(train.loc[train["season"].eq(year)].reset_index(drop=True))
        with np.load(artifact / f"joint_candidate_o{year}.npz", allow_pickle=True) as saved:
            target = saved["target"].astype(np.float64)
            if not np.array_equal(
                target, frame["control_success"].to_numpy(np.float64)
            ):
                raise ValueError(f"v19 OOF order mismatch for {year}")
            frame["target"] = target
            frame["v19"] = saved["candidate"].astype(np.float64)
            frame["domain3"] = saved["domain3"].astype(str)
        frame["residual"] = frame["target"] - frame["v19"]
        folds[year] = frame
    return folds


def _mapped_stats(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    columns: tuple[str, ...],
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    fit_mask = np.ones(len(fit), dtype=bool)
    audit_mask = np.ones(len(audit), dtype=bool)
    if domain != "ALL":
        fit_mask &= fit["domain3"].eq(domain).to_numpy()
        audit_mask &= audit["domain3"].eq(domain).to_numpy()
    source = fit.loc[fit_mask]
    source_keys = pd.MultiIndex.from_frame(source.loc[:, list(columns)])
    codes, uniques = pd.factorize(source_keys, sort=False)
    residual = source["residual"].to_numpy(np.float64)
    residual_sum = np.bincount(codes, weights=residual, minlength=len(uniques))
    n = np.bincount(codes, minlength=len(uniques)).astype(np.float64)

    destination = audit.loc[audit_mask, list(columns)]
    destination_keys = pd.MultiIndex.from_frame(destination)
    destination_codes = uniques.get_indexer(destination_keys)
    known = destination_codes >= 0
    positions = np.flatnonzero(audit_mask)[known]
    mapped_sum = np.zeros(len(audit), dtype=np.float64)
    mapped_n = np.zeros(len(audit), dtype=np.float64)
    mapped_sum[positions] = residual_sum[destination_codes[known]]
    mapped_n[positions] = n[destination_codes[known]]
    return mapped_sum, mapped_n


def _correction_from_stats(
    mapped_sum: np.ndarray, mapped_n: np.ndarray, alpha: float
) -> np.ndarray:
    output = np.zeros_like(mapped_sum)
    known = mapped_n > 0.0
    output[known] = mapped_sum[known] / (mapped_n[known] + alpha)
    return output


def _gain(target: np.ndarray, incumbent: np.ndarray, correction: np.ndarray) -> float:
    return _bss(target, np.clip(incumbent + correction, 0.001, 0.999)) - _bss(
        target, incumbent
    )


def _splits(folds: dict[int, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    output = {"y2023_to_y2024": (folds[2023], folds[2024])}
    for year in (2023, 2024):
        frame = folds[year]
        output[f"y{year}_early_to_late"] = (
            frame.loc[frame["game_month"].le(7)].reset_index(drop=True),
            frame.loc[frame["game_month"].ge(8)].reset_index(drop=True),
        )
    return output


def _diagnostics(
    audit: pd.DataFrame, correction: np.ndarray
) -> tuple[float, float, float, float]:
    target = audit["target"].to_numpy(np.float64)
    incumbent = audit["v19"].to_numpy(np.float64)
    candidate = np.clip(incumbent + correction, 0.001, 0.999)
    month_gains = []
    for month in sorted(audit["game_month"].unique()):
        mask = audit["game_month"].eq(month).to_numpy()
        month_gains.append(_bss(target[mask], candidate[mask]) - _bss(target[mask], incumbent[mask]))
    domain_gains = []
    for domain in sorted(audit["domain3"].unique()):
        mask = audit["domain3"].eq(domain).to_numpy()
        domain_gains.append(_bss(target[mask], candidate[mask]) - _bss(target[mask], incumbent[mask]))
    return (
        _gain(target, incumbent, correction),
        float(min(month_gains)),
        float(np.mean(np.asarray(month_gains) > 0.0)),
        float(min(domain_gains)),
    )


def _quadratic_gain_grid(
    audit: pd.DataFrame, correction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact unclipped BSS gains for every configured weight.

    EB corrections are far from the probability boundaries in this screen, so
    ``p + w*c`` never clips.  The Brier difference is therefore an exact
    quadratic in ``w`` and each segment only needs to be scanned once.
    """

    target = audit["target"].to_numpy(np.float64)
    incumbent = audit["v19"].to_numpy(np.float64)
    residual = target - incumbent
    weights = np.asarray(WEIGHTS, dtype=np.float64)

    def segment(mask: np.ndarray) -> np.ndarray:
        segment_target = target[mask]
        segment_residual = residual[mask]
        segment_correction = correction[mask]
        rate = float(np.mean(segment_target))
        scale = 100000.0 / (rate * (1.0 - rate))
        linear = scale * 2.0 * np.mean(segment_residual * segment_correction)
        square = scale * np.mean(segment_correction**2)
        return linear * weights - square * weights**2

    overall = segment(np.ones(len(audit), dtype=bool))
    months = np.vstack(
        [
            segment(audit["game_month"].eq(month).to_numpy())
            for month in sorted(audit["game_month"].unique())
        ]
    )
    domains = np.vstack(
        [
            segment(audit["domain3"].eq(domain).to_numpy())
            for domain in sorted(audit["domain3"].unique())
        ]
    )
    return overall, months.min(axis=0), (months > 0.0).mean(axis=0), domains.min(axis=0)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    folds = _load_folds(project)
    splits = _splits(folds)
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        for domain in DOMAINS:
            split_stats = {
                split_name: _mapped_stats(fit, audit, group.columns, domain)
                for split_name, (fit, audit) in splits.items()
            }
            for alpha in ALPHAS:
                for split_name, (fit, audit) in splits.items():
                    raw = _correction_from_stats(*split_stats[split_name], alpha)
                    gains, worst_months, month_fractions, worst_domains = (
                        _quadratic_gain_grid(audit, raw)
                    )
                    for weight_index, weight in enumerate(WEIGHTS):
                        rows.append(
                            {
                                "group": group.name,
                                "columns": "+".join(group.columns),
                                "domain": domain,
                                "alpha": alpha,
                                "weight": weight,
                                "split": split_name,
                                "gain": float(gains[weight_index]),
                                "worst_month_gain": float(worst_months[weight_index]),
                                "month_positive_fraction": float(
                                    month_fractions[weight_index]
                                ),
                                "worst_domain_gain": float(
                                    worst_domains[weight_index]
                                ),
                            }
                        )
    metrics = pd.DataFrame(rows)
    robust = (
        metrics.groupby(["group", "columns", "domain", "alpha", "weight"], as_index=False)
        .agg(
            min_gain=("gain", "min"),
            mean_gain=("gain", "mean"),
            max_gain=("gain", "max"),
            min_worst_month_gain=("worst_month_gain", "min"),
            min_month_positive_fraction=("month_positive_fraction", "min"),
            min_worst_domain_gain=("worst_domain_gain", "min"),
        )
        .sort_values(["min_gain", "mean_gain"], ascending=False)
        .reset_index(drop=True)
    )

    # Limit the ensemble search to independently useful lookup families.  One
    # best shrinkage level is chosen per family by the three-axis minimum.
    positive = robust.loc[robust["min_gain"] > 0.0]
    family_best = (
        positive.sort_values(["min_gain", "mean_gain"], ascending=False)
        .drop_duplicates(["group", "domain"])
        .head(12)
        .reset_index(drop=True)
    )
    group_by_name = {group.name: group for group in GROUPS}
    correction_bank: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    for choice in family_best.to_dict("records"):
        key = (
            str(choice["group"]),
            str(choice["domain"]),
            float(choice["alpha"]),
        )
        group = group_by_name[key[0]]
        correction_bank[key] = {}
        for split_name, (fit, audit) in splits.items():
            mapped = _mapped_stats(fit, audit, group.columns, key[1])
            correction_bank[key][split_name] = _correction_from_stats(
                *mapped, key[2]
            )
    family_records = family_best.to_dict("records")
    family_keys = [
        (str(row["group"]), str(row["domain"]), float(row["alpha"]))
        for row in family_records
    ]
    quadratic: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, (_, audit) in splits.items():
        signals = np.column_stack(
            [correction_bank[key][split_name] for key in family_keys]
        )
        target = audit["target"].to_numpy(np.float64)
        incumbent = audit["v19"].to_numpy(np.float64)
        residual = target - incumbent
        rate = float(np.mean(target))
        scale = 100000.0 / (rate * (1.0 - rate))
        linear = scale * 2.0 * np.mean(residual[:, None] * signals, axis=0)
        cross = scale * (signals.T @ signals) / len(signals)
        quadratic[split_name] = (linear, cross)
    ensemble_rows: list[dict[str, object]] = []
    ensemble_choices: list[tuple[int, ...]] = []
    for size in (1, 2, 3):
        ensemble_choices.extend(itertools.combinations(range(len(family_records)), size))
    ensemble_weights = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00)
    for choice_indices in ensemble_choices:
        choices = tuple(family_records[index] for index in choice_indices)
        index = np.asarray(choice_indices, dtype=np.int64)
        for weights in itertools.product(ensemble_weights, repeat=len(choice_indices)):
            weight_array = np.asarray(weights, dtype=np.float64)
            split_gains: dict[str, float] = {}
            for split_name in splits:
                linear, cross = quadratic[split_name]
                split_gains[split_name] = float(
                    linear[index] @ weight_array
                    - weight_array @ cross[np.ix_(index, index)] @ weight_array
                )
            ensemble_rows.append(
                {
                    "members": ";".join(
                        f"{row['group']}|{row['domain']}|a{float(row['alpha']):g}|w{weight:g}"
                        for row, weight in zip(choices, weights, strict=True)
                    ),
                    **split_gains,
                    "min_gain": min(split_gains.values()),
                    "mean_gain": float(np.mean(list(split_gains.values()))),
                }
            )
    ensembles = pd.DataFrame(ensemble_rows).sort_values(
        ["min_gain", "mean_gain"], ascending=False
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust.to_csv(output_dir / "robust.csv", index=False)
    family_best.to_csv(output_dir / "family_best.csv", index=False)
    ensembles.to_csv(output_dir / "ensembles.csv", index=False)
    summary = {
        "protocol": "V19_FORWARD_EB_RESIDUAL_OVERLAY_V1",
        "splits": list(splits),
        "n_individual_candidates": int(len(robust)),
        "n_ensemble_candidates": int(len(ensembles)),
        "best_individual": robust.head(20).to_dict("records"),
        "best_ensemble": ensembles.head(20).to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v20_residual_overlay_20260816_01"),
    )
    args = parser.parse_args()
    summary = run(args.project, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
