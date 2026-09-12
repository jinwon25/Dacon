"""Strict next-season audit for a temporal-stable conditional residual.

The candidate is deliberately narrow.  It protects F and R_ANCHOR and learns
only on R_CORE.  For source season ``s`` its 25 row-local features are built
from season ``s - 1`` target summaries.  The fitted Ridge is then applied to
season ``s + 1`` using summaries from season ``s``.  Consequently no target
from the audit season can enter its features or fit.

The 24 conditional features are six uncertainty-aware EB statistics at four
levels: pitcher, pitcher x batter hand, pitcher x pressure, and pitcher x
pressure x batter hand.  The final feature is the protected base probability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ANCHOR_TEAM = 13
BETA = 0.5534087425546688
ADVANCED_ETA = 0.90
GAMMAS = (0.55, 0.65, 0.75, 0.85, 0.95)
ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
FROZEN_ALPHA = 10_000.0
FROZEN_GAMMA = 0.75
STAT_NAMES = ("rate", "rel", "delta", "post_sd", "log_n", "delta_rel")
FEATURE_PREFIXES = ("pitcher", "pitcher_hand", "pitcher_pressure", "pitcher_pressure_hand")


@dataclass(frozen=True)
class Paths:
    corrected_cb: Path
    advanced: Path
    legacy_root: Path


def _pressure(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].to_numpy()
    strikes = frame["strikes_before"].to_numpy()
    return np.where(balls == 3, "threeball", np.where(strikes == 2, "twostrike", "normal"))


def _add_domain_and_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    regular = out["game_type"].eq("R")
    anchor = (out["pitcher_team_id"] == ANCHOR_TEAM) | (
        out["batter_team_id"] == ANCHOR_TEAM
    )
    out["domain3"] = np.where(
        ~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE")
    )
    out["pressure"] = _pressure(out)
    return out


def _bss(target: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(target))
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(prediction - target)))
    return 100_000.0 * (1.0 - brier / reference)


def _find_legacy_file(root: Path, year: int) -> Path:
    matches = sorted(root.glob(f"legacy_cb_diversity_o{str(year)[-2:]}_*/legacy_baseline_s42_o{year}.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one legacy OOF file for {year}, found {matches}")
    return matches[0]


def _load_base(train: pd.DataFrame, paths: Paths, year: int) -> pd.DataFrame:
    with np.load(paths.corrected_cb / f"corrected_cb_o{year}.npz", allow_pickle=True) as saved:
        row_id = saved["row_id"].astype(str)
        target = saved["target"].astype(np.float64)
        v2 = saved["v2"].astype(np.float64)
        game_type = saved["game_type"].astype(str)
    with np.load(paths.advanced / f"correction_compact_l15_o{year}.npz", allow_pickle=True) as saved:
        if not np.array_equal(row_id, saved["row_id"].astype(str)):
            raise ValueError(f"advanced OOF row mismatch for {year}")
        advanced = saved["correction"].astype(np.float64)
    with np.load(_find_legacy_file(paths.legacy_root, year), allow_pickle=True) as saved:
        if not np.array_equal(row_id, saved["row_id"].astype(str)):
            raise ValueError(f"legacy OOF row mismatch for {year}")
        legacy = saved["prediction"].astype(np.float64)

    prediction = v2.copy()
    regular = game_type == "R"
    prediction[regular] = np.clip(
        prediction[regular] + ADVANCED_ETA * advanced[regular], 1e-6, 1.0 - 1e-6
    )
    prediction[regular] = np.clip(
        prediction[regular] + BETA * (legacy[regular] - v2[regular]),
        1e-6,
        1.0 - 1e-6,
    )

    train_index = pd.Index(train["row_id"].astype(str)).get_indexer(row_id)
    if (train_index < 0).any():
        raise ValueError(f"OOF row missing in train for {year}")
    expected = train.iloc[train_index]["control_success"].to_numpy(np.float64)
    if not np.array_equal(target, expected):
        raise ValueError(f"OOF target mismatch for {year}")
    return pd.DataFrame(
        {
            "train_index": train_index,
            "target": target,
            "base": prediction,
            "game_type": game_type,
        }
    )


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, observed=True, sort=False)["control_success"]
        .agg([("success", "sum"), ("n", "size")])
        .reset_index()
    )


def _eb_table(
    history: pd.DataFrame,
    keys: list[str],
    parent: pd.DataFrame | None,
    parent_keys: list[str] | None,
    strength: float,
    global_rate: float,
) -> pd.DataFrame:
    table = _aggregate(history, keys)
    if parent is None:
        table["parent_rate"] = global_rate
    else:
        assert parent_keys is not None
        table = table.merge(
            parent[parent_keys + ["rate"]].rename(columns={"rate": "parent_rate"}),
            on=parent_keys,
            how="left",
            validate="many_to_one",
        )
        table["parent_rate"] = table["parent_rate"].fillna(global_rate)
    table["rate"] = (
        table["success"] + strength * table["parent_rate"]
    ) / (table["n"] + strength)
    table["rel"] = table["n"] / (table["n"] + strength)
    table["delta"] = table["rate"] - table["parent_rate"]
    table["post_sd"] = np.sqrt(
        np.clip(table["rate"] * (1.0 - table["rate"]) / (table["n"] + strength + 1.0), 0.0, None)
    )
    table["log_n"] = np.log1p(table["n"])
    table["delta_rel"] = table["delta"] * table["rel"]
    return table[keys + list(STAT_NAMES)]


def build_bank(history: pd.DataFrame) -> dict[str, object]:
    history = history.loc[history["domain3"].eq("R_CORE")].copy()
    if history.empty:
        raise ValueError("empty R_CORE history")
    global_rate = float(history["control_success"].mean())
    pitcher = _eb_table(history, ["pitcher_id"], None, None, 100.0, global_rate)
    pitcher_hand = _eb_table(
        history,
        ["pitcher_id", "batter_hand"],
        pitcher,
        ["pitcher_id"],
        38.0,
        global_rate,
    )
    pitcher_pressure = _eb_table(
        history,
        ["pitcher_id", "pressure"],
        pitcher,
        ["pitcher_id"],
        30.0,
        global_rate,
    )
    pitcher_pressure_hand = _eb_table(
        history,
        ["pitcher_id", "pressure", "batter_hand"],
        pitcher_pressure,
        ["pitcher_id", "pressure"],
        30.0,
        global_rate,
    )
    return {
        "global_rate": global_rate,
        "pitcher": pitcher,
        "pitcher_hand": pitcher_hand,
        "pitcher_pressure": pitcher_pressure,
        "pitcher_pressure_hand": pitcher_pressure_hand,
    }


def _lookup(
    rows: pd.DataFrame,
    table: pd.DataFrame,
    keys: list[str],
    prefix: str,
    global_rate: float,
) -> pd.DataFrame:
    merged = rows[keys].reset_index(drop=True).merge(
        table, on=keys, how="left", sort=False, validate="many_to_one"
    )
    output = pd.DataFrame(index=np.arange(len(rows)))
    for name in STAT_NAMES:
        fill = global_rate if name == "rate" else 0.0
        output[f"{prefix}__{name}"] = merged[name].fillna(fill).to_numpy(np.float64)
    return output


def build_features(rows: pd.DataFrame, base: np.ndarray, bank: dict[str, object]) -> pd.DataFrame:
    global_rate = float(bank["global_rate"])
    blocks = [
        _lookup(rows, bank["pitcher"], ["pitcher_id"], "pitcher", global_rate),
        _lookup(
            rows,
            bank["pitcher_hand"],
            ["pitcher_id", "batter_hand"],
            "pitcher_hand",
            global_rate,
        ),
        _lookup(
            rows,
            bank["pitcher_pressure"],
            ["pitcher_id", "pressure"],
            "pitcher_pressure",
            global_rate,
        ),
        _lookup(
            rows,
            bank["pitcher_pressure_hand"],
            ["pitcher_id", "pressure", "batter_hand"],
            "pitcher_pressure_hand",
            global_rate,
        ),
    ]
    features = pd.concat(blocks, axis=1)
    features["base_probability"] = np.asarray(base, dtype=np.float64)
    if features.shape[1] != 25 or not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"invalid stable feature matrix {features.shape}")
    return features


def _paired_cluster_bootstrap(
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = 3000,
    seed: int = 20260815,
) -> dict[str, float]:
    reference = float(np.mean(target)) * (1.0 - float(np.mean(target)))
    row_gain = (np.square(base - target) - np.square(candidate - target)) * (100_000.0 / reference)
    codes, unique = pd.factorize(pd.Series(clusters), sort=True)
    sums = np.bincount(codes, weights=row_gain, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 200):
        stop = min(n_resamples, start + 200)
        sample = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        values[start:stop] = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p95": float(np.quantile(values, 0.95)),
        "prob_positive": float(np.mean(values > 0.0)),
        "n_pitchers": int(len(unique)),
        "n_resamples": int(n_resamples),
    }


def _direct_residual_views(
    source_rows: pd.DataFrame,
    source_residual: np.ndarray,
    audit_rows: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Past-only hierarchical residual lookups used as an independent audit."""
    work = source_rows[["pitcher_id", "batter_hand", "pressure"]].copy()
    work["residual"] = np.asarray(source_residual, dtype=np.float64)
    global_mean = float(work["residual"].mean())

    def table(keys: list[str], strength: float, parent: pd.DataFrame | None = None, parent_keys: list[str] | None = None) -> pd.DataFrame:
        grouped = work.groupby(keys, observed=True, sort=False)["residual"].agg(["sum", "size"]).reset_index()
        if parent is None:
            grouped["parent"] = global_mean
        else:
            assert parent_keys is not None
            grouped = grouped.merge(
                parent[parent_keys + ["value"]].rename(columns={"value": "parent"}),
                on=parent_keys,
                how="left",
                validate="many_to_one",
            )
            grouped["parent"] = grouped["parent"].fillna(global_mean)
        grouped["value"] = (grouped["sum"] + strength * grouped["parent"]) / (grouped["size"] + strength)
        return grouped[keys + ["value"]]

    pitcher = table(["pitcher_id"], 200.0)
    hand = table(["pitcher_id", "batter_hand"], 80.0, pitcher, ["pitcher_id"])
    pressure = table(["pitcher_id", "pressure"], 80.0, pitcher, ["pitcher_id"])
    pressure_hand = table(
        ["pitcher_id", "pressure", "batter_hand"],
        50.0,
        pressure,
        ["pitcher_id", "pressure"],
    )

    def lookup(tbl: pd.DataFrame, keys: list[str]) -> np.ndarray:
        return (
            audit_rows[keys]
            .reset_index(drop=True)
            .merge(tbl, on=keys, how="left", sort=False, validate="many_to_one")["value"]
            .fillna(global_mean)
            .to_numpy(np.float64)
        )

    p = lookup(pitcher, ["pitcher_id"])
    h = lookup(hand, ["pitcher_id", "batter_hand"])
    pr = lookup(pressure, ["pitcher_id", "pressure"])
    ph = lookup(pressure_hand, ["pitcher_id", "pressure", "batter_hand"])
    return {
        "intercept": np.full(len(audit_rows), global_mean, dtype=np.float64),
        "pitcher": p,
        "pitcher_hand": h,
        "pitcher_pressure": pr,
        "pitcher_pressure_hand": ph,
        "hand_pressure_mean": 0.5 * (h + ph),
    }


def run(project: Path, research_project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    research_project = research_project.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = research_project / "artifacts" / "top10_20260814"
    paths = Paths(
        corrected_cb=artifact_root / "corrected_cb_oof_20260814_01",
        advanced=artifact_root / "advanced_domain_residual_20260814_01",
        legacy_root=artifact_root,
    )
    columns = [
        "row_id",
        "season",
        "game_month",
        "inning",
        "base_state",
        "li",
        "game_type",
        "balls_before",
        "strikes_before",
        "pitcher_id",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "control_success",
    ]
    train = pd.read_csv(project / "data" / "train.csv", usecols=columns, low_memory=False)
    train = _add_domain_and_pressure(train)
    bases = {year: _load_base(train, paths, year) for year in (2022, 2023, 2024)}
    banks = {
        year: build_bank(train.loc[train["season"].eq(year)]) for year in (2021, 2022, 2023)
    }

    domain_bias_rows: list[dict[str, object]] = []
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source_fold = bases[source_year]
        audit_fold = bases[audit_year]
        source_raw = train.iloc[source_fold["train_index"].to_numpy(np.int64)].reset_index(drop=True)
        audit_raw = train.iloc[audit_fold["train_index"].to_numpy(np.int64)].reset_index(drop=True)
        routed = audit_fold["base"].to_numpy(np.float64, copy=True)
        for domain in ("R_CORE", "R_ANCHOR", "F"):
            source_mask = source_raw["domain3"].eq(domain).to_numpy()
            audit_mask = audit_raw["domain3"].eq(domain).to_numpy()
            bias = float(
                np.mean(
                    source_fold.loc[source_mask, "target"].to_numpy()
                    - source_fold.loc[source_mask, "base"].to_numpy()
                )
            )
            domain_candidate = np.clip(
                audit_fold.loc[audit_mask, "base"].to_numpy() + bias, 1e-6, 1.0 - 1e-6
            )
            routed[audit_mask] = domain_candidate
            domain_bias_rows.append(
                {
                    "source_year": source_year,
                    "audit_year": audit_year,
                    "domain": domain,
                    "n_rows": int(audit_mask.sum()),
                    "source_bias": bias,
                    "audit_bias": float(
                        np.mean(
                            audit_fold.loc[audit_mask, "target"].to_numpy()
                            - audit_fold.loc[audit_mask, "base"].to_numpy()
                        )
                    ),
                    "gain": _bss(
                        audit_fold.loc[audit_mask, "target"].to_numpy(), domain_candidate
                    )
                    - _bss(
                        audit_fold.loc[audit_mask, "target"].to_numpy(),
                        audit_fold.loc[audit_mask, "base"].to_numpy(),
                    ),
                }
            )
        domain_bias_rows.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "domain": "ALL_ROUTED",
                "n_rows": len(routed),
                "source_bias": np.nan,
                "audit_bias": float(
                    np.mean(audit_fold["target"].to_numpy() - audit_fold["base"].to_numpy())
                ),
                "gain": _bss(audit_fold["target"].to_numpy(), routed)
                - _bss(audit_fold["target"].to_numpy(), audit_fold["base"].to_numpy()),
            }
        )
    pd.DataFrame(domain_bias_rows).to_csv(output_dir / "domain_bias_transfer.csv", index=False)

    prepared: dict[int, dict[str, object]] = {}
    for year in (2022, 2023, 2024):
        fold = bases[year]
        raw = train.iloc[fold["train_index"].to_numpy(np.int64)].reset_index(drop=True)
        core = raw["domain3"].eq("R_CORE").to_numpy()
        prepared[year] = {
            "rows": raw.loc[core].reset_index(drop=True),
            "target": fold.loc[core, "target"].to_numpy(np.float64),
            "base": fold.loc[core, "base"].to_numpy(np.float64),
        }
        prepared[year]["features"] = build_features(
            prepared[year]["rows"], prepared[year]["base"], banks[year - 1]
        )

    metric_rows: list[dict[str, object]] = []
    corrections: dict[tuple[int, int, float], np.ndarray] = {}
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source = prepared[source_year]
        audit = prepared[audit_year]
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(source["features"])
        audit_x = scaler.transform(audit["features"])
        residual = source["target"] - source["base"]
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            model.fit(fit_x, residual)
            correction = np.clip(model.predict(audit_x), -0.08, 0.08)
            corrections[(source_year, audit_year, alpha)] = correction
            for gamma in GAMMAS:
                candidate = np.clip(audit["base"] + gamma * correction, 1e-6, 1.0 - 1e-6)
                metric_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "alpha": alpha,
                        "gamma": gamma,
                        "n_rows": len(candidate),
                        "base_bss": _bss(audit["target"], audit["base"]),
                        "candidate_bss": _bss(audit["target"], candidate),
                        "gain": _bss(audit["target"], candidate) - _bss(audit["target"], audit["base"]),
                        "correction_mean": float(correction.mean()),
                        "correction_std": float(correction.std()),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "temporal_grid.csv", index=False)

    direct_rows: list[dict[str, object]] = []
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source = prepared[source_year]
        audit = prepared[audit_year]
        views = _direct_residual_views(
            source["rows"], source["target"] - source["base"], audit["rows"]
        )
        for name, correction in views.items():
            for gamma_direct in (0.25, 0.50, 0.75, 1.00):
                prediction = np.clip(
                    audit["base"] + gamma_direct * correction, 1e-6, 1.0 - 1e-6
                )
                direct_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "view": name,
                        "gamma": gamma_direct,
                        "gain": _bss(audit["target"], prediction)
                        - _bss(audit["target"], audit["base"]),
                        "correction_mean": float(correction.mean()),
                        "correction_std": float(correction.std()),
                    }
                )
    pd.DataFrame(direct_rows).to_csv(output_dir / "direct_residual_grid.csv", index=False)

    combo_rows: list[dict[str, object]] = []
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source = prepared[source_year]
        audit = prepared[audit_year]
        ridge_correction = corrections[(source_year, audit_year, FROZEN_ALPHA)]
        centered = ridge_correction - float(ridge_correction.mean())
        source_bias = float(np.mean(source["target"] - source["base"]))
        for intercept_gamma in (0.75, 1.00):
            for conditional_gamma in (0.55, 0.75, 0.95):
                correction = intercept_gamma * source_bias + conditional_gamma * centered
                prediction = np.clip(audit["base"] + correction, 1e-6, 1.0 - 1e-6)
                combo_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "intercept_gamma": intercept_gamma,
                        "conditional_gamma": conditional_gamma,
                        "gain": _bss(audit["target"], prediction)
                        - _bss(audit["target"], audit["base"]),
                        "source_bias": source_bias,
                        "correction_mean": float(correction.mean()),
                        "correction_std": float(correction.std()),
                    }
                )
    pd.DataFrame(combo_rows).to_csv(output_dir / "bias_centered_ridge_grid.csv", index=False)

    context_rows: list[dict[str, object]] = []
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source = prepared[source_year]
        audit = prepared[audit_year]

        def row_context(rows: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
            z = pd.DataFrame(index=np.arange(len(rows)))
            z["base_probability"] = base
            z["count"] = rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str)
            z["pressure_hand"] = rows["pressure"].astype(str) + "-" + rows["batter_hand"].astype(str)
            z["month"] = rows["game_month"].astype(str)
            z["inning_phase"] = pd.cut(
                rows["inning"], [0, 3, 6, 9, np.inf], labels=["early", "middle", "late", "extra"]
            ).astype(str)
            z["base_state"] = rows["base_state"].astype(str)
            z["li_bucket"] = pd.cut(
                rows["li"], [-np.inf, 0.75, 1.5, 3.0, np.inf], labels=["low", "medium", "high", "very_high"]
            ).astype(str)
            return z

        a = row_context(source["rows"], source["base"])
        b = row_context(audit["rows"], audit["base"])
        both = pd.concat([a, b], ignore_index=True)
        matrix = pd.get_dummies(both, columns=[c for c in both.columns if c != "base_probability"], dtype=np.float64)
        fit_x = matrix.iloc[: len(a)].to_numpy(np.float64)
        audit_x = matrix.iloc[len(a) :].to_numpy(np.float64)
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(fit_x)
        audit_x = scaler.transform(audit_x)
        model = Ridge(alpha=10_000.0, solver="cholesky")
        model.fit(fit_x, source["target"] - source["base"])
        correction = np.clip(model.predict(audit_x), -0.08, 0.08)
        for gamma_context in (0.55, 0.75, 1.00):
            prediction = np.clip(
                audit["base"] + gamma_context * correction, 1e-6, 1.0 - 1e-6
            )
            context_rows.append(
                {
                    "source_year": source_year,
                    "audit_year": audit_year,
                    "gamma": gamma_context,
                    "feature_count": fit_x.shape[1],
                    "gain": _bss(audit["target"], prediction)
                    - _bss(audit["target"], audit["base"]),
                    "correction_mean": float(correction.mean()),
                    "correction_std": float(correction.std()),
                }
            )
    pd.DataFrame(context_rows).to_csv(output_dir / "row_context_ridge_grid.csv", index=False)

    state_columns = [
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    state_rows: list[dict[str, object]] = []
    state_corrections: dict[tuple[int, int, float], np.ndarray] = {}
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source = prepared[source_year]
        audit = prepared[audit_year]

        def state_matrix(rows: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
            z = rows[state_columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
            z["log_pitcher_n"] = np.log1p(pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").clip(lower=0)).to_numpy()
            z["log_batter_n"] = np.log1p(pd.to_numeric(rows["asof_batter_n"], errors="coerce").clip(lower=0)).to_numpy()
            z["log_pitchmix_n"] = np.log1p(pd.to_numeric(rows["asof_pitcher_pitchmix_n"], errors="coerce").clip(lower=0)).to_numpy()
            z["recent_1_minus_5"] = z["asof_pitcher_prev1_game_success_rate"] - z["asof_pitcher_prev5_game_success_rate"]
            z["recent_3_minus_career"] = z["asof_pitcher_prev3_game_success_rate"] - z["asof_pitcher_success_rate"]
            z["pitcher_batter_gap"] = z["asof_pitcher_success_rate"] - z["asof_batter_success_rate"]
            z["strike_minus_ball"] = z["asof_pitcher_strike_rate"] - z["asof_pitcher_ball_rate"]
            z["base_probability"] = base
            return z.replace([np.inf, -np.inf], np.nan)

        a = state_matrix(source["rows"], source["base"])
        b = state_matrix(audit["rows"], audit["base"])
        med = a.median().fillna(0.0)
        fit_raw = a.fillna(med).to_numpy(np.float64)
        audit_raw = b.fillna(med).to_numpy(np.float64)
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(fit_raw)
        audit_x = scaler.transform(audit_raw)
        for alpha_state in (10_000.0, 50_000.0, 100_000.0):
            model = Ridge(alpha=alpha_state, solver="cholesky")
            model.fit(fit_x, source["target"] - source["base"])
            correction = np.clip(model.predict(audit_x), -0.08, 0.08)
            state_corrections[(source_year, audit_year, alpha_state)] = correction
            for gamma_state in (0.55, 0.75, 1.00):
                prediction = np.clip(
                    audit["base"] + gamma_state * correction, 1e-6, 1.0 - 1e-6
                )
                state_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "alpha": alpha_state,
                        "gamma": gamma_state,
                        "feature_count": fit_x.shape[1],
                        "gain": _bss(audit["target"], prediction)
                        - _bss(audit["target"], audit["base"]),
                        "correction_mean": float(correction.mean()),
                        "correction_std": float(correction.std()),
                    }
                )
    pd.DataFrame(state_rows).to_csv(output_dir / "row_state_ridge_grid.csv", index=False)

    ensemble_rows: list[dict[str, object]] = []
    for source_year, audit_year in ((2022, 2023), (2023, 2024)):
        source = prepared[source_year]
        audit = prepared[audit_year]
        bias = float(np.mean(source["target"] - source["base"]))
        stable = corrections[(source_year, audit_year, FROZEN_ALPHA)]
        stable = stable - float(stable.mean())
        state = state_corrections[(source_year, audit_year, 100_000.0)]
        state = state - float(state.mean())
        for stable_weight in (0.35, 0.55):
            for state_weight in (0.25, 0.45):
                correction = bias + stable_weight * stable + state_weight * state
                prediction = np.clip(audit["base"] + correction, 1e-6, 1.0 - 1e-6)
                ensemble_rows.append(
                    {
                        "source_year": source_year,
                        "audit_year": audit_year,
                        "stable_weight": stable_weight,
                        "state_weight": state_weight,
                        "gain": _bss(audit["target"], prediction)
                        - _bss(audit["target"], audit["base"]),
                        "correction_mean": float(correction.mean()),
                        "correction_std": float(correction.std()),
                    }
                )
    pd.DataFrame(ensemble_rows).to_csv(output_dir / "centered_ensemble_grid.csv", index=False)

    # This is a genuinely frozen prior, not selected on either target season:
    # strong standardized Ridge alpha=10,000 and gamma=.75 were specified by
    # the public temporal-stable protocol.  The complete grid is diagnostic.
    alpha = FROZEN_ALPHA
    gamma = FROZEN_GAMMA
    choice = metrics.loc[
        metrics["audit_year"].eq(2023)
        & metrics["alpha"].eq(alpha)
        & metrics["gamma"].eq(gamma)
    ].iloc[0]
    confirmation = metrics.loc[
        metrics["audit_year"].eq(2024)
        & metrics["alpha"].eq(alpha)
        & metrics["gamma"].eq(gamma)
    ].iloc[0]
    audit = prepared[2024]
    correction = corrections[(2023, 2024, alpha)]
    candidate = np.clip(audit["base"] + gamma * correction, 1e-6, 1.0 - 1e-6)
    bootstrap = _paired_cluster_bootstrap(
        audit["target"],
        audit["base"],
        candidate,
        audit["rows"]["pitcher_id"].to_numpy(),
    )
    month_rows = []
    for month, index in audit["rows"].groupby("game_month", observed=True).groups.items():
        idx = np.asarray(list(index), dtype=np.int64)
        month_rows.append(
            {
                "game_month": int(month),
                "n_rows": len(idx),
                "base_bss": _bss(audit["target"][idx], audit["base"][idx]),
                "candidate_bss": _bss(audit["target"][idx], candidate[idx]),
            }
        )
    months = pd.DataFrame(month_rows)
    months["gain"] = months["candidate_bss"] - months["base_bss"]
    months.to_csv(output_dir / "confirmation_months.csv", index=False)

    summary = {
        "protocol": "TEMPORAL_STABLE_HAND_PRESSURE_COUNT_RIDGE_V1_INDEPENDENT",
        "anchor_team": ANCHOR_TEAM,
        "feature_count": 25,
        "selection_rule": "external-prior frozen alpha=10000 and gamma=0.75; grid diagnostic only",
        "selected_alpha": alpha,
        "selected_gamma": gamma,
        "discovery_2022_to_2023_gain": float(choice["gain"]),
        "confirmation_2023_to_2024_gain": float(confirmation["gain"]),
        "confirmation_base_bss": float(confirmation["base_bss"]),
        "confirmation_candidate_bss": float(confirmation["candidate_bss"]),
        "bootstrap": bootstrap,
        "months_positive": int((months["gain"] > 0).sum()),
        "months_total": int(len(months)),
        "worst_month_gain": float(months["gain"].min()),
        "mean_month_gain": float(months["gain"].mean()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--research-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.research_project, args.output_dir)


if __name__ == "__main__":
    main()
