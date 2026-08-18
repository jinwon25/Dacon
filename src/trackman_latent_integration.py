"""Integrate a privileged TrackMan teacher over past pitcher pitch profiles.

At audit inference no current-pitch TrackMan value is available.  Instead, the
full teacher is evaluated once per pitch family using pitcher x family means
from TrackMan seasons no later than the source year.  The four predictions are
averaged with row-local pitch-mix weights.  This approximates E[p(y|x,z)|x]
without using any audit-season TrackMan row.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.trackman_privileged_distillation import (
    CATEGORICAL_COLUMNS,
    PHYSICAL_COLUMNS,
    V17_NAME,
    _diagnostics,
    _model,
    _prepare_frames,
)


TRANSITIONS = ((2022, 2023), (2023, 2024))
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
NUMERIC_PHYSICAL = tuple(column for column in PHYSICAL_COLUMNS if column != "pitch_type_group")


def _profile_tables(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    main_index: np.ndarray,
    trackman_index: np.ndarray,
    season: np.ndarray,
    cutoff: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    mask = season <= cutoff
    profile = trackman.iloc[trackman_index[mask]].reset_index(drop=True).copy()
    profile["pitcher_id"] = main.iloc[main_index[mask]]["pitcher_id"].to_numpy()
    profile["pitch_type_group"] = profile["pitch_type_group"].astype("string")
    tables: dict[str, pd.DataFrame] = {}
    for group in PITCH_GROUPS:
        subset = profile.loc[profile["pitch_type_group"].eq(group)]
        tables[group] = subset.groupby("pitcher_id", observed=True)[list(NUMERIC_PHYSICAL)].mean()
    global_means = profile.groupby("pitch_type_group", observed=True)[list(NUMERIC_PHYSICAL)].mean()
    counts = (
        profile.groupby(["pitcher_id", "pitch_type_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PITCH_GROUPS, fill_value=0)
    )
    return tables, global_means, counts


def _pitch_weights(rows: pd.DataFrame, counts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    historical = rows[["pitcher_id"]].merge(
        counts.reset_index(), on="pitcher_id", how="left", validate="many_to_one"
    )
    count_values = historical[list(PITCH_GROUPS)].fillna(0.0).to_numpy(np.float64)
    total = count_values.sum(axis=1)
    other = np.divide(
        count_values[:, 3],
        total,
        out=np.zeros(len(rows), dtype=np.float64),
        where=total > 0,
    )
    row_mix = rows[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    row_sum = np.nansum(row_mix, axis=1)
    fallback = count_values[:, :3]
    fallback_sum = fallback.sum(axis=1)
    fallback = np.divide(
        fallback,
        fallback_sum[:, None],
        out=np.full_like(fallback, 1.0 / 3.0),
        where=fallback_sum[:, None] > 0,
    )
    row_mix = np.divide(
        np.nan_to_num(row_mix, nan=0.0),
        row_sum[:, None],
        out=fallback.copy(),
        where=row_sum[:, None] > 0,
    )
    weights = np.column_stack(((1.0 - other)[:, None] * row_mix, other))
    reliability = total / (total + 200.0)
    return weights, reliability


def _fill_group_profile(
    audit_x: pd.DataFrame,
    audit_rows: pd.DataFrame,
    group: str,
    table: pd.DataFrame,
    global_means: pd.DataFrame,
) -> pd.DataFrame:
    output = audit_x.copy()
    lookup = audit_rows[["pitcher_id"]].merge(
        table.reset_index(), on="pitcher_id", how="left", validate="many_to_one"
    )
    for column in NUMERIC_PHYSICAL:
        fallback = float(global_means.loc[group, column])
        output[f"tm_{column}"] = (
            pd.to_numeric(lookup[column], errors="coerce")
            .fillna(fallback)
            .to_numpy(np.float32)
        )
    column = "tm_pitch_type_group"
    categories = audit_x[column].cat.categories
    output[column] = pd.Categorical(
        np.repeat(group, len(output)), categories=categories
    )
    return output


def run(project: Path, alignment_dir: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = (project / alignment_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = np.load(alignment_dir / "pitch_alignment.npz")
    aligned_main_index = alignment["main_index"].astype(np.int64)
    aligned_trackman_index = alignment["trackman_index"].astype(np.int64)
    aligned_season = alignment["season"].astype(np.int16)
    main = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=PHYSICAL_COLUMNS,
        low_memory=False,
    )
    safe_columns = [c for c in main.columns if c not in {"row_id", "control_success"}]
    full_columns = safe_columns + [f"tm_{column}" for column in PHYSICAL_COLUMNS]
    metric_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for source_year, audit_year in TRANSITIONS:
        source_mask = aligned_season == source_year
        source_index = aligned_main_index[source_mask]
        source = main.iloc[source_index].reset_index(drop=True).copy()
        audit = main.loc[main["season"].eq(audit_year)].reset_index(drop=True).copy()
        physical = trackman.iloc[aligned_trackman_index[source_mask]].reset_index(drop=True)
        for column in PHYSICAL_COLUMNS:
            source[f"tm_{column}"] = physical[column].to_numpy()
            audit[f"tm_{column}"] = np.nan
        source_x, _, audit_x = _prepare_frames(
            source, source.iloc[:0].copy(), audit, full_columns
        )
        target = source["control_success"].to_numpy(np.float64)
        safe_model = _model(student=False, seed=3100 + audit_year)
        safe_model.fit(
            source_x[safe_columns],
            target,
            categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in safe_columns],
        )
        safe_prediction = safe_model.predict(audit_x[safe_columns]).astype(np.float64)
        full_model = _model(student=False, seed=3200 + audit_year)
        full_model.fit(
            source_x[full_columns],
            target,
            categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in full_columns],
        )
        tables, global_means, counts = _profile_tables(
            main,
            trackman,
            aligned_main_index,
            aligned_trackman_index,
            aligned_season,
            source_year,
        )
        weights, reliability = _pitch_weights(audit, counts)
        expected_full = np.zeros(len(audit), dtype=np.float64)
        for group_index, group in enumerate(PITCH_GROUPS):
            group_x = _fill_group_profile(
                audit_x, audit, group, tables[group], global_means
            )
            prediction = full_model.predict(group_x[full_columns]).astype(np.float64)
            expected_full += weights[:, group_index] * prediction
            del group_x, prediction
            gc.collect()
        delta = expected_full - safe_prediction
        artifact = np.load(
            project
            / "artifacts"
            / "v16_multiseason_20260815_02"
            / f"{V17_NAME}_o{audit_year}.npz"
        )
        incumbent = artifact["candidate"].astype(np.float64)
        audit_target = artifact["target"].astype(np.float64)
        profile_rows.append(
            {
                "source_year": source_year,
                "audit_year": audit_year,
                "profile_cutoff": source_year,
                "profile_pitchers": int(len(counts)),
                "audit_profile_coverage": float(audit["pitcher_id"].isin(counts.index).mean()),
                "delta_mean": float(delta.mean()),
                "delta_sd": float(delta.std()),
                "median_profile_reliability": float(np.median(reliability)),
            }
        )
        domains = {
            "ALL": np.ones(len(audit), dtype=bool),
            "R_CORE": (
                audit["game_type"].eq("R")
                & audit["pitcher_team_id"].ne(13)
                & audit["batter_team_id"].ne(13)
            ).to_numpy(),
            "R_ANCHOR": (
                audit["game_type"].eq("R")
                & (
                    audit["pitcher_team_id"].eq(13)
                    | audit["batter_team_id"].eq(13)
                )
            ).to_numpy(),
            "F": audit["game_type"].ne("R").to_numpy(),
        }
        for shrinkage in (0.0, 50.0, 200.0, 800.0):
            if shrinkage == 0.0:
                shrunk = delta
            else:
                profile_n = reliability * 200.0 / np.clip(1.0 - reliability, 1e-9, None)
                shrunk = delta * profile_n / (profile_n + shrinkage)
            for domain, apply_mask in domains.items():
                for weight in (0.01, 0.025, 0.05, 0.10, 0.20, 0.35):
                    candidate = incumbent.copy()
                    candidate[apply_mask] = np.clip(
                        candidate[apply_mask] + weight * shrunk[apply_mask],
                        0.001,
                        0.999,
                    )
                    diagnostic = _diagnostics(
                        audit, audit_target, incumbent, candidate
                    )
                    metric_rows.append(
                        {
                            "source_year": source_year,
                            "audit_year": audit_year,
                            "candidate": f"latent_{domain}_k{shrinkage:g}",
                            "weight": weight,
                            "gain_vs_v17": diagnostic["gain"],
                            "month_positive_fraction": float(
                                np.mean([row["gain"] > 0 for row in diagnostic["months"]])
                            ),
                            "worst_month_gain": float(
                                min(row["gain"] for row in diagnostic["months"])
                            ),
                            "minimum_domain_gain": float(
                                min(row["gain"] for row in diagnostic["domains"])
                            ),
                        }
                    )
        # Diversity controls: both predictions are legal at audit inference.
        # The safe model sees only main-table fields; expected_full sees only
        # past pitcher x pitch-family profiles, never the audit pitch.
        for name, raw_prediction in (
            ("safe_aligned_model", safe_prediction),
            ("past_profile_teacher_expectation", expected_full),
        ):
            for domain, apply_mask in domains.items():
                for weight in (0.005, 0.01, 0.025, 0.05, 0.10, 0.20):
                    candidate = incumbent.copy()
                    candidate[apply_mask] = np.clip(
                        candidate[apply_mask]
                        + weight
                        * (raw_prediction[apply_mask] - incumbent[apply_mask]),
                        0.001,
                        0.999,
                    )
                    diagnostic = _diagnostics(
                        audit, audit_target, incumbent, candidate
                    )
                    metric_rows.append(
                        {
                            "source_year": source_year,
                            "audit_year": audit_year,
                            "candidate": f"{name}_{domain}",
                            "weight": weight,
                            "gain_vs_v17": diagnostic["gain"],
                            "month_positive_fraction": float(
                                np.mean([row["gain"] > 0 for row in diagnostic["months"]])
                            ),
                            "worst_month_gain": float(
                                min(row["gain"] for row in diagnostic["months"])
                            ),
                            "minimum_domain_gain": float(
                                min(row["gain"] for row in diagnostic["domains"])
                            ),
                        }
                    )
        del source_x, audit_x, safe_model, full_model, source
        gc.collect()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    robust = (
        metrics.groupby(["candidate", "weight"], observed=True)["gain_vs_v17"]
        .agg(min_gain="min", mean_gain="mean", max_gain="max")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False)
    )
    robust.to_csv(output_dir / "robust.csv", index=False)
    summary = {
        "protocol": "TRACKMAN_LATENT_PROFILE_INTEGRATION_FORWARD_V1",
        "profiles": profile_rows,
        "robust": robust.head(40).to_dict(orient="records"),
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
        "--alignment-dir",
        type=Path,
        default=Path("artifacts/trackman_privileged_20260816"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/trackman_latent_integration_20260816_01"),
    )
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.output_dir)


if __name__ == "__main__":
    main()
