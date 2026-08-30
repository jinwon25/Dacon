"""Audit prior-season public PITCHf/x command profiles above the v290 parent.

The public Naver play-by-play file contains realised pitch locations.  Only the
2023 file is used here, and only to build pitcher-level profiles that precede
the labelled 2024 audit season.  No 2025 play-by-play or evaluation rows are
read.  A strongly regularised pitcher-level residual model is evaluated with
both held-out-pitcher and held-out-team folds before any deployment asset is
considered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROTOCOL = "V297_PRIOR_PITCHFX_COMMAND_SIGNAL_V1"
TARGET = "control_success"
RIDGE_ALPHA = 100.0
MIN_AUDIT_PITCHES = 100
RANDOM_STATE = 2970


def bss(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(y.mean() * (1.0 - y.mean()))
    return float(100000.0 * (1.0 - np.mean((y - prediction) ** 2) / denominator))


def _pooled_within_sd(
    frame: pd.DataFrame, metric: str, output_name: str
) -> pd.Series:
    grouped = frame.groupby(["pitcher_code", "pitch_type"], observed=True)[
        metric
    ].agg(["count", "var"])
    numerator = ((grouped["count"] - 1.0) * grouped["var"].fillna(0.0)).groupby(
        level=0
    ).sum()
    denominator = (grouped["count"] - 1.0).groupby(level=0).sum()
    result = np.sqrt(numerator / denominator.where(denominator.gt(0.0)))
    result.name = output_name
    return result


def make_pitchfx_profiles(pbp: pd.DataFrame) -> pd.DataFrame:
    """Build label-free, prior-season location profiles by real KBO pitcher code."""

    columns = [
        "game_pk",
        "pitcher",
        "pitch_type",
        "balls",
        "strikes",
        "stand",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
    ]
    missing = sorted(set(columns).difference(pbp.columns))
    if missing:
        raise ValueError(f"PITCHf/x columns missing: {missing}")
    frame = pbp.loc[:, columns].copy()
    frame["pitcher_code"] = pd.to_numeric(frame["pitcher"], errors="coerce")
    for column in ["balls", "strikes", "plate_x", "plate_z", "sz_top", "sz_bot"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["pitcher_code"].notna()
        & frame["plate_x"].between(-4.0, 4.0)
        & frame["plate_z"].between(-1.0, 7.0)
        & frame["sz_top"].between(2.0, 5.0)
        & frame["sz_bot"].between(0.5, 2.5)
        & frame["sz_top"].gt(frame["sz_bot"])
    ].copy()
    frame["pitcher_code"] = frame["pitcher_code"].astype(np.int64)
    frame["z_norm"] = (frame["plate_z"] - frame["sz_bot"]) / (
        frame["sz_top"] - frame["sz_bot"]
    )
    frame = frame.loc[frame["z_norm"].between(-2.0, 3.0)].copy()
    frame["abs_x"] = frame["plate_x"].abs()
    frame["zone"] = frame["abs_x"].le(0.7083) & frame["z_norm"].between(0.0, 1.0)
    frame["heart"] = frame["abs_x"].le(0.35) & frame["z_norm"].between(0.25, 0.75)
    frame["shadow"] = (
        frame["abs_x"].between(0.55, 0.90)
        | frame["z_norm"].between(-0.15, 0.15)
        | frame["z_norm"].between(0.85, 1.15)
    )
    frame["edge_x"] = (frame["abs_x"] - 0.7083).abs()
    frame["edge_z"] = np.minimum(frame["z_norm"].abs(), (frame["z_norm"] - 1.0).abs())
    frame["center_distance"] = np.sqrt(
        (frame["plate_x"] / 0.7083) ** 2 + ((frame["z_norm"] - 0.5) / 0.5) ** 2
    )
    frame["count_group"] = np.select(
        [frame["balls"].gt(frame["strikes"]), frame["strikes"].gt(frame["balls"])],
        ["batter_ahead", "pitcher_ahead"],
        default="even",
    )
    frame["two_strike"] = frame["strikes"].ge(2)
    frame["pitch_type"] = frame["pitch_type"].fillna("UNK").astype(str)
    frame["stand"] = frame["stand"].fillna("U").astype(str)

    grouped = frame.groupby("pitcher_code", observed=True, sort=False)
    profile = grouped.agg(
        pfx_n=("plate_x", "size"),
        pfx_games=("game_pk", "nunique"),
        pfx_x_mean=("plate_x", "mean"),
        pfx_x_sd=("plate_x", "std"),
        pfx_abs_x_mean=("abs_x", "mean"),
        pfx_z_mean=("z_norm", "mean"),
        pfx_z_sd=("z_norm", "std"),
        pfx_zone_rate=("zone", "mean"),
        pfx_heart_rate=("heart", "mean"),
        pfx_shadow_rate=("shadow", "mean"),
        pfx_edge_x_mean=("edge_x", "mean"),
        pfx_edge_z_mean=("edge_z", "mean"),
        pfx_center_distance_mean=("center_distance", "mean"),
        pfx_center_distance_sd=("center_distance", "std"),
    )
    profile["pfx_log_n"] = np.log1p(profile["pfx_n"])
    profile["pfx_log_games"] = np.log1p(profile["pfx_games"])
    profile = profile.join(
        _pooled_within_sd(frame, "plate_x", "pfx_within_type_x_sd")
    ).join(_pooled_within_sd(frame, "z_norm", "pfx_within_type_z_sd"))

    type_counts = frame.groupby(
        ["pitcher_code", "pitch_type"], observed=True
    ).size()
    type_share = type_counts / type_counts.groupby(level=0).sum()
    entropy = (-(type_share * np.log(type_share.clip(lower=1e-12)))).groupby(
        level=0
    ).sum()
    profile["pfx_pitch_type_entropy"] = entropy
    profile["pfx_pitch_type_count"] = type_counts.groupby(level=0).size()

    def add_conditional(keys: list[str], values: Iterable[Any], prefix: str) -> None:
        conditional = frame.groupby(
            ["pitcher_code", *keys], observed=True
        ).agg(
            n=("plate_x", "size"),
            zone=("zone", "mean"),
            heart=("heart", "mean"),
            x_sd=("plate_x", "std"),
            z_sd=("z_norm", "std"),
        )
        for value in values:
            key = value if isinstance(value, tuple) else (value,)
            try:
                lookup = key if len(key) > 1 else key[0]
                levels: str | list[str] = keys if len(keys) > 1 else keys[0]
                part = conditional.xs(lookup, level=levels)
            except KeyError:
                continue
            label = "_".join(str(item).lower() for item in key)
            for metric in ["n", "zone", "heart", "x_sd", "z_sd"]:
                profile[f"pfx_{prefix}_{label}_{metric}"] = part[metric]

    add_conditional(["count_group"], ["batter_ahead", "even", "pitcher_ahead"], "count")
    add_conditional(["two_strike"], [False, True], "two_strike")
    add_conditional(["stand"], ["L", "R"], "stand")
    return profile.reset_index()


def _model() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)),
        ]
    )


def _fit_fold(
    table: pd.DataFrame,
    features: list[str],
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> np.ndarray:
    train = table.iloc[train_index]
    valid = table.iloc[valid_index]
    weight = train["audit_n"].to_numpy(np.float64)
    weight = weight / weight.mean()
    target = train["residual"].to_numpy(np.float64)
    target = target - np.average(target, weights=weight)
    model = _model()
    model.fit(train[features], target, ridge__sample_weight=weight)
    return model.predict(valid[features]).astype(np.float64)


def cross_validated_signal(
    table: pd.DataFrame, features: list[str], scheme: str
) -> np.ndarray:
    output = np.full(len(table), np.nan, dtype=np.float64)
    if scheme == "pitcher_5fold":
        splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = splitter.split(table)
    elif scheme == "team_holdout":
        splitter = GroupKFold(n_splits=int(table["pitcher_team_id"].nunique()))
        splits = splitter.split(table, groups=table["pitcher_team_id"])
    else:
        raise ValueError(f"unknown CV scheme: {scheme}")
    for train_index, valid_index in splits:
        output[valid_index] = _fit_fold(table, features, train_index, valid_index)
    if not np.isfinite(output).all():
        raise AssertionError(f"non-finite OOF signal for {scheme}")
    return output


def optimal_dose(y: np.ndarray, parent: np.ndarray, signal: np.ndarray) -> float:
    covariance = float(np.mean(signal * (y - parent)))
    square = float(np.mean(signal**2))
    return 0.0 if square <= 0.0 else float(np.clip(covariance / square, 0.0, 1.0))


def run(
    train_csv: Path,
    parent_axes: Path,
    pbp_2023: Path,
    pitcher_map_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    audit = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    with np.load(parent_axes, allow_pickle=False) as saved:
        parent = saved["candidate_full_2024"].astype(np.float64)
    if len(audit) != len(parent):
        raise ValueError("full-2024 parent alignment mismatch")

    pbp = pd.read_parquet(pbp_2023)
    profiles = make_pitchfx_profiles(pbp)
    profiles.to_parquet(output_dir / "pitchfx_profiles_2023.parquet", index=False)
    mapping = pd.read_csv(pitcher_map_csv)
    mapping = mapping.loc[
        mapping["conf"].ge(0.90),
        ["pitcher_id", "pitcher_trackman_id", "conf"],
    ].sort_values("conf", ascending=False).drop_duplicates("pitcher_id")
    map_code = mapping.set_index("pitcher_id")["pitcher_trackman_id"]

    regular = audit["game_type"].astype(str).eq("R").to_numpy()
    row_code = audit["pitcher_id"].map(map_code)
    covered = regular & row_code.isin(profiles["pitcher_code"]).to_numpy()
    audit_work = audit.assign(
        _parent=parent,
        _residual=audit[TARGET].to_numpy(np.float64) - parent,
        _pitcher_code=row_code,
    )
    pitcher = audit_work.loc[covered].groupby("pitcher_id", sort=False).agg(
        audit_n=(TARGET, "size"),
        residual=("_residual", "mean"),
        actual=(TARGET, "mean"),
        parent=("_parent", "mean"),
        pitcher_team_id=("pitcher_team_id", lambda values: values.mode().iloc[0]),
        pitcher_code=("_pitcher_code", "first"),
    ).reset_index()
    pitcher = pitcher.loc[pitcher["audit_n"].ge(MIN_AUDIT_PITCHES)].copy()
    pitcher = pitcher.merge(profiles, on="pitcher_code", how="inner", validate="many_to_one")
    feature_columns = sorted(column for column in profiles.columns if column != "pitcher_code")

    y = audit[TARGET].to_numpy(np.float64)
    results: dict[str, Any] = {}
    saved_arrays: dict[str, np.ndarray] = {"parent_full_2024": parent}
    for scheme in ["pitcher_5fold", "team_holdout"]:
        pitcher_signal = cross_validated_signal(pitcher, feature_columns, scheme)
        signal_map = pd.Series(pitcher_signal, index=pitcher["pitcher_id"])
        row_signal = audit["pitcher_id"].map(signal_map).fillna(0.0).to_numpy(np.float64)
        row_signal[~regular] = 0.0
        dose = optimal_dose(y, parent, row_signal)
        candidate = np.clip(parent + dose * row_signal, 0.001, 0.999)
        active = row_signal != 0.0
        monthly: dict[str, float] = {}
        for month in sorted(audit.loc[active, "game_month"].unique()):
            mask = active & audit["game_month"].eq(month).to_numpy()
            monthly[str(int(month))] = bss(y[mask], candidate[mask]) - bss(y[mask], parent[mask])
        results[scheme] = {
            "dose": dose,
            "gain": bss(y, candidate) - bss(y, parent),
            "whole_row_rms": float(np.sqrt(np.mean((candidate - parent) ** 2))),
            "active_row_rms": float(np.sqrt(np.mean((candidate[active] - parent[active]) ** 2))) if active.any() else 0.0,
            "active_rows": int(active.sum()),
            "active_fraction": float(active.mean()),
            "positive_month_fraction": float(np.mean(np.asarray(list(monthly.values())) > 0.0)) if monthly else 0.0,
            "worst_month_gain": min(monthly.values()) if monthly else 0.0,
            "monthly_gain": monthly,
            "pitcher_signal_correlation_with_residual": float(np.corrcoef(pitcher_signal, pitcher["residual"])[0, 1]),
        }
        saved_arrays[f"signal_{scheme}_full_2024"] = row_signal
        saved_arrays[f"candidate_{scheme}_full_2024"] = candidate

    profile_correlations = {}
    for feature in feature_columns:
        valid = pitcher[[feature, "residual"]].dropna()
        if len(valid) >= 20 and valid[feature].nunique() > 1:
            profile_correlations[feature] = float(valid.corr().iloc[0, 1])
    strongest = sorted(
        profile_correlations.items(), key=lambda item: abs(item[1]), reverse=True
    )[:20]
    summary = {
        "protocol": PROTOCOL,
        "status": "research_audit_complete",
        "source_external_season": 2023,
        "audit_label_season": 2024,
        "ridge_alpha": RIDGE_ALPHA,
        "minimum_audit_pitches_per_pitcher": MIN_AUDIT_PITCHES,
        "feature_count": len(feature_columns),
        "pitchers_in_residual_model": int(len(pitcher)),
        "mapped_regular_rows": int(covered.sum()),
        "mapped_regular_fraction": float(covered.sum() / max(regular.sum(), 1)),
        "modelled_regular_rows": int(audit["pitcher_id"].isin(pitcher["pitcher_id"]).to_numpy().sum()),
        "results": results,
        "strongest_univariate_profile_correlations": strongest,
        "restrictions": {
            "public_pbp_2023_only": True,
            "public_pbp_2024_not_read": True,
            "public_pbp_2025_not_read": True,
            "test_csv_not_read": True,
            "public_score_not_used_for_model_selection": True,
            "pitcher_residuals_out_of_fold": True,
            "global_level_shift_removed_within_each_fold": True,
        },
    }
    pitcher.to_parquet(output_dir / "pitcher_training_table.parquet", index=False)
    np.savez_compressed(output_dir / "audit_axes.npz", **saved_arrays)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--parent-axes", type=Path, required=True)
    parser.add_argument("--pbp-2023", type=Path, required=True)
    parser.add_argument("--pitcher-map-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.train_csv,
                args.parent_axes,
                args.pbp_2023,
                args.pitcher_map_csv,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
