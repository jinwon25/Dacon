"""Small, bounded residual learners on top of the strict nested v2 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.archive.data import TARGET_COL, read_main
from src.metrics import brier_score


def context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only as-of, low-cardinality context (never IDs or target encoding)."""
    out = pd.DataFrame(index=frame.index)
    def num(name: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(frame.get(name, default), errors="coerce").fillna(default).astype(float)

    balls, strikes = num("balls_before"), num("strikes_before")
    count = balls.astype(int).astype(str) + "-" + strikes.astype(int).astype(str)
    ph = frame.get("pitcher_hand", "__MISSING__").astype("string").fillna("__MISSING__")
    bh = frame.get("batter_hand", "__MISSING__").astype("string").fillna("__MISSING__")
    out["count_state"] = count
    out["platoon"] = ph + "_" + bh
    out["count_platoon"] = count + "_" + out["platoon"]
    out["risp"] = ((num("runner_on_2b") > 0) | (num("runner_on_3b") > 0)).astype(int)
    out["base_state"] = frame.get("base_state", "__MISSING__").astype("string").fillna("__MISSING__")
    out["outs"] = num("outs_before").clip(0, 3)
    out["late_inning"] = (num("inning") >= 7).astype(int)
    out["li_log"] = np.log1p(num("li").clip(lower=0))
    out["close_score"] = (num("score_diff_pitcher_team").abs() <= 1).astype(int)
    pn = num("asof_pitcher_n"); bn = num("asof_batter_n");
    out["pitcher_conf"] = pn / (pn + 200.0)
    out["batter_conf"] = bn / (bn + 200.0)
    s1, s3, s5 = num("asof_pitcher_prev1_game_success_rate"), num("asof_pitcher_prev3_game_success_rate"), num("asof_pitcher_prev5_game_success_rate")
    m1, m3, m5 = num("asof_pitcher_prev1_game_middle_rate"), num("asof_pitcher_prev3_game_middle_rate"), num("asof_pitcher_prev5_game_middle_rate")
    out["d_s13"], out["d_s35"] = s1 - s3, s3 - s5
    out["D_S"] = np.sqrt(out["d_s13"] ** 2 + out["d_s35"] ** 2)
    out["d_m13"], out["d_m35"] = m1 - m3, m3 - m5
    out["D_M"] = np.sqrt(out["d_m13"] ** 2 + out["d_m35"] ** 2)
    success, middle, ball = num("asof_pitcher_success_rate"), num("asof_pitcher_middle_rate"), num("asof_pitcher_ball_rate")
    out["strike_minus_success"] = num("asof_pitcher_strike_rate") - success
    out["middle_minus_ball"] = middle - ball
    mix = np.column_stack([num("asof_pitcher_fastball_rate"), num("asof_pitcher_breaking_rate"), num("asof_pitcher_offspeed_rate")])
    mix = np.clip(mix, 1e-8, 1.0)
    out["pitchmix_entropy"] = -(mix * np.log(mix)).sum(axis=1)
    out["pitchmix_conf"] = num("asof_pitcher_pitchmix_n") / (num("asof_pitcher_pitchmix_n") + 200.0)
    # Explicitly registered interactions; redundant runner/WE representations are excluded.
    out["count_platoon_conf_DM"] = out["pitcher_conf"] * out["D_M"]
    out["three_ball_middle_ball"] = (balls >= 3).astype(int) * out["middle_minus_ball"]
    out["two_strike_strike_success"] = (strikes >= 2).astype(int) * out["strike_minus_success"]
    out["risp_li_conf_DS"] = out["risp"] * out["li_log"] * out["pitcher_conf"] * out["D_S"]
    return out.reset_index(drop=True)


def _zero_center(y: np.ndarray, group: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"y": y, "g": group.astype("string")})
    means = frame.groupby("g", observed=True)["y"].transform("mean").to_numpy()
    return y - means


def eb_delta(x: pd.DataFrame, residual: np.ndarray, alpha_parent: float = 5000.0, alpha_leaf: float = 2000.0) -> np.ndarray:
    parent = x["count_platoon"].astype("string")
    # Train-origin tertiles are deterministic and are not fit on the outer row.
    leaf_signal = x["D_M"].rank(method="first", pct=True).mul(3).astype(int).clip(0, 2).astype(str)
    leaf = parent + "_" + leaf_signal
    global_mean = 0.0
    p = pd.DataFrame({"key": parent, "r": residual}).groupby("key", observed=True)["r"].agg(["sum", "count"])
    l = pd.DataFrame({"key": leaf, "r": residual}).groupby("key", observed=True)["r"].agg(["sum", "count"])
    parent_rate = (p["sum"] + alpha_parent * global_mean) / (p["count"] + alpha_parent)
    leaf_rate = (l["sum"] + alpha_leaf * parent_rate.reindex(l.index).fillna(global_mean).to_numpy()) / (l["count"] + alpha_leaf)
    return leaf_rate.reindex(leaf).fillna(0.0).to_numpy(dtype=float).clip(-0.0125, 0.0125)


def ridge_fit_predict(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame, alpha: float) -> np.ndarray:
    combined = pd.concat([train_x, test_x], ignore_index=True)
    cat = [c for c in combined.columns if str(combined[c].dtype) in ("string", "object")]
    numeric = [c for c in combined.columns if c not in cat]
    encoded = pd.get_dummies(combined[cat], columns=cat, dtype=float) if cat else pd.DataFrame(index=combined.index)
    values = pd.concat([combined[numeric].astype(float).reset_index(drop=True), encoded.reset_index(drop=True)], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler().fit(values.iloc[: len(train_x)])
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(scaler.transform(values.iloc[: len(train_x)]), train_y)
    return model.predict(scaler.transform(values.iloc[len(train_x):]))


def shallow_lgb(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    combo = pd.concat([train_x, test_x], ignore_index=True)
    cats = [c for c in combo if str(combo[c].dtype) in ("string", "object")]
    for col in cats:
        combo[col] = combo[col].astype("category")
    combo = combo.apply(lambda s: s.cat.codes if str(s.dtype) == "category" else pd.to_numeric(s, errors="coerce")).fillna(0)
    tr, te = combo.iloc[: len(train_x)], combo.iloc[len(train_x):]
    dtrain = lgb.Dataset(tr, label=train_y, free_raw_data=True)
    booster = lgb.train({"objective": "regression_l2", "metric": "l2", "verbosity": -1, "num_threads": 6, "deterministic": True, "force_col_wise": True, "num_leaves": 7, "max_depth": 3, "min_data_in_leaf": 2000, "learning_rate": 0.03, "lambda_l2": 50, "max_bin": 63, "seed": 42}, dtrain, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
    return booster.predict(te)


def _load_nested(project: Path) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    payload = np.load(project / "artifacts/followup/v2_nested_predictions.npz", allow_pickle=False)
    years = sorted({int(key.split("_", 1)[0]) for key in payload.files})
    result = {year: {key.split("_", 1)[1]: payload[key] for key in payload.files if key.startswith(f"{year}_")} for year in years}
    return pd.DataFrame(), result


def run(project: Path) -> pd.DataFrame:
    _, preds = _load_nested(project)
    train = read_main(project / "data/train.csv")
    features = context_features(train)
    rows: list[dict] = []
    all_saved: dict[str, np.ndarray] = {}
    recipes = ["eb_count_platoon", "ridge_zero_intercept", "shallow_lgb_residual"]
    for year in sorted(preds):
        target = preds[year]["target"].astype(float)
        base = preds[year]["A_original"].astype(float)
        idx = preds[year]["valid_idx"].astype(int)
        # model fit years exclude the immediately preceding validation year;
        # that year is reserved for pre-registered eta selection.
        source_years = [y for y in sorted(preds) if y < year - 1]
        tune_year = year - 1 if year - 1 in preds else None
        if not source_years:
            for recipe in recipes:
                rows.append({"recipe": recipe, "outer_validation_season": year, "brier": brier_score(target, base), "delta_v2": 0.0, "eta": 0.0, "status": "no_prior_oof_source; correction_zero"})
            all_saved[f"{year}_base"] = base
            continue
        meta_idx = np.concatenate([preds[s]["valid_idx"].astype(int) for s in source_years])
        meta_y = np.concatenate([preds[s]["target"].astype(float) - preds[s]["A_original"].astype(float) for s in source_years])
        meta_x = features.iloc[meta_idx].copy()
        meta_y = _zero_center(meta_y, meta_x["platoon"])
        test_x = features.iloc[idx].copy()
        if tune_year is not None:
            tune_idx = preds[tune_year]["valid_idx"].astype(int)
            tune_target = preds[tune_year]["target"].astype(float)
            tune_base = preds[tune_year]["A_original"].astype(float)
            tune_x = features.iloc[tune_idx].copy()
        else:
            tune_target = tune_base = tune_x = None

        raw_predictions: dict[str, np.ndarray] = {}
        raw_predictions["eb_count_platoon"] = eb_delta(meta_x, meta_y)
        raw_predictions["ridge_zero_intercept"] = ridge_fit_predict(meta_x, meta_y, test_x, 1e-3)
        raw_predictions["shallow_lgb_residual"] = shallow_lgb(meta_x, meta_y, test_x)
        for recipe, raw in raw_predictions.items():
            raw = np.asarray(raw, dtype=float)
            if recipe == "eb_count_platoon":
                raw = raw.clip(-0.0125, 0.0125)
            # eta is chosen on the immediately preceding year only; absent a
            # tuning fold, use zero rather than touch the outer target.
            if tune_year is None:
                eta = 0.0
            else:
                if recipe == "eb_count_platoon":
                    tune_raw = eb_delta(tune_x, tune_target - tune_base)
                elif recipe == "ridge_zero_intercept":
                    tune_raw = ridge_fit_predict(meta_x, meta_y, tune_x, 1e-3)
                else:
                    tune_raw = shallow_lgb(meta_x, meta_y, tune_x)
                candidates = [0.0, 0.25, 0.5, 1.0]
                scores = [brier_score(tune_target, np.clip(tune_base + e * tune_raw, 1e-4, 1 - 1e-4)) for e in candidates]
                eta = float(candidates[int(np.argmin(scores))])
            correction = np.clip(eta * raw, -0.05, 0.05)
            pred = np.clip(base + correction, 1e-4, 1.0 - 1e-4)
            rows.append({"recipe": recipe, "outer_validation_season": year, "brier": brier_score(target, pred), "delta_v2": brier_score(target, pred) - brier_score(target, base), "eta": eta, "correction_abs_p995": float(np.quantile(np.abs(correction), 0.995)), "correction_abs_max": float(np.max(np.abs(correction))), "correction_mean": float(np.mean(correction)), "status": "evaluated_nested"})
            all_saved[f"{year}_{recipe}"] = pred
        del meta_x, test_x, raw_predictions
    out = pd.DataFrame(rows)
    out.to_csv(project / "reports/residual_recipe_results.csv", index=False)
    np.savez_compressed(project / "artifacts/followup/residual_predictions.npz", **all_saved)
    # A compact subgroup report is generated from each outer prediction; it
    # intentionally does not expose player IDs or any target-derived feature.
    subgroup_rows: list[dict] = []
    for year in sorted(preds):
        idx = preds[year]["valid_idx"].astype(int); y = preds[year]["target"].astype(float); base = preds[year]["A_original"].astype(float)
        for recipe in recipes:
            key = f"{year}_{recipe}"; p = all_saved.get(key, base)
            subset = features.iloc[idx].copy(); subset["y"], subset["p"], subset["base"] = y, p, base
            for name, mask in {"R": subset.get("count_state", pd.Series(index=subset.index)).astype(str).ne("__MISSING__"), "cold_pitcher": subset["pitcher_conf"].eq(0), "known_pitcher": subset["pitcher_conf"].gt(0)}.items():
                mask = np.asarray(mask, dtype=bool)
                if mask.any(): subgroup_rows.append({"recipe": recipe, "outer_validation_season": year, "subgroup": name, "n_rows": int(mask.sum()), "brier": brier_score(y[mask], p[mask]), "delta_v2": brier_score(y[mask], p[mask]) - brier_score(y[mask], base[mask])})
    pd.DataFrame(subgroup_rows).to_csv(project / "reports/residual_subgroup_results.csv", index=False)
    print(out.to_string(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__":
    main()
