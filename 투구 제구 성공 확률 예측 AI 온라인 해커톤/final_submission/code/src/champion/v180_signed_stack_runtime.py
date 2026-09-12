"""Standalone runtime for the source-selected v178 signed stack.

Every lookup in this module is frozen from official 2019--2024 training rows.
Prediction for one evaluation row never depends on any other evaluation row.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ID_COL = "row_id"
TARGET_COL = "control_success"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_PATH = BASE_DIR / "data" / "test.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"
OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"

ANCHOR_TEAM = 13
H1_WEIGHT = 0.16
H1_BASE_WEIGHT = 0.15
C3_WEIGHT = 0.5
MEAN_RECENT_WEIGHT = 0.25
MEAN_RECENT_BASE_WEIGHT = 0.15
BRIDGE_SCALE = 1.2

STACK_SCALE = 0.25
V114_WEIGHT = 0.06988829665167925
V131_WEIGHT = -0.20028521988178444
V135_WEIGHT = 0.18470959209793483
V160_WEIGHT = 0.04511689136862597
V112_ALPHA = 0.25
V135_C3_ETA = 0.75
AFFINE_ALPHA = 1.09
AFFINE_CENTER = 0.5854452601930041

V114_SAFE = {
    "count": ("0-0", "0-2", "1-1", "1-2", "2-2", "3-0"),
    "history": ("0-29", "1000+", "200-999"),
    "platoon": ("1-1", "2-1", "2-2"),
}
COMPONENT_PRIOR = 0.53


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict_parent(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module("v180_original_parent", MODEL_DIR / "v124" / "script.py")
    return np.asarray(module.predict_dataframe(frame), dtype=np.float64)


def _predict_bridge_parent(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module(
        "v180_bridge_parent", MODEL_DIR / "v124_bridge" / "script.py"
    )
    return np.asarray(module.predict_dataframe(frame), dtype=np.float64)


def _prepare_h1(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = MODEL_DIR / "h1"
    module = _load_module("v180_h1_component", root / "script.py")
    bundle = joblib.load(root / "model" / "rf.pkl")
    prepared = module.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in module.CAAFE_COLS):
        prepared = module.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in module.ASOF_COLS):
        prepared = module.attach_asof_state(prepared, bundle)
    features = module.build_features(prepared, bundle)

    depth8 = np.mean(
        np.column_stack(
            [model.predict_proba(features)[:, 1] for model in bundle["models"]]
        ),
        axis=1,
    )
    post4 = np.asarray(module.platoon_adjust(bundle, features), dtype=np.float64)
    h1_identity = depth8 + post4
    h1_affine = np.clip(
        AFFINE_CENTER + AFFINE_ALPHA * (h1_identity - AFFINE_CENTER), 0.0, 1.0
    )
    proxy_model = joblib.load(MODEL_DIR / "proxy_h1_seed42.joblib")
    h1_proxy = np.asarray(proxy_model.predict_proba(features)[:, 1], dtype=np.float64) + post4
    return h1_identity, h1_affine, h1_proxy


def _window_matrix(
    frame: pd.DataFrame, bundle: dict[str, object]
) -> np.ndarray:
    pitcher = frame["pitcher_id"].astype("int64")
    contexts = (
        (
            frame["pitcher_hand"].astype("int64")
            == frame["batter_hand"].astype("int64")
        ).to_numpy(),
        (frame["strikes_before"].astype("int64") == 2).to_numpy(),
        (frame["num_runners_on"].astype("int64") > 0).to_numpy(),
    )
    columns = []
    for window in bundle["window_order"]:
        output = np.zeros(len(frame), dtype=np.float64)
        for label, context in zip(("hand", "two", "runner"), contexts):
            magnitude = (
                pitcher.map(bundle["tables"][window][label])
                .fillna(0.0)
                .to_numpy(np.float64)
            )
            scale = float(bundle["contrast_scale"])
            output += np.where(context, scale * magnitude, -scale * magnitude)
        columns.append(output)
    return np.column_stack(columns)


def _c3_parts(frame: pd.DataFrame, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = joblib.load(MODEL_DIR / filename)
    matrix = _window_matrix(frame, bundle)
    agreed = np.all(matrix > 0.0, axis=1) | np.all(matrix < 0.0, axis=1)
    sign_all = matrix.mean(axis=1) * agreed
    mean_recent = matrix[:, :2].mean(axis=1)
    return sign_all, mean_recent, matrix[:, 0]


def _rcore(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].astype("int64").eq(ANCHOR_TEAM)
        | frame["batter_team_id"].astype("int64").eq(ANCHOR_TEAM)
    ).to_numpy()
    return regular & ~anchor


def _v114_mask(frame: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    count = np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))
    history = (
        pd.cut(
            pd.to_numeric(frame["asof_pitcher_n"], errors="raise"),
            bins=[-np.inf, 29, 199, 999, np.inf],
            labels=["0-29", "30-199", "200-999", "1000+"],
        )
        .astype(str)
        .to_numpy()
    )
    platoon = np.char.add(
        np.char.add(frame["pitcher_hand"].astype(str).to_numpy(), "-"),
        frame["batter_hand"].astype(str).to_numpy(),
    )
    return (
        np.isin(count, V114_SAFE["count"])
        & np.isin(history, V114_SAFE["history"])
        & np.isin(platoon, V114_SAFE["platoon"])
    )


def _component_add_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["count_state"] = output["balls_before"] * 3 + output["strikes_before"]
    output["same_hand"] = (output["pitcher_hand"] == output["batter_hand"]).astype(int)
    output["abs_score_diff"] = output["score_diff_pitcher_team"].abs()
    output["log_pitcher_n"] = np.log1p(output["asof_pitcher_n"])
    output["log_batter_n"] = np.log1p(output["asof_batter_n"])
    for n_col, rate_col in (
        ("asof_pitcher_n", "asof_pitcher_success_rate"),
        ("asof_batter_n", "asof_batter_success_rate"),
    ):
        n = output[n_col].fillna(0)
        rate = output[rate_col].fillna(COMPONENT_PRIOR)
        output[f"shrunk_{rate_col}"] = (n * rate + 300 * COMPONENT_PRIOR) / (n + 300)
    output["form_diff_1"] = output["asof_pitcher_prev1_game_success_rate"] - output["asof_pitcher_success_rate"]
    output["form_diff_3"] = output["asof_pitcher_prev3_game_success_rate"] - output["asof_pitcher_success_rate"]
    output["form_diff_5"] = output["asof_pitcher_prev5_game_success_rate"] - output["asof_pitcher_success_rate"]
    output["middle_diff_3"] = output["asof_pitcher_prev3_game_middle_rate"] - output["asof_pitcher_middle_rate"]
    output["behind_count"] = (output["balls_before"] - output["strikes_before"]).clip(lower=0)
    output["pitcher_x_behind"] = output["shrunk_asof_pitcher_success_rate"] * output["behind_count"]
    output["li_x_pitcher"] = output["li"] * output["shrunk_asof_pitcher_success_rate"]
    output["runners_scoring_pos"] = output["runner_on_2b"] + output["runner_on_3b"]
    output["ball_minus_strike_rate"] = output["asof_pitcher_ball_rate"] - output["asof_pitcher_strike_rate"]
    mix = output[
        ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    ].fillna(1.0 / 3.0).clip(1e-6, 1.0)
    output["pitchmix_entropy"] = -(mix * np.log(mix)).sum(axis=1)
    output["pb_success_gap"] = output["shrunk_asof_pitcher_success_rate"] - output["shrunk_asof_batter_success_rate"]
    output["full_count"] = ((output["balls_before"] == 3) & (output["strikes_before"] == 2)).astype(int)
    output["pressure"] = output["li"] * output["num_runners_on"]
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(value) - np.log1p(-value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _predict_component(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "component"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    platoon = pd.read_csv(root / "platoon_2025.csv")
    count = pd.read_csv(root / "count_2025.csv")
    prepared = frame.copy()
    prepared["count_state"] = prepared["balls_before"] * 3 + prepared["strikes_before"]
    n_rows = len(prepared)
    prepared = prepared.merge(platoon, on=["pitcher_id", "batter_hand"], how="left")
    prepared = prepared.merge(count, on=["pitcher_id", "count_state"], how="left")
    if len(prepared) != n_rows:
        raise ValueError("component lookup tables are not unique")
    for column, fill in (
        ("plat_rate", COMPONENT_PRIOR), ("plat_n", 0), ("plat_dev", 0),
        ("cnt_rate", COMPONENT_PRIOR), ("cnt_n", 0), ("cnt_dev", 0),
    ):
        prepared[column] = prepared[column].fillna(fill)
    prepared["log_plat_n"] = np.log1p(prepared["plat_n"])
    prepared["log_cnt_n"] = np.log1p(prepared["cnt_n"])
    features = _component_add_features(prepared).reindex(columns=config["feature_cols"])
    for column in config["cat_cols"]:
        features[column] = pd.Categorical(
            features[column], categories=config["cat_maps"][column]
        )
    lgb_model = lgb.Booster(model_file=str(root / "lgbm_a_42.txt"))
    lgb_probability = lgb_model.predict(
        features, num_iteration=int(config["lgb_iterations"])
    )
    cat_frame = features.copy()
    for column in config["cat_cols"]:
        cat_frame[column] = cat_frame[column].astype(str)
    cat_model = CatBoostClassifier()
    cat_model.load_model(root / "cat_bayes.cbm")
    cat_probability = cat_model.predict_proba(
        Pool(cat_frame, cat_features=config["cat_cols"]),
        ntree_end=int(config["cat_iterations"]),
    )[:, 1]
    return _sigmoid(
        0.5 * (_logit(lgb_probability) + _logit(cat_probability))
        + float(config["delta"])
    )


def _compose_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    parent = _predict_parent(frame)
    h1_identity, h1_affine, h1_proxy = _prepare_h1(frame)
    proxy_sign, proxy_recent, proxy_last1 = _c3_parts(frame, "c3_sign_all.joblib")
    exact_sign, exact_recent, _ = _c3_parts(frame, "c3_exact.joblib")
    exact_c3 = (1.0 - MEAN_RECENT_BASE_WEIGHT) * exact_sign + MEAN_RECENT_BASE_WEIGHT * exact_recent
    active = _rcore(frame)

    v158 = parent.copy()
    v158[active] = np.clip(
        0.85 * parent[active] + 0.15 * h1_identity[active] + 0.5 * exact_c3[active],
        0.001, 0.999,
    )
    v131 = parent.copy()
    v131[active] = np.clip(
        0.85 * parent[active] + 0.15 * h1_proxy[active], 0.001, 0.999
    )
    v135 = v131.copy()
    v135[active] = np.clip(v135[active] + V135_C3_ETA * proxy_last1[active], 0.001, 0.999)
    v160 = v158.copy()
    v160[active] = np.clip(
        v160[active] + 0.15 * (h1_affine[active] - h1_identity[active]),
        0.001, 0.999,
    )
    component = _predict_component(frame)
    v114 = parent.copy()
    stable = active & _v114_mask(frame)
    v114[stable] = np.clip(
        parent[stable] + V112_ALPHA * (component[stable] - parent[stable]),
        0.001, 0.999,
    )
    direction = (
        V114_WEIGHT * (v114 - v158)
        + V131_WEIGHT * (v131 - v158)
        + V135_WEIGHT * (v135 - v158)
        + V160_WEIGHT * (v160 - v158)
    )
    return {
        "parent": parent,
        "h1_identity": h1_identity,
        "h1_affine": h1_affine,
        "proxy_sign": proxy_sign,
        "proxy_recent": proxy_recent,
        "v114": v114,
        "v131": v131,
        "v135": v135,
        "v158": v158,
        "v160": v160,
        "direction": direction,
        "rcore": active,
    }


def _predict_current_jy(frame: pd.DataFrame, values: dict[str, np.ndarray]) -> np.ndarray:
    parent = values["parent"]
    bridge_parent = _predict_bridge_parent(frame)
    h1 = values["h1_affine"]
    c3_base = (1.0 - MEAN_RECENT_BASE_WEIGHT) * values["proxy_sign"] + MEAN_RECENT_BASE_WEIGHT * values["proxy_recent"]
    c3_active = (1.0 - MEAN_RECENT_WEIGHT) * values["proxy_sign"] + MEAN_RECENT_WEIGHT * values["proxy_recent"]
    base_active = values["rcore"]
    runners = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy() > 0
    high_li = pd.to_numeric(frame["li"], errors="coerce").fillna(0.0).to_numpy() >= 1.5
    active = base_active & (runners | high_li)
    bridge = parent + BRIDGE_SCALE * (bridge_parent - parent)
    output = parent.copy()
    output[base_active] = np.clip(
        0.85 * parent[base_active] + 0.15 * h1[base_active] + C3_WEIGHT * c3_base[base_active],
        0.001, 0.999,
    )
    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * bridge[active] + H1_WEIGHT * h1[active] + C3_WEIGHT * c3_active[active],
        0.001, 0.999,
    )
    return output


def predict_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    values = _compose_components(frame)
    values["current_jy"] = _predict_current_jy(frame, values)
    values["candidate"] = np.clip(
        values["current_jy"] + STACK_SCALE * values["direction"], 0.001, 0.999
    )
    return values


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    return predict_components(frame)["candidate"]


def main() -> None:
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample submission columns are invalid")
    if len(test) != len(sample) or set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test/sample row_id mismatch")
    if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
        raise ValueError("test row_id is invalid")
    probability = predict_dataframe(test)
    sample[TARGET_COL] = sample[ID_COL].map(dict(zip(test[ID_COL], probability, strict=True)))
    if sample[TARGET_COL].isna().any() or not sample[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("v180 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v180_signed_stack | rows={len(sample)} | "
        f"mean={probability.mean():.6f} | min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
