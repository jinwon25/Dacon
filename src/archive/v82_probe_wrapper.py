"""Standalone v82 wrapper: champion plus 10% EXP-021 strict on R_CORE."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEST_PATH = BASE_DIR / "data" / "test.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"
OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"
ANCHOR_TEAM = 13
STRICT_WEIGHT = 0.10


def probe_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R")
    anchor = frame["pitcher_team_id"].eq(ANCHOR_TEAM) | frame[
        "batter_team_id"
    ].eq(ANCHOR_TEAM)
    return (regular & ~anchor).to_numpy()


def blend_predictions(
    champion: np.ndarray,
    strict: np.ndarray,
    active: np.ndarray,
    weight: float = STRICT_WEIGHT,
) -> np.ndarray:
    parent = np.asarray(champion, dtype=np.float64)
    challenger = np.asarray(strict, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if not (parent.shape == challenger.shape == mask.shape and parent.ndim == 1):
        raise ValueError("v82 blend arrays must be aligned one-dimensional vectors")
    if not (0.0 <= float(weight) <= 1.0):
        raise ValueError("v82 blend weight must be in [0, 1]")
    if not (np.isfinite(parent).all() and np.isfinite(challenger).all()):
        raise ValueError("v82 blend input contains non-finite values")
    output = parent.copy()
    output[mask] = (1.0 - float(weight)) * parent[mask] + float(weight) * challenger[mask]
    return np.clip(output, 0.001, 0.999)


def _load_component(name: str):
    path = MODEL_DIR / "components" / f"{name}_script.py"
    spec = importlib.util.spec_from_file_location(f"v82_{name}_component", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load v82 component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_DIR = MODEL_DIR / name
    return module


def _strict_predict(frame: pd.DataFrame, strict) -> np.ndarray:
    metadata = json.loads(
        (strict.MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (strict.MODEL_DIR / "history_state.json").read_text(encoding="utf-8")
    )
    multirate = json.loads(
        (strict.MODEL_DIR / "multirate_state.json").read_text(encoding="utf-8")
    )
    schemas = json.loads(
        (strict.MODEL_DIR / "feature_schemas.json").read_text(encoding="utf-8")
    )
    rows = strict.add_static_features(frame.drop(columns=[ID_COL]))
    rows = strict.attach_temporal_features(rows, history)
    rows = strict.attach_multirate_features(rows, multirate)
    temporal_base = rows["temporal_base_global_30"].to_numpy(dtype=float)
    group_state = json.loads(
        (strict.MODEL_DIR / "group_effects.json").read_text(encoding="utf-8")
    )
    group_base = np.clip(
        temporal_base + strict.map_exp018_group(rows, group_state), 0.0, 1.0
    )
    booster = lgb.Booster(
        model_str=(strict.MODEL_DIR / "rfull_lightgbm.txt").read_text(
            encoding="utf-8"
        )
    )
    hgb_state = json.loads(
        (strict.MODEL_DIR / "histgradientboosting.json").read_text(
            encoding="utf-8"
        )
    )
    lgb_correction = booster.predict(
        strict.encoded_matrix(rows, schemas["lightgbm"])
    ).astype(float)
    hgb_correction = strict.predict_histgradientboosting(
        strict.encoded_matrix(rows, schemas["histgradientboosting"]), hgb_state
    )
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    lightgbm_branch = group_base.copy()
    histgradient_branch = group_base.copy()
    lightgbm_branch[regular] = np.clip(
        group_base[regular] + 0.75 * lgb_correction[regular], 0.0, 1.0
    )
    histgradient_branch[regular] = np.clip(
        group_base[regular] + hgb_correction[regular], 0.0, 1.0
    )
    backbone = 0.5 * lightgbm_branch + 0.5 * histgradient_branch
    team_state = json.loads(
        (strict.MODEL_DIR / "team_effects.json").read_text(encoding="utf-8")
    )
    team_base = np.clip(
        backbone + strict.map_team_effects(rows, team_state), 0.0, 1.0
    )
    if str(metadata["candidate"]) != "strict_lowrank_s300_r6":
        raise ValueError("v82 requires EXP-021 strict_lowrank_s300_r6")
    lowrank_state = json.loads(
        (strict.MODEL_DIR / "lowrank_effects.json").read_text(encoding="utf-8")
    )
    return np.clip(
        team_base + strict.map_lowrank_effects(rows, lowrank_state), 0.0, 1.0
    )


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    champion = _load_component("champion")
    strict = _load_component("strict")
    parent = np.asarray(champion.predict_dataframe(frame), dtype=np.float64)
    challenger = _strict_predict(frame, strict)
    return blend_predictions(parent, challenger, probe_mask(frame))


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
    prediction_map = dict(zip(test[ID_COL], probability, strict=True))
    sample[TARGET_COL] = sample[ID_COL].map(prediction_map)
    if sample[TARGET_COL].isna().any() or not sample[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("v82 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v82_v57_strict_w010 | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
