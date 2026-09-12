"""Standalone v141 runtime: v124/H1/C3 absolute blend on R_CORE."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
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
H1_WEIGHT = 0.15
C3_WEIGHT = 0.5


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict_v124(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module("v142_v124_component", MODEL_DIR / "v124" / "script.py")
    return np.asarray(module.predict_dataframe(frame), dtype=np.float64)


def _predict_h1(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "h1"
    module = _load_module("v142_h1_component", root / "script.py")
    bundle = joblib.load(root / "model" / "rf.pkl")
    prepared = module.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in module.CAAFE_COLS):
        prepared = module.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in module.ASOF_COLS):
        prepared = module.attach_asof_state(prepared, bundle)
    features = module.build_features(prepared, bundle)
    return np.asarray(module.predict_proba(bundle, features), dtype=np.float64)


def _window_adjustment(
    frame: pd.DataFrame, tables: dict[str, dict[int, float]], scale: float
) -> np.ndarray:
    pitcher = frame["pitcher_id"].astype("int64")
    contexts = (
        (frame["pitcher_hand"].astype("int64") == frame["batter_hand"].astype("int64")).to_numpy(),
        (frame["strikes_before"].astype("int64") == 2).to_numpy(),
        (frame["num_runners_on"].astype("int64") > 0).to_numpy(),
    )
    output = np.zeros(len(frame), dtype=np.float64)
    for label, context in zip(("hand", "two", "runner"), contexts):
        magnitude = pitcher.map(tables[label]).fillna(0.0).to_numpy(np.float64)
        output += np.where(context, float(scale) * magnitude, -float(scale) * magnitude)
    return output


def _predict_c3(frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(MODEL_DIR / "c3_sign_all.joblib")
    matrix = np.column_stack(
        [
            _window_adjustment(frame, bundle["tables"][window], bundle["contrast_scale"])
            for window in bundle["window_order"]
        ]
    )
    agreed = np.all(matrix > 0.0, axis=1) | np.all(matrix < 0.0, axis=1)
    return matrix.mean(axis=1) * agreed


def _active_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].astype("int64").eq(ANCHOR_TEAM)
        | frame["batter_team_id"].astype("int64").eq(ANCHOR_TEAM)
    ).to_numpy()
    return regular & ~anchor


def predict_components(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = _predict_v124(frame)
    h1 = _predict_h1(frame)
    c3 = _predict_c3(frame)
    active = _active_mask(frame)
    output = parent.copy()
    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * parent[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    return parent, h1, c3, active, output


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    return predict_components(frame)[-1]


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
        raise ValueError("v142 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v142_v124_h1_c3 | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
