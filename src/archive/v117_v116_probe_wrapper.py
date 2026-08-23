"""Standalone v116 failure-prior probe above the frozen Public-1162 v104 package."""

from __future__ import annotations

import importlib.util
import json
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


def _load_component(name: str, filename: str):
    path = MODEL_DIR / "components" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_parent():
    module = _load_component("v116_parent_component", "parent_script.py")
    module.MODEL_DIR = MODEL_DIR / "parent"
    return module


def _domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].eq("R").to_numpy()
    anchor = frame["pitcher_team_id"].eq(ANCHOR_TEAM).to_numpy() | frame[
        "batter_team_id"
    ].eq(ANCHOR_TEAM).to_numpy()
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def predict_failure_correction(frame: pd.DataFrame) -> np.ndarray:
    component = _load_component("v116_failure_prior_component", "failure_prior.py")
    root = MODEL_DIR / "failure_prior"
    bank = joblib.load(root / "bank.joblib")
    spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    rows = frame.copy()
    rows["count_state"] = (
        pd.to_numeric(rows["balls_before"], errors="raise").astype(np.int8) * 3
        + pd.to_numeric(rows["strikes_before"], errors="raise").astype(np.int8)
    )
    parts = component.predict_failure_components(rows, bank)
    weights = np.asarray(spec["weights"], dtype=np.float64)
    if parts.shape != (len(frame), len(weights)):
        raise ValueError("failure-prior component shape mismatch")
    return parts @ weights


def apply_v116(
    parent: np.ndarray, frame: pd.DataFrame, correction: np.ndarray
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if parent.shape != correction.shape or parent.shape != (len(frame),):
        raise ValueError("v116 prediction shape mismatch")
    active = _domain(frame) == "R_CORE"
    return np.clip(parent + np.where(active, correction, 0.0), 0.001, 0.999)


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    parent = np.asarray(_load_parent().predict_dataframe(frame), dtype=np.float64)
    correction = predict_failure_correction(frame)
    return apply_v116(parent, frame, correction)


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
        raise ValueError("v116 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v116_failure_prior | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
