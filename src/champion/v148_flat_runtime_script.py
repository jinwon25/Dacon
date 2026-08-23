"""Standalone v148 runtime in the flat single-generation layout.

Prediction-identical to the original ``submit_v148.zip`` runtime.  The only
change is structural: components are imported statically from ``lib`` instead
of being re-executed through ``importlib.util.spec_from_file_location`` at
every call, and every model artifact lives one level under ``model/``.

The submission contract fixes the archive root to ``model/``, ``script.py`` and
``requirements.txt``, so the component package ships as ``model/lib/``.  That
directory is put on ``sys.path`` below -- before any component import and
relative to this file rather than the working directory, because the
evaluation server does not guarantee which directory it runs from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "model") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "model"))

from lib import conditional_overlay, form_context_rf  # noqa: E402
from lib.paths import MODEL_ROOT  # noqa: E402


ID_COL = "row_id"
TARGET_COL = "control_success"
TEST_PATH = BASE_DIR / "data" / "test.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"
OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"
ANCHOR_TEAM = 13
H1_WEIGHT = 0.15
C3_WEIGHT = 0.5
MEAN_RECENT_WEIGHT = 0.15


def _predict_parent(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(conditional_overlay.predict_dataframe(frame), dtype=np.float64)


def _predict_h1(frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(MODEL_ROOT / "form_context_rf" / "rf.pkl")
    prepared = form_context_rf.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in form_context_rf.CAAFE_COLS):
        prepared = form_context_rf.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in form_context_rf.ASOF_COLS):
        prepared = form_context_rf.attach_asof_state(prepared, bundle)
    features = form_context_rf.build_features(prepared, bundle)
    return np.asarray(form_context_rf.predict_proba(bundle, features), dtype=np.float64)


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
    bundle = joblib.load(MODEL_ROOT / "c3_sign_all.joblib")
    matrix = np.column_stack(
        [
            _window_adjustment(frame, bundle["tables"][window], bundle["contrast_scale"])
            for window in bundle["window_order"]
        ]
    )
    agreed = np.all(matrix > 0.0, axis=1) | np.all(matrix < 0.0, axis=1)
    sign_all = matrix.mean(axis=1) * agreed
    mean_recent = matrix[:, :2].mean(axis=1)
    return (1.0 - MEAN_RECENT_WEIGHT) * sign_all + MEAN_RECENT_WEIGHT * mean_recent


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
    parent = _predict_parent(frame)
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
        raise ValueError("v148 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v148_flat | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
