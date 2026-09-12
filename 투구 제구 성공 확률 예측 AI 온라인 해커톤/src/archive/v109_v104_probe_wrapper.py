"""Standalone v104 probe above the frozen Public-1161 v84 package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
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
EPS = 1e-5
FM_ETA = 0.10
CONDITIONAL_ETA = 0.10
SAFE = {
    "count": ["0-0", "1-1", "1-2", "2-0", "3-1", "3-2"],
    "history": ["1000+", "200-999", "30-199"],
    "platoon": ["1-1", "2-1", "2-2"],
}


def _load_component(name: str, filename: str):
    path = MODEL_DIR / "components" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_parent():
    module = _load_component("v104_parent_component", "parent_script.py")
    module.MODEL_DIR = MODEL_DIR / "parent"
    return module


def _prepare_fields(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    regular = output["game_type"].eq("R")
    anchor = output["pitcher_team_id"].eq(ANCHOR_TEAM) | output[
        "batter_team_id"
    ].eq(ANCHOR_TEAM)
    output["domain3"] = np.where(
        ~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE")
    )
    balls = output["balls_before"].to_numpy()
    strikes = output["strikes_before"].to_numpy()
    output["pressure"] = np.where(
        balls == 3, "threeball", np.where(strikes == 2, "twostrike", "normal")
    )
    output["count_state"] = (
        output["balls_before"].astype("Int64").astype(str)
        + "-"
        + output["strikes_before"].astype("Int64").astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__")
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__")
    )
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string")
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(value / (1.0 - value))


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _load_fm(path: Path) -> tuple[dict, dict[str, dict[str, int]], list[np.ndarray]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "V109_INDEPENDENT_R_FM_NUMPY_V1":
        raise ValueError("unexpected v109 FM protocol")
    vocabularies = json.loads((path / "vocabularies.json").read_text(encoding="utf-8"))
    with np.load(path / "embeddings.npz", allow_pickle=False) as saved:
        embeddings = [
            saved[f"field_{index}"].astype(np.float32)
            for index in range(len(metadata["fields"]))
        ]
    return metadata, vocabularies, embeddings


def _fm_correction(frame: pd.DataFrame, path: Path) -> np.ndarray:
    prepared = _prepare_fields(frame)
    metadata, vocabularies, embeddings = _load_fm(path)
    columns = []
    for field in metadata["fields"]:
        values = prepared[field].astype("string").fillna("__MISSING__")
        columns.append(values.map(vocabularies[field]).fillna(0).to_numpy(np.int64))
    code = np.column_stack(columns)
    raw = np.zeros(len(code), dtype=np.float64)
    for left, right in metadata["pair_index"]:
        raw += np.sum(
            embeddings[left][code[:, left]] * embeddings[right][code[:, right]],
            axis=1,
            dtype=np.float64,
        )
    domains = prepared["domain3"].astype(str).to_numpy()
    correction = raw - np.asarray(
        [float(metadata["domain_centres"].get(domain, 0.0)) for domain in domains]
    )
    return np.clip(
        correction,
        -float(metadata["correction_clip"]),
        float(metadata["correction_clip"]),
    )


def predict_fm_correction(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "r_fm"
    older = _fm_correction(frame, root / "older")
    recent = _fm_correction(frame, root / "recent")
    return np.clip(0.5 * (older + recent), -0.25, 0.25)


def predict_conditional_correction(frame: pd.DataFrame) -> np.ndarray:
    component = _load_component("v104_feature_component", "v104_features.py")
    root = MODEL_DIR / "conditional"
    bank = joblib.load(root / "bank.joblib")
    spec = json.loads((root / "feature_spec.json").read_text(encoding="utf-8"))
    conditional = component.feature_frame(frame, bank)
    baseline = conditional.drop(columns=list(component.CONDITIONAL_COLUMNS))
    conditional = component.apply_model_spec(conditional, spec["conditional"])
    baseline = component.apply_model_spec(baseline, spec["baseline"])
    conditional_model = lgb.Booster(model_file=str(root / "conditional_lgb.txt"))
    baseline_model = lgb.Booster(model_file=str(root / "baseline_lgb.txt"))
    conditional_prediction = conditional_model.predict(conditional)
    baseline_prediction = baseline_model.predict(baseline)
    return np.clip(conditional_prediction, 0.001, 0.999) - np.clip(
        baseline_prediction, 0.001, 0.999
    )


def stability_mask(frame: pd.DataFrame) -> np.ndarray:
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int8)
    count = np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))
    history = pd.cut(
        pd.to_numeric(frame["asof_pitcher_n"], errors="raise"),
        bins=[-np.inf, 29, 199, 999, np.inf],
        labels=["0-29", "30-199", "200-999", "1000+"],
    ).astype(str).to_numpy()
    platoon = np.char.add(
        np.char.add(frame["pitcher_hand"].astype(str).to_numpy(), "-"),
        frame["batter_hand"].astype(str).to_numpy(),
    )
    votes = (
        np.isin(count, SAFE["count"]).astype(np.int8)
        + np.isin(history, SAFE["history"]).astype(np.int8)
        + np.isin(platoon, SAFE["platoon"]).astype(np.int8)
    )
    return votes >= 2


def apply_v104(
    parent: np.ndarray,
    frame: pd.DataFrame,
    fm_correction: np.ndarray,
    conditional_correction: np.ndarray,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    prepared = _prepare_fields(frame)
    domains = prepared["domain3"].astype(str).to_numpy()
    active = domains == "R_CORE"
    raw = parent.copy()
    raw[active] = _expit(_logit(parent[active]) + FM_ETA * fm_correction[active])
    raw[active] = np.clip(
        raw[active] + CONDITIONAL_ETA * conditional_correction[active],
        0.001,
        0.999,
    )
    mask = stability_mask(frame)
    return np.where(mask, raw, parent).astype(np.float64)


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    parent = np.asarray(_load_parent().predict_dataframe(frame), dtype=np.float64)
    fm = predict_fm_correction(frame)
    conditional = predict_conditional_correction(frame)
    return np.clip(apply_v104(parent, frame, fm, conditional), 0.001, 0.999)


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
        raise ValueError("v104 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v104_source_stability_mask | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
