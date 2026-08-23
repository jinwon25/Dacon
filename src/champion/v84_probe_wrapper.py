"""Standalone v84 wrapper: Public-1159 parent plus fixed v56 FM on F."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
ETA = 0.10
EPS = 1e-5


def _load_parent():
    path = MODEL_DIR / "components" / "parent_script.py"
    spec = importlib.util.spec_from_file_location("v84_parent_component", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load v84 parent component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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


def _load_fm(model_dir: Path) -> tuple[dict, dict[str, dict[str, int]], list[np.ndarray]]:
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "V84_FIXED_V56_F_ROUTE_NUMPY_EXPORT_V1":
        raise ValueError("unexpected v84 FM protocol")
    vocabularies = json.loads(
        (model_dir / "vocabularies.json").read_text(encoding="utf-8")
    )
    with np.load(model_dir / "embeddings.npz", allow_pickle=False) as saved:
        embeddings = [
            saved[f"field_{index}"].astype(np.float32)
            for index in range(len(metadata["fields"]))
        ]
    return metadata, vocabularies, embeddings


def _encode(
    frame: pd.DataFrame,
    fields: list[str],
    vocabularies: dict[str, dict[str, int]],
) -> np.ndarray:
    columns = []
    for field in fields:
        values = frame[field].astype("string").fillna("__MISSING__")
        columns.append(
            values.map(vocabularies[field]).fillna(0).to_numpy(np.int64)
        )
    return np.column_stack(columns)


def _raw_fm(
    code: np.ndarray,
    embeddings: list[np.ndarray],
    pair_index: list[list[int]],
) -> np.ndarray:
    output = np.zeros(len(code), dtype=np.float64)
    for left, right in pair_index:
        output += np.sum(
            embeddings[left][code[:, left]] * embeddings[right][code[:, right]],
            axis=1,
            dtype=np.float64,
        )
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(value / (1.0 - value))


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def apply_fixed_v56(
    parent: np.ndarray,
    frame: pd.DataFrame,
    fm_model_dir: Path,
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    if parent.ndim != 1 or len(parent) != len(frame) or not np.isfinite(parent).all():
        raise ValueError("v84 parent prediction is invalid")
    prepared = _prepare_fields(frame)
    metadata, vocabularies, embeddings = _load_fm(fm_model_dir)
    code = _encode(prepared, metadata["fields"], vocabularies)
    raw = _raw_fm(code, embeddings, metadata["pair_index"])
    domains = prepared["domain3"].astype(str).to_numpy()
    centres = metadata["domain_centres"]
    correction = raw - np.asarray(
        [float(centres.get(domain, 0.0)) for domain in domains]
    )
    correction = np.clip(
        correction,
        -float(metadata["correction_clip"]),
        float(metadata["correction_clip"]),
    )
    active = domains == str(metadata["route"])
    output = parent.copy()
    output[active] = _expit(
        _logit(parent[active]) + float(metadata["eta"]) * correction[active]
    )
    return np.clip(output, 0.001, 0.999)


def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
    parent_module = _load_parent()
    parent = np.asarray(parent_module.predict_dataframe(frame), dtype=np.float64)
    return apply_fixed_v56(parent, frame, MODEL_DIR / "v56_fm")


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
        raise ValueError("v84 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v84_fixed_v56_f_route | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
