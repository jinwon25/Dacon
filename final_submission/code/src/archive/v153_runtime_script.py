"""Standalone v153 runtime: May maturity gate over the v142-to-v138 bridge."""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
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
MEAN_RECENT_WEIGHT = 0.85
MATURITY_MONTHS = (5,)
MATURITY_SPEC_NAMES = (
    "v14_refinement_spec.json",
    "v16_residual_spec.json",
    "v20_target1160_spec.json",
    "v21_context_state_eb_spec.json",
    "v22_low_variance_spec.json",
    "v25_postbreak_anchor_spec.json",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maturity_mask(frame: pd.DataFrame) -> np.ndarray:
    month = pd.to_numeric(frame["game_month"], errors="raise").to_numpy(np.int16)
    return np.isin(month, MATURITY_MONTHS)


@contextmanager
def _mature_spec_context(champion_dir: Path):
    alternate = MODEL_DIR / "mature_specs"
    original_read_text = Path.read_text
    replacements = {
        str((champion_dir / name).resolve()): original_read_text(
            alternate / name, encoding="utf-8"
        )
        for name in MATURITY_SPEC_NAMES
    }

    def maturity_read_text(path: Path, *args, **kwargs):
        replacement = replacements.get(str(path.resolve()))
        if replacement is not None:
            return replacement
        return original_read_text(path, *args, **kwargs)

    Path.read_text = maturity_read_text
    try:
        yield
    finally:
        Path.read_text = original_read_text


def _predict_parent_pair(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the expensive common backbone once and branch only six overlays."""

    maturity = _maturity_mask(frame)
    top = _load_module("v153_intermediate_parent", MODEL_DIR / "intermediate" / "script.py")
    v84 = top._load_parent()
    inner = v84._load_parent()
    champion = inner._load_component("champion")
    strict = inner._load_component("strict")

    original_recent = champion.apply_recent_exact_overlay
    downstream_names = (
        "apply_recent_exact_overlay",
        "apply_joint_state_mode_overlay",
        "apply_v20_target1160_overlay",
        "apply_v21_context_state_eb_overlay",
        "apply_v22_low_variance_overlay",
        "apply_v25_postbreak_anchor_overlay",
        "apply_trackman_asof_gate_overlay",
    )
    downstream = {name: getattr(champion, name) for name in downstream_names}
    captured: dict[str, object] = {}

    def capture_pre_recent(probability, *_args, **_kwargs):
        captured["pre_recent"] = np.asarray(probability, dtype=np.float64).copy()
        return probability

    champion.apply_recent_exact_overlay = capture_pre_recent
    for name in downstream_names[1:]:
        setattr(champion, name, lambda probability, *_args, **_kwargs: probability)
    try:
        champion.predict_dataframe(frame)
    finally:
        for name, function in downstream.items():
            setattr(champion, name, function)
    pre_recent = np.asarray(captured["pre_recent"], dtype=np.float64)

    original_v14 = champion.apply_v14_refinement
    original_v16 = champion.apply_v16_residual
    recent_parts: dict[str, object] = {}

    def capture_v14(probability, _frame, exact_lgb, exact_numeric):
        recent_parts["common"] = np.asarray(probability, dtype=np.float64).copy()
        recent_parts["exact_lgb"] = np.asarray(exact_lgb, dtype=np.float64)
        recent_parts["exact_numeric"] = exact_numeric
        return probability

    champion.apply_v14_refinement = capture_v14
    champion.apply_v16_residual = lambda probability, _frame: probability
    try:
        original_recent(pre_recent, frame)
    finally:
        champion.apply_v14_refinement = original_v14
        champion.apply_v16_residual = original_v16
    common = np.asarray(recent_parts["common"], dtype=np.float64)
    exact_lgb = np.asarray(recent_parts["exact_lgb"], dtype=np.float64)
    exact_numeric = recent_parts["exact_numeric"]

    intermediate_champion = original_v14(common, frame, exact_lgb, exact_numeric)
    intermediate_champion = original_v16(intermediate_champion, frame)
    intermediate_champion = champion.apply_joint_state_mode_overlay(intermediate_champion, frame)
    intermediate_champion = champion.apply_v20_target1160_overlay(intermediate_champion, frame)
    intermediate_champion = champion.apply_v21_context_state_eb_overlay(intermediate_champion, frame)
    intermediate_champion = champion.apply_v22_low_variance_overlay(intermediate_champion, frame)
    intermediate_champion = champion.apply_v25_postbreak_anchor_overlay(intermediate_champion, frame)
    ensemble = json.loads(
        (champion.MODEL_DIR / "ensemble.json").read_text(encoding="utf-8")
    )
    intermediate_champion = champion.apply_trackman_asof_gate_overlay(
        intermediate_champion, frame, float(ensemble["global_rate"])
    )

    mature_champion = intermediate_champion[maturity].copy()
    if maturity.any():
        subset = frame.loc[maturity].reset_index(drop=True)
        positions = np.flatnonzero(maturity)
        common_subset = common[maturity]
        exact_subset = exact_lgb[maturity]
        numeric_subset = exact_numeric.iloc[positions].reset_index(drop=True)
        with _mature_spec_context(champion.MODEL_DIR):
            mature_champion = original_v14(
                common_subset, subset, exact_subset, numeric_subset
            )
            mature_champion = original_v16(mature_champion, subset)
            mature_champion = champion.apply_joint_state_mode_overlay(mature_champion, subset)
            mature_champion = champion.apply_v20_target1160_overlay(mature_champion, subset)
            mature_champion = champion.apply_v21_context_state_eb_overlay(mature_champion, subset)
            mature_champion = champion.apply_v22_low_variance_overlay(mature_champion, subset)
            mature_champion = champion.apply_v25_postbreak_anchor_overlay(mature_champion, subset)
            mature_champion = champion.apply_trackman_asof_gate_overlay(
                mature_champion, subset, float(ensemble["global_rate"])
            )

    challenger = inner._strict_predict(frame, strict)
    probe = inner.probe_mask(frame)
    intermediate_inner = inner.blend_predictions(
        intermediate_champion, challenger, probe
    )
    mature_inner = inner.blend_predictions(
        mature_champion, challenger[maturity], probe[maturity]
    ) if maturity.any() else mature_champion

    intermediate_v84 = v84.apply_fixed_v56(
        intermediate_inner, frame, v84.MODEL_DIR / "v56_fm"
    )
    if maturity.any():
        mature_v84 = v84.apply_fixed_v56(
            mature_inner, subset, v84.MODEL_DIR / "v56_fm"
        )
    fm = top.predict_fm_correction(frame)
    conditional = top.predict_conditional_correction(frame)
    intermediate = top.apply_v104(intermediate_v84, frame, fm, conditional)
    mature = intermediate.copy()
    if maturity.any():
        mature[maturity] = top.apply_v104(
            mature_v84, subset, fm[maturity], conditional[maturity]
        )
    return intermediate, mature, maturity


def _predict_h1(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "h1"
    module = _load_module("v153_h1_component", root / "script.py")
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


def predict_components(frame: pd.DataFrame):
    intermediate, mature, maturity = _predict_parent_pair(frame)
    h1 = _predict_h1(frame)
    c3 = _predict_c3(frame)
    active = _active_mask(frame)
    output = intermediate.copy()
    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * intermediate[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    output[maturity] = mature[maturity]
    return intermediate, mature, h1, c3, active, maturity, output


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
        raise ValueError("v153 produced invalid probabilities")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved: {OUTPUT_PATH} | candidate=v153_may_maturity_bridge | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
