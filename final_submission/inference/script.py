"""Standalone v148 runtime: conservative bridge from v142 toward v138."""

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
H1_WEIGHT = 0.16
H1_BASE_WEIGHT = 0.15
C3_WEIGHT = 0.5
MEAN_RECENT_WEIGHT = 0.25
MEAN_RECENT_BASE_WEIGHT = 0.15
BRIDGE_SCALE = 1.2
XGB_ACTIVE_WEIGHT = 0.30
XGB_ACTIVE_THRESHOLD = 0.50
XGB_BOUNDARY_LOW = 0.48
XGB_BOUNDARY_HIGH = 0.50
XGB_BOUNDARY_AGREEMENT = 0.02
XGB_BOUNDARY_WEIGHT = 0.45
XGB_NONPRESSURE_SAME_WEIGHT = 0.15
XGB_NONPRESSURE_OPPOSITE_THRESHOLD = 0.52
XGB_NONPRESSURE_OPPOSITE_WEIGHT = 0.15


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict_parent(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module("jy_original_parent", MODEL_DIR / "v124" / "script.py")
    return np.asarray(module.predict_dataframe(frame), dtype=np.float64)


def _predict_bridge_parent(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module("jy_bridge_parent", MODEL_DIR / "v124_bridge" / "script.py")
    return np.asarray(module.predict_dataframe(frame), dtype=np.float64)


def _predict_h1(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "h1"
    module = _load_module("v148_h1_component", root / "script.py")
    bundle = joblib.load(root / "model" / "rf.pkl")
    prepared = module.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in module.CAAFE_COLS):
        prepared = module.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in module.ASOF_COLS):
        prepared = module.attach_asof_state(prepared, bundle)
    features = module.build_features(prepared, bundle)
    return np.asarray(module.predict_proba(bundle, features), dtype=np.float64)


def _predict_fallback_xgb(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module("jy_fallback_xgb", MODEL_DIR / "fallback_xgb" / "runtime.py")
    return np.asarray(module.predict(frame, MODEL_DIR / "fallback_xgb"), dtype=np.float64)



def _predict_recent_futures(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module(
        "recent_futures_expert", MODEL_DIR / "recent_futures" / "runtime.py"
    )
    return module.predict(frame, MODEL_DIR / "recent_futures")


def _predict_futures_lowrank(frame: pd.DataFrame) -> np.ndarray:
    """Map frozen prior-season OOF matrices using only fields in this row."""

    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int16)
    hand = pd.to_numeric(frame["batter_hand"], errors="raise").to_numpy(np.int16)
    if not (
        np.isin(balls, np.arange(4)).all()
        and np.isin(strikes, np.arange(3)).all()
        and np.isin(hand, (1, 2)).all()
    ):
        raise ValueError("unexpected count or batter-hand value")
    context = ((balls * 3 + strikes) * 2 + (hand - 1)).astype(np.int16)
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(np.int64)
    output = np.zeros(len(frame), dtype=np.float64)
    with np.load(MODEL_DIR / "futures_lowrank" / "lookup.npz", allow_pickle=False) as saved:
        years = saved["source_years"].astype(np.int16)
        for year in years:
            pitcher_ids = saved[f"pitcher_ids_{int(year)}"].astype(np.int64)
            matrix = saved[f"matrix_{int(year)}"].astype(np.float64)
            index = pd.Index(pitcher_ids).get_indexer(pitcher)
            seen = index >= 0
            output[seen] += matrix[index[seen], context[seen]]
    return output / float(len(years))


def _minimum_common_denominator(
    success_rate: float, middle_rate: float, max_denominator: int
) -> tuple[int, bool]:
    rates = np.asarray([success_rate, middle_rate], dtype=np.float64)
    if not np.all(np.isfinite(rates)) or np.any((rates < 0.0) | (rates > 1.0)):
        return 0, False
    denominators = np.arange(1, int(max_denominator) + 1, dtype=np.float64)
    reconstructed = np.rint(denominators[:, None] * rates[None, :])
    reconstructed /= denominators[:, None]
    error = np.max(np.abs(reconstructed - rates[None, :]), axis=1)
    compatible = np.flatnonzero(error <= 0.5e-6 + 1e-12)
    index = int(compatible[0]) if compatible.size else int(np.argmin(error))
    return index + 1, bool(compatible.size)


def _attach_workload_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct v209 denominator features independently for every row."""

    output = frame.copy()
    career_success = pd.to_numeric(
        output["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    career_middle = pd.to_numeric(
        output["asof_pitcher_middle_rate"], errors="coerce"
    ).fillna(0.2).to_numpy(np.float64)
    denominators = {}
    for horizon, max_denominator, shrinkage in (
        (1, 240, 20.0), (3, 600, 60.0), (5, 1000, 100.0)
    ):
        success = pd.to_numeric(
            output[f"asof_pitcher_prev{horizon}_game_success_rate"],
            errors="coerce",
        ).to_numpy(np.float64)
        middle = pd.to_numeric(
            output[f"asof_pitcher_prev{horizon}_game_middle_rate"],
            errors="coerce",
        ).to_numpy(np.float64)
        n = np.zeros(len(output), dtype=np.float64)
        fit = np.zeros(len(output), dtype=bool)
        for index, (success_rate, middle_rate) in enumerate(zip(success, middle)):
            denominator, matched = _minimum_common_denominator(
                success_rate, middle_rate, max_denominator
            )
            n[index] = denominator
            fit[index] = matched
        valid = fit & np.isfinite(success) & np.isfinite(middle) & (n > 0.0)
        denominators[horizon] = n
        reliability = n / (n + shrinkage)
        output[f"workload_prev{horizon}_log_n"] = np.log1p(n).astype(np.float32)
        output[f"workload_prev{horizon}_missing"] = (~valid).astype(np.float32)
        output[f"workload_prev{horizon}_reliability"] = reliability.astype(np.float32)
        output[f"workload_prev{horizon}_success_reliable_delta"] = np.where(
            valid, reliability * (success - career_success), 0.0
        ).astype(np.float32)
        output[f"workload_prev{horizon}_middle_reliable_delta"] = np.where(
            valid, reliability * (middle - career_middle), 0.0
        ).astype(np.float32)
    n1, n3, n5 = (denominators[horizon] for horizon in (1, 3, 5))
    output["workload_prev3_minus_prev1_log"] = np.log1p(
        np.maximum(n3 - n1, 0.0)
    ).astype(np.float32)
    output["workload_prev5_minus_prev3_log"] = np.log1p(
        np.maximum(n5 - n3, 0.0)
    ).astype(np.float32)
    output["workload_prev1_over_prev3"] = np.divide(
        n1, n3, out=np.zeros(len(output)), where=n3 > 0.0
    ).astype(np.float32)
    output["workload_prev3_over_prev5"] = np.divide(
        n3, n5, out=np.zeros(len(output)), where=n5 > 0.0
    ).astype(np.float32)
    return output


def _predict_workload_h1(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "workload_h1"
    module = _load_module("v343_workload_h1_component", MODEL_DIR / "h1" / "script.py")
    bundle = joblib.load(root / "model" / "rf.pkl")
    prepared = module.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in module.CAAFE_COLS):
        prepared = module.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in module.ASOF_COLS):
        prepared = module.attach_asof_state(prepared, bundle)
    prepared = _attach_workload_features(prepared)
    features = module.build_features(prepared, bundle)
    return np.asarray(module.predict_proba(bundle, features), dtype=np.float64)


def _beta_cell_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    support = pd.to_numeric(
        frame["asof_pitcher_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    pitchmix_n = pd.to_numeric(
        frame["asof_pitcher_pitchmix_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    pitchmix = frame[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy(np.float64)
    mixed = (np.max(pitchmix, axis=1) < 0.50) & (pitchmix_n >= 100.0)
    developing = (support >= 100.0) & (support < 800.0)
    return regular & ~anchor & developing & mixed


def _beta_season_posterior(
    frame: pd.DataFrame,
    bundle: dict,
    prefix: str,
    prior: np.ndarray,
) -> np.ndarray:
    identifier = pd.to_numeric(
        frame[f"{prefix}_id"], errors="raise"
    ).astype("int64")
    opening_n = identifier.map(bundle[f"{prefix}_opening_n"]).fillna(0.0).to_numpy(np.float64)
    opening_success = identifier.map(
        bundle[f"{prefix}_opening_success"]
    ).fillna(0.0).to_numpy(np.float64)
    career_n = pd.to_numeric(
        frame[f"asof_{prefix}_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    career_rate = pd.to_numeric(
        frame[f"asof_{prefix}_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    season_n = np.maximum(career_n - opening_n, 0.0)
    season_success = np.clip(
        career_n * career_rate - opening_success, 0.0, season_n
    )
    concentration = float(bundle["concentration"])
    return (season_success + concentration * prior) / (season_n + concentration)


def _predict_beta_cell(frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(MODEL_DIR / "beta_cell" / "spec.joblib")
    prior = frame["game_type"].astype(str).map(
        bundle["prior_by_game_type"]
    ).fillna(float(bundle["global_prior"])).to_numpy(np.float64)
    pitcher = _beta_season_posterior(frame, bundle, "pitcher", prior)
    batter = _beta_season_posterior(frame, bundle, "batter", prior)
    career = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    career = np.where(np.isfinite(career), career, prior)
    recent = pd.to_numeric(
        frame["asof_pitcher_prev5_game_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    recent = np.where(np.isfinite(recent), recent, career)
    matrix = np.clip(
        np.column_stack((pitcher, batter, prior, career, recent)),
        0.001,
        0.999,
    )
    probability = matrix @ np.asarray(bundle["weights"], dtype=np.float64)
    return np.clip(probability, 0.001, 0.999)


def _predict_player_transition(frame: pd.DataFrame) -> np.ndarray:
    """Frozen 2025 pitcher transition/count correction; each row maps alone."""

    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(np.int64)
    current_team = pd.to_numeric(
        frame["pitcher_team_id"], errors="raise"
    ).to_numpy(np.int64)
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int16)
    game_type = frame["game_type"].astype(str).to_numpy()
    with np.load(
        MODEL_DIR / "player_transition" / "lookup.npz", allow_pickle=False
    ) as saved:
        known_pitchers = saved["pitcher_ids"].astype(np.int64)
        index = pd.Index(known_pitchers).get_indexer(pitcher)
        seen = index >= 0
        last_year = np.full(len(frame), -1, dtype=np.int16)
        previous_team = np.full(len(frame), -1, dtype=np.int64)
        last_year[seen] = saved["last_year"].astype(np.int16)[index[seen]]
        previous_team[seen] = saved["previous_team"].astype(np.int64)[index[seen]]
        status = np.full(len(frame), "SWITCH", dtype="<U6")
        status[~seen] = "NEW"
        status[seen & (last_year < 2024)] = "RETURN"
        status[seen & (last_year == 2024) & (previous_team == current_team)] = "SAME"
        count = np.char.add(
            np.char.add(balls.astype(str), "-"), strikes.astype(str)
        )
        query_key = np.char.add(
            np.char.add(np.char.add(np.char.add(status, "|"), count), "|"),
            game_type,
        )
        correction = pd.Series(
            saved["corrections"].astype(np.float64),
            index=saved["correction_keys"].astype(str),
        ).reindex(query_key).fillna(0.0).to_numpy(np.float64)
    return correction


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


def _predict_c3(frame: pd.DataFrame, recent_weight: float = MEAN_RECENT_WEIGHT) -> np.ndarray:
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
    return (1.0 - recent_weight) * sign_all + recent_weight * mean_recent


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
    bridge_parent = _predict_bridge_parent(frame)
    h1 = _predict_h1(frame)
    c3_base = _predict_c3(frame, MEAN_RECENT_BASE_WEIGHT)
    c3 = _predict_c3(frame, MEAN_RECENT_WEIGHT)
    base_active = _active_mask(frame)
    runners_on = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy() > 0
    high_li = pd.to_numeric(frame["li"], errors="coerce").fillna(0.0).to_numpy() >= 1.5
    pressure_count = (
        (pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy() >= 3)
        | (pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy() >= 2)
    )
    active = base_active & (runners_on | high_li)
    effective_bridge = parent + BRIDGE_SCALE * (bridge_parent - parent)
    output = parent.copy()
    output[base_active] = np.clip(
        (1.0 - H1_BASE_WEIGHT) * parent[base_active]
        + H1_BASE_WEIGHT * h1[base_active]
        + C3_WEIGHT * c3_base[base_active],
        0.001,
        0.999,
    )
    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * effective_bridge[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    jy_probability = output.copy()
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    rcore = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    pressure = rcore & (
        (frame["num_runners_on"].to_numpy(dtype=np.float64) > 0.0)
        | (frame["li"].to_numpy(dtype=np.float64) >= 1.5)
    )
    same_hand = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).to_numpy()
    xgb_probability = _predict_fallback_xgb(frame)
    deployed = pressure & (jy_probability >= XGB_ACTIVE_THRESHOLD)
    boundary = (
        pressure
        & (jy_probability >= XGB_BOUNDARY_LOW)
        & (jy_probability < XGB_BOUNDARY_HIGH)
        & (np.abs(xgb_probability - jy_probability) <= XGB_BOUNDARY_AGREEMENT)
    )
    nonpressure_same = rcore & ~pressure & same_hand
    nonpressure_opposite = (
        rcore
        & ~pressure
        & ~same_hand
        & (jy_probability >= XGB_NONPRESSURE_OPPOSITE_THRESHOLD)
    )
    routes = (
        (deployed, XGB_ACTIVE_WEIGHT),
        (boundary, XGB_BOUNDARY_WEIGHT),
        (nonpressure_same, XGB_NONPRESSURE_SAME_WEIGHT),
        (nonpressure_opposite, XGB_NONPRESSURE_OPPOSITE_WEIGHT),
    )
    route_sum = np.column_stack([mask for mask, _weight in routes]).sum(axis=1)
    if np.any(route_sum > 1):
        raise RuntimeError("fallback routes overlap")
    for mask, weight in routes:
        output[mask] = np.clip(
            jy_probability[mask]
            + weight * (xgb_probability[mask] - jy_probability[mask]),
            0.001,
            0.999,
        )
    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    lowrank_probability_delta = None
    if np.any(futures):
        futures_probability = _predict_recent_futures(frame)
        lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.20 * (futures_probability[futures] - output[futures])
            + 0.50 * lowrank_probability_delta[futures],
            0.001,
            0.999,
        )
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    if np.any(anchor):
        if lowrank_probability_delta is None:
            lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[anchor] = np.clip(
            output[anchor] + 0.50 * lowrank_probability_delta[anchor],
            0.001,
            0.999,
        )
    v335_probability = output.copy()
    rcore = regular & ~anchor
    if np.any(rcore):
        transition_delta = _predict_player_transition(frame)
        output[rcore] = np.clip(
            output[rcore] + 0.25 * transition_delta[rcore],
            0.001,
            0.999,
        )
    workload_active = active & rcore
    if np.any(workload_active):
        workload_h1 = _predict_workload_h1(
            frame.loc[workload_active].reset_index(drop=True)
        )
        workload_proposal = np.clip(
            0.82 * effective_bridge[workload_active]
            + 0.18 * workload_h1
            + C3_WEIGHT * c3[workload_active],
            0.001,
            0.999,
        )
        workload_delta = workload_proposal - jy_probability[workload_active]
        output[workload_active] = np.clip(
            output[workload_active] + workload_delta,
            0.001,
            0.999,
        )
    beta_active = _beta_cell_mask(frame)
    if np.any(beta_active):
        beta_probability = _predict_beta_cell(
            frame.loc[beta_active].reset_index(drop=True)
        )
        beta_delta = 0.10 * (
            beta_probability - v335_probability[beta_active]
        )
        output[beta_active] = np.clip(
            output[beta_active] + beta_delta,
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
        f"Saved: {OUTPUT_PATH} | candidate=v148_v142_v138_bridge | "
        f"rows={len(sample)} | mean={probability.mean():.6f} | "
        f"min={probability.min():.6f} | max={probability.max():.6f}"
    )


if __name__ == "__main__":
    main()
