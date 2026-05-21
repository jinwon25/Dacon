import argparse
import json
import os
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
GROUP_COL = "scenario_id"
CAT_COLS = ["layout_id", "layout_type", "layout_cluster"]

# layout_info.csv에서 가져오는 정적 메타 컬럼 (각 layout_id마다 고정값).
# adversarial AUC=1.0 진단 결과, 이 피처들의 조합이 train/test layout을 완벽 분리.
# train/test layout이 거의 disjoint이므로 직접 사용하면 일반화 저하 → 선택적으로 드롭 가능.
LAYOUT_STATIC_COLS = [
    "aisle_width_avg", "intersection_count", "one_way_ratio",
    "pack_station_count", "charger_count", "layout_compactness",
    "zone_dispersion", "robot_total", "building_age_years",
    "floor_area_sqm", "ceiling_height_m", "fire_sprinkler_count",
    "emergency_exit_count",
]
# layout 정적 메타에서 파생된 컬럼 (피처 엔지니어링 결과)
LAYOUT_DERIVED_COLS = [
    "station_density_layout", "intersection_density_layout",
    "robot_density_layout", "pack_load_per_station",
    "charger_load",
]

# adversarial validation 결과 train/test를 가르는 추가 누설 피처.
# - robot_total_observed: layout 고정 로봇 합계 (식별자)
# - layout_type: 4-카테고리 분포 차이
# - sku_concentration: 정적성
# - *_seq_mean/std/trend/vs_seq_mean: 시나리오 단위 집계 → layout 요약치
# 보존: *_seq_interp (값 수준 결측 보간이라 식별자로 작용 약함)
LEAKY_EXTRA_COLS = ["robot_total_observed", "layout_type", "sku_concentration"]
LEAKY_SEQ_SUFFIXES = ("_seq_mean", "_seq_std", "_seq_trend", "_vs_seq_mean")

LAG_COLS = [
    "order_inflow_15m",
    "unique_sku_15m",
    "avg_items_per_order",
    "robot_active",
    "robot_idle",
    "robot_charging",
    "robot_utilization",
    "battery_mean",
    "battery_std",
    "low_battery_ratio",
    "charge_queue_length",
    "avg_charge_wait",
    "congestion_score",
    "max_zone_density",
    "blocked_path_15m",
    "near_collision_15m",
    "fault_count_15m",
    "avg_recovery_time",
    "pack_utilization",
    "loading_dock_util",
    "staging_area_util",
    "kpi_otd_pct",
]

SEQ_COLS = [
    "order_inflow_15m",
    "unique_sku_15m",
    "avg_items_per_order",
    "robot_active",
    "robot_idle",
    "robot_charging",
    "robot_utilization",
    "battery_mean",
    "low_battery_ratio",
    "charge_queue_length",
    "avg_charge_wait",
    "congestion_score",
    "max_zone_density",
    "blocked_path_15m",
    "near_collision_15m",
    "fault_count_15m",
    "avg_recovery_time",
    "pack_utilization",
    "loading_dock_util",
    "staging_area_util",
    "outbound_truck_wait_min",
    "kpi_otd_pct",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    return left / right.replace(0, np.nan)


def add_features(df: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(layout, on="layout_id", how="left")
    out["_origin_order"] = np.arange(len(out))

    sort_cols = [GROUP_COL, ID_COL] if ID_COL in out.columns else [GROUP_COL]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    group = out.groupby(GROUP_COL, sort=False)

    out["slot"] = group.cumcount().astype(np.int16)
    out["slot_sin"] = np.sin(2 * np.pi * out["slot"] / 25)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot"] / 25)
    out["is_early_slot"] = (out["slot"] <= 3).astype(np.int8)
    out["is_late_slot"] = (out["slot"] >= 20).astype(np.int8)

    # 대회 규칙상 같은 시나리오의 25개 슬롯은 추론 시 모두 볼 수 있다.
    # 원본 결측은 남겨두고, 시나리오 내부 보간/요약값만 별도 피처로 추가한다.
    seq_features = {}
    for col in SEQ_COLS:
        if col not in out.columns:
            continue
        filled = group[col].transform(lambda s: s.interpolate(limit_direction="both"))
        mean = group[col].transform("mean")
        first = group[col].transform("first")
        last = group[col].transform("last")
        seq_features[f"{col}_seq_interp"] = filled
        seq_features[f"{col}_seq_mean"] = mean
        seq_features[f"{col}_seq_std"] = group[col].transform("std")
        seq_features[f"{col}_seq_trend"] = last - first
        seq_features[f"{col}_vs_seq_mean"] = out[col] - mean
        # within-scenario rank (분포 자유, layout disjoint에 강건)
        seq_features[f"{col}_seq_rank"] = group[col].rank(method="average", pct=True)
        # anomaly: |x - median| / MAD (robust z-score)
        median = group[col].transform("median")
        mad = group[col].transform(lambda s: (s - s.median()).abs().median())
        seq_features[f"{col}_seq_anomaly"] = (out[col] - median).abs() / mad.replace(0, np.nan)
    if seq_features:
        out = pd.concat([out, pd.DataFrame(seq_features, index=out.index)], axis=1)

    robot_total = out["robot_active"] + out["robot_idle"] + out["robot_charging"]
    out["robot_total_observed"] = robot_total
    out["active_robot_share"] = safe_divide(out["robot_active"], robot_total)
    out["idle_robot_share"] = safe_divide(out["robot_idle"], robot_total)
    out["charging_robot_share"] = safe_divide(out["robot_charging"], robot_total)
    out["orders_per_active_robot"] = safe_divide(out["order_inflow_15m"], out["robot_active"])
    out["orders_per_total_robot"] = safe_divide(out["order_inflow_15m"], robot_total)
    out["sku_per_order"] = safe_divide(out["unique_sku_15m"], out["order_inflow_15m"])
    out["estimated_item_inflow"] = out["order_inflow_15m"] * out["avg_items_per_order"]

    out["pack_load_per_station"] = safe_divide(out["order_inflow_15m"], out["pack_station_count"])
    out["charger_load"] = safe_divide(out["robot_charging"] + out["charge_queue_length"], out["charger_count"])
    out["staff_load"] = safe_divide(out["order_inflow_15m"], out["staff_on_floor"])
    out["forklift_load"] = safe_divide(out["order_inflow_15m"], out["forklift_active_count"] + 1)
    out["robot_density_layout"] = safe_divide(out["robot_total"], out["floor_area_sqm"])
    out["station_density_layout"] = safe_divide(out["pack_station_count"], out["floor_area_sqm"])
    out["intersection_density_layout"] = safe_divide(out["intersection_count"], out["floor_area_sqm"])

    out["battery_pressure"] = out["low_battery_ratio"] * robot_total
    out["charge_pressure"] = out["battery_pressure"] + out["charge_queue_length"]
    out["congestion_load"] = out["congestion_score"] * out["order_inflow_15m"]
    out["zone_load"] = out["max_zone_density"] * out["order_inflow_15m"]
    out["dock_pressure"] = out["loading_dock_util"] * out["outbound_truck_wait_min"]
    out["picking_complexity"] = out["pick_list_length_avg"] * out["sku_concentration"]
    out["cold_chain_load"] = out["order_inflow_15m"] * out["cold_chain_ratio"]
    out["heavy_item_load"] = out["order_inflow_15m"] * out["heavy_item_ratio"]
    out["urgent_order_load"] = out["order_inflow_15m"] * out["urgent_order_ratio"]
    out["quality_backorder_pressure"] = out["quality_check_rate"] * out["backorder_ratio"]

    lag_features = {}
    for col in LAG_COLS:
        if col not in out.columns:
            continue
        lag1 = group[col].shift(1)
        lag_features[f"{col}_lag1"] = lag1
        lag_features[f"{col}_lag2"] = group[col].shift(2)
        lag_features[f"{col}_diff1"] = out[col] - lag1
        lag_features[f"{col}_roll3_mean"] = (
            lag1.groupby(out[GROUP_COL], sort=False)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        # 더 긴 윈도우 추가 (전문가 권고: rolling 5/12)
        lag_features[f"{col}_roll5_mean"] = (
            lag1.groupby(out[GROUP_COL], sort=False)
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        lag_features[f"{col}_roll5_std"] = (
            lag1.groupby(out[GROUP_COL], sort=False)
            .rolling(5, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )
    if lag_features:
        out = pd.concat([out, pd.DataFrame(lag_features, index=out.index)], axis=1)

    out = out.sort_values("_origin_order").drop(columns=["_origin_order"]).reset_index(drop=True)

    for col in CAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out


def make_folds(train: pd.DataFrame, n_splits: int, seed: int, group_col: str = None):
    """group_col 기본값은 GROUP_COL(scenario_id). 'layout_id' 지정 시 layout 단위 disjoint CV."""
    if group_col is None:
        group_col = GROUP_COL
    groups = train[group_col].to_numpy()
    y = train[TARGET].to_numpy()
    try:
        bins = pd.qcut(y, q=20, labels=False, duplicates="drop")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(train, bins, groups)
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        yield from splitter.split(train, y, groups)


def train_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    n_splits: int,
    seed: int,
    objective: str,
    log_target: bool = False,
    tweedie_variance_power: float | None = None,
    huber_alpha: float | None = None,
    sample_weight: np.ndarray | None = None,
    group_by_layout: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    importances = []

    params = {
        "objective": objective,
        "n_estimators": 8000,
        "learning_rate": 0.025,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if tweedie_variance_power is not None:
        params["tweedie_variance_power"] = tweedie_variance_power
    if huber_alpha is not None:
        params["alpha"] = huber_alpha

    fold_group_col = "layout_id" if group_by_layout else None
    for fold, (tr_idx, val_idx) in enumerate(make_folds(train, n_splits, seed, fold_group_col), start=1):
        X_tr = train.iloc[tr_idx][feature_cols]
        y_tr_raw = train.iloc[tr_idx][TARGET]
        X_val = train.iloc[val_idx][feature_cols]
        y_val_raw = train.iloc[val_idx][TARGET]

        if log_target:
            y_tr = np.log1p(y_tr_raw.clip(lower=0))
            y_val = np.log1p(y_val_raw.clip(lower=0))
        else:
            y_tr = y_tr_raw
            y_val = y_val_raw

        fit_kwargs = dict(
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, first_metric_only=True),
                lgb.log_evaluation(period=100),
            ],
        )
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[tr_idx]
            fit_kwargs["eval_sample_weight"] = [sample_weight[val_idx]]

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, **fit_kwargs)

        val_pred_raw = model.predict(X_val, num_iteration=model.best_iteration_)
        test_pred_fold = model.predict(test[feature_cols], num_iteration=model.best_iteration_)
        if log_target:
            val_pred = np.expm1(val_pred_raw)
            test_pred_fold = np.expm1(test_pred_fold)
        else:
            val_pred = val_pred_raw
        oof[val_idx] = val_pred
        test_pred += test_pred_fold / n_splits

        fold_mae = mean_absolute_error(y_val_raw, val_pred)
        print(f"fold {fold} MAE: {fold_mae:.6f}, best_iteration: {model.best_iteration_}")

        importances.append(
            {
                "fold": fold,
                "features": feature_cols,
                "gain": model.booster_.feature_importance(importance_type="gain").tolist(),
                "split": model.booster_.feature_importance(importance_type="split").tolist(),
            }
        )

    return oof, test_pred, importances


def save_importance(importances: list[dict], output_dir: Path) -> None:
    rows = []
    for item in importances:
        for feature, gain, split in zip(item["features"], item["gain"], item["split"]):
            rows.append(
                {
                    "fold": item["fold"],
                    "feature": feature,
                    "gain": gain,
                    "split": split,
                }
            )
    imp = pd.DataFrame(rows)
    summary = (
        imp.groupby("feature", as_index=False)[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    summary.to_csv(output_dir / "feature_importance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="regression_l1")
    parser.add_argument("--drop-layout-id", action="store_true")
    parser.add_argument("--drop-layout-features", action="store_true",
                        help="layout_info에서 가져온 정적 메타 + 파생 피처 모두 드롭 (adversarial AUC 진단 결과 반영)")
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--tweedie-variance-power", type=float, default=None)
    parser.add_argument("--huber-alpha", type=float, default=None)
    parser.add_argument("--use-cache", action="store_true",
                        help="data/cache/{train,test}_features.parquet에서 피처 로드")
    parser.add_argument("--use-cluster-cache", action="store_true",
                        help="layout cluster 피처 추가된 캐시 사용 (data/cache/*_features_cluster.parquet)")
    parser.add_argument("--sample-weights", type=str, default=None,
                        help="train ID별 sample_weight CSV 경로 (ID, sample_weight 컬럼)")
    parser.add_argument("--drop-leaky-features", action="store_true",
                        help="adversarial 식별 누설 피처(robot_total_observed, layout_type, sku_concentration, *_seq_mean/std/trend/vs_seq_mean) 드롭")
    parser.add_argument("--group-by-layout", action="store_true",
                        help="layout_id 단위 GroupKFold (LB 분포 시뮬레이션)")
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = project_path(args.data_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading data")
    train_raw = pd.read_csv(data_dir / "train.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if args.use_cluster_cache:
        print("loading cluster-enhanced cached features")
        from feature_cache import load_cached
        train, test = load_cached(data_dir, cluster=True)
    elif args.use_cache:
        print("loading cached features")
        from feature_cache import load_cached
        train, test = load_cached(data_dir)
    else:
        test_raw = pd.read_csv(data_dir / "test.csv")
        layout = pd.read_csv(data_dir / "layout_info.csv")
        print("building features")
        train = add_features(train_raw, layout)
        test = add_features(test_raw, layout)

    drop_cols = [ID_COL, GROUP_COL, TARGET]
    if args.drop_layout_id:
        drop_cols.append("layout_id")
    if args.drop_layout_features:
        drop_cols.extend(LAYOUT_STATIC_COLS)
        drop_cols.extend(LAYOUT_DERIVED_COLS)
    if args.drop_leaky_features:
        drop_cols.extend(LEAKY_EXTRA_COLS)
        drop_cols.extend([c for c in train.columns if c.endswith(LEAKY_SEQ_SUFFIXES)])
    feature_cols = [c for c in train.columns if c not in drop_cols]
    categorical_cols = [c for c in CAT_COLS if c in feature_cols]

    print(f"train shape: {train.shape}, test shape: {test.shape}")
    print(f"feature count: {len(feature_cols)}, categorical: {categorical_cols}")

    sample_weight = None
    if args.sample_weights:
        sample_weights_path = project_path(args.sample_weights)
        print(f"loading sample weights from {sample_weights_path}")
        sw_df = pd.read_csv(sample_weights_path)
        sw_df = sw_df.set_index(ID_COL).loc[train_raw[ID_COL].to_numpy()]
        sample_weight = sw_df["sample_weight"].to_numpy()
        print(f"  weights stats: mean={sample_weight.mean():.4f}, p50={np.median(sample_weight):.4f}, p90={np.quantile(sample_weight, 0.9):.4f}, max={sample_weight.max():.4f}")

    oof, pred, importances = train_lgbm(
        train=train,
        test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        n_splits=args.n_splits,
        seed=args.seed,
        objective=args.objective,
        log_target=args.log_target,
        tweedie_variance_power=args.tweedie_variance_power,
        huber_alpha=args.huber_alpha,
        sample_weight=sample_weight,
        group_by_layout=args.group_by_layout,
    )

    pred = np.clip(pred, 0, None)
    oof_clipped = np.clip(oof, 0, None)
    mae = mean_absolute_error(train[TARGET], oof_clipped)
    print(f"OOF MAE clipped: {mae:.6f}")

    sample[TARGET] = pred
    sample.to_csv(output_dir / "submission.csv", index=False)

    oof_df = train_raw[[ID_COL, GROUP_COL, "layout_id", TARGET]].copy()
    oof_df["pred"] = oof_clipped
    oof_df["abs_error"] = (oof_df[TARGET] - oof_df["pred"]).abs()
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    save_importance(importances, output_dir)

    metadata = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        "objective": args.objective,
        "log_target": args.log_target,
        "tweedie_variance_power": args.tweedie_variance_power,
        "huber_alpha": args.huber_alpha,
        "drop_layout_id": args.drop_layout_id,
        "drop_layout_features": args.drop_layout_features,
        "drop_leaky_features": args.drop_leaky_features,
        "sample_weights": args.sample_weights,
        "use_cache": args.use_cache,
        "oof_mae": mae,
        "feature_count": len(feature_cols),
        "categorical_cols": categorical_cols,
        "python": os.sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
