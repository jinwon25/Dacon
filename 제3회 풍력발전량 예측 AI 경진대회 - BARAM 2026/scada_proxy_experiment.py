from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.features import TIME_COL, build_features
from src.metrics import CAPACITY_KWH, evaluate_competition, evaluate_group
from train import calibrate, make_catboost_model, make_model, select_feature_columns


SCADA_POWER_LIMIT = 10_000


def _clean_power(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    power_cols = [c for c in df.columns if "_power_" in c]
    for col in power_cols:
        df.loc[(df[col] < 0) | (df[col] > SCADA_POWER_LIMIT), col] = np.nan
    return df


def _hourly_scada(data_dir: Path) -> pd.DataFrame:
    vestas = _clean_power(pd.read_csv(data_dir / "train" / "scada_vestas_train.csv", encoding="utf-8-sig"))
    unison = _clean_power(pd.read_csv(data_dir / "train" / "scada_unison_train.csv", encoding="utf-8-sig"))
    vestas["kst_dtm"] = pd.to_datetime(vestas["kst_dtm"])
    unison["kst_dtm"] = pd.to_datetime(unison["kst_dtm"])

    groups = {
        "kpx_group_1": (vestas, "vestas", range(1, 7)),
        "kpx_group_2": (vestas, "vestas", range(7, 13)),
        "kpx_group_3": (unison, "unison", range(1, 6)),
    }
    out = []
    for target, (df, prefix, turbines) in groups.items():
        power_cols = [f"{prefix}_wtg{i:02d}_power_kw10m" for i in turbines]
        ws_cols = [f"{prefix}_wtg{i:02d}_ws" for i in turbines]
        wd_cols = [f"{prefix}_wtg{i:02d}_wd" for i in turbines]

        tmp = pd.DataFrame()
        # A +60 minute shift aligns the 10-minute SCADA interval sums with the hourly KPX label timestamp.
        tmp[TIME_COL] = df["kst_dtm"] + pd.Timedelta(minutes=60)
        tmp[f"{target}__scada_power_sum"] = df[power_cols].sum(axis=1, min_count=1)
        tmp[TIME_COL] = tmp[TIME_COL].dt.floor("h")
        agg = tmp.groupby(TIME_COL).agg(
            {
                f"{target}__scada_power_sum": "sum",
            }
        )
        out.append(agg)

    return pd.concat(out, axis=1).sort_index().astype("float32")


def _proxy_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l1",
        n_estimators=250,
        learning_rate=0.06,
        num_leaves=32,
        min_child_samples=35,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )


def _make_proxy_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scada: pd.DataFrame,
    valid_time: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    base_cols = select_feature_columns(X_train, "kpx_group_1", "base")
    X_base = X_train[base_cols]
    X_test_base = X_test.reindex(columns=base_cols)
    proxy_train = pd.DataFrame(index=X_train.index)
    proxy_test = pd.DataFrame(index=X_test.index)
    report: dict[str, object] = {}

    for j, col in enumerate(scada.columns, start=1):
        y = scada[col].reindex(X_train.index)
        pre_valid = (~valid_time) & y.notna()
        valid = valid_time & y.notna()
        if pre_valid.sum() < 1000:
            continue

        model = _proxy_model(8100 + j)
        eval_set = [(X_base.loc[valid], y.loc[valid])] if valid.sum() else None
        callbacks = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)] if valid.sum() else [lgb.log_evaluation(0)]
        model.fit(X_base.loc[pre_valid], y.loc[pre_valid], eval_set=eval_set, eval_metric="l1", callbacks=callbacks)

        # Use the pre-valid model for 2024 holdout to avoid using 2024 SCADA in validation features.
        train_pred = pd.Series(model.predict(X_base), index=X_train.index)
        score = None
        if valid.sum():
            score = float(np.corrcoef(train_pred.loc[valid], y.loc[valid])[0, 1])

        final_mask = y.notna()
        final_model = _proxy_model(9100 + j)
        final_model.fit(X_base.loc[final_mask], y.loc[final_mask], callbacks=[lgb.log_evaluation(0)])

        proxy_name = f"proxy__{col}"
        proxy_train[proxy_name] = train_pred.astype("float32")
        proxy_test[proxy_name] = final_model.predict(X_test_base).astype("float32")
        report[proxy_name] = {
            "pre_valid_rows": int(pre_valid.sum()),
            "valid_rows": int(valid.sum()),
            "valid_corr": score,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
        }
    return proxy_train, proxy_test, report


def _fit_candidate(
    family: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
) -> tuple[object, int]:
    if family == "lgbm":
        model = make_model(seed)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        return model, int(model.best_iteration_)
    if family == "catboost":
        model = make_catboost_model(seed)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        return model, int(model.get_best_iteration() + 1)
    raise ValueError(f"Unknown family: {family}")


def _fit_final(family: str, X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_estimators: int) -> object:
    if family == "lgbm":
        model = make_model(seed, n_estimators=max(100, n_estimators))
        model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])
        return model
    if family == "catboost":
        model = make_catboost_model(seed, iterations=max(100, n_estimators))
        model.fit(X_train, y_train)
        return model
    raise ValueError(f"Unknown family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts_scada_proxy")
    parser.add_argument("--output", default="submissions/scada_proxy_v1.csv")
    parser.add_argument("--valid-start", default="2024-01-01 00:00:00")
    parser.add_argument("--include-catboost", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Building NWP features...", flush=True)
    X = build_features(data_dir, "train")
    X_test = build_features(data_dir, "test")
    valid_time = X.index >= pd.Timestamp(args.valid_start)

    print("Building SCADA proxy features...", flush=True)
    scada = _hourly_scada(data_dir)
    proxy_train, proxy_test, proxy_report = _make_proxy_features(X, X_test, scada, valid_time)

    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm").reindex(X.index)
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    submission = sample.set_index(TIME_COL)

    base_cols = select_feature_columns(X, "kpx_group_1", "base")
    X_aug = X[base_cols].join(proxy_train)
    X_test_aug = X_test.reindex(columns=base_cols).join(proxy_test)

    candidate_families = ["lgbm"]
    if args.include_catboost:
        candidate_families.append("catboost")
    candidate_variants = ["all", "eligible_only"]
    report: dict[str, object] = {
        "valid_start": args.valid_start,
        "proxy_report": proxy_report,
        "n_features": X_aug.shape[1],
        "targets": {},
    }
    valid_pred: dict[str, np.ndarray] = {}
    valid_truth: dict[str, np.ndarray] = {}

    for i, (target, capacity) in enumerate(CAPACITY_KWH.items(), start=1):
        y = labels[target]
        valid = valid_time & y.notna()
        valid_truth[target] = y.loc[valid].to_numpy()
        best = {"score": -np.inf}

        for family in candidate_families:
            for variant in candidate_variants:
                train_mask = (~valid_time) & y.notna()
                if variant == "eligible_only":
                    train_mask &= y >= 0.10 * capacity

                model, best_iteration = _fit_candidate(
                    family,
                    X_aug.loc[train_mask],
                    y.loc[train_mask],
                    X_aug.loc[valid],
                    y.loc[valid],
                    seed=10_000 + i * 100 + len(family) + len(variant),
                )
                raw = np.clip(model.predict(X_aug.loc[valid]), 0, capacity)
                scale, offset, metric = calibrate(y.loc[valid].to_numpy(), raw, capacity)
                print(target, family, variant, metric, flush=True)
                if metric["score"] > best["score"]:
                    best = {
                        "score": metric["score"],
                        "family": family,
                        "variant": variant,
                        "best_iteration": best_iteration,
                        "scale": scale,
                        "offset": offset,
                        "metric": metric,
                    }

        full_mask = y.notna()
        if best["variant"] == "eligible_only":
            full_mask &= y >= 0.10 * capacity
        final_model = _fit_final(
            best["family"],
            X_aug.loc[full_mask],
            y.loc[full_mask],
            seed=11_000 + i,
            n_estimators=int(best["best_iteration"]),
        )
        pred_valid_raw = np.clip(
            final_model.predict(X_aug.loc[valid]),
            0,
            capacity,
        )
        # Reported validation uses candidate holdout predictions above; final model is only for test inference.
        pred_test = np.clip(final_model.predict(X_test_aug) * best["scale"] + best["offset"], 0, capacity)
        submission[target] = pred_test
        report["targets"][target] = best

        # Refit of final model touches 2024; use the metric from validation candidate for summary only.
        valid_pred[target] = np.clip(pred_valid_raw * best["scale"] + best["offset"], 0, capacity)

    output = submission.reset_index()[sample.columns]
    output[TIME_COL] = output[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    # This summary is optimistic because valid_pred uses final models; target metrics above are the fair holdout checks.
    report["final_refit_valid_metric_note"] = evaluate_competition(valid_truth, valid_pred)
    (artifact_dir / "scada_proxy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["targets"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
