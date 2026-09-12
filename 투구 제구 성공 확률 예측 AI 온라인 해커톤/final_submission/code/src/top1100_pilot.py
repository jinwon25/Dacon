"""Small, reproducible Top-1100 structural-model screen.

This is deliberately a screen rather than a promotion pipeline.  It evaluates
one locked 2024 outer year (train 2019--2023) and never uses 2024 for fitting,
early stopping, or blending.  The feature builder itself only uses official
as-of values and prior-season snapshots.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import read_main
from src.metrics import brier_score, brier_skill_score
from src.top1100_features import build_features


def _metrics(name: str, y: np.ndarray, p: np.ndarray, baseline: float) -> dict:
    bs = brier_score(y, p)
    return {
        "model": name,
        "outer_validation_season": 2024,
        "n_rows": int(len(y)),
        "brier": bs,
        "delta_vs_v2_exact": bs - baseline,
        "score_equivalent": brier_skill_score(y, p),
        "finite": bool(np.isfinite(p).all()),
    }


def _clean_cat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=["category", "object", "string"]).columns:
        out[col] = out[col].astype("string").fillna("__MISSING__")
    return out


def _run_logit(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame) -> np.ndarray:
    cats = list(x_train.select_dtypes(include=["category", "object", "string"]).columns)
    nums = [c for c in x_train.columns if c not in cats]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
        ],
        sparse_threshold=0.99,
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("clf", SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=25, tol=1e-4, average=True, random_state=42, n_jobs=6)),
        ]
    )
    model.fit(_clean_cat(x_train), y_train)
    return np.asarray(model.predict_proba(_clean_cat(x_valid))[:, 1], dtype=float)


def _run_catboost(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame) -> np.ndarray:
    from catboost import CatBoostClassifier

    cats = list(x_train.select_dtypes(include=["category", "object", "string"]).columns)
    xt = _clean_cat(x_train)
    xv = _clean_cat(x_valid)
    # A fixed, pre-registered screen budget.  The outer target is never an
    # eval_set, so this cannot choose iterations from 2024.
    model = CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        l2_leaf_reg=30,
        random_strength=0.5,
        random_seed=42,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
        one_hot_max_size=32,
    )
    model.fit(xt, y_train, cat_features=cats)
    return np.asarray(model.predict_proba(xv)[:, 1], dtype=float)


def _run_lgb_state(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    xt, xv = x_train.copy(), x_valid.copy()
    cats = list(xt.select_dtypes(include=["category", "object", "string"]).columns)
    for col in cats:
        values = pd.Index(xt[col].astype("string").fillna("__MISSING__").unique())
        xt[col] = pd.Categorical(xt[col].astype("string").fillna("__MISSING__"), categories=values)
        xv[col] = pd.Categorical(xv[col].astype("string").fillna("__MISSING__"), categories=values)
    train_set = lgb.Dataset(xt, label=y_train, categorical_feature=cats, free_raw_data=True)
    model = lgb.train({
        "objective": "binary", "metric": "None", "verbosity": -1, "num_threads": 6,
        "deterministic": True, "force_col_wise": True, "seed": 42,
        "num_leaves": 31, "learning_rate": 0.04, "min_data_in_leaf": 1200,
        "lambda_l2": 20.0, "feature_fraction": 0.85, "bagging_fraction": 0.9, "bagging_freq": 1,
    }, train_set, num_boost_round=350)
    return np.asarray(model.predict(xv), dtype=float)


def _encode_fm(x_train: pd.DataFrame, x_valid: pd.DataFrame):
    """Encode categories with train-only vocabularies and standardize numerics."""
    cat_cols = list(x_train.select_dtypes(include=["category", "object", "string"]).columns)
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    train_cat, valid_cat, cards = [], [], []
    for col in cat_cols:
        vocab = pd.Index(x_train[col].astype("string").fillna("__MISSING__").unique())
        mapping = {str(v): i + 1 for i, v in enumerate(vocab)}
        train_cat.append(x_train[col].astype("string").fillna("__MISSING__").map(mapping).fillna(0).to_numpy(np.int64))
        valid_cat.append(x_valid[col].astype("string").fillna("__MISSING__").map(mapping).fillna(0).to_numpy(np.int64))
        cards.append(len(mapping) + 1)
    def numeric(frame):
        value = frame[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        return value
    a, b = numeric(x_train), numeric(x_valid)
    med = np.nanmedian(a, axis=0); med[~np.isfinite(med)] = 0
    a = np.where(np.isfinite(a), a, med); b = np.where(np.isfinite(b), b, med)
    mean = a.mean(axis=0); scale = a.std(axis=0); scale[scale < 1e-4] = 1
    return np.stack(train_cat, axis=1), np.stack(valid_cat, axis=1), (a.astype(np.float32) - mean.astype(np.float32)) / scale, (b.astype(np.float32) - mean.astype(np.float32)) / scale, (scale.astype(np.float32), cards)


def _run_fm(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(42)
    torch.set_num_threads(6)
    train_cat, valid_cat, train_num, valid_num, (_, cards) = _encode_fm(x_train, x_valid)
    device = torch.device("cpu")
    class FM(nn.Module):
        def __init__(self, cards, n_num, dim=8):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in cards])
            self.linear_num = nn.Linear(n_num, 1)
            self.linear_cat = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
            self.mlp = nn.Sequential(nn.Linear(n_num + dim, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1))
        def forward(self, cat, num):
            fields = [e(cat[:, i]) for i, e in enumerate(self.emb)]
            stacked = torch.stack(fields, dim=1)
            summed = stacked.sum(dim=1)
            fm = 0.5 * (summed.square() - stacked.square().sum(dim=1)).sum(dim=1, keepdim=True)
            linear = self.linear_num(num) + sum(e(cat[:, i]) for i, e in enumerate(self.linear_cat))
            return linear + fm + self.mlp(torch.cat([num, summed], dim=1))
    model = FM(cards, train_num.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(42)
    batch = 8192
    model.train()
    for epoch in range(5):
        order = rng.permutation(len(y_train))
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            cat = torch.from_numpy(train_cat[idx]).to(device)
            num = torch.from_numpy(train_num[idx]).to(device)
            target = torch.from_numpy(y_train[idx].astype(np.float32)).to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(cat, num), target)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(idx)
        print(f"[fm] epoch={epoch+1} loss={total/len(y_train):.6f}", flush=True)
    model.eval(); values = []
    with torch.no_grad():
        for start in range(0, len(x_valid), batch):
            cat = torch.from_numpy(valid_cat[start:start + batch]).to(device)
            num = torch.from_numpy(valid_num[start:start + batch]).to(device)
            values.append(torch.sigmoid(model(cat, num)).cpu().numpy().ravel())
    return np.concatenate(values)


def run(project: Path, train_cap: int = 450_000, models: tuple[str, ...] = ("logit", "catboost")) -> pd.DataFrame:
    started = time.perf_counter()
    train = read_main(project / "data" / "train.csv")
    train_hist = train.loc[train["season"] < 2024]
    valid = train.loc[train["season"] == 2024].copy()
    # Build once from the complete history so each row gets its strict prior
    # season state.  We fit only on pre-2024 rows below.
    all_features = build_features(train, train, include_ids=True)
    x_train = all_features.loc[train["season"].to_numpy() < 2024].reset_index(drop=True)
    x_valid = all_features.loc[train["season"].to_numpy() == 2024].reset_index(drop=True)
    y_train = train_hist["control_success"].to_numpy(dtype=np.int8)
    y_valid = valid["control_success"].to_numpy(dtype=np.int8)
    if len(x_train) > train_cap:
        # Deterministic stratified-ish thinning, preserving the latest seasons.
        rng = np.random.default_rng(42)
        idx = np.concatenate([np.flatnonzero(train_hist["season"].to_numpy() == s) for s in sorted(train_hist["season"].unique())])
        take = rng.choice(idx, size=train_cap, replace=False)
        take.sort()
        x_fit, y_fit = x_train.iloc[take], y_train[take]
    else:
        x_fit, y_fit = x_train, y_train

    import script as champion_script

    v2 = np.asarray(champion_script.predict_dataframe(valid.drop(columns=["control_success"])), dtype=float)
    baseline_bs = brier_score(y_valid, v2)
    rows = [_metrics("v2_package_exact_legacy_2024", y_valid, v2, baseline_bs)]

    runners = []
    if "logit" in models:
        runners.append(("dynamic_logit_screen", _run_logit))
    if "catboost" in models:
        runners.append(("categorical_catboost_screen", _run_catboost))
    if "lgb" in models:
        runners.append(("p0a_lgb_state_screen", _run_lgb_state))
    if "fm" in models:
        runners.append(("field_aware_fm_screen", _run_fm))
    for name, runner in runners:
        t0 = time.perf_counter()
        pred = runner(x_fit, y_fit, x_valid)
        row = _metrics(name, y_valid, pred, baseline_bs)
        row["fit_rows"] = int(len(x_fit)); row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        np.save(project / "artifacts" / "top1100" / f"{name}_2024.npy", pred)
        print(row, flush=True)

    result = pd.DataFrame(rows)
    result["run_seconds"] = time.perf_counter() - started
    out = project / "reports" / "top1100" / "fold_metrics.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    merged = pd.concat([old, result], ignore_index=True) if not old.empty else result
    merged.to_csv(out, index=False)
    (project / "artifacts" / "top1100" / "pilot_provenance.json").write_text(
        json.dumps({"outer": 2024, "train_period": "2019-2023", "valid_period": "2024", "train_cap": train_cap, "outer_target_used_for_selection": False}, indent=2),
        encoding="utf-8",
    )
    print(result.to_string(index=False), flush=True)
    del all_features, train, x_train, x_valid
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--train-cap", type=int, default=450_000)
    parser.add_argument("--models", default="logit,catboost", help="comma-separated: logit,catboost")
    args = parser.parse_args()
    run(args.project_dir.resolve(), args.train_cap, tuple(x.strip() for x in args.models.split(",") if x.strip()))


if __name__ == "__main__":
    main()
