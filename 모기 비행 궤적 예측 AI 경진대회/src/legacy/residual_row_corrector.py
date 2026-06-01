from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

import pipeline as p


def _base_prediction(cands: np.ndarray, scores: np.ndarray, prior: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, float]]:
    if mode == "soft":
        temp = 0.05
        return p.soft_select(cands, scores, temp), {"temperature": temp}
    if mode == "projected":
        residual_weight = 0.9
        temp = 0.03
        projected = p.physics_project_scores(scores, prior, residual_weight)
        return p.soft_select(cands, projected, temp), {"temperature": temp, "residual_weight": residual_weight}
    if mode == "argmax":
        return cands[np.arange(len(cands)), np.argmax(scores, axis=1)], {}
    raise ValueError(mode)


def _row_features(x: np.ndarray, cands: np.ndarray, scores: np.ndarray, prior: np.ndarray, pred: np.ndarray) -> np.ndarray:
    p0 = x[:, -1]
    rel = x[:, -6:] - p0[:, None, :]
    seq_flat = rel.reshape(len(x), -1)
    ctx = p.turn_model_features_from_context(p.turn_context_features(x, x.shape[1] - 1))
    order = np.argsort(scores, axis=1)
    top_idx = order[:, -5:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    top_prior = np.take_along_axis(prior, top_idx, axis=1)
    top_cands = cands[np.arange(len(cands))[:, None], top_idx].reshape(len(cands), -1)
    soft05 = p.soft_select(cands, scores, 0.05)
    soft10 = p.soft_select(cands, scores, 0.10)
    arg = cands[np.arange(len(cands)), order[:, -1]]
    margin = (scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]])[:, None]
    spread = np.sqrt(np.sum((cands - soft05[:, None, :]) ** 2, axis=2))
    spread_stats = np.stack(
        [
            spread.mean(axis=1),
            np.quantile(spread, 0.25, axis=1),
            np.quantile(spread, 0.50, axis=1),
            np.quantile(spread, 0.75, axis=1),
            spread.min(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate(
        [
            seq_flat,
            ctx,
            pred - p0,
            soft05 - p0,
            soft10 - p0,
            arg - p0,
            pred - soft05,
            pred - arg,
            top_cands - np.repeat(p0, 5, axis=0).reshape(len(x), -1),
            top_scores,
            top_prior,
            top_scores - top_prior,
            margin,
            spread_stats,
        ],
        axis=1,
    ).astype(np.float32)


def _fit_one(x: np.ndarray, y: np.ndarray, seed: int, args: argparse.Namespace) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def _cap_vectors(vec: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec * np.minimum(1.0, cap / (norm + p.EPS))


def _search_correction(base_pred: np.ndarray, delta: np.ndarray, y: np.ndarray) -> dict[str, object]:
    best = None
    best_key = (-1, -float("inf"))
    for cap in [0.0, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]:
        capped = _cap_vectors(delta, cap)
        for scale in [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]:
            pred = base_pred + scale * capped
            m = p.metrics(pred, y)
            key = (int(m["hits"]), -float(m["mean"]))
            if best is None or key > best_key:
                best = {
                    "cap": cap,
                    "scale": scale,
                    "metrics": m,
                }
                best_key = key
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Row-level residual corrector for selector predictions.")
    parser.add_argument("--selector-out", type=Path, default=p.WORK_DIR / "selector_full")
    parser.add_argument("--out-dir", type=Path, default=p.WORK_DIR / "05_row_residual")
    parser.add_argument("--base-mode", choices=["soft", "projected", "argmax"], default="projected")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=120)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--make-test", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve() / f"base_{args.base_mode}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector_out = args.selector_out.resolve()

    train_ids, y = p.read_labels(p.DATA_ROOT / "train_labels.csv")
    train_x = p.load_stack(p.DATA_ROOT / "train", train_ids)
    with np.load(selector_out / "oof_selector_scores.npz", allow_pickle=True) as z:
        bank = {k: z[k] for k in z.files}
    cands = bank["cands"].astype(np.float32)
    scores = bank["ens_scores"].astype(np.float32)
    prior = bank["ens_prior"].astype(np.float32)
    base_pred, base_params = _base_prediction(cands, scores, prior, args.base_mode)
    feats = _row_features(train_x, cands, scores, prior, base_pred)
    residual = (y - base_pred).astype(np.float32)
    fold_ids = np.asarray([p.stable_fold_id(sample_id, args.folds) for sample_id in train_ids])
    oof_delta = np.zeros_like(residual, dtype=np.float32)
    fold_reports = []
    for fold in range(args.folds):
        va = fold_ids == fold
        tr = ~va
        fold_delta = []
        for dim in range(3):
            model = _fit_one(feats[tr], residual[tr, dim], args.seed + fold * 100 + dim, args)
            fold_delta.append(model.predict(feats[va]))
        oof_delta[va] = np.stack(fold_delta, axis=1).astype(np.float32)
        fold_best = _search_correction(base_pred[va], oof_delta[va], y[va])
        fold_reports.append({"fold": fold, "best": fold_best})
        print(
            "[FOLD]",
            fold,
            f"hit={fold_best['metrics']['hit']:.6f}",  # type: ignore[index]
            f"cap={fold_best['cap']}",
            f"scale={fold_best['scale']}",
            flush=True,
        )

    base_metrics = p.metrics(base_pred, y)
    best = _search_correction(base_pred, oof_delta, y)
    summary: dict[str, object] = {
        "base_mode": args.base_mode,
        "base_params": base_params,
        "base_metrics": base_metrics,
        "best": best,
        "fold_reports": fold_reports,
    }
    np.savez_compressed(
        out_dir / "oof_row_residual.npz",
        ids=np.asarray(train_ids, dtype=object),
        y=y.astype(np.float32),
        base_pred=base_pred.astype(np.float32),
        delta=oof_delta.astype(np.float32),
    )

    if args.make_test:
        test_ids = p.read_submission_ids(p.DATA_ROOT / "sample_submission.csv")
        test_x = p.load_stack(p.DATA_ROOT / "test", test_ids)
        with np.load(selector_out / "test_selector_scores.npz", allow_pickle=True) as z:
            test_bank = {k: z[k] for k in z.files}
        test_cands = test_bank["cands"].astype(np.float32)
        test_scores = test_bank["ens_scores"].astype(np.float32)
        test_prior = test_bank["ens_prior"].astype(np.float32)
        test_base, _ = _base_prediction(test_cands, test_scores, test_prior, args.base_mode)
        test_feats = _row_features(test_x, test_cands, test_scores, test_prior, test_base)
        test_delta = []
        for dim in range(3):
            model = _fit_one(feats, residual[:, dim], args.seed + 999 + dim, args)
            test_delta.append(model.predict(test_feats))
        test_delta_arr = np.stack(test_delta, axis=1).astype(np.float32)
        cap = float(best["cap"])
        scale = float(best["scale"])
        test_pred = test_base + scale * _cap_vectors(test_delta_arr, cap)
        sub_path = out_dir / f"submission_row_residual_{args.base_mode}_cap{str(cap).replace('.', 'p')}_scale{str(scale).replace('.', 'p')}.csv"
        p.write_submission(sub_path, test_ids, test_pred)
        summary["test_file"] = str(sub_path)

    (out_dir / "row_residual_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
