from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

import pipeline as p


def _rank_features(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[np.arange(len(scores))[:, None], order] = np.arange(scores.shape[1], dtype=np.float32)
    centered = scores - scores.mean(axis=1, keepdims=True)
    top = scores[np.arange(len(scores)), order[:, -1]][:, None]
    second = scores[np.arange(len(scores)), order[:, -2]][:, None]
    margin_to_top = top - scores
    top_margin = np.repeat(top - second, scores.shape[1], axis=1)
    return np.stack(
        [
            scores,
            centered,
            ranks / max(scores.shape[1] - 1, 1),
            margin_to_top,
            top_margin,
        ],
        axis=2,
    ).astype(np.float32)


def _build_features(
    x: np.ndarray,
    cands: np.ndarray,
    score_bank: dict[str, np.ndarray],
) -> np.ndarray:
    cf = p.make_candidate_features(x, x.shape[1] - 1, cands, horizon=2)
    ens = score_bank["ens_scores"].astype(np.float32)
    prior = score_bank["ens_prior"].astype(np.float32)
    residual = ens - prior
    score_feats = np.concatenate(
        [
            _rank_features(ens),
            _rank_features(prior),
            _rank_features(residual),
            residual[:, :, None],
        ],
        axis=2,
    )
    cand_ids = np.arange(cands.shape[1], dtype=np.float32)[None, :, None]
    cand_ids = np.repeat(cand_ids / max(cands.shape[1] - 1, 1), len(cands), axis=0)
    family = p.CANDIDATE_FAMILY.astype(np.float32)[None, :, None]
    family = np.repeat(family / max(len(p.FAMILY_NAMES) - 1, 1), len(cands), axis=0)
    return np.concatenate([cf, score_feats, cand_ids, family], axis=2).astype(np.float32)


def _target_scores(cands: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    err = np.linalg.norm(cands - y[:, None, :], axis=2)
    hit = (err <= p.R_HIT).astype(np.float32)
    if mode == "neg_err":
        return (-err).astype(np.float32)
    if mode == "hit":
        return hit
    if mode == "utility":
        score = -err / 0.0045 + hit * 0.90
        return score.astype(np.float32)
    raise ValueError(mode)


def _flat(a: np.ndarray) -> np.ndarray:
    return a.reshape(a.shape[0] * a.shape[1], a.shape[2])


def _fit_regressor(x: np.ndarray, y: np.ndarray, seed: int, args: argparse.Namespace) -> lgb.LGBMRegressor:
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


def _search_modes(cands: np.ndarray, scores: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return {
        "soft": p.search_temperature(cands, scores, y),
        "gate": p.search_argmax_soft_gate(cands, scores, y),
        "argmax": p.metrics(cands[np.arange(len(cands)), np.argmax(scores, axis=1)], y),
    }


def _write_submission_variants(
    out_dir: Path,
    test_ids: list[str],
    test_cands: np.ndarray,
    test_scores: np.ndarray,
    search: dict[str, object],
    prefix: str,
) -> list[str]:
    files: list[str] = []
    soft_temp = float(search["soft"]["temperature"])  # type: ignore[index]
    soft_pred = p.soft_select(test_cands, test_scores, soft_temp)
    soft_file = out_dir / f"submission_{prefix}_soft.csv"
    p.write_submission(soft_file, test_ids, soft_pred)
    files.append(str(soft_file))

    gate = search["gate"]  # type: ignore[index]
    gate_pred = p.argmax_soft_gate_select(
        test_cands,
        test_scores,
        float(gate["temperature"]),  # type: ignore[index]
        float(gate["margin_threshold"]),  # type: ignore[index]
    )
    gate_file = out_dir / f"submission_{prefix}_gate.csv"
    p.write_submission(gate_file, test_ids, gate_pred)
    files.append(str(gate_file))

    arg_pred = test_cands[np.arange(len(test_cands)), np.argmax(test_scores, axis=1)]
    arg_file = out_dir / f"submission_{prefix}_argmax.csv"
    p.write_submission(arg_file, test_ids, arg_pred)
    files.append(str(arg_file))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM level-2 candidate ranker over selector OOF scores.")
    parser.add_argument("--selector-out", type=Path, default=p.WORK_DIR / "selector_full")
    parser.add_argument("--out-dir", type=Path, default=p.WORK_DIR / "04_lgbm_ranker")
    parser.add_argument("--target-mode", choices=["neg_err", "hit", "utility"], default="utility")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--make-test", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve() / f"target_{args.target_mode}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    selector_out = args.selector_out.resolve()

    train_ids, y = p.read_labels(p.DATA_ROOT / "train_labels.csv")
    train_x = p.load_stack(p.DATA_ROOT / "train", train_ids)
    with np.load(selector_out / "oof_selector_scores.npz", allow_pickle=True) as z:
        oof_bank = {k: z[k] for k in z.files}
    cands = oof_bank["cands"].astype(np.float32)
    features = _build_features(train_x, cands, oof_bank)
    targets = _target_scores(cands, y, args.target_mode)

    fold_ids = np.asarray([p.stable_fold_id(sample_id, args.folds) for sample_id in train_ids])
    oof_scores = np.zeros((len(y), cands.shape[1]), dtype=np.float32)
    fold_reports: list[dict[str, object]] = []
    for fold in range(args.folds):
        va = fold_ids == fold
        tr = ~va
        model = _fit_regressor(
            _flat(features[tr]),
            targets[tr].reshape(-1),
            args.seed + fold * 997,
            args,
        )
        pred = model.predict(_flat(features[va])).reshape(np.sum(va), cands.shape[1])
        oof_scores[va] = pred.astype(np.float32)
        fold_search = _search_modes(cands[va], oof_scores[va], y[va])
        fold_reports.append(
            {
                "fold": fold,
                "soft": fold_search["soft"]["metrics"],
                "gate": fold_search["gate"]["metrics"],
                "argmax": fold_search["argmax"],
            }
        )
        print(
            "[FOLD]",
            fold,
            f"gate={fold_search['gate']['metrics']['hit']:.6f}",  # type: ignore[index]
            f"soft={fold_search['soft']['metrics']['hit']:.6f}",  # type: ignore[index]
            flush=True,
        )

    search = _search_modes(cands, oof_scores, y)
    summary: dict[str, object] = {
        "target_mode": args.target_mode,
        "seed": args.seed,
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
        },
        "fold_reports": fold_reports,
        "oof": search,
    }
    np.savez_compressed(
        out_dir / "oof_ranker_scores.npz",
        ids=np.asarray(train_ids, dtype=object),
        y=y.astype(np.float32),
        cands=cands.astype(np.float32),
        scores=oof_scores.astype(np.float32),
        candidate_names=oof_bank["candidate_names"],
    )

    if args.make_test:
        test_ids = p.read_submission_ids(p.DATA_ROOT / "sample_submission.csv")
        test_x = p.load_stack(p.DATA_ROOT / "test", test_ids)
        with np.load(selector_out / "test_selector_scores.npz", allow_pickle=True) as z:
            test_bank = {k: z[k] for k in z.files}
        test_cands = test_bank["cands"].astype(np.float32)
        test_features = _build_features(test_x, test_cands, test_bank)
        full_model = _fit_regressor(_flat(features), targets.reshape(-1), args.seed + 9999, args)
        test_scores = full_model.predict(_flat(test_features)).reshape(len(test_ids), test_cands.shape[1]).astype(np.float32)
        np.savez_compressed(
            out_dir / "test_ranker_scores.npz",
            ids=np.asarray(test_ids, dtype=object),
            cands=test_cands,
            scores=test_scores,
            candidate_names=test_bank["candidate_names"],
        )
        summary["test_files"] = _write_submission_variants(out_dir, test_ids, test_cands, test_scores, search, f"lgbm_{args.target_mode}")

    (out_dir / "ranker_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["oof"], indent=2), flush=True)
    print(f"[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
