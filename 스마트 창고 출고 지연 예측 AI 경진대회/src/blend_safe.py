"""LB-안전 블렌드 비교: global SLSQP / equal mean / rank mean / geometric mean / power mean.

per-bin 같이 OOF에 강하게 fitting되는 방식 대신, 단순/견고한 가중치만 사용.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from scipy.stats import rankdata


TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
EPS = 1e-9
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def slsqp(y, P):
    def f(w):
        return mean_absolute_error(y, np.clip(P @ w, 0, None))
    n = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnd = [(0, 1)] * n
    starts = [np.ones(n) / n] + [np.eye(n)[i] for i in range(n)]
    best = None
    for s in starts:
        r = minimize(f, s, method="SLSQP", bounds=bnd, constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def equal(P):
    return np.ones(P.shape[1]) / P.shape[1]


def rank_avg(P_oof, P_test):
    """각 컬럼을 rank로 변환 후 평균, OOF mean/std로 다시 회귀 척도로 매핑."""
    # train rank-based
    oof_rank = np.column_stack([rankdata(P_oof[:, i]) for i in range(P_oof.shape[1])])
    avg_oof_rank = oof_rank.mean(axis=1)
    test_rank = np.column_stack([rankdata(P_test[:, i]) for i in range(P_test.shape[1])])
    avg_test_rank = test_rank.mean(axis=1)
    # rank → 회귀 값으로: 평균 OOF prediction의 sorted값을 lookup
    sorted_pred = np.sort(P_oof.mean(axis=1))
    n = len(sorted_pred)
    # rank 1..N을 0..N-1 인덱스로
    oof_pred = sorted_pred[np.clip((avg_oof_rank - 1).astype(int), 0, n - 1)]
    # test: rank 1..M → train sorted_pred에서 비율 위치
    m = len(avg_test_rank)
    pos = np.clip(((avg_test_rank - 1) / (m - 1) * (n - 1)).astype(int), 0, n - 1)
    test_pred = sorted_pred[pos]
    return oof_pred, test_pred


def geometric(P):
    """기하 평균 (log1p로 안정)."""
    return np.expm1(np.log1p(np.clip(P, 0, None)).mean(axis=1))


def power_mean(P, p=0.5):
    """power mean p ∈ (0,1] — geometric(0)과 arithmetic(1) 사이."""
    P = np.clip(P, EPS, None)
    return (P ** p).mean(axis=1) ** (1 / p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models/blend_safe"))
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [project_path(p) for p in args.model_dirs]

    print(f"loading {len(dirs)} models")
    oofs = [pd.read_csv(p / "oof_predictions.csv") for p in dirs]
    subs = [pd.read_csv(p / "submission.csv") for p in dirs]
    base_ids = oofs[0][ID_COL].to_numpy()
    for i, oof in enumerate(oofs[1:], 1):
        if not np.array_equal(base_ids, oof[ID_COL].to_numpy()):
            oofs[i] = oof.set_index(ID_COL).loc[base_ids].reset_index()
    sub_ids = subs[0][ID_COL].to_numpy()
    for i, sub in enumerate(subs[1:], 1):
        if not np.array_equal(sub_ids, sub[ID_COL].to_numpy()):
            subs[i] = sub.set_index(ID_COL).loc[sub_ids].reset_index()

    y = oofs[0][TARGET].to_numpy()
    P_oof = np.column_stack([np.clip(o["pred"].to_numpy(), 0, None) for o in oofs])
    P_test = np.column_stack([np.clip(s[TARGET].to_numpy(), 0, None) for s in subs])

    results = {}

    # 1) Global SLSQP
    w, mae = slsqp(y, P_oof)
    results["global_slsqp"] = {"oof_mae": float(mae), "weights": w.tolist(),
                                "oof": P_oof @ w, "test": P_test @ w}
    print(f"global SLSQP: {mae:.6f}")
    for d, ww in zip(dirs, w):
        print(f"  {d.name}: {ww:.4f}")

    # 2) Equal mean
    w = equal(P_oof)
    pred_oof = np.clip(P_oof @ w, 0, None)
    pred_test = np.clip(P_test @ w, 0, None)
    mae = mean_absolute_error(y, pred_oof)
    results["equal"] = {"oof_mae": float(mae), "oof": pred_oof, "test": pred_test}
    print(f"equal: {mae:.6f}")

    # 3) Geometric mean
    pred_oof = np.clip(geometric(P_oof), 0, None)
    pred_test = np.clip(geometric(P_test), 0, None)
    mae = mean_absolute_error(y, pred_oof)
    results["geometric"] = {"oof_mae": float(mae), "oof": pred_oof, "test": pred_test}
    print(f"geometric: {mae:.6f}")

    # 4) Power mean p=0.5
    pred_oof = np.clip(power_mean(P_oof, 0.5), 0, None)
    pred_test = np.clip(power_mean(P_test, 0.5), 0, None)
    mae = mean_absolute_error(y, pred_oof)
    results["power_0.5"] = {"oof_mae": float(mae), "oof": pred_oof, "test": pred_test}
    print(f"power_0.5: {mae:.6f}")

    # 5) Rank average
    pred_oof, pred_test = rank_avg(P_oof, P_test)
    pred_oof = np.clip(pred_oof, 0, None)
    pred_test = np.clip(pred_test, 0, None)
    mae = mean_absolute_error(y, pred_oof)
    results["rank_avg"] = {"oof_mae": float(mae), "oof": pred_oof, "test": pred_test}
    print(f"rank_avg: {mae:.6f}")

    # 베스트 & 폴백 (global SLSQP)
    best_name = min(results.keys(), key=lambda k: results[k]["oof_mae"])
    best = results[best_name]
    print(f"\n=== BEST: {best_name} OOF MAE {best['oof_mae']:.6f} ===")

    # save best as submission
    sub_out = subs[0].copy()
    sub_out[TARGET] = best["test"]
    sub_out.to_csv(out_dir / "submission.csv", index=False)

    # save all variants for inspection
    for name, r in results.items():
        s = subs[0].copy()
        s[TARGET] = r["test"]
        s.to_csv(out_dir / f"submission_{name}.csv", index=False)

    # OOF
    o = oofs[0][[ID_COL, TARGET]].copy()
    o["pred"] = best["oof"]
    o.to_csv(out_dir / "oof_predictions.csv", index=False)

    metadata = {
        "model_dirs": [str(p) for p in dirs],
        "best_method": best_name,
        "all_results": {k: {"oof_mae": v["oof_mae"], "weights": v.get("weights")} for k, v in results.items()},
    }
    with open(out_dir / "blend_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
