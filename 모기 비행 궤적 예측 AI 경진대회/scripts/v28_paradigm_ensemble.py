"""v28_paradigm_ensemble.py — Paradigm ensemble between selector(0.6834) and v26 Kalman(0.6702).

5월 9일 best `submission_best_public_0p6834_boundary_gate.csv` (LB 0.6834, selector + boundary
correction)와 이번 세션의 v26 (LB 0.6702, Kalman residual + GRU + boundary MLP)는 완전히 다른
paradigm. → 단순 weighted avg로 LB 0.685~0.69 도달 시도.

진단 출력:
  - 두 prediction 거리 분포 (얼마나 다른가)
  - axis별 차이
  - 다양한 weight (0.5, 0.6, 0.7, 0.8)로 ensemble CSV 생성

각 ensemble은 별도 CSV로 저장. 사용자가 1~2개 선택 제출.

사용법:
  python scripts/v28_paradigm_ensemble.py
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "open"
BEST_PATH = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
V26_PATH  = DATA_DIR / "submission_v26_cpu_fast.csv"
V27_PATH  = DATA_DIR / "submission_v27_cpu_fast.csv"


def main():
    print("=" * 60)
    print("v28 paradigm ensemble: selector(LB 0.6834) × Kalman v26(LB 0.6702)")
    print("=" * 60)

    assert BEST_PATH.exists(), f"missing: {BEST_PATH}"
    assert V26_PATH.exists(), f"missing: {V26_PATH}"

    df_best = pd.read_csv(BEST_PATH)
    df_v26  = pd.read_csv(V26_PATH)
    df_v27  = pd.read_csv(V27_PATH) if V27_PATH.exists() else None

    # id 정렬 일치 확인
    assert (df_best["id"].values == df_v26["id"].values).all()
    if df_v27 is not None:
        assert (df_best["id"].values == df_v27["id"].values).all()

    pred_best = df_best[["x","y","z"]].values
    pred_v26  = df_v26[["x","y","z"]].values
    pred_v27  = df_v27[["x","y","z"]].values if df_v27 is not None else None

    # --- Prediction 거리 분석 (OOF 없음, prediction 자체 비교) ---
    dist_best_v26 = np.linalg.norm(pred_best - pred_v26, axis=-1)
    print(f"\n=== 두 prediction 거리 (best LB 0.6834 vs v26 LB 0.6702) ===")
    print(f"  mean    : {dist_best_v26.mean()*100:.3f} cm")
    print(f"  median  : {np.median(dist_best_v26)*100:.3f} cm")
    print(f"  p90     : {np.percentile(dist_best_v26, 90)*100:.3f} cm")
    print(f"  p99     : {np.percentile(dist_best_v26, 99)*100:.3f} cm")
    print(f"  max     : {dist_best_v26.max()*100:.3f} cm")
    print(f"  < 1cm   : {(dist_best_v26 < 0.01).mean()*100:.2f}%")
    print(f"  < 2cm   : {(dist_best_v26 < 0.02).mean()*100:.2f}%")

    for j, ax in enumerate(["x", "y", "z"]):
        d = pred_best[:, j] - pred_v26[:, j]
        print(f"  axis {ax}: mean diff={d.mean()*100:+.3f}cm, std={d.std()*100:.3f}cm")

    # --- 다양한 weight ensemble 생성 ---
    print(f"\n=== Generating ensemble candidates ===")
    sub_template = df_best[["id"]].copy()

    # best (LB 0.6834) 가중치 변화
    candidates = []
    for w_best in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        w_v26 = 1.0 - w_best
        ens = w_best * pred_best + w_v26 * pred_v26
        name = f"v28_best{w_best:.2f}_v26{w_v26:.2f}"
        path = DATA_DIR / f"submission_{name}.csv"
        out_df = sub_template.copy()
        out_df["x"] = ens[:, 0]; out_df["y"] = ens[:, 1]; out_df["z"] = ens[:, 2]
        out_df.to_csv(path, index=False)
        candidates.append((name, w_best, w_v26, dist_best_v26.mean() * w_v26, path))
        print(f"  saved: {path.name}  (w_best={w_best:.2f}, w_v26={w_v26:.2f})")

    # boundary-aware: 두 예측이 2cm 안에 있으면 평균, 멀면 best
    THR = 0.02
    close = dist_best_v26 < THR
    bnd = np.where(close[:, None], (pred_best + pred_v26) / 2, pred_best)
    out_df = sub_template.copy()
    out_df["x"] = bnd[:, 0]; out_df["y"] = bnd[:, 1]; out_df["z"] = bnd[:, 2]
    bnd_path = DATA_DIR / "submission_v28_boundary_aware_2cm.csv"
    out_df.to_csv(bnd_path, index=False)
    print(f"  saved: {bnd_path.name}  (boundary-aware, close→avg / far→best)")

    # v27 포함 3-way ensemble (있을 때)
    if pred_v27 is not None:
        for w_best in [0.6, 0.7]:
            w_other = (1 - w_best) / 2
            ens = w_best * pred_best + w_other * pred_v26 + w_other * pred_v27
            name = f"v28_3way_best{w_best:.2f}"
            path = DATA_DIR / f"submission_{name}.csv"
            out_df = sub_template.copy()
            out_df["x"] = ens[:, 0]; out_df["y"] = ens[:, 1]; out_df["z"] = ens[:, 2]
            out_df.to_csv(path, index=False)
            print(f"  saved: {path.name}  (best={w_best}, v26={w_other:.2f}, v27={w_other:.2f})")

    # --- 추정 LB ---
    print(f"\n=== 추정 LB (단순 가중 평균 기준, 보수) ===")
    print(f"  best alone     : LB 0.6834 (실측)")
    print(f"  v26 alone      : LB 0.6702 (실측)")
    print(f"  v27 alone      : LB 0.6682 (실측)")
    print(f"  LB difference  : 0.013 (best - v26)")
    print(f"  추정 ensemble  : LB 0.683~0.692  (paradigm diversity 크면 +0.005~0.010 위)")

    # --- run_log ---
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v28_paradigm_ensemble",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "Selector+boundary(LB 0.6834) × Kalman+GRU+v26(LB 0.6702) ensembles",
        "best_alone_lb": 0.6834,
        "v26_alone_lb": 0.6702,
        "v27_alone_lb": 0.6682,
        "prediction_diff_stats": {
            "mean_cm": float(dist_best_v26.mean()*100),
            "p99_cm": float(np.percentile(dist_best_v26, 99)*100),
            "pct_within_1cm": float((dist_best_v26 < 0.01).mean()),
        },
        "candidates_generated": [
            "submission_v28_best{0.5,0.55,0.6,0.65,0.7,0.75,0.8}_v26{...}.csv",
            "submission_v28_boundary_aware_2cm.csv",
            "submission_v28_3way_best{0.6,0.7}.csv",
        ],
    }
    logs = []
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if not isinstance(logs, list): logs = [logs]
        except Exception: logs = []
    logs.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"\n[run_log] {log_path}")
    print("\n생성된 CSV들 → 사용자가 제출해서 LB 확인 필요")
    print("일일 제출 한도 고려해서 1~2개 선택:")
    print(f"  추천 1: submission_v28_best0.65_v26{1-0.65:.2f}.csv  (paradigm 평형)")
    print(f"  추천 2: submission_v28_best0.70_v26{1-0.70:.2f}.csv  (best 우세, 보수)")


if __name__ == "__main__":
    main()
