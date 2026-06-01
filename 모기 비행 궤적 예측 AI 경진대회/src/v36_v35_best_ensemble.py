"""v36_v35_best_ensemble.py — v35 (LB 0.6874) × 사용자 best CSV (LB 0.6834) 단순 ensemble.

LB 검증된 두 모델의 paradigm 다양성:
  - v35 = v30 (Kalman+GRU+adv) + boundary MLP — paradigm: residual prediction
  - best = selector + boundary_tiny — paradigm: candidate framework

OOF 분석 후 best 후보 3개 CSV 저장.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"

BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"


def rhit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def main():
    print("=" * 60)
    print("v36: v35 (LB 0.6874) × best (LB 0.6834) ensemble")
    print("=" * 60)

    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)

    # v35
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    oof_v35 = st35["oof_v35"]; test_v35 = st35["test_v35"]
    rh_v35 = rhit(oof_v35, y_train)
    print(f"\nv35 OOF: {rh_v35:.4f}  (LB 0.6874)")

    # best
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    test_best = df_best[["x","y","z"]].values.astype(np.float64)
    rh_gate = rhit(gate_oof, y_train)
    print(f"gate OOF: {rh_gate:.4f}  (LB 0.6834)")

    # Hit pattern
    d_v35 = np.linalg.norm(oof_v35 - y_train, axis=-1)
    d_gate = np.linalg.norm(gate_oof - y_train, axis=-1)
    h_v35 = d_v35 <= 0.01; h_gate = d_gate <= 0.01
    either = (h_v35 | h_gate).mean()
    both = (h_v35 & h_gate).mean()
    only_v35 = (h_v35 & ~h_gate).mean()
    only_gate = (~h_v35 & h_gate).mean()
    print(f"\n=== Hit pattern ===")
    print(f"  both    : {both:.4f}")
    print(f"  only v35: {only_v35:.4f}")
    print(f"  only gate: {only_gate:.4f}")
    print(f"  EITHER  : {either:.4f}  ★ (ensemble ceiling)")

    # Prediction distance
    dist = np.linalg.norm(oof_v35 - gate_oof, axis=-1)
    print(f"\n=== Prediction distance ===")
    print(f"  mean   : {dist.mean()*100:.3f}cm")
    print(f"  p99    : {np.percentile(dist, 99)*100:.3f}cm")
    print(f"  <1cm   : {(dist < 0.01).mean()*100:.2f}%")
    print(f"  <2cm   : {(dist < 0.02).mean()*100:.2f}%")

    dist_test = np.linalg.norm(test_v35 - test_best, axis=-1)
    print(f"\n=== Test prediction distance ===")
    print(f"  mean: {dist_test.mean()*100:.3f}cm, p99: {np.percentile(dist_test, 99)*100:.3f}cm")

    # Grid weight ensemble (v35 weight)
    print(f"\n=== weight grid (v35 weight) ===")
    results = {"v35 alone": rh_v35, "gate alone": rh_gate}
    test_map = {"v35 alone": test_v35, "gate alone": test_best}
    best_a, best_r = 1.0, rh_v35
    for a in np.linspace(0.0, 1.0, 21):
        ens = a * oof_v35 + (1-a) * gate_oof
        r = rhit(ens, y_train)
        name = f"v35={a:.2f}"
        results[name] = r
        test_map[name] = a * test_v35 + (1-a) * test_best
        if r > best_r: best_r, best_a = r, a
    print(f"  best a (v35) = {best_a:.2f}, OOF {best_r:.4f}")

    # Boundary-aware: 두 가까우면 평균, 멀면 v35 (LB 더 강한 쪽)
    THR = 0.02
    close = dist < THR
    close_test = dist_test < THR
    ens_bnd_v35 = np.where(close[:, None], (oof_v35 + gate_oof)/2, oof_v35)
    test_bnd_v35 = np.where(close_test[:, None], (test_v35 + test_best)/2, test_v35)
    results["boundary_aware_far_v35"] = rhit(ens_bnd_v35, y_train)
    test_map["boundary_aware_far_v35"] = test_bnd_v35

    # Boundary-aware: 가까우면 평균, 멀면 gate
    ens_bnd_gate = np.where(close[:, None], (oof_v35 + gate_oof)/2, gate_oof)
    test_bnd_gate = np.where(close_test[:, None], (test_v35 + test_best)/2, test_best)
    results["boundary_aware_far_gate"] = rhit(ens_bnd_gate, y_train)
    test_map["boundary_aware_far_gate"] = test_bnd_gate

    # Print all
    print(f"\n=== All ensembles (OOF) ===")
    for name, r in sorted(results.items(), key=lambda x: -x[1])[:12]:
        d_v35 = r - rh_v35; d_gate = r - rh_gate
        print(f"  {name:<35}: {r:.4f}  (Δ v35 {d_v35:+.4f}, Δ gate {d_gate:+.4f})")

    # Best
    best_name = max(results, key=results.get)
    print(f"\n★★ Best OOF: {best_name}  {results[best_name]:.4f}")

    # Save top 3 CSV
    print(f"\n=== Top 3 ensemble CSVs ===")
    saved = []
    for i, (name, r) in enumerate(sorted(results.items(), key=lambda x: -x[1])[:5]):
        if name in ("v35 alone", "gate alone"): continue  # 이미 있음
        if name not in test_map: continue
        safe = name.replace("=","").replace(",","").replace(" ","").replace("(","").replace(")","")[:35]
        out_csv = DATA_DIR / f"submission_v36_top{i+1}_{safe}.csv"
        tp = test_map[name]
        pd.DataFrame({"id": sub["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}
                     ).to_csv(out_csv, index=False)
        saved.append((name, r, out_csv))
        print(f"  #{i+1} {name}: OOF {r:.4f} → {out_csv.name}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v36_v35_best_ensemble",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v35 (LB 0.6874) × best (LB 0.6834) weight grid + boundary-aware",
        "v35_oof": float(rh_v35), "gate_oof": float(rh_gate),
        "v35_lb": 0.6874, "gate_lb": 0.6834,
        "oracle_either": float(either),
        "only_v35": float(only_v35), "only_gate": float(only_gate),
        "best_name": best_name, "best_oof": float(results[best_name]),
        "all_results": {k: float(v) for k, v in results.items()},
        "saved_csvs": [str(p) for _,_,p in saved],
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
    print(f"\n[log] {log_path}")


if __name__ == "__main__":
    main()
