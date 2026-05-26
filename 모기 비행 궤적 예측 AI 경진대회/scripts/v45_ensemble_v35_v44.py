"""v45_ensemble_v35_v44.py — v35 (GRU+boundary) × v44 (Trans+boundary) ensemble.

v35 OOF 0.6725 (LB 0.6874) + v44 OOF 0.6713. paradigm 다양성 (GRU vs Transformer base).
+ gate (0.6619) + v16 (0.6343) 통합 grid search.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"


def rhit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def main():
    print("=" * 60)
    print("v45: v35 × v44 × gate × v16 ensemble grid")
    print("=" * 60)

    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]

    # Load 4 models
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    oof_v35, test_v35 = st35["oof_v35"], st35["test_v35"]

    st44 = np.load(CACHE_DIR / "v44_state.npz")
    oof_v44, test_v44 = st44["oof_v44"], st44["test_v44"]

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    test_gate = df_best[["x","y","z"]].values.astype(np.float64)

    st16 = np.load(V16_PATH)
    oof_v16, test_v16 = st16["oof"].astype(np.float64), st16["test"].astype(np.float64)

    rh_v35 = rhit(oof_v35, y_train); rh_v44 = rhit(oof_v44, y_train)
    rh_gate = rhit(gate_oof, y_train); rh_v16 = rhit(oof_v16, y_train)
    print(f"v35 OOF: {rh_v35:.4f} (LB 0.6874)")
    print(f"v44 OOF: {rh_v44:.4f}")
    print(f"gate OOF: {rh_gate:.4f} (LB 0.6834)")
    print(f"v16 OOF: {rh_v16:.4f} (LB 0.6488)")

    # Hit pattern
    d_v35 = np.linalg.norm(oof_v35 - y_train, axis=-1)
    d_v44 = np.linalg.norm(oof_v44 - y_train, axis=-1)
    d_gate = np.linalg.norm(gate_oof - y_train, axis=-1)
    h_v35, h_v44, h_gate = d_v35 <= 0.01, d_v44 <= 0.01, d_gate <= 0.01
    either = (h_v35 | h_v44 | h_gate).mean()
    only_v44 = (h_v44 & ~h_v35 & ~h_gate).mean()
    print(f"\nOracle 3way (v35|v44|gate): {either:.4f}")
    print(f"only_v44 (unique to Transformer base + boundary): {only_v44:.4f}")

    # 2-way grid
    print(f"\n=== 2-way grid (v35 × v44) ===")
    best_a, best_r = 0.5, 0
    for a in np.linspace(0.0, 1.0, 21):
        ens = a * oof_v35 + (1-a) * oof_v44
        r = rhit(ens, y_train)
        if r > best_r: best_r, best_a = r, a
    print(f"  best a (v35) = {best_a:.2f}, OOF {best_r:.4f}")

    # 3-way grid (v35, v44, gate)
    print(f"\n=== 3-way grid (v35 + v44 + gate) ===")
    best_3 = None; best_r3 = 0
    for a in np.linspace(0.2, 0.9, 15):
        for b in np.linspace(0.05, min(0.7, 1-a), 14):
            c = 1 - a - b
            if c < 0 or c > 0.5: continue
            ens = a*oof_v35 + b*oof_v44 + c*gate_oof
            r = rhit(ens, y_train)
            if r > best_r3: best_r3, best_3 = r, (a, b, c)
    a3, b3, c3 = best_3
    print(f"  v35={a3:.2f}, v44={b3:.2f}, gate={c3:.2f} → OOF {best_r3:.4f}")

    # 4-way grid (v35, v44, gate, v16)
    print(f"\n=== 4-way grid ===")
    best_4 = None; best_r4 = 0
    for a in np.linspace(0.1, 0.8, 15):
        for b in np.linspace(0.0, min(0.7, 1-a), 15):
            for c in np.linspace(0.0, min(0.5, 1-a-b), 11):
                d = 1 - a - b - c
                if d < 0 or d > 0.2: continue
                ens = a*oof_v35 + b*oof_v44 + c*gate_oof + d*oof_v16
                r = rhit(ens, y_train)
                if r > best_r4: best_r4, best_4 = r, (a, b, c, d)
    a4, b4, c4, d4 = best_4
    print(f"  v35={a4:.2f}, v44={b4:.2f}, gate={c4:.2f}, v16={d4:.2f} → OOF {best_r4:.4f}")

    # Test predictions
    test_2way = best_a * test_v35 + (1-best_a) * test_v44
    test_3way = a3 * test_v35 + b3 * test_v44 + c3 * test_gate
    test_4way = a4 * test_v35 + b4 * test_v44 + c4 * test_gate + d4 * test_v16

    # Summary
    results = {
        "v35 alone (LB 0.6874)": rh_v35,
        "v44 alone": rh_v44,
        "gate alone (LB 0.6834)": rh_gate,
        f"2way v35={best_a:.2f}": best_r,
        f"3way v35={a3:.2f}/v44={b3:.2f}/gate={c3:.2f}": best_r3,
        f"4way v35={a4:.2f}/v44={b4:.2f}/gate={c4:.2f}/v16={d4:.2f}": best_r4,
    }
    test_map = {
        f"2way v35={best_a:.2f}": test_2way,
        f"3way v35={a3:.2f}/v44={b3:.2f}/gate={c3:.2f}": test_3way,
        f"4way v35={a4:.2f}/v44={b4:.2f}/gate={c4:.2f}/v16={d4:.2f}": test_4way,
    }

    print("\n=== All OOF ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        d = v - rh_v35
        print(f"  {k:<55}: {v:.4f}  (Δ vs v35 {d:+.4f})")

    best_name = max(results, key=results.get)
    print(f"\n★★ Best OOF: {best_name}  {results[best_name]:.4f}")
    print(f"   LB 추정 (변환률 +0.0146): {results[best_name] + 0.0146:.4f}")

    # Save top 3 CSVs
    print("\n=== Top 3 CSVs ===")
    saved = []
    for i, (name, r) in enumerate(sorted(results.items(), key=lambda x: -x[1])[:5]):
        if name not in test_map: continue
        safe = name.replace("=","").replace("/","_").replace(",","").replace(" ","").replace("(","").replace(")","")[:50]
        out = DATA_DIR / f"submission_v45_top{i+1}_{safe}.csv"
        tp = test_map[name]
        pd.DataFrame({"id": sub["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}
                     ).to_csv(out, index=False)
        saved.append((name, r, out))
        print(f"  #{i+1} {name}: OOF {r:.4f} → {out.name}")

    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v45_v35_v44_ensemble",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v35 × v44 (GRU vs Trans boundary) × gate × v16 grid",
        "v35_oof": float(rh_v35), "v44_oof": float(rh_v44),
        "gate_oof": float(rh_gate), "v16_oof": float(rh_v16),
        "oracle_3way": float(either),
        "all_results": {k: float(v) for k, v in results.items()},
        "best_name": best_name, "best_oof": float(results[best_name]),
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


if __name__ == "__main__":
    main()
