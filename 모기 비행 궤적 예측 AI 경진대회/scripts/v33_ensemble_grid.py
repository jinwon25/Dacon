"""v33_ensemble_grid.py — v30 (advanced v23) + 사용자 best (gate) + v16 OOF ensemble grid.

v29 교훈: OOF rank ≠ LB rank. 그러나 OOF strong + paradigm 다양 ensemble은 안전.

전략:
  - v30 OOF (0.6588, 5-fold + multi-seed + adv reweight) — paradigm: Kalman residual + GRU
  - gate OOF (0.6619, 5-fold selector + boundary) — paradigm: candidate framework
  - v23 fast OOF (0.6516) — 같은 paradigm as v30 (제외)
  - v16 OOF (0.6343) — paradigm: stacking residual MLP

3 paradigm × grid search:
  1. v30/gate 2-way grid (21 weights)
  2. v30/gate/v16 3-way grid (66 combos)
  3. Sample-wise meta-LGB (v29 같은 함정 회피 위해 per-axis 아닌 joint regression)
  4. Boundary-aware (v30/gate 가까운 sample만 평균)

Test prediction CSV로 top 3 후보 저장.
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
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"


def rhit(p, y, mask=None):
    if mask is not None: p, y = p[mask], y[mask]
    return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def main():
    print("=" * 60)
    print("v33 ensemble grid: v30 + gate (사용자 best) + v16")
    print("=" * 60)

    # --- Load ---
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]

    # v30
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    oof_v30_res  = (st30["oof_A"] + st30["oof_B"]) / 2
    test_v30_res = (st30["test_A"] + st30["test_B"]) / 2
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    oof_v30  = kalman_train + oof_v30_res  * ALPHA
    test_v30 = kalman_test  + test_v30_res * ALPHA
    print(f"v30 OOF R-Hit: {rhit(oof_v30, y_train):.4f}")

    # 사용자 best
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    print(f"gate OOF R-Hit: {rhit(gate_oof, y_train):.4f}")

    # v16
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)
    print(f"v16 OOF R-Hit: {rhit(oof_v16, y_train):.4f}")

    # gate test
    df_best = pd.read_csv(BEST_TEST)
    test_gate = df_best[["x","y","z"]].values.astype(np.float64)

    # --- Hit pattern ---
    d_v30  = np.linalg.norm(oof_v30 - y_train, axis=-1)
    d_gate = np.linalg.norm(gate_oof - y_train, axis=-1)
    d_v16  = np.linalg.norm(oof_v16 - y_train, axis=-1)
    h_v30 = d_v30 <= 0.01; h_gate = d_gate <= 0.01; h_v16 = d_v16 <= 0.01
    either3 = (h_v30 | h_gate | h_v16).mean()
    all3 = (h_v30 & h_gate & h_v16).mean()
    only_v30 = (h_v30 & ~h_gate & ~h_v16).mean()
    only_gate = (~h_v30 & h_gate & ~h_v16).mean()
    only_v16 = (~h_v30 & ~h_gate & h_v16).mean()
    print(f"\n=== 3-way hit pattern ===")
    print(f"  all 3   : {all3:.4f}")
    print(f"  only v30: {only_v30:.4f}")
    print(f"  only gate: {only_gate:.4f}")
    print(f"  only v16: {only_v16:.4f}")
    print(f"  EITHER 3 (oracle): {either3:.4f}  ★")

    rh_v30, rh_gate, rh_v16 = rhit(oof_v30, y_train), rhit(gate_oof, y_train), rhit(oof_v16, y_train)

    # --- 2-way grid (v30 × gate) ---
    print(f"\n=== 2-way grid (v30 × gate) ===")
    best_2 = (0.5, 0.5, 0.0)
    best_r = 0
    for a in np.linspace(0.0, 1.0, 21):
        ens = a * oof_v30 + (1-a) * gate_oof
        r = rhit(ens, y_train)
        if r > best_r:
            best_r, best_2 = r, a
    print(f"  best a (v30) = {best_2:.2f}, OOF {best_r:.4f}")
    best_2_combo = (best_2, 1 - best_2, 0.0)
    ens_2way = best_2 * oof_v30 + (1-best_2) * gate_oof
    test_2way = best_2 * test_v30 + (1-best_2) * test_gate

    # --- 3-way grid (v30 × gate × v16) ---
    print(f"\n=== 3-way grid (v30 × gate × v16) ===")
    best_3 = None; best_r3 = 0
    for a in np.linspace(0.1, 0.9, 17):
        for b in np.linspace(0.05, min(0.9, 1-a), 18):
            c = 1 - a - b
            if c < 0.0 or c > 0.5: continue
            ens = a * oof_v30 + b * gate_oof + c * oof_v16
            r = rhit(ens, y_train)
            if r > best_r3:
                best_r3, best_3 = r, (a, b, c)
    a, b, c = best_3
    ens_3way = a * oof_v30 + b * gate_oof + c * oof_v16
    test_3way = a * test_v30 + b * test_gate + c * test_v16
    print(f"  best weights: v30={a:.2f}, gate={b:.2f}, v16={c:.2f} → OOF {best_r3:.4f}")

    # --- Boundary-aware (v30 vs gate 가까우면 평균) ---
    dist_vg = np.linalg.norm(oof_v30 - gate_oof, axis=-1)
    close = dist_vg < 0.02
    ens_bnd = np.where(close[:, None], (oof_v30 + gate_oof)/2, gate_oof)
    rh_bnd = rhit(ens_bnd, y_train)
    print(f"\nboundary-aware (close→avg, far→gate): OOF {rh_bnd:.4f}")
    dist_vg_test = np.linalg.norm(test_v30 - test_gate, axis=-1)
    close_test = dist_vg_test < 0.02
    test_bnd = np.where(close_test[:, None], (test_v30 + test_gate)/2, test_gate)

    # --- Boundary-aware with v30 preference (v30 strong here) ---
    ens_bnd_v30 = np.where(close[:, None], (oof_v30 + gate_oof)/2, oof_v30)
    rh_bnd_v30 = rhit(ens_bnd_v30, y_train)
    print(f"boundary-aware (close→avg, far→v30): OOF {rh_bnd_v30:.4f}")
    test_bnd_v30 = np.where(close_test[:, None], (test_v30 + test_gate)/2, test_v30)

    # --- Joint meta-LGB (v29 함정 회피: joint 3D regression, residual capped) ---
    print(f"\n=== Joint meta-LGB (joint residual prediction, cap 5mm) ===")
    try:
        import lightgbm as lgb
        feat = np.concatenate([
            oof_v30, gate_oof, oof_v16,
            oof_v30 - gate_oof, oof_v30 - oof_v16, gate_oof - oof_v16,
            np.linalg.norm(oof_v30 - gate_oof, axis=-1, keepdims=True),
            np.linalg.norm(oof_v30 - kalman_train, axis=-1, keepdims=True),
            np.linalg.norm(gate_oof - kalman_train, axis=-1, keepdims=True),
        ], axis=1)
        # Target = best single (gate) + residual capped
        ens_meta = gate_oof.copy()
        kf = KFold(5, shuffle=True, random_state=0)
        for j, ax in enumerate(["x","y","z"]):
            target = y_train[:, j] - gate_oof[:, j]
            pred = np.zeros(len(y_train))
            for tr, va in kf.split(feat):
                gbm = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200,
                                         min_child_samples=30, verbose=-1, random_state=0)
                gbm.fit(feat[tr], target[tr])
                pred[va] = gbm.predict(feat[va])
            ens_meta[:, j] = gate_oof[:, j] + np.clip(pred, -0.005, 0.005)
        rh_meta = rhit(ens_meta, y_train)
        print(f"  meta-LGB OOF: {rh_meta:.4f}")

        # Test
        gbm_full = {}
        feat_test = np.concatenate([
            test_v30, test_gate, test_v16,
            test_v30 - test_gate, test_v30 - test_v16, test_gate - test_v16,
            np.linalg.norm(test_v30 - test_gate, axis=-1, keepdims=True),
            np.linalg.norm(test_v30 - kalman_test, axis=-1, keepdims=True),
            np.linalg.norm(test_gate - kalman_test, axis=-1, keepdims=True),
        ], axis=1)
        test_meta = test_gate.copy()
        for j in range(3):
            target = y_train[:, j] - gate_oof[:, j]
            gbm = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200,
                                     min_child_samples=30, verbose=-1, random_state=0)
            gbm.fit(feat, target)
            pred = np.clip(gbm.predict(feat_test), -0.005, 0.005)
            test_meta[:, j] = test_gate[:, j] + pred
    except Exception as e:
        print(f"  meta-LGB skip: {e}")
        rh_meta = 0; ens_meta = None; test_meta = None

    # --- Summary + top 3 ---
    results = {
        f"v30 alone": rh_v30,
        f"gate alone": rh_gate,
        f"v16 alone": rh_v16,
        f"2way_v30={best_2:.2f}": best_r,
        f"3way_v30={a:.2f}_gate={b:.2f}_v16={c:.2f}": best_r3,
        "boundary_aware_far_gate": rh_bnd,
        "boundary_aware_far_v30": rh_bnd_v30,
    }
    if ens_meta is not None:
        results["meta_lgb_capped5mm"] = rh_meta

    print("\n=== All OOF results ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        d_v30 = v - rh_v30; d_gate = v - rh_gate
        print(f"  {k:<45s}: {v:.4f}  (Δ v30 {d_v30:+.4f}, Δ gate {d_gate:+.4f})")

    # Best 선택
    best_name = max(results, key=results.get)
    print(f"\n★★ Best OOF: {best_name}  {results[best_name]:.4f}")

    # 저장: top 3
    test_map = {
        "v30 alone": test_v30,
        "gate alone": test_gate,
        f"2way_v30={best_2:.2f}": test_2way,
        f"3way_v30={a:.2f}_gate={b:.2f}_v16={c:.2f}": test_3way,
        "boundary_aware_far_gate": test_bnd,
        "boundary_aware_far_v30": test_bnd_v30,
    }
    if ens_meta is not None:
        test_map["meta_lgb_capped5mm"] = test_meta

    top3 = sorted(results.items(), key=lambda x: -x[1])[:5]
    print("\n=== Top 5 OOF → CSV 저장 ===")
    saved = []
    for i, (name, r) in enumerate(top3):
        if name not in test_map: continue
        tp = test_map[name]
        safe = name.replace("=","").replace(",","").replace("(","").replace(")","").replace(" ","_")[:40]
        out_csv = DATA_DIR / f"submission_v33_top{i+1}_{safe}.csv"
        pd.DataFrame({"id": sub["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}
                     ).to_csv(out_csv, index=False)
        saved.append((name, r, out_csv))
        print(f"  #{i+1} {name}: OOF {r:.4f} → {out_csv.name}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v33_oof_ensemble",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v30 (advanced v23) × gate (사용자 best) × v16 grid search ensemble",
        "v30_oof": float(rh_v30),
        "gate_oof": float(rh_gate),
        "v16_oof": float(rh_v16),
        "oracle_3way": float(either3),
        "all_results": {k: float(v) for k, v in results.items()},
        "best_name": best_name,
        "best_oof": float(results[best_name]),
        "top_csvs": [str(p) for _,_,p in saved],
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


if __name__ == "__main__":
    main()
