"""v29_oof_ensemble.py — 사용자의 best OOF (selector+boundary, 0.6619) × 우리 v23/v26 통합.

발견:
  - outputs/02_boundary_oof/cap0p004_apply1_seed20260606/boundary_oof_predictions.npz
    └── soft_pred (0.6605), gate_pred (0.6619), argmax_pred (0.6595) [5-fold OOF, 10000 rows]
  - 사용자 best LB 0.6834 = gate_pred → test → 0.6834

직접 OOF 비교 + sample-wise selection으로 최적 ensemble 도출.
test prediction은 outputs/00_submit/ (gate) + 우리 submission_v23/v26/v27 CSV에서.

ensemble candidates:
  1. 사용자 gate alone (baseline)
  2. 단순 평균 (gate + v23 + v26)
  3. weighted (OOF rhit 비례)
  4. per-axis grid
  5. sample-wise meta-LGB
  6. boundary-aware (둘 가까우면 평균, 멀면 gate)

사용법: python scripts/v29_oof_ensemble.py
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
BEST_TEST_GATE = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
BEST_TEST_SOFT = PROJECT_DIR / "outputs" / "01_best_public_0p6834" / "boundary_gate_seed20260606_apply1p0" / "submission_boundary_tiny_soft.csv"
BEST_TEST_ARGMAX = PROJECT_DIR / "outputs" / "01_best_public_0p6834" / "boundary_gate_seed20260606_apply1p0" / "submission_boundary_tiny_argmax.csv"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"


def rhit(p, y, mask=None, thr=0.01):
    if mask is not None: p, y = p[mask], y[mask]
    return float((np.linalg.norm(p - y, axis=-1) <= thr).mean())


def main():
    print("=" * 60)
    print("v29: 사용자 selector+boundary OOF × 우리 v23/v26 통합")
    print("=" * 60)

    # --- Load 사용자의 best OOF ---
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    y_oof    = bo["y"].astype(np.float64)
    gate_oof = bo["gate_pred"].astype(np.float64)
    soft_oof = bo["soft_pred"].astype(np.float64)
    argmax_oof = bo["argmax_pred"].astype(np.float64)
    print(f"\n[best OOF] {len(y_oof)} rows from {BEST_OOF_PATH.name}")
    print(f"  gate   R-Hit: {rhit(gate_oof, y_oof):.4f}")
    print(f"  soft   R-Hit: {rhit(soft_oof, y_oof):.4f}")
    print(f"  argmax R-Hit: {rhit(argmax_oof, y_oof):.4f}")

    # --- Load 우리 v23/v26/v27 OOF (cache + 빌드) ---
    # y_train order = sorted train_files
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids_ours = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])

    # id 정렬 동일 확인
    if not np.array_equal(best_ids, train_ids_ours):
        print(f"  ⚠️ id 순서 다름. user_ids[0]={best_ids[0]}, ours[0]={train_ids_ours[0]}")
        # reindex
        idx_map = {i: k for k, i in enumerate(train_ids_ours)}
        perm = np.array([idx_map[i] for i in best_ids])
    else:
        perm = None

    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    y_train = labels.set_index("id").loc[list(train_ids_ours)][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]

    st = np.load(CACHE_DIR / "v23_state_fast.npz")
    oof_A, test_A = st["oof_A"], st["test_A"]
    oof_B, test_B = st["oof_B"], st["test_B"]
    fold_mask = st["fold_mask_A"] & st["fold_mask_B"]
    oof_v23_res  = (oof_A + oof_B) / 2
    test_v23_res = (test_A + test_B) / 2
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    oof_v23  = kalman_train + oof_v23_res  * ALPHA
    test_v23 = kalman_test  + test_v23_res * ALPHA
    print(f"\n[ours] v23 OOF R-Hit (covered {fold_mask.sum()}): {rhit(oof_v23, y_train, fold_mask):.4f}")

    # v16
    st16 = np.load(V16_PATH)
    oof_v16  = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)
    print(f"[ours] v16 OOF R-Hit: {rhit(oof_v16, y_train, fold_mask):.4f}")

    # --- best (사용자) OOF/test 우리 순서로 reindex ---
    if perm is not None:
        gate_oof_r = gate_oof[perm]; soft_oof_r = soft_oof[perm]; argmax_oof_r = argmax_oof[perm]
    else:
        gate_oof_r, soft_oof_r, argmax_oof_r = gate_oof, soft_oof, argmax_oof
    print(f"[best] gate OOF R-Hit (covered {fold_mask.sum()}): {rhit(gate_oof_r, y_train, fold_mask):.4f}")

    # --- Test predictions 로드 ---
    df_best_test = pd.read_csv(BEST_TEST_GATE)
    test_ids_sub = df_best_test["id"].values

    # 우리 sub의 id 순서와 같은지
    sub_template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert np.array_equal(df_best_test["id"].values, sub_template["id"].values), "best test id mismatch"
    test_best = df_best_test[["x","y","z"]].values.astype(np.float64)

    df_v26_test = pd.read_csv(DATA_DIR / "submission_v26_cpu_fast.csv")
    test_v26 = df_v26_test[["x","y","z"]].values.astype(np.float64)

    # --- Hit pattern 분석 ---
    print("\n=== Hit pattern overlap (covered) ===")
    d_gate = np.linalg.norm(gate_oof_r - y_train, axis=-1)
    d_v23  = np.linalg.norm(oof_v23 - y_train, axis=-1)
    d_v16  = np.linalg.norm(oof_v16 - y_train, axis=-1)
    hit_gate = d_gate <= 0.01
    hit_v23  = d_v23 <= 0.01
    hit_v16  = d_v16 <= 0.01
    m = fold_mask
    either3 = (hit_gate[m] | hit_v23[m] | hit_v16[m]).mean()
    all3    = (hit_gate[m] & hit_v23[m] & hit_v16[m]).mean()
    only_gate = (hit_gate[m] & ~hit_v23[m] & ~hit_v16[m]).mean()
    only_v23  = (~hit_gate[m] & hit_v23[m] & ~hit_v16[m]).mean()
    only_v16  = (~hit_gate[m] & ~hit_v23[m] & hit_v16[m]).mean()
    print(f"  gate hit  : {hit_gate[m].mean():.4f}")
    print(f"  v23 hit   : {hit_v23[m].mean():.4f}")
    print(f"  v16 hit   : {hit_v16[m].mean():.4f}")
    print(f"  ALL 3 hit : {all3:.4f}")
    print(f"  only gate : {only_gate:.4f}")
    print(f"  only v23  : {only_v23:.4f}")
    print(f"  only v16  : {only_v16:.4f}")
    print(f"  EITHER 3  : {either3:.4f}  ★ oracle 3-way")

    # --- Ensemble candidates ---
    print("\n=== Ensemble candidates (OOF, covered) ===")
    candidates_oof = {}
    candidates_oof["gate alone"] = gate_oof_r
    candidates_oof["v23 alone"] = oof_v23
    candidates_oof["v26 alone"] = None  # cap에서 추가

    # 단순 평균
    candidates_oof["avg(gate, v23)"] = (gate_oof_r + oof_v23) / 2
    candidates_oof["avg(gate, v23, v16)"] = (gate_oof_r + oof_v23 + oof_v16) / 3
    candidates_oof["avg(gate, v16)"] = (gate_oof_r + oof_v16) / 2

    # OOF rhit 비례 weight
    rh_gate = rhit(gate_oof_r, y_train, m)
    rh_v23  = rhit(oof_v23, y_train, m)
    rh_v16  = rhit(oof_v16, y_train, m)
    w_g = rh_gate / (rh_gate + rh_v23 + rh_v16)
    w_2 = rh_v23  / (rh_gate + rh_v23 + rh_v16)
    w_3 = rh_v16  / (rh_gate + rh_v23 + rh_v16)
    candidates_oof[f"weighted_{w_g:.2f}_{w_2:.2f}_{w_3:.2f}"] = w_g*gate_oof_r + w_2*oof_v23 + w_3*oof_v16

    # Global α grid (gate vs v23)
    best_a, best_r = 0.5, 0
    for a in np.linspace(0.0, 1.0, 21):
        ens = a * gate_oof_r + (1-a) * oof_v23
        r = rhit(ens, y_train, m)
        if r > best_r: best_r, best_a = r, a
    candidates_oof[f"global_α_gate={best_a:.2f}"] = best_a * gate_oof_r + (1-best_a) * oof_v23

    # 3-way grid (gate, v23, v16 weights sum to 1)
    best_g, best_2, best_3, best_r3 = 0.5, 0.3, 0.2, 0
    for a in np.linspace(0.3, 0.9, 13):
        for b in np.linspace(0.0, 1-a, 11):
            c = 1 - a - b
            if c < 0: continue
            ens = a * gate_oof_r + b * oof_v23 + c * oof_v16
            r = rhit(ens, y_train, m)
            if r > best_r3:
                best_r3, best_g, best_2, best_3 = r, a, b, c
    candidates_oof[f"3way_{best_g:.2f}_{best_2:.2f}_{best_3:.2f}"] = (
        best_g * gate_oof_r + best_2 * oof_v23 + best_3 * oof_v16)

    # Boundary-aware: gate vs v23 가까울 때만 평균
    dist_gv = np.linalg.norm(gate_oof_r - oof_v23, axis=-1)
    close = dist_gv < 0.02
    candidates_oof["boundary_aware(gate, v23, 2cm)"] = np.where(close[:,None],
                                                                  (gate_oof_r + oof_v23)/2, gate_oof_r)

    # Sample-wise meta-LGB
    try:
        import lightgbm as lgb
        feat = np.concatenate([
            gate_oof_r, oof_v23, oof_v16,
            gate_oof_r - oof_v23, gate_oof_r - oof_v16, oof_v23 - oof_v16,
            np.linalg.norm(gate_oof_r - oof_v23, axis=-1, keepdims=True),
            np.linalg.norm(gate_oof_r - kalman_train, axis=-1, keepdims=True),
            np.linalg.norm(oof_v23 - kalman_train, axis=-1, keepdims=True),
        ], axis=1)

        ens_meta_oof = np.zeros_like(gate_oof_r)
        kf = KFold(5, shuffle=True, random_state=0)
        # Per-axis LGB regress to y - gate (residual, capped)
        for j, ax in enumerate(["x","y","z"]):
            target = y_train[:, j] - gate_oof_r[:, j]
            pred_oof = np.zeros(len(y_train))
            for tr, va in kf.split(feat[m]):
                tr_idx = np.where(m)[0][tr]; va_idx = np.where(m)[0][va]
                gbm = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200,
                                          min_child_samples=20, verbose=-1, random_state=0)
                gbm.fit(feat[tr_idx], target[tr_idx])
                pred_oof[va_idx] = gbm.predict(feat[va_idx])
            # Cap correction to 5mm
            pred_oof = np.clip(pred_oof, -0.005, 0.005)
            ens_meta_oof[:, j] = gate_oof_r[:, j] + pred_oof
        candidates_oof["meta_lgb_residual_5mm"] = ens_meta_oof
    except Exception as e:
        print(f"  meta_lgb skip: {e}")

    # 결과
    results = {}
    for name, ens in candidates_oof.items():
        if ens is None: continue
        r = rhit(ens, y_train, m)
        results[name] = r
        delta = r - rh_gate
        print(f"  {name:<48}: {r:.4f}  (Δ vs gate {delta:+.4f})")

    best_name = max(results, key=results.get)
    best_rh = results[best_name]
    print(f"\n★★ Best OOF ensemble: {best_name}  {best_rh:.4f}  (vs gate alone Δ {best_rh - rh_gate:+.4f})")

    # --- Test 생성 ---
    # gate, v23, v16 test 다 있음. 같은 weight로 test 생성.
    def test_ensemble(name):
        if name == "gate alone": return test_best
        if name == "v23 alone":  return test_v23
        if name.startswith("avg(gate, v23, v16)"): return (test_best + test_v23 + test_v16) / 3
        if name == "avg(gate, v23)": return (test_best + test_v23) / 2
        if name == "avg(gate, v16)": return (test_best + test_v16) / 2
        if name.startswith("weighted_"):
            return w_g*test_best + w_2*test_v23 + w_3*test_v16
        if name.startswith("global_α_gate"):
            return best_a * test_best + (1-best_a) * test_v23
        if name.startswith("3way_"):
            return best_g * test_best + best_2 * test_v23 + best_3 * test_v16
        if name.startswith("boundary_aware"):
            dist = np.linalg.norm(test_best - test_v23, axis=-1)
            return np.where((dist < 0.02)[:,None], (test_best + test_v23)/2, test_best)
        if name == "meta_lgb_residual_5mm":
            # Train full meta on covered
            feat_te = np.concatenate([
                test_best, test_v23, test_v16,
                test_best - test_v23, test_best - test_v16, test_v23 - test_v16,
                np.linalg.norm(test_best - test_v23, axis=-1, keepdims=True),
                np.linalg.norm(test_best - kalman_test, axis=-1, keepdims=True),
                np.linalg.norm(test_v23 - kalman_test, axis=-1, keepdims=True),
            ], axis=1)
            out_te = test_best.copy()
            for j in range(3):
                target = (y_train[:, j] - gate_oof_r[:, j])[m]
                gbm = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200,
                                          min_child_samples=20, verbose=-1, random_state=0)
                gbm.fit(feat[m], target)
                pred = np.clip(gbm.predict(feat_te), -0.005, 0.005)
                out_te[:, j] = test_best[:, j] + pred
            return out_te
        return None

    test_pred = test_ensemble(best_name)
    if test_pred is None:
        print(f"!! name not mapped: {best_name}, fallback gate")
        test_pred = test_best

    # Top 3 후보도 csv 저장 (제출 다양성)
    top3 = sorted(results.items(), key=lambda x: -x[1])[:3]
    print(f"\n=== Top 3 OOF ensembles ===")
    saved = []
    for i, (name, r) in enumerate(top3):
        tp = test_ensemble(name)
        if tp is None: continue
        safe = name.replace("=","").replace(",","").replace("(","").replace(")","").replace(" ","_").replace("__","_")[:40]
        out_csv = DATA_DIR / f"submission_v29_top{i+1}_{safe}.csv"
        pd.DataFrame({"id": sub_template["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}
                     ).to_csv(out_csv, index=False)
        saved.append((name, r, out_csv))
        print(f"  #{i+1} {name}: OOF {r:.4f} → {out_csv.name}")

    # --- run_log ---
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v29_oof_ensemble",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "사용자 selector+boundary OOF (0.6619) × 우리 v23(0.6516)/v16(0.6342) 통합",
        "gate_alone_oof": float(rh_gate),
        "v23_alone_oof": float(rh_v23),
        "v16_alone_oof": float(rh_v16),
        "oracle_3way_either": float(either3),
        "all_oof_candidates": {k: float(v) for k, v in results.items()},
        "best_oof_name": best_name,
        "best_oof_rhit": float(best_rh),
        "top3_csvs": [str(p) for _,_,p in saved],
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
    print("\n" + "=" * 60)
    print(f"Best: {best_name}  OOF {best_rh:.4f}")
    print(f"  vs gate alone OOF 0.6619 → LB 0.6834")
    print(f"  추정 LB +0.020 (gate 변환률 기준): {best_rh + 0.0215:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
