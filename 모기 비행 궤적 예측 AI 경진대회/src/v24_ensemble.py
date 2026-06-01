"""v24_ensemble.py — v23 (CPU/{mode}) + v16 (archive/v16_stack_oof.npz) ensemble.

학습 없음 (~1분). 6가지 ensemble 후보 OOF 평가 → best 선택 → submission_v24_cpu_{mode}.csv

사용법:
  python scripts/v24_ensemble.py --v23-mode micro
  python scripts/v24_ensemble.py --v23-mode fast
  python scripts/v24_ensemble.py --v23-mode full

전제:
  - cache/v23_state_{mode}.npz (v23 학습 완료)
  - cache/kalman.npz
  - archive/v16_stack_oof.npz
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"


def rhit(p, y, mask=None):
    if mask is not None:
        p, y = p[mask], y[mask]
    return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23-mode", default="micro", choices=["micro", "fast", "full"])
    args = parser.parse_args()
    mode = args.v23_mode

    print("=" * 60)
    print(f"v24 ensemble (v23 mode={mode} + v16 stacking)")
    print("=" * 60)

    # --- Load ---
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub    = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]

    state_path = CACHE_DIR / f"v23_state_{mode}.npz"
    assert state_path.exists(), f"v23 state 없음. 먼저 v23 실행: {state_path}"
    st = np.load(state_path)
    oof_A, test_A = st["oof_A"], st["test_A"]
    fold_mask_A = st["fold_mask_A"]
    has_B = bool(st.get("has_B", np.array(False)))
    if has_B:
        oof_B, test_B = st["oof_B"], st["test_B"]
        fold_mask_B = st["fold_mask_B"]
        oof_sub09 = (oof_A + oof_B) / 2
        test_sub09 = (test_A + test_B) / 2
        eval_mask = fold_mask_A & fold_mask_B
    else:
        oof_sub09 = oof_A.copy(); test_sub09 = test_A.copy()
        eval_mask = fold_mask_A

    print(f"v23 covered rows: {eval_mask.sum()}/{len(y_train)}  (has_B={has_B})")

    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    oof_v23_abs  = kalman_train + oof_sub09  * ALPHA
    test_v23_abs = kalman_test  + test_sub09 * ALPHA

    assert V16_PATH.exists(), f"v16 stack 없음: {V16_PATH}"
    st16 = np.load(V16_PATH)
    oof_v16_abs  = st16["oof"].astype(np.float64)
    test_v16_abs = st16["test"].astype(np.float64)

    rh_v23 = rhit(oof_v23_abs, y_train, eval_mask)
    rh_v16 = rhit(oof_v16_abs, y_train, eval_mask)
    print(f"v23 OOF R-Hit (covered): {rh_v23:.4f}")
    print(f"v16 OOF R-Hit (same rows): {rh_v16:.4f}")

    # --- Pattern overlap ---
    d23 = np.linalg.norm(oof_v23_abs - y_train, axis=-1)
    d16 = np.linalg.norm(oof_v16_abs - y_train, axis=-1)
    hit23 = d23 <= 0.01; hit16 = d16 <= 0.01
    m = eval_mask
    either  = (hit23[m] | hit16[m]).mean()
    both    = (hit23[m] & hit16[m]).mean()
    only_23 = (hit23[m] & ~hit16[m]).mean()
    only_16 = (~hit23[m] & hit16[m]).mean()

    print("\n=== Hit pattern overlap ===")
    print(f"  both hit  : {both:.4f}")
    print(f"  only v23  : {only_23:.4f}")
    print(f"  only v16  : {only_16:.4f}")
    print(f"  EITHER (ensemble ceiling): {either:.4f}  ★")

    dist12 = np.linalg.norm(oof_v23_abs - oof_v16_abs, axis=-1)
    print(f"\nv23 vs v16 예측 거리 (covered): mean={dist12[m].mean()*100:.3f}cm, "
          f"<1cm 비율={(dist12[m] < 0.01).mean():.4f}")

    # --- Ensemble candidates ---
    print("\n=== 6 ensemble candidates (OOF) ===")
    candidates_oof = {}
    candidates_oof["simple_avg"] = (oof_v23_abs + oof_v16_abs) / 2

    w23 = rh_v23 / max(rh_v23 + rh_v16, 1e-6); w16 = 1 - w23
    candidates_oof[f"weighted_w23={w23:.3f}"] = w23 * oof_v23_abs + w16 * oof_v16_abs

    THR = 0.02
    close = dist12 < THR
    best_single = oof_v23_abs if rh_v23 >= rh_v16 else oof_v16_abs
    candidates_oof["boundary_aware_2cm"] = np.where(close[:, None],
                                                      (oof_v23_abs + oof_v16_abs) / 2, best_single)

    best_a, best_rh_a = 0.5, -1
    for a in np.linspace(0.0, 1.0, 21):
        ens = a * oof_v23_abs + (1 - a) * oof_v16_abs
        r = rhit(ens, y_train, eval_mask)
        if r > best_rh_a: best_rh_a, best_a = r, a
    candidates_oof[f"global_α={best_a:.2f}"] = best_a * oof_v23_abs + (1 - best_a) * oof_v16_abs

    per_axis = np.full(3, best_a)
    for j in range(3):
        best_aj, best_rj = best_a, -1
        for a in np.linspace(0.0, 1.0, 21):
            ens = best_a * oof_v23_abs + (1 - best_a) * oof_v16_abs
            ens[:, j] = a * oof_v23_abs[:, j] + (1 - a) * oof_v16_abs[:, j]
            r = rhit(ens, y_train, eval_mask)
            if r > best_rj: best_rj, best_aj = r, a
        per_axis[j] = best_aj
    candidates_oof[f"per_axis_α={per_axis.round(2).tolist()}"] = (
        per_axis[None, :] * oof_v23_abs + (1 - per_axis[None, :]) * oof_v16_abs)

    # Meta-LGB
    try:
        import lightgbm as lgb
        feat = np.concatenate([
            oof_v23_abs, oof_v16_abs, oof_v23_abs - oof_v16_abs, dist12[:, None],
            np.linalg.norm(oof_v23_abs - kalman_train, axis=-1, keepdims=True),
            np.linalg.norm(oof_v16_abs - kalman_train, axis=-1, keepdims=True),
        ], axis=1)
        target = (d23 < d16).astype(np.float32)
        feat_em, target_em = feat[eval_mask], target[eval_mask]
        meta_w_em = np.zeros(eval_mask.sum())
        kf = KFold(5, shuffle=True, random_state=0)
        for tr, va in kf.split(feat_em):
            gbm = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200,
                                     min_child_samples=20, verbose=-1, random_state=0)
            gbm.fit(feat_em[tr], target_em[tr])
            meta_w_em[va] = np.clip(gbm.predict(feat_em[va]), 0.0, 1.0)
        meta_w = np.full(len(y_train), 0.5)
        meta_w[eval_mask] = meta_w_em
        candidates_oof["meta_lgb"] = meta_w[:, None] * oof_v23_abs + (1 - meta_w[:, None]) * oof_v16_abs
        print(f"meta_lgb mean w={meta_w_em.mean():.3f}")
    except Exception as e:
        print(f"meta_lgb skip: {e}")
        meta_w = None

    results_oof = {}
    print(f"  v23 single:   {rh_v23:.4f}")
    print(f"  v16 single:   {rh_v16:.4f}")
    print(f"  oracle:       {either:.4f}")
    for name, ens in candidates_oof.items():
        r = rhit(ens, y_train, eval_mask)
        results_oof[name] = r
        d = r - max(rh_v23, rh_v16)
        print(f"  {name:<40}: {r:.4f}  (Δ {d:+.4f})")

    best_name = max(results_oof, key=results_oof.get)
    best_rh = results_oof[best_name]
    print(f"\n★★ Best OOF: {best_name}  {best_rh:.4f}")

    # --- Apply to test ---
    dist12_test = np.linalg.norm(test_v23_abs - test_v16_abs, axis=-1)
    test_candidates = {
        "simple_avg": (test_v23_abs + test_v16_abs) / 2,
        f"weighted_w23={w23:.3f}": w23 * test_v23_abs + w16 * test_v16_abs,
    }
    test_close = dist12_test < THR
    test_best_single = test_v23_abs if rh_v23 >= rh_v16 else test_v16_abs
    test_candidates["boundary_aware_2cm"] = np.where(test_close[:, None],
                                                       (test_v23_abs + test_v16_abs) / 2, test_best_single)
    test_candidates[f"global_α={best_a:.2f}"] = best_a * test_v23_abs + (1 - best_a) * test_v16_abs
    test_candidates[f"per_axis_α={per_axis.round(2).tolist()}"] = (
        per_axis[None, :] * test_v23_abs + (1 - per_axis[None, :]) * test_v16_abs)

    if "meta_lgb" in candidates_oof:
        feat_test = np.concatenate([
            test_v23_abs, test_v16_abs, test_v23_abs - test_v16_abs, dist12_test[:, None],
            np.linalg.norm(test_v23_abs - kalman_test, axis=-1, keepdims=True),
            np.linalg.norm(test_v16_abs - kalman_test, axis=-1, keepdims=True),
        ], axis=1)
        gbm_full = lgb.LGBMRegressor(num_leaves=15, learning_rate=0.05, n_estimators=200,
                                       min_child_samples=20, verbose=-1, random_state=0)
        target = (d23 < d16).astype(np.float32)
        feat_full = np.concatenate([
            oof_v23_abs, oof_v16_abs, oof_v23_abs - oof_v16_abs, dist12[:, None],
            np.linalg.norm(oof_v23_abs - kalman_train, axis=-1, keepdims=True),
            np.linalg.norm(oof_v16_abs - kalman_train, axis=-1, keepdims=True),
        ], axis=1)
        gbm_full.fit(feat_full[eval_mask], target[eval_mask])
        w_test = np.clip(gbm_full.predict(feat_test), 0.0, 1.0)
        test_candidates["meta_lgb"] = w_test[:, None] * test_v23_abs + (1 - w_test[:, None]) * test_v16_abs

    chosen = None
    for k in test_candidates:
        if k == best_name or k.split("=")[0] == best_name.split("=")[0]:
            chosen = k; break
    if chosen is None:
        chosen = max(test_candidates.keys(), key=lambda k: results_oof.get(k, 0.0))
    test_final = test_candidates[chosen]
    print(f"★ chosen test ensemble: {chosen}")

    # --- Submission ---
    out_csv = DATA_DIR / f"submission_v24_cpu_{mode}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_final[:,0], "y": test_final[:,1], "z": test_final[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"\n[submission] {out_csv}")

    # --- run_log ---
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": f"v24_cpu_{mode}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v23 (Kalman+GRU) + v16 (stacking) ensemble",
        "v23_mode": mode,
        "v23_oof_rhit": rh_v23, "v16_oof_rhit": rh_v16,
        "oracle_either": float(either),
        "all_oof_candidates": {k: float(v) for k, v in results_oof.items()},
        "best_oof_name": best_name, "best_oof_rhit": float(best_rh),
        "chosen_test_ensemble": chosen,
        "covered_rows": int(eval_mask.sum()),
        "submission_path": str(out_csv),
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

    print("\n" + "=" * 60)
    print(f"v24 ensemble ({mode}) 완료")
    print("=" * 60)
    print(f"  v23 alone : {rh_v23:.4f}")
    print(f"  v16 alone : {rh_v16:.4f}")
    print(f"  ensemble  : {best_rh:.4f}  ({best_name})")
    print(f"  oracle    : {either:.4f}")
    print(f"  CSV       : {out_csv}")


if __name__ == "__main__":
    main()
