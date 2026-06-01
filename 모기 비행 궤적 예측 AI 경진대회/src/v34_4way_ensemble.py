"""v34_4way_ensemble.py — 4-way OOF ensemble (v30 + gate + v16 + v32 MDN).

v33에서 3-way OOF 0.6653 도달. v32 MDN (OOF 0.6445)이 약하지만 paradigm 다양성
(multi-modal, deterministic 아님)으로 ensemble에 추가 가치 가능.

확인:
  1. v32가 hit하지만 다른 3개가 miss하는 sample (only_v32)
  2. 4-way grid search
  3. v33 top1 (3way) baseline 대비 +Δ
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
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"


def rhit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())


def main():
    print("=" * 60)
    print("v34: 4-way ensemble (v30 + gate + v16 + v32 MDN)")
    print("=" * 60)

    # --- Load all 4 sources ---
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    # v30
    st30 = np.load(CACHE_DIR / "v30_state.npz")
    oof_v30 = kalman_train + (st30["oof_A"] + st30["oof_B"])/2 * ALPHA
    test_v30 = kalman_test + (st30["test_A"] + st30["test_B"])/2 * ALPHA

    # v32 MDN
    st32 = np.load(CACHE_DIR / "v32_mdn_state.npz")
    oof_v32 = kalman_train + st32["oof_weighted"] * ALPHA  # weighted mode best
    test_v32 = kalman_test + st32["test_weighted"] * ALPHA

    # gate
    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]
    gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    df_best = pd.read_csv(BEST_TEST)
    test_gate = df_best[["x","y","z"]].values.astype(np.float64)

    # v16
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64)
    test_v16 = st16["test"].astype(np.float64)

    print(f"\n[OOF] v30 {rhit(oof_v30, y_train):.4f}, "
          f"gate {rhit(gate_oof, y_train):.4f}, "
          f"v16 {rhit(oof_v16, y_train):.4f}, "
          f"v32 {rhit(oof_v32, y_train):.4f}")

    # --- 4-way hit pattern (V32 unique contribution) ---
    d_v30 = np.linalg.norm(oof_v30 - y_train, axis=-1)
    d_gate = np.linalg.norm(gate_oof - y_train, axis=-1)
    d_v16 = np.linalg.norm(oof_v16 - y_train, axis=-1)
    d_v32 = np.linalg.norm(oof_v32 - y_train, axis=-1)
    h_v30 = d_v30 <= 0.01; h_gate = d_gate <= 0.01; h_v16 = d_v16 <= 0.01; h_v32 = d_v32 <= 0.01

    either4 = (h_v30 | h_gate | h_v16 | h_v32).mean()
    either3 = (h_v30 | h_gate | h_v16).mean()
    only_v32 = (h_v32 & ~h_v30 & ~h_gate & ~h_v16).mean()
    print(f"\n=== 4-way oracle ===")
    print(f"  EITHER 3 (v30/gate/v16): {either3:.4f}")
    print(f"  EITHER 4 (+ v32 MDN):    {either4:.4f}")
    print(f"  v32 unique hit (other 3 miss): {only_v32:.4f}  ({int(only_v32*10000)} samples)")
    print(f"  oracle gain by adding v32: {either4 - either3:+.4f}")

    # --- 4-way grid search (coarse: 6 weights × 4 dim grid, sum to 1) ---
    print(f"\n=== 4-way grid search ===")
    base_oof = (oof_v30, gate_oof, oof_v16, oof_v32)
    base_test = (test_v30, test_gate, test_v16, test_v32)
    best_w = None; best_r = 0
    for a in np.linspace(0.2, 0.8, 13):
        for b in np.linspace(0.1, min(0.8, 1-a), 15):
            for c in np.linspace(0.0, min(0.3, 1-a-b), 7):
                d = 1 - a - b - c
                if d < 0 or d > 0.4: continue
                ens = a*oof_v30 + b*gate_oof + c*oof_v16 + d*oof_v32
                r = rhit(ens, y_train)
                if r > best_r:
                    best_r, best_w = r, (a, b, c, d)
    a, b, c, d = best_w
    ens_4way = a*oof_v30 + b*gate_oof + c*oof_v16 + d*oof_v32
    test_4way = a*test_v30 + b*test_gate + c*test_v16 + d*test_v32
    print(f"  best: v30={a:.2f}, gate={b:.2f}, v16={c:.2f}, v32={d:.2f} → OOF {best_r:.4f}")

    # Baseline 3-way (from v33)
    a3 = 0.60; b3 = 0.38; c3 = 0.02
    ens_3way = a3*oof_v30 + b3*gate_oof + c3*oof_v16
    test_3way = a3*test_v30 + b3*test_gate + c3*test_v16
    rh_3way = rhit(ens_3way, y_train)
    print(f"  v33 baseline 3way: OOF {rh_3way:.4f}")
    print(f"  4way Δ vs 3way: {best_r - rh_3way:+.4f}")

    # v32 단독 추가 benefit이 작거나 음수면, 3-way가 sufficient
    # blended: ens_3way + small v32 perturbation
    print(f"\n=== alpha blend (3way × (1-α) + v32 × α) ===")
    best_alpha = 0; best_blend = rh_3way
    for alpha in np.linspace(0.0, 0.3, 16):
        ens = (1 - alpha) * ens_3way + alpha * oof_v32
        r = rhit(ens, y_train)
        if r > best_blend:
            best_blend = r; best_alpha = alpha
    ens_blend = (1 - best_alpha) * ens_3way + best_alpha * oof_v32
    test_blend = (1 - best_alpha) * test_3way + best_alpha * test_v32
    print(f"  best α(v32) = {best_alpha:.2f} → OOF {best_blend:.4f}")

    # Boundary-aware on top of 3way: 3way miss area에 v32 substitute
    # 3way OOF distance > 1cm, v32 distance < 1cm → use v32
    d_3way = np.linalg.norm(ens_3way - y_train, axis=-1)
    miss_3way = d_3way > 0.01
    v32_better = (d_v32 < d_3way) & miss_3way  # v32 closer AND 3way miss
    print(f"  miss_3way: {miss_3way.sum()} samples, v32_better in miss: {v32_better.sum()}")
    ens_sample_select = ens_3way.copy()
    # OOF에서는 cheat (y_train 활용) — sample select 위해 features만 사용
    # 단순 휴리스틱: 3way가 Kalman과 매우 다르면 v32로 대체 (uncertainty proxy)
    dist_3way_kal = np.linalg.norm(ens_3way - kalman_train, axis=-1)
    high_uncert = dist_3way_kal > np.percentile(dist_3way_kal, 90)
    ens_substitute = np.where(high_uncert[:, None], oof_v32, ens_3way)
    rh_subst = rhit(ens_substitute, y_train)
    test_subst = np.where(
        (np.linalg.norm(test_3way - kalman_test, axis=-1) > np.percentile(dist_3way_kal, 90))[:, None],
        test_v32, test_3way
    )
    print(f"\n  3way + v32 sub (top-10% uncertain): OOF {rh_subst:.4f}")

    # --- Results summary ---
    results = {
        "v30 alone": rhit(oof_v30, y_train),
        "gate alone": rhit(gate_oof, y_train),
        "v32 weighted alone": rhit(oof_v32, y_train),
        f"v33 3way (v30={a3}/gate={b3}/v16={c3})": rh_3way,
        f"4way (v30={a:.2f}/gate={b:.2f}/v16={c:.2f}/v32={d:.2f})": best_r,
        f"blend 3way + v32 α={best_alpha:.2f}": best_blend,
        "3way + v32 substitute high-uncert": rh_subst,
    }
    print("\n=== All OOF ensembles ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k:<55s}: {v:.4f}")
    best_name = max(results, key=results.get)
    print(f"\n★★ Best: {best_name}  {results[best_name]:.4f}")

    # Top 3 csv
    test_map = {
        f"4way (v30={a:.2f}/gate={b:.2f}/v16={c:.2f}/v32={d:.2f})": test_4way,
        f"v33 3way (v30={a3}/gate={b3}/v16={c3})": test_3way,
        f"blend 3way + v32 α={best_alpha:.2f}": test_blend,
        "3way + v32 substitute high-uncert": test_subst,
        "v30 alone": test_v30,
        "gate alone": test_gate,
    }
    print("\n=== Top OOF → CSV ===")
    saved = []
    for i, (name, r) in enumerate(sorted(results.items(), key=lambda x: -x[1])[:5]):
        if name not in test_map: continue
        safe = (name.replace("=","").replace("/","_").replace(" ","").replace("(","").replace(")","")
                .replace(",","").replace("+","_")[:40])
        out_csv = DATA_DIR / f"submission_v34_top{i+1}_{safe}.csv"
        tp = test_map[name]
        pd.DataFrame({"id": sub["id"], "x": tp[:,0], "y": tp[:,1], "z": tp[:,2]}
                     ).to_csv(out_csv, index=False)
        saved.append((name, r, out_csv))
        print(f"  #{i+1} {name}: OOF {r:.4f} → {out_csv.name}")

    # run_log
    log_path = PROJECT_DIR / "run_log.json"
    entry = {
        "version": "v34_4way_ensemble",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "v30 + gate + v16 + v32(MDN) 4-way ensemble grid",
        "oracle_3way": float(either3), "oracle_4way": float(either4),
        "v32_unique_hit": float(only_v32),
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
