"""v63_paradigm_blend.py — v62 (CA paradigm) + v48 3-way grid blend.

목적:
  v48 3-way (OOF 0.6748) + v62 (OOF 0.5902, paradigm 다름) blend 시 lift 측정.
  v62 단독은 약하지만, sample-wise 다양성으로 ensemble lift 가능성 확인.

검증:
  - base: v48 3-way = 0.70*v48 + 0.12*v46 + 0.18*v35 (OOF 0.6748)
  - blend: (1-w) * base + w * v62  for w in [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
  - OOF R-Hit 기준 best w 선택, 테스트 CSV 생성
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data, yaw_angle, inverse_rotate_xy

PROJECT_DIR = SCRIPT_DIR.parent
CACHE = PROJECT_DIR / "cache"
DATA = PROJECT_DIR / "open"


def main():
    X_train, X_test, y_train, sub = load_data()
    DT = 0.040
    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)

    # ---- 기존 stacker pool (final positions) ----
    v48 = np.load(CACHE / "v48_state.npz"); v48_oof, v48_te = v48["oof_v48"], v48["test_v48"]
    v46 = np.load(CACHE / "v46_state.npz"); v46_oof, v46_te = v46["oof_v46"], v46["test_v46"]
    v35 = np.load(CACHE / "v35_state.npz"); v35_oof, v35_te = v35["oof_v35"], v35["test_v35"]

    rh = lambda p: float((np.linalg.norm(p - y_train, axis=-1) <= 0.01).mean())
    print(f"[base] v48 OOF: {rh(v48_oof):.4f}  v46 OOF: {rh(v46_oof):.4f}  v35 OOF: {rh(v35_oof):.4f}")

    # v48 3-way (memory: 0.70*v48 + 0.12*v46 + 0.18*v35)
    base_oof = 0.70 * v48_oof + 0.12 * v46_oof + 0.18 * v35_oof
    base_te  = 0.70 * v48_te  + 0.12 * v46_te  + 0.18 * v35_te
    print(f"[base 3-way] OOF: {rh(base_oof):.4f}  (memory 0.6748)")

    # ---- v62 final position (CA paradigm) ----
    v62 = np.load(CACHE / "v62_state.npz")
    oof_avg = (v62["oof_A"] + v62["oof_B"]) / 2
    test_avg = (v62["test_A"] + v62["test_B"]) / 2
    kalman_train = v62["kalman_train"]
    kalman_test = v62["kalman_test"]
    v62_oof_pos = kalman_train + inverse_rotate_xy(oof_avg, theta_train)
    v62_te_pos = kalman_test + inverse_rotate_xy(test_avg, theta_test)
    print(f"[v62] OOF: {rh(v62_oof_pos):.4f}  (CA paradigm)")

    # ---- grid blend ----
    print("\n--- blend grid: (1-w)*v48_3way + w*v62 ---")
    best_w, best_rh = 0.0, rh(base_oof)
    results = []
    for w in [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        blend_oof = (1 - w) * base_oof + w * v62_oof_pos
        r = rh(blend_oof)
        results.append((w, r))
        marker = " ★" if r > best_rh else ""
        print(f"  w={w:.2f}: OOF {r:.4f}  (Δ vs base {r - rh(base_oof):+.4f}){marker}")
        if r > best_rh:
            best_rh = r; best_w = w

    print(f"\n[best] w={best_w:.2f}  OOF={best_rh:.4f}  (vs base 3-way {rh(base_oof):.4f}, Δ {best_rh - rh(base_oof):+.4f})")

    # 제출 CSV 생성 (best blend)
    blend_te = (1 - best_w) * base_te + best_w * v62_te_pos
    out_csv = DATA / f"submission_v63_blend_w{int(best_w*100):02d}.csv"
    pd.DataFrame({"id": sub["id"], "x": blend_te[:,0], "y": blend_te[:,1], "z": blend_te[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"[submission] {out_csv}")

    # 추가: oracle pool diversity 확인
    pool_oof = np.stack([v48_oof, v46_oof, v35_oof, v62_oof_pos], axis=0)  # (4, N, 3)
    d = np.linalg.norm(pool_oof - y_train[None, :, :], axis=-1)  # (4, N)
    any_hit = (d <= 0.01).any(axis=0).mean()
    print(f"\n[oracle 4-model (v48+v46+v35+v62)] any-hit upper bound: {any_hit:.4f}")
    pool_no62 = np.stack([v48_oof, v46_oof, v35_oof], axis=0)
    d2 = np.linalg.norm(pool_no62 - y_train[None, :, :], axis=-1)
    any_hit2 = (d2 <= 0.01).any(axis=0).mean()
    print(f"[oracle 3-model (no v62)]                                : {any_hit2:.4f}  (v62 adds {any_hit - any_hit2:+.4f})")


if __name__ == "__main__":
    main()
