"""eda_post_v122c.py — v122c LB 0.6912 이후 EDA 심층 분석.

목적:
  1) v122c vs v112 OOF/test 차이 분석 (어디서 이겼나, 어디서 새 paradigm 효과?)
  2) 1-3cm boundary subset 정밀 분석 (boundary refinement 추가 잠재력)
  3) fast+turn 좁은 subset에서 v122c 성능 (NN 약점 부분의 paradigm 효과)
  4) train vs test 분포 차이 (covariate shift 여지)
  5) 추가 시도 카드의 잠재 lift 추정

산출: reports/eda_post_v122c.md + .json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
OPEN = PROJECT / "open"
CACHE = PROJECT / "cache"
REPORTS = PROJECT / "reports"
REPORTS.mkdir(exist_ok=True)

DT = 0.040
T_PRED = 0.080


def load_X():
    cache = CACHE / "xtrain_xtest.npz"
    if not cache.exists():
        raise FileNotFoundError(f"missing {cache}")
    nc = np.load(cache)
    return nc["X_train"], nc["X_test"]


def load_labels():
    df = pd.read_csv(OPEN / "train_labels.csv").sort_values("id").reset_index(drop=True)
    return df[["x","y","z"]].values.astype(np.float64)


def load_sub(path):
    df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    return df[["x","y","z"]].values.astype(np.float64)


def hit(pred, y):
    return (np.linalg.norm(pred - y, axis=1) < 0.01).mean()


def main():
    print("="*70); print("EDA post v122c LB 0.6912"); print("="*70)
    X_tr, X_te = load_X()
    y_tr = load_labels()
    N = X_tr.shape[0]
    Nte = X_te.shape[0]
    print(f"X_tr={X_tr.shape}, X_te={X_te.shape}, y_tr={y_tr.shape}")

    # last obs / velocity / accel 계산
    last_obs_tr = X_tr[:, -1]
    last_obs_te = X_te[:, -1]
    delta_tr = np.diff(X_tr, axis=1) / DT  # velocity per step (10 steps)
    delta_te = np.diff(X_te, axis=1) / DT
    accel_tr = np.diff(delta_tr, axis=1) / DT
    accel_te = np.diff(delta_te, axis=1) / DT

    speed_last_tr = np.linalg.norm(delta_tr[:, -1], axis=-1)
    speed_last_te = np.linalg.norm(delta_te[:, -1], axis=-1)
    accel_mean_tr = np.linalg.norm(accel_tr, axis=-1).mean(axis=1)
    accel_mean_te = np.linalg.norm(accel_te, axis=-1).mean(axis=1)
    turn_max_tr = np.zeros(N)
    turn_max_te = np.zeros(Nte)
    for arr_v, out in [(delta_tr, turn_max_tr), (delta_te, turn_max_te)]:
        # angle change between consecutive velocity vectors
        v_a = arr_v[:, :-1]
        v_b = arr_v[:, 1:]
        na = np.linalg.norm(v_a, axis=-1) + 1e-12
        nb = np.linalg.norm(v_b, axis=-1) + 1e-12
        cos = (v_a * v_b).sum(-1) / (na * nb)
        cos = np.clip(cos, -1, 1)
        ang = np.arccos(cos)  # (N, T-2)
        out[:] = ang.max(axis=1)

    print("\n=== Train vs Test 분포 비교 ===")
    for name, tr, te in [
        ("speed_last", speed_last_tr, speed_last_te),
        ("accel_mean", accel_mean_tr, accel_mean_te),
        ("turn_max",   turn_max_tr,  turn_max_te),
    ]:
        print(f"  {name:12s}: train mean={tr.mean():.4f} std={tr.std():.4f}  |  test mean={te.mean():.4f} std={te.std():.4f}  |  ks~{abs(np.median(tr)-np.median(te))/(tr.std()+1e-9):.3f}")

    # v122c와 v112 submission 로드
    print("\n=== Submission 파일 로드 ===")
    sub_v122c = OPEN / "submission_v122c_v121diverse_oof0.6769.csv"
    sub_v112 = OPEN / "submission_v112_v107_diverse_oof0.6768.csv"
    sub_v106 = OPEN / "submission_v106_DE15w_oof0.6770.csv"
    sub_v121c10 = OPEN / "submission_v121_cap10.csv"
    sub_v120 = OPEN / "submission_v120_full.csv"
    p_v122c = load_sub(sub_v122c)
    p_v112 = load_sub(sub_v112)
    p_v106 = load_sub(sub_v106)
    p_v121c10 = load_sub(sub_v121c10)
    p_v120 = load_sub(sub_v120)

    # baseline: last obs
    p_last = last_obs_te.copy()
    # constant velocity extrapolation
    v_te = (X_te[:, -1] - X_te[:, -2]) / DT
    p_const = last_obs_te + v_te * T_PRED

    # OOF cache for v122c: read it from cache
    print("\n=== v122c OOF에 대한 fold 마스크/예측 분석 ===")
    cache_v122c = CACHE / "v122c_v121diverse_weights.npz"
    cache_v112 = CACHE / "v112_v107_diverse_weights.npz"
    if cache_v122c.exists():
        c = np.load(cache_v122c, allow_pickle=True)
        print(f"  v122c cache keys: {list(c.files)}")
        if "oof" in c.files:
            oof_v122c = c["oof"]
            print(f"  v122c oof shape: {oof_v122c.shape} hit={hit(oof_v122c, y_tr):.4f}")
        elif "oof_pred" in c.files:
            oof_v122c = c["oof_pred"]
            print(f"  v122c oof shape: {oof_v122c.shape} hit={hit(oof_v122c, y_tr):.4f}")
    if cache_v112.exists():
        c = np.load(cache_v112, allow_pickle=True)
        print(f"  v112 cache keys: {list(c.files)}")

    # Test prediction 차이 분석 (OOF 없이 test side만)
    print("\n=== Test predictions: v122c vs v112 (paradigm diversity proxy) ===")
    d_v122c_v112 = np.linalg.norm(p_v122c - p_v112, axis=1) * 1000  # mm
    d_v112_v106 = np.linalg.norm(p_v112 - p_v106, axis=1) * 1000
    d_v120_v112 = np.linalg.norm(p_v120 - p_v112, axis=1) * 1000
    print(f"  |v122c - v112| (mm): mean={d_v122c_v112.mean():.3f} med={np.median(d_v122c_v112):.3f} q90={np.quantile(d_v122c_v112,0.9):.3f}")
    print(f"  |v112  - v106| (mm): mean={d_v112_v106.mean():.3f} med={np.median(d_v112_v106):.3f} q90={np.quantile(d_v112_v106,0.9):.3f}")
    print(f"  |v120  - v112| (mm): mean={d_v120_v112.mean():.3f} med={np.median(d_v120_v112):.3f} q90={np.quantile(d_v120_v112,0.9):.3f}")

    # train side: hit-rate by accel/turn bin for v112 oof (from cache if exists)
    print("\n=== Train OOF hit-rate by subset (v112 OOF reconstructed from saved sub if avail) ===")
    # v112 OOF cache load
    if cache_v112.exists():
        c = np.load(cache_v112, allow_pickle=True)
        oof_v112 = None
        for k in ["oof_pred", "oof", "oof_global"]:
            if k in c.files:
                oof_v112 = c[k]; break
        if oof_v112 is None:
            # try other keys
            for k in c.files:
                arr = c[k]
                if arr.shape == y_tr.shape:
                    oof_v112 = arr
                    print(f"    using key '{k}' as v112 OOF")
                    break
        if oof_v112 is not None:
            base_hit = hit(oof_v112, y_tr)
            print(f"  v112 OOF total hit: {base_hit:.4f}")

            # subset analysis
            for fname, ftr in [
                ("accel_mean", accel_mean_tr), ("turn_max", turn_max_tr),
                ("speed_last", speed_last_tr),
            ]:
                qs = np.quantile(ftr, [0.2,0.4,0.6,0.8])
                bins = np.digitize(ftr, qs)
                hits = []
                for b in range(5):
                    m = bins == b
                    if m.sum() > 0:
                        hits.append(hit(oof_v112[m], y_tr[m]))
                    else:
                        hits.append(np.nan)
                print(f"  {fname:12s} Q1-Q5 hit: {[f'{h:.3f}' for h in hits]}")

            # fast+turn high subset
            mask_hard = (speed_last_tr > np.quantile(speed_last_tr, 0.8)) & (turn_max_tr > np.quantile(turn_max_tr, 0.8))
            n_hard = mask_hard.sum()
            print(f"\n  Hard subset (speed top20 & turn top20): n={n_hard} hit={hit(oof_v112[mask_hard], y_tr[mask_hard]):.4f}")

            # 1-3cm boundary subset
            d_v112 = np.linalg.norm(oof_v112 - y_tr, axis=1)
            for lo, hi in [(0,0.005),(0.005,0.01),(0.01,0.015),(0.015,0.02),(0.02,0.03),(0.03,0.05),(0.05,10)]:
                m = (d_v112 >= lo) & (d_v112 < hi)
                print(f"  d {lo*100:.1f}-{hi*100:.1f}cm: n={m.sum():4d} ({m.mean()*100:5.2f}%)")

    print("\n=== 마지막: oracle bound (v112 + v120 + boundary refine) per-sample best 검증 ===")
    if cache_v112.exists():
        c12 = np.load(cache_v112, allow_pickle=True)
        oof_v112 = None
        for k in c12.files:
            a = c12[k]
            if a.shape == y_tr.shape:
                oof_v112 = a; break
        cache_v120 = CACHE / "v120_full_state.npz"
        if cache_v120.exists() and oof_v112 is not None:
            v120 = np.load(cache_v120, allow_pickle=True)
            print(f"  v120 cache keys: {list(v120.files)}")
            oof_v120 = None
            for k in ["oof_global", "oof_local", "oof"]:
                if k in v120.files:
                    arr = v120[k]
                    if arr.shape == y_tr.shape:
                        oof_v120 = arr; break
            if oof_v120 is not None:
                d_v112 = np.linalg.norm(oof_v112 - y_tr, axis=1)
                d_v120 = np.linalg.norm(oof_v120 - y_tr, axis=1)
                d_oracle = np.minimum(d_v112, d_v120)
                print(f"  v112 OOF hit: {(d_v112<0.01).mean():.4f}")
                print(f"  v120 OOF hit: {(d_v120<0.01).mean():.4f}")
                print(f"  oracle min(v112,v120) hit: {(d_oracle<0.01).mean():.4f}  ← per-sample selector 천장")
                # disjoint hits
                only_v112 = (d_v112<0.01) & (d_v120>=0.01)
                only_v120 = (d_v120<0.01) & (d_v112>=0.01)
                both = (d_v112<0.01) & (d_v120<0.01)
                neither = (d_v112>=0.01) & (d_v120>=0.01)
                print(f"  both hit: {both.mean():.4f}  only v112: {only_v112.mean():.4f}  only v120: {only_v120.mean():.4f}  neither: {neither.mean():.4f}")
                print(f"  → selector가 완벽하면 추가 lift = {(d_oracle<0.01).mean() - (d_v112<0.01).mean():.4f}")

if __name__ == "__main__":
    main()
