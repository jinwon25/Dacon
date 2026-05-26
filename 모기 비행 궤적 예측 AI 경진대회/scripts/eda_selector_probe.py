"""eda_selector_probe.py — only-v120-hit 221개와 only-v112-hit 379개 sample 특성 분석.

질문:
  - v120이 잡는 unique sample은 어떤 동역학 특성인가?
  - v112가 잡는 unique sample은 어떤 동역학 특성인가?
  - 메타특징으로 두 그룹을 구분 가능한가?  → conditional weight / selector 잠재력
  - v122c 안에서 v120 weight를 sample-conditional로 조정하면 어디까지 갈 수 있나?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT = Path(__file__).resolve().parent
PROJ = SCRIPT.parent
CACHE = PROJ / "cache"
OPEN = PROJ / "open"

DT = 0.040

def meta_feats(X):
    """returns dict of per-sample meta features."""
    N = X.shape[0]
    delta = np.diff(X, axis=1) / DT
    accel = np.diff(delta, axis=1) / DT
    jerk  = np.diff(accel, axis=1) / DT
    speed = np.linalg.norm(delta, axis=-1)
    accel_m = np.linalg.norm(accel, axis=-1).mean(axis=1)
    jerk_m  = np.linalg.norm(jerk,  axis=-1).mean(axis=1)
    speed_last = speed[:, -1]
    speed_max  = speed.max(axis=1)
    speed_mean = speed.mean(axis=1)
    # turn / kappa
    va, vb = delta[:, :-1], delta[:, 1:]
    na = np.linalg.norm(va, axis=-1) + 1e-12
    nb = np.linalg.norm(vb, axis=-1) + 1e-12
    cos = np.clip((va*vb).sum(-1)/(na*nb), -1, 1)
    ang = np.arccos(cos)
    turn_max = ang.max(axis=1)
    turn_mean= ang.mean(axis=1)
    # decel deceleration vs accel
    speed_diff = np.diff(speed, axis=1)  # (N, 9)
    decel_max = (-speed_diff).max(axis=1)  # max deceleration
    return dict(
        accel_m=accel_m, jerk_m=jerk_m,
        speed_last=speed_last, speed_max=speed_max, speed_mean=speed_mean,
        turn_max=turn_max, turn_mean=turn_mean,
        decel_max=decel_max,
    )


def main():
    X_tr = np.load(CACHE / "xtrain_xtest.npz")["X_train"]
    y_tr = pd.read_csv(OPEN / "train_labels.csv").sort_values("id")[["x","y","z"]].values
    c12 = np.load(CACHE / "v112_v107_diverse_weights.npz", allow_pickle=True)
    c20 = np.load(CACHE / "v120_full_state.npz", allow_pickle=True)
    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz", allow_pickle=True)
    oof_v112 = c12["oof_pred"]
    oof_v120 = c20["oof_global"]
    oof_v122c = c22["oof_pred"]

    d12 = np.linalg.norm(oof_v112 - y_tr, axis=1)
    d20 = np.linalg.norm(oof_v120 - y_tr, axis=1)
    d22 = np.linalg.norm(oof_v122c - y_tr, axis=1)
    hit12 = d12 < 0.01; hit20 = d20 < 0.01; hit22 = d22 < 0.01

    only_v112 = hit12 & ~hit20
    only_v120 = hit20 & ~hit12
    both = hit12 & hit20
    neither = ~hit12 & ~hit20

    print(f"v112 OOF hit:  {hit12.mean():.4f}")
    print(f"v120 OOF hit:  {hit20.mean():.4f}")
    print(f"v122c OOF hit: {hit22.mean():.4f}")
    print(f"both:    {both.sum():4d} ({both.mean():.4f})")
    print(f"only112: {only_v112.sum():4d} ({only_v112.mean():.4f})")
    print(f"only120: {only_v120.sum():4d} ({only_v120.mean():.4f})")
    print(f"neither: {neither.sum():4d} ({neither.mean():.4f})")
    print(f"oracle min: {(np.minimum(d12,d20)<0.01).mean():.4f}")
    print(f"v122c hit:  {hit22.mean():.4f}")

    # v122c가 oracle 대비 얼마나 잘 따라가는가
    oracle_hits = hit12 | hit20  # 둘 중 하나라도 hit
    v22c_captures_oracle = (hit22 & oracle_hits).mean() / max(oracle_hits.mean(),1e-9)
    print(f"\nv122c가 oracle hit 중에서 잡은 비율: {v22c_captures_oracle:.4f}")
    # neither subset에서 v122c가 어쩌다 잡는 sample
    v22c_in_neither = (hit22 & neither).sum()
    print(f"v122c가 neither(둘 다 miss)에서 잡은 sample: {v22c_in_neither}")

    # meta features
    m = meta_feats(X_tr)

    print("\n=== Subset별 메타특징 평균 (Z-score 대비 전체 평균) ===")
    fields = ["accel_m", "jerk_m", "speed_last", "speed_max", "turn_max", "turn_mean", "decel_max"]
    header = "subset           n     " + "  ".join(f"{f:>10s}" for f in fields)
    print(header)
    for name, mask in [("ALL", np.ones(len(X_tr), dtype=bool)), ("both", both), ("only_v112", only_v112), ("only_v120", only_v120), ("neither", neither)]:
        row = f"{name:14s}  {mask.sum():4d}  "
        for f in fields:
            v_all = m[f].mean(); v_subset = m[f][mask].mean()
            z = (v_subset - v_all) / (m[f].std() + 1e-9)
            row += f"  {v_subset:5.3f}({z:+.2f})"
        print(row)

    # 가장 식별력 있는 feature 정렬 (only_v120 vs only_v112 mean shift)
    print("\n=== only_v120 vs only_v112 식별력 (큰 차이 = paradigm 식별 가능) ===")
    diffs = []
    for f in fields:
        a = m[f][only_v120].mean()
        b = m[f][only_v112].mean()
        d = (a - b) / (m[f].std() + 1e-9)
        diffs.append((abs(d), f, d, a, b))
    diffs.sort(reverse=True)
    for ad, f, d, a, b in diffs:
        print(f"  {f:12s}  Δz={d:+.3f}  only_v120={a:.3f}  only_v112={b:.3f}")

    # 1cm boundary 내부 분석 (v122c 0.5-1.5cm subset의 v112/v120)
    print("\n=== v122c가 boundary miss (0.8-1.5cm)인 sample에서 selector 잠재 ===")
    bnd_mask = (d22 >= 0.008) & (d22 <= 0.015)
    print(f"  boundary count n={bnd_mask.sum()}")
    print(f"  boundary 안에서 v112 hit: {hit12[bnd_mask].mean():.4f}")
    print(f"  boundary 안에서 v120 hit: {hit20[bnd_mask].mean():.4f}")
    print(f"  boundary 안에서 oracle hit: {(hit12[bnd_mask]|hit20[bnd_mask]).mean():.4f}")

    # neither subset 안에서 v120 paradigm 추가 효과 측정
    print("\n=== neither subset 안에서 v122c 평균 distance ===")
    print(f"  v112 mean d (neither): {d12[neither].mean()*1000:.2f}mm")
    print(f"  v120 mean d (neither): {d20[neither].mean()*1000:.2f}mm")
    print(f"  v122c mean d (neither): {d22[neither].mean()*1000:.2f}mm")
    print(f"  per-sample min(v112,v120) (neither): {np.minimum(d12,d20)[neither].mean()*1000:.2f}mm")

    # selector simulation: per-sample 가중치 0.5/0.5로 했을 때 vs oracle 가중치
    print("\n=== Heuristic conditional weight 시뮬레이션 ===")
    # 메타특징으로 v120 weight 결정: high accel/turn ↘ v120 favor (hard subset)
    accel_z = (m["accel_m"] - m["accel_m"].mean()) / m["accel_m"].std()
    turn_z = (m["turn_max"] - m["turn_max"].mean()) / m["turn_max"].std()
    speed_z = (m["speed_last"] - m["speed_last"].mean()) / m["speed_last"].std()
    hard_score = (accel_z + turn_z + speed_z) / 3

    for w_v120 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        blend = (1-w_v120) * oof_v112 + w_v120 * oof_v120
        d = np.linalg.norm(blend - y_tr, axis=1)
        print(f"  static v120_w={w_v120:.1f}: hit={(d<0.01).mean():.4f}")

    print("\n  conditional weight (hard subset에서만 v120 weight 올리기):")
    for tau, wh, wl in [(0.5, 0.6, 0.3), (1.0, 0.6, 0.3), (0.5, 0.7, 0.25), (1.0, 0.7, 0.25), (0.5, 0.5, 0.3), (1.0, 0.5, 0.3), (1.5, 0.7, 0.3)]:
        mask_hard = hard_score > tau
        w = np.where(mask_hard, wh, wl)[:, None]
        blend = (1-w) * oof_v112 + w * oof_v120
        d = np.linalg.norm(blend - y_tr, axis=1)
        print(f"    hard_tau={tau:.1f} (n={mask_hard.sum()}) wh={wh} wl={wl}: hit={(d<0.01).mean():.4f}")

    # binary selector simulation: per-sample one-hot
    # 학습 가능한 형태: meta features → P(v120 더 정확) 분류기
    print("\n=== Oracle selector vs static blend ===")
    pick_v120 = d20 < d12  # 1 = v120 더 정확
    print(f"  pick_v120 비율: {pick_v120.mean():.4f}")
    pred_oracle = np.where(pick_v120[:,None], oof_v120, oof_v112)
    d_oracle = np.linalg.norm(pred_oracle - y_tr, axis=1)
    print(f"  oracle selector hit: {(d_oracle<0.01).mean():.4f}")
    # 만약 selector가 80% 정확도로 픽하면 (random 50% pick)
    rng = np.random.default_rng(0)
    for acc in [0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
        # acc 비율로 정답 픽
        mask_correct = rng.random(len(pick_v120)) < acc
        guess = np.where(mask_correct, pick_v120, ~pick_v120)
        pred_s = np.where(guess[:,None], oof_v120, oof_v112)
        d_s = np.linalg.norm(pred_s - y_tr, axis=1)
        print(f"  selector acc={acc:.2f}: hit={(d_s<0.01).mean():.4f}")

if __name__ == "__main__":
    main()
