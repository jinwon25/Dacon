"""v69_confidence_route.py — Confidence-thresholded routing on top of v48 3-way.

진단:
  - v68 selector accuracy 14-19% (random 9% 대비 미세). 약한 paradigm 모델(v62/v65)에 over-route.
  - hard routing (v68): 0.6374, top-2: 0.6402 — base 0.6748 대비 -0.04 (over-aggressive)
  - oracle 0.7506은 perfect selector 기준 매우 높음

설계:
  - v68 selector probability max(p) ≥ T → use selected model
  - else → fall back to v48 3-way base (safety net)
  - T grid search: [0.20, 0.25, ..., 0.95]
  - 또한 sel != base_argmax 케이스에만 적용 (boundary sample 가설)

이것도 안 통하면 → v70 multi-label BCE / weighted CE / per-anchor router.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data

CACHE = SCRIPT_DIR.parent / "data/cache"
DATA = SCRIPT_DIR.parent / "data"


def rh(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01)


def main():
    X_train, X_test, y_train, sub = load_data()

    # v68 outputs
    v68 = np.load(CACHE / "v68_state.npz", allow_pickle=True)
    sel_idx = v68["sel_idx_oof"]; prob = v68["prob_oof"]
    names = list(v68["names"])
    K = len(names)
    print(f"v68 K={K} names={names}")

    # reconstruct OOF pool same as v68
    nc = np.load(CACHE / "xtrain_xtest.npz"); kc = np.load(CACHE / "kalman.npz")
    kt = kc["kalman_train"]; ke = kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]
    st30 = np.load(CACHE / "v30_state.npz")
    st35 = np.load(CACHE / "v35_state.npz")
    st41 = np.load(CACHE / "v41_state.npz")
    st44 = np.load(CACHE / "v44_state.npz")
    st39 = np.load(CACHE / "v39_state.npz")
    BO_PATH = SCRIPT_DIR.parent / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
    BEST_TEST = SCRIPT_DIR.parent / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
    import glob, os
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    bo = np.load(BO_PATH, allow_pickle=True); gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(bo["ids"], train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in bo["ids"]]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)

    st65 = np.load(CACHE / "v65_K64_state.npz")
    st62 = np.load(CACHE / "v62_state.npz")
    from v23_train import yaw_angle, inverse_rotate_xy
    v_last_train = (nc["X_train"][:, -1] - nc["X_train"][:, -2]) / 0.040
    v_last_test  = (nc["X_test"][:, -1]  - nc["X_test"][:, -2])  / 0.040
    th_tr, th_te = yaw_angle(v_last_train), yaw_angle(v_last_test)
    v62o = st62["kalman_train"] + inverse_rotate_xy((st62["oof_A"] + st62["oof_B"])/2, th_tr)
    v62t = st62["kalman_test"] + inverse_rotate_xy((st62["test_A"] + st62["test_B"])/2, th_te)

    pool_oof = [
        kt + st30["oof_A"]*ALPHA, kt + st30["oof_B"]*ALPHA,
        st35["oof_v35"].astype(np.float64),
        kt + st41["oof_A"]*ALPHA, kt + st41["oof_B"]*ALPHA,
        st44["oof_v44"].astype(np.float64),
        gate_o, st39["oof_v39"].astype(np.float64),
        st65["oof_soft"].astype(np.float64), st65["oof_hard"].astype(np.float64),
        v62o,
    ]
    pool_te = [
        ke + st30["test_A"]*ALPHA, ke + st30["test_B"]*ALPHA,
        st35["test_v35"].astype(np.float64),
        ke + st41["test_A"]*ALPHA, ke + st41["test_B"]*ALPHA,
        st44["test_v44"].astype(np.float64),
        gate_t, st39["test_v39"].astype(np.float64),
        st65["test_soft"].astype(np.float64), st65["test_hard"].astype(np.float64),
        v62t,
    ]
    assert len(pool_oof) == K == len(pool_te)

    # base v48 3-way
    v48 = np.load(CACHE / "v48_state.npz"); v46 = np.load(CACHE / "v46_state.npz")
    base_o = 0.70*v48["oof_v48"] + 0.12*v46["oof_v46"] + 0.18*st35["oof_v35"]
    base_t = 0.70*v48["test_v48"] + 0.12*v46["test_v46"] + 0.18*st35["test_v35"]
    rh_base = rh(base_o, y_train).mean()
    print(f"\nbase v48 3-way OOF: {rh_base:.4f}")

    # selector picks predictions
    N = len(y_train)
    Nt = len(pool_te[0])
    sel_pred_o = np.stack([pool_oof[i] for i in range(K)], axis=1)[np.arange(N), sel_idx]
    sel_pred_t = np.stack([pool_te[i] for i in range(K)], axis=1)
    rh_sel_alone = rh(sel_pred_o, y_train).mean()
    print(f"v68 selector OOF (alone): {rh_sel_alone:.4f}")

    # need test selector indices — re-derive from prob? prob is OOF only.
    # Test selector not available without retraining. Use OOF prob mean per fold? Not stored.
    # Workaround: average test probs would need v68 to store test_prob. It only stores test_hard/test_top2.
    test_hard = v68["test_hard"]; test_top2 = v68["test_top2"]

    max_prob = prob.max(axis=-1)  # (N,)
    print(f"\n=== Confidence threshold sweep (use selector if max_prob ≥ T, else base) ===")
    best_T, best_r = None, rh_base
    for T in np.linspace(0.1, 0.95, 18):
        use_sel = max_prob >= T
        final = np.where(use_sel[:, None], sel_pred_o, base_o)
        r = rh(final, y_train).mean()
        n_sel = use_sel.sum()
        flag = " ★" if r > best_r else ""
        print(f"  T={T:.2f}: routed {n_sel:5d} ({n_sel/N*100:5.1f}%)  OOF={r:.4f}  (Δ {r - rh_base:+.4f}){flag}")
        if r > best_r:
            best_r, best_T = r, T

    if best_T is not None:
        print(f"\n  best T={best_T:.2f} → OOF {best_r:.4f}  (Δ vs base {best_r - rh_base:+.4f})")
        # apply to test
        # Note: test sel_idx 추정 불가 (v68에 test prob 저장 안 함). v68 test_hard 자체에 threshold 적용 가능
        # 대안: test_hard에 단순 hard prediction이 들어있음 (selector argmax) — 따라서 그대로 사용
        # max_prob 비교는 test에는 없음 → simple full apply 시 test_hard hybrid 안 됨
        # 빠른 우회: test_hard hybrid는 의미 없고, OOF의 best T 비율만큼 random 적용? 부정확.
        # 그러므로 v69는 OOF level diagnostic만 신뢰 (test 적용은 v70에서 test_prob도 저장)
        print(f"  (test 적용은 v70에서 test_prob 저장 후 재시도)")
    else:
        print(f"\n  threshold routing 효과 없음 (best_r={rh_base:.4f})")

    # 추가 진단: selector pick이 v48 3-way가 miss하는 sample만으로 한정
    base_miss = ~rh(base_o, y_train)
    print(f"\n=== base miss sample만 routing (n={base_miss.sum()}) ===")
    sel_in_miss = sel_pred_o.copy()
    final_miss_only = np.where(base_miss[:, None], sel_in_miss, base_o)
    rh_miss_only = rh(final_miss_only, y_train).mean()
    print(f"  unconditional miss-route: OOF {rh_miss_only:.4f}  (Δ {rh_miss_only - rh_base:+.4f})")
    # with T
    for T in np.linspace(0.1, 0.9, 17):
        use_sel = base_miss & (max_prob >= T)
        final = np.where(use_sel[:, None], sel_pred_o, base_o)
        r = rh(final, y_train).mean()
        n_sel = use_sel.sum()
        flag = " ★" if r > rh_base else ""
        if r > rh_base:
            print(f"  T={T:.2f}: routed {n_sel:5d}  OOF={r:.4f}  (Δ {r - rh_base:+.4f}){flag}")

    # ground truth oracle: base가 miss & v65/v62 중 hit 있는 sample 통계
    print(f"\n=== Routing potential (oracle) ===")
    pool_hits = np.stack([rh(p, y_train) for p in pool_oof])  # (K, N)
    base_hit = rh(base_o, y_train)
    # base miss이지만 v65s/v65h/v62 중 하나가 hit
    rescue_models = [names.index(n) for n in ["v65s", "v65h", "v62"]]
    rescue_hit_mask = pool_hits[rescue_models].any(axis=0)
    rescue_potential = (~base_hit) & rescue_hit_mask
    print(f"  base miss & (v65s|v65h|v62) hit: {rescue_potential.sum()} = {rescue_potential.mean():.4f}")
    print(f"  oracle upper bound (base hit OR rescue): {(base_hit | rescue_potential).mean():.4f}")


if __name__ == "__main__":
    main()
