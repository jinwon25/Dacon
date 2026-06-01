"""v134_imm.py — Multiple-Model (CV/CA/CT) 필터 — turn-decorrelated analytic base.

전 pool이 단일 constant-velocity Kalman base (corr~0.99 floor). CT(constant-turn)
모델은 CV가 구조적으로 표현 못 하는 곡선 운동을 예측 → turn subset에서 decorrelated.

per-window Multiple-Model: 각 모델 독립 KF로 11점 필터링 → +80ms(2 step) 예측.
모델별 fit cost(정규화 innovation²)로 softmax 가중 결합. (state mixing 없음 → dim 혼합 안전)
순수 numpy, 전 샘플 벡터화. OOF 계산용 라벨 = train_labels.

usage: python scripts/v134_imm.py
"""
from __future__ import annotations
import sys, glob, os, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
DATA = Path("data"); CACHE = Path("data/cache"); DT = 0.040

def load_xyz():
    nc = np.load(CACHE / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]  # (N,11,3)
    labels = pd.read_csv(DATA / "train_labels.csv")
    tr = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in tr])
    y = labels.set_index("id").loc[list(ids)][["x", "y", "z"]].values.astype(np.float64)
    return X_train, X_test, y

def bmm(A, B):  # batched (N,a,b)@(N,b,c)
    return np.einsum("nij,njk->nik", A, B)
def bmv(A, v):  # (N,a,b)@(N,b)
    return np.einsum("nij,nj->ni", A, v)

def kf_run(X, F, H, Q, R, x0, P0):
    """벡터화 KF. X(N,T,3) obs. F(N,d,d) or (d,d). returns final state/cov + fit_cost.
    fit_cost = mean over steps of normalized innovation^2 (model fit quality)."""
    N, T, _ = X.shape
    d = x0.shape[1]
    if F.ndim == 2: F = np.broadcast_to(F, (N, d, d))
    x = x0.copy(); P = P0.copy()
    Ht = H.transpose(0, 2, 1) if H.ndim == 3 else np.broadcast_to(H.T, (N, d, H.shape[0]))
    Hb = H if H.ndim == 3 else np.broadcast_to(H, (N, H.shape[0], d))
    Rb = np.broadcast_to(R, (N, 3, 3))
    Qb = Q if Q.ndim == 3 else np.broadcast_to(Q, (N, d, d))
    cost = np.zeros(N); nupd = 0
    eye = np.broadcast_to(np.eye(d), (N, d, d))
    for t in range(T):
        # predict (skip predict on first obs; just init-update)
        if t > 0:
            x = bmv(F, x)
            P = bmm(bmm(F, P), F.transpose(0, 2, 1)) + Qb
        # update with obs X[:,t]
        z = X[:, t, :]
        yk = z - bmv(Hb, x)                      # innovation (N,3)
        S = bmm(bmm(Hb, P), Ht) + Rb             # (N,3,3)
        Sinv = np.linalg.inv(S)
        K = bmm(bmm(P, Ht), Sinv)                # (N,d,3)
        x = x + bmv(K, yk)
        P = bmm(eye - bmm(K, Hb), P)
        if t > 0:
            cost += np.einsum("ni,nij,nj->n", yk, Sinv, yk)
            nupd += 1
    cost /= max(nupd, 1)
    return x, P, cost

def predict_steps(x, F, n=2):
    if F.ndim == 2:
        for _ in range(n): x = x @ F.T
    else:
        for _ in range(n): x = bmv(F, x)
    return x

def build_models(X):
    N = X.shape[0]
    # initial velocity/accel from first points
    v0 = (X[:, 1] - X[:, 0]) / DT
    # --- CV: state [p(3),v(3)] ---
    F_cv = np.eye(6); F_cv[0, 3] = F_cv[1, 4] = F_cv[2, 5] = DT
    H6 = np.zeros((3, 6)); H6[0, 0] = H6[1, 1] = H6[2, 2] = 1.0
    x0_cv = np.zeros((N, 6)); x0_cv[:, :3] = X[:, 0]; x0_cv[:, 3:] = v0
    # --- CA: state [p,v,a] (9) ---
    F_ca = np.eye(9)
    for i in range(3):
        F_ca[i, 3 + i] = DT; F_ca[i, 6 + i] = 0.5 * DT * DT; F_ca[3 + i, 6 + i] = DT
    H9 = np.zeros((3, 9)); H9[0, 0] = H9[1, 1] = H9[2, 2] = 1.0
    a0 = (X[:, 2] - 2 * X[:, 1] + X[:, 0]) / (DT * DT)
    x0_ca = np.zeros((N, 9)); x0_ca[:, :3] = X[:, 0]; x0_ca[:, 3:6] = v0; x0_ca[:, 6:] = a0
    # --- CT: coordinated turn in xy with per-sample omega, z = CV ---
    # estimate omega from heading change of xy-velocity over the window
    v = np.diff(X, axis=1) / DT                  # (N,10,3)
    phi = np.arctan2(v[:, :, 1], v[:, :, 0])     # (N,10)
    dphi = np.diff(np.unwrap(phi, axis=1), axis=1)  # (N,9)
    omega = np.clip(np.median(dphi, axis=1) / DT, -15, 15)  # rad/s, robust
    c = np.cos(omega * DT); s = np.sin(omega * DT)
    F_ct = np.broadcast_to(np.eye(6), (N, 6, 6)).copy()
    # position update via velocity over dt with rotated velocity (approx CT discretization)
    # standard CT: p += (1/w)*... ; use rotation of velocity + average displacement
    for n in range(N):
        w = omega[n]
        if abs(w) < 1e-3:
            F_ct[n, 0, 3] = DT; F_ct[n, 1, 4] = DT; F_ct[n, 2, 5] = DT
        else:
            sw, cw = s[n], c[n]
            F_ct[n, 0, 3] = sw / w; F_ct[n, 0, 4] = -(1 - cw) / w
            F_ct[n, 1, 3] = (1 - cw) / w; F_ct[n, 1, 4] = sw / w
            F_ct[n, 3, 3] = cw; F_ct[n, 3, 4] = -sw
            F_ct[n, 4, 3] = sw; F_ct[n, 4, 4] = cw
            F_ct[n, 2, 5] = DT
    return [
        ("CV", F_cv, H6, x0_cv, 6),
        ("CA", F_ca, H9, x0_ca, 9),
        ("CT", F_ct, H6, x0_cv.copy(), 6),
    ]

def run_imm(X, q_cv, q_ca, q_ct, r_obs, beta):
    N = X.shape[0]
    R = np.eye(3) * (r_obs ** 2)
    preds = []; costs = []
    for name, F, H, x0, d in build_models(X):
        q = {"CV": q_cv, "CA": q_ca, "CT": q_ct}[name]
        Q = np.eye(d) * q
        if d == 6: Q[3:, 3:] *= 10.0      # more uncertainty on velocity
        if d == 9: Q[6:, 6:] *= 10.0      # accel
        P0 = np.eye(d) * 1.0
        P0b = np.broadcast_to(P0, (N, d, d)).copy()
        xf, Pf, cost = kf_run(X, F, H, Q, R, x0, P0b)
        xp = predict_steps(xf, F, n=2)    # +80ms
        preds.append(xp[:, :3]); costs.append(cost)
    P = np.stack(preds, axis=1)           # (N,3models,3)
    C = np.stack(costs, axis=1)           # (N,3)
    w = np.exp(-beta * (C - C.min(1, keepdims=True)))
    w = w / w.sum(1, keepdims=True)       # (N,3)
    pred = (w[:, :, None] * P).sum(1)
    return pred, w, P

def hit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())

def main():
    X_train, X_test, y = load_xyz()
    print(f"loaded train {X_train.shape} test {X_test.shape}")
    # baseline CV kalman cache for reference
    kc = np.load(CACHE / "kalman.npz"); rh_kal = hit(kc["kalman_train"], y)
    print(f"[ref] cache CV-kalman OOF={rh_kal:.4f}")

    # light grid on process/obs noise + beta (OOF). 약과적합 위험 작음(few hyperparams).
    best = (None, -1, None)
    for r_obs in [0.0008, 0.0015, 0.003]:
        for q in [1e-4, 1e-3, 1e-2]:
            for beta in [0.5, 1.0, 2.0]:
                pred, w, _ = run_imm(X_train, q_cv=q, q_ca=q*5, q_ct=q*5, r_obs=r_obs, beta=beta)
                rh = hit(pred, y)
                if rh > best[1]: best = ((r_obs, q, beta), rh, w)
    (r_obs, q, beta), rh, w = best
    print(f"[best] r_obs={r_obs} q={q} beta={beta}  OOF={rh:.4f}  (model-weight mean CV/CA/CT={w.mean(0)})")

    pred_tr, w_tr, _ = run_imm(X_train, q, q*5, q*5, r_obs, beta)
    pred_te, w_te, _ = run_imm(X_test, q, q*5, q*5, r_obs, beta)
    rh = hit(pred_tr, y)
    # decorrelation vs v120, v122c, kalman
    for f, k, nm in [("v120_full_state.npz", "test_global", "v120"),
                     ("v122c_v121diverse_weights.npz", "test_pred", "v122c")]:
        st = np.load(CACHE / f); t = st[k]
        print(f"  decorr L2(test vs {nm})={np.linalg.norm(pred_te-t,axis=-1).mean()*1000:.2f}mm")
    print(f"  decorr L2(test vs kalman)={np.linalg.norm(pred_te-kc['kalman_test'],axis=-1).mean()*1000:.2f}mm")

    # save as pool member (v120-compatible schema: oof_global/test_global/fold_mask)
    np.savez(CACHE / "v134_imm_state.npz",
             oof_global=pred_tr.astype(np.float64), test_global=pred_te.astype(np.float64),
             fold_mask=np.ones(len(y), dtype=bool), rh_oof=rh)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub[["x", "y", "z"]] = pred_te
    sub.to_csv(DATA / "submission_v134_imm.csv", index=False)
    print(f"[v134] OOF={rh:.4f}  saved v134_imm_state.npz")

if __name__ == "__main__":
    main()
