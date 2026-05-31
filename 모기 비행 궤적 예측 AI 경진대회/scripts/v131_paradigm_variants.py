"""v131_paradigm_variants.py — genuinely decorrelated Neural ODE variants.

v120_big/n2는 같은 구조(yaw frame + MLP encoder) → 블렌드 기여 미미.
진짜 decorrelation은 inductive bias 변경에서. 두 축으로 변형:

  --frame yaw|frenet
     yaw   : v120 동일 (속도 xy-angle만 회전, z 별도)
     frenet: 속도(tangent)+가속도(normal)로 완전 3D 직교 프레임. z 처리가
             근본적으로 달라져 kalman/yaw framework와 최대 decorrelation.
  --encoder mlp|gru
     mlp : v120 동일 (flatten→Linear backbone)
     gru : 11-step 시퀀스를 GRU로 인코딩 → 다른 feature 추출 경로.

나머지(RK4 dynamics, loss, mirror+TTA, target=y-last_obs)는 v120과 동일하게 유지.
검증: OOF R-Hit ≥0.65 & L2(vs v120, vs v112) ~2mm 이면 pool 멤버 자격.

usage:
  python scripts/v131_paradigm_variants.py --frame frenet --encoder mlp --mode full --tag frenet
  python scripts/v131_paradigm_variants.py --frame yaw --encoder gru --mode full --tag gru
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (DT, T_PRED, load_data, get_kalman, get_scalar_feats,
                       build_seq, build_tier3, yaw_angle, normalize_seq)
# reuse proven model pieces
from v120_neural_ode import ResBlock, AccelField, NeuralODEModel, loss_combined

PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
OUT_DIR = PROJECT_DIR / "open"

MODE_CONFIGS = {
    "smoke": dict(n_folds=1, n_seeds=1, max_epochs=30, patience=8, batch=256, lr=2e-3, wd=1e-3, mirror=False),
    "fast":  dict(n_folds=2, n_seeds=1, max_epochs=60, patience=12, batch=256, lr=2e-3, wd=1e-3, mirror=True),
    "full":  dict(n_folds=5, n_seeds=2, max_epochs=80, patience=15, batch=256, lr=2e-3, wd=1e-3, mirror=True),
}

# ============================================================
# Coordinate frames: return per-sample rotation R (N,3,3) mapping
#   local = R @ global_vec.  inverse = R^T @ local.
# ============================================================
def yaw_frame(X):
    v = (X[:, -1] - X[:, -2]) / DT
    theta = yaw_angle(v)
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((X.shape[0], 3, 3))
    # x' = c*x + s*y ; y' = -s*x + c*y ; z'=z   (matches rotate_xy)
    R[:, 0, 0] = c;  R[:, 0, 1] = s
    R[:, 1, 0] = -s; R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R

def frenet_frame(X):
    """tangent=last vel, normal from last accel (Gram-Schmidt), binormal=t×n.
    degenerate(저속/직선) 샘플은 yaw frame으로 fallback."""
    N = X.shape[0]
    v = (X[:, -1] - X[:, -2]) / DT                       # (N,3)
    a = (X[:, -1] - 2 * X[:, -2] + X[:, -3]) / (DT * DT)  # (N,3)
    nv = np.linalg.norm(v, axis=1)
    t = v / np.clip(nv[:, None], 1e-9, None)
    # normal = a - (a·t)t
    a_perp = a - (np.sum(a * t, axis=1, keepdims=True)) * t
    na = np.linalg.norm(a_perp, axis=1)
    n = a_perp / np.clip(na[:, None], 1e-9, None)
    b = np.cross(t, n)
    R = np.stack([t, n, b], axis=1)  # rows = frame axes → R@g = local
    # fallback: degenerate where |v|<1e-4 or |a_perp|<1e-6 → use yaw frame
    Ry = yaw_frame(X)
    bad = (nv < 1e-4) | (na < 1e-6)
    R[bad] = Ry[bad]
    return R

def apply_R_seq(seq, R):
    """seq (N,T,9)=[rel,v,a] 각 3-block에 R 적용 (local = R@vec)."""
    out = np.empty_like(seq)
    for k in range(3):
        blk = seq[..., 3*k:3*k+3]                       # (N,T,3)
        out[..., 3*k:3*k+3] = np.einsum("nij,ntj->nti", R, blk)
    return out.astype(np.float32)

def apply_R_vec(vec, R):       # (N,3) global→local
    return np.einsum("nij,nj->ni", R, vec).astype(np.float32)

def inv_R_vec(vec, R):         # (N,3) local→global  (R^T)
    return np.einsum("nji,nj->ni", R, vec)

def mirror_seq(seq, axis=1):
    """물리적 좌우반사를 local 좌표로 변환한 mirror.
    yaw frame: world-y negate -> local axis=1 (+ vel/accel blocks).
    frenet frame: 좌우반사 M=diag(1,-1,1)이 Frenet-local에서 binormal(z)만 negate
                  -> axis=2. (유도: local=(t·g, n·g, b·g), 반사 후 b_new=-Mb -> z성분 부호반전)"""
    out = seq.copy()
    for blk in range(3):
        out[..., 3 * blk + axis] *= -1
    return out

def mirror_vec(v, axis=1):
    out = v.copy(); out[..., axis] *= -1; return out

# ============================================================
# GRU-encoder ODE model (frame-agnostic; works on (N,T,C) seq)
# ============================================================
class GRUODEModel(nn.Module):
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64,
                 dt_pred=0.080, n_steps=1):
        super().__init__()
        self.dt_pred = dt_pred; self.n_steps = n_steps
        self.gru = nn.GRU(seq_channels, latent_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.scal_proj = nn.Sequential(nn.Linear(scal_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2*latent_dim, latent_dim), nn.LayerNorm(latent_dim),
                                  nn.GELU(), ResBlock(latent_dim))
        self.accel_field = AccelField(latent_dim=latent_dim, hidden=hidden)
        self.learned_damping = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self.local_bias = nn.Parameter(torch.zeros(3))
        self._last_accels = []

    def _ode_deriv(self, pos, vel, latent, speed):
        a = self.accel_field(pos, vel, latent, speed)
        return vel, -self.learned_damping * vel + a, a

    def _rk4(self, pos, vel, latent, speed, dt):
        dp1, dv1, a1 = self._ode_deriv(pos, vel, latent, speed)
        dp2, dv2, a2 = self._ode_deriv(pos + .5*dt*dp1, vel + .5*dt*dv1, latent, speed)
        dp3, dv3, a3 = self._ode_deriv(pos + .5*dt*dp2, vel + .5*dt*dv2, latent, speed)
        dp4, dv4, a4 = self._ode_deriv(pos + dt*dp3, vel + dt*dv3, latent, speed)
        np_ = pos + (dt/6)*(dp1+2*dp2+2*dp3+dp4)
        nv_ = vel + (dt/6)*(dv1+2*dv2+2*dv3+dv4)
        return np_, nv_, [a1, a2, a3, a4]

    def forward(self, seq, scal, init_vel, speed):
        # seq: (B,T,C)
        h = self.gru(seq)[0][:, -1]                     # last hidden (B,latent)
        latent = self.fuse(torch.cat([h, self.scal_proj(scal)], dim=-1))
        pos = torch.zeros_like(init_vel); vel = init_vel
        dt = self.dt_pred / self.n_steps; accels = []
        for _ in range(self.n_steps):
            pos, vel, ac = self._rk4(pos, vel, latent, speed, dt); accels.extend(ac)
        self._last_accels = accels
        return pos + self.local_bias


def run_kfold(X_train, X_test, y_train, R_train, R_test, X_scal_tr, X_scal_te,
              cfg, encoder="mlp", mirror_axis=1, device="cpu"):
    n_folds, n_seeds = cfg["n_folds"], cfg["n_seeds"]
    max_epochs, patience = cfg["max_epochs"], cfg["patience"]
    batch, lr, wd, mirror_on = cfg["batch"], cfg["lr"], cfg["wd"], cfg["mirror"]
    N = X_train.shape[0]

    seq_tr = apply_R_seq(build_seq(X_train), R_train)
    seq_te = apply_R_seq(build_seq(X_test), R_test)
    tier3_tr, tier3_te = build_tier3(X_train), build_tier3(X_test)

    init_vel_tr = seq_tr[:, -1, 3:6].astype(np.float32)
    init_vel_te = seq_te[:, -1, 3:6].astype(np.float32)
    speed_tr = np.linalg.norm(init_vel_tr, axis=-1).astype(np.float32)
    speed_te = np.linalg.norm(init_vel_te, axis=-1).astype(np.float32)

    # target: (y - last_obs) in local frame
    target_local = apply_R_vec(y_train - X_train[:, -1], R_train)

    scal_tr_full = np.concatenate([X_scal_tr, tier3_tr], axis=-1).astype(np.float32)
    scal_te_full = np.concatenate([X_scal_te, tier3_te], axis=-1).astype(np.float32)
    scal_dim = scal_tr_full.shape[1]
    C = seq_tr.shape[2]; seq_flat_dim = seq_tr.shape[1] * C
    print(f"[v131:{encoder}] N={N} seqC={C} scal_dim={scal_dim}")

    if mirror_on:
        seq_tr_m = mirror_seq(seq_tr, mirror_axis)
        init_vel_tr_m = seq_tr_m[:, -1, 3:6].astype(np.float32)
        target_local_m = mirror_vec(target_local, mirror_axis)

    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))[:n_folds]

    oof_local = np.zeros((N, 3), dtype=np.float32)
    fold_mask = np.zeros(N, dtype=bool)
    test_per_fold, fold_rh_list = [], []
    t0 = time.time()

    def make_model():
        if encoder == "gru":
            return GRUODEModel(seq_channels=C, scal_dim=scal_dim,
                               latent_dim=cfg.get("latent_dim", 64), hidden=cfg.get("hidden", 64),
                               n_steps=cfg.get("n_steps", 1))
        return NeuralODEModel(seq_dim=seq_flat_dim, scal_dim=scal_dim,
                              latent_dim=cfg.get("latent_dim", 64), hidden=cfg.get("hidden", 64),
                              n_steps=cfg.get("n_steps", 1))

    def prep_seq(s):   # mlp flat vs gru seq
        return s.reshape(s.shape[0], -1) if encoder == "mlp" else s

    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_tr[tr].reshape(-1, C))
        sc_scal = StandardScaler().fit(scal_tr_full[tr])
        seq_n = normalize_seq(seq_tr, sc_seq); seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_n = sc_scal.transform(scal_tr_full).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te_full).astype(np.float32)
        if mirror_on: seq_m_n = normalize_seq(seq_tr_m, sc_seq)

        def T(a): return torch.from_numpy(np.ascontiguousarray(a)).to(device)
        if mirror_on:
            seq_in = np.concatenate([prep_seq(seq_n)[tr], prep_seq(seq_m_n)[tr]], 0)
            scal_in = np.concatenate([scal_n[tr], scal_n[tr]], 0)
            vel_in = np.concatenate([init_vel_tr[tr], init_vel_tr_m[tr]], 0)
            sp_in = np.concatenate([speed_tr[tr], speed_tr[tr]], 0)
            tgt_in = np.concatenate([target_local[tr], target_local_m[tr]], 0)
        else:
            seq_in, scal_in, vel_in, sp_in, tgt_in = prep_seq(seq_n)[tr], scal_n[tr], init_vel_tr[tr], speed_tr[tr], target_local[tr]
        seq_tr_t, scal_tr_t, vel_tr_t, sp_tr_t, tgt_tr_t = T(seq_in), T(scal_in), T(vel_in), T(sp_in), T(tgt_in)
        seq_va_t, scal_va_t = T(prep_seq(seq_n)[va]), T(scal_n[va])
        vel_va_t, sp_va_t = T(init_vel_tr[va]), T(speed_tr[va])
        seq_te_t, scal_te_t = T(prep_seq(seq_te_n)), T(scal_te_n)
        vel_te_t, sp_te_t = T(init_vel_te), T(speed_te)

        test_fold = np.zeros((seq_te.shape[0], 3), dtype=np.float32)
        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = make_model().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh, best_state, best_ep, bad = -1.0, None, 0, 0
            n_tr = seq_tr_t.shape[0]
            for ep in range(1, max_epochs + 1):
                model.train(); perm = torch.randperm(n_tr); el = 0; nb = 0
                for s in range(0, n_tr, batch):
                    idx = perm[s:s+batch]
                    pred = model(seq_tr_t[idx], scal_tr_t[idx], vel_tr_t[idx], sp_tr_t[idx])
                    loss, _, _ = loss_combined(pred, tgt_tr_t[idx], model._last_accels, 100.0, 1.0, 1e-4)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    el += loss.item(); nb += 1
                sch.step()
                model.eval()
                with torch.no_grad():
                    pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                pv_g = X_train[va, -1] + inv_R_vec(pv, R_train[va])
                rh = float((np.linalg.norm(pv_g - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh, best_ep, bad = rh, ep, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else: bad += 1
                if ep <= 3 or ep % 10 == 0 or bad >= patience:
                    print(f"  f{fi}s{seed} ep{ep:3d} loss={el/nb:.3f} vaRH={rh:.4f} best={best_rh:.4f}@{best_ep}")
                if bad >= patience:
                    print(f"  -> early stop ep{ep}"); break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                pte = model(seq_te_t, scal_te_t, vel_te_t, sp_te_t).cpu().numpy()
                if mirror_on:
                    seq_va_m = prep_seq(seq_m_n)[va]; vel_va_m = init_vel_tr_m[va]
                    pv_m = model(T(seq_va_m), scal_va_t, T(vel_va_m), sp_va_t).cpu().numpy()
                    pv = 0.5 * (pv + mirror_vec(pv_m, mirror_axis))
                    # TEST mirror TTA: mirror RAW seq -> raw vel -> normalize (v120 순서; 정규화 seq를
                    # mirror하면 velocity 입력이 정규화 스케일이 되어 ODE가 망가짐 — test 예측 손상 버그였음)
                    seq_te_m_raw = mirror_seq(seq_te, mirror_axis)
                    vel_te_m = seq_te_m_raw[:, -1, 3:6].astype(np.float32)
                    seq_te_m_n = normalize_seq(seq_te_m_raw, sc_seq)
                    pte_m = model(T(prep_seq(seq_te_m_n)), scal_te_t, T(vel_te_m), sp_te_t).cpu().numpy()
                    pte = 0.5 * (pte + mirror_vec(pte_m, mirror_axis))
            oof_local[va] += pv / n_seeds; test_fold += pte / n_seeds
        fold_mask[va] = True; test_per_fold.append(test_fold); fold_rh_list.append(best_rh)
        print(f"[v131] fold{fi} RH={best_rh:.4f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)

    oof_g = X_train[fold_mask, -1] + inv_R_vec(oof_local[fold_mask], R_train[fold_mask])
    rh_oof = float((np.linalg.norm(oof_g - y_train[fold_mask], axis=-1) <= 0.01).mean())
    print(f"[v131] OOF R-Hit = {rh_oof:.4f} (covered {fold_mask.sum()}/{N})")
    test_local = np.mean(test_per_fold, axis=0)
    test_global = X_test[:, -1] + inv_R_vec(test_local, R_test)
    oof_global_full = np.zeros((N, 3), dtype=np.float32)
    oof_global_full[fold_mask] = oof_g
    return oof_global_full, fold_mask, test_global, rh_oof, fold_rh_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODE_CONFIGS.keys()))
    ap.add_argument("--frame", default="frenet", choices=["yaw", "frenet"])
    ap.add_argument("--encoder", default="mlp", choices=["mlp", "gru"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_steps", type=int, default=1)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_seeds", type=int, default=0, help=">0이면 mode 기본 seed 수를 override (seed 앙상블 강화)")
    args = ap.parse_args()
    cfg = {**MODE_CONFIGS[args.mode], "n_steps": args.n_steps,
           "latent_dim": args.latent_dim, "hidden": args.hidden}
    if args.n_seeds > 0:
        cfg["n_seeds"] = args.n_seeds
    tag = args.tag or f"{args.frame}_{args.encoder}"
    state_file = CACHE_DIR / f"v131_{tag}_state.npz"

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu"); torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v131 frame={args.frame} encoder={args.encoder} mode={args.mode} cfg={cfg}")
    print(f"torch={torch.__version__} threads={torch.get_num_threads()}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    M_noise = {"loo_sample": 2000}
    X_scal_tr, X_scal_te = get_scalar_feats(X_train, X_test, M_noise, "fast")

    frame_fn = frenet_frame if args.frame == "frenet" else yaw_frame
    R_train, R_test = frame_fn(X_train), frame_fn(X_test)
    # sanity: orthonormal
    err = np.abs(np.einsum("nij,nkj->nik", R_train, R_train) - np.eye(3)).max()
    print(f"[frame] {args.frame} orthonormality max-err={err:.2e}")

    # frame-aware mirror axis: yaw->world-y(1), frenet->binormal-z(2)
    mirror_axis = 2 if args.frame == "frenet" else 1
    print(f"[mirror] axis={mirror_axis} ({'binormal-z' if mirror_axis==2 else 'lateral-y'})")

    t0 = time.time()
    oof_g, fold_mask, test_global, rh_oof, fold_rh = run_kfold(
        X_train, X_test, y_train, R_train, R_test, X_scal_tr, X_scal_te,
        cfg, encoder=args.encoder, mirror_axis=mirror_axis, device=device)
    print(f"[v131] total {(time.time()-t0)/60:.1f}m")

    np.savez(state_file, oof_global=oof_g, fold_mask=fold_mask, test_global=test_global,
             rh_oof=rh_oof, fold_rh=np.array(fold_rh))
    print(f"[v131] saved {state_file.name}")

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub[["x", "y", "z"]] = test_global
    sub.to_csv(OUT_DIR / f"submission_v131_{tag}.csv", index=False)

    # decorrelation vs v120 / v112
    for ref, rn in [("v120_full_state.npz", "v120"), ("v122c_v121diverse_weights.npz", "v122c")]:
        p = CACHE_DIR / ref
        if p.exists():
            st = np.load(p)
            t = st["test_global"] if "test_global" in st.files else st["test_pred"]
            L2 = float(np.linalg.norm(test_global - t, axis=-1).mean()) * 1000
            print(f"[decorr] L2(test vs {rn}) = {L2:.2f}mm")

    print("[v131] done.")


if __name__ == "__main__":
    main()
