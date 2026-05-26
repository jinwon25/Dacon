"""v120_neural_ode.py — Neural ODE backbone (RK4 dynamics integration).

LB 0.6+ Neural ODE 게시글 (CREE 2026-05-25, dacon codeshare 14002)을 참고한
별도 paradigm. v112 pool 멤버는 모두 kalman residual 공유 → corr~0.99 floor.
v120은 kalman 미사용, position-velocity 6D 상태계에 가속도 필드 + 학습 댐핑 + RK4
적분으로 80ms 예측 → residual basis 자체가 다름. (corr<0.93 가능성 목표)

핵심 차이:
  - 기존 pool: target = rotate_xy(y - kalman, theta) — kalman residual
  - v120:      target = rotate_xy(y - X[:,-1], theta) — last-obs residual
  - 구조: continuous dynamics integration via Runge-Kutta 4th order

사용:
  python scripts/v120_neural_ode.py --mode smoke   # fold0 1-seed 30ep
  python scripts/v120_neural_ode.py --mode fast    # 2-fold 1-seed 60ep
  python scripts/v120_neural_ode.py --mode full    # 5-fold 2-seed 80ep
"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    DT, T_PRED, load_data, get_kalman, get_scalar_feats,
    build_seq, build_tier3,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
)

PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"
OUT_DIR = PROJECT_DIR / "open"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Config
# ============================================================
MODE_CONFIGS = {
    "smoke": dict(n_folds=1, n_seeds=1, max_epochs=30, patience=8,
                  batch=256, lr=2e-3, wd=1e-3, mirror=False),
    "fast":  dict(n_folds=2, n_seeds=1, max_epochs=60, patience=12,
                  batch=256, lr=2e-3, wd=1e-3, mirror=True),
    "full":  dict(n_folds=5, n_seeds=2, max_epochs=80, patience=15,
                  batch=256, lr=2e-3, wd=1e-3, mirror=True),
}


# ============================================================
# Yaw rotation (vectorized matrix form, used in-graph)
# ============================================================
def rotate_xy_seq(seq_xyz: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """rotate (N, T, 3) by theta (N,) around z. y_local = -x*sin + y*cos."""
    c = np.cos(theta)[:, None]; s = np.sin(theta)[:, None]
    x_new = seq_xyz[..., 0] * c + seq_xyz[..., 1] * s
    y_new = -seq_xyz[..., 0] * s + seq_xyz[..., 1] * c
    return np.stack([x_new, y_new, seq_xyz[..., 2]], axis=-1).astype(np.float32)


# ============================================================
# Mirror aug (y-flip) — v90/v104 패턴
# ============================================================
def mirror_seq(seq):
    out = seq.copy()
    out[..., 1] *= -1; out[..., 4] *= -1; out[..., 7] *= -1
    return out

def mirror_target(t):
    out = t.copy(); out[..., 1] *= -1; return out


# ============================================================
# Neural ODE model
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim, dim),
        )
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        return self.ln(x + self.net(x))


class AccelField(nn.Module):
    """가속도 필드 a(pos, vel, latent, speed) → R^3."""
    def __init__(self, latent_dim=64, hidden=64):
        super().__init__()
        # pos(3) + vel(3) + latent + speed_scalar(1) = 7 + latent
        self.net = nn.Sequential(
            nn.Linear(3 + 3 + latent_dim + 1, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            ResBlock(hidden),
            nn.Linear(hidden, 3),
        )

    def forward(self, pos, vel, latent, speed):
        if speed.dim() == 1: speed = speed.unsqueeze(-1)
        x = torch.cat([pos, vel, latent, speed], dim=-1)
        return self.net(x)


class NeuralODEModel(nn.Module):
    """Position-velocity 6D state, RK4 single-step 80ms integration.

    Encoder: flatten(seq local-rotated) + scalar feats → latent z₀
    Dynamics: dp/dt = v ; dv/dt = -damping ⊙ v + a_neural(p,v,z,speed)
    Integration: RK4 (4 evals, dt=80ms)
    Output: final local position + local_bias → rotate back + last_obs + global_bias
    """
    def __init__(self, seq_dim=99, scal_dim=40, latent_dim=64, hidden=64,
                  dt_pred=0.080, n_steps=1):
        super().__init__()
        self.dt_pred = dt_pred
        self.n_steps = n_steps
        in_dim = seq_dim + scal_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim), nn.GELU(),
            ResBlock(latent_dim),
            ResBlock(latent_dim),
        )
        self.accel_field = AccelField(latent_dim=latent_dim, hidden=hidden)
        self.learned_damping = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self.local_bias = nn.Parameter(torch.zeros(3))
        self.global_bias = nn.Parameter(torch.zeros(3))
        self._last_accels = []

    def _ode_deriv(self, pos, vel, latent, speed):
        a = self.accel_field(pos, vel, latent, speed)
        dp = vel
        dv = -self.learned_damping * vel + a
        return dp, dv, a

    def _rk4_step(self, pos, vel, latent, speed, dt):
        dp1, dv1, a1 = self._ode_deriv(pos, vel, latent, speed)
        pos2 = pos + 0.5 * dt * dp1; vel2 = vel + 0.5 * dt * dv1
        dp2, dv2, a2 = self._ode_deriv(pos2, vel2, latent, speed)
        pos3 = pos + 0.5 * dt * dp2; vel3 = vel + 0.5 * dt * dv2
        dp3, dv3, a3 = self._ode_deriv(pos3, vel3, latent, speed)
        pos4 = pos + dt * dp3; vel4 = vel + dt * dv3
        dp4, dv4, a4 = self._ode_deriv(pos4, vel4, latent, speed)
        new_pos = pos + (dt / 6.0) * (dp1 + 2 * dp2 + 2 * dp3 + dp4)
        new_vel = vel + (dt / 6.0) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)
        return new_pos, new_vel, [a1, a2, a3, a4]

    def forward(self, seq_flat, scal, init_vel, speed):
        """Returns predicted local-frame displacement from last obs.

        seq_flat: (B, seq_dim) flattened normalized seq (local-rotated)
        scal:     (B, scal_dim)
        init_vel: (B, 3) initial velocity in local frame
        speed:    (B,) magnitude of init_vel (separate input for stability)
        """
        latent = self.backbone(torch.cat([seq_flat, scal], dim=-1))
        pos = torch.zeros_like(init_vel)
        vel = init_vel
        dt = self.dt_pred / self.n_steps
        accels = []
        for _ in range(self.n_steps):
            pos, vel, ac = self._rk4_step(pos, vel, latent, speed, dt)
            accels.extend(ac)
        self._last_accels = accels
        return pos + self.local_bias


# ============================================================
# Loss
# ============================================================
def loss_huber(pred, true, delta=0.001):
    return F.huber_loss(pred, true, delta=delta)

def loss_softhit(pred, true, k=300.0, c=0.01):
    d = torch.sqrt(((pred - true) ** 2).sum(-1) + 1e-12)
    return torch.sigmoid((d - c) * k).mean()


def loss_combined(pred, true, accels, w_huber=100.0, w_hit=1.0, w_reg=1e-4):
    huber = loss_huber(pred, true)
    hit = loss_softhit(pred, true)
    reg = 0.0
    if accels:
        reg = sum(a.pow(2).sum(-1).mean() for a in accels) / len(accels)
    return w_hit * hit + w_huber * huber + w_reg * reg, huber.item(), hit.item()


# ============================================================
# Fold runner
# ============================================================
def run_kfold(X_train, X_test, y_train,
              kalman_train, kalman_test,
              theta_train, theta_test,
              X_scal_tr, X_scal_te,
              cfg, device="cpu"):
    n_folds = cfg["n_folds"]; n_seeds = cfg["n_seeds"]
    max_epochs = cfg["max_epochs"]; patience = cfg["patience"]
    batch = cfg["batch"]; lr = cfg["lr"]; wd = cfg["wd"]
    mirror_on = cfg["mirror"]

    N = X_train.shape[0]

    # --- local-rotated seq + scalar feats ---
    # seq from build_seq: (N, 11, 9) = rel(3) + v_pad(3) + a_pad(3) all in raw frame
    # We rotate position/velocity/accel parts in xy by theta_train (in-place rotation)
    seq_tr_raw = build_seq(X_train); seq_te_raw = build_seq(X_test)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    # apply yaw rotation: each 3-channel block (rel, v_pad, a_pad)
    def rot_seq(seq, theta):
        out = seq.copy()
        for k in range(3):
            blk = out[..., 3*k:3*k+3]
            out[..., 3*k:3*k+3] = rotate_xy_seq(blk, theta)
        return out
    seq_tr = rot_seq(seq_tr_raw, theta_train)
    seq_te = rot_seq(seq_te_raw, theta_test)

    # initial velocity in local frame: from v_pad[:,-1] (already rotated)
    init_vel_tr = seq_tr[:, -1, 3:6].astype(np.float32)  # local v at last step
    init_vel_te = seq_te[:, -1, 3:6].astype(np.float32)
    speed_tr = np.linalg.norm(init_vel_tr, axis=-1).astype(np.float32)
    speed_te = np.linalg.norm(init_vel_te, axis=-1).astype(np.float32)

    # target: y_train in canonical local frame (vs last obs)
    target_local = rotate_xy(y_train - X_train[:, -1], theta_train).astype(np.float32)

    # scalar features
    scal_tr_full = np.concatenate([X_scal_tr, tier3_tr], axis=-1).astype(np.float32)
    scal_te_full = np.concatenate([X_scal_te, tier3_te], axis=-1).astype(np.float32)

    seq_flat_dim = seq_tr.shape[1] * seq_tr.shape[2]  # 11*9 = 99
    scal_dim = scal_tr_full.shape[1]
    print(f"[v120] N={N}, seq_flat_dim={seq_flat_dim}, scal_dim={scal_dim}")

    # mirror copies (full set; gather later by fold tr indices)
    if mirror_on:
        seq_tr_m = mirror_seq(seq_tr)
        init_vel_tr_m = seq_tr_m[:, -1, 3:6].astype(np.float32)
        target_local_m = mirror_target(target_local)

    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))
    fold_iter = fold_iter[:n_folds]

    oof_local = np.zeros((N, 3), dtype=np.float32)
    fold_mask = np.zeros(N, dtype=bool)
    test_per_fold = []
    fold_rh_list = []
    t0 = time.time()

    for fi, (tr, va) in enumerate(fold_iter):
        # normalize seq + scal on tr
        sc_seq = StandardScaler().fit(seq_tr[tr].reshape(-1, seq_tr.shape[2]))
        sc_scal = StandardScaler().fit(scal_tr_full[tr])
        seq_n = normalize_seq(seq_tr, sc_seq)
        seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_n = sc_scal.transform(scal_tr_full).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te_full).astype(np.float32)
        if mirror_on:
            seq_m_n = normalize_seq(seq_tr_m, sc_seq)

        # flatten
        seq_flat = seq_n.reshape(N, -1)
        seq_te_flat = seq_te_n.reshape(seq_te.shape[0], -1)
        if mirror_on:
            seq_m_flat = seq_m_n.reshape(N, -1)

        # tensors
        def T(a, d=device): return torch.from_numpy(a).to(d)

        # training set (optional 2x mirror)
        tr_idx = tr
        if mirror_on:
            seq_tr_t = torch.from_numpy(np.concatenate([seq_flat[tr_idx], seq_m_flat[tr_idx]], axis=0)).to(device)
            scal_tr_t = torch.from_numpy(np.concatenate([scal_n[tr_idx], scal_n[tr_idx]], axis=0)).to(device)
            vel_tr_t = torch.from_numpy(np.concatenate([init_vel_tr[tr_idx], init_vel_tr_m[tr_idx]], axis=0)).to(device)
            sp_tr_t = torch.from_numpy(np.concatenate([speed_tr[tr_idx], speed_tr[tr_idx]], axis=0)).to(device)
            tgt_tr_t = torch.from_numpy(np.concatenate([target_local[tr_idx], target_local_m[tr_idx]], axis=0)).to(device)
        else:
            seq_tr_t = T(seq_flat[tr_idx])
            scal_tr_t = T(scal_n[tr_idx])
            vel_tr_t = T(init_vel_tr[tr_idx])
            sp_tr_t = T(speed_tr[tr_idx])
            tgt_tr_t = T(target_local[tr_idx])

        seq_va_t = T(seq_flat[va]); scal_va_t = T(scal_n[va])
        vel_va_t = T(init_vel_tr[va]); sp_va_t = T(speed_tr[va])
        tgt_va_t = T(target_local[va])

        seq_te_t = T(seq_te_flat); scal_te_t = T(scal_te_n)
        vel_te_t = T(init_vel_te); sp_te_t = T(speed_te)

        # accumulate test predictions over seeds
        test_fold = np.zeros((seq_te.shape[0], 3), dtype=np.float32)

        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = NeuralODEModel(seq_dim=seq_flat_dim, scal_dim=scal_dim,
                                    latent_dim=cfg.get("latent_dim", 64),
                                    hidden=cfg.get("hidden", 64),
                                    n_steps=cfg.get("n_steps", 1)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

            best_oof_rh = -1.0; best_state = None; best_ep = 0; bad = 0
            n_tr = seq_tr_t.shape[0]

            for ep in range(1, max_epochs + 1):
                model.train()
                perm = torch.randperm(n_tr)
                ep_loss = 0.0; ep_huber = 0.0; ep_hit = 0.0; nb = 0
                for s in range(0, n_tr, batch):
                    idx = perm[s:s+batch]
                    pred = model(seq_tr_t[idx], scal_tr_t[idx], vel_tr_t[idx], sp_tr_t[idx])
                    loss, h, hit = loss_combined(pred, tgt_tr_t[idx], model._last_accels,
                                                  w_huber=100.0, w_hit=1.0, w_reg=1e-4)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item(); ep_huber += h; ep_hit += hit; nb += 1
                sch.step()

                # eval
                model.eval()
                with torch.no_grad():
                    pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                # un-rotate to global to compute true R-Hit
                pv_global = X_train[va, -1] + inverse_rotate_xy(pv, theta_train[va])
                d = np.linalg.norm(pv_global - y_train[va], axis=-1)
                rh = float((d <= 0.01).mean())
                if rh > best_oof_rh:
                    best_oof_rh = rh; best_ep = ep; bad = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if ep <= 5 or ep % 5 == 0 or bad >= patience:
                    print(f"  fold{fi} seed{seed} ep{ep:3d}/{max_epochs}: loss={ep_loss/nb:.4f} "
                          f"huber={ep_huber/nb:.5f} hit={ep_hit/nb:.4f}  va R-Hit={rh:.4f} best={best_oof_rh:.4f}@ep{best_ep}")
                if bad >= patience:
                    print(f"  -> early stop ep{ep} (patience {patience})"); break

            # load best
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                if mirror_on:
                    # TTA mirror: predict on mirrored input then un-mirror
                    seq_va_m_flat = seq_m_flat[va]
                    init_vel_va_m = init_vel_tr_m[va]
                    pv_m = model(T(seq_va_m_flat), scal_va_t, T(init_vel_va_m), sp_va_t).cpu().numpy()
                    pv = 0.5 * (pv + mirror_target(pv_m))
                pte = model(seq_te_t, scal_te_t, vel_te_t, sp_te_t).cpu().numpy()
                if mirror_on:
                    seq_te_m_flat = mirror_seq(seq_te)
                    # mirror rotated seq; init vel mirror
                    init_vel_te_m = seq_te_m_flat[:, -1, 3:6].astype(np.float32)
                    seq_te_m_n = normalize_seq(seq_te_m_flat, sc_seq).reshape(seq_te.shape[0], -1)
                    pte_m = model(T(seq_te_m_n), scal_te_t, T(init_vel_te_m), sp_te_t).cpu().numpy()
                    pte = 0.5 * (pte + mirror_target(pte_m))

            # average over seeds
            oof_local[va] += pv / n_seeds
            test_fold += pte / n_seeds

        fold_mask[va] = True
        test_per_fold.append(test_fold)
        fold_rh_list.append(best_oof_rh)
        print(f"[v120] fold{fi} best R-Hit={best_oof_rh:.4f}  elapsed {(time.time()-t0)/60:.1f}m")

    # global R-Hit
    oof_global = X_train[fold_mask, -1] + inverse_rotate_xy(oof_local[fold_mask], theta_train[fold_mask])
    rh_oof = float((np.linalg.norm(oof_global - y_train[fold_mask], axis=-1) <= 0.01).mean())
    print(f"[v120] OOF R-Hit = {rh_oof:.4f}  (covered {fold_mask.sum()}/{N})")

    test_local = np.mean(test_per_fold, axis=0)
    test_global = X_test[:, -1] + inverse_rotate_xy(test_local, theta_test)

    # full oof in global (for downstream blend)
    oof_global_full = np.zeros((N, 3), dtype=np.float32)
    oof_global_full[fold_mask] = oof_global

    return oof_local, oof_global_full, fold_mask, test_global, rh_oof, fold_rh_list


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODE_CONFIGS.keys()))
    ap.add_argument("--tag", default=None, help="output suffix")
    ap.add_argument("--n_steps", type=int, default=1, help="RK4 multi-step count (dt = 0.080/n_steps)")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    cfg = MODE_CONFIGS[args.mode]
    cfg = {**cfg, "n_steps": args.n_steps, "latent_dim": args.latent_dim, "hidden": args.hidden}

    tag = args.tag or args.mode
    state_file = CACHE_DIR / f"v120_{tag}_state.npz"
    sub_file = OUT_DIR / f"submission_v120_{tag}.csv"

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 60)
    print(f"v120 Neural ODE  mode={args.mode}  cfg={cfg}")
    print(f"torch={torch.__version__} threads={torch.get_num_threads()}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, _ = get_kalman(X_train, X_test)
    rh_kal = float((np.linalg.norm(kalman_train - y_train, axis=-1) <= 0.01).mean())
    print(f"[kalman baseline] R-Hit train: {rh_kal:.4f}")

    # scalar feats (existing cache)
    cfg_noise_mode = "fast"  # noise cache "fast" has full LOO subset (2000)
    M_noise = {"loo_sample": 2000}
    X_scal_tr, X_scal_te = get_scalar_feats(X_train, X_test, M_noise, cfg_noise_mode)

    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test = (X_test[:, -1] - X_test[:, -2]) / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)

    t0 = time.time()
    oof_local, oof_global_full, fold_mask, test_global, rh_oof, fold_rh_list = run_kfold(
        X_train, X_test, y_train,
        kalman_train, kalman_test, theta_train, theta_test,
        X_scal_tr, X_scal_te, cfg, device=device,
    )
    print(f"[v120] total {(time.time()-t0)/60:.1f}m")

    # save state for downstream blending
    np.savez(state_file,
              oof_local=oof_local, oof_global=oof_global_full, fold_mask=fold_mask,
              test_global=test_global, rh_oof=rh_oof,
              fold_rh=np.array(fold_rh_list), theta_train=theta_train, theta_test=theta_test)
    print(f"[v120] state saved: {state_file}")

    # write submission
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub[["x", "y", "z"]] = test_global
    sub.to_csv(sub_file, index=False)
    print(f"[v120] submission saved: {sub_file}")

    # correlation with v112 OOF (if exists)
    v112_csv = PROJECT_DIR / "final_candidates" / "submission_v112_v107_diverse_oof0.6768.csv"
    v106_csv = PROJECT_DIR / "final_candidates" / "submission_v106_DE15w_oof0.6770.csv"
    for ref_csv in [v112_csv, v106_csv]:
        if ref_csv.exists():
            ref = pd.read_csv(ref_csv)
            ref_xyz = ref[["x", "y", "z"]].values
            # correlation between test predictions (proxy for residual diversity)
            corr_per_axis = [
                float(np.corrcoef(test_global[:, k], ref_xyz[:, k])[0, 1])
                for k in range(3)
            ]
            diff = test_global - ref_xyz
            mean_diff = float(np.linalg.norm(diff, axis=-1).mean())
            print(f"[v120][corr-test {ref_csv.name}] x={corr_per_axis[0]:.4f} "
                  f"y={corr_per_axis[1]:.4f} z={corr_per_axis[2]:.4f} | "
                  f"mean L2 diff={mean_diff*1000:.2f}mm")

    # log
    log_entry = {
        "version": "v120",
        "mode": args.mode,
        "cfg": cfg,
        "oof_rhit": rh_oof,
        "fold_rh": fold_rh_list,
        "submission": str(sub_file),
        "state": str(state_file),
        "elapsed_min": (time.time()-t0)/60,
    }
    log_path = PROJECT_DIR / "run_log.json"
    try:
        if log_path.exists():
            arr = json.loads(log_path.read_text())
        else:
            arr = []
        arr.append(log_entry)
        log_path.write_text(json.dumps(arr, indent=2, default=str))
    except Exception as e:
        print(f"[log] append failed: {e}")

    print("[v120] done.")


if __name__ == "__main__":
    main()
