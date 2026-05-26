"""v127_neural_cde.py — Neural Controlled Differential Equation backbone.

참고: Kidger 2020 NeurIPS "Neural CDEs for Irregular Time Series" (torchcde 라이브러리).

핵심 차이 (v120 Neural ODE 대비):
  - v120: encoder → fixed latent z0, ODE state (pos, vel) 만 진화
  - v127: CDE state z(t) = f_θ(z) dX(t)/dt  — 관측 시퀀스 X가 hidden state 진화의 "control"
          → time-continuous representation of obs, 그리고 80ms 외삽

Setup:
  - 11 obs (t = -400..0ms by 40ms) → cubic Hermite spline interpolation
  - z0 = encoder(obs[0])
  - integrate z over [-400, 0]ms via CDE
  - decoder(z(0)) → output: 80ms 후 좌표 (last_obs + residual)
  - 또는 z를 80ms 이후로 ODE 식으로 연장 후 decoder

설치:
  pip install torchcde  # 추가 의존성

사용:
  python scripts/v127_neural_cde.py --mode smoke
  python scripts/v127_neural_cde.py --mode full   # Colab T4 추천
"""
from __future__ import annotations
import argparse, os, random, sys, time
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

try:
    import torchcde
except ImportError:
    print("ERROR: torchcde not installed. Run: pip install torchcde")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (
    DT, T_PRED, load_data, get_kalman, get_scalar_feats,
    build_seq, build_tier3,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
)
from v120_neural_ode import mirror_seq, mirror_target, rotate_xy_seq

PROJ = SCRIPT_DIR.parent
DATA = PROJ / "open"
CACHE = PROJ / "cache"
OUT = PROJ / "open"

MODE = {
    "smoke": dict(n_folds=1, n_seeds=1, max_epochs=30, patience=8, batch=128, lr=2e-3, wd=1e-3, mirror=False, hidden=32),
    "fast":  dict(n_folds=2, n_seeds=1, max_epochs=60, patience=12, batch=128, lr=2e-3, wd=1e-3, mirror=True,  hidden=48),
    "full":  dict(n_folds=5, n_seeds=2, max_epochs=80, patience=15, batch=128, lr=2e-3, wd=1e-3, mirror=True,  hidden=64),
}

T_OBS = np.linspace(-0.400, 0.0, 11, dtype=np.float32)  # 11 obs at 40ms apart


class CDEFunc(nn.Module):
    """f_θ(t, z): hidden_dim × control_dim 출력 (matrix). dZ/dt = f(z) · dX/dt."""
    def __init__(self, hidden=64, control_dim=4):
        super().__init__()
        self.hidden = hidden
        self.control_dim = control_dim
        self.net = nn.Sequential(
            nn.Linear(hidden, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, hidden * control_dim),
        )
    def forward(self, t, z):
        # z: (B, hidden)
        out = self.net(z).view(-1, self.hidden, self.control_dim)
        return torch.tanh(out)  # bound for stability


class NeuralCDEModel(nn.Module):
    def __init__(self, scal_dim=40, hidden=64, control_dim=4):
        super().__init__()
        self.hidden = hidden
        self.control_dim = control_dim
        self.initial = nn.Sequential(
            nn.Linear(control_dim + scal_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.cdefunc = CDEFunc(hidden, control_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, coeffs, scal, t0_X):
        """coeffs: hermite cubic coeffs for control (B, T, control_dim) → torchcde format
        scal: (B, scal_dim) static features
        t0_X: (B, control_dim) X at time 0 (last obs).
        Returns: (B, 3) predicted local-frame residual at t = +80ms.
        """
        X = torchcde.CubicSpline(coeffs)  # B time x control
        z0 = self.initial(torch.cat([t0_X, scal], dim=-1))
        # integrate from t=-0.4 to t=0
        t = torch.tensor([-0.400, 0.0], device=coeffs.device, dtype=coeffs.dtype)
        zT = torchcde.cdeint(X=X, func=self.cdefunc, z0=z0, t=t,
                              adjoint=False, method="rk4",
                              options={"step_size": 0.040})[:, -1]
        # readout at z(0) → linear extrapolation 80ms 후 좌표
        out = self.readout(zT) + self.bias
        return out


def build_control_seq(X, theta):
    """X: (N, 11, 3) rotated by theta → return (N, 11, control_dim=4): [t, x, y, z]."""
    N = X.shape[0]
    # rotate position by theta around z
    c = np.cos(theta)[:, None]; s = np.sin(theta)[:, None]
    x_new = X[..., 0] * c + X[..., 1] * s
    y_new = -X[..., 0] * s + X[..., 1] * c
    pos_local = np.stack([x_new, y_new, X[..., 2]], axis=-1).astype(np.float32)
    # subtract last obs → local-rotated rel
    pos_local = pos_local - pos_local[:, -1:, :]
    t_col = np.broadcast_to(T_OBS[None, :, None], (N, 11, 1)).copy()
    return np.concatenate([t_col, pos_local], axis=-1).astype(np.float32)  # (N, 11, 4)


def loss_combo(pred, true, w_h=100.0, w_hit=1.0):
    huber = F.huber_loss(pred, true, delta=0.001)
    d = torch.sqrt(((pred - true)**2).sum(-1) + 1e-12)
    hit = torch.sigmoid((d - 0.01) * 300.0).mean()
    return w_h * huber + w_hit * hit, huber.item(), hit.item()


def run_kfold(X_train, X_test, y_train, theta_tr, theta_te, scal_tr, scal_te, cfg, device="cpu"):
    n_folds=cfg["n_folds"]; n_seeds=cfg["n_seeds"]; max_epochs=cfg["max_epochs"]
    patience=cfg["patience"]; batch=cfg["batch"]; lr=cfg["lr"]; wd=cfg["wd"]
    mirror_on=cfg["mirror"]; hidden=cfg["hidden"]
    N = X_train.shape[0]
    Nte = X_test.shape[0]
    ctl_tr = build_control_seq(X_train, theta_tr)
    ctl_te = build_control_seq(X_test, theta_te)
    target_local = rotate_xy(y_train - X_train[:, -1], theta_tr).astype(np.float32)

    # mirror in y axis (control x stays, y flips, z stays)
    if mirror_on:
        ctl_tr_m = ctl_tr.copy(); ctl_tr_m[..., 2] *= -1
        target_local_m = target_local.copy(); target_local_m[:, 1] *= -1

    print(f"[v127] control shape: {ctl_tr.shape}, scal dim: {scal_tr.shape[1]}")
    print(f"[v127] target local sample: {target_local[0]}")

    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))[:n_folds]
    oof_local = np.zeros((N, 3), dtype=np.float32)
    fold_mask = np.zeros(N, dtype=bool)
    test_per_fold = []
    fold_rh = []
    t0 = time.time()

    for fi, (tr, va) in enumerate(fold_iter):
        sc_scal = StandardScaler().fit(scal_tr[tr])
        scal_n = sc_scal.transform(scal_tr).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te).astype(np.float32)

        coeffs_tr = torchcde.hermite_cubic_coefficients_with_backward_differences(
            torch.from_numpy(ctl_tr).to(device))
        coeffs_te = torchcde.hermite_cubic_coefficients_with_backward_differences(
            torch.from_numpy(ctl_te).to(device))
        if mirror_on:
            coeffs_tr_m = torchcde.hermite_cubic_coefficients_with_backward_differences(
                torch.from_numpy(ctl_tr_m).to(device))
        t0_X_tr = torch.from_numpy(ctl_tr[:, -1, :]).to(device)
        t0_X_te = torch.from_numpy(ctl_te[:, -1, :]).to(device)
        if mirror_on:
            t0_X_tr_m = torch.from_numpy(ctl_tr_m[:, -1, :]).to(device)

        scal_t = torch.from_numpy(scal_n).to(device)
        scal_te_t = torch.from_numpy(scal_te_n).to(device)
        tgt_t = torch.from_numpy(target_local).to(device)
        if mirror_on:
            tgt_t_m = torch.from_numpy(target_local_m).to(device)

        test_fold = np.zeros((Nte, 3), dtype=np.float32)
        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = NeuralCDEModel(scal_dim=scal_n.shape[1], hidden=hidden, control_dim=4).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh = -1; bad = 0; best_ep = 0; best_state = None
            for ep in range(1, max_epochs + 1):
                model.train()
                perm_idx = np.array(tr)
                np.random.shuffle(perm_idx)
                ep_loss = 0; nb = 0
                for s in range(0, len(perm_idx), batch):
                    idx = perm_idx[s:s+batch]
                    pred = model(coeffs_tr[idx], scal_t[idx], t0_X_tr[idx])
                    loss, h, hit = loss_combo(pred, tgt_t[idx])
                    if mirror_on:
                        pred_m = model(coeffs_tr_m[idx], scal_t[idx], t0_X_tr_m[idx])
                        loss_m, _, _ = loss_combo(pred_m, tgt_t_m[idx])
                        loss = 0.5 * (loss + loss_m)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item(); nb += 1
                sch.step()
                # eval
                model.eval()
                with torch.no_grad():
                    pv = model(coeffs_tr[va], scal_t[va], t0_X_tr[va]).cpu().numpy()
                pv_global = X_train[va, -1] + inverse_rotate_xy(pv, theta_tr[va])
                rh = float((np.linalg.norm(pv_global - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh; best_ep = ep; bad = 0
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad += 1
                if ep <= 5 or ep % 5 == 0 or bad >= patience:
                    print(f"  fold{fi} seed{seed} ep{ep:3d}/{max_epochs}: loss={ep_loss/nb:.4f} va R-Hit={rh:.4f} best={best_rh:.4f}@ep{best_ep}")
                if bad >= patience: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv = model(coeffs_tr[va], scal_t[va], t0_X_tr[va]).cpu().numpy()
                if mirror_on:
                    pv_m = model(coeffs_tr_m[va], scal_t[va], t0_X_tr_m[va]).cpu().numpy()
                    pv_m[..., 1] *= -1
                    pv = 0.5 * (pv + pv_m)
                pte = model(coeffs_te, scal_te_t, t0_X_te).cpu().numpy()
                if mirror_on:
                    ctl_te_m = ctl_te.copy(); ctl_te_m[..., 2] *= -1
                    coeffs_te_m = torchcde.hermite_cubic_coefficients_with_backward_differences(
                        torch.from_numpy(ctl_te_m).to(device))
                    t0_X_te_m = torch.from_numpy(ctl_te_m[:, -1, :]).to(device)
                    pte_m = model(coeffs_te_m, scal_te_t, t0_X_te_m).cpu().numpy()
                    pte_m[..., 1] *= -1
                    pte = 0.5 * (pte + pte_m)
            oof_local[va] += pv / n_seeds
            test_fold += pte / n_seeds
        fold_mask[va] = True
        test_per_fold.append(test_fold)
        fold_rh.append(best_rh)
        print(f"[v127] fold{fi} best R-Hit={best_rh:.4f}  elapsed {(time.time()-t0)/60:.1f}m")

    oof_global = X_train[fold_mask, -1] + inverse_rotate_xy(oof_local[fold_mask], theta_tr[fold_mask])
    rh_oof = float((np.linalg.norm(oof_global - y_train[fold_mask], axis=-1) <= 0.01).mean())
    print(f"[v127] OOF R-Hit = {rh_oof:.4f}  (covered {fold_mask.sum()}/{N})")
    test_local = np.mean(test_per_fold, axis=0)
    test_global = X_test[:, -1] + inverse_rotate_xy(test_local, theta_te)
    oof_global_full = np.zeros((N, 3), dtype=np.float32)
    oof_global_full[fold_mask] = oof_global
    return oof_local, oof_global_full, fold_mask, test_global, rh_oof, fold_rh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODE.keys()))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = MODE[args.mode]
    tag = args.tag or args.mode
    state_file = CACHE / f"v127_{tag}_state.npz"
    sub_file = OUT / f"submission_v127_{tag}.csv"

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"v127 Neural CDE mode={args.mode}  device={device}")

    X_train, X_test, y_train, _ = load_data()
    kalman_train, kalman_test, _ = get_kalman(X_train, X_test)
    X_scal_tr, X_scal_te = get_scalar_feats(X_train, X_test, {"loo_sample":2000}, "fast")
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    scal_tr = np.concatenate([X_scal_tr, tier3_tr], axis=-1).astype(np.float32)
    scal_te = np.concatenate([X_scal_te, tier3_te], axis=-1).astype(np.float32)
    v_last_tr = (X_train[:,-1]-X_train[:,-2])/DT
    v_last_te = (X_test[:,-1]-X_test[:,-2])/DT
    theta_tr, theta_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)

    t0 = time.time()
    oof_local, oof_global, fold_mask, test_global, rh_oof, fold_rh = run_kfold(
        X_train, X_test, y_train, theta_tr, theta_te, scal_tr, scal_te, cfg, device)
    print(f"[v127] total {(time.time()-t0)/60:.1f}m")

    np.savez(state_file,
              oof_local=oof_local, oof_global=oof_global, fold_mask=fold_mask,
              test_global=test_global, rh_oof=rh_oof,
              fold_rh=np.array(fold_rh), theta_train=theta_tr, theta_test=theta_te)
    print(f"[v127] saved {state_file}")
    sub = pd.read_csv(DATA / "sample_submission.csv")
    sub[["x","y","z"]] = test_global
    sub.to_csv(sub_file, index=False)
    print(f"[v127] submission saved: {sub_file}")

if __name__ == "__main__":
    main()
