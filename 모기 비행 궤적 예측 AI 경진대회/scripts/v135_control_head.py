"""v135_control_head.py — control-head analytic-integrator 멤버 (decorrelated by target parametrization).

RK4 ODE(v120/v131)와 달리 NN이 control(accel, optional jerk)을 출력하고 닫힌형으로 적분:
    p_disp = v0*T + 0.5*a*T^2 (+ 1/6*j*T^3)    (T=80ms, last-obs 기준 변위)
analytic CA(유한차분 accel, 0.08)와 다름: NN이 80ms-ahead용 effective control을 hit-loss로 학습.
새 error 구조 → kalman-residual/RK4-ODE 양쪽과 decorrelated. (Trajectron++ control-integration)

v131의 (수정된) 프레임/mirror 파이프라인을 그대로 재사용 — frame={yaw,frenet}, mirror 일관성 보장.
usage: python scripts/v135_control_head.py --frame frenet --order accel --mode full --tag ch_frenet
"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
SCRIPT_DIR = Path(__file__).resolve().parent; sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import (DT, T_PRED, load_data, get_scalar_feats, build_seq, build_tier3, normalize_seq)
from v120_neural_ode import ResBlock
# 수정된 프레임/mirror 헬퍼 재사용 (v131)
from v131_paradigm_variants import (frenet_frame, yaw_frame, apply_R_seq, apply_R_vec,
                                     inv_R_vec, mirror_seq, mirror_vec)

PROJECT_DIR = SCRIPT_DIR.parent; DATA_DIR = PROJECT_DIR / "open"; CACHE_DIR = PROJECT_DIR / "cache"
MODE_CONFIGS = {
    "smoke": dict(n_folds=1, n_seeds=1, max_epochs=30, patience=8, batch=256, lr=2e-3, wd=1e-3, mirror=False),
    "fast":  dict(n_folds=2, n_seeds=1, max_epochs=60, patience=12, batch=256, lr=2e-3, wd=1e-3, mirror=True),
    "full":  dict(n_folds=5, n_seeds=2, max_epochs=80, patience=15, batch=256, lr=2e-3, wd=1e-3, mirror=True),
}

def loss_combined(pred, true):
    import torch.nn.functional as F
    huber = F.huber_loss(pred, true, delta=0.001)
    d = torch.sqrt(((pred - true) ** 2).sum(-1) + 1e-12)
    hit = torch.sigmoid((d - 0.01) * 300.0).mean()
    return 100.0 * huber + 1.0 * hit

class ControlHead(nn.Module):
    """seq+scal -> latent -> control(accel[, jerk]); analytic integrate to +80ms disp."""
    def __init__(self, seq_dim, scal_dim, latent_dim=64, order="accel", T=0.080):
        super().__init__()
        self.T = T; self.order = order
        self.backbone = nn.Sequential(
            nn.Linear(seq_dim + scal_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(),
            ResBlock(latent_dim), ResBlock(latent_dim))
        nc = 3 if order == "accel" else 6  # accel | accel+jerk
        self.head = nn.Linear(latent_dim, nc)
        self.local_bias = nn.Parameter(torch.zeros(3))
        self.v_scale = nn.Parameter(torch.ones(1))   # learned trust on init velocity

    def forward(self, seq_flat, scal, init_vel, speed):
        z = self.backbone(torch.cat([seq_flat, scal], dim=-1))
        c = self.head(z); T = self.T
        a = c[:, :3]
        disp = self.v_scale * init_vel * T + 0.5 * a * (T * T)
        if self.order == "jerk":
            j = c[:, 3:6]; disp = disp + (1.0 / 6.0) * j * (T ** 3)
        return disp + self.local_bias

def run_kfold(X_train, X_test, y_train, R_train, R_test, X_scal_tr, X_scal_te,
              cfg, order="accel", mirror_axis=1, device="cpu"):
    n_folds, n_seeds = cfg["n_folds"], cfg["n_seeds"]
    max_epochs, patience = cfg["max_epochs"], cfg["patience"]
    batch, lr, wd, mirror_on = cfg["batch"], cfg["lr"], cfg["wd"], cfg["mirror"]
    N = X_train.shape[0]
    seq_tr = apply_R_seq(build_seq(X_train), R_train); seq_te = apply_R_seq(build_seq(X_test), R_test)
    tier3_tr, tier3_te = build_tier3(X_train), build_tier3(X_test)
    init_vel_tr = seq_tr[:, -1, 3:6].astype(np.float32); init_vel_te = seq_te[:, -1, 3:6].astype(np.float32)
    speed_tr = np.linalg.norm(init_vel_tr, axis=-1).astype(np.float32); speed_te = np.linalg.norm(init_vel_te, axis=-1).astype(np.float32)
    target_local = apply_R_vec(y_train - X_train[:, -1], R_train)
    scal_tr_full = np.concatenate([X_scal_tr, tier3_tr], -1).astype(np.float32)
    scal_te_full = np.concatenate([X_scal_te, tier3_te], -1).astype(np.float32)
    scal_dim = scal_tr_full.shape[1]; C = seq_tr.shape[2]; seq_flat_dim = seq_tr.shape[1] * C
    print(f"[v135:{order}] N={N} seq_flat={seq_flat_dim} scal={scal_dim}")
    if mirror_on:
        seq_tr_m = mirror_seq(seq_tr, mirror_axis); init_vel_tr_m = seq_tr_m[:, -1, 3:6].astype(np.float32)
        target_local_m = mirror_vec(target_local, mirror_axis)
    kf = KFold(n_splits=max(n_folds, 2), shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))[:n_folds]
    oof_local = np.zeros((N, 3), np.float32); fold_mask = np.zeros(N, bool)
    test_per_fold, fold_rh = [], []; t0 = time.time()
    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_tr[tr].reshape(-1, C)); sc_scal = StandardScaler().fit(scal_tr_full[tr])
        seq_n = normalize_seq(seq_tr, sc_seq).reshape(N, -1); seq_te_n = normalize_seq(seq_te, sc_seq).reshape(seq_te.shape[0], -1)
        scal_n = sc_scal.transform(scal_tr_full).astype(np.float32); scal_te_n = sc_scal.transform(scal_te_full).astype(np.float32)
        if mirror_on: seq_m_n = normalize_seq(seq_tr_m, sc_seq).reshape(N, -1)
        def T(a): return torch.from_numpy(np.ascontiguousarray(a)).to(device)
        if mirror_on:
            seq_in = np.concatenate([seq_n[tr], seq_m_n[tr]], 0); scal_in = np.concatenate([scal_n[tr], scal_n[tr]], 0)
            vel_in = np.concatenate([init_vel_tr[tr], init_vel_tr_m[tr]], 0); sp_in = np.concatenate([speed_tr[tr], speed_tr[tr]], 0)
            tgt_in = np.concatenate([target_local[tr], target_local_m[tr]], 0)
        else:
            seq_in, scal_in, vel_in, sp_in, tgt_in = seq_n[tr], scal_n[tr], init_vel_tr[tr], speed_tr[tr], target_local[tr]
        seq_tr_t, scal_tr_t, vel_tr_t, sp_tr_t, tgt_tr_t = T(seq_in), T(scal_in), T(vel_in), T(sp_in), T(tgt_in)
        seq_va_t, scal_va_t, vel_va_t, sp_va_t = T(seq_n[va]), T(scal_n[va]), T(init_vel_tr[va]), T(speed_tr[va])
        seq_te_t, scal_te_t, vel_te_t, sp_te_t = T(seq_te_n), T(scal_te_n), T(init_vel_te), T(speed_te)
        test_fold = np.zeros((seq_te.shape[0], 3), np.float32)
        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = ControlHead(seq_flat_dim, scal_dim, latent_dim=cfg.get("latent_dim", 64), order=order, T=T_PRED).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh, best_state, best_ep, bad = -1.0, None, 0, 0; n_tr = seq_tr_t.shape[0]
            for ep in range(1, max_epochs + 1):
                model.train(); perm = torch.randperm(n_tr)
                for s in range(0, n_tr, batch):
                    idx = perm[s:s+batch]
                    pred = model(seq_tr_t[idx], scal_tr_t[idx], vel_tr_t[idx], sp_tr_t[idx])
                    loss = loss_combined(pred, tgt_tr_t[idx])
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                sch.step(); model.eval()
                with torch.no_grad(): pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                pv_g = X_train[va, -1] + inv_R_vec(pv, R_train[va])
                rh = float((np.linalg.norm(pv_g - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh: best_rh, best_ep, bad = rh, ep, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else: bad += 1
                if bad >= patience: break
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv = model(seq_va_t, scal_va_t, vel_va_t, sp_va_t).cpu().numpy()
                pte = model(seq_te_t, scal_te_t, vel_te_t, sp_te_t).cpu().numpy()
                if mirror_on:
                    pv_m = model(T(seq_m_n[va]), scal_va_t, T(init_vel_tr_m[va]), sp_va_t).cpu().numpy()
                    pv = 0.5 * (pv + mirror_vec(pv_m, mirror_axis))
                    seq_te_m_raw = mirror_seq(seq_te, mirror_axis); vel_te_m = seq_te_m_raw[:, -1, 3:6].astype(np.float32)
                    seq_te_m_n = normalize_seq(seq_te_m_raw, sc_seq).reshape(seq_te.shape[0], -1)
                    pte_m = model(T(seq_te_m_n), scal_te_t, T(vel_te_m), sp_te_t).cpu().numpy()
                    pte = 0.5 * (pte + mirror_vec(pte_m, mirror_axis))
            oof_local[va] += pv / n_seeds; test_fold += pte / n_seeds
        fold_mask[va] = True; test_per_fold.append(test_fold); fold_rh.append(best_rh)
        print(f"[v135] fold{fi} RH={best_rh:.4f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    oof_g = X_train[fold_mask, -1] + inv_R_vec(oof_local[fold_mask], R_train[fold_mask])
    rh_oof = float((np.linalg.norm(oof_g - y_train[fold_mask], axis=-1) <= 0.01).mean())
    print(f"[v135] OOF R-Hit = {rh_oof:.4f}")
    test_global = X_test[:, -1] + inv_R_vec(np.mean(test_per_fold, 0), R_test)
    oof_full = np.zeros((N, 3), np.float32); oof_full[fold_mask] = oof_g
    return oof_full, fold_mask, test_global, rh_oof, fold_rh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke", choices=list(MODE_CONFIGS.keys()))
    ap.add_argument("--frame", default="frenet", choices=["yaw", "frenet"])
    ap.add_argument("--order", default="accel", choices=["accel", "jerk"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = MODE_CONFIGS[args.mode]
    tag = args.tag or f"ch_{args.frame}_{args.order}"
    os.environ["PYTHONHASHSEED"] = "0"; random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"v135 control-head frame={args.frame} order={args.order} mode={args.mode}")
    X_train, X_test, y_train, sub = load_data()
    X_scal_tr, X_scal_te = get_scalar_feats(X_train, X_test, {"loo_sample": 2000}, "fast")
    frame_fn = frenet_frame if args.frame == "frenet" else yaw_frame
    R_train, R_test = frame_fn(X_train), frame_fn(X_test)
    mirror_axis = 2 if args.frame == "frenet" else 1
    oof_g, fold_mask, test_global, rh_oof, fold_rh = run_kfold(
        X_train, X_test, y_train, R_train, R_test, X_scal_tr, X_scal_te, cfg, order=args.order,
        mirror_axis=mirror_axis, device="cpu")
    np.savez(CACHE_DIR / f"v135_{tag}_state.npz", oof_global=oof_g, fold_mask=fold_mask,
             test_global=test_global, rh_oof=rh_oof, fold_rh=np.array(fold_rh))
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv"); sub[["x","y","z"]] = test_global
    sub.to_csv(DATA_DIR / f"submission_v135_{tag}.csv", index=False)
    for ref, k, nm in [("v120_full_state.npz", "test_global", "v120"), ("v122c_v121diverse_weights.npz", "test_pred", "v122c")]:
        st = np.load(CACHE_DIR / ref); t = st[k]
        # OOF-vs-TEST 일관성 동시 확인
        oofref = st["oof_global"] if "oof_global" in st.files else st["oof_pred"]
        print(f"[decorr {nm}] TEST={np.linalg.norm(test_global-t,axis=-1).mean()*1000:.2f}mm  OOF={np.linalg.norm(oof_g-oofref,axis=-1).mean()*1000:.2f}mm")
    print(f"[v135] {tag} OOF={rh_oof:.4f} saved")

if __name__ == "__main__":
    main()
