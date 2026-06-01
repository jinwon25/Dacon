"""v109_mdn_wta.py - BiGRU + MDN-WTA head + mirror aug + TTA.

paradigm shift: trajectory future is intrinsically multi-modal
(avoid / continue / U-turn). single-mode regression averages modes
and loses 1cm hit rate.

design:
  - backbone: v77 BiGRUMultiAux (replace main_head with K-mixture head)
  - head: K=4 means (rotated frame, tanh * scale) + K logits
  - loss WTA: min over K of euclid + softhit, on argmin mode only
    + small entropy reg on softmax weights (prevent mode collapse)
    + weighted-mean loss (small) to keep weights calibrated
  - aux F/W heads kept single (not multi-mode) for simplicity
  - mirror aug (y-flip same as v90) + TTA (2-view inference avg)

inference cache (per fold/seed mean):
  - means_rot: (N, K, 3) in rotated frame
  - logits:    (N, K)
  - weighted_mean_rot: (N, 3)  = sum_k softmax(logits)_k * means_k
  - argmax_rot:        (N, 3)  = means at argmax(logits)
  -> ensemble pool에는 (1) weighted_mean (2) argmax 두 개 추가

separate cache: v109_state.npz (full output bundle)

기대:
  v77 BiGRU OOF ~ 0.66, v90 mirror ~ 0.664
  v109 MDN base OOF ~ 0.665+ (다른 head paradigm),
  ensemble lift +0.0010~0.0040 (v104b 진정성 패턴)
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, json, os, sys, time
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
    load_data, get_kalman, get_scalar_feats, build_tier3, build_seq,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
    MODE_CONFIGS, CACHE_DIR, DATA_DIR,
    loss_euclid,
)

PROJECT = SCRIPT_DIR.parent
DT = 0.040


# ============================================================
# y-mirror augmentation (v90 동일)
# ============================================================
def mirror_seq(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    out[..., 1] *= -1; out[..., 4] *= -1; out[..., 7] *= -1
    return out

def mirror_target(t: np.ndarray) -> np.ndarray:
    out = t.copy(); out[..., 1] *= -1
    return out

def unflip_pred_y(pred: np.ndarray) -> np.ndarray:
    out = pred.copy(); out[..., 1] *= -1
    return out


# ============================================================
# MDN-WTA model
# ============================================================
class BiGRU_MDN(nn.Module):
    """양방향 GRU + K-mixture density head + aux F/W single heads."""
    def __init__(self, n_channels=9, scal_dim=40, hidden=64, fc=128, p=0.2,
                 K=4, aux_dims=(3, 3), main_scale_cm=2.5):
        super().__init__()
        self.K = K
        self.gru = nn.GRU(n_channels, hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden + scal_dim, fc)
        self.fc2 = nn.Linear(fc, fc // 2)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        # K means (3 * K) + K logits
        self.head_means = nn.Linear(fc // 2, K * 3)
        self.head_logits = nn.Linear(fc // 2, K)
        # aux heads (single)
        self.aux_heads = nn.ModuleList([nn.Linear(fc // 2, d) for d in aux_dims])
        self.main_scale = main_scale_cm / 100.0
        # mode-init bias: spread initial K means away from origin to encourage spec
        with torch.no_grad():
            self.head_means.bias.zero_()
            # set bias of each mode to small offsets in different directions
            # K=4: along +x, -x, +y, -y in scaled space (will be * main_scale via tanh)
            offsets = torch.tensor([
                [+0.5, +0.0, +0.0],
                [-0.5, +0.0, +0.0],
                [+0.0, +0.5, +0.0],
                [+0.0, -0.5, +0.0],
            ])
            for k in range(min(K, 4)):
                self.head_means.bias[k*3:(k+1)*3] = offsets[k]

    def forward(self, seq, scal):
        out, _ = self.gru(seq)
        fwd_last = out[:, -1, :out.shape[-1] // 2]
        bwd_first = out[:, 0, out.shape[-1] // 2:]
        h_cat = torch.cat([fwd_last, bwd_first], dim=1)
        z = torch.cat([h_cat, scal], dim=1)
        z = self.act(self.fc1(z)); z = self.drop(z)
        z = self.act(self.fc2(z))
        B = z.shape[0]
        means = torch.tanh(self.head_means(z)).view(B, self.K, 3) * self.main_scale
        logits = self.head_logits(z)
        aux = [h(z) for h in self.aux_heads]
        return means, logits, aux


# ============================================================
# WTA loss combo
# ============================================================
def loss_wta(means, logits, target, ent_w=0.01, wmean_w=0.1):
    """means (B,K,3), logits (B,K), target (B,3).
    WTA: min_k loss_combo(means[k], target).
    + entropy reg on softmax weights to prevent collapse.
    + small loss on weighted mean to keep weights calibrated."""
    B, K, _ = means.shape
    t = target.unsqueeze(1).expand(-1, K, -1)
    d = torch.sqrt(((means - t) ** 2).sum(dim=-1) + 1e-12)  # (B,K)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    combo = d + 0.3 * sh  # (B,K)
    # WTA: select min mode per sample
    min_combo, _ = combo.min(dim=1)  # (B,)
    wta = min_combo.mean()
    # weighted-mean loss
    w = F.softmax(logits, dim=-1)  # (B,K)
    wmean = (w.unsqueeze(-1) * means).sum(dim=1)  # (B,3)
    d_w = torch.sqrt(((wmean - target) ** 2).sum(dim=-1) + 1e-12)
    wmean_loss = d_w.mean()
    # entropy reg (encourage spread)
    log_w = F.log_softmax(logits, dim=-1)
    ent = -(w * log_w).sum(dim=-1).mean()  # max entropy = log(K)
    target_ent = float(np.log(K))
    ent_loss = (target_ent - ent)  # >=0, minimized when uniform
    return wta + wmean_w * wmean_loss + ent_w * ent_loss, wta.item(), wmean_loss.item(), ent.item()


# ============================================================
# K-fold runner
# ============================================================
def run_kfold_mdn(target_main, target_F, target_W,
                   seq_arr, scal_arr, seq_te, scal_te,
                   kalman_train, theta_train, theta_test, y_train,
                   config, n_folds, n_seeds, max_epochs, patience, batch,
                   K, mirror_on=True, lambda_F=0.3, lambda_W=0.3,
                   ent_w=0.01, wmean_w=0.1, device="cpu"):
    N = len(target_main)
    oof_means_rot = np.zeros((N, K, 3), dtype=np.float32)
    oof_logits = np.zeros((N, K), dtype=np.float32)
    fold_mask = np.zeros(N, dtype=bool)
    test_means_per_fold = []
    test_logits_per_fold = []
    fold_rh = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(N)))
    t0 = time.time()

    if mirror_on:
        seq_arr_m = mirror_seq(seq_arr)
        target_main_m = mirror_target(target_main)
        target_F_m = mirror_target(target_F)
        target_W_m = mirror_target(target_W)
        seq_te_m = mirror_seq(seq_te)

    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
        sc_scal = StandardScaler().fit(scal_arr[tr])
        if mirror_on:
            seq_tr_2x = np.concatenate([seq_arr[tr], seq_arr_m[tr]], axis=0)
            scal_tr_2x = np.concatenate([scal_arr[tr], scal_arr[tr]], axis=0)
            tgt_tr_2x = np.concatenate([target_main[tr], target_main_m[tr]], axis=0)
            F_tr_2x = np.concatenate([target_F[tr], target_F_m[tr]], axis=0)
            W_tr_2x = np.concatenate([target_W[tr], target_W_m[tr]], axis=0)
        else:
            seq_tr_2x = seq_arr[tr]; scal_tr_2x = scal_arr[tr]
            tgt_tr_2x = target_main[tr]; F_tr_2x = target_F[tr]
            W_tr_2x = target_W[tr]

        seq_tr_n = normalize_seq(seq_tr_2x, sc_seq)
        scal_tr_n = sc_scal.transform(scal_tr_2x).astype(np.float32)
        seq_va_n = normalize_seq(seq_arr[va], sc_seq)
        scal_va_n = sc_scal.transform(scal_arr[va]).astype(np.float32)
        seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_te_n = sc_scal.transform(scal_te).astype(np.float32)
        if mirror_on:
            seq_va_m_n = normalize_seq(seq_arr_m[va], sc_seq)
            seq_te_m_n = normalize_seq(seq_te_m, sc_seq)

        def T(a): return torch.from_numpy(a).to(device)
        seq_tr_t, scal_tr_t = T(seq_tr_n), T(scal_tr_n)
        tgt_tr_t = T(tgt_tr_2x.astype(np.float32))
        F_tr_t = T(F_tr_2x.astype(np.float32))
        W_tr_t = T(W_tr_2x.astype(np.float32))
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)
        if mirror_on:
            seq_va_m_t = T(seq_va_m_n); seq_te_m_t = T(seq_te_m_n)

        seed_val_means, seed_val_logits = [], []
        seed_test_means, seed_test_logits = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = BiGRU_MDN(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                hidden=config["hidden"], fc=config["fc"], p=config["p"],
                K=K,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh, best_state, no_improve = -1.0, None, 0
            n_tr_eff = seq_tr_t.shape[0]
            for ep in range(max_epochs):
                model.train()
                perm = torch.randperm(n_tr_eff)
                for i in range(0, n_tr_eff, batch):
                    idx = perm[i:i+batch]
                    opt.zero_grad()
                    means, logits, aux = model(seq_tr_t[idx], scal_tr_t[idx])
                    loss_main, _, _, _ = loss_wta(
                        means, logits, tgt_tr_t[idx],
                        ent_w=ent_w, wmean_w=wmean_w,
                    )
                    loss = loss_main
                    loss = loss + lambda_F * loss_euclid(aux[0], F_tr_t[idx])
                    loss = loss + lambda_W * loss_euclid(aux[1], W_tr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                # validation: use weighted mean (mirror TTA if on)
                model.eval()
                with torch.no_grad():
                    m_n, l_n, _ = model(seq_va_t, scal_va_t)
                    if mirror_on:
                        m_m_raw, l_m, _ = model(seq_va_m_t, scal_va_t)
                        # unflip y: means raw is in rotated mirrored frame
                        m_m_np = m_m_raw.cpu().numpy()
                        m_m_np[..., 1] *= -1  # flip y per mode
                        m_n_np = m_n.cpu().numpy()
                        means_tta = (m_n_np + m_m_np) / 2.0  # (B,K,3)
                        # weights: average softmax probs
                        w_n = F.softmax(l_n, dim=-1).cpu().numpy()
                        w_m = F.softmax(l_m, dim=-1).cpu().numpy()
                        w_tta = (w_n + w_m) / 2.0
                    else:
                        means_tta = m_n.cpu().numpy()
                        w_tta = F.softmax(l_n, dim=-1).cpu().numpy()
                    # weighted mean prediction (rotated frame)
                    wm_rot = (w_tta[..., None] * means_tta).sum(axis=1)
                pv = inverse_rotate_xy(wm_rot, theta_train[va])
                pred = kalman_train[va] + pv
                rh = float((np.linalg.norm(pred - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
                if ep == 0 or (ep + 1) % 10 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: rhit={rh:.4f} "
                          f"(best {best_rh:.4f})  [{(time.time()-t0)/60:.1f}m]", flush=True)

            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                m_n, l_n, _ = model(seq_va_t, scal_va_t)
                mt_n, lt_n, _ = model(seq_te_t, scal_te_t)
                if mirror_on:
                    m_m_raw, l_m, _ = model(seq_va_m_t, scal_va_t)
                    mt_m_raw, lt_m, _ = model(seq_te_m_t, scal_te_t)
                    m_n_np = m_n.cpu().numpy(); mt_n_np = mt_n.cpu().numpy()
                    m_m_np = m_m_raw.cpu().numpy(); mt_m_np = mt_m_raw.cpu().numpy()
                    m_m_np[..., 1] *= -1; mt_m_np[..., 1] *= -1
                    w_n_np = F.softmax(l_n, dim=-1).cpu().numpy()
                    wt_n_np = F.softmax(lt_n, dim=-1).cpu().numpy()
                    w_m_np = F.softmax(l_m, dim=-1).cpu().numpy()
                    wt_m_np = F.softmax(lt_m, dim=-1).cpu().numpy()
                    means_va_tta = (m_n_np + m_m_np) / 2.0
                    means_te_tta = (mt_n_np + mt_m_np) / 2.0
                    w_va_tta = (w_n_np + w_m_np) / 2.0
                    w_te_tta = (wt_n_np + wt_m_np) / 2.0
                    # store logits as log of averaged probs (rough but consistent)
                    eps = 1e-12
                    l_va_eff = np.log(w_va_tta + eps)
                    l_te_eff = np.log(w_te_tta + eps)
                else:
                    means_va_tta = m_n.cpu().numpy()
                    means_te_tta = mt_n.cpu().numpy()
                    l_va_eff = l_n.cpu().numpy()
                    l_te_eff = lt_n.cpu().numpy()
            seed_val_means.append(means_va_tta); seed_val_logits.append(l_va_eff)
            seed_test_means.append(means_te_tta); seed_test_logits.append(l_te_eff)
            del model; gc.collect()

        # seed average (rotated frame)
        val_means = np.mean(seed_val_means, axis=0)
        val_logits = np.mean(seed_val_logits, axis=0)
        test_means = np.mean(seed_test_means, axis=0)
        test_logits = np.mean(seed_test_logits, axis=0)
        oof_means_rot[va] = val_means
        oof_logits[va] = val_logits
        fold_mask[va] = True
        test_means_per_fold.append(test_means)
        test_logits_per_fold.append(test_logits)

        # fold report (weighted mean)
        w_val = np.exp(val_logits - val_logits.max(axis=-1, keepdims=True))
        w_val = w_val / w_val.sum(axis=-1, keepdims=True)
        wm_rot = (w_val[..., None] * val_means).sum(axis=1)
        wm_unrot = inverse_rotate_xy(wm_rot, theta_train[va])
        pred_pos = kalman_train[va] + wm_unrot
        rh_fold = float((np.linalg.norm(pred_pos - y_train[va], axis=-1) <= 0.01).mean())
        fold_rh.append(rh_fold)
        print(f"  ★ fold {fi+1}/{n_folds}: R-Hit (wmean)={rh_fold:.4f}  "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)

    # test means averaged across folds (rotated frame). logits averaged in log-space.
    test_means_avg = np.mean(test_means_per_fold, axis=0)
    test_logits_avg = np.mean(test_logits_per_fold, axis=0)

    return oof_means_rot, oof_logits, test_means_avg, test_logits_avg, fold_rh, fold_mask


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--setup", choices=["A", "B"], default="A")
    parser.add_argument("--ent-w", type=float, default=0.01)
    parser.add_argument("--wmean-w", type=float, default=0.1)
    parser.add_argument("--out-tag", default="v109",
                        help="state/csv suffix")
    parser.add_argument("--use-data-mode", default="full")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    suffix = "" if args.setup == "A" else f"_setup{args.setup}"
    state_file = CACHE_DIR / f"{args.out_tag}{suffix}_state.npz"
    sub_file = DATA_DIR / f"submission_{args.out_tag}{suffix}.csv"
    print("=" * 60)
    print(f"v109 BiGRU + MDN-WTA (K={args.K}) + mirror={args.mirror} + TTA")
    print(f"  setup={args.setup}, n_folds={args.n_folds}, n_seeds={args.n_seeds}, "
          f"max_ep={args.max_epochs}, ent_w={args.ent_w}, wmean_w={args.wmean_w}")
    print(f"  state={state_file.name}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, kalman_train_alt = get_kalman(X_train, X_test)

    M = MODE_CONFIGS[args.use_data_mode]
    X_scal_b_tr, X_scal_b_te = get_scalar_feats(X_train, X_test, M, args.use_data_mode)
    tier3_tr = build_tier3(X_train); tier3_te = build_tier3(X_test)
    X_scal_tr = np.concatenate([X_scal_b_tr, tier3_tr], axis=-1)
    X_scal_te = np.concatenate([X_scal_b_te, tier3_te], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)

    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train, theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    if state_file.exists() and not args.force_retrain:
        st = np.load(state_file)
        oof_means_rot = st["oof_means_rot"]
        oof_logits = st["oof_logits"]
        test_means_rot = st["test_means_rot"]
        test_logits = st["test_logits"]
        fold_rh = st["fold_rh"].tolist()
        print(f"[state] cache 로드: fold_rh mean={float(np.mean(fold_rh)):.4f}")
    else:
        if args.setup == "A":
            CONFIG = dict(hidden=64, fc=128, lr=5e-4, p=0.3, wd=1e-4)
        else:
            CONFIG = dict(hidden=64, fc=128, lr=1e-3, p=0.1, wd=1e-4)
        oof_means_rot, oof_logits, test_means_rot, test_logits, fold_rh, mask = \
            run_kfold_mdn(
                target_T8, target_F, target_W,
                seq_tr, X_scal_tr, seq_te, X_scal_te,
                kalman_train, theta_train, theta_test, y_train,
                config=CONFIG, n_folds=args.n_folds, n_seeds=args.n_seeds,
                max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
                K=args.K, mirror_on=args.mirror,
                ent_w=args.ent_w, wmean_w=args.wmean_w, device="cpu",
            )
        np.savez(state_file,
                 oof_means_rot=oof_means_rot, oof_logits=oof_logits,
                 test_means_rot=test_means_rot, test_logits=test_logits,
                 fold_rh=np.array(fold_rh), K=args.K, mirror=args.mirror)
        print(f"[state] 저장: {state_file}")

    # derive variants ----------------------------------------------------
    ALPHA = np.array([1.000, 0.950, 1.000])

    def to_unrot_pred(means_rot, logits, theta, kalman):
        # means_rot: (N,K,3); logits: (N,K) effective log-probs
        w = np.exp(logits - logits.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        # weighted mean (rotated)
        wm_rot = (w[..., None] * means_rot).sum(axis=1)
        wm_unrot = inverse_rotate_xy(wm_rot, theta)
        wm_pred = kalman + wm_unrot * ALPHA[None, :]
        # argmax mode (rotated)
        am = np.argmax(w, axis=-1)
        am_rot = means_rot[np.arange(len(am)), am]
        am_unrot = inverse_rotate_xy(am_rot, theta)
        am_pred = kalman + am_unrot * ALPHA[None, :]
        return wm_pred, am_pred, w

    oof_wm, oof_am, oof_w = to_unrot_pred(oof_means_rot, oof_logits, theta_train, kalman_train)
    test_wm, test_am, test_w = to_unrot_pred(test_means_rot, test_logits, theta_test, kalman_test)

    rh_wm = float((np.linalg.norm(oof_wm - y_train, axis=-1) <= 0.01).mean())
    rh_am = float((np.linalg.norm(oof_am - y_train, axis=-1) <= 0.01).mean())
    print(f"\n[v109 K={args.K}] OOF weighted_mean: {rh_wm:.4f}")
    print(f"[v109 K={args.K}] OOF argmax_mode:   {rh_am:.4f}")
    # oracle (cheat) for paradigm diversity check
    d_all = np.linalg.norm(
        inverse_rotate_xy(oof_means_rot.reshape(-1, 3), np.repeat(theta_train, args.K))
        .reshape(-1, args.K, 3) * ALPHA[None, None, :]
        + kalman_train[:, None, :] - y_train[:, None, :],
        axis=-1)
    oracle = float((d_all.min(axis=-1) <= 0.01).mean())
    print(f"[v109 K={args.K}] OOF oracle (best mode): {oracle:.4f}")
    print(f"  weight entropy mean: {(-oof_w * np.log(oof_w + 1e-12)).sum(-1).mean():.3f} "
          f"(uniform = {float(np.log(args.K)):.3f})")
    print(f"  weight dominance (max w mean): {oof_w.max(-1).mean():.3f}")

    # save submission for weighted_mean (default)
    pd.DataFrame({"id": sub["id"],
                  "x": test_wm[:,0], "y": test_wm[:,1], "z": test_wm[:,2]}
                 ).to_csv(sub_file, index=False)
    print(f"  [submission] {sub_file.name}")
    # also save argmax variant
    sub_am = DATA_DIR / f"submission_{args.out_tag}{suffix}_argmax.csv"
    pd.DataFrame({"id": sub["id"],
                  "x": test_am[:,0], "y": test_am[:,1], "z": test_am[:,2]}
                 ).to_csv(sub_am, index=False)
    print(f"  [submission] {sub_am.name}")

    # save oof_wm / test_wm and oof_am / test_am as separate npz for ensemble pool
    pool_file = CACHE_DIR / f"{args.out_tag}{suffix}_pool.npz"
    np.savez(pool_file,
             oof_wm=oof_wm, test_wm=test_wm, rh_wm=rh_wm,
             oof_am=oof_am, test_am=test_am, rh_am=rh_am)
    print(f"[pool] 저장: {pool_file.name}")

    entry = {
        "version": f"{args.out_tag}{suffix}",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"BiGRU + MDN-WTA K={args.K} + mirror={args.mirror} + TTA",
        "K": int(args.K), "ent_w": float(args.ent_w), "wmean_w": float(args.wmean_w),
        "n_folds": args.n_folds, "n_seeds": args.n_seeds, "max_epochs": args.max_epochs,
        "fold_rh_mean": float(np.mean(fold_rh)),
        "rh_wm": float(rh_wm), "rh_am": float(rh_am), "rh_oracle": float(oracle),
    }
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
