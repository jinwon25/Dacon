"""v118_aug_hit.py — BiGRU + random yaw aug + 50% y-flip + hit-aware loss + 1-3cm band weight.

STEP 3 (D-7 sprint) decorrelated 새 paradigm 카드.

설계:
  - v77 BiGRU 아키텍처 (Bidirectional GRU + Kalman residual + F + W aux)
  - canonical frame + 변위 target은 v23/v77 framework 상속 (이미 채택)
  - 신규 1: 매 batch 마다 sample별 random yaw φ∈[0,2π) + 50% y-flip
            (canonical 위에 추가 회전 → rotation/reflection equivariance 강화)
  - 신규 2: hit-aware loss = MAE + λ·soft_hit
            soft_hit = mean(sigmoid((d - 0.01)/τ)),  τ: 0.01 → 0.003 (linear schedule)
  - 신규 3: 1-3cm 밴드 sample-weight ×2.5
            밴드 판정: min(||CV_pred - y||, ||CA_pred - y||) ∈ [0.01, 0.03]
            (CV: x[-1]+v_last·2dt, CA: x[-1]+v_last·2dt+0.5·a_last·(2dt)²)

게이트 (fold0 standalone):
  - OOF R-Hit ≥ 0.665  AND  residual corr (v112_v107_diverse) < 0.93
  - PASS → STEP4 5-fold 완주

사용법:
  python scripts/v118_aug_hit.py --fold 0 --setup A --max-epochs 150
  python scripts/v118_aug_hit.py --fold 0 --setup A --smoke   # 빠른 sanity (epochs=5, 1000 sample)
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
)
from v77_bigru import BiGRUMultiAux

PROJECT = SCRIPT_DIR.parent
REPORTS = PROJECT / "docs/reports"; REPORTS.mkdir(exist_ok=True)
DT = 0.040
TWOPI = float(2 * np.pi)


# ============================================================
# CV/CA extrapolation → 1-3cm band mask
# ============================================================
def cv_ca_predict(X):
    """X shape (N, 11, 3). Return (cv_pred, ca_pred) both (N, 3) at +80ms ahead from last point."""
    v_last = (X[:, -1] - X[:, -2]) / DT                    # (N,3)
    v_prev = (X[:, -2] - X[:, -3]) / DT                    # (N,3)
    a_last = (v_last - v_prev) / DT                        # (N,3)
    dt2 = 2.0 * DT                                          # 80ms
    cv = X[:, -1] + v_last * dt2
    ca = X[:, -1] + v_last * dt2 + 0.5 * a_last * (dt2 ** 2)
    return cv, ca


def band_weight(X, y_true, lo=0.01, hi=0.03, weight=2.5):
    """1-3cm 밴드 샘플 weight ×<weight>, 그 외 ×1.0. CV/CA 외삽 오차 중 min 사용."""
    cv, ca = cv_ca_predict(X)
    err_cv = np.linalg.norm(cv - y_true, axis=-1)
    err_ca = np.linalg.norm(ca - y_true, axis=-1)
    err = np.minimum(err_cv, err_ca)
    mask = (err >= lo) & (err <= hi)
    w = np.where(mask, weight, 1.0).astype(np.float32)
    return w, mask, err


def fast_turn_mask(X, speed_thr=1.0, turn_cos_thr=0.5):
    """fast(|v_last|>1.0) AND sharp turn(turn_cos<0.5) 부분집합."""
    v_last = (X[:, -1] - X[:, -2]) / DT
    sp_last = np.linalg.norm(v_last, axis=-1)
    # turn_cos: cos angle between v_last and mean v in prev window
    disp = np.diff(X, axis=1); v = disp / DT
    v_prev_mean = v[:, :-1, :].mean(axis=1)
    na = np.linalg.norm(v_last, axis=-1); nb = np.linalg.norm(v_prev_mean, axis=-1)
    turn = np.clip((v_last * v_prev_mean).sum(-1) / np.maximum(na * nb, 1e-12), -1, 1)
    return (sp_last > speed_thr) & (turn < turn_cos_thr)


# ============================================================
# Batch-level random yaw + 50% y-flip on normalized seq + raw targets
# ============================================================
def aug_batch(seq, *targets, device):
    """seq (B,T,9) channels [rel_xyz, v_xyz, a_xyz]. targets (B,3) each in canonical.
    Returns (seq_aug, target_aug...) all with same random φ + flip per sample."""
    B = seq.shape[0]
    phi = torch.rand(B, device=device) * TWOPI                       # (B,)
    flip = (torch.rand(B, device=device) < 0.5).float() * 2 - 1      # (B,) ±1
    cosp = torch.cos(phi); sinp = torch.sin(phi)                     # (B,)
    cosp_e = cosp.unsqueeze(-1).unsqueeze(-1)                        # (B,1,1)
    sinp_e = sinp.unsqueeze(-1).unsqueeze(-1)
    flip_e = flip.unsqueeze(-1).unsqueeze(-1)

    seq_aug = seq.clone()
    for ix, iy in [(0, 1), (3, 4), (6, 7)]:
        x = seq[..., ix:ix+1]; y = seq[..., iy:iy+1]
        seq_aug[..., ix:ix+1] = cosp_e * x + sinp_e * y
        seq_aug[..., iy:iy+1] = (-sinp_e * x + cosp_e * y) * flip_e

    out = [seq_aug]
    for t in targets:
        tx = t[..., 0]; ty = t[..., 1]; tz = t[..., 2]
        nx = cosp * tx + sinp * ty
        ny = (-sinp * tx + cosp * ty) * flip
        out.append(torch.stack([nx, ny, tz], dim=-1))
    return tuple(out)


# ============================================================
# Hit-aware loss with τ schedule + sample weight
# ============================================================
def hit_aware_loss(pred, tgt, weight, lambda_hit, tau):
    """MAE + λ·soft_hit, soft_hit = sigmoid((d-0.01)/τ).  weight: (B,) sample weights."""
    # MAE per sample (mean over 3 coords)
    mae = torch.abs(pred - tgt).mean(dim=-1)                         # (B,)
    d = torch.sqrt(((pred - tgt) ** 2).sum(dim=-1) + 1e-12)          # (B,)
    sh = torch.sigmoid((d - 0.01) / tau)                              # (B,)
    return ((mae + lambda_hit * sh) * weight).sum() / weight.sum().clamp(min=1.0)


def euclid_weighted(pred, tgt, weight):
    d = torch.sqrt(((pred - tgt) ** 2).sum(dim=-1) + 1e-12)
    return (d * weight).sum() / weight.sum().clamp(min=1.0)


# ============================================================
# Single-fold training
# ============================================================
def train_one_fold(
    fold_i, tr_idx, va_idx,
    target_main, target_F, target_W,
    seq_arr, scal_arr, seq_te, scal_te,
    kalman_train, theta_train, theta_test, y_train,
    sample_weight, X_train_raw,
    config, max_epochs, patience, batch,
    lambda_F=0.3, lambda_W=0.3,
    lambda_hit=0.3, tau_start=0.01, tau_end=0.003,
    aug_on=True, device="cpu",
):
    sc_seq = StandardScaler().fit(seq_arr[tr_idx].reshape(-1, seq_arr.shape[2]))
    sc_scal = StandardScaler().fit(scal_arr[tr_idx])
    seq_tr_n = normalize_seq(seq_arr[tr_idx], sc_seq)
    seq_va_n = normalize_seq(seq_arr[va_idx], sc_seq)
    seq_te_n = normalize_seq(seq_te, sc_seq)
    scal_tr_n = sc_scal.transform(scal_arr[tr_idx]).astype(np.float32)
    scal_va_n = sc_scal.transform(scal_arr[va_idx]).astype(np.float32)
    scal_te_n = sc_scal.transform(scal_te).astype(np.float32)

    def T(a): return torch.from_numpy(a).to(device)
    seq_tr_t, scal_tr_t = T(seq_tr_n), T(scal_tr_n)
    tgt_tr_t = T(target_main[tr_idx].astype(np.float32))
    F_tr_t = T(target_F[tr_idx].astype(np.float32))
    W_tr_t = T(target_W[tr_idx].astype(np.float32))
    w_tr_t = T(sample_weight[tr_idx].astype(np.float32))
    seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
    seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)

    torch.manual_seed(config.get("seed", 0)); np.random.seed(config.get("seed", 0))
    model = BiGRUMultiAux(
        n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
        hidden=config["hidden"], fc=config["fc"], p=config["p"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_rh, best_state, no_improve = -1.0, None, 0
    n_tr = seq_tr_t.shape[0]
    t0 = time.time()
    history = []

    for ep in range(max_epochs):
        tau = tau_start + (tau_end - tau_start) * (ep / max(max_epochs - 1, 1))
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch):
            idx = perm[i:i+batch]
            seq_b = seq_tr_t[idx]; scal_b = scal_tr_t[idx]
            tgt_b = tgt_tr_t[idx]; F_b = F_tr_t[idx]; W_b = W_tr_t[idx]
            w_b = w_tr_t[idx]
            if aug_on:
                seq_b, tgt_b, F_b, W_b = aug_batch(seq_b, tgt_b, F_b, W_b, device=device)
            opt.zero_grad()
            out_main, outs_aux = model(seq_b, scal_b)
            loss = hit_aware_loss(out_main, tgt_b, w_b, lambda_hit, tau)
            loss = loss + lambda_F * euclid_weighted(outs_aux[0], F_b, w_b)
            loss = loss + lambda_W * euclid_weighted(outs_aux[1], W_b, w_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out_va_rot, _ = model(seq_va_t, scal_va_t)
            out_va_rot = out_va_rot.cpu().numpy()
        out_va = inverse_rotate_xy(out_va_rot, theta_train[va_idx])
        pred = kalman_train[va_idx] + out_va
        rh = float((np.linalg.norm(pred - y_train[va_idx], axis=-1) <= 0.01).mean())
        history.append((ep + 1, float(tau), rh))
        if rh > best_rh:
            best_rh = rh
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  fold{fold_i} ep{ep+1:3d}: τ={tau:.4f}  rhit={rh:.4f} (best {best_rh:.4f})  "
                  f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pv_rot, _ = model(seq_va_t, scal_va_t)
        pt_rot, _ = model(seq_te_t, scal_te_t)
    pv = inverse_rotate_xy(pv_rot.cpu().numpy(), theta_train[va_idx])
    pt = inverse_rotate_xy(pt_rot.cpu().numpy(), theta_test)
    pred = kalman_train[va_idx] + pv
    rh_fold = float((np.linalg.norm(pred - y_train[va_idx], axis=-1) <= 0.01).mean())
    print(f"  ★ fold{fold_i}: best R-Hit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
    return pv, pt, rh_fold, history


# ============================================================
# Residual correlation with v112_v107_diverse OOF
# ============================================================
def residual_corr_report(new_oof_full, va_idx, y_train):
    """Compute per-axis + 3D residual correlation between new model and v112_v107_diverse on va fold."""
    p = CACHE_DIR / "v112_v107_diverse_weights.npz"
    if not p.exists():
        return None
    v112 = np.load(p)["oof_pred"]   # (10000, 3)
    new = new_oof_full              # (10000, 3) — only va_idx filled
    r_new = (new[va_idx] - y_train[va_idx])      # (Nva, 3)
    r_v112 = (v112[va_idx] - y_train[va_idx])    # (Nva, 3)
    out = {}
    for ax, name in enumerate("xyz"):
        a = r_new[:, ax]; b = r_v112[:, ax]
        c = float(np.corrcoef(a, b)[0, 1])
        out[f"corr_{name}"] = c
    # 3D residual magnitude correlation
    d_new = np.linalg.norm(r_new, axis=-1)
    d_v112 = np.linalg.norm(r_v112, axis=-1)
    out["corr_3d_mag"] = float(np.corrcoef(d_new, d_v112)[0, 1])
    # cosine sim of signed residual vectors (mean over samples)
    n_new = r_new / np.maximum(np.linalg.norm(r_new, axis=-1, keepdims=True), 1e-12)
    n_v112 = r_v112 / np.maximum(np.linalg.norm(r_v112, axis=-1, keepdims=True), 1e-12)
    out["cos_sim_mean"] = float((n_new * n_v112).sum(-1).mean())
    return out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="0..4")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--setup", choices=["A", "B"], default="A")
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lambda-hit", type=float, default=0.3)
    ap.add_argument("--tau-start", type=float, default=0.01)
    ap.add_argument("--tau-end", type=float, default=0.003)
    ap.add_argument("--band-weight", type=float, default=2.5)
    ap.add_argument("--no-band", action="store_true")
    ap.add_argument("--no-aug", action="store_true", help="disable random yaw + flip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="micro sanity (1000 sample, 5 epoch)")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_num_threads(os.cpu_count() or 4)

    setup_cfg = {
        "A": dict(hidden=64, fc=128, lr=5e-4, p=0.3, wd=1e-4, seed=args.seed),
        "B": dict(hidden=64, fc=128, lr=1e-3, p=0.1, wd=1e-4, seed=args.seed),
    }[args.setup]

    tag = f"fold{args.fold}_setup{args.setup}"
    if args.no_aug: tag += "_noaug"
    if args.no_band: tag += "_noband"
    if args.smoke: tag += "_smoke"
    state_file = CACHE_DIR / f"v118_aug_hit_{tag}.npz"
    report_file = REPORTS / f"v118_aug_hit_{tag}.md"

    print("=" * 60)
    print(f"v118 aug+hit  device={device}  setup={args.setup}  fold={args.fold}/{args.n_folds}")
    print(f"  aug={'on' if not args.no_aug else 'OFF'}  band={'on' if not args.no_band else 'OFF'}")
    print(f"  λ_hit={args.lambda_hit}  τ:{args.tau_start}→{args.tau_end}  "
          f"band_w={args.band_weight}")
    print(f"  state={state_file.name}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, kalman_train_alt = get_kalman(X_train, X_test)
    M = MODE_CONFIGS["full"]
    X_scal_b_tr, X_scal_b_te = get_scalar_feats(X_train, X_test, M, "full")
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

    # sample weights from CV/CA error band
    if args.no_band:
        sw = np.ones(len(y_train), dtype=np.float32); band_mask = np.zeros(len(y_train), dtype=bool)
    else:
        sw, band_mask, err = band_weight(X_train, y_train, lo=0.01, hi=0.03, weight=args.band_weight)
    print(f"  band(1-3cm) sample n={int(band_mask.sum())}/{len(y_train)} "
          f"({100*band_mask.mean():.1f}%)  →  weight ×{args.band_weight}")

    # fast+sharp-turn subset on TRAIN
    ft_train = fast_turn_mask(X_train)
    print(f"  fast+turn (train) n={int(ft_train.sum())}/{len(y_train)} "
          f"({100*ft_train.mean():.2f}%)")

    # ---- smoke mode reductions ----
    if args.smoke:
        keep = np.arange(min(1000, len(y_train)))
        Xk = X_train[keep]; yk = y_train[keep]
        seqk = seq_tr[keep]; scalk = X_scal_tr[keep]
        kt = kalman_train[keep]; tt = theta_train[keep]
        tm = target_T8[keep]; tF = target_F[keep]; tW = target_W[keep]
        swk = sw[keep]
        # 5-fold on subset, take fold0
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        splits = list(kf.split(np.arange(len(keep))))
        tr_idx, va_idx = splits[args.fold]
        max_ep = 5; patience = max_ep
        pv, pt, rh, hist = train_one_fold(
            args.fold, tr_idx, va_idx,
            tm, tF, tW, seqk, scalk, seq_te, X_scal_te,
            kt, tt, theta_test, yk,
            swk, Xk, setup_cfg, max_ep, patience, args.batch,
            lambda_hit=args.lambda_hit, tau_start=args.tau_start, tau_end=args.tau_end,
            aug_on=not args.no_aug, device=device,
        )
        print(f"\n[SMOKE] fold{args.fold} R-Hit={rh:.4f}  (sample={len(keep)}, ep≤{max_ep})")
        return

    # ---- full-data single-fold ----
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    splits = list(kf.split(np.arange(len(y_train))))
    tr_idx, va_idx = splits[args.fold]

    if state_file.exists() and not args.force_retrain:
        st = np.load(state_file)
        pv = st["va_pred_disp"]; pt = st["test_pred_disp"]
        va_idx = st["va_idx"]; rh_fold = float(st["rh_fold"])
        print(f"[state] cache 로드: fold R-Hit={rh_fold:.4f}")
    else:
        pv, pt, rh_fold, history = train_one_fold(
            args.fold, tr_idx, va_idx,
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            sw, X_train, setup_cfg, args.max_epochs, args.patience, args.batch,
            lambda_hit=args.lambda_hit, tau_start=args.tau_start, tau_end=args.tau_end,
            aug_on=not args.no_aug, device=device,
        )
        np.savez(
            state_file,
            va_pred_disp=pv, test_pred_disp=pt,
            va_idx=va_idx, rh_fold=rh_fold,
            band_mask=band_mask, sample_weight=sw,
            history=np.array(history),
        )
        print(f"[state] 저장: {state_file}")

    # build full-size OOF (only va slot filled)
    oof_full = np.zeros_like(y_train, dtype=np.float32)
    pred_va = kalman_train[va_idx] + pv
    oof_full[va_idx] = pred_va.astype(np.float32)

    # ---- gate metrics ----
    corr = residual_corr_report(oof_full, va_idx, y_train) or {}
    err_va = np.linalg.norm(pred_va - y_train[va_idx], axis=-1)
    rh_va = float((err_va <= 0.01).mean())
    # band breakdown on va
    band_va = band_mask[va_idx]
    rh_in_band = float((err_va[band_va] <= 0.01).mean()) if band_va.any() else 0.0
    rh_out_band = float((err_va[~band_va] <= 0.01).mean()) if (~band_va).any() else 0.0
    # fast+turn subset on va
    ft_va = ft_train[va_idx]
    rh_ft_va = float((err_va[ft_va] <= 0.01).mean()) if ft_va.any() else 0.0

    # residual corr with v77 BiGRU and v90 mirror for triangulation
    extra_corr = {}
    for label, fname, key in [
        ("v77_A", "v77_bigru_state.npz", "oof_A"),
        ("v90_mirror", "v90_mirror_state.npz", "oof"),
    ]:
        p = CACHE_DIR / fname
        if not p.exists(): continue
        ref_disp = np.load(p)[key]  # canonical-frame displacement (oof)
        ref_pred = kalman_train + inverse_rotate_xy(ref_disp, theta_train)  # full
        ref_va = ref_pred[va_idx]
        r_ref = ref_va - y_train[va_idx]
        r_new = pred_va - y_train[va_idx]
        for ax, name in enumerate("xyz"):
            extra_corr[f"{label}_corr_{name}"] = float(np.corrcoef(r_new[:, ax], r_ref[:, ax])[0, 1])
        d_ref = np.linalg.norm(r_ref, axis=-1); d_new = np.linalg.norm(r_new, axis=-1)
        extra_corr[f"{label}_corr_3d_mag"] = float(np.corrcoef(d_new, d_ref)[0, 1])

    # ---- report ----
    gate_oof = rh_fold >= 0.665
    gate_corr_v112 = corr.get("corr_3d_mag", 1.0) < 0.93
    passed = gate_oof and gate_corr_v112

    lines = []
    lines.append(f"# v118 aug+hit — fold{args.fold} setup{args.setup} ({_dt.datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("## 설정")
    lines.append(f"- aug: random yaw [0,2π) + 50% y-flip = **{'ON' if not args.no_aug else 'OFF'}**")
    lines.append(f"- band weight: ×{args.band_weight} on CV/CA-err ∈ [1cm, 3cm], n={int(band_mask.sum())}")
    lines.append(f"- λ_hit={args.lambda_hit}, τ schedule {args.tau_start}→{args.tau_end}")
    lines.append(f"- setup={args.setup}: hidden={setup_cfg['hidden']}, fc={setup_cfg['fc']}, "
                 f"lr={setup_cfg['lr']}, p={setup_cfg['p']}, wd={setup_cfg['wd']}, seed={setup_cfg['seed']}")
    lines.append(f"- max_epochs={args.max_epochs}, patience={args.patience}, batch={args.batch}")
    lines.append("")
    lines.append("## 결과 (fold0 va)")
    lines.append(f"- **standalone OOF R-Hit (va): {rh_fold:.4f}**")
    lines.append(f"  - in-band(1-3cm) R-Hit: {rh_in_band:.4f} (n={int(band_va.sum())})")
    lines.append(f"  - out-band R-Hit: {rh_out_band:.4f}")
    lines.append(f"  - fast+turn subset R-Hit (va): {rh_ft_va:.4f} (n={int(ft_va.sum())})")
    lines.append("")
    lines.append("## Residual correlation (va fold)")
    lines.append("### vs v112_v107_diverse")
    for k, v in (corr or {}).items():
        lines.append(f"- {k}: {v:.4f}")
    if extra_corr:
        lines.append("### vs v77 / v90 (참고)")
        for k, v in extra_corr.items():
            lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Gate")
    lines.append(f"- OOF ≥ 0.665: {'✅' if gate_oof else '❌'} ({rh_fold:.4f})")
    lines.append(f"- corr_3d_mag(v112) < 0.93: {'✅' if gate_corr_v112 else '❌'} "
                 f"({corr.get('corr_3d_mag', float('nan')):.4f})")
    lines.append(f"- **{'PASS → STEP 4 진행' if passed else 'FAIL → STEP 4 skip, v117/v112 마감'}**")
    report_file.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[report] {report_file}")


if __name__ == "__main__":
    main()
