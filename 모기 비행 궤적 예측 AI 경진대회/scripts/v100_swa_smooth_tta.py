"""v100_swa_smooth_tta.py — v90 framework + SWA + smooth R-Hit loss curriculum + input-noise TTA.

연구 결과 기반:
  - SWA (Izmailov 2018): 마지막 30% epoch 매 epoch weight 평균 → wider optima
  - Smooth R-Hit loss curriculum (MetricOpt, Smooth Sigmoid Surrogate):
    초기 smooth β (loss flow) → 후기 sharp β (boundary focus)
  - Input-noise TTA: Gaussian jitter σ=1mm × N=8 inferences, median pool

이전 v90 mirror (setup A) OOF 0.6643 → 목표 v100 OOF 0.665~0.670
v98 5w LB 0.6882 → v100 추가로 + paradigm diversity 활용 시 0.690+ 가능성

별도 cache: v100_state.npz
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
    loss_euclid,
)
from v77_bigru import BiGRUMultiAux
from v90_yaw_mirror_aug import mirror_seq, mirror_target, unflip_pred_y

PROJECT = SCRIPT_DIR.parent
DT = 0.040


# ============================================================
# Smooth R-Hit loss with curriculum β
# ============================================================
def smooth_rhit_loss(pred, true, beta):
    """sigmoid surrogate of 1{‖pred-true‖ ≤ 0.01}. β 작을수록 sharper.
    L = 1 - sigmoid((r - d) / β) = sigmoid((d - r) / β) (miss probability)"""
    d = torch.sqrt(((pred - true) ** 2).sum(dim=-1) + 1e-12)
    return torch.sigmoid((d - 0.01) / beta).mean()


def loss_combo_curr(pred, true, beta=0.002, w_euc=1.0, w_hit=0.3):
    return w_euc * loss_euclid(pred, true) + w_hit * smooth_rhit_loss(pred, true, beta)


def get_curriculum(ep, max_epochs):
    """β anneal: 0.006 → 0.002 (smooth → sharp), w_hit: 0.3 → 1.5 (aggressive metric focus)"""
    frac = ep / max_epochs
    # β: exponential decay from 0.006 to 0.002
    beta = 0.006 * (0.002 / 0.006) ** frac
    # w_hit ramp: 0.3 (ep 0~25%) → 1.5 (ep 75~100%)
    if frac < 0.25:
        w_hit = 0.3
    elif frac < 0.75:
        w_hit = 0.3 + (1.5 - 0.3) * (frac - 0.25) * 2  # 0.3 → 1.5 over 25~75%
    else:
        w_hit = 1.5
    return float(beta), float(w_hit)


# ============================================================
# K-fold runner: mirror aug + SWA + TTA (mirror + input-noise)
# ============================================================
def run_kfold(target_main, target_F, target_W,
              seq_arr, scal_arr, seq_te, scal_te,
              kalman_train, theta_train, theta_test, y_train,
              config, n_folds, n_seeds, max_epochs, patience, batch,
              swa_start_frac=0.5, swa_period=1, tta_n=8, tta_sigma_mm=1.0,
              swa_enabled=True, smooth_enabled=True, noise_tta_enabled=False,
              device="cpu"):
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold = []; fold_rh = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
    t0 = time.time()

    # mirror pre-compute
    seq_arr_m = mirror_seq(seq_arr)
    target_main_m = mirror_target(target_main)
    target_F_m = mirror_target(target_F)
    target_W_m = mirror_target(target_W)
    seq_te_m = mirror_seq(seq_te)

    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
        sc_scal = StandardScaler().fit(scal_arr[tr])

        # 2x train data (mirror)
        seq_tr_2x = np.concatenate([seq_arr[tr], seq_arr_m[tr]], axis=0)
        scal_tr_2x = np.concatenate([scal_arr[tr], scal_arr[tr]], axis=0)
        tgt_tr_2x = np.concatenate([target_main[tr], target_main_m[tr]], axis=0)
        F_tr_2x = np.concatenate([target_F[tr], target_F_m[tr]], axis=0)
        W_tr_2x = np.concatenate([target_W[tr], target_W_m[tr]], axis=0)

        seq_tr_n = normalize_seq(seq_tr_2x, sc_seq)
        scal_tr_n = sc_scal.transform(scal_tr_2x).astype(np.float32)
        seq_va_n = normalize_seq(seq_arr[va], sc_seq)
        scal_va_n = sc_scal.transform(scal_arr[va]).astype(np.float32)
        seq_te_n = normalize_seq(seq_te, sc_seq)
        scal_te_n = sc_scal.transform(scal_te).astype(np.float32)
        seq_va_m_n = normalize_seq(seq_arr_m[va], sc_seq)
        seq_te_m_n = normalize_seq(seq_te_m, sc_seq)

        # for TTA, save sc_seq mean/std to add jitter in raw scale
        seq_mean = sc_seq.mean_.astype(np.float32)
        seq_std = np.sqrt(sc_seq.var_).astype(np.float32)

        def T(a): return torch.from_numpy(a).to(device)
        seq_tr_t, scal_tr_t = T(seq_tr_n), T(scal_tr_n)
        tgt_tr_t = T(tgt_tr_2x.astype(np.float32))
        F_tr_t = T(F_tr_2x.astype(np.float32))
        W_tr_t = T(W_tr_2x.astype(np.float32))
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)
        seq_va_m_t = T(seq_va_m_n); seq_te_m_t = T(seq_te_m_n)
        seq_va_raw = seq_arr[va]; seq_te_raw = seq_te  # for noise TTA in raw frame

        seed_val, seed_test = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = BiGRUMultiAux(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                hidden=config["hidden"], fc=config["fc"], p=config["p"],
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
            best_rh, best_state, no_improve = -1.0, None, 0
            n_tr_eff = seq_tr_t.shape[0]

            # SWA tracking
            swa_start_ep = int(max_epochs * swa_start_frac)
            swa_states = []  # list of state_dicts

            for ep in range(max_epochs):
                model.train()
                perm = torch.randperm(n_tr_eff)
                if smooth_enabled:
                    beta, w_hit = get_curriculum(ep, max_epochs)
                else:
                    beta, w_hit = 0.002, 0.3  # v90 default
                for i in range(0, n_tr_eff, batch):
                    idx = perm[i:i+batch]
                    opt.zero_grad()
                    out_main, outs_aux = model(seq_tr_t[idx], scal_tr_t[idx])
                    loss = loss_combo_curr(out_main, tgt_tr_t[idx], beta=beta, w_hit=w_hit)
                    loss = loss + 0.3 * loss_euclid(outs_aux[0], F_tr_t[idx])
                    loss = loss + 0.3 * loss_euclid(outs_aux[1], W_tr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                # SWA: state 저장 (매 epoch, swa_start_ep 이후)
                if swa_enabled and ep >= swa_start_ep:
                    swa_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})

                # Validation (mirror TTA만)
                model.eval()
                with torch.no_grad():
                    pv_normal, _ = model(seq_va_t, scal_va_t)
                    pv_mirror_raw, _ = model(seq_va_m_t, scal_va_t)
                    pv = (pv_normal.cpu().numpy() + unflip_pred_y(pv_mirror_raw.cpu().numpy())) / 2
                pv_inv = inverse_rotate_xy(pv, theta_train[va])
                pred = kalman_train[va] + pv_inv
                rh = float((np.linalg.norm(pred - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                # early stop: SWA 활성 시 SWA 충분 모일 때까지 비활성, 그 외에는 patience 적용
                if swa_enabled:
                    # SWA 활성 → max_epochs까지 학습 (early stop OFF)
                    pass
                else:
                    if no_improve >= patience: break
                if ep == 0 or (ep + 1) % 10 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: rhit={rh:.4f} (best {best_rh:.4f}) "
                          f"β={beta:.4f} w_hit={w_hit:.2f} swa={len(swa_states)}  "
                          f"[{(time.time()-t0)/60:.1f}m]", flush=True)

            # SWA: average all collected states (NN params만, BN buffer 등은 best_state 유지)
            if swa_enabled and len(swa_states) >= 5:
                avg_state = {}
                for k in swa_states[0]:
                    if swa_states[0][k].dtype.is_floating_point:
                        avg_state[k] = torch.stack([s[k].float() for s in swa_states]).mean(dim=0)
                        if swa_states[0][k].dtype != torch.float32:
                            avg_state[k] = avg_state[k].to(swa_states[0][k].dtype)
                    else:
                        avg_state[k] = swa_states[-1][k]
                model.load_state_dict(avg_state)
                # SWA + best_state 비교, 더 좋은 거 사용
                with torch.no_grad():
                    pv_normal_swa, _ = model(seq_va_t, scal_va_t)
                    pv_mirror_swa_raw, _ = model(seq_va_m_t, scal_va_t)
                    pv_swa = (pv_normal_swa.cpu().numpy() + unflip_pred_y(pv_mirror_swa_raw.cpu().numpy())) / 2
                pv_inv_swa = inverse_rotate_xy(pv_swa, theta_train[va])
                pred_swa = kalman_train[va] + pv_inv_swa
                rh_swa = float((np.linalg.norm(pred_swa - y_train[va], axis=-1) <= 0.01).mean())
                if rh_swa > best_rh:
                    print(f"  fold{fi+1} seed{s} SWA win: {rh_swa:.4f} > best {best_rh:.4f} ({len(swa_states)} ckpts)")
                else:
                    print(f"  fold{fi+1} seed{s} SWA lose: {rh_swa:.4f} < best {best_rh:.4f}, fallback")
                    model.load_state_dict(best_state)
            else:
                model.load_state_dict(best_state)
            model.eval()

            # Final inference: mirror TTA + input-noise TTA
            def infer(seq_t, scal_t):
                with torch.no_grad():
                    out, _ = model(seq_t, scal_t)
                return out.cpu().numpy()

            # mirror TTA (val, test)
            pv_normal = infer(seq_va_t, scal_va_t)
            pv_mirror = unflip_pred_y(infer(seq_va_m_t, scal_va_t))
            pt_normal = infer(seq_te_t, scal_te_t)
            pt_mirror = unflip_pred_y(infer(seq_te_m_t, scal_te_t))
            pv_mirror_avg = (pv_normal + pv_mirror) / 2
            pt_mirror_avg = (pt_normal + pt_mirror) / 2

            if noise_tta_enabled:
                # input-noise TTA: σ=tta_sigma_mm gaussian on position channels (0,1,2)
                sigma = tta_sigma_mm / 1000.0  # to meters
                pv_noise_list = [pv_mirror_avg]
                pt_noise_list = [pt_mirror_avg]
                rng = np.random.RandomState(s * 1000 + fi)
                for n in range(tta_n):
                    noise_va = rng.normal(0, sigma, size=seq_va_raw[..., :3].shape).astype(np.float32)
                    noise_te = rng.normal(0, sigma, size=seq_te_raw[..., :3].shape).astype(np.float32)
                    seq_va_noisy = seq_va_raw.copy(); seq_va_noisy[..., :3] += noise_va
                    seq_te_noisy = seq_te_raw.copy(); seq_te_noisy[..., :3] += noise_te
                    seq_va_noisy_n = normalize_seq(seq_va_noisy, sc_seq)
                    seq_te_noisy_n = normalize_seq(seq_te_noisy, sc_seq)
                    pv_noise_list.append(infer(T(seq_va_noisy_n), scal_va_t))
                    pt_noise_list.append(infer(T(seq_te_noisy_n), scal_te_t))
                pv = np.median(np.stack(pv_noise_list, axis=0), axis=0)
                pt = np.median(np.stack(pt_noise_list, axis=0), axis=0)
            else:
                pv = pv_mirror_avg
                pt = pt_mirror_avg

            seed_val.append(pv); seed_test.append(pt)
            del model; gc.collect()

        val_mean_rot = np.mean(seed_val, axis=0)
        test_mean_rot = np.mean(seed_test, axis=0)
        oof_rot[va] = val_mean_rot
        fold_mask[va] = True
        test_per_fold.append(test_mean_rot)
        val_unrot = inverse_rotate_xy(val_mean_rot, theta_train[va])
        pred_pos = kalman_train[va] + val_unrot
        rh_fold = float((np.linalg.norm(pred_pos - y_train[va], axis=-1) <= 0.01).mean())
        fold_rh.append(rh_fold)
        print(f"  ★ fold {fi+1}/{n_folds}: R-Hit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    oof_unrot_all = np.zeros_like(target_main)
    oof_unrot_all[fold_mask] = inverse_rotate_xy(oof_rot[fold_mask], theta_train[fold_mask])
    pred = kalman_train[fold_mask] + oof_unrot_all[fold_mask]
    oof_rhit = float((np.linalg.norm(pred - y_train[fold_mask], axis=-1) <= 0.01).mean())
    test_unrot = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_per_fold], axis=0)
    print(f"  OOF R-Hit: {oof_rhit:.4f}  소요 {(time.time()-t0)/60:.1f}m")
    return oof_unrot_all, test_unrot, fold_rh, oof_rhit, fold_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--setup", choices=["A", "B"], default="A")
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--tta-sigma-mm", type=float, default=1.0)
    parser.add_argument("--noise-tta", action="store_true", default=False)
    parser.add_argument("--no-swa", dest="swa", action="store_false", default=True)
    parser.add_argument("--no-smooth", dest="smooth", action="store_false", default=True)
    parser.add_argument("--use-data-mode", default="full")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    flag_tag = ""
    if not args.swa: flag_tag += "_noswa"
    if not args.smooth: flag_tag += "_nosmooth"
    if args.noise_tta: flag_tag += "_noisetta"
    suffix = ("" if args.setup == "A" else f"_setup{args.setup}") + flag_tag
    state_file = CACHE_DIR / f"v100_swa_smooth_tta{suffix}_state.npz"
    sub_file = DATA_DIR / f"submission_v100_swa_smooth_tta{suffix}.csv"

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v100 = v90 mirror + SWA + smooth R-Hit curriculum + input-noise TTA")
    print(f"  setup={args.setup}, n_folds={args.n_folds}, n_seeds={args.n_seeds}, max_epochs={args.max_epochs}")
    print(f"  tta_n={args.tta_n}, sigma_mm={args.tta_sigma_mm}")
    print(f"  state={state_file.name}, sub={sub_file.name}")
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
        oof = st["oof"]; test_pred = st["test_pred"]
        oof_rhit = float(st["oof_rhit"])
        print(f"[state] cache 로드: OOF={oof_rhit:.4f}")
    else:
        if args.setup == "A":
            CONFIG = dict(hidden=64, fc=128, lr=5e-4, p=0.3, wd=1e-4)
        else:
            CONFIG = dict(hidden=64, fc=128, lr=1e-3, p=0.1, wd=1e-4)
        oof, test_pred, fold_rh, oof_rhit, mask = run_kfold(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CONFIG, n_folds=args.n_folds, n_seeds=args.n_seeds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
            tta_n=args.tta_n, tta_sigma_mm=args.tta_sigma_mm,
            swa_enabled=args.swa, smooth_enabled=args.smooth, noise_tta_enabled=args.noise_tta,
            device=device,
        )
        np.savez(state_file, oof=oof, test_pred=test_pred, oof_rhit=oof_rhit,
                 fold_rh=np.array(fold_rh), tta_n=args.tta_n, sigma_mm=args.tta_sigma_mm)
        print(f"[state] 저장: {state_file}")

    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal = oof * ALPHA[None, :]; test_cal = test_pred * ALPHA[None, :]
    pred = kalman_train + oof_cal
    rh_cal = float((np.linalg.norm(pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n[v100 SWA+smooth+TTA {args.setup}] OOF cal: {rh_cal:.4f}")
    print(f"  baseline v90 setup A: 0.6643, v90 A+B avg: 0.6652, v96 4-view: 0.6625")

    test_pos = kalman_test + test_cal
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(sub_file, index=False)
    print(f"  [submission] {sub_file.name}")

    entry = {"version": f"v100_swa_smooth_tta{suffix}", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": f"v90 mirror + SWA blend(70/30 best) + smooth R-Hit β curriculum + input-noise TTA σ={args.tta_sigma_mm}mm N={args.tta_n}, setup {args.setup}",
             "oof_raw": float(oof_rhit), "oof_cal": float(rh_cal),
             "submission": str(sub_file)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
