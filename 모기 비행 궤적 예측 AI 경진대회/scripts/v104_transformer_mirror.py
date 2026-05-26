"""v104_transformer_mirror.py — v41 Causal Transformer + y-mirror aug + TTA.

architecture diversity 핵심 카드.
v77/v90/v100/v103 모두 BiGRU 기반 → ensemble pool 단일 backbone.
Transformer + mirror aug로 진짜 다른 backbone paradigm 확보.

기대:
  - v41 (Transformer, no mirror) full OOF ~0.6602~0.6608 (setup A/B)
  - v104 (Transformer + mirror + TTA) OOF +0.004~0.008 보장 (v77→v90 패턴)
  - v104 boundary MLP → v94/v97과 ortho ensemble lift

설계:
  - v41의 CausalTransformerMultiAux 그대로 사용
  - y-mirror aug (v90과 동일): seq y/vy/ay 부호 반전, target y 부호 반전
  - 학습: train data 2x (normal + mirror)
  - inference: 2-view TTA (mirror unflip 평균)
  - adv reweight 그대로 사용 (v30 framework)
  - 5-fold × 2-seed × 100ep × 1 setup (A 또는 B)

별도 cache: v104{suffix}_state.npz
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, glob, json, os, random, sys, time
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
    DT, build_seq, build_tier3, build_scalar_feats,
    yaw_angle, rotate_xy, inverse_rotate_xy, normalize_seq,
)
from v30_advanced_v23 import (
    compute_adv_weights, loss_combo_weighted, loss_aux_weighted,
)
from v41_transformer_base import CausalTransformerMultiAux

PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "open"
CACHE_DIR = PROJECT_DIR / "cache"


# ============================================================
# y-mirror augmentation (v90과 동일)
# ============================================================
def mirror_seq(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    out[..., 1] *= -1; out[..., 4] *= -1; out[..., 7] *= -1
    return out

def mirror_target(t: np.ndarray) -> np.ndarray:
    out = t.copy(); out[..., 1] *= -1; return out

def unflip_pred_y(pred: np.ndarray) -> np.ndarray:
    out = pred.copy(); out[..., 1] *= -1; return out


def run_kfold_trans_mirror(target_main, target_F, target_W,
                            seq_arr, scal_arr, seq_te, scal_te,
                            sample_weight,
                            kalman_train, theta_train, theta_test, y_train,
                            config, n_folds, n_seeds, max_epochs, patience, batch,
                            mirror_on=True, lambda_F=0.3, lambda_W=0.3, device="cpu"):
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold = []; fold_rh = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
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
            sw_tr_2x = np.concatenate([sample_weight[tr], sample_weight[tr]], axis=0)
        else:
            seq_tr_2x = seq_arr[tr]; scal_tr_2x = scal_arr[tr]
            tgt_tr_2x = target_main[tr]; F_tr_2x = target_F[tr]
            W_tr_2x = target_W[tr]; sw_tr_2x = sample_weight[tr]

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
        sw_tr_t = T(sw_tr_2x.astype(np.float32))
        seq_va_t, scal_va_t = T(seq_va_n), T(scal_va_n)
        seq_te_t, scal_te_t = T(seq_te_n), T(scal_te_n)
        if mirror_on:
            seq_va_m_t = T(seq_va_m_n); seq_te_m_t = T(seq_te_m_n)

        seed_val, seed_test = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = CausalTransformerMultiAux(
                n_channels=seq_arr.shape[2], scal_dim=scal_arr.shape[1],
                d_model=config["d_model"], nhead=config["nhead"],
                num_layers=config["layers"], dim_ff=config["dim_ff"],
                fc=config["fc"], p=config["p"],
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
                    out_main, outs_aux = model(seq_tr_t[idx], scal_tr_t[idx])
                    sw_b = sw_tr_t[idx]
                    loss = loss_combo_weighted(out_main, tgt_tr_t[idx], sw_b)
                    loss = loss + lambda_F * loss_aux_weighted(outs_aux[0], F_tr_t[idx], sw_b)
                    loss = loss + lambda_W * loss_aux_weighted(outs_aux[1], W_tr_t[idx], sw_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    pv_normal, _ = model(seq_va_t, scal_va_t)
                    pv_normal = pv_normal.cpu().numpy()
                    if mirror_on:
                        pv_mirror_raw, _ = model(seq_va_m_t, scal_va_t)
                        pv_mirror = unflip_pred_y(pv_mirror_raw.cpu().numpy())
                        pv_rot = (pv_normal + pv_mirror) / 2
                    else:
                        pv_rot = pv_normal
                pv = inverse_rotate_xy(pv_rot, theta_train[va])
                pred = kalman_train[va] + pv
                rh = float((np.linalg.norm(pred - y_train[va], axis=-1) <= 0.01).mean())
                if rh > best_rh:
                    best_rh = rh
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else: no_improve += 1
                if no_improve >= patience: break
                if ep == 0 or (ep + 1) % 10 == 0:
                    print(f"  fold{fi+1} seed{s} ep{ep+1:3d}: rhit={rh:.4f} (best {best_rh:.4f})  "
                          f"[{(time.time()-t0)/60:.1f}m]", flush=True)

            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                pv_normal, _ = model(seq_va_t, scal_va_t)
                pt_normal, _ = model(seq_te_t, scal_te_t)
                pv_normal = pv_normal.cpu().numpy(); pt_normal = pt_normal.cpu().numpy()
                if mirror_on:
                    pv_mirror_raw, _ = model(seq_va_m_t, scal_va_t)
                    pt_mirror_raw, _ = model(seq_te_m_t, scal_te_t)
                    pv = (pv_normal + unflip_pred_y(pv_mirror_raw.cpu().numpy())) / 2
                    pt = (pt_normal + unflip_pred_y(pt_mirror_raw.cpu().numpy())) / 2
                else: pv = pv_normal; pt = pt_normal
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
        print(f"  ★ fold {fi+1}/{n_folds}: R-Hit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)  "
              f"[mirror={mirror_on}]", flush=True)

    oof_unrot_all = np.zeros_like(target_main)
    oof_unrot_all[fold_mask] = inverse_rotate_xy(oof_rot[fold_mask], theta_train[fold_mask])
    pred = kalman_train[fold_mask] + oof_unrot_all[fold_mask]
    oof_rhit = float((np.linalg.norm(pred - y_train[fold_mask], axis=-1) <= 0.01).mean())
    test_unrot = np.mean([inverse_rotate_xy(rot, theta_test) for rot in test_per_fold], axis=0)
    print(f"  OOF R-Hit: {oof_rhit:.4f}  소요 {(time.time()-t0)/60:.1f}m  [v104 Trans+mirror]")
    return oof_unrot_all, test_unrot, fold_rh, oof_rhit, fold_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--setup", choices=["A", "B"], default="A")
    parser.add_argument("--tag", default="v104")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    suffix = "" if args.setup == "A" else f"_setup{args.setup}"
    state_file = CACHE_DIR / f"{args.tag}{suffix}_state.npz"
    sub_file = DATA_DIR / f"submission_{args.tag}{suffix}.csv"

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v104 = Transformer + y-mirror={args.mirror} + TTA  setup={args.setup}")
    print(f"  state={state_file.name}  sub={sub_file.name}")
    print("=" * 60)

    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    y_train = labels.set_index("id").loc[train_ids][["x","y","z"]].values.astype(np.float64)

    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test, kalman_train_alt = kc["kalman_train"], kc["kalman_test"], kc["kalman_train_alt"]

    nc_noise = np.load(CACHE_DIR / "noise_fast.npz")
    scal_tr = build_scalar_feats(X_train, nc_noise["noise_p"], nc_noise["noise_s"], nc_noise["noise_l"])
    scal_te = build_scalar_feats(X_test, nc_noise["noise_p_test"], nc_noise["noise_s_test"])
    LOG_COLS = ["mean_speed","max_speed","speed_std","mean_acc","max_acc","max_jerk",
                "net_disp","|v_last|","|a_last|","|a_recent|","jerk_last","jerk_recent",
                "noise_poly2","noise_savgol","noise_loo"]
    for c in LOG_COLS:
        scal_tr[c] = np.log1p(scal_tr[c]); scal_te[c] = np.log1p(scal_te[c])
    X_scal_tr = np.concatenate([scal_tr.values.astype(np.float32), build_tier3(X_train)], axis=-1)
    X_scal_te = np.concatenate([scal_te.values.astype(np.float32), build_tier3(X_test)], axis=-1)
    seq_tr = build_seq(X_train); seq_te = build_seq(X_test)

    v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_test  = (X_test[:, -1]  - X_test[:, -2])  / DT
    theta_train, theta_test = yaw_angle(v_last_train), yaw_angle(v_last_test)
    target_T8 = rotate_xy(y_train - kalman_train, theta_train)
    target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
    target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)

    sample_w, adv_auc = compute_adv_weights(X_train, X_test)

    if state_file.exists() and not args.force_retrain:
        st = np.load(state_file)
        oof = st["oof"]; test_pred = st["test_pred"]
        oof_rhit = float(st["oof_rhit"])
        print(f"[state] cache 로드: OOF={oof_rhit:.4f}")
    else:
        if args.setup == "A":
            CFG = dict(d_model=64, nhead=4, layers=2, dim_ff=128, fc=128,
                       lr=3e-4, p=0.2, wd=1e-4)
        else:
            CFG = dict(d_model=64, nhead=4, layers=2, dim_ff=128, fc=128,
                       lr=8e-4, p=0.1, wd=1e-4)
        oof, test_pred, fold_rh, oof_rhit, mask = run_kfold_trans_mirror(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te, sample_w,
            kalman_train, theta_train, theta_test, y_train,
            config=CFG, n_folds=args.n_folds, n_seeds=args.n_seeds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
            mirror_on=args.mirror, device=device,
        )
        np.savez(state_file, oof=oof, test_pred=test_pred, oof_rhit=oof_rhit,
                 fold_rh=np.array(fold_rh), mirror_on=args.mirror)
        print(f"[state] 저장: {state_file}")

    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal = oof * ALPHA[None, :]
    test_cal = test_pred * ALPHA[None, :]
    pred = kalman_train + oof_cal
    rh_cal = float((np.linalg.norm(pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n[v104 setup{args.setup}] OOF cal: {rh_cal:.4f}  (v41 full ~0.6602~0.6608, v90 mirror 0.6643)")

    test_pos = kalman_test + test_cal
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(sub_file, index=False)
    print(f"  [submission] {sub_file.name}")

    entry = {"version": f"v104_setup{args.setup}", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": f"Transformer + y-mirror={args.mirror} + TTA + adv reweight",
             "n_folds": args.n_folds, "n_seeds": args.n_seeds, "max_epochs": args.max_epochs,
             "oof_raw": float(oof_rhit), "oof_cal": float(rh_cal),
             "submission": str(sub_file)}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
