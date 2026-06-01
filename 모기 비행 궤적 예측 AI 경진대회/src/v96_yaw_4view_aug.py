"""v96_yaw_4view_aug.py — v90 framework + 4-view augmentation (normal + mirror + yaw±20°).

v90 (mirror only, 2x): OOF 0.6636 (setup A), 0.6633 (setup B)
v96 = 4 transforms: normal, mirror, yaw +20°, yaw -20°
  - train data 4x
  - TTA: 4 views 평균
  - 시간 ~ 1-1.5hr (train 4x data, BiGRU)

별도 cache: v96_4view_state.npz
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
    loss_combo, loss_euclid,
)
from v77_bigru import BiGRUMultiAux

PROJECT = SCRIPT_DIR.parent
DT = 0.040


# ============================================================
# Transforms: mirror + yaw perturbation
# ============================================================
def mirror_seq(seq):
    out = seq.copy()
    out[..., 1] *= -1; out[..., 4] *= -1; out[..., 7] *= -1
    return out

def mirror_target(t):
    out = t.copy(); out[..., 1] *= -1
    return out

def unflip_pred_y(p):
    out = p.copy(); out[..., 1] *= -1
    return out


def yaw_seq(seq, angle_rad):
    """seq channels [rel_x, rel_y, rel_z, v_x, v_y, v_z, a_x, a_y, a_z] xy rotate."""
    c = float(np.cos(angle_rad)); s = float(np.sin(angle_rad))
    out = seq.copy()
    for ix, iy in [(0, 1), (3, 4), (6, 7)]:
        x = seq[..., ix].copy()
        y = seq[..., iy].copy()
        out[..., ix] = x * c + y * s
        out[..., iy] = -x * s + y * c
    return out


def yaw_target(t, angle_rad):
    c = float(np.cos(angle_rad)); s = float(np.sin(angle_rad))
    out = t.copy()
    x = t[..., 0].copy(); y = t[..., 1].copy()
    out[..., 0] = x * c + y * s
    out[..., 1] = -x * s + y * c
    return out


def inverse_yaw_pred(p, angle_rad):
    """yaw 회전된 frame의 prediction을 원래 frame으로 되돌림 (-angle 회전)."""
    return yaw_target(p, -angle_rad)


# ============================================================
# K-fold runner: 4-view augmentation
# ============================================================
def run_kfold_4view(target_main, target_F, target_W,
                    seq_arr, scal_arr, seq_te, scal_te,
                    kalman_train, theta_train, theta_test, y_train,
                    config, n_folds, n_seeds, max_epochs, patience, batch,
                    yaw_perturb_deg=20.0, device="cpu"):
    p = float(np.deg2rad(yaw_perturb_deg))
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold = []; fold_rh = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
    t0 = time.time()

    # pre-compute 4 transforms on whole data
    transforms = {
        "normal": (seq_arr, target_main, target_F, target_W, seq_te),
        "mirror": (mirror_seq(seq_arr), mirror_target(target_main),
                    mirror_target(target_F), mirror_target(target_W), mirror_seq(seq_te)),
        "yaw_pos": (yaw_seq(seq_arr, p), yaw_target(target_main, p),
                     yaw_target(target_F, p), yaw_target(target_W, p), yaw_seq(seq_te, p)),
        "yaw_neg": (yaw_seq(seq_arr, -p), yaw_target(target_main, -p),
                     yaw_target(target_F, -p), yaw_target(target_W, -p), yaw_seq(seq_te, -p)),
    }
    # TTA unflip functions
    def unflip(name, pred):
        if name == "normal": return pred
        if name == "mirror": return unflip_pred_y(pred)
        if name == "yaw_pos": return inverse_yaw_pred(pred, p)
        if name == "yaw_neg": return inverse_yaw_pred(pred, -p)

    for fi, (tr, va) in enumerate(fold_iter):
        sc_seq = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
        sc_scal = StandardScaler().fit(scal_arr[tr])

        # 4x train data
        seq_tr_list = []; tgt_list = []; F_list = []; W_list = []; scal_tr_list = []
        for name, (sa, tm, tf, tw, se_te) in transforms.items():
            seq_tr_list.append(sa[tr])
            tgt_list.append(tm[tr]); F_list.append(tf[tr]); W_list.append(tw[tr])
            scal_tr_list.append(scal_arr[tr])  # scal invariant
        seq_tr_4x = np.concatenate(seq_tr_list, axis=0)
        scal_tr_4x = np.concatenate(scal_tr_list, axis=0)
        tgt_tr_4x = np.concatenate(tgt_list, axis=0)
        F_tr_4x = np.concatenate(F_list, axis=0)
        W_tr_4x = np.concatenate(W_list, axis=0)

        seq_tr_n = normalize_seq(seq_tr_4x, sc_seq)
        scal_tr_n = sc_scal.transform(scal_tr_4x).astype(np.float32)
        scal_va_n = sc_scal.transform(scal_arr[va]).astype(np.float32)
        scal_te_n = sc_scal.transform(scal_te).astype(np.float32)

        # validation/test 4 views (normalized)
        seq_va_views = {name: normalize_seq(t[0][va], sc_seq) for name, t in transforms.items()}
        seq_te_views = {name: normalize_seq(t[4], sc_seq) for name, t in transforms.items()}

        def T(a): return torch.from_numpy(a).to(device)
        seq_tr_t, scal_tr_t = T(seq_tr_n), T(scal_tr_n)
        tgt_tr_t = T(tgt_tr_4x.astype(np.float32))
        F_tr_t = T(F_tr_4x.astype(np.float32))
        W_tr_t = T(W_tr_4x.astype(np.float32))
        scal_va_t = T(scal_va_n); scal_te_t = T(scal_te_n)
        seq_va_view_t = {name: T(v) for name, v in seq_va_views.items()}
        seq_te_view_t = {name: T(v) for name, v in seq_te_views.items()}

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
            for ep in range(max_epochs):
                model.train()
                perm = torch.randperm(n_tr_eff)
                for i in range(0, n_tr_eff, batch):
                    idx = perm[i:i+batch]
                    opt.zero_grad()
                    out_main, outs_aux = model(seq_tr_t[idx], scal_tr_t[idx])
                    loss = loss_combo(out_main, tgt_tr_t[idx])
                    loss = loss + 0.3 * loss_euclid(outs_aux[0], F_tr_t[idx])
                    loss = loss + 0.3 * loss_euclid(outs_aux[1], W_tr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                # validation 4-view TTA
                model.eval()
                pv_views = []
                with torch.no_grad():
                    for name, vt in seq_va_view_t.items():
                        pv, _ = model(vt, scal_va_t)
                        pv_views.append(unflip(name, pv.cpu().numpy()))
                pv_rot = np.mean(pv_views, axis=0)
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
                pv_views = []; pt_views = []
                for name in transforms:
                    pv, _ = model(seq_va_view_t[name], scal_va_t)
                    pt, _ = model(seq_te_view_t[name], scal_te_t)
                    pv_views.append(unflip(name, pv.cpu().numpy()))
                    pt_views.append(unflip(name, pt.cpu().numpy()))
            seed_val.append(np.mean(pv_views, axis=0))
            seed_test.append(np.mean(pt_views, axis=0))
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
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--setup", choices=["A", "B"], default="A")
    parser.add_argument("--yaw-perturb-deg", type=float, default=20.0)
    parser.add_argument("--use-data-mode", default="full")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    suffix = "" if args.setup == "A" else f"_setup{args.setup}"
    state_file = CACHE_DIR / f"v96_4view{suffix}_state.npz"
    sub_file = DATA_DIR / f"submission_v96_4view{suffix}.csv"

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v96 4-view augmentation (normal + mirror + yaw ±{args.yaw_perturb_deg}°)")
    print(f"  setup={args.setup}, n_folds={args.n_folds}, n_seeds={args.n_seeds}, "
          f"max_epochs={args.max_epochs}")
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
        oof, test_pred, fold_rh, oof_rhit, mask = run_kfold_4view(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CONFIG, n_folds=args.n_folds, n_seeds=args.n_seeds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
            yaw_perturb_deg=args.yaw_perturb_deg, device=device,
        )
        np.savez(state_file, oof=oof, test_pred=test_pred, oof_rhit=oof_rhit,
                 fold_rh=np.array(fold_rh), yaw_perturb_deg=args.yaw_perturb_deg)
        print(f"[state] 저장: {state_file}")

    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal = oof * ALPHA[None, :]; test_cal = test_pred * ALPHA[None, :]
    pred = kalman_train + oof_cal
    rh_cal = float((np.linalg.norm(pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n[v96 4-view {args.setup}] OOF cal: {rh_cal:.4f}")
    print(f"  (v90 setup A mirror only: 0.6643, v90 A+B avg: 0.6652)")

    test_pos = kalman_test + test_cal
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(sub_file, index=False)
    print(f"  [submission] {sub_file.name}")

    entry = {"version": f"v96_4view{suffix}", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": f"v77 BiGRU + 4-view aug (normal+mirror+yaw±{args.yaw_perturb_deg}°), setup {args.setup}",
             "oof_raw": float(oof_rhit), "oof_cal": float(rh_cal),
             "submission": str(sub_file)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
