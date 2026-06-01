"""v90_yaw_mirror_aug.py — v77 BiGRU + y-mirror augmentation + 2-view TTA.

paradigm shift 카드 (LB v82=v56=0.6876 plateau 도달 후).

설계:
  - v77 framework 그대로 (Bidirectional GRU + Kalman residual + F + W aux heads)
  - 추가: y-mirror augmentation
    * raw frame에서 y → -y 대칭 (좌우 mirror, mosquito flight 좌우 대칭)
    * seq channels [rel_x, rel_y, rel_z, v_x, v_y, v_z, a_x, a_y, a_z] → y indices 1, 4, 7
    * target_T8/F/W는 rotated frame에서 y(index 1) 부호 반전
  - 학습: train 데이터 2배로 expansion (normal + mirror), 같은 fold split
  - inference: 2-view TTA (normal + mirror-then-unflip 평균)
  - scal feature는 yaw/mirror invariant (모두 norm 기반) → 변경 없음

비교:
  - --no-mirror : control (mirror 끔, fold/seed/config 동일)
  - --mirror    : mirror aug on (default)

별도 cache (v77 보호):
  - v90_mirror_state.npz / v90_nomirror_state.npz

사용법:
  python scripts/v90_yaw_mirror_aug.py --no-mirror   # baseline
  python scripts/v90_yaw_mirror_aug.py --mirror      # mirror aug

config 기본값: setup A (lr=5e-4, do=0.3), 5fold × 2seed × 150ep × 1 config
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
# y-mirror augmentation
# ============================================================
def mirror_seq(seq: np.ndarray) -> np.ndarray:
    """seq channels [rel_x, rel_y, rel_z, v_x, v_y, v_z, a_x, a_y, a_z]
    y indices: 1, 4, 7 부호 반전."""
    out = seq.copy()
    out[..., 1] *= -1
    out[..., 4] *= -1
    out[..., 7] *= -1
    return out


def mirror_target(t: np.ndarray) -> np.ndarray:
    """rotated frame target (x, y, z) → y 부호 반전."""
    out = t.copy()
    out[..., 1] *= -1
    return out


def unflip_pred_y(pred: np.ndarray) -> np.ndarray:
    """mirror inference 결과를 normal frame으로 되돌림: y 부호 반전."""
    out = pred.copy()
    out[..., 1] *= -1
    return out


# ============================================================
# K-fold runner with optional mirror augmentation
# ============================================================
def run_kfold(target_main, target_F, target_W,
              seq_arr, scal_arr, seq_te, scal_te,
              kalman_train, theta_train, theta_test, y_train,
              config, n_folds, n_seeds, max_epochs, patience, batch,
              mirror_on=True, device="cpu"):
    oof_rot = np.zeros((len(target_main), 3))
    fold_mask = np.zeros(len(target_main), dtype=bool)
    test_per_fold = []
    fold_rh = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_iter = list(kf.split(np.arange(len(target_main))))
    t0 = time.time()

    # pre-compute mirror once (memory cheap: 8000 × 11 × 9 × 4 ≈ 3MB)
    if mirror_on:
        seq_arr_m = mirror_seq(seq_arr)
        target_main_m = mirror_target(target_main)
        target_F_m = mirror_target(target_F)
        target_W_m = mirror_target(target_W)
        seq_te_m = mirror_seq(seq_te)
    else:
        seq_arr_m = target_main_m = target_F_m = target_W_m = seq_te_m = None

    for fi, (tr, va) in enumerate(fold_iter):
        # normalize fit on original tr (mirror 추가 전 raw 분포 기준)
        sc_seq = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
        sc_scal = StandardScaler().fit(scal_arr[tr])

        if mirror_on:
            # 학습 data 2x: [tr_orig, tr_mirror]
            seq_tr_2x = np.concatenate([seq_arr[tr], seq_arr_m[tr]], axis=0)
            scal_tr_2x = np.concatenate([scal_arr[tr], scal_arr[tr]], axis=0)  # invariant
            tgt_tr_2x = np.concatenate([target_main[tr], target_main_m[tr]], axis=0)
            F_tr_2x = np.concatenate([target_F[tr], target_F_m[tr]], axis=0)
            W_tr_2x = np.concatenate([target_W[tr], target_W_m[tr]], axis=0)
        else:
            seq_tr_2x = seq_arr[tr]
            scal_tr_2x = scal_arr[tr]
            tgt_tr_2x = target_main[tr]
            F_tr_2x = target_F[tr]
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
            seq_va_m_t = T(seq_va_m_n)
            seq_te_m_t = T(seq_te_m_n)

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

                # validation: TTA if mirror_on
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
                else:
                    no_improve += 1
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
                else:
                    pv = pv_normal; pt = pt_normal
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
    print(f"  OOF R-Hit: {oof_rhit:.4f}  소요 {(time.time()-t0)/60:.1f}m  [mirror={mirror_on}]")
    return oof_unrot_all, test_unrot, fold_rh, oof_rhit, fold_mask


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--setup", choices=["A", "B"], default="A",
                        help="A: lr=5e-4 do=0.3 / B: lr=1e-3 do=0.1")
    parser.add_argument("--use-data-mode", default="full",
                        help="data/noise cache mode (full/fast/micro). full 권장.")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    mirror_on = args.mirror
    tag = "mirror" if mirror_on else "nomirror"
    suffix = "" if args.setup == "A" else f"_setup{args.setup}"
    state_file = CACHE_DIR / f"v90_{tag}{suffix}_state.npz"
    sub_file = DATA_DIR / f"submission_v90_{tag}{suffix}.csv"

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v90 BiGRU + y-mirror aug + TTA")
    print(f"  mirror_on={mirror_on}, n_folds={args.n_folds}, n_seeds={args.n_seeds}, "
          f"max_epochs={args.max_epochs}, batch={args.batch}")
    print(f"  state={state_file.name}, sub={sub_file.name}")
    print("=" * 60)

    X_train, X_test, y_train, sub = load_data()
    kalman_train, kalman_test, kalman_train_alt = get_kalman(X_train, X_test)

    # MODE_CONFIGS의 "full" 사용 (loo_sample=None → 전체 sample)
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
        oof_rhit = float(st["oof_rhit"]); fold_rh = st["fold_rh"].tolist()
        print(f"[state] cache 로드: OOF={oof_rhit:.4f}")
    else:
        if args.setup == "A":
            CONFIG = dict(hidden=64, fc=128, lr=5e-4, p=0.3, wd=1e-4)  # setup A
        else:
            CONFIG = dict(hidden=64, fc=128, lr=1e-3, p=0.1, wd=1e-4)  # setup B
        oof, test_pred, fold_rh, oof_rhit, mask = run_kfold(
            target_T8, target_F, target_W,
            seq_tr, X_scal_tr, seq_te, X_scal_te,
            kalman_train, theta_train, theta_test, y_train,
            config=CONFIG, n_folds=args.n_folds, n_seeds=args.n_seeds,
            max_epochs=args.max_epochs, patience=args.patience, batch=args.batch,
            mirror_on=mirror_on, device=device,
        )
        np.savez(state_file, oof=oof, test_pred=test_pred, oof_rhit=oof_rhit,
                 fold_rh=np.array(fold_rh), mirror_on=mirror_on)
        print(f"[state] 저장: {state_file}")

    ALPHA = np.array([1.000, 0.950, 1.000])
    oof_cal = oof * ALPHA[None, :]
    test_cal = test_pred * ALPHA[None, :]
    pred = kalman_train + oof_cal
    rh_cal = float((np.linalg.norm(pred - y_train, axis=-1) <= 0.01).mean())
    print(f"\n[v90 {tag}] OOF cal: {rh_cal:.4f}")

    test_pos = kalman_test + test_cal
    pd.DataFrame({"id": sub["id"], "x": test_pos[:,0], "y": test_pos[:,1], "z": test_pos[:,2]}
                 ).to_csv(sub_file, index=False)
    print(f"  [submission] {sub_file.name}")

    entry = {"version": f"v90_{tag}", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "approach": f"v77 BiGRU + y-mirror={mirror_on} + TTA, setup A only",
             "n_folds": args.n_folds, "n_seeds": args.n_seeds, "max_epochs": args.max_epochs,
             "oof_raw": float(oof_rhit), "oof_cal": float(rh_cal),
             "submission": str(sub_file)}
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
