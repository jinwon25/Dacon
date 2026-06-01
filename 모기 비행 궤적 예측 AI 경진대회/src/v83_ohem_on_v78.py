"""v83_ohem_on_v78.py — OHEM (Online Hard Example Mining) on v78 boundary base.

v78 OOF 0.6730 (v35와 거의 동일).
OHEM: 매 step loss top-K% sample만 backward → boundary/miss sample에 집중
v78 base (v77 BiGRU + boundary MLP)에 OHEM 추가 → 학습 차별화

설계:
  - 동일한 v78 BoundaryMLP architecture
  - 학습 중 loss top-30% sample만 backward (OHEM ratio 0.7 = keep top loss)
  - cap 1.0cm, boundary weight 2.0 default
  - 5-fold × 3-seed
"""
from __future__ import annotations

import argparse, datetime as _dt, gc, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data/cache"
BEST_OOF_PATH = PROJECT_DIR / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
V16_PATH = PROJECT_DIR / "archive" / "v16_stack_oof.npz"
BEST_TEST = PROJECT_DIR / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def loss_per_sample(p, t):
    """Per-sample combo loss for OHEM"""
    d = torch.sqrt(((p - t) ** 2).sum(dim=-1) + 1e-12)
    sh = torch.sigmoid((d - 0.01) / 0.002)
    return d + 0.3 * sh  # (B,)


class BoundaryMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, p=0.2, cap_cm=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 3)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)
        self.cap = cap_cm / 100.0

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z))
        return torch.tanh(self.head(z)) * self.cap


def build_features(X, kalman, base_pred, gate_pred, v16_pred):
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    diff_bg = base_pred - gate_pred
    diff_bv = base_pred - v16_pred
    dist_bg = np.linalg.norm(diff_bg, axis=-1, keepdims=True)
    res_b_kal = base_pred - kalman
    return np.concatenate([
        base_pred, gate_pred, v16_pred,
        diff_bg, diff_bv, dist_bg,
        kalman, res_b_kal,
        last_pos, v, a, v_mean, v_std, speed, a_norm,
    ], axis=-1).astype(np.float32)


def train_one_seed(seed, feat_tr, y_tr, base_tr, feat_test, test_base, sample_w_tr,
                    args, kf):
    device = torch.device("cpu")
    oof_pred = np.zeros_like(base_tr)
    test_per_fold = []
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        feat_tr_n = sc.transform(feat_tr[tr]).astype(np.float32)
        feat_va_n = sc.transform(feat_tr[va]).astype(np.float32)
        feat_te_n = sc.transform(feat_test).astype(np.float32)
        def T(a): return torch.from_numpy(a).to(device)
        x_tr, x_va, x_te = T(feat_tr_n), T(feat_va_n), T(feat_te_n)
        base_tr_t = T(base_tr[tr].astype(np.float32))
        base_va_t = T(base_tr[va].astype(np.float32))
        base_te_t = T(test_base.astype(np.float32))
        y_tr_t = T(y_tr[tr].astype(np.float32))
        sw_t = T(sample_w_tr[tr])

        torch.manual_seed(seed); np.random.seed(seed)
        model = BoundaryMLP(in_dim=feat_tr_n.shape[1], hidden=args.hidden, p=0.2,
                            cap_cm=args.cap_cm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
        best_rh, best_state, no_improve = -1.0, None, 0
        for ep in range(args.max_epochs):
            model.train()
            perm = torch.randperm(x_tr.shape[0])
            for i in range(0, x_tr.shape[0], 256):
                idx = perm[i:i+256]
                opt.zero_grad()
                pred = base_tr_t[idx] + model(x_tr[idx])
                # OHEM: per-sample loss × sample_weight → keep top-K%
                loss_p = loss_per_sample(pred, y_tr_t[idx]) * sw_t[idx]
                # warmup: 처음 K epochs는 모든 sample 사용 (안정성)
                if ep < args.ohem_warmup:
                    loss = loss_p.mean()
                else:
                    K = max(1, int(args.ohem_ratio * len(loss_p)))
                    top_loss, _ = torch.topk(loss_p, K)
                    loss = top_loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pred_va = (base_va_t + model(x_va)).cpu().numpy()
            rh = float((np.linalg.norm(pred_va - y_tr[va], axis=-1) <= 0.01).mean())
            if rh > best_rh:
                best_rh = rh
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else: no_improve += 1
            if no_improve >= args.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_pred[va] = (base_va_t + model(x_va)).cpu().numpy()
            test_per_fold.append((base_te_t + model(x_te)).cpu().numpy())
        rh_fold = float((np.linalg.norm(oof_pred[va] - y_tr[va], axis=-1) <= 0.01).mean())
        print(f"  seed{seed} fold{fi+1}: rhit={rh_fold:.4f}  ({(time.time()-t0)/60:.1f}m)")
        del model; gc.collect()
    return oof_pred, np.mean(test_per_fold, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cap-cm", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ohem-ratio", type=float, default=0.5,
                        help="keep top ohem_ratio fraction of samples per batch (by loss)")
    parser.add_argument("--ohem-warmup", type=int, default=10,
                        help="first N epochs use all samples (stability)")
    parser.add_argument("--base", default="v78", choices=["v77", "v78", "v35"],
                        help="base model for boundary refinement")
    args = parser.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 60)
    print(f"v83 OHEM on {args.base} (ratio {args.ohem_ratio}, warmup {args.ohem_warmup}, "
          f"cap {args.cap_cm}cm, {args.n_seeds}seed)")
    print("=" * 60)

    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y_train = labels.set_index("id").loc[list(train_ids)][["x","y","z"]].values.astype(np.float64)
    nc = np.load(CACHE_DIR / "xtrain_xtest.npz")
    X_train, X_test = nc["X_train"], nc["X_test"]
    kc = np.load(CACHE_DIR / "kalman.npz")
    kalman_train, kalman_test = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    if args.base == "v77":
        st = np.load(CACHE_DIR / "v77_bigru_state.npz")
        oof_base = kalman_train + (st["oof_A"] + st["oof_B"])/2 * ALPHA
        test_base = kalman_test + (st["test_A"] + st["test_B"])/2 * ALPHA
        base_name = "v77"
    elif args.base == "v78":
        # v78 자체에 OHEM 적용은 의미 작음 — v77을 base로 받고 OHEM-boundary 재학습
        st = np.load(CACHE_DIR / "v77_bigru_state.npz")
        oof_base = kalman_train + (st["oof_A"] + st["oof_B"])/2 * ALPHA
        test_base = kalman_test + (st["test_A"] + st["test_B"])/2 * ALPHA
        base_name = "v77"  # v78의 base가 v77이므로 OHEM-boundary는 같은 v77 base에 적용
    else:  # v35
        st = np.load(CACHE_DIR / "v30_state.npz")
        oof_base = kalman_train + (st["oof_A"] + st["oof_B"])/2 * ALPHA
        test_base = kalman_test + (st["test_A"] + st["test_B"])/2 * ALPHA
        base_name = "v30"

    bo = np.load(BEST_OOF_PATH, allow_pickle=True)
    best_ids = bo["ids"]; gate_oof = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(best_ids, train_ids):
        idx_map = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([idx_map[i] for i in best_ids])
        gate_oof = gate_oof[perm]
    test_gate = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)
    st16 = np.load(V16_PATH)
    oof_v16 = st16["oof"].astype(np.float64); test_v16 = st16["test"].astype(np.float64)

    rh_base = float((np.linalg.norm(oof_base - y_train, axis=-1) <= 0.01).mean())
    print(f"{base_name} base OOF: {rh_base:.4f}")

    d_base = np.linalg.norm(oof_base - y_train, axis=-1)
    boundary_mask = (d_base > 0.005) & (d_base <= 0.03)
    sample_w = np.ones(len(y_train), dtype=np.float32)
    sample_w[boundary_mask] = args.boundary_weight
    print(f"boundary samples: {boundary_mask.sum()} ({args.boundary_weight}× weight)")

    feat_train = build_features(X_train, kalman_train, oof_base, gate_oof, oof_v16)
    feat_test = build_features(X_test, kalman_test, test_base, test_gate, test_v16)
    print(f"feat dim: {feat_train.shape[1]}")

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    oofs, tests = [], []
    for s in range(args.n_seeds):
        print(f"\n=== seed {s} ===")
        oof_s, test_s = train_one_seed(s, feat_train, y_train, oof_base,
                                        feat_test, test_base, sample_w, args, kf)
        oofs.append(oof_s); tests.append(test_s)
        rh_s = float((np.linalg.norm(oof_s - y_train, axis=-1) <= 0.01).mean())
        print(f"  seed{s} OOF: {rh_s:.4f}")

    oof_v83 = np.mean(oofs, axis=0); test_v83 = np.mean(tests, axis=0)
    rh_v83 = float((np.linalg.norm(oof_v83 - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== v83 결과 ===")
    print(f"  {base_name} base   : {rh_base:.4f}")
    print(f"  v83 (OHEM on {args.base}): {rh_v83:.4f}  (Δ vs base: {rh_v83 - rh_base:+.4f})")
    print(f"  vs v78 0.6730: {rh_v83 - 0.6730:+.4f}")

    # save
    state_name = f"v83_ohem_{args.base}_r{args.ohem_ratio:.1f}_state.npz".replace(".", "p", 1).replace(".npz", ".npz")
    # 단순화
    state_name = f"v83_ohem_state.npz"
    np.savez(CACHE_DIR / state_name,
             oof_v83=oof_v83, test_v83=test_v83, rh_v83=rh_v83,
             ohem_ratio=args.ohem_ratio, base=args.base)
    out_csv = DATA_DIR / "submission_v83_ohem.csv"
    pd.DataFrame({"id": sub["id"], "x": test_v83[:,0], "y": test_v83[:,1], "z": test_v83[:,2]}).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv.name}")

    # blend with base
    st35 = np.load(CACHE_DIR / "v35_state.npz")
    v48s = np.load(CACHE_DIR / "v48_state.npz"); v46s = np.load(CACHE_DIR / "v46_state.npz")
    bo_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*st35["oof_v35"]
    bo_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*st35["test_v35"]
    rh_b = float((np.linalg.norm(bo_o - y_train, axis=-1) <= 0.01).mean())
    print(f"\n=== blend v48 3-way + v83 ===")
    best_w, best_r = 1.0, rh_b
    for w in np.linspace(0, 1, 21):
        ens = w * bo_o + (1 - w) * oof_v83
        r = float((np.linalg.norm(ens - y_train, axis=-1) <= 0.01).mean())
        if r > best_r: best_r, best_w = r, w
    print(f"  best w={best_w:.2f} → OOF {best_r:.4f}  Δ {best_r - rh_b:+.4f}")
    if best_w < 1 and best_r > rh_b:
        hyb_t = best_w * bo_t + (1 - best_w) * test_v83
        h = DATA_DIR / f"submission_v83_hyb_basex{best_w:.2f}.csv"
        pd.DataFrame({"id": sub["id"], "x": hyb_t[:,0], "y": hyb_t[:,1], "z": hyb_t[:,2]}).to_csv(h, index=False)
        print(f"  [submission] {h.name}")

    entry = {"version": "v83_ohem", "ts": _dt.datetime.now().isoformat(timespec="seconds"),
             "base": args.base, "ohem_ratio": args.ohem_ratio,
             "rh_base": rh_base, "rh_v83": rh_v83,
             "blend_oof": float(best_r), "blend_w": float(best_w)}
    log_path = PROJECT_DIR / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
