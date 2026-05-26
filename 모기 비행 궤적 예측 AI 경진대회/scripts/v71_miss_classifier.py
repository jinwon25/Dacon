"""v71_miss_classifier.py — Learned base-miss classifier + selector routing.

진단:
  - v69: 사후 base-miss 알면 OOF 0.6935 (+0.0187 vs base 0.6748)
  - v70: unconditional uncertainty proxy 모두 실패
  - 핵심: base가 miss할지 학습으로 예측 가능?

설계:
  Stage 1 — Miss classifier:
    Input: features (X, base pred, pool preds, scalar)
    Target: 1 if base miss else 0 (binary)
    Loss: BCE
    5-fold × 3-seed, OOF prob 저장
  Stage 2 — Routing:
    p_miss ≥ T → use rescue (v65h, v65s, v62, 또는 v68 selector hard)
    else        → use base v48 3-way
    T grid 최적 OOF
  Stage 3 — apply test (selector test prob 없음 → rescue로 v65h/v65s/v62 사용)

학습 안정성:
  - features: X_train flat + base/rescue diff + pool prediction stats
  - simple MLP (hidden=128, drop=0.3)
"""
import sys, glob, os, gc, time, datetime as _dt, json
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
from v23_train import load_data, yaw_angle, inverse_rotate_xy

PROJECT = SCRIPT_DIR.parent
CACHE = PROJECT / "cache"
DATA = PROJECT / "open"
BO_PATH = PROJECT / "outputs" / "02_boundary_oof" / "cap0p004_apply1_seed20260606" / "boundary_oof_predictions.npz"
BEST_TEST = PROJECT / "outputs" / "00_submit" / "submission_best_public_0p6834_boundary_gate.csv"
DT = 0.040


def rh(p, y): return (np.linalg.norm(p - y, axis=-1) <= 0.01)


class MissNet(nn.Module):
    def __init__(self, in_dim, hidden=128, p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)
        self.act = nn.GELU(); self.drop = nn.Dropout(p)

    def forward(self, x):
        z = self.act(self.fc1(x)); z = self.drop(z)
        z = self.act(self.fc2(z)); z = self.drop(z)
        return self.head(z).squeeze(-1)  # logit


def build_features(X, base, rescues_list, pool_strong):
    """X: (N, 11, 3). base, rescues, pool_strong: (N, 3)"""
    last_pos = X[:, -1, :]
    v = (X[:, -1, :] - X[:, -2, :]) / DT
    a = (X[:, -1, :] - 2 * X[:, -2, :] + X[:, -3, :]) / (DT ** 2)
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    v_recent = np.diff(X[:, -4:], axis=1) / DT
    v_mean = v_recent.mean(axis=1); v_std = v_recent.std(axis=1)
    speed_seq = np.linalg.norm(np.diff(X, axis=1) / DT, axis=-1)
    speed_mean = speed_seq.mean(axis=1, keepdims=True)
    speed_max = speed_seq.max(axis=1, keepdims=True)
    speed_std = speed_seq.std(axis=1, keepdims=True)
    v_seq = np.diff(X, axis=1) / DT
    v_norm = v_seq / (np.linalg.norm(v_seq, axis=-1, keepdims=True) + 1e-9)
    cos_turn = (v_norm[:, :-1] * v_norm[:, 1:]).sum(axis=-1)
    turn_mean = cos_turn.mean(axis=1, keepdims=True)
    turn_min = cos_turn.min(axis=1, keepdims=True)
    # base displacement
    base_disp = base - last_pos
    base_disp_n = np.linalg.norm(base_disp, axis=-1, keepdims=True)
    # rescue diffs from base
    rescue_diffs = []
    for r in rescues_list:
        diff = r - base
        rescue_diffs.append(diff)
        rescue_diffs.append(np.linalg.norm(diff, axis=-1, keepdims=True))
    # pool stats (strong models)
    pool_stack = np.stack(pool_strong, axis=0)  # (M, N, 3)
    pool_std = pool_stack.std(axis=0)  # (N, 3)
    pool_mean = pool_stack.mean(axis=0)
    # base disagreement with pool mean
    base_vs_pool = base - pool_mean
    feats = [last_pos, v, a, v_mean, v_std, speed, a_norm,
             speed_mean, speed_max, speed_std, turn_mean, turn_min,
             base, base_disp, base_disp_n, pool_std, pool_mean, base_vs_pool] + rescue_diffs
    return np.concatenate(feats, axis=-1).astype(np.float32)


def main():
    X_train, X_test, y_train, sub = load_data()
    N, Nt = len(y_train), len(X_test)
    kc = np.load(CACHE / "kalman.npz")
    kt, ke = kc["kalman_train"], kc["kalman_test"]
    ALPHA = np.array([1.000, 0.950, 1.000])[None, :]

    st30 = np.load(CACHE / "v30_state.npz")
    st35 = np.load(CACHE / "v35_state.npz")
    st41 = np.load(CACHE / "v41_state.npz")
    st44 = np.load(CACHE / "v44_state.npz")
    st39 = np.load(CACHE / "v39_state.npz")
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    train_ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    bo = np.load(BO_PATH, allow_pickle=True); gate_o = bo["gate_pred"].astype(np.float64)
    if not np.array_equal(bo["ids"], train_ids):
        m = {i: k for k, i in enumerate(train_ids)}
        perm = np.array([m[i] for i in bo["ids"]]); gate_o = gate_o[perm]
    gate_t = pd.read_csv(BEST_TEST)[["x","y","z"]].values.astype(np.float64)

    st65 = np.load(CACHE / "v65_K64_state.npz")
    st62 = np.load(CACHE / "v62_state.npz")
    v_last_tr = (X_train[:, -1] - X_train[:, -2]) / DT
    v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
    th_tr, th_te = yaw_angle(v_last_tr), yaw_angle(v_last_te)
    v62o = st62["kalman_train"] + inverse_rotate_xy((st62["oof_A"] + st62["oof_B"])/2, th_tr)
    v62t = st62["kalman_test"] + inverse_rotate_xy((st62["test_A"] + st62["test_B"])/2, th_te)

    v48s = np.load(CACHE / "v48_state.npz"); v46s = np.load(CACHE / "v46_state.npz")
    base_o = 0.70*v48s["oof_v48"] + 0.12*v46s["oof_v46"] + 0.18*st35["oof_v35"]
    base_t = 0.70*v48s["test_v48"] + 0.12*v46s["test_v46"] + 0.18*st35["test_v35"]

    v65h_o = st65["oof_hard"].astype(np.float64); v65h_t = st65["test_hard"].astype(np.float64)
    v65s_o = st65["oof_soft"].astype(np.float64); v65s_t = st65["test_soft"].astype(np.float64)

    pool_strong_o = [kt + st30["oof_A"]*ALPHA, kt + st30["oof_B"]*ALPHA,
                     st35["oof_v35"].astype(np.float64),
                     kt + st41["oof_A"]*ALPHA, kt + st41["oof_B"]*ALPHA,
                     st44["oof_v44"].astype(np.float64), gate_o,
                     st39["oof_v39"].astype(np.float64)]
    pool_strong_t = [ke + st30["test_A"]*ALPHA, ke + st30["test_B"]*ALPHA,
                     st35["test_v35"].astype(np.float64),
                     ke + st41["test_A"]*ALPHA, ke + st41["test_B"]*ALPHA,
                     st44["test_v44"].astype(np.float64), gate_t,
                     st39["test_v39"].astype(np.float64)]
    rescue_list_o = [v65h_o, v65s_o, v62o]
    rescue_list_t = [v65h_t, v65s_t, v62t]

    rh_base = rh(base_o, y_train).mean()
    print(f"base v48 3-way OOF: {rh_base:.4f}")
    base_miss = ~rh(base_o, y_train)
    print(f"base miss: {base_miss.sum()} ({base_miss.mean()*100:.1f}%)")

    feat_tr = build_features(X_train, base_o, rescue_list_o, pool_strong_o)
    feat_te = build_features(X_test,  base_t, rescue_list_t, pool_strong_t)
    print(f"feat dim: {feat_tr.shape[1]}")

    # 5-fold × 3-seed BCE
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    prob_oof = np.zeros(N)
    prob_te = np.zeros(Nt)
    n_seeds = 3
    t0 = time.time()
    for fi, (tr, va) in enumerate(kf.split(feat_tr)):
        sc = StandardScaler().fit(feat_tr[tr])
        ft = lambda a: sc.transform(a).astype(np.float32)
        x_tr_n, x_va_n, x_te_n = ft(feat_tr[tr]), ft(feat_tr[va]), ft(feat_te)
        def T(a): return torch.from_numpy(a)
        x_tr_t, x_va_t, x_te_t = T(x_tr_n), T(x_va_n), T(x_te_n)
        y_tr_t = T(base_miss[tr].astype(np.float32))

        prob_seeds_va, prob_seeds_te = [], []
        for s in range(n_seeds):
            torch.manual_seed(s); np.random.seed(s)
            model = MissNet(in_dim=x_tr_t.shape[1], hidden=128, p=0.3)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
            best_auc, best_st, no_imp = -1.0, None, 0
            n_tr = x_tr_t.shape[0]
            for ep in range(60):
                model.train()
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr, 512):
                    idx = perm[i:i+512]
                    opt.zero_grad()
                    logit = model(x_tr_t[idx])
                    loss = F.binary_cross_entropy_with_logits(logit, y_tr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    p_va = torch.sigmoid(model(x_va_t)).numpy()
                # AUC
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(base_miss[va].astype(int), p_va)
                if auc > best_auc:
                    best_auc, best_st, no_imp = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
                else: no_imp += 1
                if no_imp >= 10: break
            model.load_state_dict(best_st); model.eval()
            with torch.no_grad():
                prob_seeds_va.append(torch.sigmoid(model(x_va_t)).numpy())
                prob_seeds_te.append(torch.sigmoid(model(x_te_t)).numpy())
        prob_oof[va] = np.mean(prob_seeds_va, axis=0)
        prob_te += np.mean(prob_seeds_te, axis=0) / 5  # 5-fold average
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(base_miss[va].astype(int), prob_oof[va])
        print(f"  fold{fi+1}: best AUC seed avg = {auc:.4f}  ({(time.time()-t0)/60:.1f}m)")
        gc.collect()

    from sklearn.metrics import roc_auc_score
    auc_total = roc_auc_score(base_miss.astype(int), prob_oof)
    print(f"\noverall miss-classifier OOF AUC: {auc_total:.4f}")

    # routing grid (rescue × T)
    rescues = {
        "v65h": (v65h_o, v65h_t),
        "v65s": (v65s_o, v65s_t),
        "v62":  (v62o, v62t),
        "v65hs_avg": ((v65h_o + v65s_o)/2, (v65h_t + v65s_t)/2),
        "v65h_v62_avg": ((v65h_o + v62o)/2, (v65h_t + v62t)/2),
        "v65s_v62_avg": ((v65s_o + v62o)/2, (v65s_t + v62t)/2),
        "all3_avg": ((v65h_o + v65s_o + v62o)/3, (v65h_t + v65s_t + v62t)/3),
    }

    print("\n=== Routing sweep ===")
    best = (rh_base, None, None, None, None)
    for rname, (ro, rt) in rescues.items():
        grid = np.percentile(prob_oof, [20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
        for q, T in zip([20,30,40,50,55,60,65,70,75,80,85,90,95], grid):
            use_r = prob_oof >= T
            n = use_r.sum()
            final_o = np.where(use_r[:, None], ro, base_o)
            r = rh(final_o, y_train).mean()
            if r > best[0]:
                # apply same T to test
                use_r_t = prob_te >= T
                final_t = np.where(use_r_t[:, None], rt, base_t)
                best = (r, rname, T, q, final_t)
                print(f"  ★ {rname:14s}  p{q:2d} T={T:.3f}  n={n:5d}  OOF={r:.4f}  Δ {r - rh_base:+.4f}")

    print(f"\n{'='*60}")
    if best[1] is None:
        print(f"NO LIFT — miss-classifier 학습이 routing benefit 못 추출.")
        print(f"   miss AUC {auc_total:.4f}, base {rh_base:.4f}")
    else:
        print(f"BEST: rescue={best[1]}  T={best[2]:.4f} (p{best[3]})  OOF={best[0]:.4f}  Δ {best[0] - rh_base:+.4f}")
        # test 적용 비율
        use_r_t_pct = (prob_te >= best[2]).mean() * 100
        print(f"test 적용 비율: {use_r_t_pct:.1f}%")

        out = DATA / f"submission_v71_miss_{best[1]}_T{best[2]:.3f}.csv"
        pd.DataFrame({"id": sub["id"], "x": best[4][:,0], "y": best[4][:,1], "z": best[4][:,2]}).to_csv(out, index=False)
        print(f"  [submission] {out.name}")

    print("="*60)

    np.savez(CACHE / "v71_miss_state.npz",
             prob_oof=prob_oof, prob_te=prob_te, auc=auc_total,
             best_oof=best[0], best_rescue=str(best[1]) if best[1] else "",
             best_T=best[2] if best[2] is not None else 0.0)

    entry = {
        "version": "v71_miss_classifier",
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": "BCE miss-classifier → confidence-gated rescue routing",
        "miss_auc": float(auc_total),
        "rh_base": float(rh_base),
        "best_oof": float(best[0]),
        "delta": float(best[0] - rh_base),
        "best_rescue": str(best[1]) if best[1] else None,
        "best_T": float(best[2]) if best[2] is not None else None,
    }
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
