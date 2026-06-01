"""v148_cree_xy2 — CREE 공개 baseline(HyperPhysics_xy2, 회전 기반 turn-rate 물리모델)을
우리 5-fold OOF 파이프라인에 포팅. 우리 frenet-ODE와 메커니즘이 근본적으로 달라 강한 decorrelation 기대.
공개 코드공유(Dacon) 기반 — 모델 구조는 보존, 데이터만 우리 것 연결.

usage:
  python scripts/v148_cree_xy2.py --mode quick   # 1 fold 빠른 검증 (OOF/decorr)
  python scripts/v148_cree_xy2.py --mode full    # 5 fold OOF -> cache/cree_xy2_state.npz
"""
from __future__ import annotations
import argparse, os, sys, time, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from v23_train import load_data
CACHE = Path("data/cache"); DATA = Path("data")

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ===================== CREE 모델 (공개 코드 그대로, device=cpu) =====================
class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, min_win=3, mode="extended", device="cpu", targets_ext=None):
        Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
        windows = []
        for i in range(len(X)):
            targets = (targets_ext or [4,5,6,7,8,9,10,12]) if mode == "extended" else [12, 10]
            for target_idx in targets:
                end_idx = target_idx - 2
                max_w = end_idx + 2 if mode == "extended" else (12 if target_idx == 12 else 10)
                for w in range(min_win, max_w):
                    windows.append((i, w, target_idx))
        Xl, yl = [], []
        for i, w, target_idx in windows:
            Xo = Xt[i]; end_idx = target_idx - 2
            pts = Xo[end_idx - w + 1: end_idx + 1]
            target = yt[i] if target_idx == 12 else Xo[target_idx]
            if w < 11:
                v0 = pts[1] - pts[0]; n_pad = 11 - w
                js = torch.arange(n_pad, 0, -1, dtype=torch.float32)
                pad = pts[0:1] - js.unsqueeze(1) * v0.unsqueeze(0)
                Xp = torch.cat([pad, pts], dim=0)
            else:
                Xp = pts.clone()
            Xl.append(Xp); yl.append(target)
        self.X_all = torch.stack(Xl).to(device); self.y_all = torch.stack(yl).to(device)
        diffs = self.X_all[:, 1:] - self.X_all[:, :-1]
        n1 = diffs[:, 1:].norm(dim=2).clamp(min=1e-8); n2 = diffs[:, :-1].norm(dim=2).clamp(min=1e-8)
        cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2) / (n1 * n2)).clamp(-1, 1)
        theta_last = torch.acos(cos_t[:, -1])
        self.theta_weights = (1.0 + 4.0 * (theta_last / 1.0).clamp(0, 1)).cpu().numpy()
    def __len__(self): return len(self.X_all)
    def __getitem__(self, idx): return self.X_all[idx], self.y_all[idx]

def _ema_va_local(diffs_local, alpha, beta):
    B, T, _ = diffs_local.shape; one_m_a = 1.0 - alpha; one_m_b = 1.0 - beta
    vs = diffs_local.new_empty(B, T, 3); v = diffs_local[:, 0]; vs[:, 0] = v
    for t in range(1, T):
        v = alpha * diffs_local[:, t] + one_m_a * v; vs[:, t] = v
    vl = vs[:, -1]; ad = vs[:, 1:] - vs[:, :-1]; a = ad[:, 0]
    for t in range(1, T - 1):
        a = beta * ad[:, t] + one_m_b * a
    return vl, a

def _soft_hit_loss(pred, target, thr=0.013012, k=408.348):
    return (1 - torch.sigmoid(-(torch.norm(pred - target, dim=1) - thr) * k)).mean()

def extract_features(X, mean_stats, std_stats, dir_net, heading_mode="3step"):
    device = X.device; p_last = X[:, 10]; diffs = X[:, 1:] - X[:, :-1]
    n1 = diffs[:, 1:].norm(dim=2, keepdim=True) + 1e-8; n2 = diffs[:, :-1].norm(dim=2, keepdim=True) + 1e-8
    cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2, keepdim=True) / (n1 * n2)).clamp(-1, 1)
    theta_seq = torch.acos(cos_t).squeeze(2)
    theta = theta_seq[:, -1:]; theta_mean = theta_seq.mean(1, keepdim=True); theta_std = theta_seq.std(1, keepdim=True)
    theta_vel = theta_seq[:, -1:] - theta_seq[:, -2:-1]
    theta_acc = theta_seq[:, -1:] - 2*theta_seq[:, -2:-1] + theta_seq[:, -3:-2]
    theta_trend = theta_seq[:, -1:] - theta_seq[:, -3:].mean(1, keepdim=True)
    if dir_net is not None:
        speed_seq = diffs.norm(dim=2); state = torch.cat([speed_seq, theta_seq], dim=1)
        if dir_net[0].in_features == 29:
            state = torch.cat([state, diffs[:, :, 2].abs()], dim=1)
        weights = F.softmax(dir_net(state), dim=1); v_sm = (diffs * weights.unsqueeze(2)).sum(dim=1)
    else:
        v_sm = (3*diffs[:, -1] + 2*diffs[:, -2] + diffs[:, -3]) / 6.0 if heading_mode == "3step" else diffs[:, -1]
    fwd = v_sm / (v_sm.norm(dim=1, keepdim=True) + 1e-8)
    up_w = torch.zeros_like(fwd); up_w[:, 2] = 1.0
    up_w[fwd[:, 2].abs() > 0.99] = torch.tensor([0., 1., 0.], device=device)
    right = torch.cross(fwd, up_w, dim=1); right = right / (right.norm(dim=1, keepdim=True) + 1e-8)
    up = torch.cross(right, fwd, dim=1); up = up / (up.norm(dim=1, keepdim=True) + 1e-8)
    R = torch.stack([fwd, right, up], dim=2)
    v_last = diffs[:, -1]; v_prev1 = diffs[:, -2]; speed = v_last.norm(dim=1, keepdim=True)
    a_last = v_last - v_prev1; acc_mag = a_last.norm(dim=1, keepdim=True)
    v_local = torch.matmul(v_last.unsqueeze(1), R).squeeze(1); a_local = torch.matmul(a_last.unsqueeze(1), R).squeeze(1)
    X_local = torch.matmul(X - p_last.unsqueeze(1), R); p_std_local = X_local.std(1); v_local_abs = v_local.abs()
    jerk_g = diffs[:, -1] - 2*diffs[:, -2] + diffs[:, -3]; jerk_l = torch.matmul(jerk_g.unsqueeze(1), R).squeeze(1)
    jerk_mag = jerk_g.norm(dim=1, keepdim=True)
    features = torch.cat([v_local, a_local, speed, acc_mag, theta, theta_mean, theta_std, theta_trend,
                          theta_vel, theta_acc, p_std_local, v_local_abs, jerk_l, jerk_mag], dim=1)
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(0, keepdim=True); std_stats = features.std(0, keepdim=True) + 1e-8
    return (features - mean_stats) / std_stats, diffs, p_last, theta, theta_mean, theta_std, theta_seq, R, speed, mean_stats, std_stats

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
    def forward(self, x): return self.ln(x + self.net(x))

class PriorBiasedLinear(nn.Module):
    def __init__(self, in_f, out_f, prior_bias):
        super().__init__(); self.linear = nn.Linear(in_f, out_f)
        self.register_buffer('prior_bias', prior_bias.clone().detach())
        with torch.no_grad(): nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x) + self.prior_bias

def rodrigues_rotate(v, w):
    theta = w.norm(dim=1, keepdim=True); k = w / (theta + 1e-8)
    cos_t = torch.cos(theta); sin_t = torch.sin(theta); dot = (v * k).sum(dim=1, keepdim=True); cross = torch.cross(k, v, dim=1)
    return v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)

class HyperPhysics_xy2(nn.Module):
    def __init__(self, input_dim=24, **kw):
        super().__init__()
        self.sh_thr=kw.pop('sh_thr',0.013012); self.sh_k=kw.pop('sh_k',408.348044); self.mse_w=kw.pop('mse_w',129.172037)
        self.local_w=kw.pop('local_w',0.050941); self.theta_thr=kw.pop('theta_thr',1.087618); self.speed_thr=kw.pop('speed_thr',0.034583)
        self.lr=0.005400; self.wd=0.005659
        self.register_buffer("mean_stats", torch.zeros(1, input_dim)); self.register_buffer("std_stats", torch.ones(1, input_dim))
        self.use_dirnet = True   # False면 고정 3-step heading (다른 프레임 → decorrelated 2nd 물리)
        prior_dir = torch.tensor([-10.,-10.,-10.,-10.,-10.,-10.,-10.,0.,0.693,1.098])
        self.dir_net = nn.Sequential(nn.Linear(29,24), nn.LayerNorm(24), nn.GELU(), PriorBiasedLinear(24,10,prior_dir))
        self.temporal_net = nn.Sequential(nn.Linear(9,32), nn.LayerNorm(32), nn.GELU(), PriorBiasedLinear(32,6,torch.zeros(6)))
        prior_dyn = torch.tensor([0.,0.,0.,0.,0.,0.]+[-4.]*24)
        self.dynamics_net = nn.Sequential(nn.Linear(input_dim,96), nn.LayerNorm(96), nn.GELU(), ResBlock(96), PriorBiasedLinear(96,30,prior_dyn))
        self.omega_w = nn.Parameter(torch.tensor([0.0,-0.5,-1.0]))
        self.omega_net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim,48), nn.GELU(), nn.Linear(48,3))
        with torch.no_grad(): nn.init.normal_(self.omega_net[-1].weight, std=0.01); nn.init.zeros_(self.omega_net[-1].bias)
        self.diffusion_net = nn.Sequential(nn.Linear(input_dim,32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32,3))
    def get_features(self, X, mean_stats=None, std_stats=None):
        return extract_features(X, mean_stats, std_stats, self.dir_net if self.use_dirnet else None, "3step")
    @staticmethod
    def _rotation_vector(d_prev, d_curr):
        n_prev = d_prev.norm(dim=1, keepdim=True).clamp(min=1e-8); n_curr = d_curr.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d_hat_prev = d_prev/n_prev; d_hat_curr = d_curr/n_curr; cross = torch.linalg.cross(d_hat_prev, d_hat_curr, dim=1)
        sin_t = cross.norm(dim=1, keepdim=True).clamp(min=1e-8); cos_t = (d_hat_prev*d_hat_curr).sum(1, keepdim=True).clamp(-0.9999, 0.9999)
        theta = torch.atan2(sin_t, cos_t); speed_gate = torch.sigmoid((n_prev+n_curr)*500-5)
        return cross/sin_t*theta*speed_gate
    def forward(self, features, diffs, p_last, theta, speed, R):
        B = diffs.shape[0]
        ema_raw = self.temporal_net(features[:, 8:17]); alpha = torch.sigmoid(ema_raw[:, 0:3])*0.8+0.1; beta = torch.sigmoid(ema_raw[:, 3:6])*0.199+0.8
        dyn_raw = self.dynamics_net(features); w_v = 2.0+dyn_raw[:, 0:3]; w_a = 1.0+dyn_raw[:, 3:6]
        v_local_abs = features[:, 17:20]; v_local_abs2 = v_local_abs*v_local_abs; theta2 = theta*theta
        exp_v = (F.softplus(dyn_raw[:,6:9])*v_local_abs + F.softplus(dyn_raw[:,9:12])*v_local_abs2 + F.softplus(dyn_raw[:,12:15])*theta + F.softplus(dyn_raw[:,15:18])*theta2)
        exp_a = (F.softplus(dyn_raw[:,18:21])*v_local_abs + F.softplus(dyn_raw[:,21:24])*v_local_abs2 + F.softplus(dyn_raw[:,24:27])*theta + F.softplus(dyn_raw[:,27:30])*theta2)
        diffs_local = torch.matmul(diffs, R); vl, al = _ema_va_local(diffs_local, alpha, beta); diff_speed = diffs_local.norm(dim=2)
        def rv_masked(ka, kb):
            rv = self._rotation_vector(diffs_local[:, ka], diffs_local[:, kb])
            valid = ((diff_speed[:, ka] > 1e-5) & (diff_speed[:, kb] > 1e-5)).float()
            return rv*valid.unsqueeze(1), valid
        ov1, vm1 = rv_masked(-2,-1); ov2, vm2 = rv_masked(-3,-2); ov3, vm3 = rv_masked(-4,-3)
        w_logits = self.omega_w.view(1,3).expand(B,-1); masks = torch.stack([vm1, vm2, vm3], dim=1)
        w_attn = F.softmax(w_logits.masked_fill(masks == 0, -1e9), dim=1)
        omega_hist = w_attn[:,0].unsqueeze(1)*ov1 + w_attn[:,1].unsqueeze(1)*ov2 + w_attn[:,2].unsqueeze(1)*ov3
        current_speed = speed.view(B,1) if speed is not None else diff_speed[:,-1].unsqueeze(1)
        omega_delta = self.omega_net(features) * torch.sigmoid(current_speed*500-5)
        theta_scalar = theta.view(B,1)
        rotation_gate = torch.sigmoid((theta_scalar-self.theta_thr)*10) * torch.sigmoid((current_speed-self.speed_thr)*200)
        omega = (omega_hist + omega_delta) * rotation_gate
        v_rotated = rodrigues_rotate(vl, omega)
        pred_local = (w_v*torch.exp(-exp_v))*v_rotated + (w_a*torch.exp(-exp_a))*al
        log_var = self.diffusion_net(features).clamp(min=-5.0, max=5.0)
        pred_global = p_last + torch.einsum('nij,nj->ni', R, pred_local)
        return pred_global, pred_local, log_var
    def compute_loss(self, pp, yr, pred_local=None, yr_local=None, log_var=None, **kw):
        loss = _soft_hit_loss(pp, yr, self.sh_thr, self.sh_k) + self.mse_w*F.mse_loss(pp, yr)
        if pred_local is not None and yr_local is not None and log_var is not None:
            nll = 0.5*(torch.exp(-log_var)*(pred_local-yr_local)**2 + log_var); loss = loss + self.local_w*nll.mean()
        return loss

# ===================== 5-fold OOF 파이프라인 =====================
@torch.no_grad()
def predict_full(model, X, device):
    Xt = torch.tensor(X, dtype=torch.float32, device=device); preds = []
    for i in range(0, len(Xt), 512):
        b = Xt[i:i+512]
        ft, df, pl, th, _, _, _, R, sp, _, _ = model.get_features(b, model.mean_stats, model.std_stats)
        pp, _, _ = model(ft, df, pl, th, sp, R); preds.append(pp.cpu().numpy())
    return np.concatenate(preds, 0)

def hit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())

def train_fold(Xtr, ytr, epochs, min_win, targets_ext, device, use_dirnet=True):
    ds = SlidingWindowDataset(Xtr, ytr, min_win=min_win, mode="extended", device=device, targets_ext=targets_ext)
    sampler = WeightedRandomSampler(ds.theta_weights, len(ds), replacement=True)
    loader = DataLoader(ds, batch_size=256, sampler=sampler)
    model = HyperPhysics_xy2().to(device); model.use_dirnet = use_dirnet
    with torch.no_grad():
        _, _, _, _, _, _, _, _, _, mn, st = model.get_features(torch.tensor(Xtr, dtype=torch.float32, device=device))
        model.mean_stats.copy_(mn); model.std_stats.copy_(st)
    opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    for ep in range(1, epochs+1):
        model.train(); tl = 0.0
        for Xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            ft, df, pl, th, _, _, _, R, sp, _, _ = model.get_features(Xb, model.mean_stats, model.std_stats)
            pp, prl, lv = model(ft, df, pl, th, sp, R)
            yr_local = torch.matmul((yb - pl).unsqueeze(1), R).squeeze(1)
            loss = model.compute_loss(pp, yb, prl, yr_local, lv)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()*len(Xb)
        sch.step()
        if ep <= 3 or ep % 4 == 0: print(f"    ep{ep}/{epochs} loss={tl/len(ds):.4f}", flush=True)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick", choices=["quick", "full"])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--min_win", type=int, default=5)
    ap.add_argument("--tag", default="xy2")
    ap.add_argument("--aug", default="reduced", choices=["reduced", "full"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heading", default="dirnet", choices=["dirnet", "3step"])
    args = ap.parse_args()
    set_seed(args.seed); device = torch.device("cpu"); torch.set_num_threads(os.cpu_count() or 4)
    targets_ext = [4, 5, 6, 7, 8, 9, 10, 12] if args.aug == "full" else [6, 7, 8, 9, 10, 12]
    print(f"=== CREE HyperPhysics_xy2 | mode={args.mode} epochs={args.epochs} min_win={args.min_win} ===", flush=True)
    X_train, X_test, y_train, sub = load_data()
    N = len(X_train); kf = KFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(kf.split(np.arange(N)))
    if args.mode == "quick": folds = folds[:1]
    oof = np.zeros((N, 3), dtype=np.float32); mask = np.zeros(N, dtype=bool); test_acc = []; t0 = time.time()
    for fi, (tr, va) in enumerate(folds):
        print(f"  [fold {fi}] train={len(tr)} val={len(va)}", flush=True)
        model = train_fold(X_train[tr], y_train[tr], args.epochs, args.min_win, targets_ext, device, use_dirnet=(args.heading == "dirnet"))
        model.eval()
        pv = predict_full(model, X_train[va], device); oof[va] = pv; mask[va] = True
        test_acc.append(predict_full(model, X_test, device))
        print(f"  [fold {fi}] val hit={hit(pv, y_train[va]):.4f}  elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    rh = hit(oof[mask], y_train[mask])
    print(f"\n[CREE xy2] OOF hit = {rh:.4f} (covered {mask.sum()}/{N})", flush=True)
    test_pred = np.mean(test_acc, axis=0)
    # decorr vs 우리 멤버
    for ref in ["v120_full_state.npz", "v131_frenet_gru_state.npz"]:
        p = CACHE / ref
        if p.exists():
            t = np.load(p)["test_global"]; print(f"  decorr L2(test vs {ref.split('_')[0]}) = {np.linalg.norm(test_pred-t,axis=-1).mean()*1000:.2f}mm", flush=True)
    if args.mode == "full":
        np.savez(CACHE / f"cree_{args.tag}_state.npz", oof_global=oof, fold_mask=mask, test_global=test_pred, rh_oof=rh)
        pd.DataFrame({"id": sub["id"], "x": test_pred[:,0], "y": test_pred[:,1], "z": test_pred[:,2]}).to_csv(DATA/f"submission_cree_{args.tag}.csv", index=False)
        print(f"  saved cree_{args.tag}_state.npz + submission", flush=True)

if __name__ == "__main__":
    main()
