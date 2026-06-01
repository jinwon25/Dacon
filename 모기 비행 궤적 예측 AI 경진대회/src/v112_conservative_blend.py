"""v112_conservative_blend.py - Conservative ensemble blend (variance-aware).

motivation (2026-05-25 실측):
  - DE pool 늘릴수록 변환률 감쇠:
      5w v98 +0.0122 → 15w v106 +0.0118 → 29w v110_v3 +0.0109
  - 핵심: OOF에 잘 fit할수록 LB에서 일반화 손해

design:
  - top-K only: OOF best K (default 6-8) 모델만 사용 — 약한 모델 (OOF<threshold) 자동 제외
  - softmax weights + single weight cap (max 0.30) via clip-renorm
  - L1 reg on weights (sparsity but not too sparse, min 5 non-trivial)
  - shallower DE search (n_iter=150, n_starts=3)
  - also: simple grid search over top-5 models as alt

usage:
  python scripts/v112_conservative_blend.py --tag v112_top7 --top-k 7 --weight-cap 0.30
  python scripts/v112_conservative_blend.py --tag v112_simple_grid --simple-grid
"""
from __future__ import annotations

import argparse, datetime as _dt, glob, json, os, sys, itertools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from v110_de_ensemble import load_pool, load_y_and_sub

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data/cache"
DATA = PROJECT / "data"


def softmax_capped(z, cap=0.30, iters=10):
    """softmax → clip at cap → re-normalize. Iterate until all <= cap."""
    e = np.exp(z - z.max())
    w = e / e.sum()
    for _ in range(iters):
        over = w > cap
        if not over.any(): break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.sum() == 0: break
        w[under] += excess * (w[under] / w[under].sum())
    return w


def hit_rate(pred, y):
    return float((np.linalg.norm(pred - y, axis=-1) <= 0.01).mean())


def fit_de_conservative(pool, y, n_iter=150, popsize=25, n_starts=3,
                          cap=0.30, l1_w=0.0001, verbose=True):
    import time as _time
    N = len(pool)
    oofs = np.stack([t[1] for t in pool])
    tests = np.stack([t[2] for t in pool])

    def neg_hit(z):
        w = softmax_capped(z, cap=cap)
        pred = (w[:, None, None] * oofs).sum(axis=0)
        # L1 penalty: encourage some sparsity but not too aggressive
        # penalize z spread (small z = uniform). subtract small reward for spread.
        h = hit_rate(pred, y)
        # active fraction (effective L1: count of weights > 0.01)
        active = (w > 0.01).sum()
        # encourage min 5 active; soft penalty if below
        active_penalty = max(0, 5 - active) * 0.001
        return -h + active_penalty

    bounds = [(-5.0, 5.0)] * N
    best = (None, np.inf, None)
    for s in range(n_starts):
        if verbose:
            print(f"  DE start {s+1}/{n_starts} (seed={s}) ...", flush=True)
            t0 = _time.time()
        res = differential_evolution(
            neg_hit, bounds, seed=s,
            maxiter=n_iter, popsize=popsize, tol=1e-6, mutation=(0.3, 1.5),
            recombination=0.8, init="sobol", polish=True, workers=1,
        )
        if res.fun < best[1]:
            best = (res.x, res.fun, s)
        if verbose:
            print(f"    best so far: hit={-res.fun:.4f}  ({_time.time()-t0:.1f}s)", flush=True)

    w_best = softmax_capped(best[0], cap=cap)
    oof_pred = (w_best[:, None, None] * oofs).sum(axis=0)
    test_pred = (w_best[:, None, None] * tests).sum(axis=0)
    return w_best, oof_pred, test_pred, -best[1]


def simple_grid(pool, y, k=5, step=0.1):
    """top-K model uniform grid (each weight in [0, 1] in step). Sum=1 enforced."""
    N = len(pool)
    oofs = np.stack([t[1] for t in pool])  # (N, 10000, 3)
    tests = np.stack([t[2] for t in pool])

    grid_vals = np.arange(0, 1.0 + 1e-9, step)
    best_w, best_rh = None, -1
    # iterate compositions summing to 1 (rounded to step)
    n_steps = int(round(1.0 / step))
    for combo in itertools.combinations_with_replacement(range(n_steps + 1), N):
        if sum(combo) != n_steps: continue
        w = np.array(combo) / n_steps
        pred = (w[:, None, None] * oofs).sum(axis=0)
        rh = hit_rate(pred, y)
        if rh > best_rh:
            best_rh = rh; best_w = w
    test_pred = (best_w[:, None, None] * tests).sum(axis=0)
    oof_pred = (best_w[:, None, None] * oofs).sum(axis=0)
    return best_w, oof_pred, test_pred, best_rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v112_top7")
    parser.add_argument("--top-k", type=int, default=7,
                        help="use top-K models by single OOF")
    parser.add_argument("--min-oof", type=float, default=0.67,
                        help="exclude models with OOF below this")
    parser.add_argument("--weight-cap", type=float, default=0.30)
    parser.add_argument("--l1-w", type=float, default=0.0001)
    parser.add_argument("--n-iter", type=int, default=150)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=25)
    parser.add_argument("--include-mdn", type=int, default=0,
                        help="0 to exclude v109 MDN (weaker single OOF)")
    parser.add_argument("--simple-grid", action="store_true",
                        help="instead of DE, use simple grid over top-K models with step=0.1")
    parser.add_argument("--force-include", default="",
                        help="comma-separated model names to force-include (paradigm diversity)")
    parser.add_argument("--grid-step", type=float, default=0.1)
    args = parser.parse_args()

    pool_all, y = load_pool(include_mdn=bool(args.include_mdn))
    # filter by min_oof
    pool_filt = [(nm, oof, te, rh) for (nm, oof, te, rh) in pool_all if rh >= args.min_oof]
    pool_filt.sort(key=lambda t: -t[3])  # desc by OOF
    pool = pool_filt[:args.top_k]
    # force-include named paradigms (preserve diversity)
    if args.force_include:
        included_names = {t[0] for t in pool}
        force_list = [n.strip() for n in args.force_include.split(",") if n.strip()]
        for fn in force_list:
            if fn in included_names: continue
            found = next((t for t in pool_all if t[0] == fn), None)
            if found is None:
                print(f"  [warn] force-include '{fn}' not found in pool"); continue
            pool.append(found)
            print(f"  [force] +{fn} (OOF={found[3]:.4f})")

    print(f"\n=== filtered pool: {len(pool)} models (top-{args.top_k}, OOF>={args.min_oof}) ===")
    for nm, _, _, rh in pool:
        print(f"  {nm:<25} OOF={rh:.4f}")
    print(f"  (excluded {len(pool_all) - len(pool)} models with OOF<{args.min_oof} or beyond top-{args.top_k})\n")

    if args.simple_grid:
        if len(pool) > 6:
            print(f"  WARNING: simple-grid with {len(pool)} models is too slow. Reducing to top-5.")
            pool = pool[:5]
        print(f"  simple grid search over {len(pool)} models (step={args.grid_step}) ...")
        w, oof_pred, test_pred, oof_rh = simple_grid(pool, y, k=len(pool), step=args.grid_step)
    else:
        w, oof_pred, test_pred, oof_rh = fit_de_conservative(
            pool, y, n_iter=args.n_iter, popsize=args.popsize,
            n_starts=args.n_starts, cap=args.weight_cap, l1_w=args.l1_w,
        )

    print(f"\n=== {args.tag} ===")
    print(f"  OOF R-Hit: {oof_rh:.4f}")
    print(f"  weights:")
    order = np.argsort(-w)
    n_active = 0
    for i in order:
        if w[i] >= 0.005:
            print(f"    {pool[i][0]:<25} {w[i]:.3f}  (single OOF={pool[i][3]:.4f})")
            n_active += 1
    # 변환률 예측
    # 보수 blend (5-8 active models, max weight <= cap) → 변환률 추정 +0.0118~+0.0122
    est_lb = oof_rh + 0.0118
    print(f"  active weights (>=0.005): {n_active}")
    print(f"  est LB (보수 blend +0.0118 가정): {est_lb:.4f}")

    _, sub, _ = load_y_and_sub()
    out_csv = DATA / f"submission_{args.tag}_oof{oof_rh:.4f}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pred[:,0], "y": test_pred[:,1], "z": test_pred[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv.name}")

    weights_npz = CACHE / f"{args.tag}_weights.npz"
    np.savez(weights_npz,
             names=np.array([t[0] for t in pool]),
             single_oof=np.array([t[3] for t in pool]),
             weights=w, oof_pred=oof_pred, test_pred=test_pred, oof_rh=oof_rh)
    print(f"  [weights] {weights_npz.name}")

    entry = {
        "version": args.tag, "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"conservative DE (top-{args.top_k}, cap={args.weight_cap}, simple_grid={args.simple_grid})",
        "n_models": len(pool), "n_active": int(n_active),
        "oof_rh": float(oof_rh), "est_lb": float(est_lb),
        "weights": {nm: float(w[i]) for i, (nm, *_) in enumerate(pool)},
    }
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
