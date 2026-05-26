"""v110_de_ensemble.py - Multi-paradigm DE blend (v106 패턴 + v109 추가).

approach (v106와 동일):
  1. all paradigm caches OOF/test load.
  2. softmax-parametrized weights (n-1 free + 1 implied) for sum=1, all >=0.
  3. multi-seed differential_evolution to find weights minimizing -hit_rate.
  4. 또한 7w / 11w 등 subset variants도 옵션으로 동시 생성.
  5. best 결과 OOF report + LB 변환률 추정 (+0.0118) + submission CSV.

usage:
  python scripts/v110_de_ensemble.py --tag v110_18w_DE
  python scripts/v110_de_ensemble.py --tag v110_18w_DE --include-mdn 1 --max-iter 300

새 cache 비교 가능 - 결과에 따라 v109 weight 등 보고.
"""
from __future__ import annotations

import argparse, datetime as _dt, glob, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "cache"
DATA = PROJECT / "open"


def softmax_weights(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def load_y_and_sub():
    labels = pd.read_csv(DATA / "train_labels.csv")
    sub = pd.read_csv(DATA / "sample_submission.csv")
    train_files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    ids = np.array([os.path.splitext(os.path.basename(f))[0] for f in train_files])
    y = labels.set_index("id").loc[list(ids)][["x","y","z"]].values.astype(np.float64)
    return y, sub, ids


def load_pool(include_mdn=True):
    """모든 paradigm cache load. (name, oof, test) tuples list 반환."""
    y, _, ids = load_y_and_sub()
    pool = []

    # ALPHA 보정 base
    ALPHA = np.array([1.000, 0.950, 1.000])
    kc = np.load(CACHE / "kalman.npz")
    kalman_tr, kalman_te = kc["kalman_train"], kc["kalman_test"]

    # 1. boundary refinements (이미 base 추가된 oof들)
    # v94 (v90 mirror A+B base + boundary cap 1.0)
    boundary_caches = [
        ("v94", "v94_state.npz"),
        ("v94c5", "v94_cap1p5_state.npz"),
        ("v97", "v97_state.npz"),
        ("v97c0p5", "v97_cap0p5_state.npz"),
        ("v97c5", "v97_cap1p5_state.npz"),
        ("v101", "v101_state.npz"),
        ("v101c5", "v101_cap1p5_state.npz"),
        ("v103b", "v103b_state.npz"),
        ("v103bc5", "v103b_cap1p5_state.npz"),
        ("v104b", "v104b_state.npz"),
        ("v104bc5", "v104b_cap1p5_state.npz"),
        ("v113", "v113_cap10_state.npz"),     # boundary on v107 deep Trans cap 1.0
        ("v113c5", "v113_cap15_state.npz"),   # cap 1.5
        ("v121", "v121_cap10_state.npz"),     # boundary on v120 Neural ODE cap 1.0
        ("v121c5", "v121_cap15_state.npz"),   # boundary on v120 Neural ODE cap 1.5
    ]
    for name, fname in boundary_caches:
        p = CACHE / fname
        if not p.exists():
            print(f"  [skip] {fname} missing"); continue
        st = np.load(p)
        oof_k = None; test_k = None
        for k in ["oof_v91","oof_v94","oof_v97","oof","oof_pred"]:
            if k in st.files: oof_k = k; break
        for k in ["test_v91","test_v94","test_v97","test_pred","test"]:
            if k in st.files: test_k = k; break
        if oof_k is None or test_k is None:
            print(f"  [skip] {fname}: no oof/test key"); continue
        oof = st[oof_k]; test = st[test_k]
        rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
        pool.append((name, oof, test, float(rh)))

    # 2. stacker output (v53)
    p = CACHE / "v53_state.npz"
    if p.exists():
        st = np.load(p)
        # v53 keys: oof_v53/test_v53
        oof_k = "oof_v53" if "oof_v53" in st.files else "oof"
        test_k = "test_v53" if "test_v53" in st.files else "test_pred"
        oof = st[oof_k]; test = st[test_k]
        rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
        pool.append(("v53", oof, test, float(rh)))

    # 3. boundary v35 (legacy, train-only paradigm)
    p = CACHE / "v35_state.npz"
    if p.exists():
        st = np.load(p)
        oof_k = "oof_v35" if "oof_v35" in st.files else "oof"
        test_k = "test_v35" if "test_v35" in st.files else "test_pred"
        oof = st[oof_k]; test = st[test_k]
        rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
        pool.append(("v35", oof, test, float(rh)))

    # 4. v78 (boundary on v77 BiGRU)
    p = CACHE / "v78_state.npz"
    if p.exists():
        st = np.load(p)
        oof_k = "oof_v78" if "oof_v78" in st.files else "oof"
        test_k = "test_v78" if "test_v78" in st.files else "test_pred"
        oof = st[oof_k]; test = st[test_k]
        rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
        pool.append(("v78", oof, test, float(rh)))

    # 5. MDN-WTA outputs (v109): glob all *_pool.npz so K=4/K=8/setupB auto-loaded
    #    exclude smoke/test artifacts (단독 OOF 너무 낮으면 noise)
    if include_mdn:
        for fname in sorted(glob.glob(str(CACHE / "v109_*_pool.npz"))):
            tag = Path(fname).stem.replace("_pool", "")
            if "smoke" in tag:
                print(f"  [skip] {tag} (smoke)"); continue
            st = np.load(fname)
            for variant, oof_k, test_k in [("wm","oof_wm","test_wm"),
                                            ("am","oof_am","test_am")]:
                if oof_k not in st.files: continue
                oof = st[oof_k]; test = st[test_k]
                rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
                pool.append((f"{tag}_{variant}", oof, test, float(rh)))

    # 6. per-axis boundary (v108 variants on v90, v114 on v107)
    for fname in sorted(glob.glob(str(CACHE / "v108_*_state.npz"))) + \
                 sorted(glob.glob(str(CACHE / "v114_*_state.npz"))):
        st = np.load(fname)
        oof = st["oof"]; test = st["test_pred"]
        rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
        name = Path(fname).stem.replace("_state","")
        pool.append((name, oof, test, float(rh)))

    # 6b. v120 Neural ODE (raw, no boundary) — paradigm diversity check
    p = CACHE / "v120_full_state.npz"
    if p.exists():
        st = np.load(p)
        oof = st["oof_global"].astype(np.float64)
        test = st["test_global"].astype(np.float64)
        rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
        pool.append(("v120", oof, test, float(rh)))

    # 6c. v120 변종 (multi-step, big capacity) + v126 FFT — paradigm pool 확장
    for tag, fname, key_oof, key_test in [
        ("v120_n2",   "v120_n2_full_state.npz",  "oof_global", "test_global"),
        ("v120_big",  "v120_big_full_state.npz", "oof_global", "test_global"),
        ("v126_fft",  "v126_full_state.npz",     "oof_global", "test_global"),
    ]:
        pp = CACHE / fname
        if pp.exists():
            st = np.load(pp)
            if key_oof in st.files:
                oof = st[key_oof].astype(np.float64)
                test = st[key_test].astype(np.float64)
                rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
                pool.append((tag, oof, test, float(rh)))

    # 7. boundary on v109 MDN (v111 + v111_K8 variants)
    for pat in ["v111_*_state.npz", "v111_K8_*_state.npz"]:
        for fname in sorted(glob.glob(str(CACHE / pat))):
            name = Path(fname).stem.replace("_state","")
            # avoid duplicate (v111_K8_* matches both patterns)
            if any(name == p[0] for p in pool): continue
            st = np.load(fname)
            oof = st["oof"]; test = st["test_pred"]
            rh = (np.linalg.norm(oof - y, axis=-1) <= 0.01).mean()
            pool.append((name, oof, test, float(rh)))

    return pool, y


def hit_rate(pred, y):
    return float((np.linalg.norm(pred - y, axis=-1) <= 0.01).mean())


def fit_de(pool, y, n_iter=300, popsize=40, n_starts=5, seed_base=0, verbose=True):
    N = len(pool)
    oofs = np.stack([t[1] for t in pool])  # (N, 10000, 3)
    tests = np.stack([t[2] for t in pool])  # (N, 10000, 3)

    def neg_hit(z):
        w = softmax_weights(z)
        pred = (w[:, None, None] * oofs).sum(axis=0)
        return -hit_rate(pred, y)

    bounds = [(-5.0, 5.0)] * N
    best = (None, np.inf, None)
    for s in range(n_starts):
        if verbose: print(f"  DE start {s+1}/{n_starts} (seed={seed_base+s}) ...", flush=True)
        t_start = __import__('time').time()
        res = differential_evolution(
            neg_hit, bounds, seed=seed_base + s,
            maxiter=n_iter, popsize=popsize, tol=1e-6, mutation=(0.3, 1.5),
            recombination=0.8, init="sobol", polish=True, workers=1,
        )
        if res.fun < best[1]:
            best = (res.x, res.fun, s)
        elapsed = __import__('time').time() - t_start
        if verbose: print(f"    best so far: hit={-res.fun:.4f}  ({elapsed:.1f}s)", flush=True)

    w_best = softmax_weights(best[0])
    oof_pred = (w_best[:, None, None] * oofs).sum(axis=0)
    test_pred = (w_best[:, None, None] * tests).sum(axis=0)
    return w_best, oof_pred, test_pred, -best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v110_DE", help="output csv tag")
    parser.add_argument("--include-mdn", type=int, default=1, help="1 to include v109 outputs")
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=40)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--save-suffix", default="")
    args = parser.parse_args()

    pool, y = load_pool(include_mdn=bool(args.include_mdn))
    print(f"\n=== loaded {len(pool)} models ===")
    for nm, _, _, rh in pool:
        print(f"  {nm:<25} OOF={rh:.4f}")
    print()

    # full N-way DE
    w, oof_pred, test_pred, oof_rh = fit_de(
        pool, y, n_iter=args.n_iter, popsize=args.popsize,
        n_starts=args.n_starts, verbose=True,
    )

    print(f"\n=== {args.tag} ===")
    print(f"  OOF R-Hit: {oof_rh:.4f}")
    print(f"  weights (sorted desc):")
    order = np.argsort(-w)
    for i in order:
        if w[i] >= 0.005:
            print(f"    {pool[i][0]:<25} {w[i]:.3f}  (single OOF={pool[i][3]:.4f})")
    # LB 추정
    est_lb_min = oof_rh + 0.0115; est_lb_max = oof_rh + 0.0122
    print(f"  est LB: {est_lb_min:.4f} ~ {est_lb_max:.4f} (변환률 +0.0115~+0.0122)")

    # save submission
    _, sub, _ = load_y_and_sub()
    out_csv = DATA / f"submission_{args.tag}_oof{oof_rh:.4f}{args.save_suffix}.csv"
    pd.DataFrame({"id": sub["id"], "x": test_pred[:,0], "y": test_pred[:,1], "z": test_pred[:,2]}
                 ).to_csv(out_csv, index=False)
    print(f"  [submission] {out_csv.name}")

    # save weights
    weights_npz = CACHE / f"{args.tag}_weights{args.save_suffix}.npz"
    np.savez(weights_npz,
             names=np.array([t[0] for t in pool]),
             single_oof=np.array([t[3] for t in pool]),
             weights=w, oof_pred=oof_pred, test_pred=test_pred, oof_rh=oof_rh)
    print(f"  [weights] {weights_npz.name}")

    entry = {
        "version": args.tag, "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "approach": f"DE blend N={len(pool)} models, multi-seed start",
        "n_models": len(pool),
        "oof_rh": float(oof_rh),
        "est_lb_min": float(est_lb_min), "est_lb_max": float(est_lb_max),
        "weights": {nm: float(w[i]) for i, (nm, *_) in enumerate(pool)},
    }
    log_path = PROJECT / "run_log.json"
    logs = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
    logs.append(entry)
    json.dump(logs, open(log_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
