"""v122d_blend_quick.py — 빠른 11-member DE blend (n_starts=2 n_iter=150).

기존 v122d_blend_after_training.py가 너무 오래 걸려서 우회용. 결과 비슷할 가능성 큼.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
from v110_de_ensemble import load_pool, softmax_weights, hit_rate

PROJ = SCRIPT.parent
CACHE = PROJ / "data/cache"
OPEN = PROJ / "data"
REPORTS = PROJ / "docs/reports"
FINAL = PROJ / "submissions/historical"
FINAL.mkdir(exist_ok=True)


def fit_de_quick(pool, y, n_iter=150, popsize=30, n_starts=2):
    N = len(pool)
    oofs = np.stack([t[1] for t in pool])
    tests = np.stack([t[2] for t in pool])
    def neg_hit(z):
        w = softmax_weights(z)
        pred = (w[:, None, None] * oofs).sum(axis=0)
        return -hit_rate(pred, y)
    bounds = [(-5.0, 5.0)] * N
    best = (None, np.inf)
    for s in range(n_starts):
        ts = time.time()
        res = differential_evolution(
            neg_hit, bounds, seed=s,
            maxiter=n_iter, popsize=popsize, tol=1e-6, mutation=(0.3, 1.5),
            recombination=0.8, init="sobol", polish=True, workers=1,
        )
        if res.fun < best[1]:
            best = (res.x, res.fun)
        print(f"  start {s+1}/{n_starts}: hit={-res.fun:.4f} ({(time.time()-ts):.1f}s)", flush=True)
    w = softmax_weights(best[0])
    return w, (w[:, None, None] * oofs).sum(axis=0), (w[:, None, None] * tests).sum(axis=0), -best[1]


def main():
    pool, y = load_pool(include_mdn=True)
    print(f"loaded {len(pool)} members", flush=True)
    names = [p[0] for p in pool]
    for nm, _, _, rh in pool:
        print(f"  {nm:<25} {rh:.4f}", flush=True)

    print("\n[A] full pool DE (n_starts=2 n_iter=150 popsize=30)", flush=True)
    t0 = time.time()
    w_full, oof_full, test_full, hit_full = fit_de_quick(pool, y, n_iter=150, popsize=30, n_starts=2)
    print(f"  full hit={hit_full:.4f}  elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    sorted_idx = np.argsort(-w_full)
    for i in sorted_idx:
        if w_full[i] > 0.01:
            print(f"    {names[i]:<25} w={w_full[i]:.4f}", flush=True)

    sub = pd.read_csv(OPEN / "sample_submission.csv")
    sub[["x","y","z"]] = test_full
    csv_d = OPEN / f"submission_v122d_quick_oof{hit_full:.4f}.csv"
    sub.to_csv(csv_d, index=False)
    np.savez(CACHE / "v122d_quick_weights.npz",
             names=np.array(names), weights=w_full,
             oof_pred=oof_full, test_pred=test_full, oof_hit=hit_full)
    print(f"  saved {csv_d}", flush=True)

    # conservative pool: top-7 by OOF + force v120/v126 family
    K = 7
    sorted_pool = sorted(range(len(pool)), key=lambda i: -pool[i][3])[:K]
    force_names = ["v120", "v120_n2", "v120_big", "v126_fft", "v121", "v121c5"]
    force_idx = [i for i,n in enumerate(names) if n in force_names]
    sel = sorted(set(sorted_pool + force_idx))
    sub_pool = [pool[i] for i in sel]
    print(f"\n[B] conservative pool (n={len(sub_pool)}):", flush=True)
    for i in sel: print(f"    {names[i]:<25} OOF={pool[i][3]:.4f}", flush=True)
    t0 = time.time()
    w_c, oof_c, test_c, hit_c = fit_de_quick(sub_pool, y, n_iter=150, popsize=30, n_starts=2)
    print(f"  conservative hit={hit_c:.4f}  elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    cons_names = [names[i] for i in sel]
    for i, w in sorted(enumerate(w_c), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"    {cons_names[i]:<25} w={w:.4f}", flush=True)
    sub = pd.read_csv(OPEN / "sample_submission.csv")
    sub[["x","y","z"]] = test_c
    csv_e = OPEN / f"submission_v122e_quick_oof{hit_c:.4f}.csv"
    sub.to_csv(csv_e, index=False)
    np.savez(CACHE / "v122e_quick_weights.npz",
             names=np.array(cons_names), weights=w_c,
             oof_pred=oof_c, test_pred=test_c, oof_hit=hit_c)
    print(f"  saved {csv_e}", flush=True)

    # compare v122c
    v22c = np.load(CACHE / "v122c_v121diverse_weights.npz", allow_pickle=True)
    oof_v22c = v22c["oof_pred"]; test_v22c = v22c["test_pred"]
    d1 = np.linalg.norm(test_full - test_v22c, axis=1) * 1000
    d2 = np.linalg.norm(test_c - test_v22c, axis=1) * 1000
    print(f"\nL2 vs v122c test (mm):", flush=True)
    print(f"  v122d_quick: mean={d1.mean():.3f} q90={np.quantile(d1,0.9):.3f}", flush=True)
    print(f"  v122e_quick: mean={d2.mean():.3f} q90={np.quantile(d2,0.9):.3f}", flush=True)

    # report
    lines = [
        f"# v122d/e quick blend ({time.strftime('%Y-%m-%d %H:%M')})",
        f"## Pool ({len(pool)} members)",
    ]
    for nm, _, _, rh in pool: lines.append(f"- {nm}: OOF={rh:.4f}")
    lines += [
        f"\n## v122d full",
        f"- OOF hit: {hit_full:.4f}",
        f"- 변환률 +0.0143 가정 LB 예상: {hit_full + 0.0143:.4f}",
        f"\n## v122e conservative",
        f"- OOF hit: {hit_c:.4f}",
        f"- 변환률 +0.0143 가정 LB 예상: {hit_c + 0.0143:.4f}",
        f"\n## vs v122c (LB 0.6912, OOF 0.6769)",
        f"- v122d L2 mean = {d1.mean():.3f}mm",
        f"- v122e L2 mean = {d2.mean():.3f}mm",
    ]
    best_oof = max(hit_full, hit_c)
    best_name = "v122d_quick" if hit_full > hit_c else "v122e_quick"
    lines.append(f"\n## 결정")
    if best_oof > 0.6769:
        lines.append(f"**{best_name} OOF {best_oof:.4f} (+{best_oof-0.6769:.4f} vs v122c) → 제출 후보**")
    else:
        lines.append("새 멤버 추가 OOF lift 미달. v122c 유지 권고.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "v122d_quick_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nreport: reports/v122d_quick_report.md", flush=True)

    if best_oof > 0.6769:
        src = csv_d if hit_full > hit_c else csv_e
        dst = FINAL / src.name
        dst.write_bytes(src.read_bytes())
        print(f"copied to final_candidates: {dst}", flush=True)

if __name__ == "__main__":
    main()
