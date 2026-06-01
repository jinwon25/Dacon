"""v122f_sweet9_blend.py — 9 active member conservative blend with Neural ODE+FFT boundary.

목표: v122c sweet spot (9 active, LB 0.6912, 변환률 +0.0143) 재현하되 새 paradigm 멤버
      (v128/v129 = FFT/big Neural ODE boundary) 끼워넣어 paradigm diversity ↑.

설계:
  - top-K (K=7) by single OOF
  - force include: v121 v121c5 v128 v128c5 v129 v129c5 (6 new Neural ODE family)
  - 최종 DE blend: union(top-7, force) → 약 9-11 active
  - n_starts=3 n_iter=200 popsize=30 (DE 학습 ~15분/start)

LB 변환 가정:
  - 9 active members + Neural ODE family ~40% weight = +0.0143 변환률
  - 10-11 active members = +0.0120 변환률 (over-fit 시작)
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


def fit_de(pool, y, n_iter=200, popsize=30, n_starts=3, label=""):
    N = len(pool)
    oofs = np.stack([t[1] for t in pool])
    tests = np.stack([t[2] for t in pool])
    def neg_hit(z):
        w = softmax_weights(z)
        return -hit_rate((w[:, None, None] * oofs).sum(axis=0), y)
    bounds = [(-5.0, 5.0)] * N
    best = (None, np.inf)
    for s in range(n_starts):
        ts = time.time()
        res = differential_evolution(
            neg_hit, bounds, seed=s, maxiter=n_iter, popsize=popsize,
            tol=1e-6, mutation=(0.3, 1.5), recombination=0.8, init="sobol",
            polish=True, workers=1,
        )
        if res.fun < best[1]: best = (res.x, res.fun)
        print(f"  [{label}] start {s+1}/{n_starts}: hit={-res.fun:.4f}  ({(time.time()-ts):.1f}s)", flush=True)
    w = softmax_weights(best[0])
    return w, (w[:, None, None] * oofs).sum(axis=0), (w[:, None, None] * tests).sum(axis=0), -best[1]


def main():
    pool, y = load_pool(include_mdn=True)
    names = [p[0] for p in pool]
    print(f"loaded {len(pool)} pool members", flush=True)
    for nm, _, _, rh in pool:
        marker = " ←" if nm in {"v128","v128c5","v129","v129c5","v126_fft","v120_big","v120_n2"} else ""
        print(f"  {nm:<25} OOF={rh:.4f}{marker}", flush=True)

    # Sweet9 selection
    K = 7
    sorted_by_oof = sorted(range(len(pool)), key=lambda i: -pool[i][3])
    top_k = sorted_by_oof[:K]
    force_names = ["v121", "v121c5", "v128", "v128c5", "v129", "v129c5"]
    force_idx = [i for i, n in enumerate(names) if n in force_names]
    sel = sorted(set(top_k + force_idx))
    sub_pool = [pool[i] for i in sel]
    print(f"\n[Sweet9 selection, n={len(sub_pool)}]:", flush=True)
    for i in sel:
        is_force = names[i] in force_names
        is_top = i in top_k
        marks = ("[top]" if is_top else "") + ("[force]" if is_force else "")
        print(f"  {names[i]:<25} OOF={pool[i][3]:.4f}  {marks}", flush=True)

    print("\n=== DE blend (n_starts=3, n_iter=200, popsize=30) ===", flush=True)
    t0 = time.time()
    w, oof_pred, test_pred, hit = fit_de(sub_pool, y, n_iter=200, popsize=30, n_starts=3, label="sweet9")
    print(f"\n  FINAL hit = {hit:.4f}  elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    sel_names = [names[i] for i in sel]
    print("\n  Final weights (sorted):", flush=True)
    active = 0
    v120_family = {"v120", "v120_n2", "v120_big", "v126_fft", "v121", "v121c5", "v128", "v128c5", "v129", "v129c5"}
    v120_w_sum = 0.0
    for i, ww in sorted(enumerate(w), key=lambda x: -x[1]):
        if ww > 0.01:
            active += 1
            print(f"    {sel_names[i]:<25} w={ww:.4f}", flush=True)
            if sel_names[i] in v120_family:
                v120_w_sum += ww
    print(f"\n  active members: {active}", flush=True)
    print(f"  Neural ODE family weight sum: {v120_w_sum:.4f}", flush=True)

    sub = pd.read_csv(OPEN / "sample_submission.csv")
    sub[["x","y","z"]] = test_pred
    csv_f = OPEN / f"submission_v122f_sweet9_oof{hit:.4f}.csv"
    sub.to_csv(csv_f, index=False)
    np.savez(CACHE / "v122f_sweet9_weights.npz",
              names=np.array(sel_names), weights=w,
              oof_pred=oof_pred, test_pred=test_pred, oof_hit=hit,
              active_count=active, v120_family_weight=v120_w_sum)
    print(f"\n  saved {csv_f}", flush=True)

    # Comparison vs v122c
    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz", allow_pickle=True)
    test_v22c = c22["test_pred"]
    d = np.linalg.norm(test_pred - test_v22c, axis=1) * 1000
    print(f"\n  L2 vs v122c: mean={d.mean():.3f}mm  q90={np.quantile(d,0.9):.3f}mm", flush=True)

    # report
    lines = [
        f"# v122f Sweet9 blend ({time.strftime('%Y-%m-%d %H:%M')})",
        f"## Pool ({len(pool)} members), selected ({len(sub_pool)})",
        f"OOF hit: **{hit:.4f}**",
        f"Active members: {active}",
        f"Neural ODE family weight: {v120_w_sum:.4f}",
        f"L2 vs v122c: {d.mean():.3f}mm",
        "",
        f"## LB 변환률 가정",
        f"- v122c 9 active +0.0143: 예상 LB = {hit + 0.0143:.4f}",
        f"- conservative (smaller diff) +0.0127: 예상 LB = {hit + 0.0127:.4f}",
        f"- over-fit (14+ active) +0.0101: 예상 LB = {hit + 0.0101:.4f}",
        "",
        "## Active weights",
    ]
    for i, ww in sorted(enumerate(w), key=lambda x: -x[1]):
        if ww > 0.01:
            lines.append(f"- {sel_names[i]}: w={ww:.4f}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "v122f_sweet9_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  report: reports/v122f_sweet9_report.md", flush=True)

    # copy to final
    if hit > 0.6769:
        dst = FINAL / csv_f.name
        dst.write_bytes(csv_f.read_bytes())
        print(f"  copied to final_candidates: {dst}", flush=True)

if __name__ == "__main__":
    main()
