"""v142 lean — v141 conservative recipe에 FFT paradigm(v126/v128) force 주입. 경량 DE, line-buffered 모니터링.
multistep/big raw(v120_n2/big)는 dTEST 0.5mm로 v120과 사실상 동일(다양성 없음) → 제외. FFT(v128 2.05mm)만 주입.
v142a = +FFT, v142b = +FFT +big-boundary(v129). v141(LB 0.697)은 floor 유지 → downside 0.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception: pass
from v132_final_blend import add_new_members, conservative_subset, de_fit, hit
from v110_de_ensemble import load_pool, load_y_and_sub
CACHE = Path("data/cache"); DATA = Path("data")
def decorr(a, b): return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)

print("loading pool...", flush=True)
pool, y = load_pool(include_mdn=True)
pool, added = add_new_members(pool, y)
names = [p[0] for p in pool]
oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
ref = {n: i for i, n in enumerate(names)}
print(f"pool={len(pool)} loaded", flush=True)

force_v141 = [n for n in names if n.startswith("v131") or n.startswith("v135")
              or n in ("v120", "v121", "v121c5", "v120_big")]
c22 = np.load(CACHE / "v122c_v121diverse_weights.npz")["test_pred"]
v141 = pd.read_csv(DATA / "submission_v141_newconservative_oof0.6805.csv")[["x", "y", "z"]].to_numpy()
sub_id = load_y_and_sub()[1]["id"]

def run(tag, extra):
    force = force_v141 + [n for n in extra if n in ref]
    cons = conservative_subset(pool, y, force, top_k=8, oof_floor=0.67)
    print(f"\n[{tag}] subset={len(cons)} members, fitting DE (n_starts=3)...", flush=True)
    w, rh = de_fit(oofs[cons], y, n_iter=140, popsize=24, n_starts=3)
    test_c = (w[:, None, None] * tests[cons]).sum(0)
    na = int((w >= 0.01).sum())
    print(f"[{tag}] OOF={rh:.4f}  active={na}", flush=True)
    pw = {}
    for j in np.argsort(-w):
        if w[j] >= 0.01:
            nm = names[cons[j]]
            grp = ("frenet/ch" if nm.startswith(("v131", "v135")) else "fft" if nm in ("v126_fft", "v128", "v128c5")
                   else "big" if nm in ("v120_big", "v129", "v129c5") else "base")
            pw[grp] = pw.get(grp, 0.) + w[j]
            print(f"    {nm:<22} w={w[j]:.3f} oof={pool[cons[j]][3]:.4f} [{grp}]", flush=True)
    print("    paradigm: " + "  ".join(f"{k}={v:.3f}" for k, v in pw.items()), flush=True)
    print(f"    L2 vs v141={decorr(test_c, v141):.2f}mm   vs v122c={decorr(test_c, c22):.2f}mm", flush=True)
    out = DATA / f"submission_{tag}_oof{rh:.4f}.csv"
    pd.DataFrame({"id": sub_id, "x": test_c[:, 0], "y": test_c[:, 1], "z": test_c[:, 2]}).to_csv(out, index=False)
    print(f"    saved {out.name}", flush=True)

run("v142a_fft", ["v126_fft", "v128", "v128c5"])
run("v142b_fftbig", ["v126_fft", "v128", "v128c5", "v129c5", "v129"])
print("\nDONE", flush=True)
