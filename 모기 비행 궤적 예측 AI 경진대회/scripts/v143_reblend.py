"""v143_reblend — 강화된 frenet_gru 멤버(s4 시드앙상블 / n2 multistep / big 대용량)를
conservative 블렌드에 넣어 v141(LB 0.697) 상회 시도. 학습 완료 후 실행.

검증된 레버: "직교 AND 고OOF 멤버"가 DE weight를 받아 LB 변환(v141의 frenet=0.668 weight).
s4는 dominant 멤버(v131frenet_gru_c15, 0.677) 강화, n2/big은 신규 직교 멤버.

출력: 새 멤버 진단(OOF/decorr/일관성) + conservative 블렌드 후보 + nested-CV 정직성 + 제출 저장.
usage: python scripts/v143_reblend.py [--nestedcv]
"""
from __future__ import annotations
import argparse, sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception: pass
from v132_final_blend import add_new_members, conservative_subset, de_fit, hit
from v110_de_ensemble import load_pool, load_y_and_sub
from sklearn.model_selection import KFold
CACHE = Path("cache"); DATA = Path("open")
def decorr(a, b): return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)

def nested_cv(oofs_subset, y, outer=5, n_iter=140, popsize=26, n_starts=2):
    kf = KFold(n_splits=outer, shuffle=True, random_state=42); N = y.shape[0]; vh = []
    for tr, va in kf.split(np.arange(N)):
        w, _ = de_fit(oofs_subset[:, tr], y[tr], n_iter=n_iter, popsize=popsize, n_starts=n_starts)
        vh.append(hit((w[:, None, None] * oofs_subset[:, va]).sum(0), y[va]))
    return float(np.mean(vh))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nestedcv", action="store_true"); args = ap.parse_args()
    pool, y = load_pool(include_mdn=True)
    pool, added = add_new_members(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    ref = {n: i for i, n in enumerate(names)}
    print(f"pool={len(pool)}  added={added}\n", flush=True)

    # --- new-member diagnostic ---
    v120i = ref["v120"]
    newcands = [n for n in names if any(t in n for t in ["frenet_gru_s4", "frenet_gru_n2", "frenet_gru_big"])]
    refcands = ["v131_frenet_gru", "v131frenet_gru_c15", "v131frenet_gru_c10", "v121"]
    print("=== new vs reference members (decorr vs v120) ===")
    print(f"{'member':<26}{'OOF':>8}{'dOOF':>8}{'dTEST':>8}  consistency")
    for nm in refcands + newcands:
        if nm not in ref:
            print(f"{nm:<26}  (missing)"); continue
        i = ref[nm]; d_o = decorr(oofs[i], oofs[v120i]); d_t = decorr(tests[i], tests[v120i])
        flag = "OK" if d_t <= 3.0 * max(d_o, 0.3) + 2.0 else "!! TEST>>OOF BUG?"
        tagmark = "NEW" if nm in newcands else ""
        print(f"{nm:<26}{pool[i][3]:>8.4f}{d_o:>8.2f}{d_t:>8.2f}  {flag} {tagmark}", flush=True)

    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz")["test_pred"]
    v141 = pd.read_csv(DATA / "submission_v141_newconservative_oof0.6805.csv")[["x", "y", "z"]].to_numpy()
    sub_id = load_y_and_sub()[1]["id"]

    # force lists: v141 baseline + new members
    force_base = [n for n in names if n.startswith("v131") or n.startswith("v135")
                  or n in ("v120", "v121", "v121c5", "v120_big")]
    # variants to try
    variants = {
        "v143a_s4":     [n for n in names if "frenet_gru_s4" in n],                      # strengthen dominant only
        "v143b_s4n2":   [n for n in names if ("frenet_gru_s4" in n or "frenet_gru_n2" in n)],
        "v143c_all":    [n for n in names if any(t in n for t in ["frenet_gru_s4","frenet_gru_n2","frenet_gru_big"])],
    }
    results = {}
    for tag, extra in variants.items():
        force = list(dict.fromkeys(force_base + extra))
        cons = conservative_subset(pool, y, force, top_k=8, oof_floor=0.67)
        w, rh = de_fit(oofs[cons], y, n_iter=190, popsize=28, n_starts=4)
        test_c = (w[:, None, None] * tests[cons]).sum(0)
        na = int((w >= 0.01).sum())
        print(f"\n[{tag}] subset={len(cons)} OOF={rh:.4f} active={na}", flush=True)
        neww = 0.0
        for j in np.argsort(-w):
            if w[j] >= 0.01:
                nm = names[cons[j]]; isnew = any(t in nm for t in ["frenet_gru_s4","frenet_gru_n2","frenet_gru_big"])
                if isnew: neww += w[j]
                print(f"    {nm:<26} w={w[j]:.3f} oof={pool[cons[j]][3]:.4f} {'NEW' if isnew else ''}", flush=True)
        print(f"    new-member weight={neww:.3f}  L2 vs v141={decorr(test_c,v141):.2f}mm  vs v122c={decorr(test_c,c22):.2f}mm", flush=True)
        out = DATA / f"submission_{tag}_oof{rh:.4f}.csv"
        pd.DataFrame({"id": sub_id, "x": test_c[:,0], "y": test_c[:,1], "z": test_c[:,2]}).to_csv(out, index=False)
        print(f"    saved {out.name}", flush=True)
        results[tag] = (cons, rh, decorr(test_c, v141))

    if args.nestedcv:
        print("\n=== NESTED-CV honest (v141 honest=0.6762 기준) ===", flush=True)
        # v141-equivalent baseline (no new members)
        base_cons = conservative_subset(pool, y, force_base, top_k=8, oof_floor=0.67)
        print(f"  [v141-equiv] honest={nested_cv(oofs[base_cons], y):.4f}", flush=True)
        for tag, (cons, rh, l2) in results.items():
            print(f"  [{tag:<12}] in-sample={rh:.4f}  honest={nested_cv(oofs[cons], y):.4f}  L2vs141={l2:.2f}mm", flush=True)

if __name__ == "__main__":
    main()
