"""v143_reblend — 강화된 frenet_gru 멤버(s4 시드앙상블 / n2 multistep / big 대용량)를
conservative 블렌드에 넣어 v141(LB 0.697) 상회 시도. 학습 완료 후 실행.

LB 피드백이 없는 자율 모드 → 신뢰 신호 = honest nested-CV + 신규멤버 실제 weight + OOF/TEST 일관성.
검증된 레버: "직교 AND 고OOF 멤버"가 DE weight를 받아 LB 변환(v141 frenet=0.668 weight).

핵심: 변종별로 '허용 멤버 집합(allowed)'을 분리. v141equiv는 신규멤버를 풀에서 완전 배제(공정 비교).
출력: 신규멤버 진단(버그검증) → v141equiv + v143a/b/c → nested-CV → 자동추천(final_candidates/ 복사).
usage: python scripts/v143_reblend.py
"""
from __future__ import annotations
import sys, shutil, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception: pass
from v132_final_blend import add_new_members, de_fit, hit
from v110_de_ensemble import load_pool, load_y_and_sub
from sklearn.model_selection import KFold
CACHE = Path("data/cache"); DATA = Path("data"); FINAL = Path("final_candidates")
NEWTAGS = ["frenet_gru_s4", "frenet_gru_n2", "frenet_gru_big"]
def decorr(a, b): return float(np.linalg.norm(a - b, axis=-1).mean() * 1000)
def isnew(nm): return any(t in nm for t in NEWTAGS)

def nested_cv(oofs_subset, y, outer=5, n_iter=120, popsize=22, n_starts=2):
    kf = KFold(n_splits=outer, shuffle=True, random_state=42); N = y.shape[0]; vh = []
    for tr, va in kf.split(np.arange(N)):
        w, _ = de_fit(oofs_subset[:, tr], y[tr], n_iter=n_iter, popsize=popsize, n_starts=n_starts)
        vh.append(hit((w[:, None, None] * oofs_subset[:, va]).sum(0), y[va]))
    return float(np.mean(vh))

def main():
    pool, y = load_pool(include_mdn=True)
    pool, added = add_new_members(pool, y)
    names = [p[0] for p in pool]
    oofs = np.stack([p[1] for p in pool]); tests = np.stack([p[2] for p in pool])
    ref = {n: i for i, n in enumerate(names)}
    print(f"pool={len(pool)}\n", flush=True)

    # === 신규멤버 버그검증: OOF + decorr(vs v120) + OOF/TEST 일관성 ===
    v120i = ref["v120"]
    diag_members = ["v131_frenet_gru", "v131frenet_gru_c15", "v131frenet_gru_c10", "v121"] + [n for n in names if isnew(n)]
    print("=== member diagnostic (decorr vs v120) ===")
    print(f"{'member':<28}{'OOF':>8}{'dOOF':>7}{'dTEST':>7}  consistency")
    bug = set()
    for nm in diag_members:
        if nm not in ref: continue
        i = ref[nm]; d_o = decorr(oofs[i], oofs[v120i]); d_t = decorr(tests[i], tests[v120i])
        ok = d_t <= 3.0 * max(d_o, 0.3) + 2.0
        if isnew(nm) and not ok: bug.add(nm)
        print(f"{nm:<28}{pool[i][3]:>8.4f}{d_o:>7.2f}{d_t:>7.2f}  {'OK' if ok else '!! TEST>>OOF BUG'} {'NEW' if isnew(nm) else ''}", flush=True)
    if bug: print(f"\n[!! WARNING] OOF/TEST 불일치 {sorted(bug)} — 블렌드에서 제외.", flush=True)

    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz")["test_pred"]
    v141 = pd.read_csv(DATA / "submission_v141_newconservative_oof0.6805.csv")[["x", "y", "z"]].to_numpy()
    sub_id = load_y_and_sub()[1]["id"]

    # conservative subset over a RESTRICTED allowed set (top-8 OOF + force diverse), 신규멤버는 변종별로만 허용
    force_nonnew = [n for n in names if (n.startswith("v131") or n.startswith("v135")
                    or n in ("v120", "v121", "v121c5", "v120_big")) and not isnew(n)]
    nonnew = {n for n in names if not isnew(n)}
    def newset(*subtags):
        return {n for n in names if any(s in n for s in subtags) and n not in bug}
    variant_allowed = {
        "v141equiv":  nonnew,
        "v143a_s4":   nonnew | newset("frenet_gru_s4"),
        "v143b_s4n2": nonnew | newset("frenet_gru_s4", "frenet_gru_n2"),
        "v143c_all":  nonnew | newset("frenet_gru_s4", "frenet_gru_n2", "frenet_gru_big"),
    }
    def build_subset(allowed, force_names, top_k=8, floor=0.67):
        aidx = [i for i in range(len(names)) if names[i] in allowed]
        chosen = []
        for i in sorted(aidx, key=lambda i: -pool[i][3]):
            if pool[i][3] >= floor and len(chosen) < top_k: chosen.append(i)
        for i in aidx:
            if names[i] in force_names and i not in chosen: chosen.append(i)
        return chosen

    results = {}
    for tag, allowed in variant_allowed.items():
        force_names = force_nonnew + [n for n in allowed if isnew(n)]   # 신규멤버 force-include (변종 한정)
        cons = build_subset(allowed, force_names)
        w, rh = de_fit(oofs[cons], y, n_iter=180, popsize=26, n_starts=3)
        test_c = (w[:, None, None] * tests[cons]).sum(0)
        na = int((w >= 0.01).sum()); neww = float(sum(w[j] for j in range(len(w)) if w[j] >= 0.01 and isnew(names[cons[j]])))
        l2_141 = decorr(test_c, v141)
        print(f"\n[{tag}] subset={len(cons)} OOF={rh:.4f} active={na} new_w={neww:.3f} L2vs141={l2_141:.2f}mm", flush=True)
        for j in np.argsort(-w):
            if w[j] >= 0.01:
                nm = names[cons[j]]
                print(f"    {nm:<28} w={w[j]:.3f} oof={pool[cons[j]][3]:.4f} {'NEW' if isnew(nm) else ''}", flush=True)
        if tag != "v141equiv":
            out = DATA / f"submission_{tag}_oof{rh:.4f}.csv"
            pd.DataFrame({"id": sub_id, "x": test_c[:,0], "y": test_c[:,1], "z": test_c[:,2]}).to_csv(out, index=False)
            print(f"    saved {out.name}", flush=True)
        results[tag] = dict(cons=cons, insample=rh, new_w=neww, l2_141=l2_141, test=test_c)

    # === NESTED-CV honest (LB 없을 때 핵심 신호) ===
    print("\n=== NESTED-CV honest (5 outer folds; v141 실측 LB 0.697) ===", flush=True)
    for tag in results:
        h = nested_cv(oofs[results[tag]["cons"]], y); results[tag]["honest"] = h
        print(f"  [{tag:<11}] in-sample={results[tag]['insample']:.4f}  honest={h:.4f}  new_w={results[tag]['new_w']:.3f}  L2vs141={results[tag]['l2_141']:.2f}mm", flush=True)

    # === 자동 추천 ===
    base_h = results["v141equiv"]["honest"]
    cand = {k: v for k, v in results.items() if k != "v141equiv"}
    best = max(cand, key=lambda k: cand[k]["honest"]); bh = cand[best]["honest"]; margin = bh - base_h
    print("\n" + "=" * 64 + "\nRECOMMENDATION\n" + "=" * 64, flush=True)
    print(f"  v141equiv honest nested-CV = {base_h:.4f}", flush=True)
    print(f"  best v143 = {best}  honest={bh:.4f} (Δ{margin:+.4f} vs equiv)  new_w={cand[best]['new_w']:.3f}  L2vs141={cand[best]['l2_141']:.2f}mm", flush=True)
    credible = (margin >= 0.0005) and (cand[best]["new_w"] >= 0.05) and (cand[best]["l2_141"] >= 0.25)
    if credible:
        rh = cand[best]["insample"]; src = DATA / f"submission_{best}_oof{rh:.4f}.csv"; dst = FINAL / f"submission_{best}_oof{rh:.4f}.csv"
        try: shutil.copy(src, dst); print(f"  → {best}이 honest CV에서 v141 상회 + 진짜 다른 예측. final_candidates/ 복사: {dst.name}", flush=True)
        except Exception as e: print(f"  (copy 실패: {e})", flush=True)
        print(f"  ★ 제출 권고: 1순위 {best} (상방), 2순위 v141 (floor 0.697). 둘 다 final 2슬롯.", flush=True)
    else:
        why = []
        if margin < 0.0005: why.append(f"honest Δ{margin:+.4f} (노이즈 ±0.0017 내)")
        if cand[best]["new_w"] < 0.05: why.append(f"신규weight {cand[best]['new_w']:.3f} 미미")
        if cand[best]["l2_141"] < 0.25: why.append(f"L2 {cand[best]['l2_141']:.2f}mm (v141과 거의 동일)")
        print(f"  → v143이 v141을 정직하게 못 넘음 ({'; '.join(why)}).", flush=True)
        print(f"  ★ 제출 권고: v141(0.697)+v122c(0.6912) 확정. v143은 여유분 있을 때 free-roll.", flush=True)

if __name__ == "__main__":
    main()
