"""v122d_blend_after_training.py — 새 paradigm 멤버 추가 후 DE/conservative blend 재계산.

학습 완료 후 자동 실행:
  - v110_de_ensemble.py 로직 재사용 (load_pool 통해 자동으로 v120_n2/big/v126 포함)
  - DE 학습 → v122d 후보
  - v112_conservative_blend.py 로직: force-include v120 family → v122e 후보
  - 두 후보의 OOF + L2 distance vs v122c 비교 → LB 변환 예측
  - 결과를 reports/v122d_blend_report.md에 저장
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
from v110_de_ensemble import load_pool, fit_de, softmax_weights, hit_rate

PROJ = SCRIPT.parent
CACHE = PROJ / "cache"
OPEN = PROJ / "open"
REPORTS = PROJ / "reports"
FINAL = PROJ / "final_candidates"
FINAL.mkdir(exist_ok=True)


def main():
    pool, y = load_pool(include_mdn=True)
    print(f"\n=== loaded {len(pool)} models ===")
    names = []
    for nm, _, _, rh in pool:
        names.append(nm); print(f"  {nm:<25} OOF={rh:.4f}")

    # 1) DE on all members
    print("\n[DE-1] full pool DE blend")
    w_full, oof_full, test_full, hit_full = fit_de(pool, y, n_iter=300, popsize=40, n_starts=5)
    print(f"  hit = {hit_full:.4f}, top weights:")
    idx_sorted = np.argsort(-w_full)
    for i in idx_sorted[:10]:
        if w_full[i] > 0.01:
            print(f"    {names[i]:<25} w={w_full[i]:.4f}  oof={pool[i][3]:.4f}")

    # save v122d candidate
    csv_d = OPEN / f"submission_v122d_full_oof{hit_full:.4f}.csv"
    sub = pd.read_csv(OPEN / "sample_submission.csv")
    sub[["x","y","z"]] = test_full
    sub.to_csv(csv_d, index=False)
    print(f"  saved {csv_d}")

    # save state
    np.savez(CACHE / "v122d_full_weights.npz",
              names=np.array(names), weights=w_full,
              oof_pred=oof_full, test_pred=test_full,
              oof_hit=hit_full)

    # 2) Conservative subset (top-7 by single OOF + force include v120 family)
    print("\n[DE-2] conservative pool: top-K by OOF + force v120 paradigm")
    K = 7
    sorted_pool = sorted(range(len(pool)), key=lambda i: -pool[i][3])[:K]
    force_names = ["v120", "v120_n2", "v120_big", "v126_fft", "v121", "v121c5"]
    force_idx = [i for i,n in enumerate(names) if n in force_names]
    sel = sorted(set(sorted_pool + force_idx))
    sub_pool = [pool[i] for i in sel]
    print(f"  conservative subset (n={len(sub_pool)}):")
    for i in sel: print(f"    {names[i]:<25} OOF={pool[i][3]:.4f}")

    w_cons, oof_cons, test_cons, hit_cons = fit_de(sub_pool, y, n_iter=300, popsize=40, n_starts=5)
    print(f"  conservative hit = {hit_cons:.4f}")
    cons_names = [names[i] for i in sel]
    for i, w in sorted(enumerate(w_cons), key=lambda x: -x[1])[:10]:
        if w > 0.01:
            print(f"    {cons_names[i]:<25} w={w:.4f}")
    csv_e = OPEN / f"submission_v122e_conservative_oof{hit_cons:.4f}.csv"
    sub = pd.read_csv(OPEN / "sample_submission.csv")
    sub[["x","y","z"]] = test_cons
    sub.to_csv(csv_e, index=False)
    print(f"  saved {csv_e}")
    np.savez(CACHE / "v122e_conservative_weights.npz",
              names=np.array(cons_names), weights=w_cons,
              oof_pred=oof_cons, test_pred=test_cons, oof_hit=hit_cons)

    # 3) Compare with current v122c
    v122c_state = CACHE / "v122c_v121diverse_weights.npz"
    if v122c_state.exists():
        c22 = np.load(v122c_state, allow_pickle=True)
        oof_v22c = c22["oof_pred"]
        test_v22c = c22["test_pred"]
        d_full_v22c = np.linalg.norm(test_full - test_v22c, axis=1) * 1000
        d_cons_v22c = np.linalg.norm(test_cons - test_v22c, axis=1) * 1000
        print(f"\n[comparison] test L2 vs v122c (mm):")
        print(f"  v122d full vs v122c: mean={d_full_v22c.mean():.3f}  q90={np.quantile(d_full_v22c,0.9):.3f}")
        print(f"  v122e cons vs v122c: mean={d_cons_v22c.mean():.3f}  q90={np.quantile(d_cons_v22c,0.9):.3f}")

    # 4) Save report
    report = []
    report.append(f"# v122d/e blend after Neural ODE pool 확장 ({time.strftime('%Y-%m-%d %H:%M')})\n")
    report.append(f"## Pool members ({len(pool)})\n")
    for nm, _, _, rh in pool:
        report.append(f"- {nm}: OOF={rh:.4f}")
    report.append(f"\n## v122d (full pool DE)")
    report.append(f"- OOF hit: {hit_full:.4f}")
    report.append(f"- LB 변환률 +0.0143 가정 시 예상 LB: {hit_full + 0.0143:.4f}")
    report.append(f"\n## v122e (conservative + force v120 family)")
    report.append(f"- OOF hit: {hit_cons:.4f}")
    report.append(f"- LB 변환률 +0.0143 가정 시 예상 LB: {hit_cons + 0.0143:.4f}")
    report.append(f"\n## v122c base (LB 0.6912, 변환률 +0.0143)")
    report.append(f"- OOF hit: 0.6769")
    report.append(f"")
    report.append(f"## 결정")
    best_oof = max(hit_full, hit_cons)
    best_name = "v122d" if hit_full > hit_cons else "v122e"
    if best_oof > 0.6769:
        report.append(f"**제출 후보: {best_name} (OOF {best_oof:.4f}, +{best_oof-0.6769:.4f} vs v122c)**")
    else:
        report.append(f"새 멤버 추가 lift 미달. v122c 유지 권고.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "v122d_blend_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport saved: reports/v122d_blend_report.md")

    # 5) Copy best to final_candidates
    if best_oof > 0.6769:
        src = csv_d if hit_full > hit_cons else csv_e
        dst = FINAL / src.name
        dst.write_bytes(src.read_bytes())
        print(f"copied to final_candidates: {dst}")

if __name__ == "__main__":
    main()
