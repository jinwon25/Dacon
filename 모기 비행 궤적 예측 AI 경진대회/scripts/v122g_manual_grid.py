"""v122g_manual_grid.py — v122c base 위에 v128/v129 paradigm 작은 weight 수동 grid search.

이유: DE는 작은 pool에서 v128/v129 burning. v122c가 이미 LB 0.6912 챔피언이라 그 base 유지하면서
       paradigm diversity 약간 추가하여 LB lift 시도.

Setup:
  - target = v122c base + α₁ * v128c5 + α₂ * v129c5 + α₃ * v126_fft (Neural ODE+FFT)
  - α small (5-15%), 합 weight 보존 (renormalize)
  - 모든 조합을 OOF에서 평가, top OOF 후보 csv 저장
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass

SCRIPT = Path(__file__).resolve().parent
PROJ = SCRIPT.parent
CACHE = PROJ / "cache"
OPEN = PROJ / "open"
FINAL = PROJ / "final_candidates"
REPORTS = PROJ / "reports"

def hit(p, y): return (np.linalg.norm(p-y, axis=1) < 0.01).mean()

def main():
    y = pd.read_csv(OPEN / "train_labels.csv").sort_values("id")[["x","y","z"]].values

    # base = v122c
    c22 = np.load(CACHE / "v122c_v121diverse_weights.npz", allow_pickle=True)
    base_oof = c22["oof_pred"].astype(np.float64)
    base_test = c22["test_pred"].astype(np.float64)
    base_hit = hit(base_oof, y)
    print(f"v122c base OOF: {base_hit:.4f}")

    # Neural ODE+FFT family supplements
    members = {}
    for name, fname, key_oof, key_test in [
        ("v128c5",  "v128_cap15_state.npz",  "oof_v91",   "test_v91"),
        ("v129c5",  "v129_cap15_state.npz",  "oof_v91",   "test_v91"),
        ("v128",    "v128_cap10_state.npz",  "oof_v91",   "test_v91"),
        ("v129",    "v129_cap10_state.npz",  "oof_v91",   "test_v91"),
        ("v126fft", "v126_full_state.npz",   "oof_global","test_global"),
        ("v120big", "v120_big_full_state.npz","oof_global","test_global"),
        ("v120n2",  "v120_n2_full_state.npz","oof_global","test_global"),
    ]:
        p = CACHE / fname
        if not p.exists(): continue
        s = np.load(p, allow_pickle=True)
        oof = s[key_oof].astype(np.float64)
        test = s[key_test].astype(np.float64)
        members[name] = (oof, test, hit(oof, y))
        print(f"  {name:<10} OOF={hit(oof,y):.4f}")

    # 1) Single-member alpha sweep
    print("\n=== Single-member alpha sweep on v122c ===")
    best_blend = base_oof.copy(); best_test = base_test.copy()
    best_hit_val = base_hit; best_label = "v122c"
    results = []
    for name, (oof, test, _) in members.items():
        for alpha in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
            blend_oof = (1-alpha) * base_oof + alpha * oof
            blend_test = (1-alpha) * base_test + alpha * test
            h = hit(blend_oof, y)
            results.append((h, name, alpha))
            if h > best_hit_val:
                best_hit_val = h; best_blend = blend_oof; best_test = blend_test
                best_label = f"v122c+{alpha:.3f}*{name}"
                print(f"  ★ {best_label}: OOF {h:.4f} (+{h-base_hit:+.4f})")
    results.sort(reverse=True)
    print(f"\n  Top 8 single-blend:")
    for h, name, a in results[:8]:
        print(f"    v122c + {a:.3f}*{name}: OOF={h:.4f}  (Δ {h-base_hit:+.5f})")

    # 2) Two-member combination (top single + another)
    if best_label != "v122c":
        print(f"\n=== Two-member combo on top: {best_label} ===")
        # find which single member is winning
        top_single = best_label.split("*")[1]
        a1 = float(best_label.split("*")[0].split("+")[1])
        oof1, test1, _ = members[top_single]
        for name, (oof, test, _) in members.items():
            if name == top_single: continue
            for a2 in [0.025, 0.05, 0.075, 0.10]:
                rem = 1 - a1 - a2
                if rem < 0.5: continue
                blend_oof = rem * base_oof + a1 * oof1 + a2 * oof
                blend_test = rem * base_test + a1 * test1 + a2 * test
                h = hit(blend_oof, y)
                if h > best_hit_val:
                    best_hit_val = h
                    best_blend = blend_oof; best_test = blend_test
                    best_label = f"{(1-a1-a2):.3f}*v122c + {a1:.3f}*{top_single} + {a2:.3f}*{name}"
                    print(f"  ★ {best_label}: OOF {h:.4f} (+{h-base_hit:+.5f})")

    print(f"\n=== Best blend: {best_label} → OOF {best_hit_val:.4f} ===")
    print(f"  Δ vs v122c base: {best_hit_val - base_hit:+.5f}")

    # L2 vs v122c
    d = np.linalg.norm(best_test - base_test, axis=1) * 1000
    print(f"  L2 vs v122c test: mean={d.mean():.3f}mm  q90={np.quantile(d,0.9):.3f}mm")

    # save
    sub = pd.read_csv(OPEN / "sample_submission.csv")
    sub[["x","y","z"]] = best_test
    label_safe = best_label.replace("*","x").replace(" ","").replace("+","_")[:80]
    csv = OPEN / f"submission_v122g_manual_oof{best_hit_val:.4f}.csv"
    sub.to_csv(csv, index=False)
    print(f"\n  saved: {csv}")

    # report
    lines = [
        f"# v122g manual grid (v122c base + Neural ODE family supplement)",
        f"## base: v122c OOF 0.6769, LB 0.6912",
        f"## best blend: {best_label}",
        f"  OOF: **{best_hit_val:.4f}** (Δ {best_hit_val-base_hit:+.5f})",
        f"  L2 vs v122c: {d.mean():.3f}mm",
        f"",
        f"## Top 10 single-member alpha sweep",
    ]
    for h, name, a in results[:10]:
        lines.append(f"- v122c + {a:.3f}*{name}: OOF={h:.4f}  (Δ {h-base_hit:+.5f})")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "v122g_manual_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  report: reports/v122g_manual_report.md")

    if best_hit_val > base_hit:
        dst = FINAL / csv.name
        dst.write_bytes(csv.read_bytes())
        print(f"  copied to final_candidates: {dst}")

if __name__ == "__main__":
    main()
