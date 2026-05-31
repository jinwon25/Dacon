"""v133_modeseek.py — metric-aware aggregation (mode-seeking vs weighted mean).

가설: R-Hit@1cm 최적 점예측 = conditional MODE (1cm 공 안 확률질량 최대점).
DE blend는 weighted MEAN → 멤버가 두 군집으로 갈리면 사이에 앉아 둘 다 miss.
멤버 예측 구름에서 mean-shift로 densest cluster 중심(mode)으로 이동하면 hit↑ 가능.
selector(개별 멤버 픽, dead)와 다른 '집계 방식' 변경.

CV-정직 평가 불필요(학습 파라미터 없음, geometric). h(bandwidth) 1개만 OOF로 선택→약과적합.
"""
from __future__ import annotations
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
from v110_de_ensemble import load_pool
CACHE = Path("cache")
def hit(p, y): return float((np.linalg.norm(p - y, axis=-1) <= 0.01).mean())

def main():
    pool, y = load_pool(include_mdn=True)
    od = {p[0]: p[1].astype(np.float64) for p in pool}
    w22 = np.load(CACHE / "v122c_v121diverse_weights.npz")
    names22 = list(w22["names"]); wts22 = w22["weights"]
    base = w22["oof_pred"]
    print(f"v122c base OOF={hit(base,y):.4f}")

    # active members of v122c (w>=0.01) + their weights
    act = [(names22[i], wts22[i]) for i in range(len(names22)) if wts22[i] >= 0.01 and names22[i] in od]
    P = np.stack([od[n] for n, _ in act])            # (M,N,3)
    W = np.array([w for _, w in act]); W = W / W.sum()
    print(f"active members ({len(act)}): {[n for n,_ in act]}")
    wmean = (W[:, None, None] * P).sum(0)
    print(f"weighted-mean reconstruct OOF={hit(wmean,y):.4f}")

    N = y.shape[0]
    # mean-shift toward mode of member cloud, init at weighted mean
    def mean_shift(h, iters=8):
        x = wmean.copy()                              # (N,3)
        for _ in range(iters):
            diff = P - x[None]                        # (M,N,3)
            d2 = (diff ** 2).sum(-1)                  # (M,N)
            k = np.exp(-0.5 * d2 / (h * h)) * W[:, None]   # gaussian kernel * member weight
            ksum = k.sum(0) + 1e-12
            x = (k[:, :, None] * P).sum(0) / ksum[:, None]
        return x
    print("\n[mean-shift] bandwidth sweep (mm):")
    best = (None, hit(wmean, y), "wmean")
    for h_mm in [3, 4, 5, 6, 8, 10, 15]:
        x = mean_shift(h_mm / 1000.0)
        rh = hit(x, y)
        tag = f"h={h_mm}mm"
        L2 = np.linalg.norm(x - wmean, axis=-1).mean() * 1000
        print(f"  {tag:<8} OOF={rh:.4f}  (Δ={rh-hit(wmean,y):+.4f})  L2vs_wmean={L2:.3f}mm")
        if rh > best[1]: best = (x, rh, tag)

    # blend mode-seek with mean (partial shift) — robustness
    print("\n[partial shift alpha] (x = (1-a)*wmean + a*modeseek, h=5mm):")
    ms = mean_shift(0.005)
    for a in [0.25, 0.5, 0.75, 1.0]:
        x = (1 - a) * wmean + a * ms
        rh = hit(x, y)
        print(f"  a={a}  OOF={rh:.4f}  (Δ={rh-hit(wmean,y):+.4f})")

    # weighted geometric median (Weiszfeld) — robust center, another mode proxy
    def geomed(iters=10):
        x = wmean.copy()
        for _ in range(iters):
            d = np.linalg.norm(P - x[None], axis=-1) + 1e-9   # (M,N)
            wd = (W[:, None] / d)
            x = (wd[:, :, None] * P).sum(0) / wd.sum(0)[:, None]
        return x
    gm = geomed()
    print(f"\n[weighted geomedian] OOF={hit(gm,y):.4f}  (Δ={hit(gm,y)-hit(wmean,y):+.4f})")
    print(f"\nBEST: {best[2]}  OOF={best[1]:.4f}")

if __name__ == "__main__":
    main()
