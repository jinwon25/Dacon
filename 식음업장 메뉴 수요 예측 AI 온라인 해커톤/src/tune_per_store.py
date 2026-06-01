"""업장별 블렌드 가중·스케일 튜닝 — 가중 SMAPE 는 업장별 분해되므로 업장별 최적화=전체 최적화.

각 업장에서 (weekday_mean, mean7, mean28) 심플렉스 가중 + 전역 스케일을 4폴드 풀에서 탐색.
과적합 점검: 튜닝 설정의 폴드별 global 점수 일관성도 출력.

    python src/tune_per_store.py
"""
import numpy as np
import pandas as pd

from experiments import stacked_frame
from metric import score_long, weighted_smape, store_of

COMP = ["wd", "m28", "nz", "wd_nz"]  # 요일평균 / 28일평균 / 비-0평균 / 요일별 비-0평균
GLOBAL = dict(wd=0.4, m28=0.6, nz=0.0, wd_nz=0.0, c=1.0)  # 직전 전역 기준


def simplex(step=0.25, n=3):
    """ks=1/step 을 n 성분에 분배하는 모든 조합 (합=1)."""
    ks = int(round(1 / step))

    def comps(total, parts):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in comps(total - i, parts - 1):
                yield (i,) + rest
    return [tuple(round(x * step, 4) for x in c) for c in comps(ks, n)]


def pred_of(df, cfg):
    return cfg["c"] * sum(cfg[k] * df[k] for k in COMP)


def load_stacked(n_folds=4):
    df = stacked_frame(n_folds=n_folds)
    df["store"] = df["item"].map(store_of)
    return df


def search(df, scales=None):
    """업장별 (wd,m28,nz) 심플렉스 + 스케일 탐색 → {store: (cfg, score)}."""
    if scales is None:
        scales = np.round(np.arange(0.60, 1.51, 0.05), 2)
    simp = simplex(0.25, n=len(COMP))
    best = {}
    for store, g in df.groupby("store"):
        true = g["true"].to_numpy()
        comps = {k: g[k].to_numpy() for k in COMP}
        bcfg, bsc = None, 9.0
        for weights in simp:
            base = sum(w * comps[k] for w, k in zip(weights, COMP))
            for s in scales:
                sc = weighted_smape(true, np.clip(s * base, 0, None))
                if sc < bsc:
                    cfg = {k: w for k, w in zip(COMP, weights)}
                    cfg["c"] = float(s)
                    bcfg, bsc = cfg, sc
        best[store] = (bcfg, bsc)
    return best


def main():
    df = load_stacked()
    best = search(df)

    # 전역 vs 튜닝 예측 조립
    df["pred_glob"] = np.clip(pred_of(df, GLOBAL), 0, None)
    df["pred_tuned"] = 0.0
    for store, idx in df.groupby("store").groups.items():
        df.loc[idx, "pred_tuned"] = np.clip(pred_of(df.loc[idx], best[store][0]), 0, None)

    gl = score_long(df.assign(pred=df["pred_glob"]))
    tu = score_long(df.assign(pred=df["pred_tuned"]))
    print(f"\n=== 전역(0.4wd+0.6m28) vs 업장튜닝 — pooled 가중SMAPE ===")
    print(f"  global : {gl:.4f}")
    print(f"  tuned  : {tu:.4f}   (delta {tu - gl:+.4f})")
    print("  폴드별 (과적합 점검):")
    for k in sorted(df["fold"].unique()):
        s = df[df["fold"] == k]
        print(f"    fold{k}: global {score_long(s.assign(pred=s['pred_glob'])):.4f}"
              f"   tuned {score_long(s.assign(pred=s['pred_tuned'])):.4f}")

    rows = []
    for store, (cfg, sc) in best.items():
        g = df[df["store"] == store]
        before = weighted_smape(g["true"].to_numpy(), g["pred_glob"].to_numpy())
        rows.append(dict(store=store, high_w=store in ("담하", "미라시아"),
                         wd=cfg["wd"], m28=cfg["m28"], nz=cfg["nz"], wd_nz=cfg["wd_nz"],
                         c=cfg["c"], before=round(before, 4), after=round(sc, 4),
                         gain=round(before - sc, 4)))
    t = pd.DataFrame(rows).sort_values("gain", ascending=False)
    print("\n=== 업장별 최적 설정 + 개선 (gain 큰 순) ===")
    print(t.to_string(index=False))

    # 튜닝 설정 dict 출력 (제출 스크립트에 주입용)
    print("\nPER_STORE = {")
    for store, (cfg, _) in best.items():
        print(f"    {store!r}: {cfg},")
    print("}")
    return best


if __name__ == "__main__":
    main()
