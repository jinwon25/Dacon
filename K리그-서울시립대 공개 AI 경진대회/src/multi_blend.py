"""다중 멤버 블렌드 — GBDT + 여러 윈도우 GRU(K=24/12/40)를 최적 가중으로 섞는다.

서로 다른 윈도우의 GRU는 '보는 시간 범위'가 달라 decorrelation이 커진다(4th place 트릭).
가중은 비음수·합1 제약으로 OOF의 (x,y) 제곱오차 합을 최소화(SLSQP) — 유클리드의 좋은 근사.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
import sys  # noqa: E402
sys.path.insert(0, "src")
from features_v4 import build_features_v4  # noqa: E402

MEMBERS = {
    "GBDT": "data/cache/gbdt_oof.npz",
    "GRU_K24": "data/cache/gru_oof.npz",
    "GRU_K12": "data/cache/gru_oof_K12.npz",
    "GRU_K40": "data/cache/gru_oof_K40.npz",
}
MEMBERS = {k: v for k, v in MEMBERS.items() if os.path.exists(v)}
names = list(MEMBERS)
print("[멤버]", names)

# 정답
tr = build_features_v4("'data/train.csv'")
tr = tr[tr["tgt_x"].notna()][["game_episode", "tgt_x", "tgt_y"]].rename(columns={"game_episode": "ep"})

# OOF 정렬 정합 (모든 멤버를 ep 기준 inner join)
oof = tr.copy()
test = None
for n in names:
    d = np.load(MEMBERS[n], allow_pickle=True)
    oof = oof.merge(pd.DataFrame({"ep": d["ep"], f"{n}_x": d["oof_x"], f"{n}_y": d["oof_y"]}), on="ep")
    t = pd.DataFrame({"ep": d["test_ep"], f"{n}_x": d["pred_x"], f"{n}_y": d["pred_y"]})
    test = t if test is None else test.merge(t, on="ep")

Ax = oof[[f"{n}_x" for n in names]].to_numpy()
Ay = oof[[f"{n}_y" for n in names]].to_numpy()
tx, ty = oof["tgt_x"].to_numpy(), oof["tgt_y"].to_numpy()


def euclid(w):
    return np.sqrt((Ax @ w - tx) ** 2 + (Ay @ w - ty) ** 2).mean()


def mse(w):  # 최적화용 (볼록)
    return ((Ax @ w - tx) ** 2 + (Ay @ w - ty) ** 2).mean()


print("\n[단독 멤버 CV]")
for i, n in enumerate(names):
    w = np.zeros(len(names)); w[i] = 1
    print(f"  {n:8s} {euclid(w):.4f}")

w0 = np.ones(len(names)) / len(names)
cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
bnds = [(0, 1)] * len(names)
res = minimize(mse, w0, method="SLSQP", bounds=bnds, constraints=cons)
w = res.x
print("\n[최적 가중]", {n: round(float(wi), 3) for n, wi in zip(names, w)})
print(f"  ★ 다중 블렌드 CV {euclid(w):.4f}   (이전 best 13.95)")

# 제출
Tx = test[[f"{n}_x" for n in names]].to_numpy()
Ty = test[[f"{n}_y" for n in names]].to_numpy()
out = pd.DataFrame({"game_episode": test["ep"],
                    "end_x": np.clip(Tx @ w, 0, 105),
                    "end_y": np.clip(Ty @ w, 0, 68)})
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(out, on="game_episode", how="left")
out.to_csv("submissions/blend_multi.csv", index=False)
print(f"\n[완료] submissions/blend_multi.csv (행수={len(out)}, 결측 {out['end_x'].isna().sum()})")
