"""최종 앙상블 — GBDT(v4) + GRU 를 섞는다. 핵심: 둘이 '얼마나 다른지'와 최적 가중 찾기.

- 두 모델의 OOF 예측을 game_episode로 정렬해 맞춘다.
- 예측 상관(corr)을 본다: GBDT끼린 ~0.99였지만, GRU는 시야가 달라 더 낮을 것 → 앙상블 이득 기대.
- 가중 w 를 0~1 격자탐색해 OOF 유클리드 거리를 최소화. 그 w로 test 예측을 섞어 제출 생성.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

g = np.load("data/cache/gbdt_oof.npz", allow_pickle=True)
r = np.load("data/cache/gru_oof.npz", allow_pickle=True)

# game_episode 로 정렬 정합
gb = pd.DataFrame({"ep": g["ep"], "gx": g["oof_x"], "gy": g["oof_y"]})
ru = pd.DataFrame({"ep": r["ep"], "rx": r["oof_x"], "ry": r["oof_y"]})
oof = gb.merge(ru, on="ep")
# 정답 회수 (features에서 다시) — gbdt oof는 train 순서, 정답은 features_v4로
import sys  # noqa: E402
sys.path.insert(0, "src")
from features_v4 import build_features_v4  # noqa: E402
tr = build_features_v4("'data/train.csv'")
tr = tr[tr["tgt_x"].notna()][["game_episode", "tgt_x", "tgt_y"]].rename(columns={"game_episode": "ep"})
oof = oof.merge(tr, on="ep")


def cv(px, py):
    return np.sqrt((px - oof["tgt_x"]) ** 2 + (py - oof["tgt_y"]) ** 2).mean()


cv_g = cv(oof["gx"], oof["gy"])
cv_r = cv(oof["rx"], oof["ry"])
corr_x = np.corrcoef(oof["gx"], oof["rx"])[0, 1]
print(f"[단독] GBDT v4 {cv_g:.4f} | GRU {cv_r:.4f}")
print(f"[상관] GBDT vs GRU 예측 corr(x) = {corr_x:.3f}  (낮을수록 앙상블 이득 큼)")

best_w, best_cv = 0.0, 1e9
for w in np.linspace(0, 1, 51):  # w = GBDT 비중
    c = cv(w * oof["gx"] + (1 - w) * oof["rx"], w * oof["gy"] + (1 - w) * oof["ry"])
    if c < best_cv:
        best_cv, best_w = c, w
print(f"\n  ★ 최적 가중 GBDT {best_w:.2f} / GRU {1-best_w:.2f}  → 앙상블 CV {best_cv:.4f}")
print(f"     (단독 최고 {min(cv_g, cv_r):.4f} 대비 {min(cv_g, cv_r)-best_cv:+.4f})")

# test 예측 섞어 제출
gt = pd.DataFrame({"ep": g["test_ep"], "gx": g["pred_x"], "gy": g["pred_y"]})
rt = pd.DataFrame({"ep": r["test_ep"], "rx": r["pred_x"], "ry": r["pred_y"]})
te = gt.merge(rt, on="ep")
te["end_x"] = best_w * te["gx"] + (1 - best_w) * te["rx"]
te["end_y"] = best_w * te["gy"] + (1 - best_w) * te["ry"]
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(te[["ep", "end_x", "end_y"]].rename(columns={"ep": "game_episode"}),
                on="game_episode", how="left")
out["end_x"] = out["end_x"].clip(0, 105)
out["end_y"] = out["end_y"].clip(0, 68)
out.to_csv("submissions/ensemble_v5.csv", index=False)
print(f"\n[완료] submissions/ensemble_v5.csv (행수={len(out)}, 결측 {out['end_x'].isna().sum()})")
