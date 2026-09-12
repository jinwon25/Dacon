"""로컬 검증(Local CV): 제출 없이 '내 예측이 평균 몇 점인지' 직접 계산한다.

핵심 아이디어
- train 에피소드에는 정답(마지막 패스 도착점)이 있다.
- 그 도착점을 '가렸다고 치고' 여러 방법으로 예측 → 실제와의 평균 유클리드 거리
  (= 대회 채점식)를 잰다.
- 이 로컬 점수가 리더보드와 같이 움직이면, 이후로는 제출 없이 로컬에서만
  빠르게 좋아졌는지 판단할 수 있다.

leakage(정보 누수) 방지
- game_id 기준으로 fit(80%) / valid(20%)를 나눈다. 한 경기의 에피소드는 한 쪽에만.
- '학습되는 값'(평균 전진량 등)은 fit에서만 구하고, 점수는 valid에서만 잰다.

실행 (대회 폴더에서):
    python src/local_cv.py
"""
import os

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
con = duckdb.connect()

# 에피소드별: 마지막 패스(start, 정답 end) + 직전 이벤트의 이동벡터(모멘텀)
SQL = """
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
    FROM 'data/train.csv'
),
last_pass AS (                         -- rn=1 : 예측 대상(마지막 패스)
    SELECT game_id, game_episode,
           start_x, start_y,
           end_x AS tgt_x, end_y AS tgt_y
    FROM ranked WHERE rn = 1
),
prev_ev AS (                           -- rn=2 : 직전 이벤트(공이 어디서 어디로 움직여 왔나)
    SELECT game_episode,
           (end_x - start_x) AS pdx,
           (end_y - start_y) AS pdy
    FROM ranked WHERE rn = 2
)
SELECT lp.*,
       COALESCE(pe.pdx, 0) AS pdx,
       COALESCE(pe.pdy, 0) AS pdy
FROM last_pass lp
LEFT JOIN prev_ev pe USING (game_episode)
WHERE lp.tgt_x IS NOT NULL
"""
df = con.execute(SQL).df()
print(f"[데이터] 에피소드 {len(df):,}개")

# --- fit / valid 분리 (game_id 기준 20%를 valid 로) ---
valid_mask = (df["game_id"] % 5 == 0)
fit, val = df[~valid_mask].copy(), df[valid_mask].copy()
print(f"[분할] fit {len(fit):,} / valid {len(val):,}  (game_id %% 5 == 0 → valid)")

# --- fit 에서 '평균 전진량' 학습 ---
DX = (fit["tgt_x"] - fit["start_x"]).mean()
DY = (fit["tgt_y"] - fit["start_y"]).mean()
print(f"[학습] 평균 전진 DX={DX:+.2f}, DY={DY:+.2f}  (fit 에서만 계산)")


def score(px, py):
    """대회 채점식: 평균 유클리드 거리 (낮을수록 좋음)."""
    d = np.sqrt((px - val["tgt_x"]) ** 2 + (py - val["tgt_y"]) ** 2)
    return d.mean()


def clip_xy(px, py):
    return px.clip(0, 105), py.clip(0, 68)  # 경기장 밖 예측 방지


sx, sy, pdx, pdy = val["start_x"], val["start_y"], val["pdx"], val["pdy"]

methods = {
    "A. 도착=출발 (end=start)":              (sx,            sy),
    "B. 출발+평균전진":                       (sx + DX,       sy + DY),
    "B'. 출발+평균전진 (경기장 clip)":         clip_xy(sx + DX, sy + DY),
    "C1. 출발+직전이동(모멘텀)":              (sx + pdx,      sy + pdy),
    "C2. 출발+½평균전진+½모멘텀":             (sx + 0.5 * DX + 0.5 * pdx,
                                              sy + 0.5 * DY + 0.5 * pdy),
}

rows = []
for name, (px, py) in methods.items():
    rows.append((name, round(score(px, py), 4)))
res = pd.DataFrame(rows, columns=["방법", "valid 평균거리(↓)"]).sort_values("valid 평균거리(↓)")
print("\n===== 로컬 검증 결과 (낮을수록 좋음) =====")
print(res.to_string(index=False))
print("\n참고: 리더보드 end=start 실측 = 20.7314  →  로컬 A와 비슷하면 검증 신뢰 OK")
