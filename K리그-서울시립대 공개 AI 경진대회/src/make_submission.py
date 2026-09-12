"""개선 베이스라인(B') 제출 파일 생성.

방법:  도착 = clip( 출발 + 평균전진(DX, DY) )  ,  경기장 [0,105]x[0,68] 안으로 clip
평균전진(DX, DY)은 train 전체의 (마지막 패스 도착 - 출발) 평균으로 계산.
로컬 검증(src/local_cv.py)에서 valid 18.16 로, end=start(20.34)보다 우수했던 방법.

실행 (대회 폴더에서):
    python src/make_submission.py
출력: submissions/baseline_v2_start_plus_forward.csv
"""
import os

import duckdb
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
con = duckdb.connect()

# 1) train 전체에서 평균 전진량 학습
DX, DY = con.execute("""
    WITH r AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
        FROM 'data/train.csv'
    )
    SELECT AVG(end_x - start_x), AVG(end_y - start_y)
    FROM r WHERE rn = 1 AND end_x IS NOT NULL
""").fetchone()
print(f"[학습] 평균 전진 DX={DX:+.2f}, DY={DY:+.2f}  (train 전체)")

# 2) test 각 에피소드의 마지막 패스 출발점
pred = con.execute("""
    WITH ev AS (
        SELECT game_episode, start_x, start_y,
               ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
        FROM read_csv('data/test/*/*.csv', union_by_name = true)
    )
    SELECT game_episode, start_x, start_y FROM ev WHERE rn = 1
""").df()

# 3) 예측 = clip(출발 + 평균전진)
pred["end_x"] = (pred["start_x"] + DX).clip(0, 105)
pred["end_y"] = (pred["start_y"] + DY).clip(0, 68)

# 4) sample_submission 순서에 맞추기
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(pred[["game_episode", "end_x", "end_y"]], on="game_episode", how="left")
miss = int(out["end_x"].isna().sum())
out["end_x"] = out["end_x"].fillna(52.5)
out["end_y"] = out["end_y"].fillna(34.0)

os.makedirs("submissions", exist_ok=True)
out_path = "submissions/baseline_v2_start_plus_forward.csv"
out.to_csv(out_path, index=False)
print(f"[완료] {out_path}  (행수={len(out)}, 매칭실패={miss})")
print(out.head().to_string(index=False))
