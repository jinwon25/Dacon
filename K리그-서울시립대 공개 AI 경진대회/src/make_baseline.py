"""end = start 베이스라인 제출 파일 생성 (DuckDB SQL).

아이디어: 테스트의 마지막 패스는 '출발점(start_x, start_y)'만 주어지고 도착점이 가려져 있다.
가장 단순한 추측 = "공은 출발한 자리 근처에 떨어진다" → 도착 예측 = 출발 좌표.
(EDA에서 마지막 패스 평균 길이가 ~20.4였으므로, 이 베이스라인의 기대 점수도 대략 그 부근.)

실행 (대회 폴더에서):
    python src/make_baseline.py
출력: submissions/baseline_end_eq_start.csv
"""
import os

import duckdb
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)  # data/, submissions/ 를 상대경로로 쓰기 위해 대회 폴더로 이동

con = duckdb.connect()

# 1) 모든 테스트 에피소드 CSV를 한 번에 읽어, 에피소드별 '마지막 이벤트(=예측 대상 패스)'만 추출
PRED_SQL = """
WITH ev AS (
    SELECT
        game_episode, action_id, start_x, start_y,
        ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
    FROM read_csv('data/test/*/*.csv', union_by_name = true)
)
SELECT
    game_episode,
    start_x AS end_x,   -- 도착 ≈ 출발 (베이스라인 가정)
    start_y AS end_y
FROM ev
WHERE rn = 1
"""
pred = con.execute(PRED_SQL).df()

# 2) sanity check: 마지막 행이 정말 '도착좌표가 가려진(NULL)' 행이 맞는지 확인
CHECK_SQL = """
WITH ev AS (
    SELECT game_episode, end_x,
           ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
    FROM read_csv('data/test/*/*.csv', union_by_name = true)
)
SELECT COUNT(*) AS 마지막행수,
       SUM(CASE WHEN end_x IS NULL THEN 1 ELSE 0 END) AS end_결측수
FROM ev WHERE rn = 1
"""
print("[확인] 테스트 마지막 행 / 도착좌표 결측:")
print(con.execute(CHECK_SQL).df().to_string(index=False))

# 3) sample_submission 순서/행에 맞춰 정렬 (제출은 행 순서·구성이 정확히 맞아야 함)
sub = pd.read_csv("data/sample_submission.csv")[["game_episode"]]
out = sub.merge(pred, on="game_episode", how="left")
missing = int(out["end_x"].isna().sum())
out["end_x"] = out["end_x"].fillna(52.5)  # 혹시 매칭 안 되면 경기장 중앙으로 보정
out["end_y"] = out["end_y"].fillna(34.0)

os.makedirs("submissions", exist_ok=True)
out_path = "submissions/baseline_end_eq_start.csv"
out.to_csv(out_path, index=False)

print(f"\n[완료] {out_path}  (행수={len(out)}, sample 매칭 실패={missing})")
print(out.head().to_string(index=False))
