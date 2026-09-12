"""에피소드 → 피처 한 줄 만들기 (DuckDB SQL).

상위 솔루션들의 공통 인사이트: "답은 *마지막 몇 개 이벤트의 위치·방향·거리*가 결정한다."
그래서 각 에피소드에서 아래를 한 줄(=한 행)로 요약한다.

[알려진 정보 — 예측 대상 패스(rn=1)]
  start_x, start_y : 마지막 패스의 출발점 (★가장 중요. B' 베이스라인의 핵심)
  time_s           : 그 패스 시점(초)
  is_home, period_id, n_events : 맥락(홈/원정, 전·후반, 에피소드 길이)

[빌드업 — 직전 이벤트들에서]
  prev_dx, prev_dy   : 직전 이벤트(rn=2)의 이동벡터 = "공이 어느 방향으로 들어왔나"
  prev2_dx, prev2_dy : 그 전(rn=3)의 이동벡터
  w5_dx, w5_dy       : 최근 5개(rn 2~6) 순이동 합 = "공격이 향하던 방향" (한 수보다 덜 noisy)
  w5_pathlen         : 최근 5개 이동거리 합 = "빌드업이 얼마나 분주했나"
  w5_npass, w5_ncarry: 최근 5개 중 패스/드리블 개수
  prev_type          : 직전 이벤트 종류(범주) — 드리블 중이었나 패스였나 등

[파생]
  dist_to_goal_x  : 105 - start_x  (상대 골라인까지 X거리)
  dist_from_cen_y : |start_y - 34| (세로 중앙에서 벗어난 정도)

정답(학습용): tgt_x, tgt_y = 마지막 패스 도착 좌표 (test 에서는 NULL).
"""
import duckdb

FEATURE_SQL = """
WITH ev AS (
    SELECT
        game_episode, game_id, action_id, time_seconds, type_name,
        team_id, player_id,
        start_x, start_y, end_x, end_y,
        CAST(is_home AS INT) AS is_home, period_id,
        ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn,
        COUNT(*)     OVER (PARTITION BY game_episode) AS n_events
    FROM {source}
)
SELECT
    game_episode,
    MAX(game_id) AS game_id,
    -- 예측 대상 패스(rn=1)의 '알려진' 정보
    MAX(CASE WHEN rn = 1 THEN start_x END)      AS start_x,
    MAX(CASE WHEN rn = 1 THEN start_y END)      AS start_y,
    MAX(CASE WHEN rn = 1 THEN time_seconds END) AS time_s,
    MAX(CASE WHEN rn = 1 THEN is_home END)      AS is_home,
    MAX(CASE WHEN rn = 1 THEN period_id END)    AS period_id,
    MAX(CASE WHEN rn = 1 THEN n_events END)     AS n_events,
    -- 타깃 인코딩용 식별자 (모델에 raw로 넣지 않고, 그룹 평균으로 변환해 사용)
    MAX(CASE WHEN rn = 1 THEN team_id END)      AS team_id,
    MAX(CASE WHEN rn = 1 THEN player_id END)    AS player_id,
    -- 정답 (test 에서는 NULL)
    MAX(CASE WHEN rn = 1 THEN end_x END)        AS tgt_x,
    MAX(CASE WHEN rn = 1 THEN end_y END)        AS tgt_y,
    -- 직전 이벤트(rn=2): 공이 들어온 방향
    MAX(CASE WHEN rn = 2 THEN end_x - start_x END) AS prev_dx,
    MAX(CASE WHEN rn = 2 THEN end_y - start_y END) AS prev_dy,
    MAX(CASE WHEN rn = 2 THEN type_name END)       AS prev_type,
    -- 그 전(rn=3)
    MAX(CASE WHEN rn = 3 THEN end_x - start_x END) AS prev2_dx,
    MAX(CASE WHEN rn = 3 THEN end_y - start_y END) AS prev2_dy,
    -- 최근 5개 빌드업(rn 2~6) 요약
    SUM(CASE WHEN rn BETWEEN 2 AND 6 THEN end_x - start_x ELSE 0 END) AS w5_dx,
    SUM(CASE WHEN rn BETWEEN 2 AND 6 THEN end_y - start_y ELSE 0 END) AS w5_dy,
    SUM(CASE WHEN rn BETWEEN 2 AND 6
             THEN sqrt((end_x - start_x) * (end_x - start_x)
                     + (end_y - start_y) * (end_y - start_y))
             ELSE 0 END)                                              AS w5_pathlen,
    SUM(CASE WHEN rn BETWEEN 2 AND 6 AND type_name = 'Pass'  THEN 1 ELSE 0 END) AS w5_npass,
    SUM(CASE WHEN rn BETWEEN 2 AND 6 AND type_name = 'Carry' THEN 1 ELSE 0 END) AS w5_ncarry
FROM ev
GROUP BY game_episode
"""

# 모델에 넣을 피처 목록 (id/정답 제외)
FEATURE_COLS = [
    "start_x", "start_y", "time_s", "is_home", "period_id", "n_events",
    "prev_dx", "prev_dy", "prev2_dx", "prev2_dy",
    "w5_dx", "w5_dy", "w5_pathlen", "w5_npass", "w5_ncarry",
    "dist_to_goal_x", "dist_from_cen_y", "prev_type",
]
CATEGORICAL = ["prev_type"]


def build_features(source: str):
    """source 예: \"'data/train.csv'\"  또는  \"read_csv('data/test/*/*.csv', union_by_name=true)\""""
    con = duckdb.connect()
    df = con.execute(FEATURE_SQL.format(source=source)).df()
    df["dist_to_goal_x"] = 105 - df["start_x"]
    df["dist_from_cen_y"] = (df["start_y"] - 34).abs()
    return df


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    tr = build_features("'data/train.csv'")
    print("train 피처 shape:", tr.shape)
    print(tr[FEATURE_COLS].head().to_string())
