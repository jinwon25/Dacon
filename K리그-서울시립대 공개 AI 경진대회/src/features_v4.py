"""v4 풍부한 피처 — '이벤트 종류·맥락'에 신호가 있다는 단서(prev_type 0.8)를 밀고 나간다.

추가하는 것
  · 같은 팀 직전 터치(st_*): 지금 prev_*는 상대팀 동시 이벤트가 섞일 수 있음 → 공격팀의 진짜 직전 터치만.
  · 더 깊은 과거: prev3/prev4, w3/w10 윈도우(짧은·긴 빌드업 동시에).
  · 이벤트 종류 흐름: 최근 10개 중 Cross/Take-On/Throw-In/Shot/세트피스/Duel/Recovery 개수.
  · 각도 sin/cos 분해(±π 끊김 방지) + 템포(직전 이벤트와의 시간 간격).
파생 좌표/각도는 파이썬에서 계산.
"""
import duckdb
import numpy as np

SQL = """
WITH ev AS (
    SELECT
        game_episode, game_id, action_id, time_seconds, type_name, team_id, player_id,
        start_x, start_y, end_x, end_y, CAST(is_home AS INT) AS is_home, period_id,
        ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn,
        COUNT(*)     OVER (PARTITION BY game_episode) AS n_events
    FROM {source}
),
fp AS (   -- 마지막 패스(예측 대상)
    SELECT game_episode, team_id AS fteam, action_id AS fpa FROM ev WHERE rn = 1
),
samet AS ( -- 마지막 패스와 '같은 팀'의, 패스 이전 터치들 (최근순)
    SELECT e.game_episode, e.start_x, e.start_y, e.end_x, e.end_y, e.time_seconds,
           ROW_NUMBER() OVER (PARTITION BY e.game_episode ORDER BY e.action_id DESC) AS srn
    FROM ev e JOIN fp f ON e.game_episode = f.game_episode
    WHERE e.team_id = f.fteam AND e.action_id < f.fpa
),
st AS (   -- 같은 팀 직전 터치 1개
    SELECT game_episode,
           end_x - start_x AS st_dx, end_y - start_y AS st_dy,
           end_x AS st_end_x, end_y AS st_end_y, time_seconds AS st_time
    FROM samet WHERE srn = 1
),
st2 AS (  -- 같은 팀 그 전 터치
    SELECT game_episode, end_x - start_x AS st2_dx, end_y - start_y AS st2_dy
    FROM samet WHERE srn = 2
),
agg AS (
    SELECT
        game_episode,
        MAX(game_id) AS game_id,
        MAX(CASE WHEN rn = 1 THEN start_x END)      AS start_x,
        MAX(CASE WHEN rn = 1 THEN start_y END)      AS start_y,
        MAX(CASE WHEN rn = 1 THEN time_seconds END) AS time_s,
        MAX(CASE WHEN rn = 1 THEN is_home END)      AS is_home,
        MAX(CASE WHEN rn = 1 THEN period_id END)    AS period_id,
        MAX(CASE WHEN rn = 1 THEN n_events END)     AS n_events,
        MAX(CASE WHEN rn = 1 THEN team_id END)      AS team_id,
        MAX(CASE WHEN rn = 1 THEN player_id END)    AS player_id,
        MAX(CASE WHEN rn = 1 THEN end_x END)        AS tgt_x,
        MAX(CASE WHEN rn = 1 THEN end_y END)        AS tgt_y,
        -- 직전 이벤트들(원본 rn 기준, 상대 이벤트 섞일 수 있음 — 그래도 신호)
        MAX(CASE WHEN rn = 2 THEN end_x - start_x END) AS prev_dx,
        MAX(CASE WHEN rn = 2 THEN end_y - start_y END) AS prev_dy,
        MAX(CASE WHEN rn = 2 THEN type_name END)       AS prev_type,
        MAX(CASE WHEN rn = 2 THEN end_x END)           AS prev_end_x,
        MAX(CASE WHEN rn = 2 THEN end_y END)           AS prev_end_y,
        MAX(CASE WHEN rn = 2 THEN time_seconds END)    AS prev_time,
        MAX(CASE WHEN rn = 3 THEN time_seconds END)    AS prev2_time,
        MAX(CASE WHEN rn = 4 THEN time_seconds END)    AS prev3_time,
        MAX(CASE WHEN rn = 6 THEN time_seconds END)    AS t_at6,
        MAX(CASE WHEN rn = 3 THEN end_x - start_x END) AS prev2_dx,
        MAX(CASE WHEN rn = 3 THEN end_y - start_y END) AS prev2_dy,
        MAX(CASE WHEN rn = 4 THEN end_x - start_x END) AS prev3_dx,
        MAX(CASE WHEN rn = 4 THEN end_y - start_y END) AS prev3_dy,
        MAX(CASE WHEN rn = 5 THEN end_x - start_x END) AS prev4_dx,
        MAX(CASE WHEN rn = 5 THEN end_y - start_y END) AS prev4_dy,
        -- 윈도우 순이동 / 경로
        SUM(CASE WHEN rn BETWEEN 2 AND 4  THEN end_x - start_x ELSE 0 END) AS w3_dx,
        SUM(CASE WHEN rn BETWEEN 2 AND 4  THEN end_y - start_y ELSE 0 END) AS w3_dy,
        SUM(CASE WHEN rn BETWEEN 2 AND 6  THEN end_x - start_x ELSE 0 END) AS w5_dx,
        SUM(CASE WHEN rn BETWEEN 2 AND 6  THEN end_y - start_y ELSE 0 END) AS w5_dy,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 THEN end_x - start_x ELSE 0 END) AS w10_dx,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 THEN end_y - start_y ELSE 0 END) AS w10_dy,
        SUM(CASE WHEN rn BETWEEN 2 AND 6
                 THEN sqrt((end_x-start_x)*(end_x-start_x)+(end_y-start_y)*(end_y-start_y))
                 ELSE 0 END) AS w5_pathlen,
        SUM(CASE WHEN rn BETWEEN 2 AND 11
                 THEN sqrt((end_x-start_x)*(end_x-start_x)+(end_y-start_y)*(end_y-start_y))
                 ELSE 0 END) AS w10_pathlen,
        SUM(CASE WHEN rn BETWEEN 2 AND 6 AND type_name='Pass'  THEN 1 ELSE 0 END) AS w5_npass,
        SUM(CASE WHEN rn BETWEEN 2 AND 6 AND type_name='Carry' THEN 1 ELSE 0 END) AS w5_ncarry,
        -- 이벤트 종류 흐름 (최근 10개)
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name='Cross'    THEN 1 ELSE 0 END) AS n_cross,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name='Take-On'  THEN 1 ELSE 0 END) AS n_takeon,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name='Throw-In' THEN 1 ELSE 0 END) AS n_throwin,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name LIKE 'Shot%' THEN 1 ELSE 0 END) AS n_shot,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name IN ('Pass_Freekick','Pass_Corner','Goal Kick')
                 THEN 1 ELSE 0 END) AS n_setpiece,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name='Duel'     THEN 1 ELSE 0 END) AS n_duel,
        SUM(CASE WHEN rn BETWEEN 2 AND 11 AND type_name='Recovery' THEN 1 ELSE 0 END) AS n_recovery
    FROM ev GROUP BY game_episode
)
SELECT a.*, s.st_dx, s.st_dy, s.st_end_x, s.st_end_y, s.st_time,
       s2.st2_dx, s2.st2_dy
FROM agg a LEFT JOIN st s USING (game_episode)
           LEFT JOIN st2 s2 USING (game_episode)
"""

CATEGORICAL_V4 = ["prev_type"]


def _ang(dx, dy):
    a = np.arctan2(dy, dx)
    return np.sin(a), np.cos(a)


def build_features_v4(source: str):
    con = duckdb.connect()
    df = con.execute(SQL.format(source=source)).df()
    eps = 1e-6
    # 위치/거리
    df["dist_to_goal_x"] = 105 - df["start_x"]
    df["dist_from_cen_y"] = (df["start_y"] - 34).abs()
    df["dist_to_goal"] = np.hypot(105 - df["start_x"], 34 - df["start_y"])
    # 윈도우 직선 이동량/직선성
    for w in ("w3", "w5", "w10"):
        df[f"{w}_netdist"] = np.hypot(df[f"{w}_dx"], df[f"{w}_dy"])
    df["w5_straight"] = df["w5_netdist"] / (df["w5_pathlen"] + eps)
    df["w10_straight"] = df["w10_netdist"] / (df["w10_pathlen"] + eps)
    # 각도 sin/cos
    df["w5_ang_sin"], df["w5_ang_cos"] = _ang(df["w5_dx"], df["w5_dy"])
    df["w10_ang_sin"], df["w10_ang_cos"] = _ang(df["w10_dx"], df["w10_dy"])
    df["prev_ang_sin"], df["prev_ang_cos"] = _ang(df["prev_dx"].fillna(0), df["prev_dy"].fillna(0))
    df["goal_ang_sin"], df["goal_ang_cos"] = _ang(105 - df["start_x"], 34 - df["start_y"])
    # 같은 팀 직전 터치
    df["st_dist"] = np.hypot(df["st_dx"], df["st_dy"])
    df["st_ang_sin"], df["st_ang_cos"] = _ang(df["st_dx"].fillna(0), df["st_dy"].fillna(0))
    df["st_gap"] = df["time_s"] - df["st_time"]
    # 템포 (t_gap1이 v4 importance 1위 → 타이밍 신호 더 캔다)
    df["t_gap1"] = df["time_s"] - df["prev_time"]
    df["t_gap2"] = df["prev_time"] - df["prev2_time"]
    df["t_gap3"] = df["prev2_time"] - df["prev3_time"]
    df["span5"] = df["time_s"] - df["t_at6"]          # 최근 5개 이벤트가 걸린 시간(전개 속도)
    df["mean_gap5"] = df["span5"] / 5.0
    # 같은 팀 2-터치 누적 방향 + 그 전 터치
    df["st2_dist"] = np.hypot(df["st2_dx"], df["st2_dy"])
    df["st_net_dx"] = df["st_dx"].fillna(0) + df["st2_dx"].fillna(0)
    df["st_net_dy"] = df["st_dy"].fillna(0) + df["st2_dy"].fillna(0)
    return df


FEATURE_COLS_V4 = [
    "start_x", "start_y", "time_s", "is_home", "period_id", "n_events",
    "dist_to_goal_x", "dist_from_cen_y", "dist_to_goal",
    "prev_dx", "prev_dy", "prev2_dx", "prev2_dy", "prev3_dx", "prev3_dy", "prev4_dx", "prev4_dy",
    "prev_type", "prev_end_x", "prev_end_y",
    "w3_dx", "w3_dy", "w5_dx", "w5_dy", "w10_dx", "w10_dy",
    "w5_pathlen", "w10_pathlen", "w5_npass", "w5_ncarry",
    "w3_netdist", "w5_netdist", "w10_netdist", "w5_straight", "w10_straight",
    "w5_ang_sin", "w5_ang_cos", "w10_ang_sin", "w10_ang_cos",
    "prev_ang_sin", "prev_ang_cos", "goal_ang_sin", "goal_ang_cos",
    "n_cross", "n_takeon", "n_throwin", "n_shot", "n_setpiece", "n_duel", "n_recovery",
    "t_gap1", "t_gap2", "t_gap3", "span5", "mean_gap5",
    "st_dx", "st_dy", "st_dist", "st_ang_sin", "st_ang_cos", "st_end_x", "st_end_y", "st_gap",
    "st2_dx", "st2_dy", "st2_dist", "st_net_dx", "st_net_dy",
]
