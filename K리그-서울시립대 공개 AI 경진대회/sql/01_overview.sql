-- =====================================================================
-- 01_overview.sql  —  데이터 첫 탐색 (EDA)
-- 실행:  python src/run_sql.py sql/01_overview.sql   (대회 폴더 안에서)
-- =====================================================================

-- @1. train 전체 규모: 이벤트(행) 수 · 경기 수 · 에피소드 수
SELECT
    COUNT(*)                     AS 이벤트_행수,
    COUNT(DISTINCT game_id)      AS 경기수,
    COUNT(DISTINCT game_episode) AS 에피소드수
FROM 'data/train.csv';

-- @2. 이벤트 종류(type_name)별 빈도 — 상위 15개
--     SUM(COUNT(*)) OVER () : 전체 합계를 같이 구해 비율(%)을 계산하는 윈도우 함수
SELECT
    type_name,
    COUNT(*)                                        AS 건수,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS 비율_pct
FROM 'data/train.csv'
GROUP BY type_name
ORDER BY 건수 DESC
LIMIT 15;

-- @3. 에피소드당 이벤트 개수 분포 (한 플레이가 보통 몇 개 이벤트로 이뤄지나)
WITH ep AS (
    SELECT game_episode, COUNT(*) AS n_events
    FROM 'data/train.csv'
    GROUP BY game_episode
)
SELECT
    MIN(n_events)            AS 최소,
    ROUND(AVG(n_events), 1)  AS 평균,
    MEDIAN(n_events)         AS 중앙값,
    QUANTILE_CONT(n_events, 0.90) AS p90,
    MAX(n_events)            AS 최대
FROM ep;

-- @4. 각 에피소드의 '마지막 이벤트'는 무슨 종류인가? (= 우리가 예측할 대상 확인)
--     ROW_NUMBER() ... PARTITION BY game_episode ORDER BY action_id DESC : 에피소드별 맨 끝 행 뽑기
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
    FROM 'data/train.csv'
)
SELECT
    type_name AS 마지막_이벤트_종류,
    COUNT(*)  AS 에피소드수,
    SUM(CASE WHEN end_x IS NULL THEN 1 ELSE 0 END) AS end좌표_없음
FROM ranked
WHERE rn = 1
GROUP BY type_name
ORDER BY 에피소드수 DESC;

-- @5. 예측 대상(마지막 패스)의 도착 좌표 분포 + 시작좌표와의 관계 (★ 베이스라인 직관)
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY game_episode ORDER BY action_id DESC) AS rn
    FROM 'data/train.csv'
)
SELECT
    ROUND(AVG(start_x), 1) AS 시작x평균, ROUND(AVG(start_y), 1) AS 시작y평균,
    ROUND(AVG(end_x), 1)   AS 도착x평균, ROUND(AVG(end_y), 1)   AS 도착y평균,
    ROUND(AVG(end_x - start_x), 1) AS dx평균_전진정도,
    ROUND(AVG(end_y - start_y), 1) AS dy평균,
    ROUND(AVG(sqrt((end_x - start_x) * (end_x - start_x)
                 + (end_y - start_y) * (end_y - start_y))), 2) AS 패스길이평균
FROM ranked
WHERE rn = 1 AND type_name = 'Pass' AND end_x IS NOT NULL;
