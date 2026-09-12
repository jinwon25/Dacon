# 데이터 및 누수 감사

- 생성 명령: `python -m src.audit --project-dir .`
- 원본 CSV를 수정하지 않았다. 명시적 `int32`/`float32`/`category` dtype으로 각각 한 번만 로드해 정확 통계를 계산했다.
- 배포 `test.csv`와 `sample_submission.csv`는 형식 확인용 5행이므로 범주 커버리지는 실제 비공개 평가셋을 대표하지 않는다.

## 파일 크기와 메모리

| file | rows | columns | csv_mb | optimized_memory_mb |
| --- | --- | --- | --- | --- |
| train.csv | 1475092 | 49 | 351.455 | 355.911 |
| test.csv | 5 | 48 | 0.002 | 0.002 |
| sample_submission.csv | 5 | 2 | 0.000 | 0.000 |
| trackman_history.csv | 1793078 | 30 | 337.432 | 157.990 |

## Train 컬럼 프로파일

| column | dtype | missing | missing_rate | nunique | constant |
| --- | --- | --- | --- | --- | --- |
| row_id | string | 0 | 0.000000 | 1475092 | False |
| season | int32 | 0 | 0.000000 | 6 | False |
| game_month | int32 | 0 | 0.000000 | 8 | False |
| game_dayofweek | int32 | 0 | 0.000000 | 7 | False |
| inning | int32 | 0 | 0.000000 | 13 | False |
| top_bottom | category | 0 | 0.000000 | 2 | False |
| game_type | category | 0 | 0.000000 | 2 | False |
| balls_before | int32 | 0 | 0.000000 | 4 | False |
| strikes_before | int32 | 0 | 0.000000 | 3 | False |
| outs_before | int32 | 0 | 0.000000 | 3 | False |
| run_top_before | int32 | 0 | 0.000000 | 30 | False |
| run_bot_before | int32 | 0 | 0.000000 | 25 | False |
| run_total_before | int32 | 0 | 0.000000 | 38 | False |
| score_diff_home | int32 | 0 | 0.000000 | 47 | False |
| score_diff_pitcher_team | int32 | 0 | 0.000000 | 52 | False |
| runner_on_1b | int32 | 0 | 0.000000 | 2 | False |
| runner_on_2b | int32 | 0 | 0.000000 | 2 | False |
| runner_on_3b | int32 | 0 | 0.000000 | 2 | False |
| num_runners_on | int32 | 0 | 0.000000 | 4 | False |
| base_state | category | 0 | 0.000000 | 8 | False |
| home_win_expectancy | float32 | 0 | 0.000000 | 977 | False |
| away_win_expectancy | float32 | 0 | 0.000000 | 981 | False |
| li | float32 | 0 | 0.000000 | 502 | False |
| pitcher_id | int32 | 0 | 0.000000 | 792 | False |
| batter_id | int32 | 0 | 0.000000 | 830 | False |
| pitcher_hand | int32 | 0 | 0.000000 | 2 | False |
| batter_hand | int32 | 0 | 0.000000 | 2 | False |
| pitcher_team_id | int32 | 0 | 0.000000 | 13 | False |
| batter_team_id | int32 | 0 | 0.000000 | 13 | False |
| asof_pitcher_n | int32 | 0 | 0.000000 | 15450 | False |
| asof_pitcher_success_rate | float32 | 792 | 0.000537 | 220897 | False |
| asof_pitcher_reverse_rate | float32 | 792 | 0.000537 | 228113 | False |
| asof_pitcher_middle_rate | float32 | 792 | 0.000537 | 120994 | False |
| asof_pitcher_ball_rate | float32 | 792 | 0.000537 | 144810 | False |
| asof_pitcher_strike_rate | float32 | 792 | 0.000537 | 131533 | False |
| asof_pitcher_prev1_game_success_rate | float32 | 29185 | 0.019785 | 1925 | False |
| asof_pitcher_prev3_game_success_rate | float32 | 29185 | 0.019785 | 6320 | False |
| asof_pitcher_prev5_game_success_rate | float32 | 29185 | 0.019785 | 9315 | False |
| asof_pitcher_prev1_game_middle_rate | float32 | 29185 | 0.019785 | 1338 | False |
| asof_pitcher_prev3_game_middle_rate | float32 | 29185 | 0.019785 | 4654 | False |
| asof_pitcher_prev5_game_middle_rate | float32 | 29185 | 0.019785 | 7297 | False |
| asof_batter_n | int32 | 0 | 0.000000 | 13928 | False |
| asof_batter_success_rate | float32 | 830 | 0.000563 | 166833 | False |
| asof_batter_middle_rate | float32 | 830 | 0.000563 | 92884 | False |
| asof_pitcher_pitchmix_n | int32 | 0 | 0.000000 | 15450 | False |
| asof_pitcher_fastball_rate | float32 | 792 | 0.000537 | 337589 | False |
| asof_pitcher_breaking_rate | float32 | 792 | 0.000537 | 386614 | False |
| asof_pitcher_offspeed_rate | float32 | 792 | 0.000537 | 449455 | False |
| control_success | int32 | 0 | 0.000000 | 2 | False |

## Trackman 컬럼 프로파일

| column | dtype | missing | missing_rate | nunique | constant |
| --- | --- | --- | --- | --- | --- |
| trackman_id | int32 | 0 | 0.000000 | 1793078 | False |
| season | int32 | 0 | 0.000000 | 6 | False |
| game_date | category | 0 | 0.000000 | 1059 | False |
| game_month | int32 | 0 | 0.000000 | 9 | False |
| game_dayofweek | int32 | 0 | 0.000000 | 7 | False |
| trackman_game_id | category | 0 | 0.000000 | 5980 | False |
| pitch_no | int32 | 0 | 0.000000 | 475 | False |
| inning | int32 | 0 | 0.000000 | 14 | False |
| top_bottom | category | 0 | 0.000000 | 2 | False |
| balls_before | int32 | 0 | 0.000000 | 5 | False |
| strikes_before | int32 | 0 | 0.000000 | 4 | False |
| outs_before | int32 | 0 | 0.000000 | 5 | False |
| pitch_of_pa | int32 | 0 | 0.000000 | 19 | False |
| pitcher_trackman_id | int32 | 0 | 0.000000 | 906 | False |
| batter_trackman_id | int32 | 0 | 0.000000 | 913 | False |
| pitcher_hand | category | 0 | 0.000000 | 2 | False |
| batter_hand | category | 0 | 0.000000 | 2 | False |
| pitcher_team | category | 0 | 0.000000 | 26 | False |
| batter_team | category | 0 | 0.000000 | 26 | False |
| tagged_pitch_type | category | 0 | 0.000000 | 17 | False |
| auto_pitch_type | category | 72 | 0.000040 | 12 | False |
| pitch_type_group | category | 0 | 0.000000 | 4 | False |
| rel_speed | float32 | 7617 | 0.004248 | 53443 | False |
| spin_rate | float32 | 12465 | 0.006952 | 200301 | False |
| induced_vert_break | float32 | 10062 | 0.005612 | 811701 | False |
| horz_break | float32 | 10333 | 0.005763 | 932261 | False |
| extension | float32 | 7716 | 0.004303 | 97799 | False |
| rel_height | float32 | 7617 | 0.004248 | 156473 | False |
| rel_side | float32 | 7618 | 0.004249 | 987657 | False |
| zone_speed | float32 | 7921 | 0.004418 | 52276 | False |

## 중복·식별자

- train 완전 중복 행: **0**
- train `row_id` 제외 입력+target 중복 행: **0**
- train 중복 `row_id`: **0**; 5행 test 중복 `row_id`: **0**
- Trackman 완전 중복 행: **0**; 중복 `trackman_id`: **0**
- 상수 train 컬럼: **없음**
- `row_id`는 제출 정렬/무결성 확인에만 쓰고 모델 입력에서 제외한다.

## Target과 주요 그룹 성공률

- 전체 `control_success=1` 비율: **0.523766**

| group | value | n | success_rate |
| --- | --- | --- | --- |
| season | 2019 | 237413 | 0.564670 |
| season | 2020 | 244087 | 0.532712 |
| season | 2021 | 247088 | 0.532762 |
| season | 2022 | 247472 | 0.528920 |
| season | 2023 | 245525 | 0.499957 |
| season | 2024 | 253507 | 0.486105 |
| game_month | 3 | 25756 | 0.537506 |
| game_month | 4 | 206638 | 0.525963 |
| game_month | 5 | 255547 | 0.532290 |
| game_month | 6 | 253454 | 0.526979 |
| game_month | 7 | 177147 | 0.521110 |
| game_month | 8 | 221570 | 0.519023 |
| game_month | 9 | 227845 | 0.520626 |
| game_month | 10 | 107135 | 0.509171 |
| game_dayofweek | 0 | 14754 | 0.499661 |
| game_dayofweek | 1 | 232390 | 0.524515 |
| game_dayofweek | 2 | 241756 | 0.524487 |
| game_dayofweek | 3 | 236322 | 0.522410 |
| game_dayofweek | 4 | 247962 | 0.524548 |
| game_dayofweek | 5 | 252121 | 0.524708 |
| game_dayofweek | 6 | 249787 | 0.523350 |
| top_bottom | B | 722280 | 0.524443 |
| top_bottom | T | 752812 | 0.523116 |
| game_type | F | 161004 | 0.603271 |
| game_type | R | 1314088 | 0.514025 |
| pitcher_hand | 1 | 381351 | 0.516705 |
| pitcher_hand | 2 | 1093741 | 0.526228 |
| batter_hand | 1 | 695747 | 0.520939 |
| batter_hand | 2 | 779345 | 0.526289 |
| balls_before | 0 | 652052 | 0.527568 |
| balls_before | 1 | 444780 | 0.525941 |
| balls_before | 2 | 254237 | 0.520782 |
| balls_before | 3 | 124023 | 0.502092 |
| strikes_before | 0 | 608121 | 0.524874 |
| strikes_before | 1 | 447852 | 0.527643 |
| strikes_before | 2 | 419119 | 0.518015 |
| outs_before | 0 | 505924 | 0.521398 |
| outs_before | 1 | 490634 | 0.523894 |
| outs_before | 2 | 478534 | 0.526138 |

표본 수 상위 투수 20명:

| pitcher_id | n | success_rate |
| --- | --- | --- |
| 23633 | 15450 | 0.537282 |
| 23719 | 14775 | 0.589442 |
| 22759 | 14288 | 0.500770 |
| 21961 | 13876 | 0.544249 |
| 22890 | 12939 | 0.490687 |
| 23745 | 12787 | 0.543443 |
| 22277 | 12743 | 0.532842 |
| 23782 | 12393 | 0.550149 |
| 23561 | 12331 | 0.572216 |
| 23767 | 11644 | 0.552559 |
| 23880 | 11433 | 0.573253 |
| 22908 | 11157 | 0.536793 |
| 21955 | 11115 | 0.521278 |
| 21916 | 11088 | 0.555826 |
| 22708 | 10202 | 0.497255 |
| 22834 | 10198 | 0.472151 |
| 22249 | 9878 | 0.502531 |
| 23903 | 9876 | 0.433678 |
| 22380 | 9788 | 0.529935 |
| 23987 | 9769 | 0.537210 |

## 시간축·검증 단위

- 메인 시즌: **[2019, 2020, 2021, 2022, 2023, 2024]**; Trackman 시즌: **[2019, 2020, 2021, 2022, 2023, 2024]**.
- 메인에는 `season`, 월, 요일만 있고 정확한 `game_date`, 경기 ID, 경기 내 투구 번호가 없다. 따라서 같은 경기 분리를 정확히 보장하는 game-group split은 구현할 수 없다.
- Trackman에만 `game_date`, `trackman_game_id`, `pitch_no`, `pitch_of_pa`가 있다. 두 테이블은 행 단위 대응 관계가 아니다.
- `row_id` 숫자 부분은 파일 순서와 함께 증가하지만, 식별자/순서를 피처로 쓰지 않는다. 실제 test 행은 독립 예측하며 다른 test 행을 참조하지 않는다.
- 실제 평가가 2025년이고 학습 최종 시즌이 2024년이므로, **2019~2023 train / 2024 validation**을 primary split으로 선택한다.

## 엔터티·Trackman 조인 감사

- 메인 고유 투수/타자: **792 / 830**; Trackman 고유 투수/타자: **906 / 913**.
- `pitcher_id = pitcher_trackman_id` 교집합: **0개**, train 행 커버리지 **0.000000%**, 미매칭 **100.000000%**.
- `batter_id = batter_trackman_id` 교집합: **0개**, train 행 커버리지 **0.000000%**, 미매칭 **100.000000%**.
- 따라서 제공 파일만으로 확인되는 선수 단위 실제 조인 키는 **없다**. 이름이 비슷하다는 이유로 ID를 결합하면 안 된다.
- 공통 경기상황 7열 조합은 Trackman 98,647개이며 행 중복률이 94.498455%라서 1:1 키가 아니라 과거 league-context 집계 키로만 사용 가능하다.
- Trackman의 손잡이는 `Right/Left`, 메인은 `1/2`; 팀도 문자열과 익명 정수로 표현되어 공식 매핑이 없다. 초기 안전 모델은 선수 단위 Trackman 결합을 하지 않고, 별도 실험에서 시즌 이전 Trackman만 사용한 경기상황 league-context 집계만 검증한다.

## 누수 점검

- 연속 투수 기록에서 직전 target을 반영한 다음 행의 `asof_pitcher_success_rate` 갱신식 일치: **1,473,508/1,473,508 (100.000000%)**. 이는 `asof_*`가 현재 행 이전 이력임을 강하게 지지한다.
- 단, train의 다음 행 `asof_*`를 역방향으로 당기면 현재 target을 복원할 수 있다. 따라서 어떤 backward shift/다음 행 참조도 금지하고 현재 행에 공식 제공된 값만 사용한다.
- `asof_*success/middle/reverse/ball/strike_rate`는 target 관련 과거 집계라 누수 민감 피처로 분류하지만, 설명서가 투구 직전 계산을 명시하므로 현재 행 값은 사용 가능하다. 성능은 해당 블록 제거 ablation으로 점검한다.
- 현재 투구 위치·판정·실제 구종·Trackman 측정값에 해당하는 컬럼은 메인 입력에 없다. `home_win_expectancy`, `away_win_expectancy`, `li`도 설명서상 투구 직전 값이다.
- `run_total_before`, 점수 차, `num_runners_on`, `base_state`는 다른 pre-pitch 열에서 재구성되는 중복 정보이나 target 누수는 아니다.

## 고카디널리티와 미지 범주

| column | nunique |
| --- | --- |
| row_id | 1475092 |
| home_win_expectancy | 977 |
| away_win_expectancy | 981 |
| li | 502 |
| pitcher_id | 792 |
| batter_id | 830 |
| asof_pitcher_n | 15450 |
| asof_pitcher_success_rate | 220897 |
| asof_pitcher_reverse_rate | 228113 |
| asof_pitcher_middle_rate | 120994 |
| asof_pitcher_ball_rate | 144810 |
| asof_pitcher_strike_rate | 131533 |
| asof_pitcher_prev1_game_success_rate | 1925 |
| asof_pitcher_prev3_game_success_rate | 6320 |
| asof_pitcher_prev5_game_success_rate | 9315 |
| asof_pitcher_prev1_game_middle_rate | 1338 |
| asof_pitcher_prev3_game_middle_rate | 4654 |
| asof_pitcher_prev5_game_middle_rate | 7297 |
| asof_batter_n | 13928 |
| asof_batter_success_rate | 166833 |
| asof_batter_middle_rate | 92884 |
| asof_pitcher_pitchmix_n | 15450 |
| asof_pitcher_fastball_rate | 337589 |
| asof_pitcher_breaking_rate | 386614 |
| asof_pitcher_offspeed_rate | 449455 |

- 5행 sample test의 train-known 투수/타자 비율: **40.00% / 60.00%**. 실제 평가 커버리지는 알 수 없다.
- 범주는 train에서만 사전을 학습하고 validation/test 미지값은 전용 unknown 코드로 보낸다. test 빈도나 test 전체 분포는 계산하지 않는다.

## 설명서와 실제 파일 차이

- 행/열 수는 설명서와 일치한다. 배포 test와 sample submission은 각각 5행이다.
- 공식 `baseline_submit.zip` 원본은 없고 동일 내용으로 보이는 해제 폴더 `baseline_submit/`와 학습/추론 노트북이 제공됐다.
- 설명서는 손잡이를 '코드'로만 설명하며 1/2 의미 매핑을 제공하지 않는다. 이를 자의적으로 Right/Left로 확정하지 않는다.
- 메인에는 game/date/pitch-order 열이 없어 경기 단위 중복 여부와 정확한 경기 group split을 직접 확인할 수 없다.

## 감사 결론

1. `row_id`를 제외한 공식 pre-pitch 입력과 현재 행의 `asof_*`만 기본 피처로 사용한다.
2. primary validation은 2024 season-forward holdout이며, 2023 OOF를 calibration/blend 선택용으로 별도 생성한다.
3. Trackman 선수 ID 직접 결합은 커버리지가 없어 사용하지 않는다. 누수 없는 과거 league-context 집계는 별도 ablation에서만 승격한다.
4. 실제 test의 다른 행, 순서, 빈도, 분포, 정규화 통계는 어느 단계에서도 사용하지 않는다.
