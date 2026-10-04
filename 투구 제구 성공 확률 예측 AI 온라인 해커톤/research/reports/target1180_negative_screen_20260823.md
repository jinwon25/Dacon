# 1180 상당 후보 스크리닝: 부정 결과 (2026-08-23)

## 결론

**1180 상당 후보를 확보하지 못했다.** 제안한 신규 가설 5개 전부가 저장소에 이미 구현·평가되어
기각된 계열과 사실상 동일했다 — 일부는 변수명(`win_expectancy_gap`, `ctx_win_gap`)까지 정확히
일치했다. 심화 검증(3축 부호 일관성, 클러스터 부트스트랩, Reality Check) 단계에 도달한 가설이
하나도 없어 4단계 판정표는 전부 "해당 없음"이다.

## 1단계: 기존 원장 규모

- 버전 스크립트 `src/v*.py` **180개**, 버전 config `configs/v*.json` **58개**, 리포트
  `reports/*.md` **103개**.
- `docs/PROJECT_STATUS.md`에 명시적으로 "종료 상태"로 기록된 계열만도 잔차 lookup류, 다년
  직접모형, latent state/mode, 임베딩 신경망(TabM 포함), 그룹 강건 학습, post-break spline,
  TrackMan IVB·구종·물리-teacher student(1,793,078행 전수조사 완료), LI/점수상황형, 레벨별
  선수이력형, domain×count lookup, environment-stable ridge, 계절/성숙도 브릿지(v152/v154, 오늘
  실측 기각), source-stability mask 미세조정, ABS/시즌 regime DID, 투수별 희소 ridge random
  slope, TrackMan arsenal ridge, recent-game centered spread를 포함한다.
- 결정적으로, 챔피언 계열 모델(`v104`/`v124` parent)의 `feature_spec.json`을 직접 열어
  확인한 결과 `base_columns`에 **데이터 설명서의 모든 원본 컬럼**(`base_state`,
  `home_win_expectancy`, `away_win_expectancy`, `li`, `asof_pitcher_*` 12종,
  `asof_batter_*` 3종, `asof_pitcher_pitchmix_n`/`fastball_rate`/`breaking_rate`/
  `offspeed_rate`, `game_dayofweek` 포함)가 이미 포함돼 있고, `categorical_columns`에는
  `count_state`, `platoon`, `pitcher_count`, `batter_count`, `team_matchup`,
  `situation_state` 같은 파생 교차 카테고리까지 이미 들어 있다. 즉 원본 3개 데이터 파일의
  "raw feature" 차원에서는 새로 추가할 여지가 사실상 없다.

## 2~3단계: 신규 가설과 기존 증거 대조

| 가설 | 대응하는 기존 구현/평가 | 상태 |
|---|---|---|
| `base_state` 8분류를 세분화한 잔차 축 | `context_residual_screen.py`의 `domain_teams` 등 다중 그룹, `base_state`는 이미 챔피언 categorical feature이자 `situation_state` 파생 카테고리에 포함 | 이미 존재, 재탕 |
| `home_win_expectancy`/`away_win_expectancy` 비대칭도 | `configs/v78_environment_stable_residual.json`의 `win_expectancy_gap`, `src/archive/v79_champion_offset_context.py`의 `ctx_win_gap = home_win_expectancy - away_win_expectancy` — 변수명까지 동일. PROJECT_STATUS: "v78 ... primary 두 축 모두 source eta 0", "v79 ... primary 두 축 모두 eta 0; 기각" | 이미 시도·기각 (eta=0, 양쪽 축 모두) |
| `asof_batter_*`를 주 신호로 쓰는 축 | `src/archive/v107_batter_asof_ablation.py` "Strict-forward paired removal of batter cumulative ASOF summaries" — 가설 자체가 "타자 누적 요약을 제거하면 비이식 노이즈가 줄어든다"로, 타자 축이 이미 순신호가 아니라 잡음으로 판정됨 | 이미 시도·기각 (역방향 확인: 강화가 아니라 제거가 이득) |
| `game_dayofweek` 패턴 | 챔피언 `categorical_columns`에 처음부터 포함, `v10`/`v14`/`v20`/`v23`/`v24`/`v37`/`v62`/`v97`/`v109` 등 다수 스크립트에서 이미 표준 카테고리로 사용 | 이미 base feature, 별도 신규 축 아님 |
| `asof_pitcher_pitchmix_n`/fastball·breaking·offspeed rate의 다양성(엔트로피) 축 | `src/champion/v26_exact_diversity_screen.py`, `src/archive/v30_diverse_covariance_screen.py`, `src/trackman_domain.py` — PROJECT_STATUS v65~v74 "TrackMan 1,793,078행 전수조사"에 구종/다양성 계열 포함, 기각 | 이미 시도·기각 |

5개 전부 1차 스크리닝 이전에 "이미 시도됨"으로 확정돼 실행 자체를 생략했다 (문서·코드 증거가
명확해 중복 계산으로 컴퓨팅을 낭비하지 않기 위함).

## 4단계: 심화 검증 게이트

해당 없음 — 심화 단계에 도달한 가설이 없다.

| 기준 | 결과 |
|---|---|
| 로컬 full-2024 gain ≥ +30 | 해당 없음 |
| 3축 부호 일관성 | 해당 없음 |
| 클러스터 부트스트랩 CI 하한 > 0 | 해당 없음 |
| White Reality Check | 해당 없음 |
| 규정 준수 | 해당 없음 |
| 완전히 새로운 메커니즘 | 5/5 실패 (전부 기존 계열과 동일) |

## 최종 판정

**1180 상당 후보 확보 실패.** 이유는 두 가지다.

1. **데이터 소스 고갈**: `train.csv`/`test.csv`/`trackman_history.csv` 세 파일이 제공하는
   원본 신호는 챔피언 계열 모델의 base feature 목록에 이미 전부 포함돼 있고, 파생
   교차·잔차·다양성 축까지 180개 버전 스크립트가 반복적으로 탐색했다. 오늘 제안한 5개
   신규 가설은 전부 기존 코드·결과와 동일하거나(변수명 일치 포함) 이미 반대 방향으로
   판정(타자 축)됐다.
2. **local→Public 전이율의 근본적 불확실성**: 설령 위 제약이 없었다 해도, 오늘 v152/v154가
   `locked_gate_passed=true`·다년도 축 전부 양수(+11.4~+17.6)였는데 실제 Public은
   **음의 방향(-1.898)으로** 반전한 사례가 이미 나왔다. 로컬 검증만으로 "1180 상당"을
   자체 판단하는 것 자체가 이 프로젝트의 실측 데이터로 이미 반증된 방법론이다.

## 다음 라운드 제언

현재 3개 파일 기반으로는 추가로 시도할 만한 새 가설이 사실상 남아있지 않다고 판단한다.
후속 연구가 의미를 가지려면 `docs/PROJECT_STATUS.md`의 종료 조건 1번(팀원 모델의 독립 OOF)
또는 대회 규정상 허용되는 완전히 새로운 외부 데이터 확보가 선행돼야 한다. 그마저도 없다면,
남은 유일한 저위험 레버는 이미 실측된 v142(0)→v148(0.15) 양의 기울기 구간 안에서
중간값(예: 0.2~0.35)을 다음 제출권으로 값싸게 탐침해 3번째 실측 앵커를 확보하는 것뿐이며,
이는 새 가설이 아니라 기존 v142→v148 축의 보정 개선이다.
