# 투구 제구 데이터·Trackman 전수 재점검 (2026-08-09)

## 1. 제출 결과와 현재 기준

| 제출 | 파일 | 점수 | v2 대비 | 결론 |
|---|---|---:|---:|---|
| submission3 edit (39023) | `submit_v2.zip` | 763.2665303697 | 기준 | 현재 Public champion |
| submission4 edit (40116) | `submit_v3.zip` | 761.9846367188 | -1.2818936509 | Trackman 10% 확대 폐기 |
| submission5 edit (40149) | `submit_v4.zip` | 743.6520325296 | -19.6144978401 | R-only RF 25% 폐기 |
| submission6 edit (40151) | `submit_v5.zip` | 741.0198690607 | -22.2466613090 | R-only + Trackman 폐기 |

현 시점의 운영 기준은 `submit_v2.zip`이다. v4/v5의 하락은 “regular-season 전용 모델이 더 정확하다”는 가설과 Trackman 신호를 단순 가중하는 가설 모두 Public에서 반증한다. 새 ZIP은 최신 연도 forward 검증과 독립적인 변경 근거가 없으면 만들지 않는다.

## 2. 데이터 구조와 drift

- `train.csv`: 1,475,092행 × 49열, 2019~2024, target 평균 0.523766.
- `trackman_history.csv`: 1,793,078행 × 30열, 2019~2024. 직접 join key가 없는 보조 로그다.
- `test.csv`는 공식 배포본에서 5행 확인용 샘플일 뿐이며, 서버가 숨은 전체 2025 test로 대체한다. 샘플을 전체 test의 분포로 간주하지 않는다.
- 연도별 target: 2019 0.564670, 2020 0.532712, 2021 0.532762, 2022 0.528920, 2023 0.499957, 2024 0.486105.
- `game_type=F`는 2022 0.708749에서 2023 0.472904로 급변했고, `R`은 0.503691→0.503118로 거의 유지됐다. 이 구조적 변화가 2023 calibration 붕괴의 핵심이다.
- `asof_pitcher_success_rate`와 `asof_batter_success_rate`는 가장 강한 row-local 신호다. 따라서 historical rate smoothing과 count/hand/pressure 상호작용이 Trackman 평균보다 우선이다.

## 3. Trackman 연결 품질 감사

현재 익명 ID 연결은 연간 pitch-mix trajectory + 손잡이 기반 one-to-one Hungarian 매칭이다. 2024 origin에서 711명 중 446명, 2025 origin에서 792명 중 479명만 linked이며 2025의 고신뢰는 127명, medium 163명, low 189명, unmatched 313명이다. 2024→2025 ID 안정성은 high 95.65%, medium 79.75%, low 69.94%, 전체 80.04%다.

기존 프로파일을 main target에 붙여 물리 변수의 단변량 연관을 확인한 결과, 2023~2024에서 release speed, spin, extension, release height/side의 전체 상관은 대부분 |r|<0.03이었다. 이는 물리량 자체가 무의미하다는 뜻이 아니라, (a) anonymous linkage 오류, (b) target이 location이 아닌 제구 성공 사건, (c) 시즌·상황·손잡이 조건을 제거하지 않은 평균 집계가 원인일 가능성이 높다.

이번에 `src/trackman_domain.py`를 추가해 다음을 누수 없이 계산했다.

- release/flight 지표의 std, MAD, CV, 계절별 slope 및 latest-minus-history;
- 구종별 평균·변동성, repertoire entropy/effective repertoire, fastball/off-speed 비율;
- 경기당 투구 수의 평균·표준편차·최대값, pitch number/late-pitch share, PA 내 3구 이상 비율;
- 타자 손별 구종 선택률, count별 구종 선택률.

최신 2023/2024 forward 검증에서 enriched profile 전체를 넣은 대체 LGB는 각각 Brier 0.251331/0.249038로 incumbent 0.251137/0.248461보다 악화됐다. 작은 1~10% 잔차 블렌드도 2024에서 개선되지 않았다. 따라서 이 프로파일은 분석 자산으로 보존하되 다음 제출에 포함하지 않는다.

추가로 `src/target_encoding.py`의 prequential 수축률(투수×카운트, 투수×타자손, 타자×카운트 등)을 시험했다. 2023은 Brier 0.251050으로 incumbent보다 -0.000087 개선됐지만, 2024는 자체 최적 offset에서도 0.249343으로 +0.000882 악화됐다. 연도 일반화가 없어 제출 후보에서 제외했다.

## 4. 도메인 지식과 외부 방법론의 적용 판단

릴리스 포인트 변동성과 pitch mechanics의 일관성이 location consistency와 관련된다는 biomechanics/TrackMan 연구 결과는 타당하다. 다만 이 대회에서는 직접 투수 ID가 없고 매칭 불확실성이 크므로, 평균 구속을 모델 입력으로 대체하는 대신 고신뢰 매칭에 한해 residual prior로 제한해야 한다. Trackman의 속도·회전·movement·release point가 sequencing 분석에 쓰인다는 공식 설명도 같은 방향을 지지한다.

CatBoost ordered boosting은 범주형 target encoding의 prediction shift를 줄이는 장점이 있지만, 기존 CatBoost depth 6~8 pilot은 2024 Brier 0.24984~0.24992로 incumbent보다 0.00138 이상 나빴다. 전체 교체는 중단하고, 향후 사용한다면 고신뢰 domain residual 전용으로만 재시험한다.

TabM은 parameter-efficient MLP ensemble, TabR은 retrieval/nearest-neighbor 성격의 tabular 모델이다. 이 데이터는 147만 행이고 hidden test의 전체 분포를 사용할 수 없으므로, full TabM/TabR를 즉시 제출 모델로 넣는 것은 비용·누수·calibration 위험이 크다. 작은 시간순 holdout pilot에서 incumbent 잔차를 개선할 때만 후보화한다. TabICL은 대규모 tabular ICL 연구지만 현재 운영 환경과 데이터 크기에 비해 우선순위가 낮다.

## 5. 다음 실험 우선순위

1. **v2 보존**: R-only, Trackman weight 확대는 Public에서 즉시 폐기했다.
2. **상황별 calibration 재검증**: hidden test의 game-type/season 구성은 샘플 5행으로 추정하지 않고, train-only forward 규칙으로만 고정한다. 특히 F offset 재적용은 금지한다.
3. **Trackman soft linkage**: one-to-one hard match 대신 top-k distance-weighted profile을 만들고, high-confidence rows만 residual correction 대상으로 제한한다. profile 대체 모델이 아닌 1~2% shrinkage부터 검증한다.
4. **domain residual**: count pressure × platoon × recent-middle volatility × as-of rate confidence의 소형 모델을 incumbent 잔차에 학습한다. 최신 2024 개선 및 모든 fold 악화 제한을 통과하기 전에는 ZIP을 만들지 않는다.
5. **SOTA pilot**: TabM/TabR/CatBoost는 full replacement가 아니라 2023/2024 residual pilot로만 비교한다. 시간순 분할, category map fit 분리, target-free Trackman aggregation을 지킨다.

## 6. 파일명 규칙

최종 제출 ZIP은 반드시 40자 이내이며, 버전 확인을 위해 `submit_vN.zip`처럼 짧게 만든다. 현재 기준 champion은 `submit_v2.zip`이다. 분석용 후보명은 ZIP 파일명이 아니라 `artifacts/candidates/*/manifest.json`에 기록한다.

## 외부 참고

- CatBoost ordered boosting: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- TabM (ICLR 2025) / 공식 구현: https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf / https://github.com/yandex-research/tabm
- TabR (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html
- Pitch command biomechanics: https://pubmed.ncbi.nlm.nih.gov/31449438/
- TrackMan release parameters와 location consistency: https://www.tandfonline.com/doi/abs/10.1080/02640414.2020.1868679
- Context-enhanced deep pitch-location prediction: https://link.springer.com/article/10.1007/s12283-025-00497-5
