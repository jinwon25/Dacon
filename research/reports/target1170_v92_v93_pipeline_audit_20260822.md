# 1170 후속 데이터·파이프라인 전수 점검과 v92~v93 결과 — 2026-08-22

## 결론

공식 champion은 **1161.2020600422**로 유지한다. 이번 사이클에서는 데이터 생성 구조,
평가식, exact OOF 부모, TrackMan 활용 한계와 최근 후보 전체를 다시 감사하고 독립적인
R_CORE 후보 v92 및 조건부 FM 이득 게이트 v93을 검증했다. 두 후보 모두 평가 v3의
시간·월·의존성 안정성 기준을 통과하지 못했다. 따라서 새 ZIP을 만들거나 DACON에
제출하지 않았다.

- champion: `artifacts/standalone_champion_1161/standalone_champion_1161.zip`
- SHA-256: `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`
- Public: `1161.2020600422`
- 1170까지: `8.7979399578`
- 이번 사이클 제출: `0`
- test CSV/집계/행 순서 사용: 없음
- 후보 추론 단위: 현재 행과 동결 train artifact만 사용

## 공식 평가와 규정 재감사

공식 점수는 다음과 같다.

```text
Score = max(0, 100000 * (1 - BS / (r * (1-r))))
BS    = mean((p-y)^2)
r     = hidden evaluation target mean
```

Public은 비공개 245,789행 전체 100%이고 종료 시 Private도 같은 값이다. 따라서 랜덤
holdout AUC가 아니라 엄격한 미래 연도 paired Brier 차이를 판단 기준으로 삼아야 한다.
공식 실행 한도는 600초이며 ZIP 최상위에는 `script.py`, `requirements.txt`, `model/`만
둔다. 운영 답변상 외부 정답이나 정답에 준하는 정보로 feature, 후처리, 모델 설정,
앙상블 비율을 맞추는 것은 금지되며 코드 검증 때 도출 과정도 확인된다.

이번 후보는 다음을 지켰다.

1. test 행 간 통계, 빈도, 정렬, rolling, batch normalization을 사용하지 않았다.
2. 2024 label은 레시피 선택이나 fitting이 끝난 뒤 outer audit에서만 사용했다.
3. Public 점수로 route, eta, threshold, blend weight를 역산하지 않았다.
4. v92/v93은 R_CORE에만 적용했다. 2022 historical parent는 최종 TrackMan gate가
   R_ANCHOR에서만 빠져 있으므로 R_CORE/F component parity는 exact다.
5. 통과 후보가 생길 경우에도 과거 ZIP을 실행 시 요구하지 않는 standalone ZIP으로만
   전달·제출한다.

## 데이터 생성 구조 전수 점검

### 공식 train

- 1,475,092행, 2019~2024, 49열
- 2024: 253,507행, 성공률 `0.4861049`
- 연도별 성공률은 2019 `0.564670`, 2020 `0.532712`, 2021 `0.532762`,
  2022 `0.528920`, 2023 `0.499957`, 2024 `0.486105`
- 현재 행에는 볼카운트, 아웃, 이닝, 주자, 점수, LI, 투수·타자·팀 ID, 손잡이,
  현재 행 시점까지의 ASOF 이력이 있다.
- 현재 투구의 실제 위치, 구종, 구속, 회전, 무브먼트, 포수 요구 위치는 없다.

### TrackMan history

- 1,793,078행, 2019~2024, 30열
- 현재 평가 투구와 직접 연결되는 pitch key가 없고, current-pitch 물리량도 추론 입력에
  없다. 합법적인 주 용도는 동결된 과거 pitcher/pitch-type/시즌 profile이다.
- 현재 행의 `asof_pitcher_*`, pitcher ID 및 pitch-mix가 이미 물리 profile의 상당 부분을
  대리한다. v65~v74에서 평균, 반복성, IVB, 잠재 구종, 구종 student, calendar gate를
  검증했으나 미래 연도 방향이 유지되지 않았다.
- 야구 도메인상 제구는 “의도한 위치와 실제 위치의 거리”가 핵심이다. xCTRL 계열처럼
  개인별 intent를 추정하려면 현재 투구 위치와 충분한 투수×구종×손잡이 표본이 필요하다.
  이 대회 추론 행에는 전자가 없으므로 TrackMan profile만으로 current-pitch command를
  직접 복원한다는 주장은 성립하지 않는다.

## 파이프라인 감사

### 확인된 장점

- metric 구현은 공식 Brier Skill Score와 일치한다.
- champion은 모든 lookup을 train-only artifact로 동결했고 shuffle/partition 불변이다.
- v84는 R_CORE strict, R_ANCHOR TrackMan-ASOF, F shared FM을 서로 분리해 route별
  신호 충돌을 줄였다.
- canonical ZIP은 과거 제출 ZIP 없이 단독 실행되며 공식 245,789행을 41초에 처리했다.

### 현재 핵심 병목

1. **기준 확률의 연도별 calibration 부호 반전**
   - exact v84 full-2022 평균 잔차: `-0.0184775`
   - exact v84 late-2023 평균 잔차: `+0.0048233`
   - exact v84 full-2024 평균 잔차: `-0.0017754`
   단일 global/domain offset을 다음 해로 운반하면 방향이 자주 뒤집힌다.
2. **2024 반복 사용**
   - 2024는 사실상 development-contaminated 확인축이다. 같은 축에서 소폭 양수인
     후보를 계속 고르면 winner's curse가 커진다.
3. **독립 모델보다 동일 FM 변형이 많음**
   - v86~v88, v93은 평균 headroom은 보였지만 같은 상호작용 방향을 공유해 실패 월을
     안정적으로 상쇄하지 못했다.
4. **과거 exact champion analogue 부족**
   - full-2024와 late-2023은 exact하게 복원되지만, 더 오래된 연도의 최종 champion
     전체 계보를 반복 학습하는 runner가 완성되지 않았다. 두 축만으로 family 선택과
     불확실성 판정을 동시에 수행하는 것이 현재 가장 큰 통계적 병목이다.

## v92: temporal player command empirical Bayes

가설은 champion 잔차 중 투수 또는 타자의 제구 성향이 시간 원점이 바뀌어도 같은 방향으로
남는다는 것이었다. 14개 group family, alpha 3개, eta 3개, 총 126개 source trial을
early-2022→late-2022와 full-2022→late-2023에서만 선택했다.

선택 레시피는 `pitcher × batter_hand`, alpha `10000`, eta `0.25`였다.

| 축 | gain | 양수 월 | 최악 월 |
|---|---:|---:|---:|
| late-2022 | +1.5172 | 100% | +0.8812 |
| late-2023 | +0.8749 | 66.7% | -0.6142 |
| full-2024 | +0.0932 | 37.5% | -1.3316 |
| late-2024 | -0.1024 | 0% | -1.3316 |

두 source 축에서는 평균 양수였지만 2024 후반에 반전됐다. 선수 단위 residual command는
이미 champion의 ASOF/state/interaction 모델이 대부분 흡수했고, 남은 효과는 다음 해까지
운반할 만큼 안정적이지 않다. 기각했다.

## v93: conditional FM benefit gate

v87 source-mean FM은 full-2022 `+5.8667`(모든 월 양수), late-2023 `+5.6229`였지만
late-2023 10월이 음수였다. 이에 행별 Brier 이득을 목표로 다음 row-local feature만 사용한
얕은 gate를 만들었다.

- 두 source FM correction의 방향·크기·합의·coherence
- champion parent probability
- pitcher/batter ASOF 표본 신뢰도와 성공·middle·reverse·ball·strike rate
- 최근 1/3/5경기 delta
- count, inning, outs, runners, score, LI, hand matchup, month
- 모델 family: Ridge와 depth-3/7-leaf shallow LightGBM
- gate: hard positive, source-only soft positive

선택은 early-2022→late-2022 및 full-2022→late-2023 두 전이만 사용했다. 네 레시피 모두
late-2022에서는 `+10.28~+14.82`였지만 late-2023에서는 `-0.54~-2.97`로 반전해 source
gate를 통과하지 못했다. 진단상 가장 보수적인 Ridge soft gate의 2024 결과는 다음과 같다.

| 축 | gain | 양수 월 | 최악 월 |
|---|---:|---:|---:|
| full-2024 | +1.4791 | 62.5% | -5.5383 |
| late-2024 | +1.8530 | 66.7% | -4.2909 |

평균은 양수지만 full-2024의 4월·6월·8월이 음수이고 source transfer가 실패했다. 같은
family의 v86/v87/v88/v93 조합을 추가 진단했으나 최대 평균 gain `+2.0820` 조합도
late-2024 양수 월이 33.3%뿐이었다. label로 월 route를 고르는 것은 금지했다.

### v87 의존성 부트스트랩

평균 양수인 v87을 바로 Public probe할 근거가 있는지 5,000회 paired cluster bootstrap으로
추가 확인했다.

| 범위·군집 | 개선 재표본 비율 | 95% Brier delta 구간 |
|---|---:|---:|
| full-2024·pitcher | 78.84% | `[-1.245e-05, +5.501e-06]` |
| full-2024·batter | 80.74% | `[-1.172e-05, +4.611e-06]` |
| late-2024·pitcher | 72.18% | `[-1.996e-05, +1.028e-05]` |
| late-2024·batter | 72.26% | `[-2.066e-05, +1.186e-05]` |

모든 95% 구간이 0을 포함한다. 평가 v3가 요구하는 5% 하한 양수 조건을 충족하지 못하므로
v87/v93 모두 ZIP·제출 대상이 아니다.

## 최신 모델·야구 문헌 판단

- xCTRL은 투수별 의도 위치를 확률적으로 추정하는 방향이 제구 분석에 적합함을 보여준다.
  그러나 이 대회에는 current-pitch actual/target location이 없어 직접 구현할 관측치가 없다.
- ICLR 2025 TabM은 parameter-efficient MLP ensemble로 강한 tabular DL 기준선이다. 로컬
  v45 TabM-mini는 late-2023 선택 gate를 통과하지 못했다. 다음 실행은 작은 seed/width
  재탐색이 아니라 아래의 다중-origin exact runner가 먼저 완성된 뒤 full TabM을 독립
  OOF 축으로 한 번 검증하는 방식이어야 한다.
- Nature 2025 TabPFN v2의 주 검증 범위는 최대 10,000행 규모다. 147만 행 전체를 직접
  학습하는 주력 모델로는 맞지 않으며, 표본 축소가 필요한 메타 게이트에만 제한적으로
  고려할 수 있다.

## 1170을 향한 다음 우선순위

### P0 — exact multi-origin champion runner

2020→2021, 2021→2022, 2022→2023, 2023→2024 각각에서 그 시점까지의 데이터만으로
champion 전체 계보를 재학습한다. 각 축에 row_id, target, parent, domain, month,
pitcher, batter와 artifact hash를 저장한다. 이후 family/feature/eta 선택은 앞선 두 축,
locked confirmation은 뒤의 두 축처럼 역할을 명시한다. 이것이 추가 모델보다 우선이다.

### P1 — 독립적인 대용량 tabular ensemble

동일 FM의 gate가 아니라 다음 세 모델의 strict temporal OOF를 만든다.

1. 최근 2~3시즌 CatBoost ordered boosting: ID/category interaction과 결측 자체를 처리
2. season/domain 균형 LightGBM: Brier 목적, 얕은 leaf, 다중 seed 평균
3. full TabM 또는 2026 대규모 tabular low-rank ensemble: 수치·저카디널 category 중심

세 모델의 row-level residual correlation이 champion과 충분히 다를 때만 v77
worst-month/domain constrained stack에 넣는다. 같은 family의 eta/seed 변형은 독립 후보로
세지 않는다.

### P2 — 야구 구조형 hurdle model

최종 성공을 하나의 black-box 확률로만 보지 않고 다음 latent risk를 별도 head로 둔다.

1. intent risk: count×hand×base/out×score/LI
2. execution risk: pitcher ASOF command와 최근 변동성
3. arsenal risk: 과거 TrackMan pitch-type/physics profile과 표본 신뢰도

학습 target은 여전히 공식 `control_success`만 사용하고, head는 mixture-of-experts의
regularized latent component로 구현한다. TrackMan head는 pitcher history가 충분한 행에서만
reliability gate로 열고, 현재 투구 물리량을 안다고 가정하지 않는다.

### P3 — 팀 협업 데이터 계약

팀원이 모델을 추가할 때 ZIP이나 Public 점수 대신 동일 exact OOF bundle을 먼저 전달한다.
독립 축이 확보되면 소수점 후처리보다 오류 공분산을 줄이는 ensemble이 가능하다.

## 최종 결정

v92와 v93은 재현 가능한 연구 결과로 보존하지만 승격하지 않는다. 현재 champion보다
높을 가능성을 통계적으로 뒷받침하지 못한 파일을 제출해 Public을 추가 튜닝 신호로
사용하지 않는다. 다음 제출은 P0 runner에서 최소 두 strict-forward 축, 월/domain gate,
pitcher·batter·block bootstrap, family Reality Check를 통과한 독립 후보에 한정한다.

## 참고 링크

- 공식 평가: https://dacon.io/competitions/official/236743/overview/evaluation
- 공식 데이터 구조: https://dacon.io/competitions/official/236743/data
- 행 독립성 공지: https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent
- 코드 검증·외부 정답 주의: https://dacon.io/competitions/official/236743/talkboard/417157?dtype=recent&page=1
- xCTRL: https://arxiv.org/abs/2508.19184
- TabM: https://arxiv.org/abs/2410.24210
- TabPFN v2: https://doi.org/10.1038/s41586-024-08328-6
