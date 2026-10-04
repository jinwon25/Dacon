# 1158 이후 평가체계 재감사와 1170 실행 방향 — 2026-08-22

## 결론

공식 점수 계산 코드는 맞다. 현재 가장 큰 병목은 모델 용량이 아니라 **후보 선택 편향과
불완전한 비교 기준**이다. 2024 label은 v23–v76의 설계와 기각에 반복 사용됐으므로 더 이상
독립 holdout이 아니다. 2024에서 좋아 보이는 후보를 다시 2024로 승인하는 절차는 중단한다.

현재 champion은 Public `1158.0745556751`이며 단일 실행 파일과 SHA-256은 변하지 않는다.

- 파일: `artifacts/standalone_champion_1158/standalone_champion_1158.zip`
- SHA-256: `16DAFD86C00DDE5F93D88FF6BE606F6832A7B5A3F8B546CEA299928B78036526`
- 1170까지: `+11.9254443249`
- 필요한 평균 Brier 감소량: 테스트 양성률을 `0.4861`로 가정하면 약
  `2.9791e-05` (`0.45~0.55` 범위에서도 `2.9515e-05~2.9814e-05`)

이 차이는 작지만, 이미 잘 보정된 동일 계보에 scalar 보정을 더해서 얻기에는 크다. 현재
가장 현실적인 경로는 **팀원 독립 모델의 exact OOF를 이용한 constrained blend**와
**새 모델 family의 완전한 nested temporal 평가**다.

## 1. 공식 평가·규칙 재점검

[DACON 공식 평가 페이지](https://dacon.io/competitions/official/236743/overview/evaluation)의
점수는 다음과 같다.

```text
r = mean(y)
Brier = mean((p - y)^2)
Score = max(0, 100000 * (1 - Brier / (r * (1-r))))
```

`src/metrics.py`의 구현은 이 식과 일치하며 완전예측 `100000`, 기준확률 `0`, 음수 원점수의
0 clipping을 테스트한다. 후보 간 로컬 비교에서는 두 후보가 모두 0으로 잘릴 때 정보를
잃지 않도록 paired `unclipped BSS-equivalent gain`을 사용한다. 이는 공식 점수라고 부르지
않는다.

[공식 대회 설명](https://dacon.io/en/competitions/official/236743/overview/description)에 따르면
245,789행 전체 추론 제한은 **10분(600초)이고** 환경은 Ubuntu 22.04.5, Python 3.11.15,
6 vCPU, 28GB RAM, L4 22.4GiB다. 저장소의 120초 기준은 공식 제한이 아니라 여유를 둔
내부 soft guard로 유지한다.

[행 독립 추론 공지](https://dacon.io/competitions/official/236743/talkboard/417123)에 따라 한
test 행은 현재 행, 공식 train, train에서 미리 동결한 artifact만 사용한다. 다른 test 행의
평균·빈도·순서·그룹·rolling/lag/누적값은 금지한다. 전체·singleton·shuffle·partition에서
동일 행 확률 차이 `<=1e-12`를 계속 강제한다.

[Phase 3 운영진 답변](https://dacon.io/competitions/official/236743/talkboard/417157)은 점수
재현뿐 아니라 feature, 후처리값, 앙상블 비율, 모델 설정의 도출 과정까지 확인한다고
명시한다. 그러므로 Public 점수에서 상수를 역산하는 방식은 연구·배포 근거로 쓰지 않는다.

## 2. 기존 평가체계에서 발견한 문제

| 항목 | 재감사 결과 | 조치 |
|---|---|---|
| 공식 metric | 구현·테스트 일치 | 유지 |
| incumbent | 일부 scorecard가 `gain_vs_v13`에 고정 | `gain_vs_incumbent`로 일반화, legacy만 호환 |
| 2024 독립성 | 반복 열람·모델 설계에 사용 | `development_contaminated`, 진단만 허용 |
| local→Public 환산 | 정의가 다른 한 관측으로 projection 가능 | clean transfer 3개 미만이면 출력 억제 |
| 후보 다중 탐색 | v58–v76 point gate 중심 | family 전체 ledger + White Reality Check 필수 |
| 불확실성 | 월·domain point estimate만으로 통과 가능 | pitcher, crossed pitcher×batter, 연속 block p05 모두 양수 |
| 부모 일치 | 역사 부모와 최종 부모 결과 혼용 가능 | exact parent parity를 각 축에서 필수화 |
| 시간 제한 | 120초를 공식 제한처럼 기록 | 공식 600초 / 내부 120초로 분리 |
| 상태 문서 | v26 `1157.9736`가 최신처럼 남음 | 1158 standalone 기준으로 교체 |

Varma와 Simon은 같은 CV에서 tuning과 error estimation을 함께 하면 성능 추정이 낙관적으로
치우치며, 전체 tuning을 outer fold 안에서 반복하는 nested CV가 이 편향을 크게 줄인다고
보였다. White의 Reality Check는 동일 데이터에서 여러 후보를 탐색한 뒤 최고 후보만 보고할
때 우연한 승자를 골랐을 가능성을 검정한다. 현재 실험량에서는 둘 다 선택 사항이 아니다.

## 3. v75 — 실제 headroom과 calibration 감사

최종 1158 부모의 exact OOF를 읽고 test나 Public 점수로 계수를 학습하지 않은 진단이다.

| 축 | 행 | target rate | 예측 평균 | Brier | logit intercept / slope |
|---|---:|---:|---:|---:|---:|
| full-2024 | 253,507 | 0.486105 | 0.487544 | 0.2473222624 | -0.00488 / 1.01172 |
| late-2024 | 76,896 | 0.480909 | 0.482497 | 0.2479550838 | -0.01381 / 0.89418 |

full-2024 reliability는 10/20/50-bin에서 각각 `3.76e-06`, `6.66e-06`,
`1.77e-05`다. 같은 label로 다시 맞춘 oracle additive/affine gain도 `+0.829/+1.190`에
그쳤다. 즉 champion의 전역 calibration은 이미 양호하고, 1170의 주된 headroom은
reliability보다 **새로운 resolution과 오차 다양성**에 있다.

같은 2024 label로 맞춘 domain×count oracle은 `+19.941`을 보였지만 이는 답을 본
진단이다. 실제 source-only calibration은 다음처럼 전이되지 않았다.

| source → audit | 보정 | gain | 양수 월 | 최악 월 | 최소 domain |
|---|---|---:|---:|---:|---:|
| late-2023 → full-2024 | additive | -18.509 | 25.0% | -44.141 | -20.362 |
| late-2023 → full-2024 | affine | -356.633 | 12.5% | -524.462 | -472.030 |
| early-2024 → late-2024 | additive | +0.736 | 33.3% | -13.123 | -6.367 |
| early-2024 → late-2024 | affine | -19.562 | 0% | -52.634 | -41.160 |

따라서 global center/spread, beta/logit calibration을 champion 단독에 다시 탐색하는 경로는
종료한다. 새 직교 base model이 생긴 뒤 fold 내부 calibration을 하는 경우만 남긴다.

## 4. v76 — domain×count 신호의 source-only 반증

v75 oracle의 구조가 실제 과거에서 이동하는지 확인하기 위해 domain 3개×정확한 count
12개, EB prior 2000, domain 평균 제거, weight 1.0을 고정했다. v75가 동기를 제공했으므로
v76 자체도 연구 진단이며 2024 결과로 재튜닝하지 않았다.

| 축 | gain | 양수 월 | 최악 월 | 최소 domain |
|---|---:|---:|---:|---:|
| early22→late22 | -0.199 | 66.7% | -7.272 | -2.572 |
| full22→full23 | +1.253 | 57.1% | -12.401 | +0.140 |
| full23→full24, 역사 부모 | -54.646 | 0% | -341.036 | -354.168 |
| late23→full24, exact 부모 | -45.486 | 0% | -164.505 | -173.650 |
| early24→late24, exact 부모 | +0.243 | 66.7% | -5.862 | -1.813 |

같은 축의 잔차 구조는 크지만 과거에서 미래로 이동하지 않는다. domain×count lookup의 prior,
weight, 일부 count만 다시 고르는 것은 2024 사후 탐색이므로 금지한다.

## 5. 새 평가 계약 v3

`configs/evaluation_v3.json`과 `src/evaluation_contract.py`가 다음을 기계적으로 강제한다.

1. 각 축을 `nested_outer`, `locked_shadow`, `development_contaminated` 중 하나로 선언한다.
2. 서로 다른 primary 축이 최소 2개 필요하다. 오염 축은 개수와 통과 여부에 기여하지 않는다.
3. 모든 primary 축에서 exact parent, 축을 보기 전 recipe 동결, gain `>0`, 양수 월
   `>=75%`, 최악 월 `>-5`, 최소 domain `>=0`이 필요하다.
4. pitcher, crossed pitcher×batter, 연속 block bootstrap p05가 모두 `>0`이어야 한다.
5. 같은 family에서 실행한 모든 trial을 ledger에 포함하고 Reality Check `p<=0.10`을
   통과해야 한다.
6. 로컬→Public 환산은 clean 관측 3개 전까지 출력하지 않으며, 그 이후에도 시나리오일
   뿐 승격 근거가 아니다.
7. 통과 후보만 처음부터 standalone ZIP으로 만들고 공식 600초·내부 120초, 메모리,
   오프라인, 행 독립성을 검증한다.

현재 v58–v76 중 이 계약으로 Public probe 자격을 얻는 후보는 없다. 이는 실패가 아니라
추가 leaderboard 과적합을 막는 필요한 중단선이다.

## 6. 1170을 위한 우선순위 포트폴리오

| 우선 | 경로 | 기대 근거 | 필요한 증거 | 판단 |
|---|---|---|---|---|
| P0 | 팀원 독립 exact OOF constrained blend | `2.98e-05` Brier만 줄이면 되며 작은 비상관 오차도 충분 | 동일 row_id·동일 축 OOF, exact parent, 잔차 공분산 | 가장 먼저 |
| P0 | 최종 family의 nested temporal OOF 재구축 | 현재 selection bias를 제거하고 가짜 양수를 줄임 | outer 내부에서 feature/tuning/calibration 전체 반복 | 기반 공사 |
| P1 | 환경 안정 feature-family residual | v62는 모든 feature equal-risk였고 안정성 선별은 미실시 | 두 primary 축에서 방향·gain·worst-group 동시 안정 | 신규 직교 후보 |
| P1 | ID 의존을 낮춘 recency-trajectory GAM/GBDT | 현재 calibration보다 resolution이 병목 | 사전 고정한 ASOF/최근창 contrast와 신뢰도 상호작용 | 1회 사전등록 pilot |
| P1 | 모델별 오류 공분산 기반 environment bagging | 평균 점수보다 champion과 다른 오차가 목표에 직접 연결 | 축별 `E[(p_j-y)(p_k-y)]`, 최악 group 제약 | OOF 확보 후 |
| P2 | 새 base model 뒤 교차적합 beta/logit calibration | 서로 다른 모델의 결합 calibration은 아직 열림 | 모든 calibration을 inner fold에서만 fit | 마지막 단계 |
| 보류 | TrackMan 신규 경로 | 과거 profile은 이미 전수조사, 현재 투구 물리·위치 없음 | 합법적인 현재행 pre-pitch 신호가 새로 생길 때만 | 현재 재탐색 금지 |
| 종료 | scalar 보정, count lookup, IVB/물리 profile, 물리 학생 | v59/v60, v76, v66–v74에서 시간 전이 실패 | 없음 | 재튜닝 금지 |
| 종료 | full-data TabPFN/TabR, TabM 크기·seed 재탐색 | 147만 행 부적합 또는 v45/v55 외부축 실패 | 완전히 다른 독립 소스가 없는 한 없음 | 후순위/종료 |

### P0-1. 팀원 OOF 전달 계약

각 모델은 `row_id, origin, target, prediction, domain3, month, pitcher_id, batter_id`와
모델/설정/학습데이터 해시를 전달한다. OOF는 해당 origin 이전 label로만 학습하고,
target·prediction·row_id 일치 여부를 자동 검사한다. Public 점수만 있는 모델은 blend
weight 학습에 쓰지 않는다.

blend는 다음처럼 incumbent 방향을 기준으로 제한한다.

```text
p_blend = p_incumbent + Σ α_j (p_j - p_incumbent)
α_j >= 0,  Σ α_j <= 1
```

inner 축에서 Brier를 최소화하되 month/domain 최악 손실, 잔차 상관, 가중치 상한을
제약하고 outer 축에는 한 번만 적용한다. 후보 모델을 추가하거나 feature를 바꿀 때마다
family trial count와 Reality Check를 다시 계산한다.

### P0-2. nested OOF 재구축

현재 champion의 모든 역사 ZIP을 재연결하는 것이 목표가 아니다. **새 후보를 만드는
알고리즘 전체**를 한 함수로 고정하고 각 outer origin에서 다음을 반복한다.

1. outer 이전 시즌만 열어 inner rolling-origin으로 feature family와 hyperparameter를 선택한다.
2. 선택된 recipe로 outer 이전 전체를 다시 학습한다.
3. outer를 한 번 예측하고 calibration도 inner에서 선택한 값만 사용한다.
4. 모든 candidate/trial 예측을 보존해 Reality Check 범위를 숨기지 않는다.

이미 사람이 본 2022–2024를 완전한 virgin holdout이라고 부르지는 않는다. nested replay와
전체 trial 보정으로 알고리즘 선택 편향을 줄이고, 새 locked shadow가 있다면 최종 1회만 연다.

### P1-1. 환경 안정 residual의 사전등록안

환경은 target을 보지 않고 `origin×domain3×month band`로 정한다. feature family는 공식
현재행 상황, ASOF 선수 능력, 최근창 차이, 표본수/신뢰도, 주자·count·이닝/점수 문맥처럼
추론 시 알 수 있는 것만 사용한다. 각 family를 별도 저용량 모델로 fit해 두 primary 축에서
residual 방향과 gain이 같은 family만 남긴다.

첫 pilot은 한 번만 실행한다.

- 모델: 낮은 depth의 L2 GBDT와 spline/logistic GAM 두 종류
- 목적: parent residual의 squared error 또는 직접 Brier
- 용량·seed·feature family는 실행 전 JSON으로 동결
- 비교: ERM, environment-balanced, worst-group penalty 세 개
- 성공 조건: v3 계약 전체 통과 및 parent와 충분히 낮은 residual correlation

이는 v62의 “전체 feature를 ExtraTrees에 넣고 group weight만 변경”한 실험과 다르다.
핵심은 모델 이름이 아니라 **환경 간 방향이 변하는 feature family를 학습 전에 제거**하는 데
있다.

### P1-2. 야구 도메인 기반 recency trajectory

현재 위치·의도·물리량을 복원하려 하지 않는다. 공식 행에 이미 존재하는 누적/최근 상태에서
다음과 같은 저자유도 변화량만 사전 고정해 사용한다.

- 장기 ASOF 대비 최근창 성공률 차이와 표본수에 따른 shrinkage
- 짧은창−중간창, 중간창−장기창의 기울기와 곡률
- count pressure, 주자, 이닝/점수 문맥과 위 변화량의 제한된 상호작용
- 새 선수·적은 표본을 자동으로 0으로 축소하는 신뢰도 항

xCTRL처럼 의도와 실행을 분리하는 모델은 실제 위치와 intended target이 있어야 한다.
이 데이터에서는 그 구조를 직접 구현하지 않고, pre-pitch context의 partial pooling만
사용한다.

## 7. 최신 방법론의 적용 판단

- [TabReD, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)는
  시간 분할과 실제 산업형 feature가 모델 순위를 바꾼다고 보고했다. 여기서는 새 SOTA
  이름보다 rolling-origin 평가를 우선하는 직접 근거다.
- [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)은
  parameter-efficient MLP ensemble이 강한 표형 baseline임을 보였지만, 우리 v45 mini는
  full-2024 `-8.271`이었다. 같은 width·k·epoch 재탐색은 하지 않는다.
- [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)은 최대 10,000행
  수준 benchmark에서 강하다. 1,475,092행 전체 후보에는 맞지 않고 외부 pretrained
  weight 규정도 확인해야 하므로 소규모 residual subgroup 연구 외에는 제외한다.
- [TIVA, ICML 2023](https://proceedings.mlr.press/v202/tan23b.html)는 target과 독립적인
  속성으로 환경을 나누고 invariant risk를 학습하는 방향을 제시한다. 이를 그대로 복제하지
  않고 target-free 환경 정의와 stable feature-family 선별 원칙으로 사용한다.
- [xCTRL, 2025](https://arxiv.org/abs/2508.19184)은 투구 의도와 실행을 분리하지만 실제
  pitch location과 intended location이 필요하다. 현재 test에는 없어 직접 적용하지 않는다.
- [Varma & Simon, 2006](https://pubmed.ncbi.nlm.nih.gov/16504092/)과
  [White, 2000](https://doi.org/10.1111/1468-0262.00152)은 각각 nested selection과
  다중 탐색 보정의 근거다.
- [Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437)은 Brier 같은
  proper scoring rule에서 정직한 확률 예측을 최적화해야 한다는 근거다.

## 8. 즉시 실행 순서

1. 팀원들에게 exact OOF 전달 계약을 공유하고 독립 OOF 존재 여부를 먼저 확인한다.
2. 동시에 final-family candidate ledger와 nested outer runner를 만든다.
3. OOF가 오면 모델별 gain보다 먼저 residual correlation·quadratic blend headroom을 계산한다.
4. 독립 OOF가 없다면 사전등록한 환경 안정 residual pilot **한 family**만 실행한다.
5. v3 통과 전에는 ZIP과 Public probe를 만들지 않는다.
6. 통과하면 처음부터 standalone ZIP으로 패키징하고 공식 규칙·600초·내부 120초·행 독립성을
   모두 검증한다.

재현 코드는 `src/archive/v75_evaluation_headroom_audit.py`,
`src/archive/v76_domain_count_contrast.py`이고, compact 결과는
`reports/v75_evaluation_headroom_metrics.csv`,
`reports/v76_domain_count_contrast_metrics.csv`에 남겼다. 대형 예측 cache는 판단 확정 후
삭제해 혼선을 막는다.
